"""
Model evaluation, residual diagnostics, comparison, rolling-origin evaluation, and tuning.
"""

import numpy as np
import polars as pl
from typing import Dict, Tuple, List, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import skew, kurtosis, jarque_bera

from ts_contracts import ForecastOutput, EvaluationMetrics
from ts_models import TSModel


class ModelEvaluator:
    """Standardised evaluation for all models"""

    @staticmethod
    def evaluate(y_true: np.ndarray, forecast: ForecastOutput) -> EvaluationMetrics:
        """Compute all metrics"""
        mae = mean_absolute_error(y_true, forecast.prediction)
        rmse = np.sqrt(mean_squared_error(y_true, forecast.prediction))
        # sklearn returns a fraction; scale to percent to match the "%" in reporting
        mape = mean_absolute_percentage_error(y_true, forecast.prediction) * 100

        # PI coverage: % of actual values within [lower, upper]
        coverage = np.mean((y_true >= forecast.lower) & (y_true <= forecast.upper)) * 100

        # Mean uncertainty width
        mean_width = np.mean(forecast.uncertainty_width)

        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            pi_coverage=coverage,
            mean_uncertainty_width=mean_width
        )


class ResidualDiagnostics:
    """Statistical tests for forecast residuals: does the docs.md "Confirm with:"
    line diagnostics.md points at but that no code computed - Ljung-Box for
    autocorrelated residuals, and skew/kurtosis/Jarque-Bera for heavy tails.

    Operates on `y_true - forecast.prediction` (test-set residuals), same as
    `ts_plots.TSPlotter.residuals_diagnostic`. That makes it model-agnostic: it
    applies equally to SARIMA, ExponentialSmoothing, LightGBM and SeasonalNaive,
    since it only needs actual vs. predicted values, not a model's internals.
    This is deliberately not `model_fit.resid` + `model_fit.plot_diagnostics()`
    (the SARIMAX-specific in-sample innovations diagnostic) - those rely on the
    state-space representation and have no equivalent for LightGBM/SeasonalNaive.
    """

    @staticmethod
    def ljung_box(residuals: np.ndarray, lags: Optional[List[int]] = None) -> pl.DataFrame:
        """Ljung-Box test for residual autocorrelation at the given lags.

        p < 0.05 at a lag means residuals are not white noise at that lag - the
        model left structure on the table (diagnostics.md's "autocorrelated
        residuals" case). Default lags cap at len(residuals) // 5 since the test
        is unreliable once lags approach the sample size, which matters on this
        project's short (~100-200 point) test windows.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        residuals = np.asarray(residuals)
        if lags is None:
            max_lag = max(1, min(20, len(residuals) // 5))
            lags = [max_lag]

        result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        return pl.from_pandas(result.reset_index().rename(columns={"index": "lag"}))

    @staticmethod
    def normality_stats(residuals: np.ndarray) -> Dict[str, float]:
        """Mean, std, skew, excess kurtosis and Jarque-Bera normality test.

        Excess kurtosis >> 0 is diagnostics.md's "heavy tails" case: real-world
        PI coverage will run below the reported target even if the residual mean
        is centered. Jarque-Bera folds skew and kurtosis into one normality test
        (p < 0.05 rejects normality); on short test windows (<50 points) treat it
        as a rough signal, not a firm verdict, since it needs a reasonable
        sample size to have power.
        """
        residuals = np.asarray(residuals)
        jb_stat, jb_pvalue = jarque_bera(residuals)

        return {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skew": float(skew(residuals)),
            "excess_kurtosis": float(kurtosis(residuals)),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
        }


class ModelComparison:
    """Compare and rank multiple models"""

    def __init__(self, y_train: np.ndarray, y_test: np.ndarray):
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}

    def add_model(self, model: TSModel) -> None:
        """Register a model"""
        self.models[model.name] = model

    def fit_all(self) -> None:
        """Fit all registered models"""
        for name, model in self.models.items():
            try:
                model.fit()
                print(f"[ok] {name} fitted")
            except Exception as e:
                print(f"[fail] {name} failed: {e}")

    def evaluate_all(self, confidence_level: float = 0.80) -> pl.DataFrame:
        """Evaluate all models and return ranking"""
        results = {}

        for name, model in self.models.items():
            if not model.fitted:
                print(f"Skipping {name} (not fitted)")
                continue

            try:
                forecast = model.forecast(len(self.y_test), confidence_level)
                metrics = ModelEvaluator.evaluate(self.y_test, forecast)
                results[name] = {
                    'metrics': metrics,
                    'forecast': forecast,
                    'params': model.get_params()
                }
            except Exception as e:
                print(f"[fail] {name} evaluation failed: {e}")

        self.results = results
        return self._ranking_table()

    def _ranking_table(self) -> pl.DataFrame:
        """Create ranking table with scores, ranked by RMSE (lower is better)"""
        rows = []
        for name, data in self.results.items():
            metrics = data['metrics']
            rows.append({
                'Model': name,
                'MAE': metrics.mae,
                'RMSE': metrics.rmse,
                'MAPE': metrics.mape,
                'PI Coverage %': metrics.pi_coverage,
                'Uncertainty Width': metrics.mean_uncertainty_width,
            })

        return (
            pl.DataFrame(rows)
            .with_columns(pl.col('RMSE').rank().alias('Rank'))
            .sort('Rank')
        )

    def best_model(self) -> Tuple[str, EvaluationMetrics]:
        """Return best model name and metrics"""
        if not self.results:
            raise RuntimeError("No results yet. Call evaluate_all() first.")

        best_name = min(self.results, key=lambda x: self.results[x]['metrics'].rmse)
        return best_name, self.results[best_name]['metrics']

    def get_forecast(self, model_name: str) -> ForecastOutput:
        """Retrieve forecast from evaluated model"""
        return self.results[model_name]['forecast']


class RollingOriginEvaluator:
    """Rolling-origin (walk-forward) evaluation, scored per forecast horizon.

    ModelComparison/ModelTuner fit once and score one forecast against one
    static y_val/y_test window - fast, but silent on whether that ranking
    would hold on a different window, and on how error grows with lead time.
    This class refits at each origin on an expanding window, forecasts
    `horizon` steps, and keeps every (origin, horizon-step) error so it can
    be grouped by horizon afterwards. It is a slower confirmation step, not
    a replacement: run it on the model(s) that already won a cheap
    ModelComparison/ModelTuner pass, not as the primary search loop.

    Cost and sample size on this project's data (672 points/asset, 14 days at 48
    half-hour intervals): SARIMA's own fit takes ~12s per origin here, so `step=1`
    with `horizon=48` means ~288 refits (~an hour) for one asset - use `step >=
    horizon` for SARIMA (~6 origins, ~70s). That same non-overlapping step also
    keeps origins statistically independent; a small `step` gives many rows but
    they overlap so heavily (same day scored from near-identical training sets)
    that per-horizon MAE/RMSE looks tighter than the data actually supports. See
    docs_src/theory/rolling-origin-evaluation.md for the full reasoning.
    """

    def __init__(self, model_class, y: np.ndarray, min_train_size: int,
                 horizon: int, step: int = 1):
        self.model_class = model_class
        self.y = y
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step = step
        self.records: List[Dict] = []

    def run(self, model_kwargs: Optional[Dict] = None,
            confidence_level: float = 0.80) -> pl.DataFrame:
        """Walk the origin forward by `step`, refitting and forecasting at each one.

        Returns one row per (origin, horizon-step) with actual, prediction,
        interval bounds and error - the raw material for metrics_by_horizon()
        and rolling_error_over_time().
        """
        model_kwargs = model_kwargs or {}
        self.records = []
        n = len(self.y)

        for origin in range(self.min_train_size, n - self.horizon, self.step):
            train = self.y[:origin]
            actual = self.y[origin:origin + self.horizon]

            try:
                model = self.model_class(train, **model_kwargs)
                model.fit()
                forecast = model.forecast(self.horizon, confidence_level)
            except Exception as e:
                print(f"[fail] origin {origin} failed: {e}")
                continue

            for h in range(self.horizon):
                self.records.append({
                    "origin": origin,
                    "horizon": h + 1,
                    "actual": actual[h],
                    "prediction": forecast.prediction[h],
                    "lower": forecast.lower[h],
                    "upper": forecast.upper[h],
                    "error": actual[h] - forecast.prediction[h],
                })

        return pl.DataFrame(self.records)

    @staticmethod
    def metrics_by_horizon(results: pl.DataFrame) -> pl.DataFrame:
        """MAE/RMSE/PI coverage grouped by horizon step, from run()'s output."""
        return (
            results
            .group_by("horizon")
            .agg([
                pl.col("error").abs().mean().alias("mae"),
                (pl.col("error") ** 2).mean().sqrt().alias("rmse"),
                (
                    (pl.col("actual") >= pl.col("lower"))
                    & (pl.col("actual") <= pl.col("upper"))
                ).mean().mul(100).alias("pi_coverage"),
            ])
            .sort("horizon")
        )

    @staticmethod
    def rolling_error_over_time(results: pl.DataFrame, horizon: int = 1,
                                 window: int = 12) -> pl.DataFrame:
        """Rolling RMSE at a fixed horizon (default 1-step), ordered by origin.

        Shows whether the model's near-term accuracy improves or degrades as
        the training window grows, rather than just its average over the run.
        """
        one_step = results.filter(pl.col("horizon") == horizon).sort("origin")
        return one_step.with_columns(
            (pl.col("error") ** 2).rolling_mean(window_size=window).sqrt().alias("rolling_rmse")
        )


class ModelTuner:
    """Hyperparameter tuning for a single model.

    Trials are fit on y_train and scored on y_val. Keep y_val disjoint from
    the test set used for final reporting - tuning against the test set makes
    the reported improvement optimistic by construction.
    """

    def __init__(self, model_class, y_train: np.ndarray, y_val: np.ndarray):
        self.model_class = model_class
        self.y_train = y_train
        self.y_val = y_val
        self.trials = []

    def grid_search(self, param_grid: Dict[str, List],
                   confidence_level: float = 0.80) -> pl.DataFrame:
        """Grid search over parameter combinations, ranked by RMSE on y_val"""
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            try:
                model = self.model_class(self.y_train, **params)
                model.fit()
                forecast = model.forecast(len(self.y_val), confidence_level)
                metrics = ModelEvaluator.evaluate(self.y_val, forecast)

                self.trials.append({
                    'params': params,
                    'metrics': metrics,
                    'forecast': forecast
                })
            except Exception as e:
                print(f"[fail] Params {params} failed: {e}")

        return self._trials_table()

    def _trials_table(self) -> pl.DataFrame:
        """Return trials as a dataframe ranked by RMSE (lower is better)"""
        rows = []
        for trial in self.trials:
            params = trial['params']
            metrics = trial['metrics']
            row = {
                'MAE': metrics.mae,
                'RMSE': metrics.rmse,
                'MAPE': metrics.mape,
                'PI Coverage %': metrics.pi_coverage,
            }
            row.update({f"param_{k}": str(v) for k, v in params.items()})
            rows.append(row)

        return (
            pl.DataFrame(rows)
            .with_columns(pl.col('RMSE').rank().alias('Rank'))
            .sort('Rank')
        )

    def best_params(self) -> Dict:
        """Return best hyperparameters"""
        if not self.trials:
            raise RuntimeError("No trials yet. Call grid_search() first.")

        best_trial = min(self.trials, key=lambda x: x['metrics'].rmse)
        return best_trial['params']
