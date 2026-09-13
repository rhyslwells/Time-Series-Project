"""
Time series model implementations: SeasonalNaive, SARIMA, ExponentialSmoothing, LightGBM.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import norm

from ts_contracts import ForecastOutput


class TSModel(ABC):
    """Abstract base class for time series models"""

    def __init__(self, name: str, y_train: np.ndarray):
        self.name = name
        self.y_train = y_train
        self.model = None
        self.fitted = False

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Fit model to training data"""
        pass

    @abstractmethod
    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        """Generate forecasts with prediction intervals"""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Return current model hyperparameters"""
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Set model hyperparameters"""
        pass


class SARIMAModel(TSModel):
    """SARIMA(p,d,q)x(P,D,Q,s) wrapper"""

    def __init__(
        self,
        y_train: np.ndarray,
        order: Tuple = (1, 1, 1),
        seasonal_order: Tuple = (1, 1, 1, 48),
    ):
        super().__init__("SARIMA", y_train)
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, **kwargs) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        try:
            model = SARIMAX(
                self.y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = model.fit(disp=False, maxiter=kwargs.get("maxiter", 1000))
            self.fitted = True
        except Exception as e:
            raise RuntimeError(f"SARIMA fit failed: {e}")

    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        forecast = self.model.get_forecast(steps=steps)
        yhat = np.asarray(forecast.predicted_mean)
        ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))

        return ForecastOutput(
            prediction=yhat,
            lower=ci[:, 0],
            upper=ci[:, 1],
            uncertainty_width=ci[:, 1] - ci[:, 0],
        )

    def get_params(self) -> Dict:
        return {"order": self.order, "seasonal_order": self.seasonal_order}

    def set_params(self, **kwargs) -> None:
        if "order" in kwargs:
            self.order = kwargs["order"]
        if "seasonal_order" in kwargs:
            self.seasonal_order = kwargs["seasonal_order"]
        self.fitted = False


class ExponentialSmoothingModel(TSModel):
    """Exponential Smoothing wrapper (Holt-Winters)"""

    def __init__(
        self,
        y_train: np.ndarray,
        seasonal_periods: int = 48,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = False,
    ):
        super().__init__("ExponentialSmoothing", y_train)
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend

    def fit(self, **kwargs) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        try:
            model = ExponentialSmoothing(
                self.y_train,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
            )
            self.model = model.fit(optimized=True)
            self.fitted = True
        except Exception as e:
            raise RuntimeError(f"ExponentialSmoothing fit failed: {e}")

    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        yhat = np.asarray(self.model.forecast(steps=steps))

        # Estimate prediction intervals from residuals
        residual_std = np.std(np.asarray(self.model.resid))
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std

        return ForecastOutput(
            prediction=yhat,
            lower=yhat - margin,
            upper=yhat + margin,
            uncertainty_width=np.full_like(yhat, 2 * margin),
        )

    def get_params(self) -> Dict:
        return {
            "seasonal_periods": self.seasonal_periods,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend,
        }

    def set_params(self, **kwargs) -> None:
        if "seasonal_periods" in kwargs:
            self.seasonal_periods = kwargs["seasonal_periods"]
        if "trend" in kwargs:
            self.trend = kwargs["trend"]
        if "seasonal" in kwargs:
            self.seasonal = kwargs["seasonal"]
        if "damped_trend" in kwargs:
            self.damped_trend = kwargs["damped_trend"]
        self.fitted = False


class SeasonalNaiveModel(TSModel):
    """Seasonal naive baseline: yhat_t = y_{t-s}.

    The reference every other model is scored against (MASE is the ratio of a
    model's MAE to this model's MAE on the same window). On clean data with a
    hard-coded daily profile this is a strong baseline and beating it is not
    guaranteed.
    """

    def __init__(self, y_train: np.ndarray, season_length: int = 48):
        super().__init__("SeasonalNaive", y_train)
        self.season_length = season_length

    def fit(self, **kwargs) -> None:
        s = self.season_length
        if len(self.y_train) <= s:
            raise RuntimeError(
                f"SeasonalNaive needs more than {s} training points, got {len(self.y_train)}"
            )
        # One-cycle-ahead in-sample residuals set the interval scale.
        self._residual_std = float(np.std(self.y_train[s:] - self.y_train[:-s]))
        self.model = self.y_train[-s:]
        self.fitted = True

    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        s = self.season_length
        last_season = self.model
        yhat = np.array([last_season[i % s] for i in range(steps)])

        # h-step error accumulates one residual variance per seasonal cycle.
        z_score = norm.ppf((1 + confidence_level) / 2)
        cycles = np.floor(np.arange(steps) / s) + 1
        margin = z_score * self._residual_std * np.sqrt(cycles)

        return ForecastOutput(
            prediction=yhat,
            lower=yhat - margin,
            upper=yhat + margin,
            uncertainty_width=2 * margin,
        )

    def get_params(self) -> Dict:
        return {"season_length": self.season_length}

    def set_params(self, **kwargs) -> None:
        if "season_length" in kwargs:
            self.season_length = kwargs["season_length"]
        self.fitted = False


class LightGBMModel(TSModel):
    """LightGBM for time series, with quantile regression for uncertainty.

    Two feature modes, chosen by whether `X_train` is supplied:

    - **Internal features** (`X_train=None`, the default): builds lag + hour-of-day
      features from `y_train` alone, same as before. `forecast(steps, confidence_level)`
      works standalone, recursively feeding its own median prediction forward — this is
      the mode `ModelComparison`/`ModelEvaluator`/`ModelTuner` (`ts_evaluation.py`) drive
      generically, and what the existing notebooks (`ts_model_explorer.py`,
      `lightgbm_residual_diagnostics.py`) use unchanged.
    - **External features** (`X_train` supplied, e.g. the `feat_*` columns from
      `metering_data_supervised_learning.parquet`): `y_train` must be the aligned target
      column (same length as `X_train`, already leakage-safe — see
      `docs_src/modeling/supervised_learning_models/feature_engineering.md`).
      `forecast()` then requires a matching `X_test` (already-engineered feature rows for
      the forecast horizon) — this mode is for direct notebook use, not the generic
      evaluation loop, since `TSModel.forecast` callers never pass `X_test`.

    Uncertainty is quantile regression: one `LGBMRegressor` per level in
    `quantile_levels` (default P10/P50/P90). Quantile predictions are sorted row-wise
    post-hoc to guarantee P10 <= P50 <= P90 (LightGBM's quantile objective does not
    guarantee monotonicity on its own — see
    `docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning models.md`).
    """

    def __init__(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray = None,
        lags: List[int] = None,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        quantile_levels: List[float] = None,
    ):
        super().__init__("LightGBM", y_train)
        self.lags = lags or [1, 2, 48, 96]  # 30min, 1hr, 1day, 2day
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.quantile_levels = sorted(quantile_levels or [0.1, 0.5, 0.9])
        self._median_idx = int(
            np.argmin(np.abs(np.array(self.quantile_levels) - 0.5))
        )
        self.X_train = np.asarray(X_train) if X_train is not None else None
        self.feature_names = None
        self.models: Dict[float, object] = {}

    def _create_features(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lag features and seasonality"""
        max_lag = max(self.lags)
        X = []
        targets = []

        for i in range(max_lag, len(y)):
            features = [y[i - lag] for lag in self.lags]
            # Add seasonal component (hour of day)
            hour_of_day = (i % 48) / 48.0
            features.append(hour_of_day)
            X.append(features)
            targets.append(y[i])

        self.feature_names = [f"lag_{lag}" for lag in self.lags] + ["hour_of_day"]
        return np.array(X), np.array(targets)

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is not None:
            if len(self.X_train) != len(self.y_train):
                raise ValueError(
                    f"X_train ({len(self.X_train)} rows) and y_train "
                    f"({len(self.y_train)} rows) must be the same length and "
                    "already row-aligned (see metering_data_supervised_learning.parquet)."
                )
            return self.X_train, self.y_train
        return self._create_features(self.y_train)

    def fit(self, **kwargs) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("LightGBM not installed. pip install lightgbm")

        X, y = self._prepare_training_data()

        self.models = {}
        for q in self.quantile_levels:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=kwargs.get("n_estimators", 100),
                verbose=-1,
            )
            model.fit(X, y)
            self.models[q] = model
        self.model = self.models[self.quantile_levels[self._median_idx]]
        self.fitted = True

    def _stack_quantiles(self, quantile_preds: Dict[float, np.ndarray]) -> ForecastOutput:
        """Row-wise sort quantile predictions to enforce monotonicity, then split
        into prediction/lower/upper."""
        q_arr = np.stack([quantile_preds[q] for q in self.quantile_levels], axis=1)
        q_arr_sorted = np.sort(q_arr, axis=1)

        prediction = q_arr_sorted[:, self._median_idx]
        lower = q_arr_sorted[:, 0]
        upper = q_arr_sorted[:, -1]

        return ForecastOutput(
            prediction=prediction,
            lower=lower,
            upper=upper,
            uncertainty_width=upper - lower,
        )

    def forecast(
        self, steps: int, confidence_level: float = 0.80, X_test: np.ndarray = None
    ) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.X_train is not None:
            if X_test is None:
                raise ValueError(
                    "This model was fit with X_train, so forecast() requires a "
                    "matching X_test of already-engineered feature rows."
                )
            X_test = np.asarray(X_test)
            if len(X_test) != steps:
                raise ValueError(
                    f"X_test has {len(X_test)} rows but steps={steps}."
                )
            quantile_preds = {
                q: np.asarray(self.models[q].predict(X_test))
                for q in self.quantile_levels
            }
            return self._stack_quantiles(quantile_preds)

        # Internal-features mode: recursive forecast, feeding the median
        # prediction forward as each step's lag input.
        y_recent = self.y_train.copy()
        median_q = self.quantile_levels[self._median_idx]
        quantile_preds = {q: [] for q in self.quantile_levels}

        for _ in range(steps):
            features = [
                y_recent[-lag] if lag <= len(y_recent) else y_recent[0]
                for lag in self.lags
            ]
            hour = (len(y_recent) % 48) / 48.0
            features.append(hour)

            step_preds = {}
            for q in self.quantile_levels:
                step_preds[q] = float(self.models[q].predict([features])[0])
                quantile_preds[q].append(step_preds[q])

            y_recent = np.append(y_recent, step_preds[median_q])

        quantile_preds = {q: np.array(v) for q, v in quantile_preds.items()}
        return self._stack_quantiles(quantile_preds)

    def get_params(self) -> Dict:
        return {
            "lags": self.lags,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "quantile_levels": self.quantile_levels,
        }

    def set_params(self, **kwargs) -> None:
        if "lags" in kwargs:
            self.lags = kwargs["lags"]
        if "num_leaves" in kwargs:
            self.num_leaves = kwargs["num_leaves"]
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]
        if "quantile_levels" in kwargs:
            self.quantile_levels = sorted(kwargs["quantile_levels"])
            self._median_idx = int(
                np.argmin(np.abs(np.array(self.quantile_levels) - 0.5))
            )
        self.fitted = False
