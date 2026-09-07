# Framework Usage

How to use `ts_model_framework.py` (models, comparison, tuning) and `ts_plots.py` (diagnostics) in `src/`. For the reasoning behind the metrics and models, see [Theory](../theory/index.md).

## From parquet to numpy

`metering_data.parquet` holds all 15 assets. The models take a 1-D numpy array for one
asset, so filter, sort, pull the target column, and hold back the last few days as the test
set. Data IO is polars; the models exchange numpy.

```python
import numpy as np
import polars as pl


def load_asset(asset_id: str, test_days: int = 4):
    df = pl.read_parquet("src/data/metering_data.parquet")
    asset = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")
    y = asset.select("metering_kwh").to_numpy().flatten()
    split = len(y) - test_days * 48  # 48 half-hours per day
    return y[:split], y[split:]


y_train, y_test = load_asset("ASSET_001", test_days=4)  # 480 train, 192 test
```

## Quick start

```python
import numpy as np
from ts_model_framework import (
    ModelComparison,
    SARIMAModel,
    ExponentialSmoothingModel,
    LightGBMModel,
)

y_train = np.array([...])  # e.g. 480 points = 10 days x 48 half-hours
y_test = np.array([...])  # e.g. 192 points = 4 days x 48 half-hours

comp = ModelComparison(y_train, y_test)
comp.add_model(SARIMAModel(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48)))
comp.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
comp.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))

comp.fit_all()
ranking = comp.evaluate_all()  # polars DataFrame ranked by RMSE
```

`evaluate_all()` returns a **polars** DataFrame. `SARIMAModel(..., order=(1, 0, 1), ...)` —
see [Models](../theory/models.md#sarimapdq-x-pdqs) for why `d=0` is the better default on
this trend-free data.

```python
from ts_plots import TSPlotter

best_name, metrics = comp.best_model()
forecast = comp.get_forecast(best_name)

TSPlotter.forecast_vs_actual(y_test, forecast, best_name, metrics)
TSPlotter.residuals_diagnostic(y_test, forecast, best_name)
TSPlotter.uncertainty_analysis(forecast, best_name)
TSPlotter.pi_coverage(y_test, forecast, best_name)
```

Interpreting the four plots is covered in [Diagnostics](../theory/diagnostics.md).

## Comparing all models

```python
from ts_plots import ComparisonPlotter

forecasts = {name: data["forecast"] for name, data in comp.results.items()}
metrics_dict = {name: data["metrics"] for name, data in comp.results.items()}

ComparisonPlotter.forecast_comparison(y_test, forecasts)  # overlaid forecasts
ComparisonPlotter.metrics_comparison(metrics_dict)  # MAE/RMSE/MAPE/coverage bar chart
```

## Output contracts

```python
@dataclass
class ForecastOutput:
    prediction: np.ndarray  # point forecast
    lower: np.ndarray  # lower bound (P10)
    upper: np.ndarray  # upper bound (P90)
    uncertainty_width: np.ndarray  # upper - lower


@dataclass
class EvaluationMetrics:
    mae: float
    rmse: float
    mape: float  # sklearn fraction (0-1); __repr__ prints it with a % suffix — a known bug
    pi_coverage: float  # percentage, 0-100 (not a fraction)
    mean_uncertainty_width: float
```

Interval semantics differ by model: SARIMA's `lower`/`upper` come from the state-space
confidence interval and widen with horizon; ExponentialSmoothing and LightGBM return a
single residual std broadcast across the whole horizon (`np.full_like`), so their
`uncertainty_width` is constant. See [Diagnostics](../theory/diagnostics.md).

## Tuning

`ModelTuner` grid-searches any model class against a param grid and ranks trials by RMSE.

!!! warning "The tuner selects on the test set"
    `ModelTuner.grid_search` fits each trial on `y_train` but evaluates and ranks every trial
    on `y_test` — the same data used to report final performance. The before/after RMSE
    improvement it prints is optimistically biased, and the bias grows with the grid size.
    Treat tuned metrics as an upper bound, not an estimate. A rolling-origin (walk-forward)
    split is the fix and is not yet implemented.

```python
from ts_model_framework import ModelTuner, SARIMAModel

tuner = ModelTuner(SARIMAModel, y_train, y_test)
param_grid = {
    "order": [(0, 1, 1), (1, 1, 1), (2, 1, 1)],
    "seasonal_order": [(1, 0, 1, 48), (1, 1, 1, 48)],
}
trials = tuner.grid_search(param_grid)
best_params = tuner.best_params()
```

Typical grids for the other two models:

```python
# ExponentialSmoothingModel
# "mul" needs strictly positive data — it will fail on the solar_battery assets (zero-crossing).
param_grid = {
    "trend": ["add", None],
    "seasonal": ["add"],
    "damped_trend": [True, False],
}

# LightGBMModel
# lag 336 (weekly) is not usefully estimable on 14 days of data.
param_grid = {
    "lags": [[1, 2, 48, 96], [1, 48, 96]],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
}
```

Refit the best model and compare before/after:

```python
from ts_model_framework import ModelEvaluator

final_model = SARIMAModel(y_train, **best_params)
final_model.fit()
final_forecast = final_model.forecast(len(y_test))
final_metrics = ModelEvaluator.evaluate(y_test, final_forecast)
print(f"RMSE: {baseline_metrics.rmse} -> {final_metrics.rmse}")
```

Because the tuner ranked on `y_test`, this "before -> after" gap is guaranteed to be
non-negative — it is not evidence the tuned model generalises better.

## Full workflow

```
1. ModelComparison across SARIMA / ExponentialSmoothing / LightGBM
2. Diagnostic plots for the best model
3. ComparisonPlotter across all models
4. ModelTuner grid search on the best model
5. Refit with tuned params, save forecast output (CSV) and ranking table
```

Deploy once `final_metrics` clears the [production-ready checklist](../theory/models-decisions.md#production-ready-checklist):

```python
if (
    final_metrics.pi_coverage > 75 and final_metrics.rmse < threshold
):  # coverage is 0-100
    save_model(final_model)
```

## Extending the framework

- New model: subclass `TSModel`, implement `fit()` and `forecast()` — plots and metrics work on it automatically.
- New metric: extend the `EvaluationMetrics` dataclass.
- New plot: add a method to `TSPlotter` or `ComparisonPlotter`.
- Different seasonality: change `seasonal_order`'s `s` (48 for daily on 30-min data; a weekly `s=336` is not supported by 14 days of history).

## Troubleshooting

| Symptom | Fix |
|---|---|
| SARIMA `RuntimeError` on fit | Test stationarity (ADF/KPSS) before adding `d`; reduce `seasonal_order` complexity, or pass `maxiter=2000` |
| ExponentialSmoothing singular-matrix warning | Data may have zero-variance stretches — use `trend="add"` / `seasonal="add"`, not `"mul"` |
| LightGBM `IndexError` when forecasting | Ensure `max(lags) < len(y_train)`; use smaller lags for short series (< 500 obs) |

## Files

`ts_model_framework.py` and `ts_plots.py` live in `src/` and are self-contained — no project-specific data loading inside them. Dependencies: `statsmodels`, `scikit-learn`, `lightgbm`, `scipy`, `numpy`, `plotly`. All dataframe IO is `polars`; the models exchange numpy arrays.
