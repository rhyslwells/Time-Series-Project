# Framework Usage

How to use `ts_model_framework.py` (models, comparison, tuning) and `ts_plots.py` (diagnostics) in `src/`. For the reasoning behind the metrics and models, see [Theory](../theory/index.md).

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
ranking = comp.evaluate_all()  # DataFrame ranked by RMSE
```

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
    mape: float
    pi_coverage: float
    mean_uncertainty_width: float
```

## Tuning

`ModelTuner` grid-searches any model class against a param grid and ranks trials by RMSE:

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
param_grid = {
    "trend": ["add", "mul"],
    "seasonal": ["add", "mul"],
    "damped_trend": [True, False],
}

# LightGBMModel
param_grid = {
    "lags": [[1, 2, 48, 96], [1, 48, 336]],  # up to 2 days vs up to 1 week
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
}
```

Refit the best model and compare before/after:

```python
final_model = SARIMAModel(y_train, **best_params)
final_model.fit()
final_forecast = final_model.forecast(len(y_test))
final_metrics = ModelEvaluator.evaluate(y_test, final_forecast)
print(f"RMSE: {metrics.rmse} -> {final_metrics.rmse}")
```

## Full workflow

```
1. ModelComparison across SARIMA / ExponentialSmoothing / LightGBM
2. Diagnostic plots for the best model
3. ComparisonPlotter across all models
4. ModelTuner grid search on the best model
5. Refit with tuned params, save forecast output (CSV) and ranking table
```

Deploy once `final_metrics` clears the [production-ready checklist](../theory/model-decisions.md#production-ready-checklist):

```python
if final_metrics.pi_coverage > 0.75 and final_metrics.rmse < threshold:
    save_model(final_model)
```

## Extending the framework

- New model: subclass `TSModel`, implement `fit()` and `forecast()` — plots and metrics work on it automatically.
- New metric: extend the `EvaluationMetrics` dataclass.
- New plot: add a method to `TSPlotter` or `ComparisonPlotter`.
- Different seasonality: change `seasonal_order`'s `s` (48 for daily, 336 for weekly on 30-min data).

## Troubleshooting

| Symptom | Fix |
|---|---|
| SARIMA `RuntimeError` on fit | Check stationarity (try d=1 or 2), reduce `seasonal_order` complexity, or pass `maxiter=2000` |
| ExponentialSmoothing singular-matrix warning | Data may have zero-variance stretches — try `trend="add"` instead of `"mul"` |
| LightGBM `IndexError` when forecasting | Ensure `max(lags) < len(y_train)`; use smaller lags for short series (< 500 obs) |

## Files

`ts_model_framework.py` and `ts_plots.py` live in `src/` and are self-contained — no project-specific data loading inside them. Dependencies: `statsmodels`, `lightgbm` (optional), `scikit-learn`, `plotly`, `numpy`, `pandas`.
