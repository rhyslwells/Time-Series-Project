# Time Series Model Comparison & Tuning Framework

## Overview

Generalised framework for comparing multiple forecasting models, diagnosing results, and tuning hyperparameters.

**Three layers:**
1. **Core models** (`ts_model_framework.py`) — SARIMA, ExponentialSmoothing, LightGBM
2. **Plotting** (`ts_plots.py`) — Generic diagnostic + comparison plots
3. **Workflow** (`example_model_comparison.py`) — End-to-end comparison → tuning

---

## Quick Start

### 1. Compare three models

```python
from ts_model_framework import (
    SARIMAModel, ExponentialSmoothingModel, LightGBMModel,
    ModelComparison
)

comparator = ModelComparison(y_train, y_test)

# Add models
comparator.add_model(SARIMAModel(y_train, order=(1,1,1), seasonal_order=(1,1,1,48)))
comparator.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
comparator.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))

# Fit and evaluate
comparator.fit_all()
ranking = comparator.evaluate_all()
print(ranking)

# Get best
best_name, best_metrics = comparator.best_model()
```

**Output:** DataFrame ranked by RMSE with MAE, MAPE, PI Coverage % columns.

---

### 2. Diagnostic plots for any model

```python
from ts_plots import TSPlotter

forecast = comparator.get_forecast("SARIMA")
metrics = comparator.results["SARIMA"]['metrics']

# Plot 1: Forecast vs Actual with PI
fig1 = TSPlotter.forecast_vs_actual(y_test, forecast, "SARIMA", metrics)

# Plot 2: Residuals over time + histogram
fig2 = TSPlotter.residuals_diagnostic(y_test, forecast, "SARIMA")

# Plot 3: Uncertainty width
fig3 = TSPlotter.uncertainty_analysis(forecast, "SARIMA")

# Plot 4: PI Coverage (actual vs bounds)
fig4 = TSPlotter.pi_coverage(y_test, forecast, "SARIMA")
```

All plots work for any model — no special handling needed.

---

### 3. Compare all model forecasts

```python
from ts_plots import ComparisonPlotter

forecasts = {name: data['forecast'] for name, data in comparator.results.items()}
metrics_dict = {name: data['metrics'] for name, data in comparator.results.items()}

# Overlaid forecasts (all models)
fig1 = ComparisonPlotter.forecast_comparison(y_test, forecasts)

# Bar chart: MAE, RMSE, MAPE, PI Coverage
fig2 = ComparisonPlotter.metrics_comparison(metrics_dict)
```

---

### 4. Tune hyperparameters

#### SARIMA

```python
from ts_model_framework import ModelTuner, SARIMAModel

tuner = ModelTuner(SARIMAModel, y_train, y_test)

param_grid = {
    "order": [(0,1,1), (1,1,0), (1,1,1), (1,1,2), (2,1,0)],
    "seasonal_order": [(0,0,0,48), (1,0,1,48), (1,1,1,48), (0,1,1,48)]
}

trials = tuner.grid_search(param_grid)  # Returns ranked DataFrame
best_params = tuner.best_params()
```

**Best params example:**
```
{'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 48)}
```

#### ExponentialSmoothing

```python
tuner = ModelTuner(ExponentialSmoothingModel, y_train, y_test)

param_grid = {
    "trend": ["add", "mul"],
    "seasonal": ["add", "mul"],
    "damped_trend": [True, False]
}

trials = tuner.grid_search(param_grid)
best_params = tuner.best_params()
```

#### LightGBM

```python
tuner = ModelTuner(LightGBMModel, y_train, y_test)

param_grid = {
    "lags": [[1, 2, 48, 96], [1, 48, 336]],  # 30min, 1day, 1week
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1]
}

trials = tuner.grid_search(param_grid)
best_params = tuner.best_params()
```

---

## Standard Output Contract

Every model returns:

```python
@dataclass
class ForecastOutput:
    prediction: np.ndarray        # Point forecast
    lower: np.ndarray              # Lower bound (P10)
    upper: np.ndarray              # Upper bound (P90)
    uncertainty_width: np.ndarray  # upper - lower
```

**Metrics:**

```python
@dataclass
class EvaluationMetrics:
    mae: float                  # Mean Absolute Error
    rmse: float                 # Root Mean Squared Error
    mape: float                 # Mean Absolute Percentage Error
    pi_coverage: float          # % of actuals within [lower, upper]
    mean_uncertainty_width: float
```

---

## Model-Specific Notes

### SARIMA(p,d,q)×(P,D,Q,s)

**Common params for 30-min energy data:**
- `order=(1,1,1)` — Non-seasonal AR, differencing, MA
- `seasonal_order=(1,1,1,48)` — Seasonal with s=48 (1 day)

**When to tune:**
- High residual autocorrelation (ACF plot)
- Systematic bias in forecasts
- Poor RMSE vs baseline

**Grid search strategy:**
- Start with (p,d,q) ∈ {0,1,2} × {0,1} × {0,1}
- Fix s=48, vary (P,D,Q) ∈ {0,1} × {0,1} × {0,1}
- Optimize by RMSE

### ExponentialSmoothing

**Parameters:**
- `trend="add"` or `"mul"` — Additive or multiplicative trend
- `seasonal="add"` or `"mul"` — Additive or multiplicative seasonality
- `damped_trend=True/False` — Dampen trend to flatten over time

**When to use:**
- Simpler than SARIMA, faster to fit
- Good for stable, seasonal patterns
- Less prone to overfitting

### LightGBM

**Lag-based features:**
- `lags=[1, 2, 48, 96]` → 30-min, 1-hour, 1-day, 2-day lags
- Automatically creates hour-of-day feature (0–1 scaled)

**Hyperparameters:**
- `num_leaves` — Tree complexity (15 simpler, 63 complex)
- `learning_rate` — Step size (0.01 slow, 0.1 fast)

**When to use:**
- When you have strong lag structure
- Portfolio-level forecasts (feed in multiple assets)
- Feature engineering flexibility

---

## Workflow (from example_model_comparison.py)

```
Step 1: Compare 3 baseline models
         ↓
Step 2: Diagnostics for best model (4 plots)
         ↓
Step 3: Compare all model forecasts (overlaid + metrics)
         ↓
Step 4: Grid search best model's hyperparameters
         ↓
Step 5: Refit with tuned params, save final output
```

**Saves:**
- HTML plots (interactive, shareable)
- CSV forecast output (prediction, lower, upper, actual)
- Ranking tables (model comparison)

---

## Troubleshooting

### SARIMA fails to fit

**Symptom:** `RuntimeError: SARIMA fit failed`

**Solution:**
- Check stationarity (d=1 or 2 usually works)
- Reduce seasonal_order complexity
- Try `maxiter=2000` in fit()

### ExponentialSmoothing warns about singular matrix

**Symptom:** Warnings but model still works

**Solution:**
- Data may have repeated values or zero variance
- Try `trend="add"` instead of `"mul"`

### LightGBM lags mismatch

**Symptom:** `IndexError` when forecasting

**Solution:**
- Ensure max(lags) < len(y_train)
- For short series (<500 obs), use smaller lags

---

## Next Steps

1. **Add more models** — Extend `TSModel` base class (VARIMA, Prophet, LSTM)
2. **Residual tests** — Ljung-Box for autocorrelation, Shapiro-Wilk for normality
3. **Walk-forward validation** — Iterative retraining + testing
4. **Ensemble** — Average forecasts from top 2–3 models
5. **Real-time monitoring** — Track prediction errors over time (concept drift detection)

---

## File Structure

```
/
├── ts_model_framework.py        # Core: Models, Evaluation, Tuning
├── ts_plots.py                  # Plotting: Diagnostic + Comparison
├── example_model_comparison.py  # Full workflow
└── MODEL_COMPARISON_GUIDE.md    # This file
```

All files are self-contained; copy to your project and import as needed.
