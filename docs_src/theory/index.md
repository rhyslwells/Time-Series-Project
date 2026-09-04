# Time Series Forecasting — Complete Reference

## Overview

MAybe we should move the imppmentation of code to here C:\Users\RhysL\Desktop\Time Series Project\docs_src\coding

with theory of models and math ect in 

C:\Users\RhysL\Desktop\Time Series Project\docs_src\theory

This file is a index and should just point to other files

This reference covers everything needed to understand, build, compare, and deploy time series forecasts for assets

**Structure:**
- **Code:** `ts_model_framework.py` (models + tuning) + `ts_plots.py` (plots)
- **Theory:** `MATH_AND_INTERPRETATION.md` (formulas + reasoning)
- **Examples:** `VISUAL_EXAMPLES.md` (good vs bad plots)
- **Quick lookup:** `METRIC_REFERENCE.md` (ranges + tables)
- **Examples:** `example_model_comparison.py` (full workflow)

---

## Quick Start (5 Minutes)

### Step 1: Load Data
```python
import numpy as np
from ts_model_framework import (
    ModelComparison,
    SARIMAModel,
    ExponentialSmoothingModel,
    LightGBMModel,
)

# Load your time series
y_train = np.array([...])  # 480 points (10 days × 48 half-hours)
y_test = np.array([...])  # 192 points (4 days × 48 half-hours)
```

### Step 2: Compare Models
```python
comp = ModelComparison(y_train, y_test)
comp.add_model(SARIMAModel(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48)))
comp.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
comp.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))

comp.fit_all()
ranking = comp.evaluate_all()
print(ranking)
```

### Step 3: Plot Results
```python
from ts_plots import TSPlotter, ComparisonPlotter

best_name, metrics = comp.best_model()
forecast = comp.get_forecast(best_name)

# Diagnostics
TSPlotter.forecast_vs_actual(y_test, forecast, best_name, metrics)
TSPlotter.residuals_diagnostic(y_test, forecast, best_name)
TSPlotter.pi_coverage(y_test, forecast, best_name)
```

### Step 4: Interpret
- **MAE ~0.5 kWh?** → Good point forecast
- **Coverage ~80%?** → Intervals well-calibrated
- **RMSE/MAE < 1.3?** → No huge outliers
- → **Deploy**

---


---

## The Models Explained Simply

### SARIMA (Seasonal ARIMA)

**What it does:**
Learns patterns from past values and past forecast errors, handles daily seasonality.

**Good for:**
- Stable patterns (residential)
- Strong daily cycle
- Needs quick fit

**Parameters to tune:**
- `order=(p,d,q)`: p→past values, d→smoothing, q→error correction
- `seasonal_order=(P,D,Q,s)`: P,D,Q→seasonal equivalent, s=48 for daily

**Starting point:**
```python
SARIMA(1,1,1) × (1,1,1,48)  # Works for most 30-min energy data
```

**If RMSE is high:**
- Increase p: `(2,1,1)` → Captures more past-value dependencies
- Increase d: `(1,2,1)` → Removes stronger trends
- Increase q: `(1,1,2)` → Corrects forecast errors better

### Exponential Smoothing (Holt-Winters)

**What it does:**
Weighted average of past values, adapts smoothly to level, trend, seasonality.

**Good for:**
- Simpler patterns
- Stable seasonality
- Fast fitting

**Parameters to tune:**
- `trend`: "add" (constant offset) or "mul" (multiplies by level)
- `seasonal`: "add" (constant offset) or "mul" (multiplies by level)
- `damped_trend`: True/False (flatten trend over time?)

**Starting point:**
```python
ExponentialSmoothingModel(y_train, trend="add", seasonal="add")
```

**If coverage is too low:**
- Try multiplicative: `seasonal="mul"`
- Add damping: `damped_trend=True`

### LightGBM (Gradient Boosting)

**What it does:**
Ensemble of decision trees, learns non-linear relationships from lags.

**Good for:**
- Complex patterns
- Non-linear demand curves
- Multiple lag scales

**Parameters to tune:**
- `lags`: Which past values matter? [1, 2, 48, 96] = 30min, 1hr, 1day, 2day
- `num_leaves`: Tree complexity (15=simple, 63=complex)
- `learning_rate`: How fast to learn (0.01=slow, 0.1=fast)

**Starting point:**
```python
LightGBMModel(y_train, lags=[1, 2, 48, 96], num_leaves=31, learning_rate=0.05)
```

**If RMSE is high:**
- Add longer lags: `[1, 2, 48, 96, 336]` → Include weekly pattern
- Increase capacity: `num_leaves=63` → More complex tree

---

## Interpreting Plots

### Plot 1: Forecast vs Actual (With Prediction Intervals)

```
What to look for:
✓ Actual (black) mostly inside shaded region (PI)
✓ Forecast (blue) follows actual, lags by ~1-2 timesteps
✓ PI widens/narrows with model confidence
✓ No systematic bias (actuals not all above/below forecast)

If wrong:
✗ Actual frequently outside PI → Coverage too low, intervals too narrow
✗ Forecast flat-lines → Model underfitted
✗ PI always same width → Not adapting to conditions
```

**What it tells you:**
- "Is the point forecast close to actual?" (MAE visual check)
- "Are the confidence bounds reasonable?" (PI coverage visual check)
- "Does uncertainty adapt to time-of-day?" (PI width variation)

---

### Plot 2: Residuals Diagnostic

**Left panel: Residuals over time**
```
What to look for:
✓ Bars bounce randomly between ±1-2 range
✓ Red line (uncertainty) ≈ 2 × typical |bar height|
✓ No trend (doesn't drift up/down)
✓ Occasional spikes normal

If wrong:
✗ All positive or all negative → Systematic bias
✗ Clear pattern (clustering) → Autocorrelated (model missed structure)
✗ Spikes outside red bands regularly → Uncertainty miscalibrated
```

**Right panel: Histogram**
```
What to look for:
✓ Bell-shaped (normal distribution)
✓ Centered at 0
✓ Symmetric (not skewed left or right)

If wrong:
✗ Skewed right → Forecast systematically too low
✗ Skewed left → Forecast systematically too high
✗ Heavy tails → Extreme events more common than model thinks
```

**What it tells you:**
- "Are residuals white noise?" (Random = good)
- "Is there systematic bias?" (All positive/negative = bad)
- "Are residuals normal?" (Histogram shape = affects PI validity)

---

### Plot 3: Uncertainty Width Over Time

```
What to look for:
✓ Width varies by time-of-day (narrow at noon, wide at transition)
✓ Mean width matches 2 × 1.28 × std(residuals)
✓ Width correlates with demand volatility

If wrong:
✗ Constant everywhere → Model not adapting
✗ Exploding over forecast horizon → Model losing confidence too fast
✗ Much wider than expected → Over-conservative, wasting opportunity
```

**What it tells you:**
- "Does model know when it's confident?" (Narrow in stable periods ✓)
- "Is uncertainty calibrated?" (Width = 2 × z × std ✓)

---

### Plot 4: PI Coverage (Green/Red Scatter)

```
What to look for:
✓ ~80% green dots (actual in bounds)
✓ ~20% red dots (actual outside bounds)
✓ Reds scattered randomly (no clustering by hour/day)

If wrong:
✗ Mostly red (< 50% coverage) → Intervals too narrow, over-confident
✗ All green (> 95% coverage) → Intervals too wide, under-confident
✗ Reds only in morning hours → Model doesn't understand time-of-day
```

**What it tells you:**
- "Are my confidence claims accurate?" (Coverage ≈ target confidence ✓)
- "When do I mess up?" (Where are the reds?)

---

## Decision Tree: Which Model is Best?

```
START
│
├─ Rank by RMSE (lower = better)
│
├─ CHECK: Is best model's coverage 75-85%?
│  ├─ YES → Use it!
│  └─ NO → Check if too high (widen) or too low (narrow)
│
├─ CHECK: RMSE/MAE ratio < 1.5?
│  ├─ YES → Consistent, no huge outliers ✓
│  └─ NO → Check residuals for outliers
│
├─ CHECK: MAPE < 15% (or asset-specific threshold)?
│  ├─ YES → Good
│  └─ NO → Needs more tuning
│
└─ If 3+ ✓: DEPLOY
   If 2+ ✗: TUNE MODEL
```

---

## When to Retrain

| Event | Action | Why |
|---|---|---|
| Coverage drops < 70% | Retrain immediately | Under-confident, risky |
| RMSE increases 20% | Retrain | Concept drift (patterns changed) |
| Forecast flat-lines | Retrain + check parameters | Model gave up |
| New season (spring/summer/fall/winter) | Retrain on recent 14 days | Patterns shift with weather |
| Equipment changes (new HVAC, solar installed) | Retrain + inspect residuals | Asset behavior changed |

---

## Expected Metrics by Asset Type

### Residential (e.g., house, small office)
```
MAE: 0.3-0.5 kWh     ← Patterns stable
RMSE: 0.4-0.6 kWh    ← Few outliers
MAPE: 8-12%          ← Predictable
PI Coverage: 78-82%  ← Well-calibrated
Best model: ExponentialSmoothing or SARIMA
```

### Commercial (e.g., office building, retail)
```
MAE: 0.6-1.0 kWh     ← More variable (occupancy)
RMSE: 0.8-1.3 kWh    ← Some spikes
MAPE: 12-18%         ← Harder to predict
PI Coverage: 76-84%  ← May need tuning
Best model: SARIMA(2,1,1)×(1,1,1,48) or LightGBM
```

### EV Charging Station
```
MAE: 1.0-3.0 kWh     ← Highly variable
RMSE: 1.5-4.0 kWh    ← Frequent spikes
MAPE: 25-40%         ← Chaotic
PI Coverage: 70-80%  ← Intervals must be wide
Best model: LightGBM with external features
                      (price, demand response signals)
```

---

## Key Formulas (One-Pagers)

### Metrics

```python
# MAE: Average absolute error
mae = mean(abs(y_true - forecast))

# RMSE: Root mean squared error (penalizes outliers)
rmse = sqrt(mean((y_true - forecast) ** 2))

# MAPE: Mean absolute percentage error
mape = mean(abs((y_true - forecast) / y_true)) * 100

# PI Coverage: % of actuals within bounds
coverage = mean((y_true >= lower) & (y_true <= upper)) * 100
```

### Prediction Intervals (80% Confidence)

```python
# Standard approach (assumes normal residuals)
z_80 = 1.282  # From normal distribution
upper = forecast + z_80 * std(residuals)
lower = forecast - z_80 * std(residuals)

# Check if calibrated:
expected_width = 2 * z_80 * std(residuals)
actual_width = upper - lower
if abs(expected_width - actual_width) < 0.2:
    print("Well-calibrated ✓")
```

### SARIMA Order Selection

```python
# Max reasonable complexity:
p + q < len(y_train) / 10

# For 480-point training set:
# Max p + q < 48

# Conservative starting point:
SARIMA(1,1,1) × (1,1,1,48)

# Increase complexity if RMSE high:
SARIMA(2,1,1) × (1,1,1,48)  # More AR
SARIMA(1,2,1) × (1,1,1,48)  # More differencing
SARIMA(1,1,2) × (1,1,1,48)  # More MA
```

---

## Troubleshooting Checklist

### Problem: High RMSE (> threshold)

**Diagnosis:**
```python
if forecast.std() < 0.1:
    print("Forecast flat-lined (underfitted)")
    # → Increase p, d, q
elif residual_mean > 0.2:
    print("Systematic bias (forecast too low)")
    # → Check differencing, retrain on recent data
elif rmse / mae > 1.5:
    print("Occasional huge errors (outliers)")
    # → Check data quality, use robust model
```

**Fixes (in order):**
1. Increase differencing: d=1 → d=2
2. Increase AR: p=1 → p=2
3. Switch to LightGBM (handles non-linear better)
4. Retrain on recent data (seasonal patterns shifted)

---

### Problem: Coverage Too Low (< 70%)

**Diagnosis:**
```python
if coverage < 50%:
    print("Model massively over-confident (intervals way too narrow)")
    # → Inflate intervals manually, or retrain with damped_trend=True
elif coverage < 75%:
    print("Model somewhat over-confident (needs calibration)")
    # → Widen intervals by 20%, or improve residual estimation
```

**Fixes (in order):**
1. Manually widen: `upper *= 1.2, lower *= 0.8`
2. Retrain model
3. Use quantile regression (predict P10/P90 directly)
4. Add damping: `damped_trend=True` for ExponentialSmoothing

---

### Problem: Systematic Bias (All residuals positive/negative)

**Diagnosis:**
```python
if mean(residuals) > 0.3:
    print("Forecast too low (underestimating)")
    # → Check if trend increasing, retrain on recent data
elif mean(residuals) < -0.3:
    print("Forecast too high (overestimating)")
    # → Check if data quality, outliers
```

**Fixes (in order):**
1. Retrain on recent data only (last 7–14 days)
2. Ensure differencing present: d ≥ 1
3. Check for missing external variables (temperature, holidays)
4. Decompose series (separate level, trend, seasonality)

---

## Full Workflow Example

```python
from ts_model_framework import (
    ModelComparison,
    SARIMAModel,
    ExponentialSmoothingModel,
    LightGBMModel,
    ModelTuner,
)
from ts_plots import TSPlotter, ComparisonPlotter

# 1. COMPARE
comp = ModelComparison(y_train, y_test)
comp.add_model(SARIMAModel(y_train))
comp.add_model(ExponentialSmoothingModel(y_train))
comp.add_model(LightGBMModel(y_train))
comp.fit_all()
ranking = comp.evaluate_all()
print(ranking)  # See which model wins by RMSE

# 2. DIAGNOSE BEST
best_name, metrics = comp.best_model()
forecast = comp.get_forecast(best_name)

TSPlotter.forecast_vs_actual(y_test, forecast, best_name, metrics)
TSPlotter.pi_coverage(y_test, forecast, best_name)

# Check metrics:
if metrics.pi_coverage < 75:
    print("WARNING: Coverage too low, tuning needed")
if metrics.rmse > threshold:
    print("WARNING: RMSE too high, tuning needed")

# 3. TUNE BEST MODEL
if tuning_needed:
    tuner = ModelTuner(SARIMAModel, y_train, y_test)
    param_grid = {
        "order": [(1, 1, 1), (2, 1, 1), (1, 2, 1)],
        "seasonal_order": [(1, 0, 1, 48), (1, 1, 1, 48)],
    }
    trials = tuner.grid_search(param_grid)
    best_params = tuner.best_params()

    # Refit with tuned params
    final_model = SARIMAModel(y_train, **best_params)
    final_model.fit()
    final_forecast = final_model.forecast(len(y_test))
    final_metrics = ModelEvaluator.evaluate(y_test, final_forecast)

    print(f"Improvement: {metrics.rmse} → {final_metrics.rmse}")

# 4. COMPARE FORECASTS
all_forecasts = {name: comp.get_forecast(name) for name in comp.results.keys()}
ComparisonPlotter.forecast_comparison(y_test, all_forecasts)

# 5. DEPLOY
if final_metrics.pi_coverage > 0.75 and final_metrics.rmse < threshold:
    save_model(final_model)
    print("✓ Ready to deploy!")
```

---

## File Organization (For Obsidian)

```
TimeSeries/
├── 📄 README_FORECASTING.md          ← This file
├── 📄 MATH_AND_INTERPRETATION.md    ← Theory & formulas
├── 📄 VISUAL_EXAMPLES.md            ← Good vs bad plots
├── 📄 METRIC_REFERENCE.md           ← Quick lookup tables
├── 💾 ts_model_framework.py          ← Code (models + tuning)
├── 💾 ts_plots.py                    ← Code (plotting)
└── 💾 example_model_comparison.py    ← Full example workflow
```

---

## Next Steps

1. **Copy code to your repo**
   ```bash
   cp ts_model_framework.py your_project/
   cp ts_plots.py your_project/
   ```

2. **Test on your data**
   ```bash
   python example_model_comparison.py
   ```

3. **Integrate into VS Code workflow**
   - Reference this README in `.claude/skills/` for structured prompts
   - Use `MODEL_COMPARISON_GUIDE.md` for quick recipes

4. **Monitor in production**
   - Track metrics weekly (coverage, RMSE, MAPE)
   - Retrain when coverage drops or RMSE increases 20%
   - Log residuals to detect concept drift

---

## Questions? Use This Path

| Question | Document | Section |
|---|---|---|
| "What's a good MAE?" | METRIC_REFERENCE.md | Metrics at a Glance |
| "How do I interpret this plot?" | VISUAL_EXAMPLES.md | Good vs Bad Cases |
| "Why use this formula?" | MATH_AND_INTERPRETATION.md | Part 1–5 |
| "When should I retrain?" | This document | When to Retrain |
| "How do I tune SARIMA?" | MODEL_COMPARISON_GUIDE.md | Tune Best Model |
| "Is my forecast ready to deploy?" | VISUAL_EXAMPLES.md | Section 6 Checklist |

---

**Version:** 1.0
**Last updated:** 2025-01-15
**For:**  FlexGo forecasting
