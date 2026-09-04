# Quick Reference: Metrics, Ranges, and Expected Outputs

## Metrics at a Glance

### MAE (Mean Absolute Error)

| Energy Asset Type | Good MAE | Acceptable | Poor |
|---|---|---|---|
| Residential | < 0.5 kWh | 0.5–1.0 kWh | > 1.5 kWh |
| Commercial | < 1.0 kWh | 1.0–1.5 kWh | > 2.0 kWh |
| EV Charging | < 2.0 kWh | 2.0–4.0 kWh | > 5.0 kWh |

**Interpretation:** Average absolute difference between forecast and actual
**Better:** Lower is better (0 = perfect)
**Red flag:** If significantly higher than RMSE ÷ 1.3 (indicates overfitting to outliers)

---

### RMSE (Root Mean Squared Error)

| Energy Asset Type | Good RMSE | Acceptable | Poor |
|---|---|---|---|
| Residential | < 0.7 kWh | 0.7–1.0 kWh | > 1.5 kWh |
| Commercial | < 1.3 kWh | 1.3–1.8 kWh | > 2.5 kWh |
| EV Charging | < 2.5 kWh | 2.5–5.0 kWh | > 7.0 kWh |

**Interpretation:** Penalizes large errors more than MAE
**Better:** Lower is better
**Red flag:** If RMSE >> MAE, model has occasional huge misses

**Diagnostic:**
```python
rmse_mae_ratio = rmse / mae
if rmse_mae_ratio < 1.1:
    print("Consistent errors (no big outliers) ✓")
elif 1.1 < rmse_mae_ratio < 1.5:
    print("Some outliers (normal) ✓")
else:  # > 1.5
    print("Frequent large errors (problem!) ✗")
```

---

### MAPE (Mean Absolute Percentage Error)

| Energy Asset Type | Good MAPE | Acceptable | Poor |
|---|---|---|---|
| Residential | < 10% | 10–15% | > 20% |
| Commercial | < 15% | 15–20% | > 25% |
| EV Charging | < 30% | 30–40% | > 50% |

**Interpretation:** Percentage error (scale-independent)
**Better:** Lower is better (% closer to 0)
**Caution:** Undefined if actual ≈ 0 (use max(|actual|, ε) in denominator)

**Scale-independent comparison:**
```python
# Asset A: MAE = 0.1 kWh (seems good)
#          Mean(actual) = 10 kWh
#          MAPE = 0.1/10 = 1%

# Asset B: MAE = 0.2 kWh (seems bad)
#          Mean(actual) = 1 kWh  
#          MAPE = 0.2/1 = 20%

# MAPE correctly ranks B worse!
```

---

### PI Coverage (%)

| Target Confidence | Good Coverage | Acceptable | Poor |
|---|---|---|---|
| 80% PI | 78–82% | 75–85% | < 70% or > 90% |
| 90% PI | 88–92% | 85–95% | < 85% or > 97% |
| 95% PI | 93–97% | 90–98% | < 90% or > 99% |

**Interpretation:** % of actual values within [lower, upper] bounds
**Better:** Matches target confidence level (80% target → 80% coverage)

**When coverage is wrong:**

```
Coverage Too Low (e.g., 45% when target 80%):
  → Intervals too narrow
  → Model over-confident
  → Flexibility commitments will breach

Coverage Too High (e.g., 95% when target 80%):
  → Intervals too wide
  → Model under-confident
  → Wasted conservatism (could sell more flexibility)

Ideal: Coverage ≈ target ± 5%
```

---

### Uncertainty Width

| Asset Type | Good Width | Acceptable | Note |
|---|---|---|---|
| Residential | 0.8–1.5 kWh | 1.5–2.0 kWh | Narrow = confident |
| Commercial | 1.5–2.5 kWh | 2.5–3.5 kWh | Should vary by hour |
| EV Charging | 3.0–6.0 kWh | 6.0–10.0 kWh | Large uncertainty normal |

**Interpretation:** Upper − Lower (for 80% PI)
**Better:** Matches true forecast variability
**Red flag:** If constant everywhere (should vary by time-of-day)

**Relationship to residual std:**
```python
# For 80% confidence interval:
expected_width = 2 * 1.28 * std(residuals)
#                         ↑
#                     z-score for 80% PI

# Example:
std(residuals) = 0.4 kWh
expected_width = 2 * 1.28 * 0.4 = 1.024 kWh

# Check if model matches:
actual_width = np.mean(forecast.uncertainty_width)
if abs(actual_width - expected_width) < 0.2:
    print("Width well-calibrated ✓")
else:
    print("Width miscalibrated ✗ (check PI calculation)")
```

---

## Model Output Examples

### Example 1: Good Forecast (Residential Building)

```python
# Input:
y_test = [2.1, 3.2, 2.8, 3.5, 2.3, 3.1, 2.9]  # 7 hours of actuals

# Output from SARIMA(1,1,1)×(1,1,1,48):
forecast.prediction = [2.0, 3.1, 2.9, 3.4, 2.2, 3.0, 2.8]
forecast.lower      = [1.2, 2.3, 2.1, 2.6, 1.4, 2.2, 2.0]
forecast.upper      = [2.8, 3.9, 3.7, 4.2, 3.0, 3.8, 3.6]
forecast.uncertainty_width = [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6]

# Metrics:
mae       = mean(abs([2.1-2.0, 3.2-3.1, 2.8-2.9, 3.5-3.4, 2.3-2.2, 3.1-3.0, 2.9-2.8]))
          = mean([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
          = 0.1 kWh  ← Excellent!

rmse      = sqrt(mean([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
          = 0.1 kWh

coverage  = 7/7 = 100%  ← All actuals in bounds (slightly over-confident but OK)

# Interpretation:
"Forecast is off by ~0.1 kWh on average"
"I'm 80% confident demand will be 3.0 kWh ± 0.8 kWh"
"All actual values fell within predicted bounds"
✓ Production-ready
```

---

### Example 2: Acceptable Forecast (Commercial Building)

```python
# Input:
y_test = [4.2, 5.8, 4.5, 6.2, 4.1, 5.5, 4.8]  # Peak hours have more variance

# Output from SARIMA(1,1,1)×(1,1,1,48):
forecast.prediction = [4.1, 5.5, 4.8, 5.9, 4.3, 5.3, 4.6]
forecast.lower      = [2.8, 4.2, 3.5, 4.6, 3.0, 4.0, 3.3]
forecast.upper      = [5.4, 6.8, 6.1, 7.2, 5.6, 6.6, 5.9]
forecast.uncertainty_width = [2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6]

# Errors:
errors = [0.1, 0.3, -0.3, 0.3, -0.2, 0.2, 0.2]

# Metrics:
mae       = mean(abs(errors))
          = mean([0.1, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2])
          = 0.23 kWh

rmse      = sqrt(mean([0.01, 0.09, 0.09, 0.09, 0.04, 0.04, 0.04]))
          = sqrt(0.057)
          = 0.24 kWh

coverage  = 6/7 = 85.7%  ← One outside (2.5 kWh forecast, actual 5.8 kWh)
                           But 85.7% ≈ 80% target ✓

# Interpretation:
"Forecast off by ~0.23 kWh on average (acceptable for commercial)"
"One spike caught the model by surprise (outside bounds)"
"Overall coverage 85% ≈ target (good)"
"Uncertainty width (2.6 kWh) reasonable for commercial"
✓ Acceptable, monitoring needed
```

---

### Example 3: Poor Forecast (Missing Pattern)

```python
# Input:
y_test = [2.1, 2.9, 3.5, 4.2, 3.1, 2.5, 1.8]  # Morning ramp-up

# Output from SARIMA(1,0,0)×(0,0,0,48):  ← No seasonal differencing!
forecast.prediction = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]  ← Flat!
forecast.lower      = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
forecast.upper      = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
forecast.uncertainty_width = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

# Errors:
errors = [-0.9, -0.1, 0.5, 1.2, 0.1, -0.5, -1.2]

# Metrics:
mae       = mean(abs(errors))
          = 0.78 kWh  ← High

rmse      = sqrt(mean([0.81, 0.01, 0.25, 1.44, 0.01, 0.25, 1.44]))
          = sqrt(0.60)
          = 0.77 kWh

coverage  = 7/7 = 100%  ← All in bounds (but bounds too wide!)

# Red flags:
✗ Forecast flat (doesn't adapt to ramp-up pattern)
✗ MAE = 0.78 kWh (too high for residential)
✗ Coverage 100% but width 3.0 kWh (over-conservative)
✗ RMSE/MAE ratio ≈ 1.0 (no large outliers, just consistently off)

# Diagnosis:
"Model doesn't capture daily pattern"
"SARIMA(1,0,0) without seasonal differencing is insufficient"
"Seasonal pattern at s=48 not being handled"

# Fix:
SARIMA(1,1,1)×(1,1,1,48)  ← Add differencing and seasonal component
```

---

## Decision Tables

### When to Retrain

| Condition | Action | Frequency |
|---|---|---|
| Coverage drops below 70% | Retrain, inflate intervals | Immediately |
| MAE increases 20% | Retrain, check for concept drift | Weekly |
| RMSE/MAE ratio > 2.0 | Check for outliers/data quality | When noticed |
| Residual std increases | Model less confident (expected) | Monitor |
| Forecast flat-lines | Retrain, increase p or q | Immediately |

---

### When to Switch Models

| Symptom | Try | Reason |
|---|---|---|
| MAPE > 25%, cannot improve | LightGBM + features | Need non-linear fit |
| Coverage erratic (no pattern) | Quantile regression | Residuals non-normal |
| Concept drift (patterns change) | Refit last 7–14 days only | Model adapts faster |
| Residuals autocorrelated | Increase d or D | Differencing insufficient |
| Huge spikes missed | LightGBM or robust model | Linear models fail |

---

### Tuning Priority Matrix

| Problem | Priority | Fix |
|---|---|---|
| Coverage < 75% | **High** | Inflate intervals or retrain |
| MAE > threshold | **High** | Increase model complexity (p, num_leaves) |
| RMSE/MAE > 1.5 | **Medium** | Identify/remove outliers |
| Uncertainty flat | **Medium** | Use adaptive intervals (quantile regression) |
| Coverage 85–95% | **Low** | Acceptable, monitor only |

---

## Formula Reference

### Z-Scores for Prediction Intervals

```python
from scipy.stats import norm

# 80% confidence interval
z_80 = norm.ppf(0.90)      # 1.282
upper_80 = forecast + z_80 * std_residuals
lower_80 = forecast - z_80 * std_residuals

# 90% confidence interval  
z_90 = norm.ppf(0.95)      # 1.645
upper_90 = forecast + z_90 * std_residuals
lower_90 = forecast - z_90 * std_residuals

# 95% confidence interval
z_95 = norm.ppf(0.975)     # 1.960
upper_95 = forecast + z_95 * std_residuals
lower_95 = forecast - z_95 * std_residuals
```

### SARIMA Parameters Quick Check

```python
# Time series length n:
p + q should be < n/10

# For 480-point training set (10 days × 48 half-hours):
# Max reasonable: p + q < 48

# Recommended for energy:
# p, q ∈ {0, 1, 2}, d ∈ {0, 1}
# P, Q ∈ {0, 1}, D ∈ {0, 1}
# s = 48 for daily seasonality

SARIMA(1,1,1) × (1,1,1,48)  ← Safe starting point
SARIMA(2,1,1) × (1,1,1,48)  ← More complex if needed
SARIMA(1,1,2) × (0,1,1,48)  ← Alternative
```

### Residual Diagnostics

```python
residuals = y_test - forecast.prediction

# Should pass:
mean(residuals) ≈ 0        ← No systematic bias
std(residuals) should match model estimate
np.corrcoef(residuals[:-1], residuals[1:]) ≈ 0  ← Not autocorrelated
# Histogram should be bell-shaped (Shapiro-Wilk p > 0.05)
```

---

## Quick Sanity Checks

```python
# Check 1: Is forecast reasonable?
assert forecast.prediction.min() >= y_train.min() * 0.5
assert forecast.prediction.max() <= y_train.max() * 1.5

# Check 2: Are bounds sensible?
assert (forecast.lower < forecast.prediction).all()
assert (forecast.upper > forecast.prediction).all()
assert (forecast.upper - forecast.lower > 0).all()

# Check 3: Is coverage in ballpark?
assert 0.7 < coverage < 0.95  # 70-95% reasonable

# Check 4: Any NaNs or Infs?
assert not np.isnan(forecast.prediction).any()
assert not np.isinf(forecast.prediction).any()
assert not np.isnan(forecast.uncertainty_width).any()

# If all pass:
print("✓ Forecast output is valid")
```

---

## Common Metric Combinations and Their Meaning

| MAE | RMSE | MAPE | Coverage | Interpretation |
|---|---|---|---|---|
| 0.3 | 0.4 | 8% | 80% | **Excellent** — Deploy immediately |
| 0.5 | 0.65 | 12% | 80% | **Good** — Monitor and use |
| 0.7 | 1.0 | 15% | 78% | **Acceptable** — Needs tuning soon |
| 1.0 | 1.8 | 20% | 82% | **Poor** — Retrain, increase p,q,P,Q |
| 1.2 | 1.2 | 25% | 45% | **Bad** — Model over-confident, intervals too narrow |
| 0.4 | 0.4 | 9% | 98% | **Bad** — Intervals too wide, model under-confident |
| 0.5 | 2.0 | 12% | 80% | **Bad** — Huge outliers, check data quality |

