# Metrics

## MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

Average absolute difference between actual and forecast, in the same units as the data (kWh). Scale-dependent: 0.1 kWh is excellent for a 0.1 kWh asset and poor for a 5 kWh asset — always read it against the asset-type ranges below.

## RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

Squaring means large errors dominate the sum, so RMSE is always ≥ MAE and the *gap* between them is diagnostic:

```python
ratio = rmse / mae
# < 1.2   consistent errors, no outliers
# 1.2-1.5 some outliers, normal
# > 1.5   frequent large errors — check data quality / model fit
```

## MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

Scale-independent, so it's the right metric for comparing forecast quality across differently-sized assets. **Pitfall:** blows up when $y_t \approx 0$ (common in energy off-peak periods) — use $\max(|y_t|, \epsilon)$ in the denominator, or exclude near-zero values.

## PI Coverage

$$\text{Coverage} = \frac{1}{n} \sum_{t=1}^{n} \mathbb{1}[y_t \in [\hat{L}_t, \hat{U}_t]]$$

The share of actuals that fall inside the prediction interval. For an 80% PI, coverage should land near 80%.

- **Undercoverage** (e.g. 45%): intervals too narrow, model over-confident — flexibility commitments will breach.
- **Overcoverage** (e.g. 95%): intervals too wide, model under-confident — wasted conservatism, missed revenue.

```python
from scipy.stats import norm
z_80 = norm.ppf(0.90)   # 1.282
upper = forecast + z_80 * std(residuals)
lower = forecast - z_80 * std(residuals)
```

Sanity check: `mean_width ≈ 2 * z * std(residuals)`; a gap larger than ~0.2 kWh means the interval calculation doesn't match the residual distribution.

## Expected ranges by asset type

| Metric | Residential | Commercial | EV Charging |
|---|---|---|---|
| MAE | < 0.5 kWh good, 0.5-1.0 acceptable, > 1.5 poor | < 1.0 / 1.0-1.5 / > 2.0 kWh | < 2.0 / 2.0-4.0 / > 5.0 kWh |
| RMSE | < 0.7 / 0.7-1.0 / > 1.5 kWh | < 1.3 / 1.3-1.8 / > 2.5 kWh | < 2.5 / 2.5-5.0 / > 7.0 kWh |
| MAPE | < 10% / 10-15% / > 20% | < 15% / 15-20% / > 25% | < 30% / 30-40% / > 50% |
| PI Coverage (80% target) | 78-82% good, 75-85% acceptable | 76-84% | 70-80% |
| Uncertainty width | 0.8-1.5 kWh | 1.5-2.5 kWh | 3.0-6.0 kWh |

Why the spread: residential demand follows a stable daily routine, commercial adds occupancy and HVAC non-linearity, and EV charging arrival/duration is close to a Poisson process — each step up trades predictability for MAE/MAPE headroom.

## Sanity checks before trusting a forecast

```python
assert (forecast.lower < forecast.prediction).all()
assert (forecast.upper > forecast.prediction).all()
assert 0.7 < coverage < 0.95
assert not np.isnan(forecast.prediction).any()
```
