# Metrics

This page is reference-first: the range tables and sanity checks come first so they are
quick to look up, then each metric is defined below. Read a metric's value against a
**baseline**, not against an absolute threshold — see the last section.

## Expected ranges by asset type

This project has two asset types: `ev_charging` and `solar_battery`. Values are in the units
the data uses (kWh per 30-minute interval; see [units](../data/index.md)). Ranges are
rough orientation, not acceptance criteria, and they are **horizon-dependent** — a day-ahead
(h=48) forecast has much more error than an intraday (h=1) one, so qualify any number you
quote with the horizon it was measured at.

| Metric | ev_charging | solar_battery |
|---|---|---|
| MAE | < 0.3 good, 0.3-0.6 acceptable, > 1.0 poor | < 0.3 / 0.3-0.6 / > 1.0 |
| RMSE | < 0.5 / 0.5-0.9 / > 1.4 | < 0.5 / 0.5-0.9 / > 1.4 |
| MAPE | usable, < 20% / 20-35% / > 50% | **not usable** — series crosses zero |
| PI Coverage (80% target) | 75-85% good, 70-90% acceptable | 75-85% / 70-90% |
| MASE (vs seasonal naive, s=48) | < 1 = beats the baseline; aim well below 1 | < 1 = beats the baseline |

`solar_battery` values pass through zero, so MAPE's denominator is not merely small, it is
sometimes exactly zero and often changes sign. `max(|y_t|, ε)` does not fix this — it
silently redefines the metric, so cross-asset comparison (the only reason to use MAPE) no
longer holds. Use MASE for these assets, and for cross-asset comparison generally.

## Sanity checks before trusting a forecast

The framework returns `pi_coverage` as a **percentage (0-100)**, not a fraction.

```python
assert (forecast.lower < forecast.prediction).all()
assert (forecast.upper > forecast.prediction).all()
assert 70 < metrics.pi_coverage < 95        # percentage scale
assert not np.isnan(forecast.prediction).any()
```

## MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

Average absolute difference between actual and forecast, in the same units as the data.
Scale-dependent: read it against the asset-type ranges above, or better, against MASE.

## RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

Squaring means large errors dominate the sum, so RMSE is always ≥ MAE and the *gap* between
them is diagnostic:

```python
ratio = rmse / mae
# < 1.2   consistent errors, no outliers
# 1.2-1.5 some outliers, normal
# > 1.5   frequent large errors — check data quality / model fit
```

## MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

Scale-independent, so it is tempting for comparing across differently-sized assets.
**Pitfalls:** undefined at $y_t = 0$, unstable near it, and asymmetric (it penalises
over-forecasting more than under-forecasting). For `solar_battery` assets, whose values
cross zero, it is meaningless. Prefer MASE.

Note: `ModelEvaluator.evaluate` scales sklearn's fraction to a percentage (0-100), so the
stored value matches the `%` it is printed with and thresholds like "MAPE < 15%" read
directly.

## MASE — Mean Absolute Scaled Error

$$\text{MASE} = \frac{\text{MAE of the model}}{\text{MAE of seasonal naive } (\hat{y}_t = y_{t-48})}$$

Scale-free, defined at zero, symmetric, and by construction a skill ratio against the
seasonal-naive baseline. MASE < 1 means the model beats "copy the same time yesterday";
MASE ≥ 1 means it does not. On clean synthetic data with a hard-coded daily profile,
seasonal naive is a strong baseline and beating it is not guaranteed.

## PI Coverage

$$\text{Coverage} = \frac{100}{n} \sum_{t=1}^{n} \mathbb{1}[y_t \in [\hat{L}_t, \hat{U}_t]]$$

The percentage of actuals that fall inside the prediction interval. For an 80% PI, coverage
should land near 80.

- **Undercoverage** (e.g. 45): intervals too narrow, model over-confident — flexibility commitments will breach.
- **Overcoverage** (e.g. 95): intervals too wide, model under-confident — wasted conservatism, missed revenue.

Coverage alone is gameable — you can hit 80% with enormous intervals. Pair it with a proper
scoring rule (interval/Winkler score or pinball loss) that combines width and breach into a
single number.

```python
from scipy.stats import norm

z_80 = norm.ppf(0.90)  # 1.282
upper = forecast + z_80 * std(residuals)
lower = forecast - z_80 * std(residuals)
```

Sanity check: `mean_width ≈ 2 * z * std(residuals)` for a single-std interval. Note that only
SARIMA produces a horizon-varying width in this framework; ExponentialSmoothing and LightGBM
return a constant width (see [Diagnostics](diagnostics.md)).

## Proper scoring rules for intervals

Coverage says whether the actual landed inside the band but nothing about how wide the band
had to be. These two rules fold width and breach into one number, so a model cannot win by
inflating its intervals.

**Interval (Winkler) score** for a central $(1-\alpha)$ interval $[\hat{L}_t, \hat{U}_t]$:

$$W_t = (\hat{U}_t - \hat{L}_t) + \frac{2}{\alpha}(\hat{L}_t - y_t)\,\mathbb{1}[y_t < \hat{L}_t] + \frac{2}{\alpha}(y_t - \hat{U}_t)\,\mathbb{1}[y_t > \hat{U}_t]$$

Pay the width always, plus a penalty scaled by $2/\alpha$ for each breach. Lower is better;
average over the horizon. For an 80% interval, $\alpha = 0.2$ so each breach costs $10\times$
its distance outside the band.

**Pinball loss** at quantile $q$ (the same object, per quantile rather than per interval):

$$\rho_q(y_t, \hat{y}_t^{(q)}) = \max\big(q\,(y_t - \hat{y}_t^{(q)}),\ (q-1)(y_t - \hat{y}_t^{(q)})\big)$$

Average over timesteps and over the quantiles the model reports. This is the loss a
quantile-regression model already minimises, and it is the right target when comparing
`solar_battery` interval quality where coverage is noisy and MAPE is unusable.

Neither is implemented in `ModelEvaluator` yet — compute them from `forecast.lower`,
`forecast.upper`, and the actuals.

## Read metrics against a baseline

An MAE of 0.4 means nothing on its own. The reference point is **seasonal naive**
($\hat{y}_t = y_{t-48}$ for a 30-minute series with a daily cycle). Compute the baseline's
MAE/RMSE on the same test window first, then report the model as a ratio (that ratio is
MASE). `SeasonalNaiveModel` in the framework computes this baseline directly — add it to a
`ModelComparison` and read the other models' MAE against its row.

With 14 days of data there are only two weekly cycles, so a weekly seasonal-naive baseline
($\hat{y}_t = y_{t-336}$) is barely estimable — daily seasonal naive is the honest reference here.
