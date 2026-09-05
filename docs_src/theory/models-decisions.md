# Model Decisions

## Which model is best?

1. Rank candidates by RMSE (lower is better).
2. Check the leader's PI coverage — is it within 75-85% of the 80% target? If not, note whether it's over- or under-covering.
3. Check RMSE/MAE ratio — under 1.5 means no outlier problem.
4. Check MAPE against the [expected range](metrics.md#expected-ranges-by-asset-type) for the asset type.
5. If 3+ checks pass, deploy. If 2+ fail, tune (see below) before comparing again.

**When RMSE and coverage disagree** (e.g. LightGBM has the lowest RMSE but only 68% coverage while SARIMA has slightly higher RMSE but 82% coverage): for flexibility forecasting, coverage should usually decide it. A miscalibrated interval breaches commitments regardless of how good the point forecast is; a small coverage gap can be corrected with a manual widening factor. An ensemble — LightGBM's point forecast paired with SARIMA's interval — is a reasonable middle ground when both matter.

## When to retrain

| Trigger | Action |
|---|---|
| Coverage drops below 70% | Retrain immediately — model is dangerously over-confident |
| RMSE increases 20%+ | Retrain — likely concept drift |
| Forecast flat-lines | Retrain and re-check hyperparameters — model gave up |
| New season | Retrain on the most recent 14 days |
| Equipment changes (new HVAC, solar installed) | Retrain and inspect residuals for a step change |

## When to switch model class

| Symptom | Try | Why |
|---|---|---|
| MAPE > 25% and RMSE tuning has plateaued | LightGBM + exogenous features | Need non-linear fit |
| Coverage erratic with no time-of-day pattern | Quantile regression | Residuals aren't normal |
| Concept drift (patterns visibly shifting) | Refit on the last 7-14 days only | Faster adaptation than full retrain |
| Residuals autocorrelated after tuning d/D | Increase differencing further, or add lag features | Differencing alone insufficient |
| Large spikes consistently missed | LightGBM or another non-linear model | Linear models can't capture sharp non-linearities |

## Tuning priority

| Problem | Priority | Fix |
|---|---|---|
| Coverage < 75% | High | Inflate intervals, or retrain |
| MAE above the asset-type threshold | High | Increase model complexity (p, num_leaves) |
| RMSE/MAE > 1.5 | Medium | Identify and address outliers |
| Uncertainty width flat over time | Medium | Move to adaptive/quantile intervals |
| Coverage 85-95% | Low | Acceptable — monitor only |

Model-specific tuning directions are in [Models](models.md#choosing-between-them); code for grid search is in [Coding](../coding/framework_usage.md#tuning).

## Production-ready checklist

- [ ] RMSE and MAE within the [expected range](metrics.md#expected-ranges-by-asset-type) for the asset type
- [ ] RMSE/MAE ratio < 1.5
- [ ] MAPE < 15% (or the asset-specific threshold)
- [ ] PI coverage 75-85%
- [ ] Residuals: bell-shaped, centered at 0, no trend, not autocorrelated
- [ ] Uncertainty width varies with time-of-day rather than flat
- [ ] Coverage roughly uniform across hours (no clustered under/over-coverage)

All checked -> deploy. Two or more unchecked -> tune or reconsider the model class before deploying.
