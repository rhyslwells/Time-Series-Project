# Diagnostics

How to read each of the four diagnostic plots produced by `ts_plots.TSPlotter`, and what a bad case tells you about the model.

## Forecast vs Actual (with prediction interval)

**Good case:** the actual line stays mostly inside the shaded PI band, the forecast tracks it closely (lagging by roughly one time step), and the band narrows during stable periods and widens during transitions. No systematic gap between forecast and actual in either direction.

**Overconfident (PI too narrow):** actuals repeatedly spike outside the band even though the point forecast itself looks reasonable — coverage comes out well under the 80% target (e.g. 45%). Cause: residual std underestimated, or residuals aren't actually normal. Fix: widen manually (`upper *= 1.5, lower /= 1.5`), check the residual histogram for heavy tails, or switch to quantile regression.

**Underfit (forecast flat-lines):** the actual line swings but the forecast barely moves — the model settled on predicting close to the mean every time. Coverage can look fine only because the interval is very wide, not because the model is confident. Cause: over-differencing (d too high) removed real signal, or too little AR/MA capacity. Fix: reduce d, increase p, or switch to LightGBM if the pattern is non-linear.

**Systematically biased:** point forecast tracks reasonably but sits consistently above or below actual — check the residual mean, not just MAE/RMSE, since a biased-but-small error can still look acceptable on those alone. Cause: seasonal pattern shifted (e.g. trained on a different season) or a driving variable (temperature, occupancy) is missing. Fix: retrain on recent data, or add the missing feature.

## Residuals diagnostic (time series + histogram)

**Good case:** residuals bounce randomly around 0 with no trend, mostly within ±2x the model's own uncertainty band, and the histogram is roughly bell-shaped and centered at 0.

**Autocorrelated residuals:** consecutive residuals cluster on the same side (several positive in a row, then several negative) instead of bouncing randomly. This means the model missed structure — errors compound instead of self-correcting. Fix: increase differencing (d/D) or AR terms (p/P).

**Systematic bias:** residuals sit almost entirely on one side of zero and the histogram is visibly skewed. All-positive means the forecast underestimates; all-negative means it overestimates. Fix: check for a missing trend term (ensure d ≥ 1), or retrain on data that better represents current conditions.

**Heavy tails:** the histogram has a normal-looking center but disproportionately tall bars at the extremes. The normal-residual assumption behind the PI calculation breaks down, so real-world coverage will run below what the model reports. Fix: remove/inspect outliers in training data, use a more robust model, or predict quantiles directly instead of assuming normality.

## Uncertainty width over time

**Good case:** width tracks actual volatility — narrow during stable stretches (e.g. midday for an office), wider during transitions (morning ramp-up, evening ramp-down). Mean width roughly matches `2 x z x std(residuals)`.

**Flat width:** the interval is the same size at every hour, meaning the model isn't adapting its confidence to conditions — over-wide during predictable periods (wasted conservatism) and under-wide during volatile ones (under-confident when it matters most). Usually means the interval was computed from a single global residual std rather than a conditional one; for LightGBM in particular, consider a residual-std model or explicit quantile models.

**Exploding width:** the interval balloons far faster than the forecast horizon justifies (e.g. tripling over a few hours). Typically comes from naive error propagation (std growing with $\sqrt{\text{horizon}}$) rather than recalibration against real data. Fix: cap the width, or use a recursive forecasting approach that treats each step's own forecast as the new anchor.

## PI coverage (green/red scatter)

Each point is colored by whether the actual fell inside (green) or outside (red) the interval.

**Good case:** roughly 80% green for an 80% PI, with reds scattered randomly across time rather than clustered.

**Under-coverage:** far more reds than the target implies (e.g. 15% coverage against an 80% target) — the model is dangerously over-confident and any commitment based on the envelope will breach frequently. Check the residual distribution for normality first; if it's fine, the interval math itself is likely wrong.

**Clustered reds (time-of-day pattern):** coverage looks fine in aggregate but reds concentrate in a specific window (e.g. daytime hours) while off-peak hours are all green. The model hasn't captured a time-of-day effect properly — check the seasonal component (SARIMA: P, D > 0) or the hour-of-day feature (LightGBM), and consider retraining on data from the current season.

**All-green (over-coverage):** ~100% coverage against an 80% target means the interval is far wider than necessary. Point forecast may itself be flat (model "giving up" rather than genuinely uncertain) — check `forecast.std()` isn't near zero before assuming the width is simply conservative-but-correct.

## Ranking models against each other

When RMSE and coverage disagree — one model has the lowest error but the other is better calibrated — coverage should usually win for anything feeding a flexibility commitment: a narrow-but-wrong interval breaches commitments, while a slightly wider one just costs some upside. A 1-2 point coverage gap from target is easy to correct (manual widening); a systematically miscalibrated model is not. See [Decisions](model-decisions.md) for the full selection logic.
