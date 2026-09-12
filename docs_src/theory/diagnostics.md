# Diagnostics

How to read each of the four diagnostic plots produced by `ts_plots.TSPlotter`, and what a bad case tells you about the model. Each failure mode ends with a *Confirm with:* line naming the metric or test that distinguishes it, so the page is usable without a plot in front of you.

This page would be clearer with example figures for each failure mode — worth adding once there are real forecasts from [the framework](../coding/framework_usage.md) to draw from.

## Forecast vs Actual (with prediction interval)

**Good case:** the actual line stays mostly inside the shaded PI band, the forecast tracks it closely (lagging by roughly one time step), and the band narrows during stable periods and widens during transitions. No systematic gap between forecast and actual in either direction.

**Overconfident (PI too narrow):** actuals repeatedly spike outside the band even though the point forecast itself looks reasonable — coverage comes out well under the 80% target (e.g. 45%). Cause: residual std underestimated, or residuals aren't actually normal. Fix: widen manually (`upper *= 1.5, lower /= 1.5`), check the residual histogram for heavy tails, or switch to quantile regression.

**Underfit (forecast flat-lines):** the actual line swings but the forecast barely moves — the model settled on predicting close to the mean every time. Coverage can look fine only because the interval is very wide, not because the model is confident. Cause: over-differencing (d too high) removed real signal, or too little AR/MA capacity. Fix: reduce d (this data has no trend, so `d=0` is the right default), increase p, or switch to LightGBM if the pattern is non-linear. *Confirm with:* `forecast.std()` near zero, and residuals that mirror the actual series' shape.

**Systematically biased:** point forecast tracks reasonably but sits consistently above or below actual — check the residual mean, not just MAE/RMSE, since a biased-but-small error can still look acceptable on those alone. Cause: seasonal pattern shifted (e.g. trained on a different season) or a driving variable (temperature, occupancy) is missing. Fix: retrain on recent data, or add the missing feature. *Confirm with:* mean(residuals) well away from zero relative to their std.

## Residuals diagnostic (time series + histogram)

**Good case:** residuals bounce randomly around 0 with no trend, mostly within ±2x the model's own uncertainty band, and the histogram is roughly bell-shaped and centered at 0.

**Autocorrelated residuals:** consecutive residuals cluster on the same side (several positive in a row, then several negative) instead of bouncing randomly. This means the model missed structure — errors compound instead of self-correcting. Fix: add AR terms (p/P) or seasonal differencing (D); check the ACF/PACF of the residuals for which. *Confirm with:* Ljung-Box on the residuals, or a visible spike in their ACF — see
[Confirming diagnostics numerically](#confirming-diagnostics-numerically) for both.

**Systematic bias:** residuals sit almost entirely on one side of zero and the histogram is visibly skewed. All-positive means the forecast underestimates; all-negative means it overestimates. Fix: check the seasonal component is capturing the daily shape, or retrain on data that better represents current conditions. (Adding non-seasonal differencing is not the fix here — the series has no trend.) *Confirm with:* sign test on the residuals, or simply the fraction above zero.

**Heavy tails:** the histogram has a normal-looking center but disproportionately tall bars at the extremes. The normal-residual assumption behind the PI calculation breaks down, so real-world coverage will run below what the model reports. Fix: remove/inspect outliers in training data, use a more robust model, or predict quantiles directly instead of assuming normality. *Confirm with:* excess kurtosis, or PI coverage below the reported target despite a
centered residual mean — see [Confirming diagnostics
numerically](#confirming-diagnostics-numerically) for the Q-Q plot and
`ResidualDiagnostics.normality_stats`.

## Uncertainty width over time

**Good case:** width tracks actual volatility — narrow during stable stretches (e.g. midday for an office), wider during transitions (morning ramp-up, evening ramp-down). Mean width roughly matches `2 x z x std(residuals)`.

**Flat width:** the interval is the same size at every hour. For **ExponentialSmoothing and
LightGBM this is structural, not a fitting failure** — both return `np.full_like(...)`, a
single residual std broadcast across the whole horizon (`ts_model_framework.py:175, :273`).
Only SARIMA's width is diagnostic here. Where you do want conditional width, the fix is a
residual-std model or explicit quantile models. This is also the dominant real problem in
this codebase: load-forecast residual variance scales with level and time of day, and a
constant width cannot represent that (a variance-stabilising transform for the strictly
positive EV assets, conditional variance modelling, or direct quantile regression are the
remedies). *Confirm with:* plot width against horizon — a horizontal line for ExpSmoothing
or LightGBM is expected; a horizontal line for SARIMA is the finding.

**Widening width (SARIMA):** the interval grows roughly with $\sqrt{\text{horizon}}$. For an
integrated process this is **correct behaviour**, not an artefact — h-step forecast variance
genuinely grows with h, and this is what SARIMA's own `conf_int` produces. Do not cap it:
capping produces miscalibrated intervals by construction, the very failure this page is
trying to prevent. Only treat it as a problem if the width outpaces what the residuals
justify when checked against a held-out window. *Confirm with:* PI coverage at long horizons —
if coverage stays near target as the band widens, the widening is earned.

## PI coverage (green/red scatter)

Each point is colored by whether the actual fell inside (green) or outside (red) the interval.

**Good case:** roughly 80% green for an 80% PI, with reds scattered randomly across time rather than clustered.

**Under-coverage:** far more reds than the target implies (e.g. 15% coverage against an 80% target) — the model is dangerously over-confident and any commitment based on the envelope will breach frequently. Check the residual distribution for normality first; if it's fine, the interval math itself is likely wrong.

**Clustered reds (time-of-day pattern):** coverage looks fine in aggregate but reds concentrate in a specific window (e.g. daytime hours) while off-peak hours are all green. The model hasn't captured a time-of-day effect properly — check the seasonal component (SARIMA: P, D > 0) or the hour-of-day feature (LightGBM), and consider retraining on data from the current season.

**All-green (over-coverage):** ~100% coverage against an 80% target means the interval is far wider than necessary. Point forecast may itself be flat (model "giving up" rather than genuinely uncertain) — check `forecast.std()` isn't near zero before assuming the width is simply conservative-but-correct.

## Confirming diagnostics numerically

The "Confirm with:" lines above name a test but didn't used to point at code. Two more
`TSPlotter` methods and one new class close that gap, all built on the same
`residuals = y_test - forecast.prediction` used by `residuals_diagnostic` above - so, like
that method, **they apply to every `TSModel` subclass** (SARIMA, ExponentialSmoothing,
LightGBM, SeasonalNaive), not just the statsmodels-backed ones. This is deliberately not
the same thing as `SARIMAX.plot_diagnostics()` or `model_fit.summary()`'s Jarque-Bera /
heteroskedasticity rows (see `archive/ML_Tools-TimeSeries/residual_analysis.py`) - those
read the state-space innovations off a fitted `SARIMAXResults` object and have no
equivalent for LightGBM or SeasonalNaive, whose "residuals" only exist as
actual-minus-prediction after the fact.

**`TSPlotter.residuals_qq(y_test, forecast, model_name)`** - Q-Q plot of residuals against
a normal distribution. Points hugging the red line support the normality assumption behind
the framework's z-score prediction intervals; a curved or S-shaped pattern is the visual
form of the "heavy tails" case above.

**`TSPlotter.residuals_acf(y_test, forecast, model_name, nlags=20)`** - correlogram of
residuals with a 95% white-noise band ($\pm 1.96/\sqrt{n}$). Bars outside the band are the
visual form of "autocorrelated residuals" above.

**`ResidualDiagnostics`** (in `ts_model_framework.py`) gives the numeric counterpart to
both plots:

- `ljung_box(residuals, lags=None)` - p < 0.05 at a lag means residuals are not white
  noise there (structure the model missed). Default lag caps at `len(residuals) // 5` since
  the test loses reliability as lags approach the sample size - relevant here because test
  windows are short (a 4-day test split is 192 points).
- `normality_stats(residuals)` - mean, std, skew, excess kurtosis, and a Jarque-Bera
  normality test (p < 0.05 rejects normality). On a test window under ~50 points, treat
  Jarque-Bera as a rough signal rather than a firm verdict.

```python
from ts_model_framework import ResidualDiagnostics

residuals = y_test - forecast.prediction
ResidualDiagnostics.ljung_box(residuals)        # polars DataFrame: lag, lb_stat, lb_pvalue
ResidualDiagnostics.normality_stats(residuals)  # dict: mean, std, skew, excess_kurtosis, jarque_bera_*
```

Neither test replaces the plot - a p-value gives no sense of *where* the autocorrelation
sits (which lag) or *how* the tails are heavy (see [Metrics — proper scoring
rules](metrics.md#proper-scoring-rules-for-intervals) for a related "one number isn't
enough" case). Use them together: the plot for shape, the test for a threshold to act on.

**LightGBM caveat:** a low Ljung-Box p-value here doesn't automatically mean "the model
missed structure" for LightGBM. [Models](models.md#lightgbm-gradient-boosting) notes its
`forecast()` is recursive - past step ~96 every lag feature is itself a prior prediction,
so errors compound with horizon by construction. Autocorrelated residuals from that
compounding are a forecasting-strategy artifact, not necessarily a fixable model fit
issue; check whether the autocorrelation grows with horizon (consistent with compounding)
before reaching for more AR-like structure the way you would for SARIMA.

## Ranking models against each other

When RMSE and coverage disagree — one model has the lowest error but the other is better calibrated — coverage should usually win for anything feeding a flexibility commitment: a narrow-but-wrong interval breaches commitments, while a slightly wider one just costs some upside. A 1-2 point coverage gap from target is easy to correct (manual widening); a systematically miscalibrated model is not. See [Decisions](models-decisions.md) for the full selection logic.
