# Feature validation

How to catch feature leakage in a supervised-learning time series model — the checks that
apply on top of [feature engineering](feature_engineering.md) once lag/rolling features exist.

## Chronological splitting

Validation and test sets must cover strictly later periods than the training set. Splitting
randomly (e.g. a shuffled k-fold) lets future timestamps leak into training and breaks the
autocorrelation the model relies on. See [Rolling-origin
evaluation](../theory/rolling-origin-evaluation.md) for how this project walks the split
forward instead of using one fixed cutoff.

## Positive lag shifting

Generate lag features by shifting historical values backward (`shift()`), so a predictor at
time $t$ contains only past observations ($t-1, t-2, \dots$) — never the current or a future
value. This is the same constraint [Feature engineering's
caveats](feature_engineering.md#caveats) describe for `feat_*` columns: joined same-day
metrics must be lagged by at least one day before use as a model input, or the model
conditions on information it could not have had at forecast time.

## Walk-forward validation

Evaluate by stepping the training window forward in time — refit (or reforecast) at each
origin using only data available up to that point, rather than fitting once on a fixed split.
See [Rolling-origin evaluation](../theory/rolling-origin-evaluation.md) for the implementation
(`RollingOriginEvaluator`) and this dataset's refit-cost and sample-size caveats at 14 days of
history.
