# Rolling-Origin Evaluation

## The problem with a single train/test split

`ModelComparison` and `ModelTuner` fit a model once and score one forecast against one
static held-out window. That's fast, and fine for narrowing down model family and
hyperparameters. But one number from one window can't tell you two things:

- **Whether that window was representative.** With 14 days of data, a 3-4 day test
  block is a small, specific slice — a model can look best simply because it fit that
  particular stretch (e.g. a quiet period, or one that happened to avoid an EV charging
  spike).
- **How error grows with lead time.** A forecast scored as one block treats a 1-step-ahead
  prediction and a 96-step-ahead prediction as equally weighted contributions to the same
  RMSE. It has nothing to say about whether the day-ahead (h=48) forecast is as trustworthy
  as the next-30-min (h=1) one.

## What rolling-origin does instead

Walk the training window forward one origin at a time. At each origin, refit on
everything up to that point, forecast `horizon` steps ahead, and record the error at each
step of that horizon separately. Repeating this across many origins gives many
overlapping scored windows instead of one, and grouping the results by horizon step shows
error as a function of lead time rather than a single averaged number.

This mirrors how the model would actually be used in deployment — forecasting forward from
whatever data exists *right now* — rather than training once on a fixed cutoff.

## What it answers that the single split doesn't

- **Is the ranking from `ModelComparison` stable, or an artefact of the split?** Scoring
  the same models across many origins shows whether the winner keeps winning.
- **How far ahead can this forecast be trusted?** Error-by-horizon typically increases
  with lead time; the point where it crosses an acceptable threshold is the honest limit
  of the forecast, not "the length of `y_test`".
- **Does accuracy improve as more history accumulates?** Tracking error at a fixed horizon
  (e.g. 1-step) across origins shows whether the model benefits from a growing training
  window or is roughly flat regardless of how much data it sees.

## Relationship to the existing framework

This is a slower, higher-fidelity **confirmation** step, not a replacement for
`ModelComparison`/`ModelTuner`. Those stay the cheap first pass for choosing a model
family and searching hyperparameters — refitting at every origin for every candidate
would be expensive for little benefit during search. Rolling-origin evaluation runs
afterwards, once, on whichever model already won that cheap pass, to confirm the choice
generalises across windows and to characterise how far ahead it can be trusted. It reuses
the same `TSModel` subclasses and the same `ForecastOutput`/`EvaluationMetrics`
contracts — only the evaluation loop around them differs.

```python
evaluator = RollingOriginEvaluator(SeasonalNaiveModel, y, min_train_size=336, horizon=48)
results = evaluator.run()
by_horizon = evaluator.metrics_by_horizon(results)  # mae, rmse, pi_coverage per horizon step
```

See [Metrics](metrics.md) for what MAE/RMSE/PI coverage mean, and
[Diagnostics](diagnostics.md) for how to read the resulting plots.

## Caveats specific to this dataset

These matter before you point `RollingOriginEvaluator` at real data — they don't show up
until you do the arithmetic for this project's actual size (672 points/asset: 14 days x 48
half-hour intervals).

**Refit cost.** Measured on this data: a single SARIMA `fit()+forecast()` on a 336-point
training window takes **~12 seconds**; ExponentialSmoothing ~0.3s; SeasonalNaive ~0s. A
`step=1` walk with `horizon=48` over one asset produces ~288 origins - **about an hour of
SARIMA refitting for one asset alone**. Use a `step` at least as large as `horizon` for
SARIMA (6 origins over 14 days, ~70s), or restrict rolling-origin confirmation to the
cheaper models.

**Sample size per horizon.** With `min_train_size=336` and `horizon=48`, non-overlapping
origins (`step=48`) give only **6 origins total** - `metrics_by_horizon` is then averaging
6 points at every horizon step. Overlapping origins (`step=1`) give 288 rows, but they are
not 288 independent samples: neighbouring origins share nearly all their training data and
forecast overlapping stretches of the same day, so the resulting MAE/RMSE looks far more
precise than it is. This is the same "barely estimable" problem [Metrics](metrics.md)
already flags for a weekly seasonal-naive baseline, sharper here because a single day's
worth of data is being reused across dozens of "different" origins. Treat a small-`step`
run as a smoother look at the shape of horizon decay, not as a tighter confidence interval
on it.

**Flat width is horizon-dependent, not model-dependent.** [Diagnostics](diagnostics.md#uncertainty-width-over-time)
notes only SARIMA produces horizon-varying interval width. That holds for a rolling-origin
run with `horizon <= season_length` (e.g. `horizon=48` here): `SeasonalNaiveModel`'s margin
scales with `floor(h / season_length) + 1`, which stays at 1 for every step inside one
season, so both SeasonalNaive and ExponentialSmoothing report flat width in that run. Run
`RollingOriginEvaluator` with `horizon > season_length` (e.g. 96, two days) and
SeasonalNaive's width will step up at h=49 - expected, not a bug, but worth knowing before
reading a flat-vs-growing plot as a model property rather than a horizon-length artefact.
