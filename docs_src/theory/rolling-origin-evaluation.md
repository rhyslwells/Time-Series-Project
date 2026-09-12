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
