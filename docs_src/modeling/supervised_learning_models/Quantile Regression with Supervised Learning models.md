# Quantile regression with LightGBM

How `LightGBMModel` (`src/ts_models.py`) quantifies forecast uncertainty: three separate
`LGBMRegressor` models, one per quantile level (default P10/P50/P90), instead of the
residual-std margin the other `TSModel` subclasses use.

## Quantile regression vs. mean + residual variance

`SARIMAModel` and `ExponentialSmoothingModel` predict a single conditional mean and derive
`lower`/`upper` from a constant residual-std margin (`yhat +/- z * residual_std`) — this
assumes the residual distribution is symmetric and doesn't widen or narrow with the
covariates. Quantile regression instead fits the model directly to the conditional quantile
of the target ($P(Y \le \hat{y}_\tau \mid X) = \tau$ for quantile level $\tau$), so the
interval width can vary with the input features rather than being fixed — e.g. wider
intervals at hours with historically higher variance, narrower ones where the asset behaves
predictably. LightGBM supports this natively via `objective="quantile"` with a per-model
`alpha` parameter set to the target quantile.

## One multi-objective model vs. three separate models

`LightGBMModel` fits **three separate models** (`self.models: Dict[float, LGBMRegressor]`,
keyed by quantile level), not one model with multiple outputs:

- **Pros:** each model is a standard `LGBMRegressor` — no custom multi-output wrapper,
  works with the existing `.fit()`/`.predict()` calling convention, and each quantile can
  in principle be tuned independently (different `num_leaves`/`learning_rate` per quantile)
  if the data warrants it.
- **Cons:** 3x training cost, and — the main gotcha below — no built-in guarantee that the
  three models agree with each other row-by-row.
- **Why not one multi-objective model:** LightGBM's `LGBMRegressor` takes one `objective`
  per model instance; a genuinely joint multi-quantile model needs a custom objective/loss
  (e.g. pinball loss summed across quantiles) which adds implementation complexity for a
  15-asset, 3-quantile use case where training cost is negligible either way.

## Calibration: does LightGBM guarantee P10 <= P50 <= P90?

No. Each quantile model is trained independently, so nothing stops the P10 model from
predicting *above* the P50 model's output for a given row, especially with limited training
data or extreme feature values. `LightGBMModel._stack_quantiles` handles this by sorting the
three predictions row-wise (`np.sort(q_arr, axis=1)`) after prediction, then assigning the
lowest sorted value to `lower`, the value at the level closest to 0.5 to `prediction`, and
the highest to `upper`. This is a post-hoc monotonization, not a training-time constraint —
it guarantees a valid, non-crossing interval in the output, but does not fix miscalibration
(e.g. P10 that is *systematically* too high).

## Hyperparameters and validation

- **Gotchas:** `objective="quantile"` needs its `alpha` set per model instance (0.1, 0.5, 0.9
  by default here); forgetting to vary `alpha` across the three models silently trains three
  identical mean-ish regressors. `num_leaves`/`learning_rate` are shared across all three
  quantile models in the current implementation — reasonable at this dataset size, but a
  place to revisit if the P10/P90 models underfit while P50 overfits (or vice versa).
- **Validation:** point-forecast metrics (MAE/RMSE on `prediction`, i.e. the P50 model) are
  necessary but not sufficient — they say nothing about whether the intervals are well
  calibrated. Use `pi_coverage` from `EvaluationMetrics`/`ModelEvaluator` (already wired up
  for `LightGBMModel` like every other `TSModel`) to check the fraction of actuals falling
  inside `[lower, upper]` against the nominal level implied by the chosen quantiles (P10/P90
  should cover ~80%). See [Rolling-origin evaluation](../../theory/rolling-origin-evaluation.md)
  for validating this across multiple forecast origins rather than a single train/test split.