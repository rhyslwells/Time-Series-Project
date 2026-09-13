# LightGBM Supervised Learning Forecasting — Implementation Plan

**Status: all three phases done and verified (2026-09-13).** See "Current state" for what
exists right now, and the per-task checkboxes further down for detail. One naming deviation
from the original sketch: the notebook is `src/notebooks/lightgbm_forecasting.py`, not
`lightgbm.py` (see Phase 3 below for why).

## Current state (2026-09-13)

**Phase 1 — Feature Engineering Infrastructure: done.**
- `working_notes/9_lightgbm/feature_design.md` records the Task 1.1 feature-set design (26
  features: 6 lags, 4 rolling stats, 4 cyclical encodings, 12 lagged daily-context columns).
- `src/data/generate_metering_features.py` now also builds
  `metering_data_supervised_learning.parquet` (`build_supervised_learning_features` and
  helpers) — 8,640 rows (15 assets x 576 rows after dropping the first 96 rows/asset for
  lookback), 26 `feat_*` columns, zero nulls. Kept separate from the pre-existing
  `metering_data_with_features.parquet`, which other notebooks still read unchanged.
- `docs_src/modeling/supervised_learning_models/feature_engineering.md` documents the table
  (schema, look-ahead handling, validation checks), registered in `mkdocs.yml` nav.

**Phase 2 — LightGBM Model Framework: done.**
- `LightGBMModel` in `src/ts_models.py` was extended, not replaced, to stay backward
  compatible with existing callers:
  - **Internal-features mode** (`X_train=None`, the default): unchanged from before —
    lag + hour-of-day features built from `y_train` alone, `forecast(steps,
    confidence_level)` works standalone. `ts_model_explorer.py` and
    `lightgbm_residual_diagnostics.py` call this mode via `LightGBMModel(y_train,
    lags=[1,2,48,96])` and were verified to still work unmodified.
  - **External-features mode** (`X_train` supplied, e.g. the `feat_*` columns from
    `metering_data_supervised_learning.parquet`): `y_train` must be the aligned target
    column; `forecast()` requires a matching `X_test` of pre-engineered rows.
  - Both modes now fit **three quantile models** (`objective="quantile"`, default
    `quantile_levels=[0.1, 0.5, 0.9]`) instead of one point model + residual-std margin.
    Predictions are sorted row-wise post-hoc (`_stack_quantiles`) to guarantee
    `lower <= prediction <= upper` — LightGBM's per-quantile models don't guarantee this
    on their own.
  - Verified: both modes produce valid, monotonic `ForecastOutput`; existing notebook call
    pattern still fits/forecasts correctly; full `ModelComparison` (`fit_all` +
    `evaluate_all`, alongside `SARIMAModel`) runs end-to-end.
- `docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning
  models.md` — was a question-list stub, now answers Task 2.1's open questions (one model per
  quantile vs. multi-objective, no built-in monotonicity guarantee and how it's handled,
  hyperparameter/validation gotchas), registered in `mkdocs.yml` nav.

**Phase 3 — Notebook: done, with one deviation.** Built as
`src/notebooks/lightgbm_forecasting.py`, **not** `lightgbm.py` as originally named — running
`python lightgbm.py` (or marimo loading it) from `src/notebooks/` put that directory ahead of
site-packages on `sys.path`, so `ts_models.py`'s `import lightgbm as lgb` resolved to the
notebook file itself instead of the real package (`AttributeError: module 'lightgbm' has no
attribute 'LGBMRegressor'`). This is a Python import-shadowing bug, not a design choice, so
the notebook was renamed rather than patching `ts_models.py` to route around a filename
collision in shared code. Follows the `sarima.py` template: load `metering_data_supervised_learning.parquet`
→ feature overview → 8-day/4-day train/test split → fit `LightGBMModel` in external-features
mode → feature importance → forecast → `ForecastOutput` contract → `ModelEvaluator` →
uncertainty analysis → quantile-interpolated P(forecast > threshold) → `TSPlotter` → SARIMA
comparison via `ComparisonPlotter`. Verified end-to-end (`python lightgbm_forecasting.py`,
all 12 sections run, P10 <= P50 <= P90 holds, comparison to SARIMA completes). Results on
ASSET_001: LightGBM MAE 0.26 / RMSE 0.36 vs. SARIMA MAE 0.57 / RMSE 0.74, but LightGBM's PI
coverage came in at 60.4% against an 80% target (SARIMA: 71.4%) — flagged in the notebook,
not tuned away, since the plan's success criteria call for reporting this honestly rather
than forcing a number. The notebook's Summary section is explicit that this comparison is not
fully apples-to-apples: LightGBM's test-set lag features are true observed values (leakage-safe
for training, but optimistic for evaluation), not recursively forecast the way SARIMA's are.

**Interface decision worth flagging:** the original Task 2.2 sketch below (`forecast(self,
X_test, steps, confidence_level)`, `X_train` required in the constructor) would have broken
`ModelComparison`/`RollingOriginEvaluator`/`ModelTuner` (`ts_evaluation.py`), which all call
`model.fit()` and `model.forecast(steps, confidence_level)` with no `X_test`, plus the two
existing notebooks' `LightGBMModel(y_train, lags=...)` calls. Resolved by making `X_train`
optional (see Phase 2 above) rather than following the sketch literally.

---

## Original Objective (phases below now largely implemented — see "Current state" above for
what differs from the original sketch, mainly the optional-`X_train` interface decision)

Build a production-ready LightGBM forecasting notebook (`src/notebooks/lightgbm.py`) that:
1. Demonstrates supervised learning for 30-minute energy metering forecasts
2. Implements uncertainty quantification via quantile regression (P10, P50, P90)
3. Reuses existing forecast output contracts and visualization tools (`src/ts_models.py`, `src/ts_evaluation.py`, `src/ts_plots.py`)
4. Validates feature importance to choose the optimal supervised-learning feature set
5. Achieves performance parity or improvement over SARIMA baseline

**Key Constraint:** Output must match `ForecastOutput` contract (prediction, lower, upper, uncertainty_width) for downstream compatibility.

---

## Phasing

### Phase 1: Feature Engineering Infrastructure (Blocking) — DONE

**Goal:** Extend `src/data/generate_metering_features.py` to produce supervised-learning-ready features

#### Task 1.1: Design supervised-learning feature set — DONE
- **What:** Define lag features, rolling statistics, and temporal encodings appropriate for 30-min resolution
- **Inputs:** 
  - `metering_data.parquet` (10,080 rows, 30-min intervals)
  - `daily_metrics.parquet` (behavioral context)
- **Outputs (Design doc, not code yet):**
  - List of candidate features (lag-1 to lag-4, rolling mean/std over 6h/24h windows, cyclical hour/day-of-week encoding, daily metrics broadcast)
  - Rationale for each (why it helps forecast 30-min load)
  - Expected dimensionality (e.g., ~25–40 features)
  - Handling of look-ahead leakage (which features are "safe" for real-time forecasting)
- **Actually delivered:** `working_notes/9_lightgbm/feature_design.md` — 26 features (6 lags,
  4 rolling mean/std, 4 cyclical sin/cos, 12 lagged daily-context columns), matches the
  ~25-40 estimate.

#### Task 1.2: Implement feature generation in `generate_metering_features.py` — DONE
- **What:** Add functions to build supervised-learning feature matrix
- **Output:** New parquet file: `metering_data_supervised_learning.parquet`
  - Schema: `asset_id`, `timestamp`, `metering_kwh` (target), `feat_lag_1`, `feat_lag_2`, ..., `feat_dow_sin`, `feat_hour_cos`, etc.
  - ~~Same 10,080 rows as base metering data~~ — **actually 8,640 rows**: the first 96
    rows/asset are dropped (no full lag/rolling/daily-context history yet), so this table is
    intentionally smaller than `metering_data_with_features.parquet`. See `feature_design.md`
    for the row-count derivation.
  - No nulls (verified: 0 nulls across all 26 `feat_*` columns)
- **Notes:** 
  - Kept separate from existing `metering_data_with_features.parquet` — that file and its
    consumers (`ts_model_explorer.py` etc.) are untouched.
  - Feature ordering/naming documented in `feature_engineering.md` (docs) and inline in
    `generate_metering_features.py`.

#### Task 1.3 Update documentation — DONE

`docs_src/modeling/supervised_learning_models/feature_engineering.md`, following the
structure of `docs_src/data/feature_engineering.md` (join logic, column naming, look-ahead
handling, output schema, validation). Registered in `mkdocs.yml` nav under Modeling.

---

### Phase 2: LightGBM Model Framework (Blocking) — DONE

**Goal:** Add LightGBMModel class to `src/ts_models.py` that produces ForecastOutput

#### Task 2.1: Investigate quantile regression for uncertainty quantification — DONE
- **What:** Research best practices for LightGBM quantile regression
- **Specific questions to answer:**
  - How does LightGBM's native quantile objective work (num_leaves, learning_rate tuning)?
  - Single model with multiple objectives vs. three separate models (P10, P50, P90)?
  - Calibration: Do quantiles naturally satisfy P10 < P50 < P90, or do we need post-hoc monotonization?
  - Cross-validation strategy (how to tune quantile objectives without overfitting)?
- **Output:** Decision document in `working_notes/` with recommended approach + code sketch
- **Actually delivered:** answers written into `docs_src/modeling/supervised_learning_models/
  Quantile Regression with Supervised Learning models.md` (chosen: three separate models per
  quantile, not one multi-objective model; no native monotonicity guarantee, handled by
  post-hoc row-wise sorting; shared `num_leaves`/`learning_rate` across the three quantile
  models for now) rather than a standalone `working_notes/` file, since the doc already
  existed as a question stub in the target location.

#### Task 2.2: Implement LightGBMModel class — DONE, with one interface deviation
- **What:** Add `LightGBMModel(TSModel)` to `ts_models.py`
- **Interface as originally sketched (following TSModel contract):**
  ```python
  class LightGBMModel(TSModel):
      def __init__(self, X_train, y_train, quantile_levels=[0.1, 0.5, 0.9]):
          # X_train: (n_samples, n_features) supervised-learning feature matrix
          # y_train: (n_samples,) target metering values
          # quantile_levels: which quantiles to predict
      
      def fit(self):
          # Fit 3 models (or 1 multi-objective model) for P10, P50, P90
      
      def forecast(self, X_test, steps, confidence_level=0.80):
          # Return ForecastOutput(prediction, lower, upper, uncertainty_width)
  ```
- **What actually shipped:** `X_train` is optional (`__init__(self, y_train, X_train=None,
  lags=None, num_leaves=31, learning_rate=0.05, quantile_levels=None)`), and `forecast`
  gained an optional `X_test=None` param rather than requiring it —
  `forecast(self, steps, confidence_level=0.80, X_test=None)`. Reason: the sketch above,
  taken literally, breaks `ModelComparison`/`RollingOriginEvaluator`/`ModelTuner`
  (`ts_evaluation.py`), which call `model.fit()` and `model.forecast(steps,
  confidence_level)` generically with no `X_test`, and breaks the two existing notebooks'
  `LightGBMModel(y_train, lags=[1,2,48,96])` calls. This was flagged and confirmed with the
  user before implementing (see "Interface decision" in Current state above). Fitting still
  produces 3 quantile models (`objective="quantile"`, one `LGBMRegressor` per level) with
  post-hoc row-wise sorting for monotonicity, in both the internal- and external-features
  modes.
- **Output:** `LightGBMModel` added to `ts_models.py`, fully integrated with `ModelEvaluator` (`ts_evaluation.py`)
- **Testing:** Verify `.forecast()` returns valid ForecastOutput with P10 ≤ P50 ≤ P90
- **Verified:** both feature modes produce monotonic `ForecastOutput`s; existing notebook
  call pattern (`LightGBMModel(y_train, lags=[1,2,48,96])`) still fits/forecasts correctly;
  full `ModelComparison.fit_all()` + `.evaluate_all()` runs end-to-end alongside `SARIMAModel`.

---

### Phase 3: LightGBM Notebook (Non-blocking, reuses Phases 1–2) — DONE (as `lightgbm_forecasting.py`)

**Goal:** Pedagogical marimo notebook demonstrating supervised learning + uncertainty quantification

#### Task 3.1: Build `src/notebooks/lightgbm.py` (marimo notebook)
- **Structure (following sarima.py template):**
  1. **Imports & Setup**
  2. **Summary** (markdown: what is supervised learning, why quantile regression)
  3. **Load & Explore Data** — Load `metering_data_supervised_learning.parquet`
  4. **Feature Overview** — Show candidate features, their ranges, correlations
  5. **Feature Importance (Pre-fit)** — Train a simple model, extract feature importance, visualize top 10
  6. **Train/Test Split** — 10 days train, 4 days test (same as SARIMA for comparability)
  7. **Fit LightGBM** — Train 3 quantile models (P10, P50, P90) on selected features
  8. **Generate Forecasts** — Produce ForecastOutput with prediction intervals
  9. **Standard Output Contract** — Show forecast_df schema (asset_id, timestamp, prediction, lower, upper, uncertainty_width, actual)
  10. **Model Evaluation** — Reuse `ModelEvaluator` (MAE, RMSE, MAPE, PI coverage)
  11. **Uncertainty Analysis** — Interval width distribution, calibration (% of actuals in interval)
  12. **Forecast Probability of Events** — P(forecast > threshold) using fitted quantiles
  13. **Visualizations** — Reuse `TSPlotter` methods (forecast_vs_actual, residuals_diagnostic, pi_coverage)
  14. **Comparison to SARIMA** (optional extension cell) — side-by-side metrics

#### Task 3.2: Validation & Testing
- **What:** Run notebook end-to-end, verify:
  - Feature matrix has no nulls, correct shape (8,640 rows total / 576 rows for the single
    asset the notebook filters to — see Phase 1 row-count note above)
  - LightGBM trains without errors
  - ForecastOutput is valid (P10 ≤ P50 ≤ P90, all non-null)
  - `TSPlotter` methods work with LightGBM output (no model-specific assumptions)
  - Metrics match expected format (MAE, RMSE, MAPE as floats; pi_coverage as %)

---

## Task Ordering & Dependencies

```
Phase 1 (Features)
├─ 1.1: Design feature set
└─ 1.2: Implement in generate_metering_features.py ◄── BLOCKING for 2.x and 3.x

Phase 2 (Model)
├─ 2.1: Investigate quantile regression
└─ 2.2: Implement LightGBMModel ◄── BLOCKING for 3.1

Phase 3 (Notebook)
├─ 3.1: Build notebook
└─ 3.2: Validate
```

**Critical path:** 1.1 → 1.2 → 2.1 → 2.2 → 3.1 → 3.2

---

## Success Criteria

- [x] `metering_data_supervised_learning.parquet` is generated, validates schema
- [x] `LightGBMModel` class exists, integrates with `ModelEvaluator`, produces valid `ForecastOutput`
- [x] Quantile regression produces P10 ≤ P50 ≤ P90 (monotonicity check)
- [x] `lightgbm_forecasting.py` notebook runs end-to-end without errors (renamed from
  `lightgbm.py` — see Phase 3)
- [x] Notebook demonstrates feature importance, uncertainty calibration, and side-by-side SARIMA comparison
- [x] All three `TSPlotter` methods work without modification (model-agnostic)
- [ ] PI coverage ≥ target confidence level (e.g., ≥ 80% for 80% CI) — **not met**: 60.4%
  against an 80% target on ASSET_001 (SARIMA: 71.4%). Reported honestly in the notebook
  rather than tuned to hit the number; a real follow-up (see Next Steps in the notebook and
  Task 2.1's shared-hyperparameters note).

---

## Open Questions (For Task 2.1) — ANSWERED

Answers written up in full in `docs_src/modeling/supervised_learning_models/Quantile
Regression with Supervised Learning models.md`; summarized here:

1. **Quantile regression setup:** Three separate `LGBMRegressor` models (one per quantile
   level), not one multi-objective model — simpler to implement against the existing
   `.fit()`/`.predict()` convention, at the cost of 3x training time (negligible at this
   dataset size) and no cross-quantile consistency guarantee (see #2).
2. **Monotonicity:** Not guaranteed by LightGBM — each quantile model is trained
   independently. Handled with post-hoc row-wise sorting (`_stack_quantiles` in
   `ts_models.py`), which guarantees valid non-crossing intervals but does not fix
   systematic miscalibration.
3. **Hyperparameter tuning:** Not addressed yet — `num_leaves`/`learning_rate` are shared
   across all three quantile models for now (see doc's "shared across all three" note as a
   place to revisit if P10/P90 under/overfit relative to P50). Left as a follow-up, not
   blocking for Phase 3.
4. **Feature selection:** Not resolved — Phase 1 shipped the full 26-feature set with no
   selection step. Deferred to whatever `LightGBMModel.get_params()`-driven feature
   importance work the Phase 3 notebook surfaces (Task 3.1's "Feature Importance" section).

