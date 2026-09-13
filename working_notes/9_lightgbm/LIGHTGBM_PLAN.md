# LightGBM Supervised Learning Forecasting — Implementation Plan

## Objective

Build a production-ready LightGBM forecasting notebook (`src/notebooks/lightgbm.py`) that:
1. Demonstrates supervised learning for 30-minute energy metering forecasts
2. Implements uncertainty quantification via quantile regression (P10, P50, P90)
3. Reuses existing forecast output contracts and visualization tools (`ts_model_framework.py`, `ts_plots.py`)
4. Validates feature importance to choose the optimal supervised-learning feature set
5. Achieves performance parity or improvement over SARIMA baseline

**Key Constraint:** Output must match `ForecastOutput` contract (prediction, lower, upper, uncertainty_width) for downstream compatibility.

---

## Phasing

### Phase 1: Feature Engineering Infrastructure (Blocking)
**Goal:** Extend `src/data/generate_metering_features.py` to produce supervised-learning-ready features

#### Task 1.1: Design supervised-learning feature set
- **What:** Define lag features, rolling statistics, and temporal encodings appropriate for 30-min resolution
- **Inputs:** 
  - `metering_data.parquet` (10,080 rows, 30-min intervals)
  - `daily_metrics.parquet` (behavioral context)
- **Outputs (Design doc, not code yet):**
  - List of candidate features (lag-1 to lag-4, rolling mean/std over 6h/24h windows, cyclical hour/day-of-week encoding, daily metrics broadcast)
  - Rationale for each (why it helps forecast 30-min load)
  - Expected dimensionality (e.g., ~25–40 features)
  - Handling of look-ahead leakage (which features are "safe" for real-time forecasting)

#### Task 1.2: Implement feature generation in `generate_metering_features.py`
- **What:** Add functions to build supervised-learning feature matrix
- **Output:** New parquet file: `metering_data_supervised_learning.parquet`
  - Schema: `asset_id`, `timestamp`, `metering_kwh` (target), `feat_lag_1`, `feat_lag_2`, ..., `feat_dow_sin`, `feat_hour_cos`, etc.
  - Same 10,080 rows as base metering data
  - No nulls (or clear handling of boundaries)
- **Notes:** 
  - Keep this separate from existing `metering_data_with_features.parquet` to avoid breaking existing pipelines
  - Document feature ordering and naming convention in docstrings

---

### Phase 2: LightGBM Model Framework (Blocking)
**Goal:** Add LightGBMModel class to `src/ts_model_framework.py` that produces ForecastOutput

#### Task 2.1: Investigate quantile regression for uncertainty quantification
- **What:** Research best practices for LightGBM quantile regression
- **Specific questions to answer:**
  - How does LightGBM's native quantile objective work (num_leaves, learning_rate tuning)?
  - Single model with multiple objectives vs. three separate models (P10, P50, P90)?
  - Calibration: Do quantiles naturally satisfy P10 < P50 < P90, or do we need post-hoc monotonization?
  - Cross-validation strategy (how to tune quantile objectives without overfitting)?
- **Output:** Decision document in `working_notes/` with recommended approach + code sketch

#### Task 2.2: Implement LightGBMModel class
- **What:** Add `LightGBMModel(TSModel)` to `ts_model_framework.py`
- **Interface (following TSModel contract):**
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
- **Output:** `LightGBMModel` added to `ts_model_framework.py`, fully integrated with `ModelEvaluator`
- **Testing:** Verify `.forecast()` returns valid ForecastOutput with P10 ≤ P50 ≤ P90

---

### Phase 3: LightGBM Notebook (Non-blocking, reuses Phases 1–2)
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
  - Feature matrix has no nulls, correct shape (10,080 rows)
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

- [ ] `metering_data_supervised_learning.parquet` is generated, validates schema
- [ ] `LightGBMModel` class exists, integrates with `ModelEvaluator`, produces valid `ForecastOutput`
- [ ] Quantile regression produces P10 ≤ P50 ≤ P90 (monotonicity check)
- [ ] `lightgbm.py` notebook runs end-to-end without errors
- [ ] Notebook demonstrates feature importance, uncertainty calibration, and side-by-side SARIMA comparison
- [ ] All three `TSPlotter` methods work without modification (model-agnostic)
- [ ] PI coverage ≥ target confidence level (e.g., ≥ 80% for 80% CI)

---

## Open Questions (For Task 2.1)

1. **Quantile regression setup:** One 3-objective model or three separate models?
   - Pros/cons of each for convergence, hyperparameter tuning, inference speed?
2. **Monotonicity:** Does LightGBM guarantee P10 ≤ P50 ≤ P90, or do we need post-hoc sorting?
3. **Hyperparameter tuning:** How to cross-validate 3 quantile objectives without massive overhead?
4. **Feature selection:** Start with all ~40 features, or use domain knowledge to select a subset?

