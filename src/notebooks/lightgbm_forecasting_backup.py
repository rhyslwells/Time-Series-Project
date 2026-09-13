""" """

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.insert(0, "../")

    import marimo as mo
    import polars as pl
    import numpy as np
    import optuna
    import warnings

    from ts_models import LightGBMModel, SARIMAModel
    from ts_evaluation import ModelEvaluator
    from ts_plots import TSPlotter, ComparisonPlotter

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        ComparisonPlotter,
        LightGBMModel,
        ModelEvaluator,
        SARIMAModel,
        TSPlotter,
        mo,
        np,
        optuna,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    Supervised-learning forecasting pipeline for 30-minute energy metering, using
    `LightGBMModel`'s **external-features mode**: a pre-engineered feature matrix
    (`metering_data_supervised_learning.parquet`) in, quantile regression out.

    1. **Load & Explore** — `metering_data_supervised_learning.parquet` (26 engineered
       features: lags, rolling stats, cyclical time encoding, lagged daily context)
    2. **Feature Overview** — ranges and correlation with the target
    3. **Train/Test Split** — chronological, same asset/test length as the SARIMA notebook
    4. **Fit Baseline LightGBM** — all 26 features, default hyperparams — needed to get a
       first read on feature importance before selecting anything
    5. **Feature Importance** — which engineered features the baseline model splits on
    6. **Feature Selection** — keep the top-10 by importance
    7. **Hyperparameter Tuning** — Optuna study over `num_leaves`/`learning_rate`/
       `n_estimators` on the reduced feature set, scored on a validation slice of train
    8. **Fit Final LightGBM** — top-10 features + tuned hyperparams — this is the model
       every section from here on evaluates, forecasts, and plots
    9. **Forecasting** — point forecast + 80% prediction interval from the fitted quantiles
    10. **Evaluation** — MAE, RMSE, MAPE, PI coverage (`ModelEvaluator`, same contract as SARIMA)
    11. **Uncertainty Analysis** — does interval width vary across the test set?
    12. **Probability Events** — P(forecast > threshold), interpolated from the fitted quantiles
    13. **Visualizations** — reuses `TSPlotter`, unmodified
    14. **Comparison** — baseline vs. tuned-and-reduced vs. SARIMA, same asset/test window

    ## Key Concepts

    **Supervised learning vs. classical time series**: instead of modeling the series
    directly (SARIMA), the target is regressed on engineered features built from its own
    history. See
    [`docs_src/modeling/supervised_vs_statistical_models.md`](../../docs_src/modeling/supervised_vs_statistical_models.md).

    **Quantile regression for uncertainty**: three independent models trained on the
    pinball loss for P10/P50/P90, instead of a single point forecast plus a residual-std
    margin. See
    [`docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning models.md`](<../../docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning models.md>).

    **Why a baseline fit comes before tuning/selection**: choosing which features to keep
    needs feature importances, and importances need a fitted model. So Section 4 fits once
    on everything just to get that read, Sections 6-7 use it to narrow down features and
    hyperparameters, and Section 8 is the one real fit whose forecasts, metrics, and plots
    the rest of the notebook shows — not the baseline's.

    ### A caveat this notebook does not paper over

    The feature matrix's lag columns (`feat_lag_1` ... `feat_lag_96`) hold **true observed
    values** for both train and test rows — that's what makes a leakage-safe training set
    (see [`feature_engineering.md`](<../../docs_src/modeling/supervised_learning_models/feature_engineering.md>)).
    Evaluating on the test set this way answers "how good are these forecasts if every lag
    feature is exactly correct up to the forecast target," which is **not** the same as a
    genuine multi-step-ahead deployment, where lags beyond the shortest one (`feat_lag_1`)
    would themselves have to be forecast or held fixed. `LightGBMModel`'s *internal-features*
    mode (used by `lightgbm_residual_diagnostics.py`) is recursive and does compound error
    with horizon — this notebook's external-features mode does not, and that is an
    optimistic simplification worth remembering when comparing its metrics to SARIMA's.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Load and Explore Data

    Load the supervised-learning feature matrix for a single asset.
    """)
    return


@app.cell
def _(pl):
    sup = pl.read_parquet("../../src/data/metering_data_supervised_learning.parquet")
    asset_id = "ASSET_001"
    asset_sup = sup.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    feature_cols = sorted([c for c in asset_sup.columns if c.startswith("feat_")])

    print("Data Summary:")
    print(f"  Asset: {asset_id}")
    print(f"  Rows: {asset_sup.shape[0]} ({asset_sup.shape[0] / 48:.1f} days)")
    print(f"  Features: {len(feature_cols)}")
    print(
        f"  Date range: {asset_sup['timestamp'].min()} to {asset_sup['timestamp'].max()}"
    )
    return asset_id, asset_sup, feature_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Feature Overview

    Ranges of the engineered features, and which correlate most with the target.
    """)
    return


@app.cell
def _(asset_sup, feature_cols, pl):
    feature_ranges = asset_sup.select(
        [pl.col(c).min().alias(f"{c}_min") for c in feature_cols]
        + [pl.col(c).max().alias(f"{c}_max") for c in feature_cols]
    )

    correlations = pl.DataFrame(
        {
            "feature": feature_cols,
            "corr_with_target": [
                asset_sup.select(pl.corr(c, "metering_kwh")).item()
                for c in feature_cols
            ],
        }
    ).with_columns(pl.col("corr_with_target").abs().alias("abs_corr"))

    constant_features = correlations.filter(pl.col("corr_with_target").is_null())[
        "feature"
    ].to_list()
    ranked_correlations = correlations.filter(
        pl.col("corr_with_target").is_not_null()
    ).sort("abs_corr", descending=True)

    if constant_features:
        print(
            f"Constant for this asset (correlation undefined): {constant_features}"
        )
    print("Top 10 features by |correlation| with metering_kwh:")
    print(ranked_correlations.select(["feature", "corr_with_target"]).head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Train/Test Split

    Chronological split, same 4-day test length as `sarima.py` — but note the supervised
    table starts 2 days later per asset (the first 96 rows are dropped for lookback, see
    [`feature_engineering.md`](<../../docs_src/modeling/supervised_learning_models/feature_engineering.md>)),
    so train here is 8 days rather than 10.
    """)
    return


@app.cell
def _(asset_sup, feature_cols):
    test_len = 4 * 48  # 4 days

    train_sup = asset_sup[:-test_len]
    test_sup = asset_sup[-test_len:]

    X_train = train_sup.select(feature_cols).to_numpy()
    X_test = test_sup.select(feature_cols).to_numpy()
    y_train = train_sup["metering_kwh"].to_numpy()
    y_test = test_sup["metering_kwh"].to_numpy()
    train_timestamps = train_sup["timestamp"].to_numpy()
    test_timestamps = test_sup["timestamp"].to_numpy()

    print("Train/Test Split:")
    print(f"  Train: {len(y_train)} observations ({len(y_train) / 48:.1f} days)")
    print(f"  Test:  {len(y_test)} observations ({len(y_test) / 48:.1f} days)")
    return X_test, X_train, test_timestamps, train_timestamps, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Fit Baseline LightGBM (All Features, Default Hyperparams)

    External-features mode: `X_train` supplied, so `LightGBMModel` fits three
    `LGBMRegressor` models (`objective="quantile"`, one per level in `quantile_levels`) on
    the full engineered feature matrix. This baseline exists only to produce feature
    importances for Section 6 — it is not the model the rest of the notebook evaluates.
    """)
    return


@app.cell
def _(LightGBMModel, X_train, y_train):
    print("Fitting baseline LightGBM (all 26 features, default hyperparams)...")

    lgbm_baseline = LightGBMModel(
        y_train, X_train=X_train, quantile_levels=[0.1, 0.5, 0.9]
    )
    lgbm_baseline.fit()

    print(f"  Status: {'Fitted' if lgbm_baseline.fitted else 'Failed'}")
    return (lgbm_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Feature Importance (Baseline)

    Which engineered features the baseline's P50 (median) model actually splits on.
    """)
    return


@app.cell
def _(feature_cols, lgbm_baseline, pl):
    importances = pl.DataFrame(
        {
            "feature": feature_cols,
            "importance": lgbm_baseline.models[0.5].feature_importances_,
        }
    ).sort("importance", descending=True)

    print("Feature importance (baseline P50 model), top 10:")
    print(importances.head(10))
    return (importances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Feature Selection

    Keep the top-10 features by baseline importance — the cheapest way to check whether
    the other 16 are earning their place, without a formal selection procedure
    (`docs_src/data/feature_selection.md` covers PACF/information-criteria approaches for
    when this needs to be more rigorous). This reduced set is what Section 7's tuning and
    Section 8's final fit both use.
    """)
    return


@app.cell
def _(X_test, X_train, feature_cols, importances):
    top_features = importances["feature"].head(10).to_list()
    top_idx = [feature_cols.index(f) for f in top_features]

    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    print(f"Selected features ({len(top_features)}): {top_features}")
    return X_test_sel, X_train_sel, top_features


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Hyperparameter Tuning (Optuna)

    `ModelTuner.grid_search` (`ts_evaluation.py`) only calls `model_class(y_train,
    **params)` and `forecast(steps, confidence_level)` — no `X_train`/`X_test` — so it
    doesn't support this notebook's external-features mode without extending the
    framework. An Optuna study plugged in directly here instead: `optuna.create_study`
    samples `num_leaves`/`learning_rate`/`n_estimators` (TPE sampler, the Optuna default —
    picks the next trial's parameters based on what previous trials scored, rather than
    exhausting a fixed grid), each trial fit on the Section 6 feature subset and scored on
    a validation slice carved out of *train* (last 2 days), never on `y_test` — tuning
    against the test set would make the reported improvement optimistic by construction
    (same reasoning `ModelTuner`'s docstring gives).
    """)
    return


@app.cell
def _(LightGBMModel, ModelEvaluator, X_train_sel, optuna, y_train):
    val_len = 2 * 48  # 2 days, carved out of train

    tune_train_X, tune_val_X = X_train_sel[:-val_len], X_train_sel[-val_len:]
    tune_train_y, tune_val_y = y_train[:-val_len], y_train[-val_len:]

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        }
        candidate = LightGBMModel(
            tune_train_y,
            X_train=tune_train_X,
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            quantile_levels=[0.1, 0.5, 0.9],
        )
        candidate.fit(n_estimators=params["n_estimators"])
        candidate_forecast = candidate.forecast(
            steps=len(tune_val_y), confidence_level=0.80, X_test=tune_val_X
        )
        return ModelEvaluator.evaluate(tune_val_y, candidate_forecast).rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_num_leaves = study.best_params["num_leaves"]
    best_learning_rate = study.best_params["learning_rate"]
    best_n_estimators = study.best_params["n_estimators"]

    print(f"Optuna study: {len(study.trials)} trials, scored on a 2-day validation slice of train")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return best_learning_rate, best_n_estimators, best_num_leaves


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 8: Fit Final LightGBM (Top-10 Features, Tuned Hyperparams)

    This is the model every section from here on evaluates, forecasts, and plots —
    Sections 4-7 exist to arrive at this configuration, not to be read as a fair
    default-vs-tuned comparison on their own (that comparison is Section 14).
    """)
    return


@app.cell
def _(
    LightGBMModel,
    X_train_sel,
    best_learning_rate,
    best_n_estimators,
    best_num_leaves,
    y_train,
):
    print(
        f"Fitting final LightGBM (top-10 features, num_leaves={best_num_leaves}, "
        f"learning_rate={best_learning_rate:.4f}, n_estimators={best_n_estimators})..."
    )

    lgbm = LightGBMModel(
        y_train,
        X_train=X_train_sel,
        num_leaves=best_num_leaves,
        learning_rate=best_learning_rate,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    lgbm.fit(n_estimators=best_n_estimators)

    print(f"  Status: {'Fitted' if lgbm.fitted else 'Failed'}")
    print(f"  Quantile levels: {lgbm.quantile_levels}")
    return (lgbm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 9: Generate Forecasts with Uncertainty

    `forecast()` in external-features mode requires a matching `X_test` — here, the
    Section 6 feature subset for the test window (see the Summary's caveat about what
    these lag features actually represent on the test set).
    """)
    return


@app.cell
def _(X_test_sel, lgbm, y_test):
    forecast_output = lgbm.forecast(
        steps=len(y_test), confidence_level=0.80, X_test=X_test_sel
    )
    yhat = forecast_output.prediction
    lower = forecast_output.lower
    upper = forecast_output.upper

    assert (lower <= yhat).all() and (yhat <= upper).all(), "quantile monotonicity failed"

    print(f"Generated {len(y_test)} forecasts with P10-P90 prediction interval")
    print("Monotonicity check (P10 <= P50 <= P90): OK")
    return (forecast_output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 10: Standard Forecast Output Contract

    Same contract as every other model in the framework:

    | Column | Description |
    |--------|-------------|
    | `asset_id` | Unique identifier |
    | `timestamp` | Forecast applies to this time |
    | `prediction` | Point forecast (P50) |
    | `lower` | Lower bound of prediction interval (P10) |
    | `upper` | Upper bound of prediction interval (P90) |
    | `uncertainty_width` | Interval width (upper - lower) |
    """)
    return


@app.cell
def _(asset_id, forecast_output, pl, test_timestamps, y_test):
    forecast_df = pl.DataFrame(
        {
            "asset_id": [asset_id] * len(forecast_output.prediction),
            "timestamp": test_timestamps,
            "prediction": forecast_output.prediction,
            "lower": forecast_output.lower,
            "upper": forecast_output.upper,
            "uncertainty_width": forecast_output.uncertainty_width,
            "actual": y_test,
        }
    )

    print("Forecast Output (first 5 rows):")
    print(
        forecast_df.head(5).select(
            ["timestamp", "prediction", "lower", "upper", "actual"]
        )
    )
    return (forecast_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 11: Model Evaluation
    """)
    return


@app.cell
def _(ModelEvaluator, forecast_output, y_test):
    metrics = ModelEvaluator.evaluate(y_test, forecast_output)

    print("Performance:")
    print(f"  MAE:  {metrics.mae:.4f} kWh")
    print(f"  RMSE: {metrics.rmse:.4f} kWh")
    print(f"  MAPE: {metrics.mape:.2f}%")
    print(f"  PI Coverage: {metrics.pi_coverage:.1f}% (target: 80%)")
    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 12: Uncertainty Analysis

    Unlike SARIMA/ExponentialSmoothing's constant residual-std margin, quantile
    regression's interval width can vary row-by-row with the input features.
    """)
    return


@app.cell
def _(forecast_df):
    uncertainty_mean = forecast_df["uncertainty_width"].mean()
    uncertainty_std = forecast_df["uncertainty_width"].std()
    uncertainty_min = forecast_df["uncertainty_width"].min()
    uncertainty_max = forecast_df["uncertainty_width"].max()

    print(f"Uncertainty Metrics:")
    print(f"  Mean width: {uncertainty_mean:.4f} kWh")
    print(f"  Std dev: {uncertainty_std:.4f} kWh")
    print(f"  Min: {uncertainty_min:.4f} kWh")
    print(f"  Max: {uncertainty_max:.4f} kWh")
    print(
        f"  Std dev > 0 confirms interval width varies across the test set "
        f"({'varies' if uncertainty_std > 1e-6 else 'constant'})"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 13: Forecast Probability of Events

    P(forecast > threshold), interpolated directly from the three fitted quantile
    predictions rather than assuming a normal distribution around the point forecast.
    """)
    return


@app.cell
def _(X_test_sel, lgbm, np, y_train):
    thresholds = [
        np.percentile(y_train, 25),
        np.percentile(y_train, 50),
        np.percentile(y_train, 75),
    ]

    q_levels = np.array(lgbm.quantile_levels)
    q_preds = np.stack(
        [lgbm.models[q].predict(X_test_sel) for q in lgbm.quantile_levels], axis=1
    )
    q_preds_sorted = np.sort(q_preds, axis=1)

    print("Forecast Probability of Events (interpolated from P10/P50/P90):")
    for threshold in thresholds:
        # For each row, interpolate the fitted quantile levels against the threshold to
        # estimate P(Y <= threshold), then P(Y > threshold) = 1 - that.
        prob_below = np.array(
            [
                np.interp(threshold, row, q_levels, left=0.0, right=1.0)
                for row in q_preds_sorted
            ]
        )
        prob_exceed = 1 - prob_below
        mean_prob = np.mean(prob_exceed)
        print(f"  P(forecast > {threshold:.2f} kWh) = {mean_prob:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 14: Visualizations
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, metrics, test_timestamps, y_test):
    fig1 = TSPlotter.forecast_vs_actual(
        y_test, forecast_output, "LightGBM (quantile)", metrics, x=test_timestamps
    )
    fig1
    return


@app.cell
def _(TSPlotter, forecast_output, test_timestamps, y_test):
    fig2 = TSPlotter.residuals_diagnostic(
        y_test, forecast_output, "LightGBM (quantile)", x=test_timestamps
    )
    fig2
    return


@app.cell
def _(TSPlotter, forecast_output, test_timestamps, y_test):
    fig3 = TSPlotter.pi_coverage(
        y_test, forecast_output, "LightGBM (quantile)", x=test_timestamps
    )
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 15: Comparison — Baseline vs. Tuned+Reduced vs. SARIMA

    Same asset, same test window (`test_timestamps` from Section 3). Includes the
    Section 4 baseline (all 26 features, default hyperparams) alongside the Section 8
    final model, so the selection/tuning work in Sections 6-7 shows up as a real
    before/after rather than being asserted. SARIMA is trained on all raw history
    available before the test window starts, including the 2 days the supervised table
    drops for lookback, so its training window is slightly longer.
    """)
    return


@app.cell
def _(ModelEvaluator, X_test, lgbm_baseline, y_test):
    baseline_forecast = lgbm_baseline.forecast(
        steps=len(y_test), confidence_level=0.80, X_test=X_test
    )
    baseline_metrics = ModelEvaluator.evaluate(y_test, baseline_forecast)
    return (baseline_metrics,)


@app.cell
def _(SARIMAModel, pl, test_timestamps, train_timestamps):
    raw = pl.read_parquet("../../src/data/metering_data.parquet")
    raw_asset = raw.filter(pl.col("asset_id") == "ASSET_001").sort("timestamp")

    sarima_train = raw_asset.filter(
        pl.col("timestamp") <= train_timestamps.max()
    )["metering_kwh"].to_numpy()

    sarima_test = raw_asset.filter(
        pl.col("timestamp").is_in(pl.Series(test_timestamps))
    ).sort("timestamp")["metering_kwh"].to_numpy()

    print(f"SARIMA train: {len(sarima_train)} obs, test: {len(sarima_test)} obs")

    sarima = SARIMAModel(sarima_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48))
    sarima.fit()
    sarima_forecast = sarima.forecast(steps=len(sarima_test), confidence_level=0.80)
    return sarima_forecast, sarima_test


@app.cell
def _(
    ComparisonPlotter,
    ModelEvaluator,
    baseline_metrics,
    metrics,
    sarima_forecast,
    sarima_test,
):
    sarima_metrics = ModelEvaluator.evaluate(sarima_test, sarima_forecast)

    comparison_fig = ComparisonPlotter.metrics_comparison(
        {
            "LightGBM (baseline)": baseline_metrics,
            "LightGBM (tuned+reduced)": metrics,
            "SARIMA": sarima_metrics,
        }
    )
    comparison_fig
    return (sarima_metrics,)


@app.cell
def _(baseline_metrics, metrics, sarima_metrics):
    print("Side-by-side:")
    print(f"  LightGBM (baseline):      {baseline_metrics}")
    print(f"  LightGBM (tuned+reduced): {metrics}")
    print(f"  SARIMA:                   {sarima_metrics}")

    tuning_helped = metrics.rmse < baseline_metrics.rmse
    print(
        f"\nTuning + feature selection {'improved' if tuning_helped else 'did not improve'} "
        f"RMSE vs. the baseline ({baseline_metrics.rmse:.4f} -> {metrics.rmse:.4f})."
    )
    print(
        "Remember the Summary's caveat: LightGBM's test-set lag features are true "
        "observed values, not recursively forecast ones, so the SARIMA comparison is not "
        "fully apples-to-apples with SARIMA's genuine multi-step forecast."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Next Steps

    - Quantify the optimism from using true lag features on the test set — e.g. compare
      against `LightGBMModel`'s internal-features (recursive) mode on the same asset/window.
    - Section 7's study is intentionally small (20 trials, 3 parameters); widen the search
      space or trial count, or extend `ModelTuner` to support `X_train`/`X_test` if this
      becomes a recurring need across notebooks rather than a one-off here.
    - Section 6 selects features by baseline importance alone; a more rigorous pass (PACF,
      information criteria — see `docs_src/data/feature_selection.md`) may find a different
      subset than "top-10 by importance."
    """)
    return


if __name__ == "__main__":
    app.run()
