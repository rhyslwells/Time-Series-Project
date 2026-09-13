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
    from scipy.stats import norm
    import warnings

    from ts_models import LightGBMModel, SARIMAModel
    from ts_evaluation import ModelEvaluator
    from ts_plots import TSPlotter, ComparisonPlotter

    warnings.filterwarnings("ignore")
    return (
        ComparisonPlotter,
        LightGBMModel,
        ModelEvaluator,
        SARIMAModel,
        TSPlotter,
        mo,
        np,
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
    4. **Fit LightGBM (quantile regression)** — three `LGBMRegressor` models (P10/P50/P90)
    5. **Feature Importance** — which engineered features the P50 model actually splits on
    6. **Forecasting** — point forecast + 80% prediction interval from the fitted quantiles
    7. **Evaluation** — MAE, RMSE, MAPE, PI coverage (`ModelEvaluator`, same contract as SARIMA)
    8. **Uncertainty Analysis** — does interval width vary across the test set?
    9. **Probability Events** — P(forecast > threshold), interpolated from the fitted quantiles
    10. **Visualizations** — reuses `TSPlotter`, unmodified
    11. **Comparison to SARIMA** — same asset, same test window

    ## Key Concepts

    **Supervised learning vs. classical time series**: instead of modeling the series
    directly (SARIMA), the target is regressed on engineered features built from its own
    history. See
    [`docs_src/modeling/supervised_vs_statistical_models.md`](../../docs_src/modeling/supervised_vs_statistical_models.md).

    **Quantile regression for uncertainty**: three independent models trained on the
    pinball loss for P10/P50/P90, instead of a single point forecast plus a residual-std
    margin. See
    [`docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning models.md`](<../../docs_src/modeling/supervised_learning_models/Quantile Regression with Supervised Learning models.md>).

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
    ## Section 4: Fit LightGBM (Quantile Regression)

    External-features mode: `X_train` supplied, so `LightGBMModel` fits three
    `LGBMRegressor` models (`objective="quantile"`, one per level in `quantile_levels`)
    on the engineered feature matrix instead of building its own lag features from
    `y_train` alone.
    """)
    return


@app.cell
def _(LightGBMModel, X_train, y_train):
    print("Fitting LightGBM (P10/P50/P90 quantile models)...")

    lgbm = LightGBMModel(y_train, X_train=X_train, quantile_levels=[0.1, 0.5, 0.9])
    lgbm.fit()

    print(f"  Status: {'Fitted' if lgbm.fitted else 'Failed'}")
    print(f"  Quantile levels: {lgbm.quantile_levels}")
    return (lgbm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Feature Importance

    Which engineered features the P50 (median) model actually splits on.
    """)
    return


@app.cell
def _(feature_cols, lgbm, pl):
    importances = pl.DataFrame(
        {
            "feature": feature_cols,
            "importance": lgbm.models[0.5].feature_importances_,
        }
    ).sort("importance", descending=True)

    print("Feature importance (P50 model), top 10:")
    print(importances.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Generate Forecasts with Uncertainty

    `forecast()` in external-features mode requires a matching `X_test` — the
    pre-engineered feature rows for the forecast horizon (see the caveat in the Summary
    about what these lag features actually represent on the test set).
    """)
    return


@app.cell
def _(X_test, lgbm, y_test):
    forecast_output = lgbm.forecast(
        steps=len(y_test), confidence_level=0.80, X_test=X_test
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
    ## Section 7: Standard Forecast Output Contract

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
    ## Section 8: Model Evaluation
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
    ## Section 9: Uncertainty Analysis

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
    ## Section 10: Forecast Probability of Events

    P(forecast > threshold), interpolated directly from the three fitted quantile
    predictions rather than assuming a normal distribution around the point forecast.
    """)
    return


@app.cell
def _(X_test, lgbm, np, y_train):
    thresholds = [
        np.percentile(y_train, 25),
        np.percentile(y_train, 50),
        np.percentile(y_train, 75),
    ]

    q_levels = np.array(lgbm.quantile_levels)
    q_preds = np.stack(
        [lgbm.models[q].predict(X_test) for q in lgbm.quantile_levels], axis=1
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
    ## Section 11: Visualizations
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
    ## Section 12: Comparison to SARIMA

    Same asset, same test window (`test_timestamps` from Section 3) — SARIMA is trained
    on all raw history available before the test window starts, including the 2 days the
    supervised table drops for lookback, so its training window is slightly longer.
    """)
    return


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
    metrics,
    sarima_forecast,
    sarima_test,
):
    sarima_metrics = ModelEvaluator.evaluate(sarima_test, sarima_forecast)

    comparison_fig = ComparisonPlotter.metrics_comparison(
        {"LightGBM (quantile)": metrics, "SARIMA": sarima_metrics}
    )
    comparison_fig
    return (sarima_metrics,)


@app.cell
def _(metrics, sarima_metrics):
    print("Side-by-side:")
    print(f"  LightGBM (quantile): {metrics}")
    print(f"  SARIMA:               {sarima_metrics}")

    better = "LightGBM" if metrics.rmse < sarima_metrics.rmse else "SARIMA"
    print(f"\nLower RMSE: {better}")
    print(
        "Remember the Summary's caveat: LightGBM's test-set lag features are true "
        "observed values, not recursively forecast ones, so this comparison is not "
        "fully apples-to-apples with SARIMA's genuine multi-step forecast."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Next Steps

    - Quantify the optimism from using true lag features on the test set — e.g. compare
      against `LightGBMModel`'s internal-features (recursive) mode on the same asset/window.
    - Tune `num_leaves`/`learning_rate` per quantile level rather than sharing them across
      all three (see Task 2.1's open question in `LIGHTGBM_PLAN.md`).
    - Feature selection: the full 26-feature set is used as-is; narrow it using the
      importances from Section 5 and `docs_src/data/feature_selection.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
