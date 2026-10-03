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
    import warnings

    from ts_models import SeasonalWindowAverageModel
    from ts_evaluation import ModelEvaluator
    from ts_plots import TSPlotter, ComparisonPlotter

    warnings.filterwarnings("ignore")
    return (
        ComparisonPlotter,
        ModelEvaluator,
        SeasonalWindowAverageModel,
        TSPlotter,
        mo,
        np,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    **SeasonalWindowAverage** baseline: at each time of day, average the previous
    `window` days' values at that same time. E.g. for 13:00 with `window=7`, the
    forecast is the mean of the last 7 days' 13:00 readings.

    This notebook demonstrates its use and compares it against **SeasonalNaive**
    (`window=1`, i.e. "copy the same time yesterday") to show what averaging
    buys you:

    1. **Load & Explore** — 14-day metering data (30-min intervals)
    2. **Train/Test Split** — 10 days train, 4 days test
    3. **Model Fitting** — SeasonalWindowAverage (window=7) vs SeasonalNaive (window=1)
    4. **Forecasting** — Point forecasts + 80% prediction intervals
    5. **Evaluation** — MAE, RMSE, MAPE, PI coverage, MASE vs SeasonalNaive
    6. **Visualizations** — Forecast vs actual, residuals, PI coverage, side-by-side comparison

    ### Key Takeaways

    - Averaging over several seasonal cycles smooths single-day noise that
      SeasonalNaive carries forward unchanged
    - It remains a simple, interpretable, assumption-light baseline — no trend
      or dependence structure is modeled, only the seasonal mean
    - Useful as a stronger reference point than SeasonalNaive when per-day
      noise (not trend/level shift) dominates the residual
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Load and Explore Data

    Load metering data for a single asset and examine its characteristics.
    """)
    return


@app.cell
def _(pl):
    # Load metering data
    df = pl.read_parquet("../../src/data/metering_data.parquet")
    asset_id = "ASSET_001"
    asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    print("Data Summary:")
    print(f"  Asset: {asset_id}")
    print(f"  Type: {asset_data['asset_type'][0]}")
    print(f"  Records: {asset_data.shape[0]}")
    print(
        f"  Date range: {asset_data['timestamp'].min()} to {asset_data['timestamp'].max()}"
    )

    y = asset_data.select("metering_kwh").to_numpy().flatten()
    timestamps = asset_data.select("timestamp").to_numpy().flatten()
    return asset_id, timestamps, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Train/Test Split

    Split data: 10 days training (480 obs), 4 days test (192 obs)

    `SeasonalWindowAverage(window=7)` needs at least `season_length * window =
    48 * 7 = 336` training points — the 480-observation train set covers this.
    """)
    return


@app.cell
def _(timestamps, y):
    test_split_idx = len(y) - (4 * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]
    train_timestamps = timestamps[:test_split_idx]
    test_timestamps = timestamps[test_split_idx:]

    print("Train/Test Split:")
    print(f"  Train: {len(y_train)} observations ({len(y_train) / 48:.1f} days)")
    print(f"  Test:  {len(y_test)} observations ({len(y_test) / 48:.1f} days)")
    return test_timestamps, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Fit SeasonalWindowAverage (and SeasonalNaive baseline)

    - **SeasonalWindowAverage(season_length=48, window=7)**: mean of the same
      half-hour slot across the previous 7 days
    - **SeasonalWindowAverage(season_length=48, window=1)**: the previous
      occurrence of that slot only — this is the seasonal-naive special case,
      included for comparison (it auto-names itself "SeasonalNaive")
    """)
    return


@app.cell
def _(SeasonalWindowAverageModel, y_train):
    print("Fitting SeasonalWindowAverage model...")

    swa = SeasonalWindowAverageModel(y_train, season_length=48, window=7)
    swa.fit()
    print(f"  Status: {'Fitted' if swa.fitted else 'Failed'}")
    print(f"  Params: {swa.get_params()}")

    print("\nFitting SeasonalNaive baseline (window=1)...")
    naive = SeasonalWindowAverageModel(y_train, season_length=48, window=1)
    naive.fit()
    print(f"  Status: {'Fitted' if naive.fitted else 'Failed'}")
    print(f"  Params: {naive.get_params()}")
    return (naive, swa)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Generate Forecasts with Uncertainty

    Produce 80% prediction intervals alongside point forecasts for both models.
    """)
    return


@app.cell
def _(naive, swa, y_test):
    forecast_steps = len(y_test)
    confidence_level = 0.80

    swa_forecast = swa.forecast(steps=forecast_steps, confidence_level=confidence_level)
    naive_forecast = naive.forecast(
        steps=forecast_steps, confidence_level=confidence_level
    )

    print(
        f"Generated {forecast_steps} forecasts with {confidence_level * 100:.0f}% confidence interval"
    )
    return confidence_level, forecast_steps, naive_forecast, swa_forecast


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Standard Forecast Output Contract

    All forecasts follow a standard format for reusability:

    | Column | Description |
    |--------|-------------|
    | `asset_id` | Unique identifier |
    | `timestamp` | Forecast applies to this time |
    | `prediction` | Point forecast (expected value) |
    | `lower` | Lower bound of prediction interval |
    | `upper` | Upper bound of prediction interval |
    | `uncertainty_width` | Interval width (upper - lower) |
    """)
    return


@app.cell
def _(asset_id, pl, swa_forecast, test_timestamps, y_test):
    forecast_df = pl.DataFrame(
        {
            "asset_id": [asset_id] * len(swa_forecast.prediction),
            "timestamp": test_timestamps,
            "prediction": swa_forecast.prediction,
            "lower": swa_forecast.lower,
            "upper": swa_forecast.upper,
            "uncertainty_width": swa_forecast.uncertainty_width,
            "actual": y_test,
        }
    )

    print("SeasonalWindowAverage Forecast Output (first 5 rows):")
    print(
        forecast_df.head(5).select(
            ["timestamp", "prediction", "lower", "upper", "actual"]
        )
    )
    return (forecast_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Model Evaluation

    Assess forecast quality using multiple metrics, and compute **MASE**
    (SeasonalWindowAverage's MAE / SeasonalNaive's MAE) to see whether
    averaging over 7 days actually beats copying yesterday.

    - **MAE**: Mean Absolute Error (average absolute difference)
    - **RMSE**: Root Mean Squared Error (penalizes large errors)
    - **MAPE**: Mean Absolute Percentage Error (% error)
    - **PI Coverage**: What fraction of actuals fell within the prediction interval?
    - **MASE < 1**: SeasonalWindowAverage beats SeasonalNaive; **>= 1**: it does not
    """)
    return


@app.cell
def _(ModelEvaluator, confidence_level, naive_forecast, swa_forecast, y_test):
    swa_metrics = ModelEvaluator.evaluate(y_test, swa_forecast)
    naive_metrics = ModelEvaluator.evaluate(y_test, naive_forecast)
    mase = swa_metrics.mae / naive_metrics.mae

    print("SeasonalWindowAverage:")
    print(f"  MAE:  {swa_metrics.mae:.4f} kWh")
    print(f"  RMSE: {swa_metrics.rmse:.4f} kWh")
    print(f"  MAPE: {swa_metrics.mape:.2f}%")
    print(
        f"  PI Coverage: {swa_metrics.pi_coverage:.1f}% (target: {confidence_level * 100:.0f}%)"
    )

    print("\nSeasonalNaive (baseline):")
    print(f"  MAE:  {naive_metrics.mae:.4f} kWh")
    print(f"  RMSE: {naive_metrics.rmse:.4f} kWh")
    print(f"  MAPE: {naive_metrics.mape:.2f}%")

    print(f"\nMASE (SeasonalWindowAverage vs SeasonalNaive): {mase:.3f}")
    return naive_metrics, swa_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Visualizations

    Forecast diagnostics for SeasonalWindowAverage, plus a side-by-side
    comparison against SeasonalNaive.
    """)
    return


@app.cell
def _(TSPlotter, swa_forecast, swa_metrics, test_timestamps, y_test):
    # Plot 1: Forecast vs Actual with prediction interval
    fig1 = TSPlotter.forecast_vs_actual(
        y_test, swa_forecast, "SeasonalWindowAverage", swa_metrics, x=test_timestamps
    )
    fig1
    return


@app.cell
def _(TSPlotter, swa_forecast, test_timestamps, y_test):
    # Plot 2: Residual diagnostics
    fig2 = TSPlotter.residuals_diagnostic(
        y_test, swa_forecast, "SeasonalWindowAverage", x=test_timestamps
    )
    fig2
    return


@app.cell
def _(TSPlotter, swa_forecast, test_timestamps, y_test):
    # Plot 3: Prediction interval coverage
    fig3 = TSPlotter.pi_coverage(
        y_test, swa_forecast, "SeasonalWindowAverage", x=test_timestamps
    )
    fig3
    return


@app.cell
def _(ComparisonPlotter, naive_forecast, swa_forecast, test_timestamps, y_test):
    # Plot 4: SeasonalWindowAverage vs SeasonalNaive forecasts overlaid
    fig4 = ComparisonPlotter.forecast_comparison(
        y_test,
        {"SeasonalWindowAverage": swa_forecast, "SeasonalNaive": naive_forecast},
        sample_size=96,
        x=test_timestamps,
    )
    fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Next Steps

    - Sweep `window` (e.g. 3, 5, 7, 14) to see how smoothing trades off against
      responsiveness to recent level shifts
    - Add SeasonalWindowAverage to `ts_model_explorer.py`'s full model
      comparison (done) to rank it alongside SARIMA/ExponentialSmoothing/LightGBM
    """)
    return


if __name__ == "__main__":
    app.run()
