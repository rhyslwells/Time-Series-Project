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

    from ts_model_framework import SARIMAModel, ModelEvaluator
    from ts_plots import TSPlotter

    warnings.filterwarnings("ignore")
    return ModelEvaluator, SARIMAModel, TSPlotter, mo, norm, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    Complete pipeline for energy metering forecasts with **uncertainty quantification**. Complete SARIMA forecasting pipeline demonstrated:

    1. **Load & Explore** — 14-day metering data (30-min intervals)
    2. **Train/Test Split** — 10 days train, 4 days test
    3. **Model Fitting** — SARIMA(1,1,1)×(1,1,1,48)
    4. **Forecasting** — Point forecasts + 80% prediction intervals
    5. **Evaluation** — MAE, RMSE, MAPE, PI coverage
    6. **Uncertainty Analysis** — When is the model uncertain?
    7. **Probability Events** — P(forecast > threshold)
    8. **Visualizations** — Time series, accuracy, residuals

    ## Key Concepts

    **Forecast as a Data Product**: Instead of just point forecasts, we produce:
    - Expected metering value: $E[Y_t]$
    - Prediction intervals: $P_{10}(Y_t)$, $P_{50}(Y_t)$, $P_{90}(Y_t)$

    ### Key Takeaways

    - SARIMA effectively captures daily seasonality
    - Prediction intervals quantify forecast uncertainty
    - Standard output contract enables downstream analytics
    - Multiple evaluation metrics reveal different insights
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
    ## Section 3: Fit SARIMA Model

    **SARIMA(1,1,1)×(1,1,1,48)**

    Parameters:
    - **(p,d,q) = (1,1,1)**: Non-seasonal autoregressive, differencing, moving average
    - **(P,D,Q,s) = (1,1,1,48)**: Seasonal components with s=48 (one day)

    This captures daily seasonality patterns in 30-minute energy metering data.
    """)
    return


@app.cell
def _(SARIMAModel, y_train):
    print("Fitting SARIMA model...")

    sarima = SARIMAModel(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48))
    sarima.fit()

    print(f"  Status: {'Fitted' if sarima.fitted else 'Failed'}")
    print(f"  AIC: {sarima.model.aic:.2f}")
    return (sarima,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Generate Forecasts with Uncertainty

    Produce 80% prediction intervals (P10-P90) alongside point forecasts.

    A wide interval indicates high uncertainty; a narrow interval indicates high confidence.
    """)
    return


@app.cell
def _(sarima, y_test):
    forecast_steps = len(y_test)
    confidence_level = 0.80

    forecast_output = sarima.forecast(
        steps=forecast_steps, confidence_level=confidence_level
    )
    yhat = forecast_output.prediction
    lower = forecast_output.lower
    upper = forecast_output.upper

    print(
        f"Generated {forecast_steps} forecasts with {confidence_level * 100:.0f}% confidence interval"
    )
    return confidence_level, forecast_output, lower, upper, yhat


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
    ## Section 6: Model Evaluation

    Assess forecast quality using multiple metrics:

    - **MAE**: Mean Absolute Error (average absolute difference)
    - **RMSE**: Root Mean Squared Error (penalizes large errors)
    - **MAPE**: Mean Absolute Percentage Error (% error)
    - **PI Coverage**: What fraction of actuals fell within the prediction interval?
    """)
    return


@app.cell
def _(ModelEvaluator, confidence_level, forecast_output, y_test):
    metrics = ModelEvaluator.evaluate(y_test, forecast_output)
    mae = metrics.mae
    rmse = metrics.rmse
    mape = metrics.mape * 100
    coverage = metrics.pi_coverage

    print(f"Performance:")
    print(f"  MAE:  {mae:.4f} kWh")
    print(f"  RMSE: {rmse:.4f} kWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  PI Coverage: {coverage:.1f}% (target: {confidence_level * 100:.0f}%)")
    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Uncertainty Analysis

    Understand when the model is most uncertain.

    High uncertainty can indicate:
    - Unusual patterns in the data
    - Lack of historical precedent
    - Genuine variability in the asset's behavior
    - System constraints or operating changes
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 8: Forecast Probability of Events

    Calculate P(forecast > threshold) for operational decisions.

    Example: What's the probability metering exceeds 5 kW?
    """)
    return


@app.cell
def _(lower, norm, np, upper, y_train, yhat):
    thresholds = [
        np.percentile(y_train, 25),
        np.percentile(y_train, 50),
        np.percentile(y_train, 75),
    ]

    print("Forecast Probability of Events:")
    for threshold in thresholds:
        forecast_std = (upper - lower) / (2 * 1.645)
        prob_exceed = 1 - norm.cdf(threshold, loc=yhat, scale=forecast_std)
        mean_prob = np.mean(prob_exceed)
        print(f"  P(forecast > {threshold:.2f} kWh) = {mean_prob:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 9: Visualizations

    Three key plots for forecast analysis.
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, metrics, y_test):
    # Plot 1: Forecast vs Actual with prediction interval
    fig1 = TSPlotter.forecast_vs_actual(y_test, forecast_output, "SARIMA", metrics)
    fig1
    return


@app.cell
def _(TSPlotter, forecast_output, y_test):
    # Plot 2: Residual diagnostics
    fig2 = TSPlotter.residuals_diagnostic(y_test, forecast_output, "SARIMA")
    fig2
    return


@app.cell
def _(TSPlotter, forecast_output, y_test):
    # Plot 3: Prediction interval coverage
    fig3 = TSPlotter.pi_coverage(y_test, forecast_output, "SARIMA")
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Next Steps

    - Test alternative SARIMA parameters (p, d, q)
    """)
    return


if __name__ == "__main__":
    app.run()
