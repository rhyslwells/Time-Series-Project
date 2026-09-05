import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")

# marimo edit working_notes\2_basic_forecasting\sarima_final.py


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime
    import plotly.graph_objects as go
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import norm
    import warnings

    warnings.filterwarnings("ignore")

    return (
        mo,
        pl,
        np,
        datetime,
        go,
        SARIMAX,
        mean_absolute_error,
        mean_squared_error,
        norm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
# SARIMA Time Series Forecasting for Energy Assets

Complete pipeline for energy metering forecasts with **uncertainty quantification**.

## Key Concepts

**Forecast as a Data Product**: Instead of just point forecasts, we produce:
- Expected metering value: $E[Y_t]$
- Prediction intervals: $P_{10}(Y_t)$, $P_{50}(Y_t)$, $P_{90}(Y_t)$

## Workflow

```
Raw Metering Data (14 days, 30-min intervals)
    ↓
Train/Test Split (10 days train, 4 days test)
    ↓
SARIMA(1,1,1)×(1,1,1,48) Model Fitting
    ↓
Forecast Generation (point + 80% intervals)
    ↓
Model Evaluation & Uncertainty Analysis
    ↓
Visualizations & Insights
```
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

    return asset_id, y, timestamps


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## Section 2: Train/Test Split

Split data: 10 days training (480 obs), 4 days test (192 obs)
""")
    return


@app.cell
def _(y, timestamps, pl):
    test_split_idx = len(y) - (4 * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]
    train_timestamps = timestamps[:test_split_idx]
    test_timestamps = timestamps[test_split_idx:]

    print("Train/Test Split:")
    print(f"  Train: {len(y_train)} observations ({len(y_train) / 48:.1f} days)")
    print(f"  Test:  {len(y_test)} observations ({len(y_test) / 48:.1f} days)")

    return y_train, y_test, train_timestamps, test_timestamps


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
def _(y_train, SARIMAX, pl):
    print("Fitting SARIMA model...")

    try:
        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 48),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=1000)
        status = "Success"
        aic = f"{results.aic:.2f}"
    except Exception as e:
        print(f"Error: {e}. Using simplified model...")
        model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 48))
        results = model.fit(disp=False)
        status = "Simplified"
        aic = f"{results.aic:.2f}"

    print(f"  Status: {status}")
    print(f"  AIC: {aic}")

    return model, results


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## Section 4: Generate Forecasts with Uncertainty

Produce 80% prediction intervals (P10-P90) alongside point forecasts.

A wide interval indicates high uncertainty; a narrow interval indicates high confidence.
""")
    return


@app.cell
def _(results, y_test, np, pl):
    forecast_steps = len(y_test)
    confidence_level = 0.80

    forecast = results.get_forecast(steps=forecast_steps)
    yhat = np.asarray(forecast.predicted_mean)
    forecast_ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))
    lower = forecast_ci[:, 0]
    upper = forecast_ci[:, 1]

    print(
        f"Generated {forecast_steps} forecasts with {confidence_level * 100:.0f}% confidence interval"
    )

    return forecast_steps, confidence_level, yhat, lower, upper


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
def _(asset_id, yhat, lower, upper, y_test, test_timestamps, datetime, pl):
    forecast_df = pl.DataFrame(
        {
            "asset_id": [asset_id] * len(yhat),
            "timestamp": test_timestamps,
            "prediction": yhat,
            "lower": lower,
            "upper": upper,
            "uncertainty_width": upper - lower,
            "actual": y_test,
        }
    )

    print("Forecast Output (first 5 rows):")
    print(
        forecast_df.head(5).select(
            ["timestamp", "prediction", "lower", "upper", "actual"]
        )
    )

    return forecast_df


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
def _(
    y_test,
    yhat,
    lower,
    upper,
    confidence_level,
    mean_absolute_error,
    mean_squared_error,
    np,
):
    mae = mean_absolute_error(y_test, yhat)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    mape = np.mean(np.abs((y_test - yhat) / (np.abs(y_test) + 1e-8))) * 100
    coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100

    print(f"Performance:")
    print(f"  MAE:  {mae:.4f} kWh")
    print(f"  RMSE: {rmse:.4f} kWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  PI Coverage: {coverage:.1f}% (target: {confidence_level * 100:.0f}%)")

    return mae, rmse, mape, coverage


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

    return uncertainty_mean, uncertainty_std


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## Section 8: Forecast Probability of Events

Calculate P(forecast > threshold) for operational decisions.

Example: What's the probability metering exceeds 5 kW?
""")
    return


@app.cell
def _(y_train, yhat, upper, lower, norm, np):
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
def _(y_train, y_test, yhat, lower, upper, asset_id, mae, rmse, go):
    # Plot 1: Full time series with train/test split
    fig1 = go.Figure()
    _train_range = list(range(len(y_train)))
    _test_range = list(range(len(y_train), len(y_train) + len(y_test)))

    fig1.add_trace(
        go.Scatter(
            x=_train_range,
            y=y_train,
            mode="lines",
            name="Training data",
            line=dict(width=1),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=_test_range, y=y_test, mode="lines", name="Test data", line=dict(width=1)
        )
    )
    fig1.add_vline(
        x=len(y_train),
        line_dash="dash",
        line_color="red",
        annotation_text="Train/Test split",
    )

    fig1.update_layout(
        title=f"Asset {asset_id}: Time Series with Train/Test Split",
        xaxis_title="Time (30-min intervals)",
        yaxis_title="Metering (kWh)",
        height=500,
        width=1200,
    )
    fig1

    return fig1


@app.cell
def _(y_test, yhat, lower, upper, mae, rmse, go):
    # Plot 2: Forecast vs Actual
    fig2 = go.Figure()
    _test_range = list(range(len(y_test)))

    fig2.add_trace(
        go.Scatter(
            x=_test_range, y=y_test, mode="lines", name="Actual", line=dict(width=2)
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=_test_range,
            y=yhat,
            mode="lines",
            name="Forecast",
            line=dict(width=2),
            opacity=0.7,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=_test_range + _test_range[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,100,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% PI",
        )
    )

    fig2.update_layout(
        title=f"Forecast vs Actual (MAE: {mae:.4f}, RMSE: {rmse:.4f})",
        xaxis_title="Test Period",
        yaxis_title="Metering (kWh)",
        height=500,
        width=1200,
    )
    fig2

    return fig2


@app.cell
def _(y_test, yhat, forecast_df, go):
    # Plot 3: Residuals vs Uncertainty
    fig3 = go.Figure()
    _test_range = list(range(len(y_test)))
    residuals = y_test - yhat

    fig3.add_trace(
        go.Bar(
            x=_test_range,
            y=residuals,
            name="Residuals",
            opacity=0.6,
            marker_color="steelblue",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=_test_range,
            y=forecast_df["uncertainty_width"],
            mode="lines",
            name="Uncertainty width",
            line=dict(color="red", width=2),
            yaxis="y2",
        )
    )
    fig3.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    fig3.update_layout(
        title="Residuals vs Forecast Uncertainty",
        xaxis_title="Test Period",
        yaxis_title="Residual (kWh)",
        yaxis2=dict(title="Uncertainty Width", overlaying="y", side="right"),
        height=500,
        width=1200,
    )
    fig3

    return fig3


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## Summary

Complete SARIMA forecasting pipeline demonstrated:

1. **Load & Explore** — 14-day metering data (30-min intervals)
2. **Train/Test Split** — 10 days train, 4 days test
3. **Model Fitting** — SARIMA(1,1,1)×(1,1,1,48)
4. **Forecasting** — Point forecasts + 80% prediction intervals
5. **Evaluation** — MAE, RMSE, MAPE, PI coverage
6. **Uncertainty Analysis** — When is the model uncertain?
7. **Probability Events** — P(forecast > threshold)
8. **Visualizations** — Time series, accuracy, residuals

### Key Takeaways

- SARIMA effectively captures daily seasonality
- Prediction intervals quantify forecast uncertainty
- Standard output contract enables downstream analytics
- Multiple evaluation metrics reveal different insights

### Next Steps

- Test alternative SARIMA parameters (p, d, q)
- Compare against other models (Exponential Smoothing, XGBoost)
- Aggregate forecasts across multiple assets (portfolio level)
- Derive flexibility forecasts from the distribution
- Monitor for concept drift over time
""")
    return


if __name__ == "__main__":
    app.run()
