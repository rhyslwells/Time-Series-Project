import marimo as mo
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import warnings

# marimo edit working_notes\2_basic_forecasting\sarima_marimo.py


warnings.filterwarnings("ignore")

app = mo.App()

# ============================================================================
# INTRODUCTION: Context from main_idea.md
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    # SARIMA Time Series Forecasting for Energy Assets

    This notebook builds a **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** forecast model
    for energy metering data.

    ## Key Concept: Forecast as a Data Product

    The forecast itself is not just "the next 5 days of meter values". Instead, we produce a **forecast distribution**:

    - $E[Y_t]$: Expected metering value
    - $P_{10}(Y_t)$, $P_{50}(Y_t)$, $P_{90}(Y_t)$: Prediction intervals

    This gives information about both **expected behaviour** and **uncertainty**.

    ## Architecture

    The system flows as:

    ```
    Metering Data
        ↓
    SARIMA Model (trained on history)
        ↓
    Forecast Distribution (point + intervals)
        ↓
    Derived Metrics (uncertainty, probabilities, events)
        ↓
    Decision Intelligence
    ```

    ## Sections in This Notebook

    1. **Load & Explore Data** — understand the asset and its history
    2. **Train/Test Split** — reserve test data for validation
    3. **Fit SARIMA Model** — train with (p,d,q)×(P,D,Q,s) parameters
    4. **Generate Forecasts** — produce point forecasts and prediction intervals
    5. **Standard Output Contract** — format as `asset_id`, `timestamp`, `prediction`, `lower`, `upper`
    6. **Model Evaluation** — assess accuracy (MAE, RMSE, MAPE, coverage)
    7. **Uncertainty Analysis** — understand when the model is uncertain
    8. **Probability of Events** — calculate P(forecast > threshold)
    9. **Visualizations** — time series, forecast vs actual, residuals
    10. **Summary** — final metrics and insights
    """)


# ============================================================================
# SECTION 1: LOAD AND EXPLORE DATA
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 1: Load and Explore Data

    We start by loading the raw metering data for a single asset and understanding its structure
    and characteristics.
    """)


@app.cell
def _():
    # Load metering data
    df = pl.read_parquet("../../src/data/metering_data.parquet")
    print("Data shape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nUnique assets:", df.select("asset_id").n_unique())
    print("\nAsset types:", df.select("asset_type").unique())

    # Select asset for analysis
    asset_id = "ASSET_001"
    asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    asset_info = pl.DataFrame(
        {
            "Property": ["Asset ID", "Asset Type", "Total Records", "Date Range"],
            "Value": [
                asset_id,
                asset_data["asset_type"][0],
                str(asset_data.shape[0]),
                f"{asset_data['timestamp'].min()} to {asset_data['timestamp'].max()}",
            ],
        }
    )
    print("\nAsset Summary:")
    print(asset_info)

    # Convert to time series format
    y = asset_data.select("metering_kwh").to_numpy().flatten()
    timestamps = asset_data.select("timestamp").to_numpy().flatten()

    return df, asset_id, asset_data, y, timestamps


# ============================================================================
# SECTION 2: TRAIN/TEST SPLIT
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 2: Train/Test Split

    We split the data into training (historical) and test (validation) sets.

    - **Training data**: Used to fit the SARIMA model and learn patterns
    - **Test data**: Held-out to evaluate forecast accuracy

    For 14 days of 30-minute data:
    - 1 day = 48 observations
    - 4 days = 192 observations (reserved for testing)
    """)


@app.cell
def _(y, timestamps):
    # Reserve last 4 days (4 * 48 half-hourly periods = 192 points)
    test_split_idx = len(y) - (4 * 48)

    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]

    train_timestamps = timestamps[:test_split_idx]
    test_timestamps = timestamps[test_split_idx:]

    split_info = pl.DataFrame(
        {
            "Split": ["Train", "Test"],
            "Observations": [len(y_train), len(y_test)],
            "Days": [f"{len(y_train) / 48:.1f}", f"{len(y_test) / 48:.1f}"],
        }
    )
    print("Train/Test Split:")
    print(split_info)

    return y_train, y_test, train_timestamps, test_timestamps, test_split_idx


# ============================================================================
# SECTION 3: FIT SARIMA MODEL
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 3: Fit SARIMA Model

    **SARIMA(p,d,q)×(P,D,Q,s)** is a time series model that captures:

    - **(p, d, q)**: Non-seasonal autoregressive, integrated, moving average components
    - **(P, D, Q, s)**: Seasonal versions (s = seasonal period)

    For 30-minute metering with daily seasonality (s=48):
    - Model: **SARIMA(1,1,1)×(1,1,1,48)**
    - **p=1**: Autoregressive order (past values influence current)
    - **d=1**: First differencing (make series stationary)
    - **q=1**: Moving average order (past forecast errors influence current)
    - **P=1, D=1, Q=1**: Seasonal analogs at lag 48
    - **s=48**: Seasonal period (48 half-hourly intervals = 1 day)

    The model learns to:
    1. Capture daily patterns (solar generation peaks at noon, consumption patterns)
    2. Handle trends and level shifts
    3. Produce prediction intervals (uncertainty)
    """)


@app.cell
def _(y_train):
    print("\nFitting SARIMA model...")
    model_config = pl.DataFrame(
        {
            "Parameter": ["Order (p,d,q)", "Seasonal Order (P,D,Q,s)"],
            "Value": ["(1, 1, 1)", "(1, 1, 1, 48)"],
        }
    )
    print(model_config)

    try:
        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 48),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=1000)
        model_status = "Success"
        aic_value = f"{results.aic:.2f}"
    except Exception as e:
        print(f"Error fitting model: {e}")
        print("Falling back to simpler model...")
        model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 48))
        results = model.fit(disp=False)
        model_status = "Simplified"
        aic_value = f"{results.aic:.2f}"

    model_result = pl.DataFrame(
        {"Metric": ["Status", "AIC"], "Value": [model_status, aic_value]}
    )
    print(model_result)
    print(f"\nAIC (Akaike Information Criterion): {aic_value}")
    print("Lower AIC indicates a better fit (penalizes model complexity)")

    return model, results, model_status, aic_value


# ============================================================================
# SECTION 4: GENERATE FORECASTS WITH UNCERTAINTY INTERVALS
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 4: Generate Forecasts with Uncertainty Intervals

    The SARIMA model produces:

    1. **Point forecast** ($\\hat{y}_t$): The expected value
    2. **Prediction intervals**: Lower and upper bounds around the point forecast

    We generate an **80% prediction interval** (P10 and P90 quantiles).

    This means:
    - We expect the actual value to fall within [lower, upper] ~80% of the time
    - A wide interval indicates high uncertainty; a narrow interval indicates high confidence
    """)


@app.cell
def _(results, y_test):
    forecast_steps = len(y_test)
    confidence_level = 0.80  # 80% prediction interval

    forecast = results.get_forecast(steps=forecast_steps)
    yhat = np.asarray(forecast.predicted_mean)
    forecast_ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))
    lower = forecast_ci[:, 0]
    upper = forecast_ci[:, 1]

    forecast_config = pl.DataFrame(
        {
            "Metric": ["Forecast Steps", "Prediction Interval"],
            "Value": [str(forecast_steps), f"{confidence_level * 100:.0f}% confidence"],
        }
    )
    print("Forecast Configuration:")
    print(forecast_config)

    return forecast_steps, confidence_level, yhat, lower, upper


# ============================================================================
# SECTION 5: STANDARD FORECAST OUTPUT CONTRACT
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 5: Standard Forecast Output Contract

    All forecasts follow a **standard format** for consistency and reusability:

    | Column | Type | Description |
    |--------|------|-------------|
    | `asset_id` | str | Unique identifier for the asset |
    | `timestamp` | datetime | When the forecast applies |
    | `prediction` | float | Point forecast (expected value) |
    | `lower` | float | Lower bound of prediction interval (P10) |
    | `upper` | float | Upper bound of prediction interval (P90) |
    | `uncertainty_width` | float | `upper - lower` (width of interval) |
    | `model_version` | str | Model identifier (e.g., "sarima_v1") |
    | `forecast_made_at` | datetime | When the forecast was generated |
    | `actual` | float | Actual metering (known after forecast period) |

    This contract makes forecasts reusable across the system without rebuilding downstream analytics.
    """)


@app.cell
def _(asset_id, yhat, lower, upper, y_test, test_timestamps):
    forecast_df = pl.DataFrame(
        {
            "asset_id": [asset_id] * len(yhat),
            "timestamp": test_timestamps,
            "prediction": yhat,
            "lower": lower,
            "upper": upper,
            "model_version": ["sarima_v1"] * len(yhat),
            "forecast_made_at": [datetime.now()] * len(yhat),
            "uncertainty_width": upper - lower,
            "actual": y_test,
        }
    )

    print("\nForecast Output (first 10 rows):")
    print(
        forecast_df.head(10).select(
            ["timestamp", "prediction", "lower", "upper", "actual", "uncertainty_width"]
        )
    )

    return forecast_df


# ============================================================================
# SECTION 6: MODEL EVALUATION
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 6: Model Evaluation

    We assess the forecast quality using multiple metrics:

    - **MAE** (Mean Absolute Error): Average absolute difference between forecast and actual
    - **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
    - **MAPE** (Mean Absolute Percentage Error): Error as a percentage of actual values
    - **PI Coverage**: What fraction of actuals fell within the prediction interval?
    - **Bias**: Mean forecast error (positive = over-forecast, negative = under-forecast)

    A good model should have:
    - Low MAE, RMSE, MAPE
    - PI Coverage close to the nominal level (e.g., ~80% for an 80% interval)
    - Bias close to zero
    """)


@app.cell
def _(y_test, yhat, lower, upper, confidence_level):
    mae = mean_absolute_error(y_test, yhat)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    mape = np.mean(np.abs((y_test - yhat) / (np.abs(y_test) + 1e-8))) * 100

    # Prediction interval coverage
    coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100

    # Bias
    bias = np.mean(yhat - y_test)

    print(f"\nModel Performance on Test Set:")
    print(f"  MAE:      {mae:.4f} kWh")
    print(f"  RMSE:     {rmse:.4f} kWh")
    print(f"  MAPE:     {mape:.2f}%")
    print(f"  Bias:     {bias:.4f} kWh")
    print(f"  PI Coverage: {coverage:.1f}% (expected ~{confidence_level * 100:.0f}%)")

    return mae, rmse, mape, coverage, bias


# ============================================================================
# SECTION 7: FORECAST UNCERTAINTY ANALYSIS
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 7: Forecast Uncertainty Analysis

    We analyze the **prediction interval width** to understand when the model is confident vs uncertain.

    - **Mean Width**: Average size of prediction intervals
    - **High Uncertainty Periods**: Times when the model is least confident

    High uncertainty can indicate:
    - Unusual patterns in the data
    - Lack of historical precedent
    - Genuine variability in the asset's behavior
    - System constraints or operating changes
    """)


@app.cell
def _(forecast_df):
    uncertainty_mean = forecast_df["uncertainty_width"].mean()
    uncertainty_std = forecast_df["uncertainty_width"].std()
    uncertainty_min = forecast_df["uncertainty_width"].min()
    uncertainty_max = forecast_df["uncertainty_width"].max()

    # Periods with high uncertainty (mean + 2*std)
    high_uncertainty_threshold = uncertainty_mean + 2 * uncertainty_std
    high_uncertainty_periods = forecast_df.filter(
        pl.col("uncertainty_width") > high_uncertainty_threshold
    )

    uncertainty_analysis = pl.DataFrame(
        {
            "Metric": [
                "Mean Width",
                "Std Deviation",
                "Min Width",
                "Max Width",
                "High Uncertainty Count",
                "High Uncertainty Proportion",
            ],
            "Value": [
                f"{uncertainty_mean:.4f} kWh",
                f"{uncertainty_std:.4f} kWh",
                f"{uncertainty_min:.4f} kWh",
                f"{uncertainty_max:.4f} kWh",
                f"{high_uncertainty_periods.shape[0]} / {len(forecast_df)}",
                f"{high_uncertainty_periods.shape[0] / len(forecast_df) * 100:.1f}%",
            ],
        }
    )
    print("\nForecast Uncertainty Analysis:")
    print(uncertainty_analysis)

    return (
        uncertainty_mean,
        uncertainty_std,
        uncertainty_min,
        uncertainty_max,
        high_uncertainty_threshold,
        high_uncertainty_periods,
    )


# ============================================================================
# SECTION 8: FORECAST PROBABILITY OF EVENTS
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 8: Forecast Probability of Events

    Once you have a forecast distribution, you can ask probabilistic questions:

    > "What is the probability that metering exceeds a threshold?"

    $$P(Y_t > \\text{threshold})$$

    This is more useful than a point forecast for operational decisions.

    **Example interpretation:**
    - If P(Y > 5 kW) = 0.75, the asset is **likely** to exceed 5 kW
    - If P(Y > 5 kW) = 0.25, the asset is **unlikely** to exceed 5 kW

    We calculate probabilities for the 25th, 50th, and 75th percentiles of training data.
    """)


@app.cell
def _(y_train, yhat, upper, lower):
    # Define thresholds at quartiles of training data
    thresholds = [
        np.percentile(y_train, 25),
        np.percentile(y_train, 50),
        np.percentile(y_train, 75),
    ]

    prob_data = []
    for threshold in thresholds:
        # Estimate forecast standard deviation from interval width
        forecast_std = (upper - lower) / (2 * 1.645)  # 1.645 = 90% confidence z-score
        prob_exceed = 1 - norm.cdf(threshold, loc=yhat, scale=forecast_std)
        mean_prob = np.mean(prob_exceed)
        prob_data.append(
            {
                "Threshold": f"{threshold:.2f} kWh",
                "P(Forecast > Threshold)": f"{mean_prob:.1%}",
                "Min Probability": f"{np.min(prob_exceed):.1%}",
                "Max Probability": f"{np.max(prob_exceed):.1%}",
            }
        )

    probability_events = pl.DataFrame(prob_data)
    print("\nForecast Probability of Events:")
    print(probability_events)

    return thresholds, forecast_std, prob_exceed, probability_events


# ============================================================================
# SECTION 9: VISUALIZATIONS
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 9: Visualizations

    Three key plots help interpret the forecast:

    1. **Full Time Series with Train/Test Split** — Shows how much data was used vs held-out
    2. **Forecast vs Actual (Test Period)** — How well does the forecast match reality?
    3. **Residuals vs Uncertainty Width** — Are prediction intervals capturing forecast error?
    """)


@app.cell
def _(y_train, y_test, yhat, lower, upper, asset_id, mae, rmse, confidence_level):
    # Plot 1: Full time series with train/test split
    fig1 = go.Figure()
    _train_range = list(range(len(y_train)))
    _full_range = list(range(len(y_train), len(y_train) + len(y_test)))

    fig1.add_trace(
        go.Scatter(
            x=_train_range,
            y=y_train,
            mode="lines+markers",
            name="Training data",
            line=dict(width=1),
            marker=dict(size=2),
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=_full_range,
            y=y_test,
            mode="lines+markers",
            name="Test data",
            line=dict(width=1),
            marker=dict(size=2),
        )
    )

    fig1.add_vline(
        x=len(y_train),
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="Train/Test split",
        annotation_position="top",
    )

    fig1.update_layout(
        title=f"Asset {asset_id}: Time Series with Train/Test Split",
        xaxis_title="Time Period (half-hourly intervals)",
        yaxis_title="Metering (kWh)",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1400,
    )

    print("Plot 1: Full time series")
    fig1.show()

    return fig1


@app.cell
def _(y_test, yhat, lower, upper, asset_id, mae, rmse, confidence_level):
    # Plot 2: Forecast vs Actual
    fig2 = go.Figure()
    _test_range_2 = list(range(len(y_test)))

    fig2.add_trace(
        go.Scatter(
            x=_test_range_2,
            y=y_test,
            mode="lines+markers",
            name="Actual",
            line=dict(width=1.5),
            marker=dict(size=3),
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=_test_range_2,
            y=yhat,
            mode="lines+markers",
            name="Forecast",
            line=dict(width=1.5),
            marker=dict(size=3),
            opacity=0.7,
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=_test_range_2 + _test_range_2[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{confidence_level * 100:.0f}% PI",
            hoverinfo="skip",
        )
    )

    fig2.update_layout(
        title=f"Forecast vs Actual (Test Period) — MAE: {mae:.4f}, RMSE: {rmse:.4f}",
        xaxis_title="Test Period (half-hourly intervals)",
        yaxis_title="Metering (kWh)",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1400,
    )

    print("Plot 2: Forecast vs Actual")
    fig2.show()

    return fig2


@app.cell
def _(y_test, yhat, forecast_df):
    # Plot 3: Residuals and Uncertainty Width
    fig3 = go.Figure()
    _test_range_3 = list(range(len(y_test)))
    residuals = y_test - yhat

    fig3.add_trace(
        go.Bar(
            x=_test_range_3,
            y=residuals,
            name="Residuals",
            opacity=0.6,
            marker_color="rgba(31, 119, 180, 0.6)",
        )
    )

    fig3.add_trace(
        go.Scatter(
            x=_test_range_3,
            y=forecast_df["uncertainty_width"],
            mode="lines",
            name="Uncertainty width",
            line=dict(color="red", width=1.5),
            yaxis="y2",
        )
    )

    fig3.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

    fig3.update_layout(
        title="Residuals vs Forecast Uncertainty",
        xaxis_title="Test Period (half-hourly intervals)",
        yaxis_title="Residual (kWh)",
        yaxis2=dict(
            title="Uncertainty Width (kWh)",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1400,
        legend=dict(x=0.01, y=0.99),
    )

    print("Plot 3: Residuals vs Uncertainty")
    fig3.show()

    return fig3, residuals


# ============================================================================
# SECTION 10: SUMMARY
# ============================================================================


@app.cell(hide_code=True)
def _():
    return mo.md("""
    ## Section 10: Summary

    This notebook demonstrated a complete SARIMA forecasting pipeline for energy metering.

    ### Key Takeaways

    1. **SARIMA captures seasonality** — Essential for 30-minute metering with daily patterns
    2. **Prediction intervals matter** — Uncertainty is as important as the point forecast
    3. **Standard output contract** — Enables downstream analytics and decision systems
    4. **Multiple evaluation metrics** — MAE, RMSE, MAPE, coverage all tell different stories
    5. **Forecast = data product** — Not just numbers, but information about uncertainty and events

    ### Next Steps

    You could extend this to:
    - Test other model configurations (e.g., different p, d, q values)
    - Compare SARIMA against other models (Exponential Smoothing, XGBoost, etc.)
    - Aggregate forecasts across multiple assets (portfolio level)
    - Derive flexibility forecasts from the forecast distribution
    - Monitor model drift over time
    """)


@app.cell
def _(asset_id, y_train, y_test, mae, rmse, mape, coverage, uncertainty_mean):
    summary_info = pl.DataFrame(
        {
            "Property": ["Asset", "Model", "Training Period", "Test Period"],
            "Value": [
                asset_id,
                "SARIMA (1,1,1)×(1,1,1,48)",
                f"{len(y_train)} observations",
                f"{len(y_test)} observations",
            ],
        }
    )
    print("\n" + "=" * 70)
    print("FORECASTING SUMMARY")
    print("=" * 70)
    print(summary_info)

    print("\nPerformance Metrics:")
    performance_metrics = pl.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "MAPE", "PI Coverage"],
            "Value": [
                f"{mae:.4f} kWh",
                f"{rmse:.4f} kWh",
                f"{mape:.2f}%",
                f"{coverage:.1f}%",
            ],
        }
    )
    print(performance_metrics)

    print("\nUncertainty:")
    uncertainty_summary = pl.DataFrame(
        {
            "Metric": ["Mean PI Width"],
            "Value": [f"{uncertainty_mean:.4f} kWh"],
        }
    )
    print(uncertainty_summary)
    print("=" * 70)


if __name__ == "__main__":
    app.run()
