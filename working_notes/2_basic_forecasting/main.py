import polars as pl
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 1: LOAD AND EXPLORE DATA
# ============================================================================

# Load metering data
df = pl.read_parquet("../../src/data/metering_data.parquet")

asset_id = "ASSET_001"
# filter data for the selected asset
asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")
asset_data


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
print(asset_info)

# Convert to time series format
y = asset_data.select("metering_kwh").to_numpy().flatten()
timestamps = asset_data.select("timestamp").to_numpy().flatten()

# ============================================================================
# SECTION 2: TRAIN/TEST SPLIT
# ============================================================================

# Reserve last 4 days (14 * 48 half-hourly periods = 672 points per week)
# For 14 days of data, hold out last 4 days
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
print("\nTrain/Test Split:")
print(split_info)

# ============================================================================
# SECTION 3: SELECT AND FIT FORECAST MODEL
# ============================================================================

# Choose model: "SARIMA" or "ExponentialSmoothing"
# MODEL_TYPE = "SARIMA"
MODEL_TYPE = "ExponentialSmoothing"  # Uncomment to use Exponential Smoothing

# ============================================================================
# SECTION 3.1: SARIMA MODEL
# ============================================================================

if MODEL_TYPE == "SARIMA":
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

# ============================================================================
# SECTION 3.2: EXPONENTIAL SMOOTHING MODEL (HOLT-WINTERS)
# ============================================================================

elif MODEL_TYPE == "ExponentialSmoothing":
    print("\nFitting Exponential Smoothing (Holt-Winters) model...")

    model_config = pl.DataFrame(
        {
            "Parameter": ["Trend", "Seasonal", "Seasonal Period"],
            "Value": ["Additive", "Additive", "48 (daily)"],
        }
    )
    print(model_config)

    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=48,
        )
        results = model.fit(optimized=True)
        model_status = "Success"
        aic_value = f"{results.aic:.2f}"
    except Exception as e:
        print(f"Error fitting Exponential Smoothing: {e}")
        print("Falling back to simpler ES model...")
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal=None,
        )
        results = model.fit(optimized=True)
        model_status = "Simplified"
        aic_value = f"{results.aic:.2f}"

    model_result = pl.DataFrame(
        {"Metric": ["Status", "AIC"], "Value": [model_status, aic_value]}
    )
    print(model_result)

# ============================================================================
# SECTION 4: GENERATE FORECASTS WITH UNCERTAINTY INTERVALS
# ============================================================================

forecast_steps = len(y_test)
confidence_level = 0.80  # 80% prediction interval (P10 and P90)

if MODEL_TYPE == "SARIMA":
    forecast = results.get_forecast(steps=forecast_steps)
    yhat = np.asarray(forecast.predicted_mean)
    forecast_ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))
    lower = forecast_ci[:, 0]
    upper = forecast_ci[:, 1]

elif MODEL_TYPE == "ExponentialSmoothing":
    yhat = results.forecast(steps=forecast_steps)

    # Estimate uncertainty from training residuals
    residuals = y_train - results.fittedvalues
    residual_std = np.std(residuals)
    z_score = 1.282  # 80% confidence interval
    lower = yhat - z_score * residual_std
    upper = yhat + z_score * residual_std

forecast_config = pl.DataFrame(
    {
        "Metric": ["Forecast Steps", "Prediction Interval"],
        "Value": [str(forecast_steps), f"{confidence_level * 100:.0f}% confidence"],
    }
)
print("\nForecast Configuration:")
print(forecast_config)

# ============================================================================
# SECTION 5: STANDARD FORECAST OUTPUT CONTRACT
# ============================================================================

# Create forecast dataframe in standard format
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

# Quick visualization: Forecast vs Actual
fig = go.Figure()
test_range = list(range(len(forecast_df)))

fig.add_trace(
    go.Scatter(
        x=test_range,
        y=forecast_df["actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    )
)

fig.add_trace(
    go.Scatter(
        x=test_range,
        y=forecast_df["prediction"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#ff7f0e", width=2),
        marker=dict(size=4),
        opacity=0.7,
    )
)

fig.add_trace(
    go.Scatter(
        x=test_range + test_range[::-1],
        y=forecast_df["upper"].to_list() + forecast_df["lower"].to_list()[::-1],
        fill="toself",
        fillcolor="rgba(255, 127, 14, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{confidence_level * 100:.0f}% PI",
        hoverinfo="skip",
    )
)

fig.update_layout(
    title=f"SARIMA Forecast vs Actual - {asset_id}",
    xaxis_title="Test Period (half-hourly intervals)",
    yaxis_title="Metering (kWh)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    width=1400,
)
fig.show()


# ============================================================================
# SECTION 6: MODEL EVALUATION
# ============================================================================

# Calculate metrics
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


# ============================================================================
# SECTION 7: FORECAST UNCERTAINTY ANALYSIS
# ============================================================================

# Uncertainty quantiles
uncertainty_mean = forecast_df["uncertainty_width"].mean()
uncertainty_std = forecast_df["uncertainty_width"].std()
uncertainty_min = forecast_df["uncertainty_width"].min()
uncertainty_max = forecast_df["uncertainty_width"].max()

# Periods with high uncertainty
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
            f"{high_uncertainty_periods.shape[0]} / {len(yhat)}",
            f"{high_uncertainty_periods.shape[0] / len(yhat) * 100:.1f}%",
        ],
    }
)
print("\nForecast Uncertainty Analysis:")
print(uncertainty_analysis)

# ============================================================================
# SECTION 8: FORECAST PROBABILITY OF EVENTS
# ============================================================================

# TODO: I will need help understanding the meaning of this.

# Define event thresholds
thresholds = [
    np.percentile(y_train, 25),
    np.percentile(y_train, 50),
    np.percentile(y_train, 75),
]


# Calculate probability of exceeding each threshold
from scipy.stats import norm

prob_data = []
for threshold in thresholds:
    forecast_std = (upper - lower) / (2 * 1.645)
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

# ============================================================================
# SECTION 9: FULL VISUALIZATION
# ============================================================================

# Plot 1: Full time series with train/test split
fig1 = go.Figure()
train_range = list(range(len(y_train)))
full_range = list(range(len(y_train), len(y)))

fig1.add_trace(
    go.Scatter(
        x=train_range,
        y=y_train,
        mode="lines+markers",
        name="Training data",
        line=dict(width=1),
        marker=dict(size=2),
    )
)

fig1.add_trace(
    go.Scatter(
        x=full_range,
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
fig1.show()

# Plot 2: Forecast vs Actual (test period)

# For a plot between 1 and 2 you could have plot 1 but with the 80% PI on the forecast section
# like 2 without the Actual.

# Looking at 2 some questions which will arise are to evaluate the actual versus the 80% PI interval,
# Like how many actual points fall outside the 80% PI interval.

fig2 = go.Figure()
test_range = list(range(len(y_test)))

fig2.add_trace(
    go.Scatter(
        x=test_range,
        y=y_test,
        mode="lines+markers",
        name="Actual",
        line=dict(width=1.5),
        marker=dict(size=3),
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_range,
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
        x=test_range + test_range[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{confidence_level * 100:.0f}% PI",
        hoverinfo="skip",
    )
)

fig2.update_layout(
    title=f"Forecast vs Actual (Test Period) - MAE: {mae:.4f}, RMSE: {rmse:.4f}",
    xaxis_title="Test Period (half-hourly intervals)",
    yaxis_title="Metering (kWh)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    width=1400,
)
fig2.show()

# Plot 3: Residuals and uncertainty

# An explanation will be necessary here of what residuals and uncertainty width
# are and how they relate to the forecast. The residuals are the difference between the actual values and the forecasted values, while the uncertainty width represents the range of possible values predicted by the model. This plot helps visualize how well the model is performing and where it may be uncertain.

# Are the following correct:
# What weould be expected to see is that the residuals should be randomly distributed around zero, indicating that the model is unbiased. The uncertainty width should ideally capture most of the residuals, meaning that the prediction intervals are accurate. If many residuals fall outside the uncertainty width, it suggests that the model's predictions are not reliable.
# What if the uncertaintity width is close to a straight line? This would indicate that the model is consistently uncertain about its predictions, which could be due to a lack of variability in the data or an inadequate model. It may also suggest that the model is not capturing important patterns in the data, leading to a lack of confidence in its forecasts.


fig3 = go.Figure()
residuals = y_test - yhat

fig3.add_trace(
    go.Bar(
        x=test_range,
        y=residuals,
        name="Residuals",
        opacity=0.6,
        marker_color="rgba(31, 119, 180, 0.6)",
    )
)

fig3.add_trace(
    go.Scatter(
        x=test_range,
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
fig3.show()


# ============================================================================
# SUMMARY
# ============================================================================


print("FORECASTING SUMMARY")


summary_info = pl.DataFrame(
    {
        "Property": ["Asset", "Model", "Training Period", "Test Period"],
        "Value": [
            asset_id,
            "SARIMA (1,1,1)x(1,1,1,48)",
            f"{len(y_train)} observations",
            f"{len(y_test)} observations",
        ],
    }
)
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

print("\nUncertainty Metrics:")
uncertainty_metrics = pl.DataFrame(
    {
        "Metric": [
            "Mean Width",
            "Std Deviation",
            "Min Width",
            "Max Width",
            "High Uncertainty Periods",
        ],
        "Value": [
            f"{uncertainty_mean:.4f} kWh",
            f"{uncertainty_std:.4f} kWh",
            f"{uncertainty_min:.4f} kWh",
            f"{uncertainty_max:.4f} kWh",
            f"{high_uncertainty_periods.shape[0]} / {len(yhat)}",
        ],
    }
)
print(uncertainty_metrics)
