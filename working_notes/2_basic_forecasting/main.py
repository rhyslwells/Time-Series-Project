import polars as pl
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: LOAD AND EXPLORE DATA
# ============================================================================

# Load metering data
df = pl.read_parquet("../../src/data/metering_data.parquet")
print("Data shape:", df.shape)
print("\nColumns:", df.columns)
print("\nUnique assets:", df.select("asset_id").n_unique())
print("\nAsset types:", df.select("asset_type").unique())
print("\nDate range:")

asset_id = "ASSET_001"
# filter data for the selected asset
asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")
asset_data


asset_info = pl.DataFrame({
    "Property": ["Asset ID", "Asset Type", "Total Records", "Date Range"],
    "Value": [asset_id, asset_data['asset_type'][0], str(asset_data.shape[0]),
              f"{asset_data['timestamp'].min()} to {asset_data['timestamp'].max()}"]
})
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

split_info = pl.DataFrame({
    "Split": ["Train", "Test"],
    "Observations": [len(y_train), len(y_test)],
    "Days": [f"{len(y_train)/48:.1f}", f"{len(y_test)/48:.1f}"]
})
print("\nTrain/Test Split:")
print(split_info)

# ============================================================================
# SECTION 3: BASELINE FORECAST MODEL (SARIMA)
# ============================================================================

# Fit SARIMA model
# Order: (p, d, q) = (1, 1, 1) - basic differencing
# Seasonal: (P, D, Q, s) = (1, 1, 1, 48) - 48 half-hourly periods per day

# There wont be any seasonality here as the data is only for 14 days
print("\nFitting SARIMA model...")

model_config = pl.DataFrame({
    "Parameter": ["Order (p,d,q)", "Seasonal Order (P,D,Q,s)"],
    "Value": ["(1, 1, 1)", "(1, 1, 1, 48)"]
})
print(model_config)

try:
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 48),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False, maxiter=1000)
    model_status = "Success"
    aic_value = f"{results.aic:.2f}"
except Exception as e:
    print(f"Error fitting model: {e}")
    print("Falling back to simpler model...")
    model = SARIMAX(
        y_train,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 0, 48)
    )
    results = model.fit(disp=False)
    model_status = "Simplified"
    aic_value = f"{results.aic:.2f}"

model_result = pl.DataFrame({
    "Metric": ["Status", "AIC"],
    "Value": [model_status, aic_value]
})
print(model_result)

# ============================================================================
# SECTION 4: GENERATE FORECASTS WITH UNCERTAINTY INTERVALS
# ============================================================================

# Forecast on test set
forecast_steps = len(y_test)
forecast = results.get_forecast(steps=forecast_steps)

# Extract point forecast and prediction intervals
yhat = np.asarray(forecast.predicted_mean)
confidence_level = 0.80  # 80% prediction interval (P10 and P90)
forecast_ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))
lower = forecast_ci[:, 0]
upper = forecast_ci[:, 1]

forecast_config = pl.DataFrame({
    "Metric": ["Forecast Steps", "Prediction Interval"],
    "Value": [str(forecast_steps), f"{confidence_level*100:.0f}% confidence"]
})
print("\nForecast Configuration:")
print(forecast_config)

# ============================================================================
# SECTION 5: STANDARD FORECAST OUTPUT CONTRACT
# ============================================================================

# Create forecast dataframe in standard format
forecast_df = pl.DataFrame({
    "asset_id": [asset_id] * len(yhat),
    "timestamp": test_timestamps,
    "prediction": yhat,
    "lower": lower,
    "upper": upper,
    "model_version": ["sarima_v1"] * len(yhat),
    "forecast_made_at": [datetime.now()] * len(yhat),
    "uncertainty_width": upper - lower,
    "actual": y_test
})

print("\nForecast Output (first 10 rows):")
print(forecast_df.head(10).select([
    "timestamp", "prediction", "lower", "upper", "actual", "uncertainty_width"
]))

# Quick visualization: Forecast vs Actual
fig, ax = plt.subplots(figsize=(14, 6))
test_range = range(len(forecast_df))
ax.plot(test_range, forecast_df["actual"], 'o-', label='Actual', linewidth=2, markersize=4, color='#1f77b4')
ax.plot(test_range, forecast_df["prediction"], 's-', label='Forecast', linewidth=2, markersize=4, alpha=0.7, color='#ff7f0e')
ax.fill_between(test_range, forecast_df["lower"], forecast_df["upper"], alpha=0.2, label=f'{confidence_level*100:.0f}% PI', color='#ff7f0e')
ax.set_xlabel('Test Period (half-hourly intervals)', fontsize=11)
ax.set_ylabel('Metering (kWh)', fontsize=11)
ax.set_title(f'SARIMA Forecast vs Actual - {asset_id}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


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
print(f"  PI Coverage: {coverage:.1f}% (expected ~{confidence_level*100:.0f}%)")



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

uncertainty_analysis = pl.DataFrame({
    "Metric": ["Mean Width", "Std Deviation", "Min Width", "Max Width",
               "High Uncertainty Count", "High Uncertainty Proportion"],
    "Value": [f"{uncertainty_mean:.4f} kWh", f"{uncertainty_std:.4f} kWh",
              f"{uncertainty_min:.4f} kWh", f"{uncertainty_max:.4f} kWh",
              f"{high_uncertainty_periods.shape[0]} / {len(yhat)}",
              f"{high_uncertainty_periods.shape[0]/len(yhat)*100:.1f}%"]
})
print("\nForecast Uncertainty Analysis:")
print(uncertainty_analysis)

# ============================================================================
# SECTION 8: FORECAST PROBABILITY OF EVENTS
# ============================================================================

#TODO: I will need help understanding the meaning of this.

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
    prob_data.append({
        "Threshold": f"{threshold:.2f} kWh",
        "P(Forecast > Threshold)": f"{mean_prob:.1%}",
        "Min Probability": f"{np.min(prob_exceed):.1%}",
        "Max Probability": f"{np.max(prob_exceed):.1%}"
    })

probability_events = pl.DataFrame(prob_data)
print("\nForecast Probability of Events:")
print(probability_events)

# ============================================================================
# SECTION 9: FULL VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Full time series with train/test split
ax = axes[0]
ax.plot(range(len(y_train)), y_train, 'o-', label='Training data', linewidth=1, markersize=2)
ax.plot(range(len(y_train), len(y)), y_test, 'o-', label='Test data', linewidth=1, markersize=2)
ax.axvline(x=len(y_train), color='red', linestyle='--', alpha=0.5, label='Train/Test split')
ax.set_ylabel('Metering (kWh)')
ax.set_title(f'Asset {asset_id}: Time Series with Train/Test Split')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Forecast vs Actual (test period)
ax = axes[1]
test_range = range(len(y_test))
ax.plot(test_range, y_test, 'o-', label='Actual', linewidth=1.5, markersize=3)
ax.plot(test_range, yhat, 's-', label='Forecast', linewidth=1.5, markersize=3, alpha=0.7)
ax.fill_between(test_range, lower, upper, alpha=0.3, label=f'{confidence_level*100:.0f}% PI')
ax.set_ylabel('Metering (kWh)')
ax.set_xlabel('Test Period (half-hourly intervals)')
ax.set_title(f'Forecast vs Actual (Test Period) - MAE: {mae:.4f}, RMSE: {rmse:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Residuals and uncertainty
ax = axes[2]
residuals = y_test - yhat
ax.bar(test_range, residuals, alpha=0.6, label='Residuals', width=0.8)
ax.plot(test_range, forecast_df["uncertainty_width"], 'r-', linewidth=1.5, label='Uncertainty width')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_ylabel('Residual (kWh) / Uncertainty (kWh)')
ax.set_xlabel('Test Period (half-hourly intervals)')
ax.set_title('Residuals vs Forecast Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\nVisualization saved to: working_notes/2_basic_forecasting/forecast_analysis.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("FORECASTING SUMMARY")
print("="*70)

summary_info = pl.DataFrame({
    "Property": ["Asset", "Model", "Training Period", "Test Period"],
    "Value": [asset_id, "SARIMA (1,1,1)x(1,1,1,48)", f"{len(y_train)} observations", f"{len(y_test)} observations"]
})
print(summary_info)

print("\nPerformance Metrics:")
performance_metrics = pl.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE", "PI Coverage"],
    "Value": [f"{mae:.4f} kWh", f"{rmse:.4f} kWh", f"{mape:.2f}%", f"{coverage:.1f}%"]
})
print(performance_metrics)

print("\nUncertainty Metrics:")
uncertainty_metrics = pl.DataFrame({
    "Metric": ["Mean Width", "Std Deviation", "Min Width", "Max Width", "High Uncertainty Periods"],
    "Value": [f"{uncertainty_mean:.4f} kWh", f"{uncertainty_std:.4f} kWh",
              f"{uncertainty_min:.4f} kWh", f"{uncertainty_max:.4f} kWh",
              f"{high_uncertainty_periods.shape[0]} / {len(yhat)}"]
})
print(uncertainty_metrics)
