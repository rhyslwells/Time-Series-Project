"""
Random Forest Forecasting for Future Time Series (5 Days Ahead, No Leakage)

This script demonstrates a workflow for using machine learning to forecast 
time series data in a way that respects temporal order and avoids data leakage.

Key Objectives:
1. Generate synthetic hourly usage data:
   - Mimics real-world scenarios like telecom usage or IoT sensor readings.
   - Includes daily and weekly seasonal patterns plus random noise.

2. Feature engineering for machine learning:
   - Lag features: past observations at various offsets (e.g., 1, 2, 3, 24 hours).
     These allow the model to learn short-term and daily dependencies.
   - Rolling mean features: averages over recent windows (e.g., last 3, 6, 12 hours)
     to capture local trends.
   - Time-based features: hour of day and day of week to account for seasonality.

3. Train a Random Forest model:
   - A tree-based ensemble method that can learn non-linear relationships
     from the features.
   - Trained only on historical data to prevent future information leakage.

4. Iterative forecasting for future time points:
   - Forecast horizon is 5 days (120 hours) beyond the last available data.
   - Each prediction uses only previously observed or predicted values
     to generate the next step's features.
   - This iterative approach ensures **no data leakage** from the future.

5. Inspection and understanding:
   - Print the tail of the feature matrix with lag and rolling values to
     understand what the model sees during training.
   - Print the first few forecasted values to track the iterative process.

6. Visualization:
   - Compare historical usage with the forecasted values.
   - Clearly shows how the Random Forest model predicts future behavior
     based on learned patterns from historical data.

Purpose:
- To illustrate how feature-based ML methods like Random Forest can be
  applied to time series forecasting.
- To demonstrate best practices for preventing leakage while predicting
  future points beyond the observed dataset.
- To serve as a template for extending to real datasets and other
  global ML models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------- SYNTHETIC DATA -----------------
np.random.seed(42)
date_range = pd.date_range(start="2025-01-01", periods=24*30, freq="H")  # 30 days
daily_pattern = np.sin(2 * np.pi * date_range.hour / 24)
weekly_pattern = np.sin(2 * np.pi * date_range.dayofweek / 7)
noise = np.random.normal(0, 0.2, len(date_range))
usage = 5 + 2*daily_pattern + 1.5*weekly_pattern + noise

df = pd.DataFrame({"Time": date_range, "Usage": usage}).set_index("Time")
print("Synthetic data ready. Last 5 rows:")
print(df.tail())

# ----------------- FEATURE ENGINEERING FOR TRAINING -----------------
def build_feature_matrix(series, lags=[1,2,3,24], rolling_windows=[3,6,12]):
    X, y = [], []
    for i in range(max(max(lags), max(rolling_windows)), len(series)):
        row = []
        # lag features
        for lag in lags:
            row.append(float(series.iloc[i-lag]))
        # rolling mean features
        for window in rolling_windows:
            row.append(float(series.iloc[i-window:i].mean()))
        # time features
        timestamp = series.index[i]
        row.extend([timestamp.hour, timestamp.dayofweek])
        X.append(row)
        y.append(float(series.iloc[i]))
    return pd.DataFrame(X), pd.Series(y)


X_train, y_train = build_feature_matrix(df)
print(f"\nFeature matrix for training created. Shape: {X_train.shape}")

# ----------------- TRAIN RANDOM FOREST -----------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ----------------- PRINT TAIL WITH LAG VALUES -----------------

# Add target to the feature DataFrame for inspection
X_with_target = X_train.copy()
X_with_target['Target'] = y_train.values

print("\nTail of feature matrix with lag and rolling values:")
print(X_with_target.tail())

# ----------------- ITERATIVE FORECAST INTO FUTURE -----------------
forecast_horizon = 24*5  # 5 days
lags = [1,2,3,24]
rolling_windows = [3,6,12]

pred_series = df['Usage'].copy()  # start with historical data
forecast_values = []

print("\nStarting 5-day future forecast...")
for step in range(forecast_horizon):
    current_time = pred_series.index[-1] + pd.Timedelta(hours=1)
    
    # Build features for next step
    X_next = []
    for lag in lags:
        X_next.append(pred_series.iloc[-lag])
    for window in rolling_windows:
        X_next.append(pred_series.iloc[-window:].mean())
    # time features
    X_next.extend([current_time.hour, current_time.dayofweek])
    
    # Predict next value
    y_pred = rf.predict([X_next])[0]
    forecast_values.append(y_pred)
    
    # Append predicted value to series for next iteration
    pred_series = pd.concat([pred_series, pd.Series([y_pred], index=[current_time])])
    
    if step < 5:  # print first few steps
        print(f"Step {step+1}, Forecasted value: {y_pred:.3f}, Time: {current_time}")

# ----------------- VISUALIZATION -----------------
plt.figure(figsize=(12,4))
plt.plot(df.index, df['Usage'], label='Historical Usage')
forecast_index = pd.date_range(df.index[-1]+pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')
plt.plot(forecast_index, forecast_values, color='red', linestyle='--', label='5-Day Forecast')
plt.legend()
plt.title("Random Forest - Future 5-Day Forecast (No Leakage)")
plt.show()
print("Forecast complete.")
