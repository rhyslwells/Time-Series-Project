"""
# Rolling Forecast Evaluation for Time Series Models

## What we are achieving here

The goal of this script is to evaluate how well a time series forecasting model
performs **out-of-sample**, i.e. on data that was not available when the forecast 
was made. 

We do this using a **rolling forecast (walk-forward) validation** approach:

1. Split the historical data into a **training set** (past data) and a **test set** (future data).
2. At each time step in the test set:
   - Train the model only on the data available up to that point.
   - Forecast `n` months ahead.
   - Compare forecasts to the actual observed values once they become available.
3. Collect error metrics (RMSE, MAE) for each forecast horizon (1-month ahead, 2-months ahead, …).

## Why we care

- In practice, we never have access to "future" data when making forecasts. 
  Rolling evaluation mimics the **real-world forecasting process**.
- This lets us see **how forecast accuracy changes with horizon**:
  - Usually, shorter horizons (1–2 months) are more accurate than longer horizons.
  - We can identify how far ahead forecasts remain reliable before errors grow too large.
- By repeating this process as new monthly data comes in, we can check if 
  the model **improves over time** as more training history becomes available.

In short, this script helps answer:
- How accurate is the model at different lead times?
- How quickly does forecast uncertainty grow as we look further ahead?
- Can the model adapt and improve as more data arrives?
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# ============================================================
# 1. Load and explore sample monthly data
# ============================================================

"""
We use the classic "flights" dataset: number of airline passengers per month
(1949–1960). This series is useful because it contains both a strong trend 
(increasing air travel over time) and clear yearly seasonality (summer peaks).

Forecasting models should be able to capture these dynamics.
"""

flights = sns.load_dataset("flights")
flights["month"] = pd.to_datetime(
    flights["year"].astype(str) + "-" + flights["month"].astype(str)
)
flights.set_index("month", inplace=True)
y = flights["passengers"]

# Plot the raw time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y.index, y, marker='o')
ax.set_xlabel("Time")
ax.set_ylabel("Passengers")
ax.set_title("Monthly Airline Passengers (1949–1960)")
plt.show()

# ============================================================
# 2. Define helper functions
# ============================================================

def rolling_forecast(y, max_horizon=6, train_size=60):
    """
    Perform rolling-origin (walk-forward) forecast evaluation.

    Why?
    ----
    - In real life, we never forecast the past — only the future.
    - Rolling-origin simulates deployment: at each time point, we fit the model
      only on data available up to that date, then forecast ahead.
    - This ensures evaluation is *out-of-sample*, avoiding look-ahead bias.

    Parameters
    ----------
    y : pd.Series
        Time series data with datetime index
    max_horizon : int
        Maximum forecast horizon (n-step ahead forecast)
    train_size : int
        Number of initial observations used for training before the first forecast

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns:
        - time: when the forecast target occurs
        - horizon: how many steps ahead (1 = next month, 2 = two months ahead, etc.)
        - actual: observed value
        - forecast: model prediction
        - error: difference (actual - forecast)
    """
    records = []
    
    for t in range(train_size, len(y) - max_horizon):
        # Training sample up to time t (expanding window)
        train = y.iloc[:t]
        
        # Fit Holt-Winters model with trend + seasonality
        model = ExponentialSmoothing(
            train, seasonal='mul', trend='add', seasonal_periods=12
        ).fit(optimized=True)
        
        # Forecast max_horizon months ahead
        forecast = model.forecast(max_horizon)
        
        # Store results for each horizon
        for h in range(1, max_horizon + 1):
            actual = y.iloc[t + h]
            pred = forecast.iloc[h - 1]
            records.append({
                "time": y.index[t + h],
                "horizon": h,
                "actual": actual,
                "forecast": pred,
                "error": actual - pred
            })
    
    return pd.DataFrame(records)


def compute_metrics(df):
    """
    Compute forecast error metrics by horizon.

    Why?
    ----
    - Forecast accuracy almost always declines as horizon increases.
    - By grouping by horizon, we can see how well the model performs for
      short-term vs long-term forecasts.

    Metrics:
    - RMSE (Root Mean Squared Error): penalizes large errors
    - MAE (Mean Absolute Error): easier to interpret, less sensitive to outliers
    """
    metrics = df.groupby("horizon").apply(
        lambda g: pd.Series({
            "RMSE": np.sqrt(mean_squared_error(g["actual"], g["forecast"])),
            "MAE": mean_absolute_error(g["actual"], g["forecast"])
        })
    )
    return metrics

# ============================================================
# 3. Run rolling forecast evaluation
# ============================================================

results = rolling_forecast(y, max_horizon=6, train_size=60)
metrics = compute_metrics(results)

print("Error metrics by forecast horizon:")
print(metrics)

# ============================================================
# 4. Visualize results
# ============================================================

# ---- Error vs horizon ----
"""
Plotting error by forecast horizon shows how forecast accuracy decays
as we look further into the future.
"""
plt.figure(figsize=(10, 4))
plt.plot(metrics.index, metrics["RMSE"], marker="o", label="RMSE")
plt.plot(metrics.index, metrics["MAE"], marker="s", label="MAE")
plt.xlabel("Forecast Horizon (months ahead)")
plt.ylabel("Error")
plt.title("Forecast Error vs Horizon")
plt.legend()
plt.show()

# ---- Rolling RMSE ----
"""
Here we track the 1-step ahead forecast error over time using a 12-month rolling window.

Why?
- This shows whether the model improves as more data becomes available.
- It also highlights if there are periods where forecasts are systematically worse
  (e.g., during structural changes in the data).
"""
one_step = results[results["horizon"] == 1]
window = 12  # rolling window size in months
one_step["rolling_rmse"] = (
    one_step["error"].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
)

plt.figure(figsize=(10, 4))
plt.plot(one_step["time"], one_step["rolling_rmse"], label="Rolling RMSE (1-step ahead)")
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Rolling Forecast Error Over Time (1-month ahead)")
plt.legend()
plt.show()

# ---- Rolling MAPE ----
"""
MAPE (Mean Absolute Percentage Error) expresses errors as a % of actual values.
This makes it scale-independent and easier to interpret (e.g., "on average, forecasts 
are within 8% of the actual values").

We plot a 12-month rolling MAPE for 1-step ahead forecasts.
"""
one_step["rolling_mape"] = (
    one_step["error"].rolling(window).apply(
        lambda x: np.mean(np.abs(x / one_step["actual"]))
    )
)

plt.figure(figsize=(10, 4))
plt.plot(one_step["time"], one_step["rolling_mape"], label="Rolling MAPE (1-step ahead)")
plt.xlabel("Time")
plt.ylabel("MAPE")
plt.title("Rolling Forecast Percentage Error Over Time (1-month ahead)")
plt.legend()
plt.show()
