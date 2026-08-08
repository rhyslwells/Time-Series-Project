"""
VARMAX Example: Multivariate Time Series Forecasting

This script demonstrates how to fit a VARMAX model using statsmodels.
It simulates two related time series, fits a model, evaluates the fit,
and produces forecasts.

Dependencies:
    pip install statsmodels pandas matplotlib numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX

# ----------------------------------------------------------
# 1. Simulate some example data
# ----------------------------------------------------------

np.random.seed(42)

# Create a time index
dates = pd.date_range(start="2020-01-01", periods=100, freq="M")

# Generate two correlated time series
x1 = np.cumsum(np.random.normal(0, 1, 100))             # random walk
x2 = 0.6 * x1 + np.random.normal(0, 1, 100)             # dependent on x1

# Combine into a DataFrame
df = pd.DataFrame({"series_A": x1, "series_B": x2}, index=dates)

# ----------------------------------------------------------
# 2. Split into training and test data
# ----------------------------------------------------------

train = df.iloc[:-12]  # first 88 observations
test = df.iloc[-12:]   # last 12 for forecast evaluation

# ----------------------------------------------------------
# 3. Fit a VARMAX model
# ----------------------------------------------------------

# (p,q) = (2,1): VAR(2) + MA(1)
# trend='c' adds a constant term
model = VARMAX(train, order=(2, 1), trend='c')

# disp=False suppresses verbose optimization output
results = model.fit(disp=False)
print(results.summary())

# ----------------------------------------------------------
# 4. Forecast future values
# ----------------------------------------------------------

n_forecast = len(test)
forecast = results.forecast(steps=n_forecast)

# ----------------------------------------------------------
# 5. Plot actual vs forecasted series
# ----------------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for i, col in enumerate(df.columns):
    ax[i].plot(df.index, df[col], label='Actual', color='black')
    ax[i].plot(forecast.index, forecast[col], label='Forecast', color='tab:red', linestyle='--')
    ax[i].axvline(test.index[0], color='gray', linestyle=':')
    ax[i].set_title(f"Forecast for {col}")
    ax[i].legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 6. Evaluate forecasting performance (simple RMSE)
# ----------------------------------------------------------

rmse = np.sqrt(((forecast - test) ** 2).mean())
print("\nRoot Mean Squared Error (per series):")
print(rmse)
