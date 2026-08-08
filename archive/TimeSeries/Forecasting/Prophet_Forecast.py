"""
Standalone Prophet exploration script.
This script:
1. Generates synthetic monthly time-series data
2. Prepares Prophet-formatted dataframe
3. Splits into train/test
4. Tunes changepoint and seasonality priors
5. Runs cross-validation
6. Produces forecast plot
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot


# ----------------------------------------
# Generate example synthetic data
# ----------------------------------------
# Monthly data from 2010 to 2020
dates = pd.date_range(start="2010-01-01", end="2020-12-01", freq="MS")

# Synthetic signal:
# Trend + seasonality + noise
np.random.seed(42)
n = len(dates)

trend = np.linspace(100, 200, n)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(scale=5, size=n)

values = trend + seasonality + noise

df = pd.DataFrame({
    "ds": dates,
    "y": values
})

df = df.sort_values("ds")


# ----------------------------------------
# Train/test split
# ----------------------------------------
test_size = 12   # last 12 months as test
train = df.iloc[:-test_size].copy()
test = df.iloc[-test_size:].copy()


# ----------------------------------------
# Parameter grid for tuning
# ----------------------------------------
param_grid = {
    "changepoint_prior_scale": [ 0.1, 0.5],
    "seasonality_prior_scale": [1.0]
}

params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

mses = []

# Define rolling cutoff dates for cross-validation
cutoffs = pd.date_range(
    start=train["ds"].min() + pd.DateOffset(years=2),
    end=train["ds"].max() - pd.DateOffset(years=1),
    freq="12M"
)


# ----------------------------------------
# Hyperparameter search loop
# ----------------------------------------
for param in params:

    m = Prophet(**param)
    m.add_country_holidays(country_name='US')
    m.fit(train)

    df_cv = cross_validation(
        model=m,
        horizon="365 days",
        cutoffs=cutoffs
    )

    df_p = performance_metrics(df_cv, rolling_window=1)

    mses.append(df_p["mse"].values[0])


# ----------------------------------------
# Select the best model
# ----------------------------------------
tuning_results = pd.DataFrame(params)
tuning_results["mse"] = mses

best_params = params[np.argmin(mses)]
print("Best parameters:", best_params)

m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)


# ----------------------------------------
# Forecast
# ----------------------------------------
future = m.make_future_dataframe(periods=12, freq="MS")
forecast = m.predict(future)

# Attach predictions to test slice
test = test.merge(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    on="ds",
    how="left"
)


# ----------------------------------------
# Plot results
# ----------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(train["ds"], train["y"], label="Train")
ax.plot(test["ds"], test["y"], label="Actual (Test)", color="blue")
ax.plot(test["ds"], test["yhat"], label="Forecast", linestyle="--", color="darkorange")

plt.fill_between(
    x=test["ds"],
    y1=test["yhat_lower"],
    y2=test["yhat_upper"],
    color="lightblue",
    alpha=0.3,
    label="Uncertainty Interval"
)

ax.set_title("Prophet Forecast Example (Synthetic Data)")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

plt.tight_layout()
plt.show()

