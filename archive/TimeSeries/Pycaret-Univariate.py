"""
Standalone PyCaret Time Series Forecasting Example
Demonstrates:
1. Loading example data
2. Setting up PyCaret time-series experiment
3. Comparing models
4. Analyzing model performance
5. Generating forecasts
6. Saving and loading model pipelines
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import pycaret
from pycaret.datasets import get_data
from pycaret.time_series import (
    setup,
    compare_models,
    plot_model,
    predict_model,
    check_stats,
    save_model,
    load_model
)

import matplotlib.pyplot as plt


# ----------------------------------------
# Check version
# ----------------------------------------
print("PyCaret version:", pycaret.__version__)


# ----------------------------------------
# Load example dataset
# ----------------------------------------
# Monthly airline passenger counts (standard demo dataset)
data = get_data("airline")
print("Loaded dataset shape:", data.shape)

# Optional plot
data.plot(title="Airline Passenger Series")
plt.show()


# ----------------------------------------
# Setup PyCaret time-series environment
# ----------------------------------------
exp = setup(
    data=data,
    fh=12,          # forecast horizon
    session_id=123, # reproducibility
    fold=3,         # CV folds
    verbose=True
)

# Optional stats on raw series
check_stats()


# ----------------------------------------
# Compare all available models
# ----------------------------------------
best_model = compare_models()
print("Best model selected:", best_model)


# ----------------------------------------
# Visual diagnostics for the chosen model
# ----------------------------------------
# Forecast on test horizon
plot_model(best_model, plot="forecast")

# Residual analysis
plot_model(best_model, plot="residuals")


# ----------------------------------------
# Predict on test set (holdout)
# ----------------------------------------
holdout_pred = predict_model(best_model)
print("Holdout predictions:")
print(holdout_pred.head())


# ----------------------------------------
# Forecast further into the future
# ----------------------------------------
future_pred = predict_model(best_model, fh=24)
print("24-period ahead forecast:")
print(future_pred.head())


# ----------------------------------------
# Save and load the trained pipeline
# ----------------------------------------
# save_model(best_model, "pycaret_ts_pipeline")

# loaded = load_model("pycaret_ts_pipeline")
# print("Loaded model pipeline:")
# print(loaded)

# print("Done.")
