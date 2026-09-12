"""
Residual Analysis Demonstration
--------------------------------
This script shows how to evaluate residuals from:
1. Seasonal decomposition
2. SARIMA modelling
Using:
- Q-Q plots
- Standard residual diagnostics
"""

# ------------------------------------------------------
# Imports
# ------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------
# Generate synthetic time-series data
# ------------------------------------------------------
np.random.seed(42)

n = 300
time_index = pd.date_range(start="2000-01-01", periods=n, freq="M")

trend = np.linspace(0, 20, n)
season = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(scale=2, size=n)

y = trend + season + noise

df = pd.DataFrame({"y": y}, index=time_index)

# ------------------------------------------------------
# Seasonal decomposition and residuals
# ------------------------------------------------------
decomp = seasonal_decompose(df["y"], period=12, model="additive")

df["decomp_residuals"] = decomp.resid

# Plot decomposition
decomp.plot()
plt.tight_layout()
plt.show()

# Q-Q plot: decomposition residuals
qqplot(df["decomp_residuals"].dropna(), line="45")
plt.title("Q-Q Plot: Decomposition Residuals")
plt.show()

# ------------------------------------------------------
# SARIMA model fit
# ------------------------------------------------------
# Basic SARIMA(1,0,1)x(1,1,1,12)
model = SARIMAX(df["y"], order=(1,0,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Extract model residuals
sarima_residuals = model_fit.resid

# ------------------------------------------------------
# Q-Q plot: SARIMA residuals
# ------------------------------------------------------
qqplot(sarima_residuals, line="45")
plt.title("Q-Q Plot: SARIMA Residuals")
plt.show()

# ------------------------------------------------------
# Full diagnostics
# ------------------------------------------------------
model_fit.plot_diagnostics(figsize=(10, 8))
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# Summaries
# ------------------------------------------------------
print(model_fit.summary())
