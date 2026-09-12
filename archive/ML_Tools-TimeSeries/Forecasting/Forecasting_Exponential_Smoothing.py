# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Utility function for evaluation
def forecast_evaluation(predicted, actual):
    """
    Evaluates forecast accuracy using MSE, MAE, RMSE, and MAPE.
    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Generate synthetic data with non-linear trend and slight noise
dates = pd.date_range('2020-01-01', '2020-02-20', freq='D')
base_data = np.array([100, 102, 105, 108, 110, 112, 115, 118, 120, 122, 124, 126, 130, 133, 137, 140, 143, 145, 147, 150, 
                     153, 156, 159, 162, 165, 168, 470, 172, 175, 408, 180, 183, 185, 188, 190, 192, 195, 198, 200, 202, 
                     205, 208, 210, 213, 215, 218, 220, 223, 225, 228, 130])

# Sinusoidal trend (a * sin(bx + c) + d)
sinusoidal_coeffs = [2, 100, 0, 100]  # a = amplitude, b = frequency, c = phase shift, d = vertical shift
non_linear_trend = sinusoidal_coeffs[0] * np.sin(sinusoidal_coeffs[1] * np.arange(len(base_data)) + sinusoidal_coeffs[2]) + sinusoidal_coeffs[3]

# Adding slight noise
np.random.seed(42)
noise = np.random.normal(0, 1, len(base_data))  # Increase standard deviation of noise

# Final data with non-linear trend and noise
noisy_data = non_linear_trend + noise

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Close': noisy_data})

# Defining the train and test split (Train on all data except last 'test_period' days)
train = data[:-10]  # Train on all data except the last 10 days
test = data[-10:]   # Test on the last 10 days (or adjust based on your data)

# Setting seasonal_periods based on your data's frequency (adjust accordingly)
seasonal_periods = 12  # Set this value according to your data's seasonal cycle (e.g., 12 for monthly data)

# Creating lists to store forecasts for plotting
ses_forecasts = []
holt_forecasts = []
hw_forecasts = []

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(train['Close']).fit(optimized=True)
ses_forecast = ses_model.forecast(test.shape[0])
ses_forecasts.append(ses_forecast)

# Holt's Linear Trend (Double Exponential Smoothing)
holt_model = Holt(train['Close']).fit(optimized=True)
holt_forecast = holt_model.forecast(test.shape[0])
holt_forecasts.append(holt_forecast)

# Holt-Winters' Method (Triple Exponential Smoothing)
# Ensure there are enough data points for seasonal cycles
if len(train) >= seasonal_periods * 2:
    hw_model = ExponentialSmoothing(train['Close'], trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit(optimized=True)
    hw_forecast = hw_model.forecast(test.shape[0])
    hw_forecasts.append(hw_forecast)
else:
    print(f"Not enough data for Holt-Winters. Skipping.")

# Plotting all forecasts on a single plot
plt.figure(figsize=(12,6))

# Plot training and actual test data
plt.plot(train.index, train['Close'], label='Training Data', color='blue')
plt.plot(test.index, test['Close'], label='Test Data', color='orange')

# Plotting the forecasts for each method
plt.plot(test.index, ses_forecasts[0], label='SES Forecast', color='green', linestyle='--')
plt.plot(test.index, holt_forecasts[0], label='Holt Forecast', color='red', linestyle='--')
plt.plot(test.index, hw_forecasts[0], label='Holt-Winters Forecast', color='purple', linestyle='--')

# Adding titles and labels
plt.title('Forecast Comparison: SES, Holt, and Holt-Winters')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='best')

# Show the plot
plt.show()
