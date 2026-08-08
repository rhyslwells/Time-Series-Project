"""
Random Forest Time Series Forecasting with 95% Confidence Intervals

This script demonstrates:
1. Loading time series data (CSV) and preprocessing it.
2. Feature engineering for time series:
   - Lag features
   - Rolling window statistics
   - Time-of-day and day-of-week indicators
3. Training a Random Forest Regressor to forecast future points iteratively.
4. Generating 95% confidence intervals using the distribution of individual tree predictions.
5. Visualizing historical and forecasted data including uncertainty.

Inputs:
- CSV file with columns ['Time', 'Usage'].

Outputs:
- Forecasted series for a specified horizon.
- Plot with 95% confidence intervals.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")  # Hide sklearn/pandas deprecation warnings for clarity

# ----------------- CONFIGURATION -----------------
DATE_COL = "Time"        # Column name for timestamps
VALUE_COL = "Usage"      # Column name for target variable

FIT_WEEKS = 12           # Train on last 12 weeks of data
FORECAST_FREQ = "30min"  # Frequency of data (here: 30 minutes)
FORECAST_DAYS = 7        # How many future days to forecast
PLOT_WEEKS = 4           # Show this many past weeks in the output plot

# Feature engineering configuration:
LAGS = [1, 2, 3, 48, 96, 336, 2*336, 3*336]             # half-hour, 1h, 1.5h, 1d, 2d, 1w, 2w, 3w
ROLLING_WINDOWS = [3, 6, 12, 48, 96, 336, 2*336, 3*336] # rolling averages for different horizons

# Optional tuning
DO_SMALL_GRID_SEARCH = True
# -------------------------------------------------


# ----------------- DATA FUNCTIONS -----------------
def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Load CSV, parse dates, sort chronologically, set datetime index, and interpolate missing values.
    """
    df = pd.read_csv(filepath, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)             # Ensure ascending chronological order
    df.set_index(DATE_COL, inplace=True)      # Time series index
    df[VALUE_COL] = df[VALUE_COL].interpolate(method='time')  # Fill gaps
    return df


def prepare_fit_data(df: pd.DataFrame, weeks: int = FIT_WEEKS) -> pd.DataFrame:
    """
    Extract last N weeks of data for training.
    This avoids fitting on excessively old data (which may be less relevant).
    """
    last_date = df.index.max()
    start_date = last_date - pd.Timedelta(weeks=weeks)
    return df.loc[df.index >= start_date]


# ----------------- FEATURE ENGINEERING -----------------
def generate_lag_features(series: pd.Series, lags: list[int]) -> list[float]:
    """
    For each lag, take the value at that offset from the end of the series.
    Example: lag=48 means "value 24h ago" for 30-min data.
    """
    return [float(series.iloc[-lag]) for lag in lags]


def generate_rolling_features(series: pd.Series, rolling_windows: list[int]) -> list[float]:
    """
    For each window size, compute mean of the last 'window' points.
    Example: window=48 means "average over last 24h".
    """
    return [float(series.iloc[-window:].mean()) for window in rolling_windows]


def generate_time_features(timestamp: pd.Timestamp) -> list[int]:
    """
    Encode basic calendar features:
    - Hour of day (0–23)
    - Day of week (0=Monday,...,6=Sunday)
    """
    return [timestamp.hour, timestamp.dayofweek]


def build_feature_matrix(series: pd.Series,
                         lags: list[int],
                         rolling_windows: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct supervised learning dataset from raw series.
    ##### Adds extract features constructed from the history up to time i. Makes dataframe bigger.
    Each row i contains features built from the history up to time i:
      - lag values
      - rolling means
      - calendar features
    Target y[i] = actual series value at time i.
    """
    X, y = [], []
    max_window = max(max(lags), max(rolling_windows))  # Need enough history to compute features
    for i in range(max_window, len(series)):
        hist = series.iloc[:i]              # all values up to (not including) i
        timestamp = series.index[i]

        row = []
        row += generate_lag_features(hist, lags)
        row += generate_rolling_features(hist, rolling_windows)
        row += generate_time_features(timestamp)

        X.append(row)
        y.append(float(series.iloc[i]))

    X = pd.DataFrame(X)
    y = pd.Series(y, index=series.index[max_window:])
    

    return X, y


# ----------------- MODEL TRAINING -----------------
def train_random_forest(X: pd.DataFrame,
                        y: pd.Series,
                        n_estimators: int = 100,
                        max_depth: int | None = None,
                        random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest on the given feature matrix X and target y.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # use all CPUs
    )
    rf.fit(X, y)
    return rf


def tiny_grid_search_rf(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Brute-force a very small grid of hyperparameters to improve model fit.
    Uses RMSE on validation set to choose the best config.
    """
    grid = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 150, "max_depth": 3},
        {"n_estimators": 200, "max_depth": 3},
        {"n_estimators": 150, "max_depth": 6},
        {"n_estimators": 200, "max_depth": 6},
    ]
    best_rmse = float("inf")
    best_params = grid[0]
    for params in grid:
        model = train_random_forest(X_train, y_train,
                                    n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"])
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    print(f"[Grid Search] Best params: {best_params} with RMSE={best_rmse:.3f}")
    return best_params


# ----------------- PREDICTION -----------------
def predict_next_step(model: RandomForestRegressor,
                      pred_series: pd.Series,
                      lags: list[int],
                      rolling_windows: list[int],
                      current_time: pd.Timestamp) -> tuple[float, float, float]:
    """
    Predict the next time step.
    Returns:
      - mean prediction
      - lower 95% CI
      - upper 95% CI
    CI is computed by looking at the spread of predictions across all trees.
    """
    X_next = []
    X_next += generate_lag_features(pred_series, lags)
    X_next += generate_rolling_features(pred_series, rolling_windows)
    X_next += generate_time_features(current_time)

    # Collect predictions from all trees (each tree gives a slightly different value)
    tree_preds = np.array([est.predict([X_next])[0] for est in model.estimators_])
    mean_pred = float(tree_preds.mean())
    lower = float(np.percentile(tree_preds, 2.5))    # 2.5th percentile
    upper = float(np.percentile(tree_preds, 97.5))   # 97.5th percentile
    return mean_pred, lower, upper


def iterative_forecast(model: RandomForestRegressor,
                       pred_series: pd.Series,
                       steps: int,
                       freq: str,
                       lags: list[int],
                       rolling_windows: list[int]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Forecast multiple steps into the future using recursive strategy:
      - Predict one step ahead
      - Append prediction to history
      - Use it to predict the next step
    This accumulates error but mimics how we would use the model in practice.
    """
    forecast_values, lower_bounds, upper_bounds = [], [], []
    for _ in range(steps):
        current_time = pred_series.index[-1] + pd.Timedelta(freq)
        mean_pred, lower, upper = predict_next_step(model, pred_series, lags, rolling_windows, current_time)
        forecast_values.append(mean_pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        # Append predicted value to history so it can be used for the next step
        pred_series = pd.concat([pred_series, pd.Series([mean_pred], index=[current_time])])

    forecast_index = pd.date_range(start=pred_series.index[-steps], periods=steps, freq=freq)
    return (pd.Series(forecast_values, index=forecast_index, name="forecast"),
            pd.Series(lower_bounds, index=forecast_index, name="lower"),
            pd.Series(upper_bounds, index=forecast_index, name="upper"))

def train_val_split(X, y, val_fraction=0.2):
    """
    Chronological train/validation split for time series.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.
    val_fraction : float
        Fraction of data to hold out for validation.
    
    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    split_idx = int(len(X) * (1 - val_fraction))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_val, y_train, y_val

# ----------------- MAIN FORECAST FUNCTION -----------------
def forecast_random_forest_with_uncertainty(df_fit: pd.DataFrame,
                                            forecast_days: int = FORECAST_DAYS,
                                            freq: str = FORECAST_FREQ,
                                            lags: list[int] = LAGS,
                                            rolling_windows: list[int] = ROLLING_WINDOWS
                                            ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Train a Random Forest on historical data and forecast the next N days.
    """
    X, y = build_feature_matrix(df_fit[VALUE_COL], lags, rolling_windows)
    # print(X.tail()) #to see lags ect
    print(f"Training features shape: {X.shape}")

    if DO_SMALL_GRID_SEARCH:
        X_train, X_val, y_train, y_val = train_val_split(X, y, val_fraction=0.2)

        # Now search hyperparams using only train vs validation
        best_params = tiny_grid_search_rf(X_train, y_train, X_val, y_val)
        rf = train_random_forest(X, y,
                                 n_estimators=best_params["n_estimators"],
                                 max_depth=best_params["max_depth"])
    else:
        rf = train_random_forest(X, y)

    print("Random Forest trained.")

    steps = int((24 * 2) * forecast_days)  # steps = 48 per day for 30min frequency
    return iterative_forecast(rf, df_fit[VALUE_COL].copy(), steps, freq, lags, rolling_windows)


# ----------------- PLOTTING -----------------
def plot_forecast(df_full: pd.DataFrame,
                  forecast_series: pd.Series,
                  lower_series: pd.Series,
                  upper_series: pd.Series,
                  series_name: str,
                  plot_weeks: int = PLOT_WEEKS) -> None:
    """
    Plot last N weeks of historical data plus forecast and CI.
    """
    start_date = df_full.index.max() - pd.Timedelta(weeks=plot_weeks)
    last_weeks = df_full.loc[df_full.index >= start_date]

    plt.figure(figsize=(12, 5))
    plt.plot(last_weeks.index, last_weeks[VALUE_COL], label="Historical")
    plt.plot(forecast_series.index, forecast_series, color="red", linestyle="--", label="Forecast")
    plt.fill_between(forecast_series.index,
                     lower_series.values,
                     upper_series.values,
                     color="red", alpha=0.2, label="95% CI")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.title(f"Random Forest Forecast for {series_name}")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    file_path = os.path.join("plots", f"{series_name}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {file_path}")


# ----------------- PIPELINE -----------------
def run_pipeline(csv_file: str) -> None:
    """
    Full pipeline:
    - Load and clean data
    - Restrict to last FIT_WEEKS
    - Train model and forecast FORECAST_DAYS
    - Plot and save results
    """
    df_full = load_and_prepare(csv_file)
    df_fit = prepare_fit_data(df_full)
    print(f"Fitting Random Forest on last {len(df_fit)} points ({FIT_WEEKS} weeks)...")

    forecast_series, lower_series, upper_series = forecast_random_forest_with_uncertainty(
        df_fit,
        forecast_days=FORECAST_DAYS,
        freq=FORECAST_FREQ,
        lags=LAGS,
        rolling_windows=ROLLING_WINDOWS
    )

    series_name = os.path.basename(csv_file).replace(".csv", "")
    plot_forecast(df_full, forecast_series, lower_series, upper_series, series_name)
    print("Forecast complete.")


# ----------------- CLI -----------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rf_forecast.py <csv_file>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
