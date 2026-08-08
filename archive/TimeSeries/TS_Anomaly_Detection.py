import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA

# Function for ARIMA-based anomaly detection
def detect_outliers_arima(df, order=(1, 1, 1)):
    # Ensure 'value' column is numeric and has no missing values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df = df.dropna(subset=['value'])  # Drop rows where 'value' is NaN
    
    if df['value'].empty:
        print("Error: After cleaning, 'value' column is empty.")
        return pd.DataFrame()  # Return empty DataFrame if no data left
    
    # Apply ARIMA model
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    residuals = model_fit.resid
    z_scores = zscore(residuals)
    outliers_arima = df[np.abs(z_scores) > 3]
    print(f"Potential Outliers (ARIMA Residuals): {len(outliers_arima)} detected")
    return outliers_arima

# Function to display basic dataset info
def display_basic_info(df, name):
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Time Range: {df.index.min()} to {df.index.max()}")

# Function to display missing values percentage
def display_missing_values(df):
    missing_pct = df.isnull().mean() * 100
    print("\nMissing Values (%):")
    print(missing_pct)

# Function for IQR based outlier detection
def detect_outliers_iqr(df):
    Q1, Q3 = df['value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"\nPotential Outliers (IQR): {len(outliers_iqr)} detected")
    return outliers_iqr

# Function for Z-score based outlier detection
def detect_outliers_z(df, z_thresh):
    df['zscore'] = zscore(df['value'])
    outliers_z = df[np.abs(df['zscore']) > z_thresh]
    print(f"Potential Outliers (Z-score > {z_thresh}): {len(outliers_z)} detected")
    return outliers_z

# Function for STL decomposition and residual-based outlier detection
def detect_outliers_stl(df, window):
    try:
        decomposition = seasonal_decompose(df['value'], model='additive', period=window)
        residual = decomposition.resid.dropna()
        residual_std = residual.std()
        outliers_stl = residual[np.abs(residual) > 2 * residual_std]
        print(f"Potential Outliers (STL Residuals): {len(outliers_stl)} detected")
        return outliers_stl
    except ValueError:
        print("STL Decomposition skipped (not enough data points).")
        return pd.Series(dtype="float64")

# Function for seasonality analysis (Autocorrelation)
def analyze_seasonality(df):
    autocorr_vals = acf(df['value'], nlags=min(100, len(df) - 1), fft=True)
    peak_lag = np.argmax(autocorr_vals[1:]) + 1  # Ignore lag=0
    print(f"Strongest Periodicity at lag: {peak_lag}")

# Updated function to highlight outliers
def plot_dataset(df, name, window, outliers_iqr, outliers_z, outliers_stl, outliers_arima):
    """
    Generate and display an interactive plot of the dataset, allowing toggling of components
    via the legend.

    Args:
    - df (pd.DataFrame): Time series dataframe.
    - name (str): Dataset name.
    - window (int): Rolling mean window size.
    - outliers_iqr (pd.DataFrame): IQR outliers.
    - outliers_z (pd.DataFrame): Z-score outliers.
    - outliers_stl (pd.DataFrame): STL outliers.
    - outliers_arima (pd.DataFrame): ARIMA-based outliers.

    Returns:
    - None
    """
    fig = go.Figure()

    # Raw Data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'], 
        mode='lines', name="Raw Data", 
        opacity=0.6, showlegend=True
    ))

    # Rolling Mean
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'].rolling(window=window, min_periods=1).mean(),
        mode='lines', name=f"Rolling Mean (window={window})", 
        line=dict(color='red'), showlegend=True
    ))

    # Highlight IQR Outliers
    if not outliers_iqr.empty:
        fig.add_trace(go.Scatter(
            x=outliers_iqr.index, y=outliers_iqr['value'], 
            mode='markers', name=f"IQR Outliers ({len(outliers_iqr)})", 
            marker=dict(color='purple', size=6, symbol='x'), showlegend=True
        ))

    # Highlight Z-score Outliers
    if not outliers_z.empty:
        fig.add_trace(go.Scatter(
            x=outliers_z.index, y=outliers_z['value'], 
            mode='markers', name=f"Z-score Outliers ({len(outliers_z)})", 
            marker=dict(color='blue', size=6, symbol='circle'), showlegend=True
        ))

    # Highlight STL Outliers
    if not outliers_stl.empty:
        fig.add_trace(go.Scatter(
            x=outliers_stl.index, y=df.loc[outliers_stl.index, 'value'], 
            mode='markers', name=f"STL Outliers ({len(outliers_stl)})", 
            marker=dict(color='orange', size=6, symbol='diamond'), showlegend=True
        ))

    # Highlight ARIMA Outliers
    if not outliers_arima.empty:
        fig.add_trace(go.Scatter(
            x=outliers_arima.index, y=outliers_arima['value'], 
            mode='markers', name=f"ARIMA Outliers ({len(outliers_arima)})", 
            marker=dict(color='green', size=6, symbol='star'), showlegend=True
        ))

    fig.update_layout(
        title=f"Time Series Analysis: {name}", 
        xaxis_title="Timestamp", 
        yaxis_title="Value",
        legend=dict(
            x=1,  # Position the legend outside the plot
            y=1.3,  # Place it at the top-right corner
            traceorder='normal',
            orientation='v',
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0)', 
            bordercolor='Black', 
            borderwidth=1
        ),
        margin=dict(r=150)  # Adjust the right margin to make space for the legend
    )
    fig.show()

# Main function for dataset exploration
def explore_dataset(df, name, window=50, z_thresh=3.0):
    display_basic_info(df, name)
    display_missing_values(df)
    
    outliers_iqr = detect_outliers_iqr(df)
    outliers_z = detect_outliers_z(df, z_thresh)
    outliers_stl = detect_outliers_stl(df, window)
    outliers_arima = detect_outliers_arima(df)
    
    analyze_seasonality(df)
    plot_dataset(df, name, window, outliers_iqr, outliers_z, outliers_stl, outliers_arima)

# Function to load dataset and run analysis based on user input
def analyze_single_file(file_path):
    """
    Load the dataset from the given path and perform the analysis.

    Args:
    - file_path (str): Relative or absolute path to the dataset.

    Returns:
    - None
    """

    # Load dataset
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        # Extract file name from path for display
        name = os.path.basename(file_path)
        explore_dataset(df, name)
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Example usage
begin="../../../"
filenames = [
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_no_noise.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_perfect_square_wave.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_small_noise.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_flatline.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_noisy.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_flatmiddle.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_jumpsdown.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_jumpsup.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_nojump.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_increase_spike_density.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_load_balancer_spikes.csv"
]

file = filenames[7]
file_path = begin + file
analyze_single_file(file_path)
