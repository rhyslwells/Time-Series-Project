# SARIMA Time Series Forecasting

Complete interactive exploration of SARIMA forecasting for energy metering data with uncertainty quantification.

## Overview

This notebook demonstrates a principled forecasting pipeline:

1. **Load & Explore** — 14-day metering data (30-min intervals)
2. **Train/Test Split** — 10 days train, 4 days test
3. **Model Fitting** — SARIMA(1,1,1)×(1,1,1,48) with daily seasonality
4. **Forecasting** — Point forecasts + 80% prediction intervals
5. **Evaluation** — MAE, RMSE, MAPE, PI coverage metrics
6. **Uncertainty Analysis** — When is the model uncertain?
7. **Probability Events** — P(forecast > threshold) for operational decisions
8. **Visualizations** — Time series, accuracy, residuals

## Interactive Notebook

Explore the SARIMA forecasting workflow interactively:

/// marimo-embed-file
filepath: working_notes/2_basic_forecasting/sarima_marimo.py
height: 1000px
show_source: false
///

## Key Findings

- **SARIMA(1,1,1)×(1,1,1,48)** effectively captures daily seasonality in 30-minute energy metering
- **Prediction intervals** quantify forecast uncertainty for each timestep
- **Standard output contract** enables downstream analytics and aggregation
- **Multiple evaluation metrics** (MAE, RMSE, MAPE, coverage) reveal different forecast quality aspects

## Output Contract

All forecasts follow a standard format:

| Column | Description |
|--------|-------------|
| `asset_id` | Unique identifier |
| `timestamp` | Forecast applies to this time |
| `prediction` | Point forecast (expected value) |
| `lower` | Lower bound of prediction interval (P10) |
| `upper` | Upper bound of prediction interval (P90) |
| `uncertainty_width` | Interval width (upper - lower) |

## Next Steps

- Test alternative SARIMA parameters (p, d, q)
- Compare against other models (Exponential Smoothing, XGBoost)
- Aggregate forecasts across multiple assets (portfolio level)
- Derive flexibility forecasts from the distribution
- Monitor for concept drift over time
