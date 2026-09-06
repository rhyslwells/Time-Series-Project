# Time Series Forecasting

A multi-layer forecasting system for energy assets.

**Goal: build a reusable library of code snippets for time series forecasting on energy assets, organized by forecasting task and designed for low-friction implementation.**

This is a curated collection of production-ready code patterns from the time series forecasting framework—SARIMA, Exponential Smoothing, LightGBM, and diagnostic tools—presented as self-contained snippets with hand-written narrative guidance.

Users land here, find their task (trend removal, seasonality detection, walk-forward validation, anomaly detection, model tuning, or evaluation), copy the pattern, and adapt it to their own use case.

## Concepts explored

1. Forecasting with plots
2. Comparing different models for a given asset and time period
3. Core forecasting tasks:
    - Trend removal and deseasonalization
    - Seasonality detection
    - Walk-forward validation
    - Anomaly detection
    - Model selection and tuning
    - Forecast evaluation

## Concepts to explore in future

1. **Asset profiling**: classify assets by behavioral fingerprint
2. **Daily metrics**: make use of the daily aggregate metrics
3. **Uncertainty quantification**: understand prediction intervals and confidence levels for forecasts

## Documentation

- **[Data](data/)** — Data descriptions and generation documentation
- **[Theory](theory/)** — Design rationale and methodological foundations
- **[Findings](findings/)** — Notes on forecasting tasks and the resources that support them
- **[Notebooks](notebooks/)** — Exploration notebooks covering forecasting tasks

