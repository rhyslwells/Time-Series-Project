# Time Series Forecasting

A forecasting system for energy assets with multi-layer analysis and flexibility optimization.

## Quick Start

```bash
uv sync
uv run ipython
```

See [Architecture](architecture.md) for the system design.

## Key Concepts

This project implements a layered forecasting approach:

1. Raw metering data
2. Point forecasts with uncertainty
3. Derived features (daily energy, peaks, ramps)
4. Asset profiling and behavior classification
5. Anomaly detection
6. Flexibility forecasting
7. Portfolio aggregation
8. Trading optimization

## Exploration

Active investigations are documented in [Exploration Notebooks](notebooks.md).

## Structure

- **Findings**: Validated discoveries and analysis results
- **Methodology**: How we approach problems
- **Data**: Schema and data quality information
- **Notebooks**: Interactive marimo explorations
