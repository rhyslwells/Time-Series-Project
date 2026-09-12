# Time Series Forecasting — Learning & Documentation

A learning-focused project exploring time series forecasting methods applied to energy systems data. The primary goal is to **understand forecasting approaches** through clear documentation and reusable implementations.

**Main deliverable:** [mkdocs documentation](https://rhyslwells.github.io/Time-Series-Project/) explaining forecasting methods, outcomes on realistic data, and key learnings.

**Focus:** Education and clarity, not production deployment.


## Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
uv sync
```

This creates a virtual environment and installs all dependencies.

### Running IPython

For interactive exploration:

```bash
uv run ipython
```

### Using Marimo for Exploration

For notebook-based analysis and consolidation:

```bash
uv run marimo edit
```

## How This Project Works

1. **Synthetic data generation** (`src/data/`) — Creates realistic energy metering data (14 days × 15 assets × 30-min intervals)
2. **Forecasting models** (`src/ts_model_framework.py`) — Multiple approaches (Seasonal Naive, SARIMA, Exponential Smoothing, LightGBM)
3. **Evaluation & diagnostics** (`src/ts_plots.py`) — Performance metrics, uncertainty quantification, residual analysis
4. **Documentation** (`docs_src/`) — Methodology explanations, results, and learnings captured in mkdocs

## Dependencies

Core dependencies:

- **polars**: Data operations (all I/O and transformations)
- **statsmodels, scikit-learn, lightgbm**: Forecasting engines
- **numpy**: Numerical arrays for models
- **marimo, ipython**: Exploration and consolidation
- **matplotlib, plotly**: Visualization

See `pyproject.toml` for the complete list.
