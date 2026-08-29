# Time Series Forecasting

A time series forecasting and analysis system focused on energy systems, flexibility forecasting, and optimization.

See: https://rhyslwells.github.io/Time-Series-Project/


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

## Dependencies

Core dependencies are kept minimal and focused:

- **statsmodels, scikit-learn, lightgbm**: Forecasting engines
- **sktime, tbats, pmdarima**: Time series specialized tools
- **pandas, numpy**: Data handling
- **marimo, ipython**: Exploration and visualization
- **matplotlib, plotly, seaborn**: Visualization

See `pyproject.toml` for the complete list and optional development tools.
