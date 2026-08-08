# Time Series Forecasting

A time series forecasting and analysis system focused on energy systems, flexibility forecasting, and optimization.

## Project Structure

```
.
├── README.md              # This file
├── CLAUDE.md              # AI assistant collaboration guidelines
├── pyproject.toml         # Project dependencies and config
├── notes.md               # Forecasting architecture and strategy
├── archive/               # Reference scripts and experimental work
└── src/                   # Main source code (to be created)
```

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

## Key Concepts

See **notes.md** for a comprehensive overview of the forecasting architecture, including:

- Basic forecast generation with uncertainty
- Derived features and signals
- Asset profiling and baseline estimation
- Anomaly detection
- Model health monitoring
- Portfolio-level aggregation
- Flexibility forecasting
- Trading volume estimation

## Development Workflow

1. **Exploration**: Start with IPython for rapid testing
2. **Consolidation**: Move validated work into marimo notebooks
3. **Implementation**: Clean up into reusable modules
4. **Reference**: Archive experimental or reference work to `/archive`

## Dependencies

Core dependencies are kept minimal and focused:

- **statsmodels, scikit-learn, lightgbm**: Forecasting engines
- **sktime, tbats, pmdarima**: Time series specialized tools
- **pandas, numpy**: Data handling
- **marimo, ipython**: Exploration and visualization
- **matplotlib, plotly, seaborn**: Visualization

See `pyproject.toml` for the complete list and optional development tools.
