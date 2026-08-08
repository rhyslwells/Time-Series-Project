# System Architecture

See [notes.md](../notes.md) for the complete strategic design.

## Layered Progression

```
Metering Data
    |
    v
Point Forecasts + Uncertainty
    |
    +---> Daily Energy
    +---> Peak/Ramp Rates
    +---> Asset Profiles
    |
    v
Anomaly Detection
    |
    v
Flexibility Envelope
    |
    v
Portfolio Aggregation
    |
    v
Optimization & Trading
```

## Key Principles

1. **Forecast as intermediate layer**: Not a final output, but input to downstream analysis
2. **Uncertainty-driven**: Always include prediction intervals, not just point estimates
3. **Asset profiling**: Classify assets by behavioral fingerprints
4. **Swappable models**: Standard forecast contract enables model replacement
5. **Portfolio view**: Leverage correlation and aggregation benefits

## Standard Forecast Contract

All forecasts use a consistent schema:

```python
{
    "asset_id": str,           # Unique asset identifier
    "timestamp": datetime,     # Forecast target time
    "yhat": float,            # Point forecast
    "lower": float,           # Lower prediction interval
    "upper": float,           # Upper prediction interval
    "model_version": str,     # Which model produced this
    "forecast_ts": datetime   # When forecast was made
}
```

## Directory Layout

```
src/
├── forecasting/          # Core forecasting models
├── features/             # Feature engineering
├── anomaly/              # Anomaly detection
└── utils/                # Shared utilities

docs/
├── findings/             # Validated discoveries
├── methodology/          # How we do things
└── data/                 # Data documentation

archive/
└── [experimental work]

working_notes/            # Untracked scratch space
```

## Development Workflow

1. **Explore** in IPython, save to `working_notes/`
2. **Validate** with marimo notebooks
3. **Consolidate** findings into `docs/`
4. **Implement** clean code in `src/`
