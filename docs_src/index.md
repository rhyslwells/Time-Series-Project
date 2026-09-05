# Time Series Forecasting

A forecasting system for energy assets with multi-layer analysis and flexibility optimization.

See [theory/](theory/) for the system design and architectural decisions.

## Key Concepts

This project implements a layered forecasting approach:

1. Raw metering data
2. Point forecasts with uncertainty
3. Derived features (daily energy, peaks, ramps)
4. Asset profiling and behavior classification
5. ...


## Key Principles

1. **Forecast as intermediate layer**: Not a final output, but input to downstream analysis
2. **Uncertainty-driven**: Always include prediction intervals, not just point estimates
3. **Asset profiling**: Classify assets by behavioral fingerprints
4. **Swappable models**: Standard forecast contract enables model replacement
5. **Portfolio view**: Leverage correlation and aggregation benefits

## Documentation

- **[Findings](findings/)** — Validated discoveries from analysis and modeling
- **[Data](data/)** — Data contracts and generation documentation
- **[Theory](theory/)** — Design rationale and methodological foundations
- **[Notebooks](notebooks/)** — Consolidation notebooks and exploration guides