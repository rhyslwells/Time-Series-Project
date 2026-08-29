# Forecasting System Architecture

## Core Layers (from main_idea.md)

1. **Raw metering** — Raw data (30-min intervals, kWh)
2. **Forecast models** — Point & distribution forecasts (yhat, P10, P50, P90)
3. **Derived features** — Daily energy, peak, ramp, profiles
4. **Flexibility** — Available reduction/increase, SOC projections
5. **Anomalies** — Residuals, outliers, asset health
6. **Portfolio** — Aggregated forecasts, correlation effects
7. **Trading** — Confidence-adjusted volume, tradable quantity
8. **Optimization** — Stochastic planning, scenario trees

## Design Principles

- **Separation of concerns**: each layer is independent and composable
- **Reusable intermediates**: forecasts feed downstream layers as intermediate products
- **Standard contracts**: consistent data schema across layers
- **Swappable models**: support alternative forecasting approaches
- **Transparent flow**: data moves through layers in clear progression

## Key Decisions

**Why 15 assets with 2 types:**
- Large enough to test clustering & model selection
- Two distinct patterns: predictable (EV) vs variable (solar)
- Realistic energy use cases
- Foundation for portfolio analysis

**Why 2-week window (14 days):**
- Captures full weeks (includes weekday/weekend patterns)
- Small enough for quick iteration
- Sufficient for seasonal/weekly features

## Reference

See `working_notes/main_idea.md` for detailed 15-section system design.
