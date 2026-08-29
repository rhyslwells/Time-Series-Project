# System Architecture

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
