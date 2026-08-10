# Methodology

Documented approaches and processes for the forecasting system.

## Core Methodologies

### Feature Engineering
[Feature Engineering Guide](feature_engineering.md)

- How we construct meaningful features from raw metering data.
- See [Data Generation and Calculations](../data/data_generation.md)

### Model Selection
[Model Selection Strategy - (Not yet implemented)](model_selection.md)

How we choose which forecasting model to use for each asset.

### Validation & Evaluation
[Validation Approach - (Not yet implemented)](validation.md)

How we assess model performance and avoid overfitting.

---

## Decision Log

This section documents why we chose certain approaches.

### Why Standard Forecast Contract?
- **Problem**: Different models output different schemas
- **Solution**: Standardize on `(asset_id, timestamp, yhat, lower, upper, model_version)`
- **Benefit**: Models become swappable without downstream changes

### Why Asset Profiling First?
- **Problem**: Single model doesn't work for all assets
- **Solution**: Classify assets, use targeted models per class
- **Benefit**: Better accuracy, clearer model assignment logic

### Why Prediction Intervals Over Point Forecasts?
- **Problem**: Users need to know forecast reliability
- **Solution**: Always provide lower and upper bounds
- **Benefit**: Enables risk-aware decision making in optimization

---

## Related Resources

- [Architecture Overview](../architecture.md) - System design
- [Exploration Notebooks](../notebooks.md) - How we validated these approaches
- [Findings](../findings/index.md) - What we discovered
