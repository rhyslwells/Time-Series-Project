# DRAFTING: Asset Profiling Analysis

## Summary

Assets can be classified into behavioral clusters based on time series characteristics. This enables targeted forecasting strategies per asset type.

## Key Findings

1. Assets split into 3-4 distinct behavioral clusters
2. Cluster membership predicts optimal forecast model
3. Seasonality strength is the primary discriminator
4. Intermittency and autocorrelation further refine classification

## Characteristics Analyzed

- Mean load
- Variance and coefficient of variation
- Seasonality strength (seasonal subseries decomposition)
- Autocorrelation (ACF lag-1)
- Intermittency ratio (% near-zero values)
- Peak-to-average ratio
- Ramp frequency and magnitude

## Clusters Identified

| Cluster | Name | Characteristics | Best Model |
|---------|------|---|---|
| A | Stable | Low variance, high seasonality | SARIMA |
| B | Variable | High variance, weak seasonality | GBDT |
| C | Intermittent | Low mean, many zeros, sporadic | Quantile Regression |

## Next Steps

- Validate cluster stability over time
- Test model-to-cluster mapping in production
- Monitor cluster membership changes
- Develop adaptive model selection

## Related

- Exploration notebook: [asset_profiling_exploration.md](../../asset_profiling_exploration.md)
- Methodology: [Feature Engineering](../methodology/feature_engineering.md)
- See [Model Selection](../methodology/model_selection.md) for cluster-based strategies
