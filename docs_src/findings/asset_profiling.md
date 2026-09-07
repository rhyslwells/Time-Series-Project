# Asset profiling

!!! warning "Proposed — not implemented"
    No clusters have been computed. Nothing in the repo runs a clustering step, and the
    table below is a proposed scheme, not a result. This page records the intended approach
    so it is not lost.

## Idea

Assets could be grouped into behavioural clusters from their time-series characteristics,
and each cluster routed to the model class that suits it — cheaper and more principled than
grid-searching every asset individually. The per-asset inputs this would build on already
exist in [`daily_metrics.parquet`](../data/data_generation.md).

## Characteristics to analyse

- Mean load
- Variance and coefficient of variation
- Seasonality strength — needs a definition; the standard is
  $F_S = \max(0,\, 1 - \operatorname{Var}(R)/\operatorname{Var}(S+R))$ from an STL decomposition
- Autocorrelation (ACF lag-1)
- Intermittency ratio (fraction of near-zero values)
- Peak-to-average ratio
- Ramp frequency and magnitude

## Proposed cluster scheme (illustrative)

| Cluster | Characteristics | Candidate model |
|---------|-----------------|-----------------|
| Stable | Low variance, high seasonality strength | SARIMA |
| Variable | High variance, weak seasonality | LightGBM |
| Intermittent | Low mean, many near-zero values | Quantile regression |

With only two asset types in the current dataset (`ev_charging`, `solar_battery`) and 14
days of history, this scheme cannot yet be tested meaningfully.

## Next steps

- Compute the characteristics above from `daily_metrics.parquet`
- Cluster and inspect stability over the 14 days
- Only then compare a cluster-routed model choice against per-asset selection

## Related

- [Feature engineering](../data/feature_engineering.md) — how the daily metrics are built
- [Forecast products](../theory/forecast_products.md#cross-asset-comparison-and-asset-classification) — where this fits in the layer model
