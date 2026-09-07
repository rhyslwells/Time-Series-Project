# Findings

This section is deliberately small. It holds only what the exploration has actually
established; work that is still a proposal lives in [Theory](../theory/index.md) or in the
notebooks, not here.

## Status

| Topic | Status | Where |
|---|---|---|
| SARIMA forecasting pipeline works end to end for one asset | Demonstrated on a single split | [notebook](../notebooks/sarima.md) |
| Three model classes run and can be compared on one split | Demonstrated; ranking not yet validated | [notebook](../notebooks/ts_model_explorer.md) |
| Asset behavioural clustering | Proposed scheme only — no clusters computed | [Asset profiling](asset_profiling.md) |
| Rolling-origin / walk-forward evaluation | Not started | — |
| Seasonal-naive baseline and skill scores (MASE) | Not started | [Metrics](../theory/metrics.md#read-metrics-against-a-baseline) |
| Uncertainty calibration on this data | Not started | [Diagnostics](../theory/diagnostics.md) |

## What "demonstrated" means here

The two notebooks each run one asset through one fixed 10-day / 4-day split. That is enough
to exercise the framework and the output contract, but not enough to rank the models or to
trust the reported metrics — see the caveats in [Models](../theory/models.md) and
[Framework usage](../coding/framework_usage.md#tuning). Treat this section as a record of
early-stage progress, not settled results.
