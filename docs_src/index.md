# Time series forecasting

A layered forecasting system for energy assets: raw metering → forecasts → derived features
→ flexibility → optimization. Layers 1-2 (data and per-asset forecasting) are built; the
upper layers are design, described in [Forecast products](theory/forecast_products.md).

## What exists today

- A synthetic dataset: 15 assets (EV charging and solar+battery), 14 days, 30-minute intervals, in three parquet files. See [Data](data/index.md).
- A forecasting framework in `src/ts_model_framework.py`: `SARIMAModel`, `ExponentialSmoothingModel`, `LightGBMModel`, plus comparison, tuning and evaluation. See [Coding](coding/index.md).
- Diagnostic plots in `src/ts_plots.py` and a reading guide in [Diagnostics](theory/diagnostics.md).
- Two exploration notebooks running one asset through the framework. See [Notebooks](notebooks/notebooks.md).

Not yet built: rolling-origin evaluation, a seasonal-naive baseline, asset clustering, and
everything above the per-asset forecast. [Findings](findings/index.md) tracks the honest status.

## Reading path

1. **[Data](data/index.md)** — what the dataset is, how it is generated, what each column means.
2. **[Theory](theory/index.md)** — metrics, models, diagnostics, and the model-selection logic.
3. **[Coding](coding/index.md)** — how to call the framework, starting from loading the parquet data.
4. **[Notebooks](notebooks/notebooks.md)** — the same workflow, interactive.
5. **[Findings](findings/index.md)** — what has actually been established so far.

## Find your task

| Task | Page |
|---|---|
| Load the data and run a forecast | [Framework usage](coding/framework_usage.md#from-parquet-to-numpy) |
| Call the framework classes | [Framework usage](coding/framework_usage.md) |
| Pick a metric / read it against a baseline | [Metrics](theory/metrics.md) |
| Choose between SARIMA / ExpSmoothing / LightGBM | [Models](theory/models.md), [Decisions](theory/models-decisions.md) |
| Read a diagnostic plot | [Diagnostics](theory/diagnostics.md) |
| Understand the column units (kW vs kWh) | [Data](data/index.md#units) |
