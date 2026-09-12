# Forecasting approach

## Principle

The forecasting workflow stays model-agnostic wherever practical. The raw series is
inspected independently of any forecasting model — see [Data inspection](../data/data_inspection.md)
— and model-specific requirements (a seasonal period, a stationarity assumption, an error
distribution) stay inside the individual model implementations rather than driving a shared
preprocessing step.

## Avoid unnecessary transformations

Do not transform the underlying series just to satisfy the assumptions of one particular
model. In particular, stationarity is not treated as a prerequisite for starting the
forecasting process: differencing, transformations, or other preprocessing are introduced
later, and only if a specific model needs them.

For the current short-term forecasting problem, models consume the raw metering values
directly.

## Model interchangeability

Different forecasting approaches — [SARIMA, Exponential Smoothing, LightGBM](models.md), or
others — should be substitutable without redesigning the data-inspection step that precedes
them. Each model handles the series' characteristics (seasonality, stationarity, scale)
according to its own methodology; the forecasting problem and evaluation framework
([Decisions](models-decisions.md)) stay the same regardless of which model is in use.

## Current focus

The current test case is deliberately narrow:

- Two weeks of synthetic 30-minute data
- ~8-hour forecast horizon (16 future half-hour observations)

More model-specific preprocessing and transformation decisions are added later, only if a
specific model's diagnostics call for them.
