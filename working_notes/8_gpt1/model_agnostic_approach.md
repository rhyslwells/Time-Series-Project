# Model-Agnostic Forecasting Approach

## Principle

The forecasting workflow should remain model-agnostic wherever practical.

The raw time series should be inspected independently of the forecasting model, and model-specific requirements should remain within the individual model implementations.

## Avoid unnecessary transformations

Do not transform the underlying series simply to satisfy the assumptions of a particular forecasting model.

In particular, stationarity should not be treated as a prerequisite for beginning the forecasting process. Differencing, transformations, or other preprocessing should only be introduced later if there is a specific modelling reason to do so.

For the current short-term forecasting problem, use the raw metering values directly.

## Model interchangeability

The workflow should allow different forecasting approaches to be substituted without redesigning the preceding data-inspection process.

For example:

* SARIMA
* LightGBM
* Other statistical or machine-learning forecasting models

Each model can handle the characteristics of the series according to its own methodology, while the overall forecasting problem and evaluation framework remain consistent.

## Current focus

The current test case is deliberately narrow:

* Two weeks of synthetic 30-minute data
* Approximately 8-hour forecast horizon
* 16 future half-hour observations

More model-specific preprocessing and transformation decisions can be considered later if they become necessary.
