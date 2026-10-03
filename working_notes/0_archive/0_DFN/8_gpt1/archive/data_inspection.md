# Time Series Data Inspection

## Purpose

Establish a model-independent understanding of the time series before applying forecasting models.

## Initial checks

For the current two-week, 30-minute synthetic metering dataset:

* Inspect the overall distribution and range of the target.
* Examine the series over time for trends or changes in behaviour.
* Examine time-of-day patterns.
* Examine daily and weekly seasonality where observable.
* Examine autocorrelation using ACF.
* Examine partial autocorrelation using PACF where useful.
* Examine rolling mean and variance where useful.

## Current scope

The synthetic data allows the following to be deferred:

* Missing values and data integrity checks
* Timestamp validation
* Outlier detection and treatment
* Production data-quality checks

The focus is understanding the temporal structure relevant to short-term forecasting.

## Forecast horizon

The current forecasting problem is short term:

* 30-minute observations
* Approximately two weeks of data
* Approximately 8-hour forecast horizon
* 16 future observations per forecast
