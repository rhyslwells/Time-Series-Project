# Data Documentation

Schemas, dictionaries, and quality information.

## Data Dictionary
[Complete Dictionary](dictionary.md)

Column definitions, units, and value ranges.

## Data Quality
[Data Quality Assessment](quality.md)

Known issues, missing data patterns, and outliers.

## Sources
[Data Sources](sources.md)

Where data comes from and how it's updated.

---

## Quick Reference

### Metering Data Schema

| Field | Type | Description |
|-------|------|---|
| asset_id | string | Unique asset identifier |
| timestamp | datetime | UTC, 30-minute intervals |
| value | float | Consumption/generation in kW |
| quality_flag | int | 0=good, 1=estimated, 2=missing |

### Forecast Output Schema

| Field | Type | Description |
|-------|------|---|
| asset_id | string | Which asset |
| timestamp | datetime | Target time (UTC) |
| yhat | float | Point forecast (kW) |
| lower | float | 10th percentile |
| upper | float | 90th percentile |
| model_version | string | Model ID (e.g., "lightgbm_v2.1") |
| forecast_ts | datetime | When forecast was generated |

---

## Data Flows

```
Raw Metering Data (30-min intervals)
    |
    v
Data Validation & QA
    |
    v
Feature Engineering
    |
    v
Model Training & Scoring
    |
    v
Forecast Output
```
