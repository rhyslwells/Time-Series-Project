# Data

Schemas, units, quality notes, and the generation pipeline for the synthetic metering dataset.

## Find an answer fast

| Question | Page |
|---|---|
| "What do the assets actually look like — patterns, ranges, fitness for use?" | [Synthetic metering data](synthetic_metering_data.md) |
| "How is the data generated? What are the formulas?" | [Data generation](data_generation.md) |
| "What does each `daily_metrics` column mean?" | [Data generation](data_generation.md#daily_metricsparquet-daily-aggregates-and-behavioral-features) |
| "How do the daily metrics get onto the 30-minute series?" | [Feature engineering](feature_engineering.md) |
| "kW or kWh?" | [Units](#units), below |
| "Are there known bugs in the generators?" | [Known issues](data_generation.md#known-issues) |

## The pipeline at a glance

| Stage | Script | Output | Rows |
|---|---|---|---|
| 1 | `generate_raw_data.py` | `metering_data_raw.csv` | 10,080 |
| 2 | `generate_daily_metrics.py` | `metering_data.parquet` (raw series) | 10,080 |
| 2 | `generate_daily_metrics.py` | `daily_metrics.parquet` (daily aggregates + behavioral metrics) | 210 |
| 3 | `generate_metering_features.py` | `metering_data_with_features.parquet` (series + `feat_`-prefixed daily context) | 10,080 |

All scripts and outputs live in `src/data/`. Every intermediate file is committed — its
schema is a cross-layer data contract.

## Units

`metering_kwh` is **energy in kWh delivered (positive) or exported (negative) during each
30-minute interval**. It is not an instantaneous power reading.

The `daily_metrics.parquet` columns named `daily_peak_kw`, `daily_min_kw`, `daily_avg_kw`,
`daily_std_kw` are the max / min / mean / std of that same kWh-per-30-min quantity. **Their
`_kw` suffix is misleading**: an actual power figure would be twice the half-hourly energy
value (2 × kWh per half-hour = kW). Read them as "kWh per 30 min", the same unit as
`metering_kwh`. Renaming them is noted in [Known issues](data_generation.md#known-issues).

`daily_energy_kwh` is a genuine kWh total (the sum over the day).
