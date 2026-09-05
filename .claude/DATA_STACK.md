# Data Stack & Technologies

## Polars (Not Pandas)

All data operations use **polars** exclusively:

- **Performance**: lazy evaluation, columnar storage
- **Efficiency**: memory-efficient, parquet-native I/O
- **Type preservation**: datetime, float64 precision
- **Future-proof**: supports all downstream operations

### Polars API

Common methods:
- `.filter()` — conditional selection
- `.select()` — column projection
- `.with_columns()` — add/modify columns
- `.group_by()` — aggregations
- `.write_parquet()` / `.read_parquet()` — file I/O

## Data Format Strategy

| Format | Location | Use Case | Tracked |
|--------|----------|----------|---------|
| CSV | `src/data/` | Intermediate raw output of generate_raw_data.py | Yes |
| Parquet | `src/data/` | Production, efficient, typed | Yes |

**Parquet advantages:**
- Compressed, columnar storage
- Discoverable schema (column names, types)
- Reproducible data contracts
- Enables incremental/parallel processing

## Data Generation Pipeline

Located: `src/data/`

### Flow

1. **generate_raw_data.py** → Creates synthetic metering
   - 14 days × 15 assets × 30-min intervals = 10,080 records
   - Seed=42 for reproducibility
   - Two asset types: EV charging, solar+battery
   - Output: `metering_data_raw.csv`

2. **generate_daily_metrics.py** → Analysis & export to parquet
   - Daily energy/peak/average
   - Ramp rates (30-min interval changes)
   - Behavioral metrics (coefficient of variation, intermittency, peak-to-avg ratio)
   - Output: `metering_data.parquet`, `daily_metrics.parquet`

3. **generate_metering_features.py** → Joins daily context onto the 30-min series
   - Reads `metering_data.parquet` + `daily_metrics.parquet`
   - Joins on (asset_id, date), broadcasting each daily row across its 48 half-hour rows
   - Daily columns are renamed with a `feat_` prefix to distinguish them from native 30-min columns
   - Output: `metering_data_with_features.parquet`

### Production Files (src/data/)

- `metering_data.parquet` — Raw 30-min metering (10,080 rows)
- `daily_metrics.parquet` — Daily aggregates + behavioral metrics (210 rows)
- `metering_data_with_features.parquet` — 30-min metering with daily features broadcast per day (10,080 rows)
