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
| CSV | `working_notes/` | Human-readable, temporary | No |
| Parquet | `src/data/` | Production, efficient, typed | Yes |

**Parquet advantages:**
- Compressed, columnar storage
- Discoverable schema (column names, types)
- Reproducible data contracts
- Enables incremental/parallel processing

## Data Generation Pipeline

Located: `working_notes/1_produce_data/`

### Flow

1. **generate_data.py** → Creates synthetic metering
   - 14 days × 15 assets × 30-min intervals = 10,080 records
   - Seed=42 for reproducibility
   - Two asset types: EV charging, solar+battery
   - Output: `metering_data_raw.csv`

2. **inspect_data.py** → Analysis & export to parquet
   - Daily metrics (energy, peak, average)
   - Ramp rates (30-min interval changes)
   - Behavioral fingerprints (clustering features)
   - Output: 4 parquet files in `src/data/`

### Production Files (src/data/)

- `metering_data.parquet` — Raw 30-min metering (10,080 rows)
- `daily_metrics.parquet` — Daily aggregates (210 rows)
- `ramp_rates.parquet` — Asset ramp statistics (15 rows)
- `behavioral_fingerprints.parquet` — Asset features for model selection (15 rows)
