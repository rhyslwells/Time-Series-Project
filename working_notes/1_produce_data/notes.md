# Data Generation and Inspection

Scripts to generate and analyze synthetic metering data for the forecasting system.

See [main_idea.md](../main_idea.md) for system architecture.

**Detailed documentation:** [docs_src/data/](../../docs_src/data/)
- [Synthetic Metering Data Description](../../docs_src/data/synthetic_metering_data.md) — asset types, patterns, behavioral metrics
- [Data Generation and Calculations](../../docs_src/data/data_generation.md) — generation process, formulas, schema

---

## Quick Start

```bash
# Generate synthetic data
python generate_data.py
# Creates: metering_data_raw.csv (10,080 rows)

# Analyze and export
python inspect_data.py
# Creates: src/data/metering_data.parquet, src/data/daily_metrics.parquet
```

---

## Scripts

### `generate_data.py`

Creates synthetic metering data: 15 assets, 14 days, 30-minute intervals.

**Output:** `metering_data_raw.csv`
```
asset_id, timestamp, metering_kwh, asset_type
```

**Parameters:**
- Seed: 42 (reproducible)
- Assets: 8 EV charging + 7 solar+battery

### `inspect_data.py`

Analyzes raw data and computes daily metrics.

**Outputs:** 
- `src/data/metering_data.parquet` — Raw time series (10,080 rows)
- `src/data/daily_metrics.parquet` — Daily aggregates + behavioral features (210 rows)

**Computed metrics:**
- Energy: daily_energy_kwh, daily_peak_kw, daily_min_kw, daily_avg_kw, daily_std_kw
- Ramps: mean_ramp_kw, std_ramp_kw, max_abs_ramp_kw
- Behavior: coefficient_of_variation, intermittency_ratio, peak_to_avg_ratio

---

## Data Pipeline

```
generate_data.py  →  metering_data_raw.csv
                             ↓
                       inspect_data.py
                             ↓
                    src/data/metering_data.parquet
                    src/data/daily_metrics.parquet
```

---

## Output Files

See [Data Generation and Calculations](../../docs_src/data/data_generation.md#output-data) for detailed schema.

| File | Rows | Purpose |
|------|------|---------|
| metering_data.parquet | 10,080 | 30-min time series for forecasting |
| daily_metrics.parquet | 210 | Daily aggregates + behavioral metrics |

---

## Key Features

- **Reproducible:** Seed=42, identical data across runs
- **Consolidated:** All daily analysis in single daily_metrics table
- **Behavioral:** Daily CV, intermittency, peak-to-avg ratio enable adaptive models
- **Operationally useful:** Ramp rates inform flexibility constraints

