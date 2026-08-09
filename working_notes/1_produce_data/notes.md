# Data Generation and Inspection

This folder contains scripts to generate and analyze synthetic metering data for the time series forecasting system. See [main_idea.md](../main_idea.md) for the system architecture overview.

## Overview

We generate synthetic metering data with realistic behavioral patterns to support forecasting and asset analysis:

- **15 assets** with unique identifiers (ASSET_001 to ASSET_015)
- **2 weeks** of continuous metering data (2025-01-01 to 2025-01-14)
- **30-minute intervals** (2 measurements per hour, 1,344 rows per asset, 10,080 total records)
- **No missing values**, with realistic noise patterns
- **Values in metering_kwh** (kilowatt-hours per 30-minute period)

## Asset Types

Two asset behavioral patterns are included, derived from real energy system use cases:

### 1. EV Charging Stations (8 assets)
- **Pattern:** Load drop with distinct daily peaks
- **Characteristics:**
  - Morning peak (06:00-09:00): 2.5 kW average
  - Evening peak (16:00-21:00): 3.0 kW average
  - Night charging (22:00-02:00): 1.5 kW average
  - Weekday (Mon-Fri): 1.2x multiplier
  - Weekend (Sat-Sun): 0.8x multiplier
- **Behavioral metrics:**
  - Mean load: 1.83 kW
  - Coefficient of variation: 0.79 (moderate predictability)
  - Peak-to-average ratio: 2.23
  - Autocorrelation: 0.66 (strong daily patterns)
  - Intermittency: minimal

### 2. Solar + Battery Storage (7 assets)
- **Pattern:** Net metering with solar generation and evening discharge
- **Characteristics:**
  - Solar generation peak (06:00-18:00): 2.5 kW peak at noon
  - Battery discharge (18:00-22:00): 0.5 kW average
  - Baseline consumption: 0.8 kW
  - Net metering: generation - consumption - discharge
- **Behavioral metrics:**
  - Mean load: 0.026 kW (net, mostly exporting)
  - Coefficient of variation: 45.5 (highly variable)
  - Peak-to-average ratio: 88.8 (extreme variation)
  - Autocorrelation: 0.90 (strong solar correlation)
  - Intermittency: none (continuous operation)

## Scripts

### `generate_data.py`
Generates synthetic metering data with realistic patterns based on asset type.

**Dependencies:** numpy, polars, datetime

**Output:** `metering_data_raw.csv` (10,080 rows × 4 columns)

**Columns:**
- `asset_id` (str): Unique asset identifier (ASSET_001, etc.)
- `timestamp` (datetime): 30-minute intervals
- `metering_kwh` (float): Consumption/generation in kWh
- `asset_type` (str): 'ev_charging' or 'solar_battery'

**Usage:**
```bash
python generate_data.py
```

### `inspect_data.py`
Analyzes the generated data and computes behavioral fingerprints for each asset.

**Dependencies:** polars, numpy, scipy, pathlib

**Features:**
- Daily energy, peak, and minimum calculations
- Ramp rate analysis (30-minute interval changes)
- Autocorrelation and seasonality metrics
- Behavioral fingerprinting (mean load, variance, peak-to-average ratio, intermittency)
- Asset clustering analysis by type
- Automatic parquet export to `src/data/`

**Usage:**
```bash
python inspect_data.py
```

**Output:** Terminal report with summary statistics and clustering insights

## Data Flow

```
generate_data.py
      │
      ├─ Seed: 42 (reproducible)
      └─> metering_data_raw.csv (working directory, 10,080 records)
            │
            ↓
inspect_data.py
            │
            ├─ Compute daily metrics (14 days × 15 assets = 210 rows)
            ├─ Compute ramp rates (1 row per asset = 15 rows)
            ├─ Compute behavioral fingerprints (1 row per asset = 15 rows)
            │
            ├─ Print terminal reports
            │   ├─ Data summary
            │   ├─ Asset list
            │   ├─ Daily metrics
            │   ├─ Ramp rates
            │   ├─ Behavioral fingerprints
            │   └─ Clustering analysis
            │
            └─> Save to src/data/ as parquet:
                ├─ metering_data.parquet (10,080 rows)
                ├─ daily_metrics.parquet (210 rows)
                ├─ ramp_rates.parquet (15 rows)
                └─ behavioral_fingerprints.parquet (15 rows)
```

## Output Files (src/data/)

Production-ready parquet files for model development:

- **metering_data.parquet**
  - Schema: asset_id, timestamp, metering_kwh, asset_type
  - 10,080 rows (15 assets × 14 days × 48 half-hourly periods)
  - Index: asset_id, timestamp

- **daily_metrics.parquet**
  - Schema: asset_id, date, daily_energy_kwh, daily_peak_kw, daily_min_kw, daily_avg_kw, daily_std_kw, n_samples
  - 210 rows (15 assets × 14 days)
  - Supports daily aggregation analysis

- **ramp_rates.parquet**
  - Schema: asset_id, mean_ramp_kw, std_ramp_kw, max_ramp_kw, mean_abs_ramp_kw, max_abs_ramp_kw
  - 15 rows (one per asset)
  - Supports operational constraints (ramp limits)

- **behavioral_fingerprints.parquet**
  - Schema: asset_id, asset_type, mean_load_kw, variance, coefficient_of_variation, autocorrelation, intermittency_ratio, peak_to_avg_ratio
  - 15 rows (one per asset)
  - Enables asset clustering and model selection

## Data Quality Observations

**Completeness:**
- No missing values (0 NaNs)
- No data gaps (continuous 30-minute intervals)
- All 15 assets have 1,344 complete records

**Value Distribution:**
- 2,767 negative values (18.2% of records)
  - Expected: solar_battery assets during generation periods
  - Represents net export to grid
  - Not an error; reflects realistic net metering

**Temporal Coverage:**
- 14 days × 24 hours × 4 periods = 1,344 records per asset
- Covers 2 full weeks (Mon-Sun × 2)
- Includes weekday/weekend patterns

## Key Findings & Implications

### Asset Clustering
Clear separation enables automatic model selection:
- **EV Charging cluster:** Predictable, regular peaks (low CV, low ACF)
- **Solar+Battery cluster:** Highly variable, weather-dependent (high CV, high ACF)

### Forecasting Strategy
- EV assets: suitable for SARIMA, exponential smoothing (strong seasonality)
- Solar assets: requires external variables (irradiance, temperature) or ensemble methods

### Uncertainty Quantification
- EV charging: lower confidence needed (CV 0.79)
- Solar+Battery: higher uncertainty bands needed (CV 45.5)

## Integration Points

This data pipeline feeds into:
1. **Forecasting models** (src/forecasting/) — use metering_data.parquet
2. **Model selection** (src/models/) — use behavioral_fingerprints.parquet for asset clustering
3. **Validation** (src/evaluation/) — use daily_metrics.parquet for benchmark
4. **Flexibility analysis** — use ramp_rates.parquet for operational constraints

## Technical Decisions

**Polars over Pandas:**
- Used polars for all data operations (generate_data.py, inspect_data.py)
- Reasons: performance, memory efficiency, lazy evaluation support
- All scripts use polars API (`.group_by()`, `.with_columns()`, `.write_parquet()`)

**Synthetic Data:**
- Deterministic patterns (seed=42) for reproducibility
- Added realistic noise to make data non-trivial
- Covers key forecasting challenges (seasonality, variability, net metering)

**Parquet Format:**
- Efficient columnar storage (smaller files than CSV)
- Preserves data types (datetime, float precision)
- Supports lazy loading for large datasets
- Industry standard for data pipelines

## Next Steps Ideas

1. Load parquet files from `src/data/` in forecasting models
2. Use behavioral_fingerprints for asset-level model selection
3. Build baseline forecasts against metering_data
4. Validate forecasts with daily_metrics (daily aggregation should align)
5. Test ramp-rate constraints in optimization