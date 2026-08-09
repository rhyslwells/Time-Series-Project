# Data Generation and Inspection

This folder contains scripts to generate and analyze synthetic metering data for the time series forecasting system. See [main_idea.md](../main_idea.md) for the system architecture.

## Overview

We generate synthetic metering data with realistic behavioral patterns to support forecasting and asset analysis:

- **15 assets** with unique identifiers
- **2 weeks** of continuous metering data (2025-01-01 to 2025-01-14)
- **30-minute intervals** (4 measurements per hour, 1,344 rows per asset, 20,160 total)
- **No missing values**, with realistic noise
- **Values in metering_kwh** (kilowatt-hours per 30-minute period)

## Asset Types

Two asset behavioral patterns are included:

1. **EV Charging Stations** (8 assets) — Load drop pattern
   - Morning peak (06:00-09:00)
   - Evening peak (16:00-21:00)  
   - Night charging (22:00-02:00)
   - Weekday/weekend variations

2. **Solar + Battery** (7 assets) — Generation and storage pattern
   - Solar generation peak (06:00-18:00)
   - Battery discharge in evening (18:00-22:00)
   - Daily consumption baseline

## Scripts

### `generate_data.py`
Generates synthetic metering data with realistic patterns based on asset type.

**Output:** `metering_data_raw.csv`

```bash
python generate_data.py
```

### `inspect_data.py`
Analyzes the generated data and computes behavioral fingerprints for each asset.

**Features:**
- Daily energy, peak, and minimum calculations
- Ramp rate analysis (30-minute interval changes)
- Autocorrelation and seasonality metrics
- Behavioral fingerprinting (mean load, variance, peak-to-average ratio, intermittency)
- Asset clustering analysis by type

**Usage:**
```bash
python inspect_data.py
```

**Output:** Terminal report with summary statistics and clustering insights

## Data Flow

```
generate_data.py
      ↓
metering_data_raw.csv
      ↓
inspect_data.py
      ↓
Terminal report (metrics, fingerprints, validation)
      ↓
Later: Save as Parquet in src/ for production use
```

## Next Steps

Once validated:
1. Convert `metering_data_raw.csv` to Parquet format
2. Move to `src/data/` for production use
3. Build forecasting models against this dataset