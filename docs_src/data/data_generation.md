# Data Generation and Calculations

This document describes how synthetic metering data is generated and the calculations performed to produce analysis-ready datasets.

**Source:** `src/data/` scripts (generate_raw_data.py, generate_daily_metrics.py). For the third pipeline stage, which joins these daily metrics back onto the 30-minute series, see [Feature Engineering](feature_engineering.md).

## Generation Process

### Step 1: Synthetic Metering Generation (generate_raw_data.py)

Synthetic metering data is created with realistic behavioral patterns for 15 assets over 14 days (2025-01-01 to 2025-01-14) at 30-minute intervals.

**Parameters:**
- Seed: 42 (reproducible)
- Assets: 15 (8 EV charging + 7 solar+battery)
- Duration: 14 days
- Interval: 30 minutes
- Total records: 10,080 (15 × 14 × 48)

**Asset Types:**

1. **EV Charging (8 assets)**
   - Distinct daily peaks during charging hours
   - Weekday (Mon-Fri): 1.2x intensity multiplier
   - Weekend (Sat-Sun): 0.8x intensity multiplier
   - Peak times: Morning (06:00-09:00), Evening (16:00-21:00)

2. **Solar + Battery Storage (7 assets)**
   - Solar generation peak at midday
   - Evening discharge period
   - Net metering (generation - consumption - discharge)
   - Negative values represent export to grid

**Output:** `metering_data_raw.csv` (temporary working file)

---

## Analysis and Calculations

### Step 2: Daily Metrics Computation (generate_daily_metrics.py)

Raw metering data is aggregated and analyzed to produce daily-level metrics for each asset.

#### Energy Metrics

```
daily_energy_kwh = SUM(metering_kwh) over all 30-minute intervals in day
daily_peak_kw = MAX(metering_kwh) over all 30-minute intervals in day
daily_min_kw = MIN(metering_kwh) over all 30-minute intervals in day
daily_avg_kw = MEAN(metering_kwh) over all 30-minute intervals in day
daily_std_kw = STDEV(metering_kwh) over all 30-minute intervals in day
n_samples = COUNT of 30-minute intervals (48 per day)
```

#### Ramp Rate Analysis

Ramp rates measure how quickly power changes between consecutive 30-minute intervals, important for understanding asset flexibility constraints.

```
ramp_kw = metering_kwh[t] - metering_kwh[t-1]

Per day aggregation:
mean_ramp_kw = MEAN(|ramp_kw|) over all intervals in day
std_ramp_kw = STDEV(ramp_kw) over all intervals in day
max_abs_ramp_kw = MAX(|ramp_kw|) over all intervals in day
```

Positive ramp = increase in load/generation
Negative ramp = decrease in load/generation

#### Behavioral Metrics

Daily behavioral metrics characterize asset variability and predictability, enabling adaptive forecasting and model selection.

**Coefficient of Variation (CV):**
```
CV = STDEV(metering_kwh) / MEAN(metering_kwh)

Interpretation:
- CV < 0.5: Low variability (e.g., steady EV charging)
- CV 0.5-2.0: Moderate variability (e.g., daily peaks)
- CV > 2.0: High variability (e.g., solar generation)
```

**Peak-to-Average Ratio:**
```
peak_to_avg_ratio = daily_peak_kw / daily_avg_kw

Interpretation:
- Ratio ≈ 1.0: Flat, consistent load
- Ratio > 2.0: Pronounced peaks (e.g., EV charging or solar midday spike)
- Ratio >> 1.0: Highly variable demand patterns
```

**Intermittency Ratio:**
```
intermittency_ratio = COUNT(metering_kwh == 0) / total_intervals

Interpretation:
- 0.0: Continuous operation
- > 0.0: Asset offline during some periods
- Used to identify assets with scheduled downtime or intermittent operation
```

---

## Output Data

### metering_data.parquet (Raw Time Series)

**Schema:**
```
asset_id: string                    # ASSET_001 to ASSET_015
timestamp: datetime                 # UTC, 30-minute intervals
metering_kwh: float                 # Consumption (+) or generation (-)
asset_type: string                  # 'ev_charging' or 'solar_battery'
```

**Size:** 10,080 rows (15 assets × 14 days × 48 periods)

**Usage:**
- Base dataset for all forecasting models
- Input to feature engineering pipelines
- Validation against forecasts

---

### daily_metrics.parquet (Daily Aggregates + Behavioral Features)

**Schema:**
```
asset_id: string                    # Asset identifier
date: date                          # Calendar date
daily_energy_kwh: float             # Total daily consumption/generation
daily_peak_kw: float                # Maximum 30-min value in day
daily_min_kw: float                 # Minimum 30-min value in day
daily_avg_kw: float                 # Mean 30-min value in day
daily_std_kw: float                 # Std dev of 30-min values
n_samples: int                      # Count (48 for complete days)
mean_ramp_kw: float                 # Average absolute change between intervals
std_ramp_kw: float                  # Std dev of ramp changes
max_abs_ramp_kw: float              # Largest change between any two intervals
coefficient_of_variation: float     # Daily variability metric (std/mean)
intermittency_ratio: float          # Fraction of zero-reading intervals
peak_to_avg_ratio: float            # Peak load vs average load
```

**Size:** 210 rows (15 assets × 14 days)

**Single Source of Truth:** All daily-level analysis derives from this table.

**Usage:**
- Asset characterization and behavioral profiling
- Model selection based on variability patterns
- Operational constraint validation (ramp limits)
- Time-series forecasting with lagged behavioral features
- Trend analysis and drift detection

---

## Key Design Decisions

### Why Consolidate into daily_metrics?

Previously, behavioral metrics and ramp rates were stored separately (15 rows each, asset-level aggregates). Consolidating into daily_metrics enables:

1. **Temporal variation capture:** Fingerprints change day-to-day, not static per asset
2. **Adaptive forecasting:** Yesterday's coefficient_of_variation informs today's uncertainty bands
3. **Single fact table:** Simplifies data contracts and downstream logic
4. **Drift detection:** Time-series of behavioral metrics reveals asset state changes

---

## Next Step

`daily_metrics.parquet` is joined back onto the 30-minute series to produce a model-ready
table with per-interval and daily-context features side by side. See
[Feature Engineering](feature_engineering.md).

