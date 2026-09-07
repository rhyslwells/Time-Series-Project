# Data generation and calculations

This page covers the generation process and the formulas. For what the resulting data looks
like and its fitness for use, see [Synthetic metering data](synthetic_metering_data.md).

**Source:** `src/data/` scripts (generate_raw_data.py, generate_daily_metrics.py). For the third pipeline stage, which joins these daily metrics back onto the 30-minute series, see [Feature Engineering](feature_engineering.md).

## Pipeline at a glance

```mermaid
flowchart LR
    A[generate_raw_data.py] -->|metering_data_raw.csv| B[generate_daily_metrics.py]
    B -->|metering_data.parquet| C[generate_metering_features.py]
    B -->|daily_metrics.parquet| C
    C -->|metering_data_with_features.parquet| D[Forecasting]
```

All three scripts and their outputs live in `src/data/`. Every intermediate file is
committed to the repository, since its schema is a cross-layer data contract.

## Generation process

### Step 1: synthetic metering generation (generate_raw_data.py)

Synthetic metering data is created with realistic behavioral patterns for 15 assets over 14 days (2025-01-01 to 2025-01-14) at 30-minute intervals.

**Parameters:**
- Seed: 42 (reproducible)
- Assets: 15 (8 EV charging + 7 solar+battery)
- Duration: 14 days
- Interval: 30 minutes
- Total records: 10,080 (15 × 14 × 48)

**Asset Types:**

1. **EV Charging (8 assets)** — additive components on a 0.5 base load:
   - Morning peak (06:00-09:00): +2.5
   - Evening peak (16:00-21:00): +3.0
   - Night charging (22:00-02:00): +1.5
   - Whole pattern then multiplied by a weekday factor: 1.2 (Mon-Fri) or 0.8 (Sat-Sun)
   - Gaussian noise added, std = 10% of the mean positive pattern value; result clipped at 0

2. **Solar + Battery Storage (7 assets)** — net metering (generation - consumption - discharge):
   - Solar generation: half-sine over 06:00-18:00, peak 2.5 at local noon
   - Consumption: `0.8 + 0.3 sin(2π hour / 24)`, then scaled by a weekday factor: 1.0 (Mon-Fri) or 0.85 (Sat-Sun)
   - Battery discharge: -0.5 over 18:00-22:00
   - Gaussian noise added, std = 8% of (max solar + 0.5); values are **not** clipped, so negatives occur
   - Negative values represent net export to grid

**Output:** `metering_data_raw.csv` — a committed intermediate file (the input to Step 2), not a scratch file.

---

## Analysis and calculations

### Step 2: daily metrics computation (generate_daily_metrics.py)

Raw metering data is aggregated and analyzed to produce daily-level metrics for each asset.

#### Energy metrics

```
daily_energy_kwh = SUM(metering_kwh) over all 30-minute intervals in day
daily_peak_kw = MAX(metering_kwh) over all 30-minute intervals in day
daily_min_kw = MIN(metering_kwh) over all 30-minute intervals in day
daily_avg_kw = MEAN(metering_kwh) over all 30-minute intervals in day
daily_std_kw = STDEV(metering_kwh) over all 30-minute intervals in day
n_samples = COUNT of 30-minute intervals (48 per day)
```

#### Ramp rate analysis

Ramp rates measure how quickly power changes between consecutive 30-minute intervals, important for understanding asset flexibility constraints.

```
ramp_kw = metering_kwh[t] - metering_kwh[t-1]

Per day aggregation:
mean_ramp_kw = MEAN(ramp_kw) over all intervals in day        # signed, not absolute
std_ramp_kw = STDEV(ramp_kw) over all intervals in day
max_abs_ramp_kw = MAX(|ramp_kw|) over all intervals in day
```

Positive ramp = increase in load/generation
Negative ramp = decrease in load/generation

!!! note "mean_ramp_kw is the signed mean"
    `generate_daily_metrics.py:35` computes `pl.col('ramp_kw').mean()` — the signed mean, not
    the mean absolute change. For a series that returns to a similar level each day the signed
    mean sits near zero, so **`max_abs_ramp_kw` is the ramp-magnitude column**, not this one.
    See [Known issues](#known-issues).

#### Behavioral metrics

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

## Output data

### metering_data.parquet (raw time series)

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

### daily_metrics.parquet - daily aggregates and behavioral features

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
mean_ramp_kw: float                 # Signed mean change between intervals (near zero for level-reverting series)
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

## Key design decisions

### Why consolidate into daily_metrics?

Previously, behavioral metrics and ramp rates were stored separately (15 rows each, asset-level aggregates). Consolidating into daily_metrics enables:

1. **Temporal variation capture:** Fingerprints change day-to-day, not static per asset
2. **Adaptive forecasting:** Yesterday's coefficient_of_variation informs today's uncertainty bands
3. **Single fact table:** Simplifies data contracts and downstream logic
4. **Drift detection:** Time-series of behavioral metrics reveals asset state changes

---

## Next step

`daily_metrics.parquet` is joined back onto the 30-minute series to produce a model-ready
table with per-interval and daily-context features side by side. See
[Feature Engineering](feature_engineering.md).

---

## Known issues

Open items surfaced during a documentation review. Both are intentional-for-now: the docs
describe the data as it actually is, and neither blocks use.

| Location | Issue | Effect |
|----------|-------|--------|
| `generate_daily_metrics.py:35` | `mean_ramp_kw` uses `.mean()` (signed), while its schema gloss and downstream naming imply mean-absolute | Column is near zero for level-reverting assets; easy to misread as "no ramps". `max_abs_ramp_kw` is the magnitude column |
| `daily_metrics.parquet` column names | `daily_peak_kw` / `daily_min_kw` / `daily_avg_kw` / `daily_std_kw` are statistics of a kWh-per-30-min quantity, not kW | `_kw` suffix implies power; a true kW value is 2× the stored number. Rename to `_kwh_hh` or similar deferred |

