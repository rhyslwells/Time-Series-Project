# Synthetic Metering Data Description

This document describes the characteristics, patterns, and behavior of the synthetic metering dataset (src\data\metering_data.parquet) used for model development and testing.

## Dataset Overview

**Purpose:** Provide realistic, reproducible metering data for developing and validating energy forecasting systems.

**Temporal Coverage:**
- Start: 2025-01-01 (Wednesday)
- End: 2025-01-14 (Tuesday)
- Duration: 14 days (2 full weeks)
- Frequency: 30-minute intervals
- Total records: 10,080 (15 assets × 14 days × 48 intervals)

**Assets:**
- Count: 15
- Types: 2 (EV charging, Solar+Battery storage)
- Time series per asset: 1,344 complete records

---

## Asset Types and Behavioral Patterns

### EV Charging Stations (8 assets)

**Characteristics:**
- Deterministic, predictable daily patterns
- Strong weekday/weekend separation
- Two main charging windows: morning and evening

**Hourly Pattern (Weekday Base):**
```
06:00 - 09:00: Morning charging peak    (2.5 kW avg, 1.2x multiplier)
09:00 - 16:00: Daytime taper            (0.5 kW avg)
16:00 - 21:00: Evening peak             (3.0 kW avg, 1.2x multiplier)
21:00 - 02:00: Night charging           (1.5 kW avg)
02:00 - 06:00: Overnight minimal        (0.1 kW avg)
```

**Weekday/Weekend Variation:**
- Weekday (Mon-Fri): 1.2× intensity (higher charging demand)
- Weekend (Sat-Sun): 0.8× intensity (lower, more distributed charging)

**Behavioral Metrics (Typical):**
- Mean load: 1.83 kW
- Coefficient of variation: 0.79 (moderate predictability)
- Peak-to-average ratio: 2.23 (pronounced peaks)
- Intermittency: minimal (rarely zero)
- Daily variability: Low (similar patterns day-to-day)

**Forecasting Suitability:**
- Strong seasonality (daily patterns)
- Low uncertainty (CV < 1.0)
- Suitable for: SARIMA, exponential smoothing, gradient boosting

---

### Solar + Battery Storage (7 assets)

**Characteristics:**
- Generation-led, net metering (exports to grid)
- Strong solar correlation
- Highly variable, weather-dependent
- Evening battery discharge supports grid

**Daily Pattern (Clear Sky Base):**
```
00:00 - 06:00: Minimal generation        (-0.2 kW avg, base load only)
06:00 - 09:00: Solar ramp-up             (0 to 2.0 kW)
09:00 - 12:00: Increasing generation    (2.0 to 2.5 kW)
12:00 - 15:00: Peak generation          (2.5 kW, export active)
15:00 - 18:00: Afternoon decline         (2.5 to 1.0 kW)
18:00 - 22:00: Battery discharge        (-0.5 kW avg, grid support)
22:00 - 00:00: Evening wind-down         (-0.2 to 0.0 kW)
```

**Value Interpretation:**
- Positive values: Net consumption (importing from grid)
- Negative values: Net export (generation exceeds consumption)

**Behavioral Metrics (Typical):**
- Mean load: 0.026 kW (net exporter)
- Coefficient of variation: 45.5 (highly variable)
- Peak-to-average ratio: 88.8 (extreme variation due to zero-crossing)
- Intermittency: none (always operating)
- Daily variability: High (weather-dependent)

**Forecasting Suitability:**
- Weather-dependent (requires external variables)
- High uncertainty (CV >> 1.0)
- Suitable for: Ensemble methods, with exogenous regressors (irradiance, temperature)
- Requires probabilistic forecasts to capture tail risk

---

## Data Quality and Characteristics

### Value Distribution

**Range:**
- Minimum: -1.8 kW (solar export)
- Maximum: +3.4 kW (EV charging peak)
- Mean across all records: 0.89 kW

**Negative Values:**
- Count: 2,767 records (18.2%)
- Source: Solar+battery assets during generation periods
- Status: Expected (net export to grid)

**Distribution by Asset Type:**
| Type | Positive (%) | Negative (%) | Mean (kW) |
|------|-------------|-------------|-----------|
| EV Charging | 99%+ | <1% | 1.83 |
| Solar+Battery | 40% | 60% | 0.026 |

### Temporal Patterns

**Weekday/Weekend Patterns (EV assets):**
- Weekday peak load: 20-30% higher than weekend
- Weekend patterns show distributed charging

**Daily Seasonality (All assets):**
- EV: Strong 24-hour cycle
- Solar: Strong 24-hour cycle (inverted vs EV)

**Weekly Patterns:**
- Visible Mon-Sun variation (weekday/weekend effect)
- Two complete weeks provide statistical representation

---

## Realistic Noise

The data includes realistic noise to make patterns non-trivial for forecasting:
- Random variation within ±5% of expected values
- No missing data (unrealistic but enables testing)
- No extreme outliers (exceptional weather, equipment failures)
- No trend or drift (assumes stable asset behavior)

---

## Use Cases and Limitations

### Appropriate For

✓ **Developing and validating forecasting models**
- Deterministic, repeatable patterns
- Known asset types and behaviors
- Clean data for algorithm development

✓ **Testing feature engineering pipelines**
- Sufficient temporal coverage (2 weeks)
- Clear asset differentiation (EV vs solar)
- Interpretable behavioral metrics

✓ **Prototyping forecasting workflows**
- Fast iteration (small dataset)
- Reproducible (seed=42)
- All assumptions documented

### Limitations

**Not suitable for production forecasts**
- Too short (2 weeks << seasonal patterns)
- Unrealistic (no missing data, no weather data, no real-world noise)
- Synthetic patterns won't match production assets

**Not suitable for long-term trend analysis**
- Only 14 days (no seasonal variation, no drift)
- No year-over-year comparison
- No holiday effects or special events

**Not suitable for uncertainty calibration**
- Noise levels don't reflect real forecasting error
- Weather variability not represented (solar assets)
- Operational changes not captured

---

## Asset Reference

| Asset ID | Type | Mean Load (kW) | CV | Peak Ratio | Notes |
|----------|------|---|---|---|---|
| ASSET_001-008 | EV Charging | ~1.8 | ~0.79 | ~2.2 | Predictable peaks |
| ASSET_009-015 | Solar+Battery | ~0.03 | ~45 | ~90 | Net exporter |

---

## Reproducibility

To regenerate this data:
```bash
cd working_notes/1_produce_data
python generate_data.py      # Creates metering_data_raw.csv
python inspect_data.py       # Creates src/data/*.parquet files
```

Outputs will be identical due to seed=42 reproducibility.
