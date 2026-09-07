# Synthetic metering data

This document describes the characteristics, patterns, and behavior of the synthetic metering dataset (`src/data/metering_data.parquet`) used for model development and testing. For the generation process and formulas, see [Data Generation and Calculations](data_generation.md).

## Dataset overview

**Purpose:** Provide realistic, reproducible metering data for developing and validating energy forecasting systems.

For generation parameters (seed, asset counts, interval, total records) see
[Data Generation and Calculations](data_generation.md). Relevant here: the series runs
2025-01-01 (Wednesday) to 2025-01-14 (Tuesday) — two full weeks, so weekday/weekend
effects appear twice — at 1,344 complete records per asset.

---

## Asset types and behavioral patterns

### EV charging stations (8 assets)

**Characteristics:**
- Deterministic, predictable daily patterns
- Strong weekday/weekend separation
- Two main charging windows: morning and evening

**Resulting half-hourly values** (component sum × weekday factor, before noise):

| Window | Components | Weekday (×1.2) | Weekend (×0.8) |
|--------|-----------|----------------|----------------|
| 00:00 - 02:00 | base 0.5 + night 1.5 | 2.4 | 1.6 |
| 02:00 - 06:00 | base 0.5 | 0.6 | 0.4 |
| 06:00 - 09:00 | base 0.5 + morning 2.5 | 3.6 | 2.4 |
| 09:00 - 16:00 | base 0.5 | 0.6 | 0.4 |
| 16:00 - 21:00 | base 0.5 + evening 3.0 | 4.2 | 2.8 |
| 21:00 - 22:00 | base 0.5 | 0.6 | 0.4 |
| 22:00 - 00:00 | base 0.5 + night 1.5 | 2.4 | 1.6 |

There is no separate "overnight minimal" band — outside the three charging windows the value
is just the base load. Night charging runs 22:00-02:00, not 21:00-02:00.

**Weekday/Weekend Variation:**
- Weekday (Mon-Fri): 1.2× intensity (higher charging demand)
- Weekend (Sat-Sun): 0.8× intensity (lower, more distributed charging)

**Behavioral Metrics (indicative — computed from a prior run, not re-verified here):**
- Mean load: ~1.8 kW
- Coefficient of variation: ~0.8 (moderate predictability)
- Peak-to-average ratio: ~2.2 (pronounced peaks)
- Intermittency: minimal (rarely zero)
- Daily variability: Low (similar patterns day-to-day)

**Forecasting Suitability:**
- Strong seasonality (daily patterns)
- Low uncertainty (CV < 1.0)
- Suitable for: SARIMA, exponential smoothing, gradient boosting

---

### Solar + battery storage (7 assets)

**Characteristics:**
- Generation-led, net metering (exports to grid)
- Strong solar correlation
- Highly variable, weather-dependent
- Evening battery discharge supports grid

**Daily Pattern (net = solar generation − consumption − battery discharge):**
```
Solar generation: half-sine over 06:00-18:00, peak 2.5 at ~12:00, zero otherwise
Consumption:      0.8 + 0.3 sin(2π hour / 24), scaled by a constant 0.85
Battery discharge: 0.5 subtracted over 18:00-22:00
```
Overnight the net is roughly −consumption (a small import); midday the solar term dominates
and the net goes strongly negative (export); the evening discharge deepens the 18:00-22:00 dip.

**Value Interpretation:**
- Positive values: Net consumption (importing from grid)
- Negative values: Net export (generation exceeds consumption)

!!! warning "No weekday/weekend variation"
    Unlike EV assets, solar assets are identical on weekdays and weekends — the intended
    0.85 weekend factor is applied on every day (see [generator issue](data_generation.md#reported-code-issues)).

**Behavioral Metrics (indicative — computed from a prior run, not re-verified here):**
- Mean load: near zero (net exporter)
- Coefficient of variation: very large (mean near zero makes CV unstable)
- Peak-to-average ratio: very large (same cause — denominator near zero)
- Intermittency: none (always operating)
- Daily variability: High

**Forecasting Suitability:**
- Weather-dependent (requires external variables)
- High uncertainty (CV >> 1.0)
- Suitable for: Ensemble methods, with exogenous regressors (irradiance, temperature)
- Requires probabilistic forecasts to capture tail risk

---

## Data quality and characteristics

### Value distribution

**Range (indicative, from a prior run):**
- Minimum: about -1.8 kW (solar export)
- Maximum: about +4.2 kW (EV weekday evening peak, plus noise)
- Mean across all records: order of 1 kW (EV assets dominate the count)

**Negative Values:**
- Count: 2,767 records — 27.5% of the 10,080 rows (2,767 / 10,080)
- Source: Solar+battery assets during generation periods (2,767 / 4,704 solar rows ≈ 59%)
- Status: Expected (net export to grid)

**Distribution by Asset Type (indicative):**
| Type | Positive (%) | Negative (%) | Mean (kW) |
|------|-------------|-------------|-----------|
| EV Charging | ~100% | ~0% | ~1.8 |
| Solar+Battery | ~41% | ~59% | near zero |

### Temporal patterns

**Weekday/Weekend Patterns:**
- EV assets: weekday values are exactly 1.2/0.8 = 1.5× the weekend values (50% higher), before noise
- Solar assets: none — the weekend factor is inert (see warning above)

**Daily Seasonality (all assets):**
- EV: Strong 24-hour cycle
- Solar: Strong 24-hour cycle (inverted vs EV)

**Weekly Patterns:**
- EV assets only: a weekday/weekend step, repeated across the two weeks
- Two weeks give exactly two realisations of the weekly cycle — barely enough to estimate a weekly component (see [Models](../theory/models.md))

---

## Realistic noise

The data includes realistic noise to make patterns non-trivial for forecasting:
- Additive Gaussian noise (unbounded), std = 10% of the mean positive pattern value for EV assets, 8% of (max solar + 0.5) for solar assets
- EV values are clipped at zero after noise; solar values are not, which is why solar can go negative
- No missing data (unrealistic but enables testing)
- No extreme outliers (exceptional weather, equipment failures)
- No trend or drift (assumes stable asset behavior)

---

## Use cases and limitations

### Appropriate for

**Developing and validating forecasting models**
- Deterministic, repeatable patterns
- Known asset types and behaviors
- Clean data for algorithm development

**Testing feature engineering pipelines**
- Sufficient temporal coverage (2 weeks)
- Clear asset differentiation (EV vs solar)
- Interpretable behavioral metrics

**Prototyping forecasting workflows**
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

## Asset reference

Indicative figures from a prior run; not re-verified in this review.

| Asset ID | Type | Mean Load (kW) | CV | Peak Ratio | Notes |
|----------|------|---|---|---|---|
| ASSET_001-008 | EV Charging | ~1.8 | ~0.8 | ~2.2 | Predictable peaks |
| ASSET_009-015 | Solar+Battery | near 0 | very large | very large | Net exporter; CV/ratio unstable because mean ≈ 0 |
