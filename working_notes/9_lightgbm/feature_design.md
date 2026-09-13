# Phase 1.1 — Supervised-learning feature set design

Design for `metering_data_supervised_learning.parquet`, produced by a new function in
`src/data/generate_metering_features.py`. Kept separate from
`metering_data_with_features.parquet` (used by other notebooks) to avoid breaking it.

## Feature set

All features computed per `asset_id` (partitioned), sorted by `timestamp`. Everything
that touches history uses `shift`/`rolling` **after** a shift, so a row's features only
see observations strictly before its own timestamp — no look-ahead.

| Group | Columns | Definition | Rationale |
|---|---|---|---|
| Target | `metering_kwh` | raw value | supervised target |
| Lags | `feat_lag_1`, `feat_lag_2`, `feat_lag_3`, `feat_lag_4` | `shift(1..4)` | last 30min-2hr, matches `LightGBMModel`'s existing short lags |
| Lags | `feat_lag_48`, `feat_lag_96` | `shift(48)`, `shift(96)` | same time yesterday / 2 days ago — daily seasonality, matches `LightGBMModel` defaults |
| Rolling | `feat_roll_mean_6h`, `feat_roll_std_6h` | `shift(1).rolling_mean/std(12)` | short-term level/volatility |
| Rolling | `feat_roll_mean_24h`, `feat_roll_std_24h` | `shift(1).rolling_mean/std(48)` | daily level/volatility |
| Cyclical | `feat_hour_sin`, `feat_hour_cos` | sin/cos of hour-of-day (24h period) | smooth daily cycle, no midnight discontinuity |
| Cyclical | `feat_dow_sin`, `feat_dow_cos` | sin/cos of day-of-week (7d period) | weekly cycle (documented gap in current models — see `docs_src/theory/models.md`) |
| Daily context | `feat_daily_energy_kwh` ... `feat_peak_to_avg_ratio` (12 cols) | `daily_metrics.parquet` joined on `date - 1 day` | previous-day behavioral context, matches `feature_engineering.md`'s "shift date by one day" fix for the same-day leakage caveat |

Dimensionality: 6 lag + 4 rolling + 4 cyclical + 12 daily-context = **26 features**.

## Look-ahead handling

- Lag/rolling columns: leaking rows are the leading ones with insufficient history.
  Binding constraint is `lag_96` (needs 96 prior observations) — same order as the
  24h rolling window computed on a shifted series (needs 48 prior + 1 shift = 49).
- Daily context columns: joined on the *previous* calendar day, so the first day per
  asset (48 rows) has no match and is null.
- **Row handling:** drop the first 96 rows per `asset_id` (covers both constraints
  above, since 96 > 49 and 96 > 48). Remaining rows are fully non-null by construction.
- Per asset: 672 rows (14 days x 48) - 96 dropped = 576 rows. Across 15 assets: 8,640
  rows in the output parquet (vs 10,080 in `metering_data.parquet`).

## Schema

`asset_id`, `timestamp`, `metering_kwh` (target), then the 26 `feat_*` columns above.
