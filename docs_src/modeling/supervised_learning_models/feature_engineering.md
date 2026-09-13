# Feature engineering for supervised learning models

How `metering_data_supervised_learning.parquet` is built from `metering_data.parquet` and
`daily_metrics.parquet`. This is a separate, richer table from
[`metering_data_with_features.parquet`](../data/feature_engineering.md) — that one broadcasts
same-day daily metrics for general-purpose slicing; this one is a leakage-safe model matrix
purpose-built for training a supervised regressor (LightGBM) directly against `metering_kwh`.

**Source:** `src/data/generate_metering_features.py` (`build_supervised_learning_features`)

## Why a separate table

`metering_data_with_features.parquet` joins each day's `daily_metrics.parquet` row onto its
*own* 48 half-hour rows — convenient for inspection, but it leaks same-day totals into any
row that claims to forecast that day. It also carries no lag, rolling-window, or cyclical
columns at all. Rather than complicate that table's contract for existing consumers, the
supervised-learning table is generated alongside it, unused by anything that reads the
original file.

## Feature groups

All features are computed per `asset_id`, sorted by `timestamp`, using only observations
strictly before each row's own timestamp — see [Look-ahead handling](#look-ahead-handling).

| Group | Columns | Definition |
|---|---|---|
| Target | `metering_kwh` | raw value, unchanged |
| Short lags | `feat_lag_1` .. `feat_lag_4` | `shift(1..4)` — last 30min to 2hr |
| Seasonal lags | `feat_lag_48`, `feat_lag_96` | `shift(48)`, `shift(96)` — same time 1 and 2 days ago |
| Rolling (6h) | `feat_roll_mean_6h`, `feat_roll_std_6h` | mean/std of `shift(1)` over a 12-step window |
| Rolling (24h) | `feat_roll_mean_24h`, `feat_roll_std_24h` | mean/std of `shift(1)` over a 48-step window |
| Cyclical time | `feat_hour_sin`, `feat_hour_cos` | sin/cos of hour-of-day, 24h period |
| Cyclical time | `feat_dow_sin`, `feat_dow_cos` | sin/cos of day-of-week, 7d period |
| Daily context | `feat_daily_energy_kwh` ... `feat_peak_to_avg_ratio` (12 columns) | `daily_metrics.parquet` joined on `date - 1 day` |

26 feature columns total.

## Look-ahead handling

- **Lag and rolling columns** use `shift`/`rolling` *over* `asset_id`, so no feature ever
  reads a value at or after its own row's timestamp.
- **Rolling stats** are computed on the already-`shift(1)`'d series, so a 24h rolling window
  needs 48 prior observations *plus* the shift — 49 rows of history.
- **Daily context columns** join `daily_metrics.parquet` on the *previous* calendar day
  (`date - 1`), not the same day — this is the fix `feature_engineering.md` documents as
  "proposed, not implemented" for `metering_data_with_features.parquet`; here it's applied
  directly.
- The binding constraint across all of the above is `feat_lag_96` (needs 96 prior rows), so
  the **first 96 rows of each asset are dropped**. Every remaining row is fully non-null.

## Output data

### metering_data_supervised_learning.parquet

**Schema:** `asset_id`, `timestamp`, `metering_kwh` (target), plus the 26 `feat_*` columns
above.

**Size:** 8,640 rows — 15 assets x (672 raw rows - 96 dropped for lookback) = 15 x 576.

**Usage:** direct model matrix for `LightGBMModel` and other supervised-learning estimators;
each row is leakage-safe to train and predict on as-is, with no further lagging required.

## Validation

Before training on this table, spot-check:

- Zero nulls across all `feat_*` columns (verified at generation time — see script output).
- `feat_lag_1` at row `t` equals `metering_kwh` at row `t-1` for the same `asset_id`.
- `feat_daily_energy_kwh` at any row on date `d` matches `daily_metrics.parquet`'s value for
  date `d - 1`, not `d`.

See [Feature validation](../data/feature_validation.md) for the general chronological-split
and leakage-check methodology this table is built to satisfy.
