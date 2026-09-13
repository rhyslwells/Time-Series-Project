import polars as pl
import numpy as np
from pathlib import Path

LAGS = [1, 2, 3, 4, 48, 96]
ROLLING_WINDOWS = {'6h': 12, '24h': 48}
MAX_LOOKBACK = 96  # rows to drop per asset: covers max(lags)=96 and rolling(48)+shift(1)=49

def load_data():
    data_dir = Path(__file__).parent
    metering = pl.read_parquet(data_dir / 'metering_data.parquet')
    daily = pl.read_parquet(data_dir / 'daily_metrics.parquet')
    return metering, daily

def add_daily_features(metering, daily):
    feature_cols = [c for c in daily.columns if c not in ('asset_id', 'date')]
    daily_features = daily.rename({c: f'feat_{c}' for c in feature_cols})

    enriched = metering.with_columns(
        pl.col('timestamp').dt.date().alias('date')
    ).join(
        daily_features, on=['asset_id', 'date'], how='left'
    ).drop('date')

    return enriched

def print_summary(enriched, metering):
    print("=" * 70)
    print("METERING DATA WITH FEATURES")
    print("=" * 70)
    print(f"Rows: {len(enriched):,} (source metering_data.parquet: {len(metering):,})")
    feature_cols = [c for c in enriched.columns if c.startswith('feat_')]
    print(f"Feature columns joined from daily_metrics: {len(feature_cols)}")
    for c in feature_cols:
        print(f"  {c}")
    print()

    unmatched = enriched.filter(pl.col(feature_cols[0]).is_null()).height
    print(f"Rows with no matching daily features: {unmatched}")
    print()

    print("Sample rows:")
    print(enriched.head(5))
    print()

def add_lag_and_rolling_features(metering):
    """Lag + rolling-window features per asset, using only strictly-past observations."""
    df = metering.sort(['asset_id', 'timestamp'])

    lag_exprs = [
        pl.col('metering_kwh').shift(lag).over('asset_id').alias(f'feat_lag_{lag}')
        for lag in LAGS
    ]

    # Rolling stats computed on the lag-1 series, so a row's rolling window only
    # covers observations strictly before its own timestamp.
    shifted = pl.col('metering_kwh').shift(1).over('asset_id')
    rolling_exprs = []
    for label, window in ROLLING_WINDOWS.items():
        rolling_exprs.append(
            shifted.rolling_mean(window).over('asset_id').alias(f'feat_roll_mean_{label}')
        )
        rolling_exprs.append(
            shifted.rolling_std(window).over('asset_id').alias(f'feat_roll_std_{label}')
        )

    return df.with_columns(lag_exprs + rolling_exprs)

def add_cyclical_features(df):
    """Sin/cos encodings of hour-of-day (24h) and day-of-week (7d)."""
    hour_frac = pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute() / 60.0
    dow = pl.col('timestamp').dt.weekday()  # 1 (Mon) - 7 (Sun)

    return df.with_columns(
        (2 * np.pi * hour_frac / 24.0).sin().alias('feat_hour_sin'),
        (2 * np.pi * hour_frac / 24.0).cos().alias('feat_hour_cos'),
        (2 * np.pi * dow / 7.0).sin().alias('feat_dow_sin'),
        (2 * np.pi * dow / 7.0).cos().alias('feat_dow_cos'),
    )

def add_lagged_daily_features(df, daily):
    """Join daily_metrics onto the previous calendar day, avoiding same-day leakage."""
    feature_cols = [c for c in daily.columns if c not in ('asset_id', 'date')]
    daily_features = daily.rename({c: f'feat_{c}' for c in feature_cols}).with_columns(
        (pl.col('date') + pl.duration(days=1)).alias('join_date')
    )

    return df.with_columns(
        pl.col('timestamp').dt.date().alias('date')
    ).join(
        daily_features.drop('date'), left_on=['asset_id', 'date'],
        right_on=['asset_id', 'join_date'], how='left'
    ).drop('date')

def build_supervised_learning_features(metering, daily):
    df = add_lag_and_rolling_features(metering)
    df = add_cyclical_features(df)
    df = add_lagged_daily_features(df, daily)

    # Drop the leading rows per asset that don't have full lag/rolling/daily history.
    df = df.with_columns(
        pl.col('timestamp').rank('ordinal').over('asset_id').alias('_row_rank')
    ).filter(pl.col('_row_rank') > MAX_LOOKBACK).drop('_row_rank')

    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    return df.select(['asset_id', 'timestamp', 'metering_kwh'] + sorted(feature_cols))

def print_supervised_learning_summary(supervised, metering):
    print("=" * 70)
    print("METERING DATA WITH SUPERVISED LEARNING FEATURES")
    print("=" * 70)
    print(f"Rows: {len(supervised):,} (source metering_data.parquet: {len(metering):,})")
    feature_cols = [c for c in supervised.columns if c.startswith('feat_')]
    print(f"Feature columns: {len(feature_cols)}")
    for c in feature_cols:
        print(f"  {c}")
    print()

    null_counts = supervised.select(pl.col(feature_cols).is_null().sum())
    total_nulls = sum(null_counts.row(0))
    print(f"Total nulls across feature columns: {total_nulls}")
    print()

    print("Sample rows:")
    print(supervised.head(5))
    print()

def save_supervised_learning_parquet(supervised):
    data_dir = Path(__file__).parent
    output_path = data_dir / 'metering_data_supervised_learning.parquet'
    supervised.write_parquet(output_path)

    print("=" * 70)
    print("FILE SAVED")
    print("=" * 70)
    print(f"Metering data (supervised learning): {output_path}")
    print()

def save_parquet(enriched):
    data_dir = Path(__file__).parent
    output_path = data_dir / 'metering_data_with_features.parquet'
    enriched.write_parquet(output_path)

    print("=" * 70)
    print("FILE SAVED")
    print("=" * 70)
    print(f"Metering data with features: {output_path}")
    print()

def main():
    print("\nLoading data...")
    metering, daily = load_data()

    print("Joining daily features onto 30-min metering data...")
    enriched = add_daily_features(metering, daily)

    print_summary(enriched, metering)
    save_parquet(enriched)

    print("Building supervised-learning feature matrix (lags, rolling stats, cyclical, lagged daily context)...")
    supervised = build_supervised_learning_features(metering, daily)

    print_supervised_learning_summary(supervised, metering)
    save_supervised_learning_parquet(supervised)

if __name__ == "__main__":
    main()
