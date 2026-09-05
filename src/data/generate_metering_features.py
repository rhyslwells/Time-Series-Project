import polars as pl
from pathlib import Path

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

if __name__ == "__main__":
    main()
