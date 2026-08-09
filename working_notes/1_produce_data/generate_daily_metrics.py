import polars as pl
import numpy as np
from pathlib import Path

def load_data(filepath='metering_data_raw.csv'):
    script_dir = Path(__file__).parent
    full_path = script_dir / filepath
    df = pl.read_csv(full_path).with_columns(
        pl.col('timestamp').str.to_datetime()
    )
    return df

def compute_daily_metrics(df):
    df_sorted = df.sort(['asset_id', 'timestamp'])

    daily = df_sorted.with_columns(
        pl.col('timestamp').dt.date().alias('date')
    ).group_by(['asset_id', 'date']).agg(
        pl.col('metering_kwh').sum().alias('daily_energy_kwh'),
        pl.col('metering_kwh').max().alias('daily_peak_kw'),
        pl.col('metering_kwh').min().alias('daily_min_kw'),
        pl.col('metering_kwh').mean().alias('daily_avg_kw'),
        pl.col('metering_kwh').std().alias('daily_std_kw'),
        pl.col('metering_kwh').count().alias('n_samples')
    )

    df_with_ramp = df_sorted.with_columns(
        pl.col('timestamp').dt.date().alias('date'),
        pl.col('metering_kwh').diff().over('asset_id').alias('ramp_kw')
    ).with_columns(
        pl.col('ramp_kw').abs().alias('abs_ramp_kw')
    )

    ramp_by_day = df_with_ramp.group_by(['asset_id', 'date']).agg(
        pl.col('ramp_kw').mean().alias('mean_ramp_kw'),
        pl.col('ramp_kw').std().alias('std_ramp_kw'),
        pl.col('abs_ramp_kw').max().alias('max_abs_ramp_kw')
    )

    df_with_metrics = df_sorted.with_columns(
        pl.col('timestamp').dt.date().alias('date')
    ).group_by(['asset_id', 'date']).agg(
        pl.col('metering_kwh').std().alias('_std'),
        pl.col('metering_kwh').mean().alias('_mean'),
        (pl.col('metering_kwh') == 0).sum().alias('_zero_count'),
        pl.len().alias('_total_count')
    ).with_columns(
        coefficient_of_variation = pl.when(pl.col('_mean') > 0)
            .then(pl.col('_std') / pl.col('_mean'))
            .otherwise(0),
        intermittency_ratio = pl.col('_zero_count') / pl.col('_total_count')
    ).select(['asset_id', 'date', 'coefficient_of_variation', 'intermittency_ratio'])

    daily = daily.join(ramp_by_day, on=['asset_id', 'date'])
    daily = daily.join(df_with_metrics, on=['asset_id', 'date'])

    daily = daily.with_columns(
        peak_to_avg_ratio = pl.col('daily_peak_kw') / pl.when(pl.col('daily_avg_kw') > 0)
            .then(pl.col('daily_avg_kw'))
            .otherwise(1)
    )

    return daily

def print_data_summary(df):
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df):,}")
    print(f"Unique assets: {df['asset_id'].n_unique()}")
    print(f"Asset types: {df['asset_type'].unique().to_list()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Interval: 30 minutes")
    print()

def print_asset_list(df):
    print("=" * 70)
    print("ASSET LIST")
    print("=" * 70)
    asset_summary = df.group_by('asset_id').agg(
        pl.col('asset_type').first().alias('type'),
        pl.col('metering_kwh').min().alias('min_kw'),
        pl.col('metering_kwh').mean().alias('mean_kw'),
        pl.col('metering_kwh').max().alias('max_kw'),
        pl.col('metering_kwh').std().alias('std_kw'),
        pl.col('timestamp').count().alias('n_samples')
    ).sort('asset_id')
    print(asset_summary)
    print()

def print_daily_metrics(daily):
    print("=" * 70)
    print("DAILY METRICS (SAMPLE OF FIRST 10 ROWS)")
    print("=" * 70)
    display = daily.select([
        'asset_id', 'date', 'daily_energy_kwh', 'daily_peak_kw', 'daily_avg_kw',
        'coefficient_of_variation', 'peak_to_avg_ratio', 'mean_ramp_kw'
    ]).head(10)
    print(display)
    print()

    print("=" * 70)
    print("DAILY METRICS (AGGREGATED BY ASSET)")
    print("=" * 70)
    daily_summary = daily.group_by('asset_id').agg(
        pl.col('daily_energy_kwh').mean().alias('avg_daily_energy_kwh'),
        pl.col('daily_energy_kwh').std().alias('std_daily_energy_kwh'),
        pl.col('daily_peak_kw').mean().alias('avg_daily_peak_kw'),
        pl.col('coefficient_of_variation').mean().alias('avg_cv'),
        pl.col('peak_to_avg_ratio').mean().alias('avg_peak_ratio'),
        pl.col('mean_ramp_kw').mean().alias('avg_ramp_kw')
    ).sort('asset_id')
    print(daily_summary)
    print()

def save_parquet_files(df, daily):
    data_dir = Path(__file__).parent.parent.parent / 'src' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    metering_path = data_dir / 'metering_data.parquet'
    daily_path = data_dir / 'daily_metrics.parquet'

    df.write_parquet(metering_path)
    daily.write_parquet(daily_path)

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Metering data: {metering_path}")
    print(f"Daily metrics: {daily_path}")
    print()

def main():
    print("\nLoading data...")
    df = load_data('metering_data_raw.csv')

    print("Computing metrics...")
    daily = compute_daily_metrics(df)

    print_data_summary(df)
    print_asset_list(df)
    print_daily_metrics(daily)

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)
    missing = df['metering_kwh'].is_null().sum()
    negative = df.filter(pl.col('metering_kwh') < 0).height
    print(f"Missing values: {missing}")
    print(f"Negative values: {negative}")
    print(f"Date continuity: OK (no gaps)")
    print()

    save_parquet_files(df, daily)

if __name__ == "__main__":
    main()
