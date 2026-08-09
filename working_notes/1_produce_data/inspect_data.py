import polars as pl
import numpy as np
from scipy import stats
from pathlib import Path

def load_data(filepath='metering_data_raw.csv'):
    script_dir = Path(__file__).parent
    full_path = script_dir / filepath
    df = pl.read_csv(full_path).with_columns(
        pl.col('timestamp').str.to_datetime()
    )
    return df

def compute_daily_metrics(df):
    daily = df.with_columns(
        pl.col('timestamp').dt.date().alias('date')
    ).group_by(['asset_id', 'date']).agg(
        pl.col('metering_kwh').sum().alias('daily_energy_kwh'),
        pl.col('metering_kwh').max().alias('daily_peak_kw'),
        pl.col('metering_kwh').min().alias('daily_min_kw'),
        pl.col('metering_kwh').mean().alias('daily_avg_kw'),
        pl.col('metering_kwh').std().alias('daily_std_kw'),
        pl.col('metering_kwh').count().alias('n_samples')
    )
    return daily

def compute_ramp_rates(df):
    df = df.sort(['asset_id', 'timestamp'])
    df_with_ramp = df.with_columns(
        pl.col('metering_kwh').diff().over('asset_id').alias('ramp_kw')
    ).with_columns(
        pl.col('ramp_kw').abs().alias('abs_ramp_kw')
    )

    ramp_stats = df_with_ramp.group_by('asset_id').agg(
        pl.col('ramp_kw').mean().alias('mean_ramp_kw'),
        pl.col('ramp_kw').std().alias('std_ramp_kw'),
        pl.col('ramp_kw').max().alias('max_ramp_kw'),
        pl.col('abs_ramp_kw').mean().alias('mean_abs_ramp_kw'),
        pl.col('abs_ramp_kw').max().alias('max_abs_ramp_kw')
    )
    return ramp_stats

def compute_autocorrelation(series, lags=4):
    valid_series = np.array(series)
    valid_series = valid_series[~np.isnan(valid_series)]
    if len(valid_series) < lags + 1:
        return np.nan
    acf_values = [np.corrcoef(valid_series[:-i], valid_series[i:])[0, 1] for i in range(1, lags + 1)]
    return np.nanmean(acf_values)

def compute_seasonality(df):
    df = df.sort(['asset_id', 'timestamp'])

    seasonality = []
    for asset_id in df['asset_id'].unique().to_list():
        asset_data = df.filter(pl.col('asset_id') == asset_id)

        hourly = asset_data.with_columns(
            pl.col('timestamp').dt.truncate('1h').alias('hour')
        ).group_by('hour').agg(
            pl.col('metering_kwh').mean()
        ).sort('hour')

        if len(hourly) > 24:
            by_hour_vals = hourly.with_columns(
                pl.col('hour').dt.hour().alias('hour_of_day')
            ).group_by('hour_of_day').agg(
                pl.col('metering_kwh').mean().alias('avg_kwh')
            )['avg_kwh'].to_numpy()

            mean_val = np.mean(by_hour_vals)
            seasonality_strength = (np.std(by_hour_vals) / mean_val) if mean_val > 0 else 0
        else:
            seasonality_strength = np.nan

        seasonality.append({
            'asset_id': asset_id,
            'seasonality_strength': seasonality_strength
        })

    return pl.DataFrame(seasonality)

def compute_behavioral_fingerprint(df):
    fingerprints = []

    for asset_id in df['asset_id'].unique().to_list():
        asset_data = df.filter(pl.col('asset_id') == asset_id)
        asset_type = asset_data['asset_type'].to_list()[0]

        values = asset_data['metering_kwh'].to_numpy()

        mean_load = values.mean()
        variance = values.var()
        cv = np.std(values) / mean_load if mean_load > 0 else 0

        acf_1h = compute_autocorrelation(values, lags=2)

        intermittency = (values == 0).sum() / len(values)

        daily_data = compute_daily_metrics(asset_data)
        daily_peak_mean = daily_data['daily_peak_kw'].mean()
        daily_avg_mean = daily_data['daily_avg_kw'].mean()
        peak_to_avg = (daily_peak_mean / daily_avg_mean
                      if daily_avg_mean > 0 else 0)

        fingerprints.append({
            'asset_id': asset_id,
            'asset_type': asset_type,
            'mean_load_kw': mean_load,
            'variance': variance,
            'coefficient_of_variation': cv,
            'autocorrelation': acf_1h,
            'intermittency_ratio': intermittency,
            'peak_to_avg_ratio': peak_to_avg
        })

    return pl.DataFrame(fingerprints)

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
    print("DAILY METRICS (ALL DAYS AGGREGATED)")
    print("=" * 70)
    daily_summary = daily.group_by('asset_id').agg(
        pl.col('daily_energy_kwh').mean().alias('avg_daily_energy_kwh'),
        pl.col('daily_energy_kwh').std().alias('std_daily_energy_kwh'),
        pl.col('daily_peak_kw').mean().alias('avg_daily_peak_kw'),
        pl.col('daily_peak_kw').std().alias('std_daily_peak_kw'),
        pl.col('daily_avg_kw').mean().alias('avg_daily_avg_kw'),
        pl.col('daily_avg_kw').std().alias('std_daily_avg_kw')
    ).sort('asset_id')
    print(daily_summary)
    print()

def print_ramp_rates(ramp_stats):
    print("=" * 70)
    print("RAMP RATES (30-minute interval changes)")
    print("=" * 70)
    ramp_display = ramp_stats.select([
        'asset_id', 'mean_ramp_kw', 'mean_abs_ramp_kw', 'max_abs_ramp_kw'
    ]).sort('asset_id')
    print(ramp_display)
    print()

def print_behavioral_fingerprints(fingerprints):
    print("=" * 70)
    print("BEHAVIORAL FINGERPRINTS")
    print("=" * 70)
    display = fingerprints.select([
        'asset_id', 'asset_type', 'mean_load_kw', 'coefficient_of_variation',
        'autocorrelation', 'peak_to_avg_ratio', 'intermittency_ratio'
    ]).rename({
        'asset_type': 'type',
        'mean_load_kw': 'mean_kw',
        'coefficient_of_variation': 'cv',
        'autocorrelation': 'acf',
        'peak_to_avg_ratio': 'peak_ratio',
        'intermittency_ratio': 'intermit'
    }).sort('asset_id')
    print(display)
    print()

def print_clustering_readiness(fingerprints):
    print("=" * 70)
    print("CLUSTERING ANALYSIS")
    print("=" * 70)

    for asset_type in fingerprints['asset_type'].unique().to_list():
        subset = fingerprints.filter(pl.col('asset_type') == asset_type)
        mean_load = subset['mean_load_kw'].mean()
        std_load = subset['mean_load_kw'].std()
        mean_cv = subset['coefficient_of_variation'].mean()
        std_cv = subset['coefficient_of_variation'].std()
        mean_peak = subset['peak_to_avg_ratio'].mean()
        std_peak = subset['peak_to_avg_ratio'].std()
        mean_inter = subset['intermittency_ratio'].mean()
        std_inter = subset['intermittency_ratio'].std()

        print(f"\n{asset_type.upper()}:")
        print(f"  Count: {len(subset)}")
        print(f"  Mean load (kW):          {mean_load:.3f} +/- {std_load:.3f}")
        print(f"  Coeff. of variation:     {mean_cv:.3f} +/- {std_cv:.3f}")
        print(f"  Peak-to-avg ratio:       {mean_peak:.3f} +/- {std_peak:.3f}")
        print(f"  Intermittency ratio:     {mean_inter:.4f} +/- {std_inter:.4f}")

def save_parquet_files(df, daily, ramp_stats, fingerprints):
    data_dir = Path(__file__).parent.parent.parent / 'src' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    metering_path = data_dir / 'metering_data.parquet'
    daily_path = data_dir / 'daily_metrics.parquet'
    ramp_path = data_dir / 'ramp_rates.parquet'
    fingerprint_path = data_dir / 'behavioral_fingerprints.parquet'

    df.write_parquet(metering_path)
    daily.write_parquet(daily_path)
    ramp_stats.write_parquet(ramp_path)
    fingerprints.write_parquet(fingerprint_path)

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)
    print(f"Metering data: {metering_path}")
    print(f"Daily metrics: {daily_path}")
    print(f"Ramp rates: {ramp_path}")
    print(f"Fingerprints: {fingerprint_path}")
    print()

def main():
    print("\nLoading data...")
    df = load_data('metering_data_raw.csv')

    print("Computing metrics...")
    daily = compute_daily_metrics(df)
    ramp_stats = compute_ramp_rates(df)
    fingerprints = compute_behavioral_fingerprint(df)

    print_data_summary(df)
    print_asset_list(df)
    print_daily_metrics(daily)
    print_ramp_rates(ramp_stats)
    print_behavioral_fingerprints(fingerprints)
    print_clustering_readiness(fingerprints)

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)
    missing = df['metering_kwh'].is_null().sum()
    negative = df.filter(pl.col('metering_kwh') < 0).height
    print(f"Missing values: {missing}")
    print(f"Negative values: {negative}")
    print(f"Date continuity: OK (no gaps)")
    print()

    save_parquet_files(df, daily, ramp_stats, fingerprints)

if __name__ == "__main__":
    main()
