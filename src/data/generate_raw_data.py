import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def generate_ev_charging_pattern(timestamps, asset_id, noise_level=0.1):
    n = len(timestamps)
    hour_of_day = np.array([ts.hour for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])

    base_load = 0.5

    morning_peak = ((hour_of_day >= 6) & (hour_of_day < 9)).astype(float) * 2.5
    evening_peak = ((hour_of_day >= 16) & (hour_of_day < 21)).astype(float) * 3.0
    night_charging = ((hour_of_day >= 22) | (hour_of_day < 2)).astype(float) * 1.5

    weekday_factor = 1.2 * ((day_of_week < 5).astype(float)) + 0.8 * ((day_of_week >= 5).astype(float))

    pattern = base_load + morning_peak + evening_peak + night_charging
    pattern = pattern * weekday_factor

    noise = np.random.normal(0, noise_level * np.mean(pattern[pattern > 0]), n)
    pattern = np.maximum(pattern + noise, 0)

    return pattern

def generate_solar_battery_pattern(timestamps, asset_id, noise_level=0.08):
    n = len(timestamps)
    hour_of_day = np.array([ts.hour for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])

    solar_generation = np.zeros(n)
    for i, hour in enumerate(hour_of_day):
        if 6 <= hour < 18:
            time_frac = (hour - 6) / 12
            solar_generation[i] = np.sin(time_frac * np.pi) * 2.5
        else:
            solar_generation[i] = 0

    consumption = 0.8 + 0.3 * np.sin((hour_of_day / 24) * 2 * np.pi)
    consumption = consumption * (1.0 if (day_of_week < 5).all() else 0.85)

    battery_discharge = 0.5 * ((hour_of_day >= 18) & (hour_of_day < 22)).astype(float)

    net_pattern = solar_generation - consumption - battery_discharge

    noise = np.random.normal(0, noise_level * (np.max(solar_generation) + 0.5), n)
    pattern = net_pattern + noise

    return pattern

def generate_metering_data(
    n_assets=15,
    days=14,
    interval_minutes=30,
    start_date="2025-01-01",
    asset_types=None,
    random_seed=42
):
    np.random.seed(random_seed)

    if asset_types is None:
        asset_types = (
            ["ev_charging"] * 8 +
            ["solar_battery"] * 7
        )

    n_intervals = (days * 24 * 60) // interval_minutes

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start_dt + timedelta(minutes=interval_minutes * i) for i in range(n_intervals)]

    data_rows = []

    ts_array = np.array(timestamps)
    for asset_idx, asset_type in enumerate(asset_types):
        asset_id = f"ASSET_{asset_idx+1:03d}"

        if asset_type == "ev_charging":
            metering_kwh = generate_ev_charging_pattern(ts_array, asset_id)
        else:
            metering_kwh = generate_solar_battery_pattern(ts_array, asset_id)

        for ts, value in zip(timestamps, metering_kwh):
            data_rows.append({
                'asset_id': asset_id,
                'timestamp': ts,
                'metering_kwh': value,
                'asset_type': asset_type
            })

    df = pl.DataFrame(data_rows)
    df = df.sort(['asset_id', 'timestamp'])

    return df

if __name__ == "__main__":
    df = generate_metering_data(
        n_assets=15,
        days=14,
        interval_minutes=30,
        start_date="2025-01-01"
    )

    output_path = Path(__file__).parent / 'metering_data_raw.csv'
    df.write_csv(output_path)
    print(f"Generated {len(df)} metering records")
    print(f"Assets: {df['asset_id'].n_unique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nData saved to {output_path}")
