"""Longer, more heterogeneous synthetic EV-charge-point data for model-selection experiments.

Why not `src/data/generate_raw_data.py`: that script is a production data contract (14 days, every asset of a
type identical up to noise). This one is experimental and lives here. It is meant to answer
"does a pooled model beat per-asset baselines when each asset has 6 weeks of history?".

Design (all synthetic, no trend):
  * 8 weeks, 30-min, so a 6-week training window plus a 2-week holdout.
  * Each asset has its own scale, peak timing, weekday/weekend ratio and noise level.
  * Load is session-like: a daily/weekly expected profile times random on/off sessions, so it is spiky
    and often near zero, like real charge points.
  * Optional behaviour shift: a subset of assets change their evening peak after `shift_day`,
    to test whether a 6-weekly retrain picks up new behaviour.

Run from the project root:
    uv run python mlforecast/make_long_data.py
Writes mlforecast/data/ev_long.parquet with columns: unique_id, ds, y.
"""
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).parent / "data" / "ev_long.parquet"
PERIODS_PER_DAY = 48


def _gauss(h, mu, sd):
    return np.exp(-(((h - mu) / sd) ** 2))


def make_ev_assets(
    n_assets: int = 30,
    days: int = 56,
    start: str = "2025-01-06",  # a Monday
    shift_day: int | None = 42,
    shift_fraction: float = 0.3,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ds = np.arange(
        np.datetime64(start), np.datetime64(start) + np.timedelta64(days, "D"), np.timedelta64(30, "m")
    )
    n = len(ds)
    minutes = (ds - ds.astype("datetime64[D]")).astype("timedelta64[m]").astype(int)
    hour = minutes / 60.0
    dow = (ds.astype("datetime64[D]").astype(int) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    weekend = dow >= 5
    day_idx = (ds.astype("datetime64[D]") - ds[0].astype("datetime64[D]")).astype(int)

    frames = []
    shifted = set(rng.choice(n_assets, size=int(n_assets * shift_fraction), replace=False)) if shift_day else set()
    for a in range(n_assets):
        scale = rng.lognormal(mean=0.8, sigma=0.4)  # asset size
        eve_mu = rng.normal(18.0, 1.0)  # evening peak hour
        morn_w = rng.uniform(0.2, 0.9)  # morning peak strength
        wkend = rng.uniform(0.4, 1.2)  # weekend factor vs weekday
        sess_p = rng.uniform(0.5, 0.9)  # prob. a session is active inside the peak
        noise = rng.uniform(0.05, 0.2)

        eve = np.full(n, eve_mu)
        if a in shifted:
            eve = np.where(day_idx >= shift_day, eve_mu + 2.5, eve_mu)  # peak moves later

        profile = 0.1 + _gauss(hour, 8.0, 1.5) * morn_w + _gauss(hour, eve, 2.0) * 1.0
        profile = profile * np.where(weekend, wkend, 1.0)

        # Sessions: a 2-hour (4 slot) block is on or off, so the series is spiky.
        block = np.repeat(rng.random(n // 4 + 1) < sess_p, 4)[:n]
        on = np.where(profile > 0.3, block, 1.0)
        y = scale * profile * (0.35 + 0.65 * on) + rng.normal(0, noise * scale, n)
        frames.append(pl.DataFrame({"unique_id": f"EV_{a + 1:03d}", "ds": ds.astype("datetime64[us]"), "y": np.maximum(y, 0.0)}))

    return pl.concat(frames).sort("unique_id", "ds")


if __name__ == "__main__":
    df = make_ev_assets()
    OUT.parent.mkdir(exist_ok=True)
    df.write_parquet(OUT)
    print(f"{df['unique_id'].n_unique()} assets, {len(df)} rows, {df['ds'].min()} -> {df['ds'].max()}")
    print(f"wrote {OUT}")
