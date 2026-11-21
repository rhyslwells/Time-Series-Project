"""Shared helpers for the statsforecast exploration notebooks.

statsforecast is the *statistical* counterpart to mlforecast: no feature engineering, each series
gets its own classical model (ETS, ARIMA, MSTL, seasonal baselines) fitted directly on its history.

Conventions (same as ../mlforecast)
-----------------------------------
* Half-hourly, values in kW (negative = export), `season_length = 48` (one day).
* Day-ahead: forecast `h=48` slots issued at 00:00.
* Long format `(unique_id, ds, y)`. statsforecast wants pandas/polars; we use pandas here to match mlforecast demos.
* Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, HistoricAverage, MSTL, SeasonalNaive, SeasonalWindowAverage,
)
from statsforecast.utils import ConformalIntervals

PERIODS_PER_DAY = 48
SEASON = PERIODS_PER_DAY
LEVEL = [80]
COLORS = {
    "SeasonalNaive": "#7f7f7f", "SeasonalWindowAverage": "#1f77b4", "HistoricAverage": "#bcbd22",
    "AutoETS": "#2ca02c", "MSTL": "#ff7f0e", "MSTL_ARIMA": "#9467bd",
}


# --------------------------------------------------------------------------- data
def make_synthetic(days: int = 28, noise: float = 4.0, seed: int = 0) -> pd.Series:
    """Fake net-load for one asset: evening peak, midday solar export, weekend dip, AR(1) noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-02", periods=days * PERIODS_PER_DAY, freq="30min")  # a Monday
    h = idx.hour + idx.minute / 60
    demand = 30 + 20 * np.exp(-(((h - 18) / 2.5) ** 2)) + 8 * np.exp(-(((h - 8) / 1.5) ** 2))
    demand = demand * np.where(idx.dayofweek >= 5, 0.85, 1.0)
    cloud = rng.uniform(0.3, 1.0, size=days)
    solar = 35 * np.clip(np.sin(np.pi * (h - 6) / 12), 0, None) ** 2 * cloud[(idx.normalize() - idx[0].normalize()).days]
    eps = np.zeros(len(idx))
    for i in range(1, len(idx)):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, noise)
    return pd.Series(demand - solar + eps, index=idx, name="kw")


def make_df(n_assets: int = 2, days: int = 28, noise: float = 4.0) -> pd.DataFrame:
    """Long dataframe with `n_assets` series (different seeds)."""
    frames = []
    for i in range(n_assets):
        s = make_synthetic(days=days, noise=noise, seed=i)
        frames.append(pd.DataFrame({"unique_id": f"asset_{i}", "ds": s.index, "y": s.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------- models
def make_models(conformal: bool = False) -> list:
    """The model zoo. Fast ones only: AutoARIMA with season_length=48 is far too slow, so ARIMA is
    used as MSTL's *trend* forecaster on the deseasonalised series instead."""
    ci = ConformalIntervals(h=PERIODS_PER_DAY, n_windows=3) if conformal else None
    mstl_ci = ConformalIntervals(h=PERIODS_PER_DAY, n_windows=3)  # MSTL and SeasonalWindowAverage have no parametric intervals: conformal only
    return [
        HistoricAverage(prediction_intervals=ci),
        SeasonalNaive(season_length=SEASON, prediction_intervals=ci),
        SeasonalWindowAverage(season_length=SEASON, window_size=7, alias="SeasonalWindowAverage", prediction_intervals=mstl_ci),
        AutoETS(season_length=SEASON, model="ZZN", alias="AutoETS", prediction_intervals=ci),
        MSTL(season_length=SEASON, trend_forecaster=AutoETS(model="ZZN"), alias="MSTL", prediction_intervals=mstl_ci),
        MSTL(season_length=[SEASON, SEASON * 7], trend_forecaster=AutoARIMA(), alias="MSTL_ARIMA", prediction_intervals=mstl_ci),
    ]


def make_sf(models=None, n_jobs: int = 1) -> StatsForecast:
    return StatsForecast(models=models or make_models(), freq="30min", n_jobs=n_jobs)


def model_cols(frame: pd.DataFrame) -> list[str]:
    """Point-forecast columns of a statsforecast output frame (skips -lo-/-hi- columns)."""
    skip = {"unique_id", "ds", "cutoff", "y"}
    return [c for c in frame.columns if c not in skip and "-lo-" not in c and "-hi-" not in c]


# ----------------------------------------------------------------------- metrics
def metrics_table(cv: pd.DataFrame, level: int = 80) -> pd.DataFrame:
    """MAE / RMSE / coverage (if intervals present) / skill vs seasonal naive."""
    rows = []
    for m in model_cols(cv):
        err = cv["y"] - cv[m]
        row = {"model": m, "MAE (kW)": err.abs().mean(), "RMSE (kW)": np.sqrt((err ** 2).mean())}
        lo, hi = f"{m}-lo-{level}", f"{m}-hi-{level}"
        if lo in cv:
            row[f"{level}% cover"] = float(((cv["y"] >= cv[lo]) & (cv["y"] <= cv[hi])).mean())
            row[f"{level}% width"] = float((cv[hi] - cv[lo]).mean())
        rows.append(row)
    t = pd.DataFrame(rows).set_index("model")
    if "SeasonalNaive" in t.index:
        t["skill vs naive"] = 1 - t["MAE (kW)"] / t.loc["SeasonalNaive", "MAE (kW)"]
    return t
