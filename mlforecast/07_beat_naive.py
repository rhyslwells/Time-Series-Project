# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/07_beat_naive.py
#   uv run marimo run  mlforecast/07_beat_naive.py   (app mode, code hidden)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings
    from functools import partial

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import polars as pl
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    from mlforecast.target_transforms import Differences, LocalStandardScaler
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive, SeasonalNaive, SeasonalWindowAverage
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae, mase, rmse

    warnings.filterwarnings("ignore")
    return (
        AutoETS, Differences, LocalStandardScaler, MLForecast, Naive, RollingMean,
        SeasonalNaive, SeasonalWindowAverage, StatsForecast, evaluate, lgb, mae, mase,
        mo, np, partial, pd, pl, px, rmse,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Can any model beat the naive baselines?

    Real benchmark for half-hourly energy data is **seasonal naive** (same slot yesterday = lag 48,
    same slot last week = lag 336), not plain naive. A model only earns its place if it captures
    something those miss *without* adding estimation noise.

    This notebook runs the checklist on the 15 assets in `src/data/metering_data.parquet`
    (EV chargers + solar/battery, 14 days):

    1. Baselines first (`statsforecast`): Naive, SeasonalNaive(48), SeasonalNaive(336), seasonal window average, AutoETS.
    2. LightGBM (`mlforecast`) with **only day-ahead-safe lags** (>= 48), raw vs. `Differences([48])` target.
    3. Ensembles.
    4. Rolling-origin `cross_validation` with several windows, scored with **MASE** (scale-free), per-asset **win rate**, and error **by horizon**.

    Day-ahead framing: forecast 48 slots at 00:00 from data up to the previous midnight (`h=48`).
    """)
    return


@app.cell
def _(mo):
    n_windows = mo.ui.slider(3, 6, value=6, label="backtest days (windows)")
    n_windows
    return (n_windows,)


@app.cell
def _(pd, pl):
    # polars at the file boundary, pandas only because statsforecast/mlforecast want it.
    raw = pl.read_parquet("src/data/metering_data.parquet").sort("asset_id", "timestamp")
    df = raw.select(
        unique_id="asset_id", ds="timestamp", y="metering_kwh", asset_type="asset_type"
    ).to_pandas()
    df["is_solar"] = (df["asset_type"] == "solar_battery").astype(int)
    base = df[["unique_id", "ds", "y"]]
    asset_type = df.drop_duplicates("unique_id").set_index("unique_id")["asset_type"]
    H = 48
    KEYS = ["unique_id", "ds", "cutoff"]
    pd.DataFrame({"assets": [df.unique_id.nunique()], "rows": [len(df)], "start": [df.ds.min()], "end": [df.ds.max()]})
    return H, KEYS, asset_type, base, df


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Baselines (statsforecast)

    `Naive` is flat for 48 steps, so it is a weak benchmark at day-ahead. The two seasonal
    naives carry the daily and weekly shape for free. With only 14 days, `AutoETS(season_length=48)`
    has very little to learn from.
    """)
    return


@app.cell
def _(
    AutoETS, H, Naive, SeasonalNaive, SeasonalWindowAverage, StatsForecast, base, n_windows,
):
    sf = StatsForecast(
        models=[
            Naive(alias="Naive"),
            SeasonalNaive(season_length=48, alias="SN48"),
            SeasonalNaive(season_length=336, alias="SN336"),
            SeasonalWindowAverage(season_length=48, window_size=3, alias="SWA3"),
            AutoETS(season_length=48, model="ZZN", alias="ETS"),
        ],
        freq="30min",
    )
    cv_base = sf.cross_validation(df=base, h=H, n_windows=n_windows.value, step_size=H)
    cv_base.head()
    return (cv_base,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. LightGBM (mlforecast)

    Two things matter here:

    - **Lags >= 48 only.** A lag-1 feature would force recursive prediction and let errors compound across the day.
      With lags 48/96/144 plus a rolling mean on lag 48, one `predict(h=48)` is valid and nothing leaks.
      (Lag 336 is left out: with 14 days it would delete most training rows. 336 stays a *baseline*.)
    - **Target transform.** `Differences([48])` makes the model predict the *deviation from seasonal naive*, so it
      cannot lose the level. Compare it with the raw target, and with an extra `LocalStandardScaler`.
      Trees cannot extrapolate, so on level-shifting series the differenced version is the safer bet. On a stable
      series like this, the raw one may win. Look at the numbers rather than assume.

    `static_features=["is_solar"]` tells the single global model which assets are solar.
    """)
    return


@app.cell
def _(
    Differences, H, KEYS, LocalStandardScaler, MLForecast, RollingMean, cv_base, df, lgb,
    n_windows,
):
    def make_mlf(target_transforms):
        return MLForecast(
            models={"LGB": lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                random_state=0, verbose=-1,
            )},
            freq="30min",
            lags=[48, 96, 144],
            lag_transforms={48: [RollingMean(window_size=7)]},
            date_features=["hour", "dayofweek"],
            target_transforms=target_transforms,
        )

    variants = {
        "LGB_raw": None,
        "LGB_diff48": [Differences([48])],
        "LGB_diff48_std": [Differences([48]), LocalStandardScaler()],
    }
    cv_all = cv_base
    for _name, _tt in variants.items():
        _cv = make_mlf(_tt).cross_validation(
            df=df[["unique_id", "ds", "y", "is_solar"]], h=H, n_windows=n_windows.value,
            step_size=H, static_features=["is_solar"],
        )
        # Join on keys: mlforecast and statsforecast do not return rows in the same order.
        cv_all = cv_all.merge(
            _cv[KEYS + ["LGB"]].rename(columns={"LGB": _name}), on=KEYS, how="left"
        )
    cv_all.head()
    return (cv_all,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Ensembles

    Equal-weight means. An ensemble only helps if its members are individually reasonable.
    The first is the generic "ETS + seasonal naive + LGBM" recipe, the second uses
    the members that actually did well on this data (picked on the same backtest, so treat it as optimistic).
    """)
    return


@app.cell
def _(cv_all):
    cv_ens = cv_all.copy()
    cv_ens["Ens_generic"] = cv_ens[["ETS", "SN48", "LGB_diff48"]].mean(axis=1)
    cv_ens["Ens_SN336+LGB"] = cv_ens[["SN336", "LGB_raw"]].mean(axis=1)
    MODELS = [c for c in cv_ens.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    BASELINES = ["Naive", "SN48", "SN336", "SWA3", "ETS"]
    return BASELINES, MODELS, cv_ens


@app.cell
def _(MODELS, base, cv_ens, evaluate, mae, mase, partial, pd, rmse):
    # Scale for MASE = in-sample seasonal-naive(48) MAE on history before the first cutoff.
    train = base[base["ds"] <= cv_ens["cutoff"].min()]
    _long = cv_ens.drop(columns="cutoff")
    per_asset = evaluate(
        _long, metrics=[mae, rmse, partial(mase, seasonality=48)], train_df=train, models=MODELS,
    )
    per_asset["metric"] = per_asset["metric"].replace({"mase": "MASE"})
    per_asset = per_asset.melt(id_vars=["unique_id", "metric"], var_name="model", value_name="value")
    overall = per_asset.groupby(["model", "metric"])["value"].mean().unstack("metric")
    overall = overall[["mae", "rmse", "MASE"]]
    ref = overall.loc["SN48", "mae"]
    overall["skill vs SN48"] = 1 - overall["mae"] / ref
    best_base = overall.loc[["Naive", "SN48", "SN336", "SWA3", "ETS"], "mae"].min()
    overall["skill vs best baseline"] = 1 - overall["mae"] / best_base
    overall = overall.loc[MODELS].sort_values("mae")
    pd.options.display.float_format = "{:.3f}".format
    return overall, per_asset, train


@app.cell
def _(mo, overall):
    mo.vstack([
        mo.md("""
        ## 4. Scores (mean over 15 assets)

        MASE < 1 means better than in-sample seasonal naive(48). **Skill** = `1 - MAE / MAE_reference`
        (positive = better). "Best baseline" is the hardest of the five baselines, which is the bar that counts.
        """),
        overall.round(3),
    ])
    return


@app.cell
def _(BASELINES, MODELS, mo, overall, pd, per_asset):
    _best_base_name = overall.loc[BASELINES, "mae"].idxmin()
    _wins = []
    _mae = per_asset[per_asset["metric"] == "mae"].pivot(index="unique_id", columns="model", values="value")
    for _m in MODELS:
        if _m in BASELINES:
            continue
        _wins.append({
            "model": _m,
            f"beats {_best_base_name}": f"{(_mae[_m] < _mae[_best_base_name]).mean():.0%}",
            "beats SN48": f"{(_mae[_m] < _mae['SN48']).mean():.0%}",
            "worst asset ratio vs SN48": f"{(_mae[_m] / _mae['SN48']).max():.2f}x",
        })
    mo.vstack([
        mo.md(f"""
        ## 5. Per-asset win rate

        Share of the 15 assets where the model has lower MAE than the baseline. A model that wins on most
        assets but loses badly on a few is still useful. "worst asset ratio" shows how bad the bad ones are
        (above 1.00x = worse than SN48).
        """),
        pd.DataFrame(_wins).set_index("model"),
    ])
    return


@app.cell
def _(asset_type, mo, per_asset):
    _mae = per_asset[per_asset["metric"] == "MASE"]
    _by_type = (
        _mae.assign(asset_type=_mae["unique_id"].map(asset_type))
        .groupby(["asset_type", "model"])["value"].mean().unstack("asset_type").sort_values("ev_charging")
    )
    mo.vstack([
        mo.md("## 6. MASE by asset type\n\nAggregates hide structure: EV charging and solar+battery respond differently to each model."),
        _by_type.round(3),
    ])
    return


@app.cell
def _(MODELS, cv_ens, np, pd, px):
    # Error by horizon step (hours after the 00:00 forecast issue time).
    _d = cv_ens.copy()
    _d["step"] = ((_d["ds"] - _d["cutoff"]) / np.timedelta64(30, "m")).astype(int)
    _d["hour_ahead"] = (_d["step"] - 1) // 2 + 1
    _rows = []
    for _m in MODELS:
        _g = (_d["y"] - _d[_m]).abs().groupby(_d["hour_ahead"]).mean()
        _rows.append(_g.rename(_m))
    _h = (
        pd.concat(_rows, axis=1)
        .rename_axis("hour_ahead").reset_index()
        .melt(id_vars="hour_ahead", var_name="model", value_name="MAE")
    )
    fig_h = px.line(_h, x="hour_ahead", y="MAE", color="model", title="MAE by hours ahead of the 00:00 issue time")
    fig_h.update_layout(height=420, xaxis_title="hours ahead (1-24)")
    fig_h
    return


@app.cell
def _(cv_ens, mo):
    _ids = sorted(cv_ens["unique_id"].unique())
    asset_pick = mo.ui.dropdown(_ids, value="ASSET_001", label="asset")
    show = mo.ui.multiselect(
        ["SN48", "SN336", "LGB_raw", "LGB_diff48", "Ens_SN336+LGB", "ETS"], value=["SN336", "LGB_raw"], label="models"
    )
    mo.hstack([asset_pick, show])
    return asset_pick, show


@app.cell
def _(asset_pick, cv_ens, px, show):
    _last = cv_ens[(cv_ens["unique_id"] == asset_pick.value) & (cv_ens["cutoff"] == cv_ens["cutoff"].max())]
    _long = _last.melt(id_vars=["ds"], value_vars=["y", *show.value], var_name="series", value_name="kWh")
    fig_f = px.line(_long, x="ds", y="kWh", color="series", title=f"{asset_pick.value}: last backtest day")
    fig_f.update_traces(selector=dict(name="y"), line=dict(color="black", width=3))
    fig_f.update_layout(height=420)
    fig_f
    return


@app.cell
def _(BASELINES, mo, overall):
    _best_base = overall.loc[BASELINES, "mae"].idxmin()
    _winners = overall[(~overall.index.isin(BASELINES)) & (overall["mae"] < overall.loc[_best_base, "mae"])]
    if len(_winners):
        _top = _winners["mae"].idxmin()
        _txt = (
            f"**Yes.** Hardest baseline is `{_best_base}` (MAE {overall.loc[_best_base, 'mae']:.3f}). "
            f"`{_top}` beats it with MAE {overall.loc[_top, 'mae']:.3f} "
            f"({overall.loc[_top, 'skill vs best baseline']:.1%} skill)."
        )
    else:
        _txt = f"**No.** `{_best_base}` is still the best; tying with the baseline is the expected result on low-signal data."
    mo.md(f"""
    ## Verdict

    {_txt}

    Caveats: 14 days gives only a few backtest windows, so differences of a few percent are within noise.
    The winner was picked on the same windows it is scored on. Re-check as more history arrives. Extra data
    (calendar, tariff signals, weather) is the other lever, since without a real information advantage the
    model has the baseline's information plus parameters to fit.
    """)
    return


if __name__ == "__main__":
    app.run()
