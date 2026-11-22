# Run from the project root (Time Series Project):
#   uv run marimo edit autogluon/01_beat_naive.py
#   uv run marimo run  autogluon/01_beat_naive.py   (app mode, code hidden)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    import warnings
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import polars as pl
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    warnings.filterwarnings("ignore")
    return Path, TimeSeriesDataFrame, TimeSeriesPredictor, mo, np, pd, pl, px, shutil


@app.cell
def _(mo):
    mo.md(r"""
    # Can AutoGluon beat the naive baselines?

    Same question as `mlforecast/07_beat_naive.py`, answered with `autogluon.timeseries`.
    The bar is **seasonal naive** (same slot yesterday, `m=48`), not plain naive.

    - Data: `mlforecast/data/ev_long.parquet` (30 synthetic EV charge points, 8 weeks, 30-min).
    - Day-ahead framing: forecast 48 slots from each midnight (`prediction_length=48`).
    - The predictor is **fit once** on history before the backtest. Each backtest day is then forecast from
      its own midnight with the *fitted* models (no refit), so the scores are rolling-origin out-of-sample.
    - AutoGluon's own `SeasonalNaive` / `Naive` sit in the same model zoo, so baselines and challengers share
      one code path. Scores are recomputed in numpy (MAE, RMSE, MASE) so they match the mlforecast notebook.
    """)
    return


@app.cell
def _(mo):
    n_windows = mo.ui.slider(3, 10, value=7, label="backtest days (windows)")
    run = mo.ui.run_button(label="Fit AutoGluon")
    mo.vstack([n_windows, run])
    return n_windows, run


@app.cell
def _(TimeSeriesDataFrame, n_windows, pd, pl):
    H = 48
    STEP = pd.Timedelta("30min")
    raw = pl.read_parquet("mlforecast/data/ev_long.parquet").sort("unique_id", "ds").to_pandas()
    full = TimeSeriesDataFrame.from_data_frame(raw, id_column="unique_id", timestamp_column="ds")
    _last = raw["ds"].max()
    # Midnight cutoffs: history <= cutoff, then forecast the next 48 slots.
    first_cutoff = _last - n_windows.value * H * STEP
    cutoffs = [first_cutoff + k * H * STEP for k in range(n_windows.value)]
    _ts = full.index.get_level_values("timestamp")
    train = TimeSeriesDataFrame(full[_ts <= first_cutoff])
    pd.DataFrame({
        "assets": [raw["unique_id"].nunique()], "train_end": [first_cutoff], "data_end": [_last],
    })
    return H, STEP, cutoffs, full, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Fit

    Hand-picked model zoo instead of a preset, so the run is short and the baselines are explicit:

    - Baselines: `Naive`, `SeasonalNaive` (m=48).
    - Challengers: `DirectTabular` and `RecursiveTabular` (gradient boosting on lags, pooled over all assets),
      `ETS` and `Theta` with `seasonal_period=48`, plus the `WeightedEnsemble` AutoGluon builds on top.
    - `eval_metric="MASE"` with `eval_metric_seasonal_period=48` makes model selection and ensemble weights
      optimise against the seasonal-naive scale.
    """)
    return


@app.cell
def _(H, Path, TimeSeriesPredictor, mo, run, shutil, train):
    mo.stop(not run.value, mo.md("Press **Fit AutoGluon** to train."))
    _path = Path("autogluon/models/beat_naive")
    shutil.rmtree(_path, ignore_errors=True)
    hp = {
        "Naive": {},
        "SeasonalNaive": {"seasonal_period": 48},
        "DirectTabular": [
            {"model_name": "GBM", "ag_args": {"name_suffix": "LGB"}},
            {"model_name": "XGB", "ag_args": {"name_suffix": "XGB"}},
            {"model_name": "CAT", "ag_args": {"name_suffix": "Cat"}},
        ],
        "RecursiveTabular": [
            {"model_name": "GBM", "ag_args": {"name_suffix": "LGB"}},
            {"model_name": "XGB", "ag_args": {"name_suffix": "XGB"}},
        ],
        "PerStepTabular": {},
        "ETS": {"seasonal_period": 48},
        "Theta": {"seasonal_period": 48},
    }
    predictor = TimeSeriesPredictor(
        path=str(_path),
        prediction_length=H,
        freq="30min",
        target="y",
        eval_metric="MASE",
        eval_metric_seasonal_period=48,
        verbosity=1,
    ).fit(train, hyperparameters=hp, enable_ensemble=True)
    models = predictor.model_names()
    models
    return models, predictor


@app.cell
def _(H, STEP, TimeSeriesDataFrame, cutoffs, full, models, pd, predictor):
    # Rolling-origin forecasts with the fitted models, one call per (model, cutoff).
    _ts = full.index.get_level_values("timestamp")
    _rows = []
    for _c in cutoffs:
        _hist = TimeSeriesDataFrame(full[_ts <= _c])
        _actual = full[(_ts > _c) & (_ts <= _c + H * STEP)]["y"].rename("y")
        for _m in models:
            _p = predictor.predict(_hist, model=_m)["mean"].rename("pred")
            _j = pd.concat([_actual, _p], axis=1).dropna().reset_index()
            _j["model"] = _m
            _j["cutoff"] = _c
            _rows.append(_j)
    cv = pd.concat(_rows, ignore_index=True).rename(columns={"item_id": "unique_id", "timestamp": "ds"})
    return (cv,)


@app.cell
def _(cutoffs, cv, full, np, pd):
    # MASE scale: in-sample seasonal-naive(48) MAE on history before the first cutoff.
    _hist = full[full.index.get_level_values("timestamp") <= cutoffs[0]]["y"]
    _scale = _hist.groupby(level="item_id").apply(lambda s: np.abs(s.values[48:] - s.values[:-48]).mean())
    _scale = _scale.clip(lower=1e-6)

    _e = cv.assign(abs_err=(cv["y"] - cv["pred"]).abs(), sq_err=(cv["y"] - cv["pred"]) ** 2)
    per_asset = _e.groupby(["model", "unique_id"]).agg(mae=("abs_err", "mean"), mse=("sq_err", "mean")).reset_index()
    per_asset["rmse"] = np.sqrt(per_asset["mse"])
    per_asset["MASE"] = per_asset["mae"] / per_asset["unique_id"].map(_scale)

    overall = per_asset.groupby("model")[["mae", "rmse", "MASE"]].mean()
    BASELINES = [m for m in ("Naive", "SeasonalNaive") if m in overall.index]
    overall["skill vs SeasonalNaive"] = 1 - overall["mae"] / overall.loc["SeasonalNaive", "mae"]
    overall = overall.sort_values("mae")
    pd.options.display.float_format = "{:.3f}".format
    return BASELINES, overall, per_asset


@app.cell
def _(mo, overall):
    mo.vstack([
        mo.md("""
        ## 2. Scores (mean over assets, all backtest days)

        MASE < 1 means better than in-sample seasonal naive(48). **Skill** = `1 - MAE / MAE_SeasonalNaive`
        (positive = better). `WeightedEnsemble` is AutoGluon's blend of the other models, with weights chosen
        on a validation window inside the training data (not on these backtest days).
        """),
        overall.round(3),
    ])
    return


@app.cell
def _(BASELINES, mo, overall, pd, per_asset):
    _mae = per_asset.pivot(index="unique_id", columns="model", values="mae")
    _rows = []
    for _m in overall.index:
        if _m in BASELINES:
            continue
        _rows.append({
            "model": _m,
            "beats SeasonalNaive": f"{(_mae[_m] < _mae['SeasonalNaive']).mean():.0%}",
            "worst asset ratio": f"{(_mae[_m] / _mae['SeasonalNaive']).max():.2f}x",
        })
    mo.vstack([
        mo.md("""
        ## 3. Per-asset win rate

        Share of assets where the model has lower MAE than `SeasonalNaive`. "worst asset ratio" above 1.00x
        means worse than seasonal naive on that asset.
        """),
        pd.DataFrame(_rows).set_index("model"),
    ])
    return


@app.cell
def _(cv, np):
    # Forecasts are issued at 00:00, so the horizon step is also the time of day.
    by_slot = cv.copy()
    by_slot["slot"] = ((by_slot["ds"] - by_slot["cutoff"]) / np.timedelta64(30, "m")).astype(int)
    by_slot["time"] = ((by_slot["slot"] - 1) // 2).astype(str).str.zfill(2) + ":" + np.where(
        by_slot["slot"] % 2 == 1, "00", "30"
    )
    by_slot["abs_err"] = (by_slot["y"] - by_slot["pred"]).abs()
    return (by_slot,)


@app.cell
def _(by_slot, mo, overall):
    _default = list(dict.fromkeys(["SeasonalNaive", *overall.index[:2]]))
    show = mo.ui.multiselect(list(overall.index), value=_default, label="models to plot")
    mo.vstack([
        mo.md("""
        ## 4. What the models do across the forecast day

        Averages over all assets and all backtest days, by half-hour slot of the 48-step forecast
        (slot 00:00 is the first step, 23:30 the last). The chart shows mean absolute error per slot,
        i.e. where in the day each model is good or bad.
        """),
        show,
    ])
    return (show,)


@app.cell
def _(by_slot, px, show):
    _m = by_slot[by_slot["model"].isin(show.value)]
    _h = _m.groupby(["slot", "time", "model"])["abs_err"].mean().reset_index(name="MAE").sort_values("slot")
    fig_mae = px.line(_h, x="time", y="MAE", color="model", title="Mean MAE per half-hour slot of the forecast day")
    fig_mae.update_layout(height=420, xaxis_title="time of day (forecast issued 00:00)")
    fig_mae
    return


@app.cell
def _(BASELINES, mo, overall):
    _bar = overall.loc["SeasonalNaive", "mae"]
    _winners = overall[(~overall.index.isin(BASELINES)) & (overall["mae"] < _bar)]
    if len(_winners):
        _top = _winners["mae"].idxmin()
        _txt = (
            f"**Yes.** `SeasonalNaive` has MAE {_bar:.3f}. `{_top}` beats it with MAE "
            f"{overall.loc[_top, 'mae']:.3f} ({overall.loc[_top, 'skill vs SeasonalNaive']:.1%} skill)."
        )
    else:
        _txt = f"**No.** `SeasonalNaive` (MAE {_bar:.3f}) is still the best."
    mo.md(f"""
    ## Verdict

    {_txt}

    Caveats: a handful of backtest days is a small sample, so gaps of a few percent are noise, and the
    "best row" is picked on the same days it is scored on. Spiky session-like load is hard for every model;
    any gain over seasonal naive should come from pooled learning across assets, not single-series fitting.
    """)
    return


if __name__ == "__main__":
    app.run()
