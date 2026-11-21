# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/02_evaluation.py
#   uv run marimo run  mlforecast/02_evaluation.py   (app mode, code hidden)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, mo, np, pd, plt


@app.cell
def _(mo):
    noise = mo.ui.slider(1, 10, value=4, label="Demo-data noise (kW)")
    mo.vstack([
        mo.md(
            """
            # 2 · Evaluation: is the ML actually better?

            **Rolling-origin backtest**, day-ahead: for each test day we train only on earlier days, forecast the 48 slots,
            then score against what really happened (`MLForecast.cross_validation`). Three tiers compete, all as models inside one `MLForecast` object:

            | Tier | Model | Idea |
            |---|---|---|
            | Simple | Seasonal naive | "Same as yesterday", widened by past errors |
            | Simple+ | Slot profile | Quantiles of each half-hour slot in the training window |
            | ML | Quantile GBM | LightGBM (one model per quantile) on mlforecast lag + calendar features |
            """
        ),
        mo.hstack([noise], justify="start", gap=2),
    ])
    return (noise,)


@app.cell
def _(c, mo, noise):
    s, source = c.get_series(noise.value)
    with mo.status.spinner("Running backtest…"):
        bt = c.backtest(s)
    tbl = c.metrics_table(bt)
    mo.md(f"**Data:** {source} · {bt['fold'].nunique()} test days × 48 slots = {bt['model'].eq('Quantile GBM').sum()} scored points per model")
    return bt, tbl


@app.cell
def _(mo, plt, tbl):
    _show = tbl.copy()
    _show["80% cover"] = (_show["80% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["90% cover"] = (_show["90% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["skill vs naive"] = (_show["skill vs naive"] * 100).round(0).astype(int).astype(str) + "%"
    _show = _show.round({"MAE (kW)": 2, "RMSE (kW)": 2, "pinball": 3})

    _best = tbl["pinball"].idxmin()
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 3))
    _cols = ["C7", "C0", "C1"]
    _axs[0].bar(tbl.index, tbl["MAE (kW)"], color=_cols)
    _axs[0].set_title("Median error (MAE, kW) · lower is better")
    _axs[1].bar(tbl.index, tbl["pinball"], color=_cols)
    _axs[1].set_title("Pinball loss (whole distribution) · lower is better")
    for _a in _axs:
        _a.tick_params(axis="x", labelsize=8)
    _fig.tight_layout()

    mo.vstack([
        mo.md("## Scoreboard"),
        mo.ui.table(_show.reset_index(), selection=None),
        _fig,
        mo.callout(
            mo.md(
                f"**Best on pinball loss: {_best}** (skill vs naive: {tbl.loc[_best, 'skill vs naive'] * 100:.0f}%). "
                "*Skill* = how much lower its loss is than 'same as yesterday'. "
                "If the ML model isn't clearly beating the simple profile, the extra complexity isn't paying for itself yet."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _(bt, c, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(4.6, 4.2))
    _ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for _name, _col in zip(c.MODELS, ["C7", "C0", "C1"]):
        _g = bt[bt["model"] == _name]
        _obs = [np.mean(_g["y"] <= _g[f"q{q:.2f}"]) for q in c.QUANTILES]
        _ax.plot(c.QUANTILES, _obs, "o-", color=_col, label=_name)
    _ax.set_xlabel("predicted quantile level")
    _ax.set_ylabel("share of actuals below it")
    _ax.set_title("Calibration (reliability)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(
            """
            ## Are the percentiles honest?

            If the model says P90, about **90%** of actual values should fall below it.

            - Line **above** the diagonal → forecasts too high (over-predicting)
            - Line **below** → too low
            - Line **flatter** than the diagonal → intervals too narrow (overconfident)
            - Line **steeper** → intervals too wide

            This matters most for the likelihood notebook: probabilities are only as good as this chart.
            """
        ),
    ], align="center", gap=2)
    return


@app.cell
def _(bt, mo):
    _days = sorted(bt.index.normalize().unique())
    day_pick = mo.ui.dropdown({f"{d:%a %d %b}": d for d in _days}, value=f"{_days[-1]:%a %d %b}", label="Inspect test day")
    model_pick = mo.ui.dropdown(["Seasonal naive", "Slot profile", "Quantile GBM"], value="Quantile GBM", label="model")
    mo.vstack([mo.md("## Look at one day"), mo.hstack([day_pick, model_pick], justify="start", gap=2)])
    return day_pick, model_pick


@app.cell
def _(bt, day_pick, mo, model_pick, plt):
    _g = bt[(bt["model"] == model_pick.value) & (bt.index.normalize() == day_pick.value)]
    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    _ax.fill_between(_g.index, _g["q0.05"], _g["q0.95"], color="C0", alpha=0.15, label="P5–P95")
    _ax.fill_between(_g.index, _g["q0.25"], _g["q0.75"], color="C0", alpha=0.30, label="P25–P75")
    _ax.plot(_g.index, _g["q0.50"], color="C0", lw=2, label="P50")
    _ax.plot(_g.index, _g["y"], color="k", lw=1.2, marker=".", label="actual")
    _out = (_g["y"] < _g["q0.05"]) | (_g["y"] > _g["q0.95"])
    _ax.scatter(_g.index[_out], _g["y"][_out], color="red", zorder=5, label="outside P5–P95")
    _ax.legend(ncols=5, fontsize=8, loc="upper left")
    _ax.set_ylabel("kW")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"{_out.sum()} of 48 points outside P5–P95 (expect about 5 if calibrated).")])
    return


@app.cell
def _(bt, c, mo, plt):
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    for _name, _col in zip(c.MODELS, ["C7", "C0", "C1"]):
        _g = bt[bt["model"] == _name]
        _per = _g.groupby(_g.index.normalize()).apply(lambda d: c.pinball(d["y"].to_numpy(), d[c.qcols()].to_numpy()), include_groups=False)
        _ax.plot(_per.index, _per.values, "o-", color=_col, label=_name)
    _ax.set_ylabel("pinball loss")
    _ax.set_title("Loss by test day: is one model consistently better, or does it depend on the day?")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("## Consistency across days"), _fig])
    return


@app.cell
def _(bt, mo, plt, s):
    _starts = bt.groupby("fold").apply(lambda d: d.index.min(), include_groups=False)
    _t0 = s.index[0]
    _fig, _ax = plt.subplots(figsize=(10, 0.45 * len(_starts) + 1.2))
    for _f, _d0 in _starts.items():
        _ax.barh(_f, (_d0 - _t0).total_seconds() / 86400, left=0, color="0.75", height=0.6, label="train (refit each fold)" if _f == 0 else None)
        _ax.barh(_f, 1, left=(_d0 - _t0).total_seconds() / 86400, color="C1", height=0.6, label="test day (h=48)" if _f == 0 else None)
    _ax.invert_yaxis()
    _ax.set_xlabel("days since start of data")
    _ax.set_ylabel("fold")
    _ax.legend(fontsize=8, loc="lower left")
    _ax.set_title("How cross_validation slices the data")
    _fig.tight_layout()
    mo.vstack([
        mo.md(
            """
            ## Under the hood: `MLForecast.cross_validation`

            One call produced every number above:

            ```python
            fcst.cross_validation(df, n_windows=6, h=48, step_size=48, refit=True)
            ```

            - `h=48`: each window forecasts one day (48 half-hours).
            - `step_size=48`: the next window starts one day later.
            - `n_windows`: how many windows, counted back from the end of the data.
            - `refit=True`: retrain on everything before each window (honest, slower). `False` fits once
              and reuses it, which is fast but trains on data that comes *after* early windows.

            The output has one row per (timestamp, window) with `y`, a `cutoff` column and one prediction column
            per model. All 21 models (3 tiers x 7 quantiles) are fitted and scored in that single call.
            """
        ),
        _fig,
    ])
    return


if __name__ == "__main__":
    app.run()
