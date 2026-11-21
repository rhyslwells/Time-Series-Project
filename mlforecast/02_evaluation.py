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
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, make_subplots, mo, np, pd


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
    return bt, s, tbl


@app.cell
def _(make_subplots, mo, tbl):
    _show = tbl.copy()
    _show["80% cover"] = (_show["80% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["90% cover"] = (_show["90% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["skill vs naive"] = (_show["skill vs naive"] * 100).round(0).astype(int).astype(str) + "%"
    _show = _show.round({"MAE (kW)": 2, "RMSE (kW)": 2, "pinball": 3})

    _best = tbl["pinball"].idxmin()
    _cols = ["#7f7f7f", "#1f77b4", "#ff7f0e"]
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("Median error (MAE, kW)", "Pinball loss (whole distribution)"))
    _fig.add_bar(x=list(tbl.index), y=tbl["MAE (kW)"], marker_color=_cols, row=1, col=1, showlegend=False)
    _fig.add_bar(x=list(tbl.index), y=tbl["pinball"], marker_color=_cols, row=1, col=2, showlegend=False)
    _fig.update_layout(height=320, margin=dict(l=50, r=20, t=50, b=40))

    mo.vstack([
        mo.md("## Scoreboard"),
        mo.ui.table(_show.reset_index(), selection=None),
        _fig,
        mo.callout(
            mo.md(
                f"**Best on pinball loss: {_best}** (skill vs naive: {tbl.loc[_best, 'skill vs naive'] * 100:.0f}%). "
                "*Skill* = how much lower its loss is than 'same as yesterday'. Lower bars are better. "
                "If the ML model isn't clearly beating the simple profile, the extra complexity isn't paying for itself yet."
            ),
            kind="success",
        ),
    ])
    return


@app.cell
def _(bt, c, go, mo, np):
    _cols = {"Seasonal naive": "#7f7f7f", "Slot profile": "#1f77b4", "Quantile GBM": "#ff7f0e"}
    _fig = go.Figure()
    _fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash", width=1), name="perfect")
    for _name in c.MODELS:
        _g = bt[bt["model"] == _name]
        _obs = [float(np.mean(_g["y"] <= _g[f"q{q:.2f}"])) for q in c.QUANTILES]
        _fig.add_scatter(x=c.QUANTILES, y=_obs, mode="lines+markers", name=_name, line=dict(color=_cols[_name], width=2))
    _fig.update_layout(
        title="Calibration (reliability)", xaxis_title="predicted quantile level", yaxis_title="share of actuals below it",
        height=420, width=520, margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.25),
    )
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
def _(bt, day_pick, go, mo, model_pick):
    _g = bt[(bt["model"] == model_pick.value) & (bt.index.normalize() == day_pick.value)]
    _out = (_g["y"] < _g["q0.05"]) | (_g["y"] > _g["q0.95"])
    _fig = go.Figure()
    _fig.add_scatter(x=_g.index, y=_g["q0.95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.05"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                     name="P5–P95", hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.25"], fill="tonexty", fillcolor="rgba(31,119,180,0.30)", line=dict(width=0),
                     name="P25–P75", hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.50"], name="P50", line=dict(color="#1f77b4", width=2.5))
    _fig.add_scatter(x=_g.index, y=_g["y"], name="actual", mode="lines+markers", line=dict(color="black", width=1.2), marker=dict(size=4))
    _fig.add_scatter(x=_g.index[_out], y=_g["y"][_out], mode="markers", name="outside P5–P95",
                     marker=dict(color="red", size=11, symbol="circle-open", line=dict(width=2)))
    _fig.update_layout(
        yaxis_title="kW", hovermode="x unified", height=380, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([_fig, mo.md(f"{int(_out.sum())} of 48 points outside P5–P95 (expect about 5 if calibrated).")])
    return


@app.cell
def _(bt, c, go, mo):
    _cols = {"Seasonal naive": "#7f7f7f", "Slot profile": "#1f77b4", "Quantile GBM": "#ff7f0e"}
    _fig = go.Figure()
    for _name in c.MODELS:
        _g = bt[bt["model"] == _name]
        _per = _g.groupby(_g.index.normalize()).apply(
            lambda d: c.pinball(d["y"].to_numpy(), d[c.qcols()].to_numpy()), include_groups=False
        )
        _fig.add_scatter(x=_per.index, y=_per.values, mode="lines+markers", name=_name, line=dict(color=_cols[_name], width=2))
    _fig.update_layout(
        title="Loss by test day: is one model consistently better, or does it depend on the day?",
        yaxis_title="pinball loss", hovermode="x unified", height=320, margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    _fig.update_xaxes(tickformat="%a %d %b")
    mo.vstack([mo.md("## Consistency across days"), _fig])
    return


@app.cell
def _(bt, go, mo, s):
    _starts = bt.groupby("fold").apply(lambda d: d.index.min(), include_groups=False)
    _t0 = s.index[0]
    _off = [(_d0 - _t0).total_seconds() / 86400 for _d0 in _starts.values]
    _names = [f"fold {f}" for f in _starts.index]
    _fig = go.Figure()
    _fig.add_bar(y=_names, x=_off, base=0, orientation="h", marker_color="#bbbbbb", name="train (refit each fold)",
                 hovertemplate="train: first %{x:.0f} days<extra></extra>")
    _fig.add_bar(y=_names, x=[1] * len(_off), base=_off, orientation="h", marker_color="#ff7f0e", name="test day (h=48)",
                 hovertemplate="test day<extra></extra>")
    _fig.update_layout(
        barmode="overlay", title="How cross_validation slices the data", xaxis_title="days since start of data",
        yaxis=dict(autorange="reversed"), height=50 * len(_names) + 170, margin=dict(l=70, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.3),
    )
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
