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
    mo.vstack([
        mo.md(
            """
            # 2 · Evaluation: is the ML actually better?

            **Rolling-origin backtest**, day-ahead: for each test day we train only on earlier days, forecast the 48 slots,
            then score against what really happened (`MLForecast.cross_validation`). Five models compete, all inside one `MLForecast` object, and every chart shows all of them together:

            | Tier | Model | Idea |
            |---|---|---|
            | Simple | Seasonal naive | "Same as yesterday", widened by past errors |
            | Simple+ | Slot profile | Quantiles of each half-hour slot in the training window |
            | ML | Quantile GBM | LightGBM with `objective="quantile"`, one model per quantile |
            | ML | HistGB quantile | scikit-learn `HistGradientBoostingRegressor(loss="quantile")` |
            | ML | Linear quantile | scikit-learn `QuantileRegressor`: a straight-line model |
            """
        ),
    ])
    return


@app.cell
def _(c, mo):
    s, source = c.get_series()
    tiers = c.BASELINES + c.ML_MODELS  # all five models in one MLForecast
    with mo.status.spinner("Running backtest…"):
        bt = c.backtest(s, tiers=tiers)  # cross_validation with just these tiers
    tbl = c.metrics_table(bt)
    mo.md(f"**Data:** {source} · {bt['fold'].nunique()} test days × 48 slots = {bt['model'].eq(tiers[0]).sum()} scored points per model")
    return bt, s, tbl, tiers


@app.cell
def _(c, make_subplots, mo, tbl):
    _show = tbl.copy()
    _show["80% cover"] = (_show["80% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["90% cover"] = (_show["90% cover"] * 100).round(0).astype(int).astype(str) + "%"
    _show["skill vs naive"] = (_show["skill vs naive"] * 100).round(0).astype(int).astype(str) + "%"
    _show = _show.round({"MAE (kW)": 2, "RMSE (kW)": 2, "pinball": 3})

    _best = tbl["pinball"].idxmin()
    _cols = [c.TIER_COLORS[m] for m in tbl.index]
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
def _(bt, c, go, mo, np, pd, tiers):
    # observed share of actuals below each predicted quantile, per model
    _obs = {
        _name: [float(np.mean(bt[bt["model"] == _name]["y"] <= bt[bt["model"] == _name][f"q{q:.2f}"])) for q in c.QUANTILES]
        for _name in tiers
    }
    _fig = go.Figure()
    _fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash", width=1.5), name="perfect")
    for _name in tiers:
        _fig.add_scatter(
            x=c.QUANTILES, y=_obs[_name], mode="lines+markers", name=_name, line=dict(color=c.TIER_COLORS[_name], width=2),
            hovertemplate=_name + "<br>model says P%{x:.0%}<br>actually below it: %{y:.0%}<extra></extra>",
        )
    _fig.update_layout(
        title="Calibration (reliability): when the model says P90, how often is the actual below it?",
        xaxis=dict(title="quantile the model predicted", tickformat=".0%"),
        yaxis=dict(title="share of actuals that were below it", tickformat=".0%"),
        height=430, margin=dict(l=60, r=20, t=60, b=50), legend=dict(orientation="h", y=-0.22),
    )

    # the same numbers as a table: expected vs observed
    _tbl = pd.DataFrame({"model says": [f"P{int(q * 100)}" for q in c.QUANTILES], "should be below it": [f"{q:.0%}" for q in c.QUANTILES]})
    for _name in tiers:
        _tbl[_name] = [f"{v:.0%}" for v in _obs[_name]]

    # plain-language verdict on the central 80% band (P10 to P90)
    _lines = []
    for _name in tiers:
        _g = bt[bt["model"] == _name]
        _in = float(np.mean((_g["y"] >= _g["q0.10"]) & (_g["y"] <= _g["q0.90"])))
        _verdict = "about right" if abs(_in - 0.8) <= 0.07 else ("too narrow: overconfident" if _in < 0.8 else "too wide: underconfident")
        _lines.append(f"- **{_name}**: its P10 to P90 band (meant to hold 80%) held **{_in:.0%}** of actuals, so it is {_verdict}.")

    mo.vstack([
        mo.md("## Are the percentiles honest? (calibration)"),
        mo.md(
            """
            A forecast of "P90 = 40 kW" is a *promise*: nine times out of ten, the actual will come in at or below 40 kW.
            Calibration checks whether the promise is kept.

            **How the chart is made.** For every scored half-hour and every quantile level (P5, P10, ... P95) we ask
            "was the actual value below that prediction?". The share of times the answer is yes is the point on the chart.
            A perfectly honest model puts every point on the dashed diagonal: P10 is beaten 10% of the time, P50 half
            the time, P90 90% of the time.
            """
        ),
        _fig,
        mo.md("**The same numbers as a table** (the 'should be below it' column is what an honest model would show):"),
        mo.ui.table(_tbl, selection=None),
        mo.md("**Verdict for the central 80% band:**\n\n" + "\n".join(_lines)),
        mo.accordion({
            "How to read the shape of a line": mo.md(
                """
                | What you see | What it means | Practical effect |
                |---|---|---|
                | Line sits **on** the diagonal | Honest | Probabilities in notebook 01 can be taken at face value |
                | Whole line **above** the diagonal | Too many actuals fall below every quantile: forecasts run **high** | You over-predict load |
                | Whole line **below** the diagonal | Actuals keep exceeding the forecast: forecasts run **low** | You under-predict load |
                | Line **flatter** than the diagonal (low end above it, high end below it) | The tails are too thin: real values stray further out than the model allows | **Overconfident**: bands too narrow, "unlikely" events happen more than stated |
                | Line **steeper** than the diagonal (low end below it, high end above it) | The model's spread is wider than reality | **Underconfident**: bands too wide, wasted caution |

                Example: the model's P5 is beaten only 5% of the time if it is honest. If you see 14%, then 14 in 100
                actuals fell below what the model called a 1-in-20 low.
                """
            ),
            "Why it matters": mo.md(
                """
                Notebook 01 turns these quantiles into statements like "70% chance demand exceeds 40 kW".
                That statement is only as trustworthy as the calibration shown here.
                Accuracy (MAE) and calibration are separate: a model can have a good middle line and still be overconfident.

                **Small-sample warning:** with only a few test days, each point rests on a few hundred half-hours that are
                strongly correlated with their neighbours (errors persist for hours). Treat gaps of a few percentage points
                as noise, and look for consistent patterns across levels.
                """
            ),
        }),
    ])
    return


@app.cell
def _(bt, mo, tiers):
    _days = sorted(bt.index.normalize().unique())
    day_pick = mo.ui.dropdown({f"{d:%a %d %b}": d for d in _days}, value=f"{_days[-1]:%a %d %b}", label="Inspect test day")
    model_pick = mo.ui.dropdown(tiers, value="Quantile GBM", label="Show the P5–P95 / P25–P75 bands for")
    mo.vstack([mo.md("## Look at one day"), mo.hstack([day_pick, model_pick], justify="start", gap=2)])
    return day_pick, model_pick


@app.cell
def _(bt, c, day_pick, go, mo, model_pick, tiers):
    _day = bt[bt.index.normalize() == day_pick.value]
    _g = _day[_day["model"] == model_pick.value]
    _out = (_g["y"] < _g["q0.05"]) | (_g["y"] > _g["q0.95"])
    _c0 = c.TIER_COLORS[model_pick.value]
    _fig = go.Figure()
    _fig.add_scatter(x=_g.index, y=_g["q0.95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.05"], fill="tonexty", fillcolor="rgba(120,120,120,0.15)", line=dict(width=0),
                     name=f"{model_pick.value} P5–P95", hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_g.index, y=_g["q0.25"], fill="tonexty", fillcolor="rgba(120,120,120,0.30)", line=dict(width=0),
                     name=f"{model_pick.value} P25–P75", hoverinfo="skip")
    for _m in tiers:  # every model's median, so they can be compared directly
        _d = _day[_day["model"] == _m]
        _fig.add_scatter(x=_d.index, y=_d["q0.50"], name=f"{_m} P50", line=dict(color=c.TIER_COLORS[_m], width=3 if _m == model_pick.value else 1.6))
    _fig.add_scatter(x=_g.index, y=_g["y"], name="actual", mode="lines+markers", line=dict(color="black", width=1.5), marker=dict(size=4))
    _fig.add_scatter(x=_g.index[_out], y=_g["y"][_out], mode="markers", name="actual outside P5–P95",
                     marker=dict(color="red", size=11, symbol="circle-open", line=dict(width=2)))
    _fig.update_layout(
        yaxis_title="kW", hovermode="x unified", height=420, margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([
        _fig,
        mo.md(
            f"Coloured lines: every model's median (P50) for this day. Grey bands and red rings belong to **{model_pick.value}** only "
            f"(change it above): {int(_out.sum())} of 48 points fall outside its P5–P95 (expect about 5 if calibrated)."
        ),
    ])
    return


@app.cell
def _(bt, c, go, mo, tiers):
    _cols = c.TIER_COLORS
    _fig = go.Figure()
    for _name in tiers:
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
            per model. All 21 models (3 tiers x 7 quantiles) are fitted and scored in that single call. Change the ML model above and only that tier is swapped.
            """
        ),
        _fig,
    ])
    return


if __name__ == "__main__":
    app.run()
