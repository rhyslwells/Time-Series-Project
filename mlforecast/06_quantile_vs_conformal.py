# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/06_quantile_vs_conformal.py
#   uv run marimo run  mlforecast/06_quantile_vs_conformal.py   (app mode, code hidden)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from mlforecast import MLForecast
    from mlforecast.utils import PredictionIntervals

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return MLForecast, PredictionIntervals, c, go, lgb, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 6 · Quantile models vs conformal intervals

    Two ways to get an 80% prediction interval out of mlforecast. Same features, same LightGBM, same backtest.
    Only the *source of the uncertainty* differs.

    | | Quantile models | Conformal intervals |
    |---|---|---|
    | Models fitted | **two**: one trained for the 10th percentile, one for the 90th (`objective="quantile"`) | **one** ordinary point model (squared error) |
    | Where the width comes from | the models predict the spread directly | measured past errors of the point model on held-out data |
    | mlforecast feature used | just more entries in `models={...}` | `PredictionIntervals(n_windows=k, h=48)` + `level=[80]` |
    | Extra data needed | none | `k` extra windows of history to measure errors on |

    **How conformal works in mlforecast.** Before predicting, it runs `k` backtest windows, each forecasting `h=48` steps.
    For every step ahead it collects the absolute errors, then takes their 80th percentile as the half-width:

    `interval = point forecast ± (80th percentile of past errors at that step)`

    Because we forecast at midnight, "step ahead" is the same as "half-hour slot of the day". So conformal here
    *can* be wider at midday than at 3am, but each slot's width is estimated from only `k` numbers. Move the
    calibration-windows slider and watch how noisy that gets.
    """)
    return


@app.cell
def _(mo):
    noise = mo.ui.slider(1, 10, value=4, label="Demo-data noise (kW)", show_value=True)
    days = mo.ui.slider(21, 56, value=28, step=7, label="Days of data", show_value=True)
    test_windows = mo.ui.slider(2, 7, value=4, label="Backtest days", show_value=True)
    calib = mo.ui.slider(2, 8, value=3, label="Conformal calibration windows (k)", show_value=True)
    mo.vstack([
        mo.hstack([noise, days], justify="start", gap=2),
        mo.hstack([test_windows, calib], justify="start", gap=2),
    ])
    return calib, days, noise, test_windows


@app.cell
def _(MLForecast, PredictionIntervals, c, calib, days, lgb, mo, noise, pd, test_windows):
    _s = c.make_synthetic(days=days.value, noise=noise.value)
    df = c.to_long(_s)

    def _gbm(**kw):
        return lgb.LGBMRegressor(
            max_depth=3, n_estimators=120, learning_rate=0.05, min_child_samples=10, random_state=0, verbose=-1, **kw
        )

    def run_quantile(df, n_test):
        f = MLForecast(
            models={"lo": _gbm(objective="quantile", alpha=0.10), "mid": _gbm(objective="quantile", alpha=0.50),
                    "hi": _gbm(objective="quantile", alpha=0.90)},
            **c.FEATURE_KWARGS,
        )
        r = f.cross_validation(df, n_windows=n_test, h=48, step_size=48, refit=True)
        return r.rename(columns={"lo": "lo", "mid": "pred", "hi": "hi"})

    def run_conformal(df, n_test, k):
        f = MLForecast(models={"point": _gbm()}, **c.FEATURE_KWARGS)
        r = f.cross_validation(
            df, n_windows=n_test, h=48, step_size=48, refit=True,
            prediction_intervals=PredictionIntervals(n_windows=k, h=48), level=[80],
        )
        return r.rename(columns={"point": "pred", "point-lo-80": "lo", "point-hi-80": "hi"})

    _ok = calib.value * 48 + 48 + test_windows.value * 48 + 96 < len(df)
    if not _ok:
        mo.stop(True, mo.callout(mo.md("Not enough days for that many calibration + backtest windows. Add data or reduce a slider."), kind="warn"))
    with mo.status.spinner("Running both backtests…"):
        qdf = run_quantile(df, test_windows.value)
        cdf = run_conformal(df, test_windows.value, calib.value)
    mo.md(f"Backtested **{test_windows.value}** days × 48 slots with each method (refit before each day).")
    return cdf, df, qdf, run_conformal


@app.cell
def _(cdf, go, mo, qdf):
    _day = qdf["cutoff"].max()
    _q, _c = qdf[qdf["cutoff"] == _day], cdf[cdf["cutoff"] == _day]
    _fig = go.Figure()
    _fig.add_scatter(x=_q["ds"], y=_q["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_q["ds"], y=_q["lo"], fill="tonexty", fillcolor="rgba(31,119,180,0.20)", line=dict(width=0),
                     name="quantile 80% interval", hoverinfo="skip")
    _fig.add_scatter(x=_c["ds"], y=_c["hi"], line=dict(color="rgba(255,127,14,0.9)", width=1.5, dash="dot"), name="conformal 80% (upper)")
    _fig.add_scatter(x=_c["ds"], y=_c["lo"], line=dict(color="rgba(255,127,14,0.9)", width=1.5, dash="dot"), name="conformal 80% (lower)")
    _fig.add_scatter(x=_q["ds"], y=_q["pred"], name="quantile median", line=dict(color="#1f77b4", width=2))
    _fig.add_scatter(x=_c["ds"], y=_c["pred"], name="conformal point forecast", line=dict(color="#ff7f0e", width=2))
    _fig.add_scatter(x=_q["ds"], y=_q["y"], name="actual", line=dict(color="black", width=1.5))
    _fig.update_layout(
        title="Last backtest day: both 80% intervals", yaxis_title="kW", hovermode="x unified",
        height=400, margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([
        mo.md("## One day, both intervals"),
        _fig,
        mo.md(
            "Blue band: from two quantile models. Orange dotted lines: conformal, centred on its own point forecast. "
            "Look at where each is wide or narrow, and whether the actual (black) stays inside."
        ),
    ])
    return


@app.cell
def _(cdf, mo, np, pd, qdf):
    def score(r, alpha=0.2):
        y, lo, hi = r["y"].to_numpy(), r["lo"].to_numpy(), r["hi"].to_numpy()
        width = hi - lo
        winkler = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        return {
            "coverage (target 80%)": f"{((y >= lo) & (y <= hi)).mean() * 100:.0f}%",
            "mean width (kW)": round(float(width.mean()), 2),
            "interval score (lower = better)": round(float(winkler.mean()), 2),
            "MAE of centre (kW)": round(float(np.abs(y - r["pred"].to_numpy()).mean()), 2),
        }

    _tbl = pd.DataFrame({"Quantile models": score(qdf), "Conformal": score(cdf)}).T
    mo.vstack([
        mo.md("## Scoreboard"),
        mo.ui.table(_tbl.reset_index(names="method"), selection=None),
        mo.md(
            """
            - **Coverage** should sit near 80%. Below = intervals too narrow (overconfident). Above = too wide.
            - **Mean width** on its own rewards being narrow; coverage on its own rewards being wide.
            - **Interval score** (Winkler) combines them: width plus a penalty for every kW the actual falls outside.
              It is the fairest single number for comparing intervals.
            - **MAE of centre** compares the middle lines: quantile uses its P50 model, conformal its point model.
            """
        ),
    ])
    return


@app.cell
def _(cdf, go, mo, qdf):
    def _by_slot(r):
        r = r.assign(slot=r["ds"].dt.hour * 2 + r["ds"].dt.minute // 30)
        r["width"] = r["hi"] - r["lo"]
        r["inside"] = ((r["y"] >= r["lo"]) & (r["y"] <= r["hi"])).astype(float)
        return r.groupby("slot")[["width", "inside"]].mean()

    _q, _c = _by_slot(qdf), _by_slot(cdf)
    _lab = [f"{h // 2:02d}:{'30' if h % 2 else '00'}" for h in range(48)]
    _fig = go.Figure()
    _fig.add_scatter(x=_lab, y=_q["width"], name="quantile", line=dict(color="#1f77b4", width=2))
    _fig.add_scatter(x=_lab, y=_c["width"], name="conformal", line=dict(color="#ff7f0e", width=2))
    _fig.update_layout(
        title="Mean interval width by time of day", yaxis_title="kW", height=320,
        margin=dict(l=50, r=20, t=50, b=40), hovermode="x unified", legend=dict(orientation="h", y=-0.25),
    )
    _fig.update_xaxes(tickvals=_lab[::4])
    _fig2 = go.Figure()
    _fig2.add_scatter(x=_lab, y=_q["inside"] * 100, name="quantile", line=dict(color="#1f77b4", width=2))
    _fig2.add_scatter(x=_lab, y=_c["inside"] * 100, name="conformal", line=dict(color="#ff7f0e", width=2))
    _fig2.add_hline(y=80, line=dict(color="black", dash="dash", width=1), annotation_text="target 80%")
    _fig2.update_layout(
        title="Share of actuals inside the interval, by time of day", yaxis=dict(title="%", range=[0, 105]), height=320,
        margin=dict(l=50, r=20, t=50, b=40), hovermode="x unified", legend=dict(orientation="h", y=-0.25),
    )
    _fig2.update_xaxes(tickvals=_lab[::4])
    mo.vstack([
        mo.md("## Does the width follow the uncertainty?"),
        _fig,
        _fig2,
        mo.md(
            """
            In this synthetic data, midday is more uncertain than overnight (unknown cloud cover multiplies the solar curve).
            A good interval is **wider at midday and narrower overnight**, and covers about 80% in *every* slot.

            - Quantile models learn width from the features (`slot`, yesterday's level), so they can follow the shape.
            - Conformal estimates each slot's width from only `k` past errors. The line is jagged because each point
              is a percentile of a handful of numbers. Coverage per slot wobbles accordingly.

            With only a few backtest days, per-slot coverage here is itself noisy. Raise **Backtest days** and **Days of data** to firm it up.
            """
        ),
    ])
    return


@app.cell
def _(calib, df, go, mo, np, qdf, run_conformal, test_windows):
    _ks = [k for k in range(2, 9) if k * 48 + 48 + test_windows.value * 48 + 96 < len(df)]
    _rows = []
    for _k in _ks:
        _r = run_conformal(df, test_windows.value, _k)
        _in = ((_r["y"] >= _r["lo"]) & (_r["y"] <= _r["hi"])).mean() * 100
        _rows.append((_k, _in, float((_r["hi"] - _r["lo"]).mean())))
    _q_in = ((qdf["y"] >= qdf["lo"]) & (qdf["y"] <= qdf["hi"])).mean() * 100
    _fig = go.Figure()
    _fig.add_scatter(x=[r[0] for r in _rows], y=[r[1] for r in _rows], mode="lines+markers", name="conformal coverage",
                     line=dict(color="#ff7f0e", width=2))
    _fig.add_hline(y=80, line=dict(color="black", dash="dash", width=1), annotation_text="target 80%")
    _fig.add_hline(y=_q_in, line=dict(color="#1f77b4", dash="dot", width=1.5), annotation_text="quantile models")
    _fig.update_layout(
        title="Conformal coverage vs number of calibration windows (k)", xaxis_title="k (calibration windows)",
        yaxis=dict(title="coverage (%)", range=[0, 105]), height=320, margin=dict(l=50, r=20, t=50, b=40),
    )
    mo.vstack([
        mo.md("## How much calibration history does conformal need?"),
        _fig,
        mo.md(
            "Conformal trades data for guarantees: each extra calibration window is a day you cannot train on, but it "
            "makes the error estimate steadier. Coverage usually creeps towards the target as `k` grows. "
            "Quantile models (dotted blue) need no calibration windows but come with no coverage promise at all."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Which should you use?

    - **Few data, want a coverage you can defend:** conformal. One model, error bars grounded in measured errors.
    - **Want shape-aware, asymmetric intervals and many levels (P5…P95):** quantile models. Cost: one model per level,
      and quantile models can cross or be miscalibrated, so always check coverage as in 02.
    - **Best of both (worth trying):** fit quantile models, then use conformal to correct their coverage
      (conformalised quantile regression). mlforecast does not do this out of the box, but the ingredients are here.

    **Things to try**
    1. Set noise to 10. Which method's coverage degrades more?
    2. Drop **k** to 2 (mlforecast needs at least 2). Why does the conformal width line get so jagged?
    3. In `run_conformal`, try `PredictionIntervals(..., method="conformal_error")`. How does it change the width?
    4. Ask for several levels at once: `level=[50, 80, 95]`. Conformal gives all of them from the same residuals.
    """)
    return


if __name__ == "__main__":
    app.run()
