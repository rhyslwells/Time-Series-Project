# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/04_residuals.py
#   uv run marimo run  mlforecast/04_residuals.py   (app mode, code hidden)

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
    mo.md(
        """
        # 4 · Residuals: where and how do the models go wrong?

        Residual = actual − median forecast, from the day-ahead backtest (`MLForecast.cross_validation`, refit before each
        day, so these are honest out-of-sample errors). **Every chart shows all five models together** so you can compare them:
        two simple baselines and three ML models, all inside one `MLForecast` object on the same features.

        Good residuals look like **boring noise**: centred on zero, no pattern by time of day, no memory.
        Patterns = something the model could still learn.
        """
    )
    return


@app.cell
def _(c, mo, np, pd):
    s, source = c.get_series()
    all_models = c.BASELINES + c.ML_MODELS
    with mo.status.spinner("Backtesting all five models…"):
        bt = c.backtest(s, tiers=all_models)
    gs = {}
    for _m in all_models:
        _g = bt[bt["model"] == _m].copy()
        _g["resid"] = _g["y"] - _g["q0.50"]
        gs[_m] = _g

    _rows = []
    for _m, _g in gs.items():
        _r = _g["resid"].to_numpy()
        _rows.append({
            "model": _m,
            "bias (kW)": float(_r.mean()),
            "MAE (kW)": float(np.abs(_r).mean()),
            "residual σ (kW)": float(_r.std()),
            "lag-1 autocorr": float(np.corrcoef(_r[:-1], _r[1:])[0, 1]),
            "error vs level": float(np.corrcoef(np.abs(_r), np.abs(_g["q0.50"].to_numpy()))[0, 1]),
        })
    stats = pd.DataFrame(_rows).set_index("model")
    return all_models, bt, gs, source, stats


@app.cell
def _(mo, source, stats):
    _show = stats.round(2).reset_index()
    _flags = []
    for _m, _r in stats.iterrows():
        _f = []
        if abs(_r["bias (kW)"]) > 0.15 * _r["residual σ (kW)"]:
            _f.append(f"runs {'low' if _r['bias (kW)'] > 0 else 'high'} on average ({_r['bias (kW)']:+.1f} kW)")
        if _r["lag-1 autocorr"] > 0.5:
            _f.append(f"errors persist (lag-1 = {_r['lag-1 autocorr']:.2f})")
        if _r["error vs level"] > 0.3:
            _f.append("errors grow with the load level")
        _flags.append(f"- **{_m}**: " + ("; ".join(_f) if _f else "nothing alarming in the headline numbers"))
    mo.vstack([
        mo.md(f"**Data:** {source}"),
        mo.ui.table(_show, selection=None),
        mo.md("**Flags per model**\n\n" + "\n".join(_flags)),
        mo.md(
            "*Bias* = mean residual (ideally ~0). *Lag-1 autocorr* = how much a miss now predicts a miss next half-hour "
            "(ideally ~0; high means the model is missing a driver or recent information). *Error vs level* = whether bigger "
            "loads get bigger errors (heteroscedastic)."
        ),
    ])
    return


@app.cell
def _(all_models, c, go, gs, make_subplots, mo, np):
    _col = c.TIER_COLORS
    _lay = dict(margin=dict(l=55, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.12))
    _slots = [c.slot_label(h) for h in range(48)]
    _tick = _slots[::4]

    # --- tab 1: over time, spread by slot, bias by slot
    _f1 = make_subplots(rows=3, cols=1, vertical_spacing=0.09, subplot_titles=(
        "Residuals over the test period (kW)",
        "Spread of residuals by half-hour of day (standard deviation): is there a time-of-day pattern?",
        "Average residual by half-hour of day: does the model lean one way at certain times?"))
    for _m in all_models:
        _g = gs[_m]
        _f1.add_scatter(x=_g.index, y=_g["resid"], mode="lines", line=dict(color=_col[_m], width=1.1), name=_m,
                        legendgroup=_m, row=1, col=1)
        _by = _g.groupby("hod")["resid"]
        _f1.add_scatter(x=_slots, y=_by.std().reindex(range(48)).to_numpy(), mode="lines", line=dict(color=_col[_m], width=2),
                        name=_m, legendgroup=_m, showlegend=False, row=2, col=1)
        _f1.add_scatter(x=_slots, y=_by.mean().reindex(range(48)).to_numpy(), mode="lines", line=dict(color=_col[_m], width=2),
                        name=_m, legendgroup=_m, showlegend=False, row=3, col=1)
    _f1.add_hline(y=0, line=dict(color="gray", width=1), row=1, col=1)
    _f1.add_hline(y=0, line=dict(color="gray", width=1), row=3, col=1)
    _f1.update_xaxes(tickformat="%d %b<br>%H:%M", row=1, col=1)
    for _r in (2, 3):
        _f1.update_xaxes(tickvals=_tick, row=_r, col=1)
    _f1.update_yaxes(title_text="kW", row=1, col=1)
    _f1.update_yaxes(title_text="σ (kW)", row=2, col=1)
    _f1.update_yaxes(title_text="mean (kW)", row=3, col=1)
    _f1.update_layout(height=820, hovermode="x unified", **_lay)

    # --- tab 2: distribution + residual vs forecast
    _f2 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Residual distribution (all models overlaid)", "Residual vs forecast: fan shape = error grows with level"))
    for _m in all_models:
        _g = gs[_m]
        _f2.add_histogram(x=_g["resid"], nbinsx=30, histnorm="probability density", opacity=0.45, marker_color=_col[_m],
                          name=_m, legendgroup=_m, row=1, col=1)
        _f2.add_scatter(x=_g["q0.50"], y=_g["resid"], mode="markers", marker=dict(size=4, opacity=0.45, color=_col[_m]),
                        name=_m, legendgroup=_m, showlegend=False, row=1, col=2)
    _f2.add_hline(y=0, line=dict(color="gray", width=1), row=1, col=2)
    _f2.update_xaxes(title_text="residual (kW)", row=1, col=1)
    _f2.update_xaxes(title_text="forecast (kW)", row=1, col=2)
    _f2.update_yaxes(title_text="residual (kW)", row=1, col=2)
    _f2.update_layout(height=420, barmode="overlay", **_lay)

    # --- tab 3: memory (ACF)
    _lags = np.arange(1, 97)
    _f3 = go.Figure()
    _n = 0
    for _m in all_models:
        _r = gs[_m]["resid"].to_numpy() - gs[_m]["resid"].mean()
        _acf = np.array([np.sum(_r[:-k] * _r[k:]) / np.sum(_r * _r) for k in _lags])
        _n = len(_r)
        _f3.add_scatter(x=_lags, y=_acf, mode="lines+markers", marker=dict(size=4), line=dict(color=_col[_m], width=1.6), name=_m,
                        hovertemplate=_m + "<br>lag %{x} half-hours: %{y:.2f}<extra></extra>")
    _band = 1.96 / np.sqrt(_n)
    _f3.add_hline(y=_band, line=dict(color="gray", dash="dot", width=1))
    _f3.add_hline(y=-_band, line=dict(color="gray", dash="dot", width=1))
    _f3.add_vline(x=48, line=dict(color="red", dash="dash", width=1), annotation_text="1 day")
    _f3.update_layout(title="Autocorrelation of residuals: points outside the dotted band = leftover pattern",
                      xaxis_title="lag (half-hours)", yaxis_title="autocorrelation", height=380, **_lay)

    # --- tab 4: PIT
    _edges = np.linspace(0, 1, 11)
    _mid = [f"{a:.1f}–{b:.1f}" for a, b in zip(_edges[:-1], _edges[1:])]
    _f4 = go.Figure()
    _exp = 0
    for _m in all_models:
        _pit = c.pit_values(gs[_m])
        _cnt, _ = np.histogram(_pit, bins=_edges)
        _exp = len(_pit) / 10
        _f4.add_bar(x=_mid, y=_cnt, name=_m, marker_color=_col[_m], opacity=0.9)
    _f4.add_hline(y=_exp, line=dict(color="gray", dash="dash", width=1.5), annotation_text="flat = calibrated")
    _f4.update_layout(title="PIT histogram (calibration)", xaxis_title="where the actual fell in the predicted distribution",
                      yaxis_title="count", barmode="group", height=380, **_lay)

    mo.ui.tabs({
        "Time & slot": mo.vstack([
            mo.md(
                """
                **Residual = actual − median forecast.** Above zero: the model under-forecast. Below zero: it over-forecast.
                The three charts ask different questions:

                - **Top (over time):** are the errors random flicker, or slow waves? Waves that last hours mean the model
                  keeps missing in the *same direction* for a while. If all five lines move together, the miss is in
                  the **data** (something none of them can see), not in one model's choices.
                - **Middle (spread by half-hour):** how big the errors are at each time of day. A hump at midday means that
                  time of day is hard to forecast. Compare the lines: which model keeps the hump smallest?
                - **Bottom (average error by half-hour):** a line far from zero = a *systematic* miss at that time (bias).
                """
            ),
            _f1,
            mo.accordion({
                "What to take from these charts": mo.md(
                    """
                    1. **Errors are not white noise.** Slow waves instead of static mean autocorrelated errors: a miss at 11:00
                       predicts a miss at 11:30. Confirm in the *Memory (ACF)* tab. The cause is usually a driver the models cannot
                       see. Here, one cloudiness factor per day scales the solar output.
                    2. **Errors concentrate at certain times.** A higher standard deviation at midday than overnight means
                       uncertainty is not constant through the day. Overnight, demand is regular. Around midday solar dominates,
                       and a cloudy versus a clear day swings the answer by tens of kW.
                    3. **Models that move together share a blind spot.** Large excursions that appear in every line at the same
                       moment line up with day changes: the models lean on yesterday's same slot (`lag48`), so when
                       today's weather differs from yesterday's all of them miss.
                    4. **Differences between lines are the model comparison.** If one line is clearly lower in the middle chart,
                       that model handles the hard period better. The Linear quantile model usually shows structured bias in the
                       bottom chart because a straight line cannot follow the daily curve.
                    5. **What to do about it.** More lags of the same series will not fix a hidden driver. Add one (a weather or
                       irradiance forecast, via mlforecast's `X_df`), or accept the uncertainty and make sure the prediction
                       intervals widen at midday (see notebooks 02 and 06).
                    """
                ),
                "Cautions": mo.md(
                    """
                    - **Few days, few points.** Each half-hour slot has about 6 values (one per test day), so the per-slot lines
                      are noisy. Trust the broad pattern (midday much wider than night), not individual wiggles.
                    - **Overlapping lines.** If a chart is crowded, click a model's name in the legend to hide it, or double-click
                      one to isolate it.
                    """
                ),
            }),
        ]),
        "Distribution": mo.vstack([
            _f2,
            mo.md(
                "Left: the shape of the errors. Fat tails or skew mean occasional large misses, and a model whose histogram is "
                "narrower is more accurate. Right: if points fan out as the forecast rises, errors grow with the load level."
            ),
        ]),
        "Memory (ACF)": mo.vstack([
            _f3,
            mo.md(
                "Autocorrelation at lag *k* asks: how related is a miss now to the miss *k* half-hours ago? Values near 0 are good. "
                "Errors that persist for hours, or repeat at lag 48 (the red line, one day), mean information in recent or "
                "daily history is still unused. Models that share the same pattern share the same blind spot."
            ),
        ]),
        "Calibration (PIT)": mo.vstack([
            _f4,
            mo.md(
                "For each actual, where did it fall inside the model's predicted distribution (0 = below everything, 1 = above "
                "everything)? Flat bars = honest probabilities. **U-shape** = overconfident (intervals too narrow). "
                "**Hump** = too wide. **Slope** = biased."
            ),
        ]),
    })
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Where these residuals come from

        Each residual is `y - q0.50` from `MLForecast.cross_validation(..., refit=True)`: the models for each day
        were fitted only on earlier days, so these are out-of-sample errors. All five models sit in one `MLForecast` object,
        so they saw the identical features and the identical backtest windows.

        **Ideas to try, then re-read the diagnostics above:**

        - Add `lags=[48, 96]` or `RollingMean(window_size=336)` in `common.make_forecaster` (needs more days
          of synthetic data). Does the lag-1 autocorrelation of the residuals drop?
        - Add a lag below 48 (e.g. `lags=[1]`). The backtest looks wonderful, but a real day-ahead forecast
          cannot know last half-hour at midnight. mlforecast would have to predict recursively, feeding its own
          forecasts back in. That is the leakage trap this folder avoids.
        - Compare the models: which diagnostic separates the Linear quantile model from the two tree models?
        """
    )
    return


if __name__ == "__main__":
    app.run()
