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
def _(c, mo):
    noise = mo.ui.slider(1, 10, value=4, label="Demo-data noise (kW)")
    model_pick = mo.ui.dropdown(c.MODELS, value="Quantile GBM", label="Model to diagnose")
    mo.vstack([
        mo.md(
            """
            # 4 · Residuals: where and how does the model go wrong?

            Residual = actual − median forecast, from the day-ahead backtest (so these are honest out-of-sample errors).
            Good residuals look like **boring noise**: centred on zero, no pattern by time of day, no memory.
            Patterns = something the model could still learn.
            """
        ),
        mo.hstack([noise, model_pick], justify="start", gap=2),
    ])
    return model_pick, noise


@app.cell
def _(c, model_pick, noise, np, pd):
    s, source = c.get_series(noise.value)
    bt = c.backtest(s)
    g = bt[bt["model"] == model_pick.value].copy()
    g["resid"] = g["y"] - g["q0.50"]
    _r = g["resid"].to_numpy()
    _ac1 = float(np.corrcoef(_r[:-1], _r[1:])[0, 1])
    _het = float(np.corrcoef(np.abs(_r), np.abs(g["q0.50"].to_numpy()))[0, 1])
    stats = {"bias": float(_r.mean()), "sd": float(_r.std()), "mae": float(np.abs(_r).mean()), "ac1": _ac1, "het": _het}
    return g, source, stats


@app.cell
def _(mo, source, stats):
    _flags = []
    if abs(stats["bias"]) > 0.15 * stats["sd"]:
        _flags.append(f"⚠️ **Bias**: forecasts run {'low' if stats['bias'] > 0 else 'high'} on average ({stats['bias']:+.1f} kW).")
    if stats["ac1"] > 0.5:
        _flags.append(f"⚠️ **Autocorrelated errors** (lag-1 = {stats['ac1']:.2f}): a miss now predicts a miss next half-hour. "
                      "Real fix = use recent actuals (intraday updating) or weather/other drivers.")
    if stats["het"] > 0.3:
        _flags.append("⚠️ **Heteroscedastic**: errors grow with the level of load. Quantile models handle this; plain ±σ bands wouldn't.")
    mo.vstack([
        mo.md(f"**Data:** {source}"),
        mo.hstack([
            mo.stat(value=f"{stats['bias']:+.2f}", label="bias (kW)", caption="mean residual, ideally ~0"),
            mo.stat(value=f"{stats['mae']:.2f}", label="MAE (kW)"),
            mo.stat(value=f"{stats['sd']:.2f}", label="residual σ (kW)"),
            mo.stat(value=f"{stats['ac1']:.2f}", label="lag-1 autocorr", caption="ideally ~0"),
        ], justify="start"),
        mo.md("\n\n".join(_flags) if _flags else "✅ Nothing alarming in the headline numbers."),
    ])
    return


@app.cell
def _(c, g, go, make_subplots, mo, np):
    _lay = dict(margin=dict(l=50, r=20, t=50, b=40), template="plotly")

    # --- tab 1: over time + by slot
    _f1 = make_subplots(rows=2, cols=1, subplot_titles=(
        "Residuals over the test period (kW)", "Residual spread by half-hour of day: is there a time-of-day pattern?"))
    _f1.add_scatter(x=g.index, y=g["resid"], mode="lines", line=dict(color="#1f77b4", width=1), name="residual", row=1, col=1)
    _f1.add_hline(y=0, line=dict(color="black", width=1), row=1, col=1)
    _f1.add_box(x=[c.slot_label(h) for h in g["hod"]], y=g["resid"], boxpoints=False, marker_color="#1f77b4",
                name="residual", row=2, col=1)
    _f1.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
    _f1.update_xaxes(tickformat="%d %b<br>%H:%M", row=1, col=1)
    _f1.update_xaxes(categoryorder="array", categoryarray=[c.slot_label(h) for h in range(48)],
                     tickvals=[c.slot_label(h) for h in range(0, 48, 4)], row=2, col=1)
    _f1.update_yaxes(title_text="kW", row=1, col=1)
    _f1.update_layout(height=520, showlegend=False, **_lay)

    # --- tab 2: distribution + residual vs forecast
    _mu, _sd = g["resid"].mean(), g["resid"].std()
    _x = np.linspace(g["resid"].min(), g["resid"].max(), 200)
    _f2 = make_subplots(rows=1, cols=2, subplot_titles=("Residual distribution", "Residual vs forecast: fan shape = error grows with level"))
    _f2.add_histogram(x=g["resid"], nbinsx=30, histnorm="probability density", marker_color="#1f77b4", opacity=0.8,
                      name="residuals", row=1, col=1)
    _f2.add_scatter(x=_x, y=np.exp(-0.5 * ((_x - _mu) / _sd) ** 2) / (_sd * np.sqrt(2 * np.pi)), mode="lines",
                    line=dict(color="red", dash="dash"), name="normal fit", row=1, col=1)
    _f2.add_scatter(x=g["q0.50"], y=g["resid"], mode="markers", marker=dict(size=5, opacity=0.5, color="#1f77b4"),
                    name="residual", showlegend=False, row=1, col=2)
    _f2.add_hline(y=0, line=dict(color="black", width=1), row=1, col=2)
    _f2.update_xaxes(title_text="residual (kW)", row=1, col=1)
    _f2.update_xaxes(title_text="forecast (kW)", row=1, col=2)
    _f2.update_yaxes(title_text="residual (kW)", row=1, col=2)
    _f2.update_layout(height=380, legend=dict(orientation="h", y=-0.2), **_lay)

    # --- tab 3: memory (ACF)
    _r = g["resid"].to_numpy() - g["resid"].mean()
    _lags = np.arange(1, 97)
    _acf = np.array([np.sum(_r[:-k] * _r[k:]) / np.sum(_r * _r) for k in _lags])
    _band = 1.96 / np.sqrt(len(_r))
    _f3 = go.Figure()
    _f3.add_bar(x=_lags, y=_acf, marker_color="#ff7f0e", name="autocorrelation",
                hovertemplate="lag %{x} half-hours<br>%{y:.2f}<extra></extra>")
    _f3.add_hline(y=_band, line=dict(color="black", dash="dot", width=1))
    _f3.add_hline(y=-_band, line=dict(color="black", dash="dot", width=1))
    _f3.add_vline(x=48, line=dict(color="red", dash="dash", width=1), annotation_text="1 day")
    _f3.update_layout(title="Autocorrelation of residuals: bars outside the dotted band = leftover pattern",
                      xaxis_title="lag (half-hours)", height=340, showlegend=False, **_lay)

    # --- tab 4: PIT
    _pit = c.pit_values(g)
    _f4 = go.Figure()
    _f4.add_histogram(x=_pit, xbins=dict(start=0, end=1, size=0.1), marker_color="#2ca02c", opacity=0.85, name="PIT")
    _f4.add_hline(y=len(_pit) / 10, line=dict(color="black", dash="dash", width=1), annotation_text="flat = calibrated")
    _f4.update_layout(title="PIT histogram (calibration)", xaxis_title="where the actual fell in the predicted distribution",
                      yaxis_title="count", height=340, showlegend=False, **_lay)

    mo.ui.tabs({
        "Time & slot": mo.vstack([_f1, mo.md("Look for systematic misses at particular times (e.g. always under-forecasting the evening ramp).")]),
        "Distribution": mo.vstack([_f2, mo.md("Fat tails or skew mean occasional large misses. A fan shape in the right plot means larger errors at higher loads.")]),
        "Memory (ACF)": mo.vstack([_f3, mo.md("Errors persisting for hours or repeating at lag 48 mean information in recent/daily history is still unused.")]),
        "Calibration (PIT)": mo.vstack([_f4, mo.md("Flat = honest probabilities. **U-shape** = overconfident (intervals too narrow). **Hump** = too wide. **Slope** = biased.")]),
    })
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Where these residuals come from

        Each residual is `y - q0.50` from `MLForecast.cross_validation(..., refit=True)`: the model for each day
        was fitted only on earlier days, so these are out-of-sample errors.

        **Ideas to try, then re-read the diagnostics above:**

        - Add `lags=[48, 96]` or `RollingMean(window_size=336)` in `common.make_forecaster` (needs more days
          of synthetic data). Does the lag-1 autocorrelation of the residuals drop?
        - Add a lag below 48 (e.g. `lags=[1]`). The backtest looks wonderful, but a real day-ahead forecast
          cannot know last half-hour at midnight. mlforecast would have to predict recursively, feeding its own
          forecasts back in. That is the leakage trap this folder avoids.
        - Raise the demo-data noise slider and watch which diagnostics move first.
        """
    )
    return


if __name__ == "__main__":
    app.run()
