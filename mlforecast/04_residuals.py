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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, mo, np, pd, plt


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
def _(c, g, mo, np, plt):
    # --- tab 1: over time + by slot
    _f1, _ax = plt.subplots(2, 1, figsize=(10, 5))
    _ax[0].plot(g.index, g["resid"], lw=0.8, color="C0")
    _ax[0].axhline(0, color="k", lw=0.6)
    _ax[0].set_title("Residuals over the test period (kW)")
    _data = [g.loc[g["hod"] == h, "resid"].to_numpy() for h in range(48)]
    _ax[1].boxplot(_data, showfliers=False, widths=0.6)
    _ax[1].axhline(0, color="k", lw=0.6)
    _ax[1].set_xticks(range(1, 49, 4), [c.slot_label(h) for h in range(0, 48, 4)])
    _ax[1].set_title("Residual spread by half-hour of day · is there a time-of-day pattern?")
    _f1.tight_layout()

    # --- tab 2: distribution + residual vs forecast
    _f2, _ax2 = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax2[0].hist(g["resid"], bins=30, color="C0", alpha=0.8, density=True)
    _x = np.linspace(g["resid"].min(), g["resid"].max(), 200)
    _mu, _sd = g["resid"].mean(), g["resid"].std()
    _ax2[0].plot(_x, np.exp(-0.5 * ((_x - _mu) / _sd) ** 2) / (_sd * np.sqrt(2 * np.pi)), "r--", label="normal fit")
    _ax2[0].set_title("Residual distribution")
    _ax2[0].legend(fontsize=8)
    _ax2[1].scatter(g["q0.50"], g["resid"], s=8, alpha=0.5)
    _ax2[1].axhline(0, color="k", lw=0.6)
    _ax2[1].set_xlabel("forecast (kW)")
    _ax2[1].set_ylabel("residual (kW)")
    _ax2[1].set_title("Residual vs forecast · fan shape = error grows with level")
    _f2.tight_layout()

    # --- tab 3: memory (ACF)
    _r = g["resid"].to_numpy() - g["resid"].mean()
    _lags = np.arange(1, 97)
    _acf = np.array([np.sum(_r[:-k] * _r[k:]) / np.sum(_r * _r) for k in _lags])
    _f3, _ax3 = plt.subplots(figsize=(10, 3))
    _ax3.bar(_lags, _acf, color="C1")
    _band = 1.96 / np.sqrt(len(_r))
    _ax3.axhline(_band, color="k", ls=":", lw=1)
    _ax3.axhline(-_band, color="k", ls=":", lw=1)
    _ax3.axvline(48, color="C3", ls="--", lw=0.8)
    _ax3.set_xlabel("lag (half-hours) · red line = 1 day")
    _ax3.set_title("Autocorrelation of residuals · bars outside dotted band = leftover pattern")
    _f3.tight_layout()

    # --- tab 4: PIT
    _pit = c.pit_values(g)
    _f4, _ax4 = plt.subplots(figsize=(6, 3.4))
    _ax4.hist(_pit, bins=10, range=(0, 1), color="C2", alpha=0.85)
    _ax4.axhline(len(_pit) / 10, color="k", ls="--", lw=1)
    _ax4.set_title("PIT histogram (calibration)")
    _ax4.set_xlabel("where the actual fell in the predicted distribution")
    _f4.tight_layout()

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
