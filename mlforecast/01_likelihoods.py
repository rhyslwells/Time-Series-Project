# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/01_likelihoods.py
#   uv run marimo run  mlforecast/01_likelihoods.py   (app mode, code hidden)

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
            # 1 · Likelihoods: "how likely is X tomorrow?"

            A forecast is more useful as a **distribution** than a single line.
            Here an mlforecast pipeline (lag features + one quantile LightGBM per level) predicts P5…P95 for each half-hour of tomorrow, and we turn
            that into probabilities for questions you actually care about.

            *Units: kW (average power over the half-hour). If your meter gives kWh per half-hour, multiply by 2.*
            """
        ),
        mo.hstack([noise], justify="start", gap=2),
    ])
    return (noise,)


@app.cell
def _(c, mo, noise):
    s, source = c.get_series(noise.value)
    fc = c.forecast_tomorrow(s)  # MLForecast.fit(...) then .predict(h=48); quantile LightGBM models
    mo.md(f"**Data:** {source} · {len(s)} half-hours ({len(s) // 48} days) · forecasting **{fc.index[0]:%a %d %b}**")
    return fc, s


@app.cell
def _(fc, mo, plt, s):
    _hist = s.iloc[-96:]  # last 2 days of actuals for context
    _fig, _ax = plt.subplots(figsize=(10, 3.8))
    _ax.plot(_hist.index, _hist.values, color="0.3", lw=1.2, label="actual (last 2 days)")
    _ax.fill_between(fc.index, fc["q0.05"], fc["q0.95"], color="C0", alpha=0.15, label="P5–P95")
    _ax.fill_between(fc.index, fc["q0.25"], fc["q0.75"], color="C0", alpha=0.30, label="P25–P75")
    _ax.plot(fc.index, fc["q0.50"], color="C0", lw=2, label="P50 (median)")
    _ax.axhline(0, color="k", lw=0.6, ls=":")
    _ax.set_ylabel("kW")
    _ax.legend(ncols=4, fontsize=8, loc="upper left")
    _ax.set_title("Tomorrow's forecast fan (day-ahead)")
    _fig.tight_layout()
    mo.vstack([mo.md("## The forecast fan"), _fig, mo.md("Wider fan = the model is less sure. Notice how it widens when the sun is up (cloud cover is unknown to this model).")])
    return


@app.cell
def _(c, mo, s):
    direction = mo.ui.dropdown(["above", "below"], value="above", label="Likelihood that demand is")
    threshold = mo.ui.slider(
        int(s.min()), int(s.max()), value=int(s.quantile(0.75)), label="threshold (kW)", show_value=True, full_width=True
    )
    slot = mo.ui.dropdown({c.slot_label(h): h for h in range(48)}, value="18:00", label="at half-hour")
    mo.vstack([
        mo.md("## Ask a question"),
        mo.hstack([direction, slot], justify="start", gap=2),
        threshold,
    ])
    return direction, slot, threshold


@app.cell
def _(c, direction, fc, mo, np, plt, slot, threshold):
    _x, _d = threshold.value, direction.value
    _vals = fc[c.qcols()].to_numpy()
    _p_all = np.array([c.prob_event(c.QUANTILES, r, _x, _d)[0] for r in _vals])
    _p, _txt = c.prob_event(c.QUANTILES, _vals[slot.value], _x, _d)

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.bar(range(48), _p_all * 100, color="C1", alpha=0.8)
    _ax.axvline(slot.value, color="k", ls="--", lw=1)
    _ax.set_xticks(range(0, 48, 4), [c.slot_label(h) for h in range(0, 48, 4)])
    _ax.set_ylabel("probability (%)")
    _ax.set_ylim(0, 100)
    _ax.set_title(f"P(demand {_d} {_x} kW) for every half-hour tomorrow")
    _fig.tight_layout()

    mo.vstack([
        mo.hstack([
            mo.stat(value=_txt, label=f"P(demand {_d} {_x} kW at {c.slot_label(slot.value)})", caption="from the predicted quantiles"),
            mo.stat(value=f"{_p_all.sum():.1f}", label="expected half-hours affected", caption="of 48 tomorrow"),
            mo.stat(value=f"{(_p_all > 0.5).sum()}", label="half-hours more likely than not", caption="P > 50%"),
        ], justify="start"),
        _fig,
        mo.md("*Values like `<5%` / `>95%` mean the threshold is outside the predicted P5–P95 range, so the tail can't be resolved from these quantiles.*"),
    ])
    return


@app.cell
def _(c, fc, mo, np, plt):
    _vals = fc[c.qcols()].to_numpy()
    _p_exp = np.array([c.prob_event(c.QUANTILES, r, 0.0, "below")[0] for r in _vals])
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    _ax.fill_between(range(48), _p_exp * 100, color="C2", alpha=0.6)
    _ax.set_xticks(range(0, 48, 4), [c.slot_label(h) for h in range(0, 48, 4)])
    _ax.set_ylim(0, 100)
    _ax.set_ylabel("%")
    _ax.set_title("P(net load < 0 kW) → likelihood the asset is exporting")
    _fig.tight_layout()
    _likely = [c.slot_label(h) for h in range(48) if _p_exp[h] > 0.5]
    mo.vstack([
        mo.md("## Sign flips: will the asset export?"),
        _fig,
        mo.md(f"Slots where export is more likely than not: **{_likely[0]}–{_likely[-1]}**" if _likely else "No slot where export is more likely than not."),
    ])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **How to read this honestly**
            - Probabilities come from interpolating 7 predicted quantiles, so they're smooth approximations, not exact.
            - With ~2 weeks of data the quantiles themselves are noisy. Check **notebook 2** to see whether
              "80% intervals" actually contain ~80% of outcomes (calibration) before trusting any percentage here.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(c, mo, plt, s):
    _tbl = c.feature_table(s)
    _show = _tbl.drop(columns="unique_id").head(6).round(2)
    _last = _tbl.iloc[-96:]
    _fig, _ax = plt.subplots(figsize=(10, 3.2))
    _ax.plot(_last["ds"], _last["y"], color="k", lw=1.4, label="y (target)")
    _ax.plot(_last["ds"], _last["lag48"], color="C3", lw=1, ls="--", label="lag48 = y shifted by 48 slots")
    _ax.fill_between(_last["ds"], _last["rolling_min_lag48_window_size48"], _last["rolling_max_lag48_window_size48"],
                     color="C0", alpha=0.12, label="yesterday's 24h min-max")
    _ax.set_ylabel("kW")
    _ax.legend(fontsize=8, ncols=3, loc="upper left")
    _ax.set_title("Features are just the target, shifted and summarised")
    _fig.tight_layout()
    mo.vstack([
        mo.md(
            """
            ## Under the hood: what did mlforecast feed the model?

            You never wrote a feature loop. You told `MLForecast` three things, and `fcst.preprocess(df)` returns
            the table it trains on (first rows below):

            - `lags=[48]` -> column `lag48`: the value one day (48 half-hours) earlier.
            - `lag_transforms={48: [RollingMean(3), RollingMean(48), RollingMax(48), RollingMin(48)]}` ->
              rolling stats computed *after* shifting by 48, so they only look at yesterday and earlier.
            - `date_features=[slot, weekend]` -> calendar columns from the timestamp.

            Every lag is >= 48, so at midnight all features for tomorrow's 48 slots already exist. That is
            why one `predict(h=48)` works with no step-by-step recursion.
            """
        ),
        mo.ui.table(_show.reset_index(drop=True), selection=None),
        _fig,
        mo.md("The dashed red line is the target slid forward one day. The model's job is to learn how today differs from it."),
    ])
    return


if __name__ == "__main__":
    app.run()
