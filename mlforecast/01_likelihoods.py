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
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, mo, np, pd


@app.cell
def _(c, mo):
    noise = mo.ui.slider(1, 10, value=4, label="Demo-data noise (kW)")
    model_pick = mo.ui.dropdown(c.ML_MODELS, value="Quantile GBM", label="ML model")
    mo.vstack([
        mo.md(
            """
            # 1 · Likelihoods: "how likely is X tomorrow?"

            A forecast is more useful as a **distribution** than a single line.
            Here an mlforecast pipeline (lag features, then one quantile model per level) predicts P5…P95
            for each half-hour of tomorrow, and we turn that into probabilities for questions you care about.

            Pick the model below. All three plug into the *same* `MLForecast` object and see the *same* features;
            only the estimator changes:

            | Model | What it is |
            |---|---|
            | Quantile GBM | LightGBM with `objective="quantile"` |
            | HistGB quantile | scikit-learn `HistGradientBoostingRegressor(loss="quantile")` |
            | Linear quantile | scikit-learn `QuantileRegressor`: a straight-line model, so it cannot bend around the evening peak |

            *Units: kW (average power over the half-hour).*
            """
        ),
        mo.hstack([noise, model_pick], justify="start", gap=2),
    ])
    return model_pick, noise


@app.cell
def _(c, mo, model_pick, noise):
    s, source = c.get_series(noise.value)
    fc = c.forecast_tomorrow(s, model_pick.value)  # MLForecast.fit(...) then .predict(h=48)
    mo.md(
        f"**Data:** {source} · {len(s) // 48} days · **model:** {model_pick.value} · "
        f"forecasting **{fc.index[0]:%a %d %b %Y}**"
    )
    return fc, s


@app.cell
def _(fc, go, mo, model_pick, s):
    _hist = s.iloc[-96:]  # last 2 days of actuals for context
    _fig = go.Figure()
    _fig.add_scatter(x=_hist.index, y=_hist.values, name="actual (last 2 days)", line=dict(color="gray", width=1.5))
    _fig.add_scatter(x=fc.index, y=fc["q0.95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=fc.index, y=fc["q0.05"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                     line=dict(width=0), name="P5–P95", hoverinfo="skip")
    _fig.add_scatter(x=fc.index, y=fc["q0.75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=fc.index, y=fc["q0.25"], fill="tonexty", fillcolor="rgba(31,119,180,0.30)",
                     line=dict(width=0), name="P25–P75", hoverinfo="skip")
    _fig.add_scatter(x=fc.index, y=fc["q0.50"], name="P50 (median)", line=dict(color="#1f77b4", width=2.5))
    _fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
    _fig.update_layout(
        title=f"Tomorrow's forecast fan · {model_pick.value}", yaxis_title="kW", hovermode="x unified",
        height=380, margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([
        mo.md("## The forecast fan"),
        _fig,
        mo.md("Wider fan = the model is less sure. Notice how it widens when the sun is up (cloud cover is unknown to this model). Hover for exact values."),
    ])
    return


@app.cell
def _(c, go, mo, s):
    _fig = go.Figure()
    for _m in c.ML_MODELS:
        _f = c.forecast_tomorrow(s, _m)
        _fig.add_scatter(x=_f.index, y=_f["q0.50"], name=_m, line=dict(width=2))
    _fig.update_layout(
        title="Median (P50) forecast, all three ML models", yaxis_title="kW", hovermode="x unified",
        height=320, margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.25),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([
        mo.md("## Same data, different model"),
        _fig,
        mo.md(
            "The two boosted-tree models track each other closely. The linear model usually misses the shape: "
            "it can only add up straight-line effects of `slot`, `lag48` and the rolling stats, so a "
            "non-linear daily curve is out of reach. Swapping the estimator is a one-line change in mlforecast."
        ),
    ])
    return


@app.cell
def _(mo, s):
    direction = mo.ui.dropdown(["above", "below"], value="above", label="Likelihood that demand is")
    threshold = mo.ui.slider(
        int(s.min()), int(s.max()), value=int(s.quantile(0.75)), label="threshold (kW)", show_value=True, full_width=True
    )
    mo.vstack([
        mo.md("## Ask a question"),
        direction,
        threshold,
    ])
    return direction, threshold


@app.cell
def _(c, direction, fc, go, mo, np, threshold):
    _x, _d = threshold.value, direction.value
    _vals = fc[c.qcols()].to_numpy()
    _p_all = np.array([c.prob_event(c.QUANTILES, r, _x, _d)[0] for r in _vals])
    _labels = [c.prob_event(c.QUANTILES, r, _x, _d)[1] for r in _vals]
    _peak = int(_p_all.argmax())

    _fig = go.Figure()
    _fig.add_bar(
        x=fc.index, y=_p_all * 100, marker_color="#ff7f0e", opacity=0.85, customdata=_labels,
        hovertemplate="%{x|%a %d %b, %H:%M}<br>P = %{customdata}<extra></extra>",
    )
    _fig.add_hline(y=50, line=dict(color="black", width=1, dash="dash"), annotation_text="50%")
    _fig.update_layout(
        title=f"P(demand {_d} {_x} kW) for every half-hour tomorrow",
        yaxis=dict(title="probability (%)", range=[0, 100]),
        height=340, margin=dict(l=50, r=20, t=50, b=40), bargap=0.1,
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M", dtick=4 * 3600 * 1000)

    mo.vstack([
        mo.hstack([
            mo.stat(value=_labels[_peak], label="peak probability", caption=f"at {fc.index[_peak]:%a %d %b %H:%M}"),
            mo.stat(value=f"{_p_all.sum():.1f}", label="expected half-hours affected", caption="of 48 tomorrow"),
            mo.stat(value=f"{(_p_all > 0.5).sum()}", label="half-hours more likely than not", caption="P > 50%"),
        ], justify="start"),
        _fig,
        mo.md("*Values like `<5%` / `>95%` mean the threshold is outside the predicted P5–P95 range, so the tail can't be resolved from these quantiles. Hover any bar for its probability.*"),
    ])
    return


@app.cell
def _(c, fc, go, mo, np):
    _vals = fc[c.qcols()].to_numpy()
    _p_exp = np.array([c.prob_event(c.QUANTILES, r, 0.0, "below")[0] for r in _vals])
    _fig = go.Figure()
    _fig.add_scatter(
        x=fc.index, y=_p_exp * 100, fill="tozeroy", line=dict(color="#2ca02c"), name="P(export)",
        hovertemplate="%{x|%a %d %b, %H:%M}<br>%{y:.0f}%<extra></extra>",
    )
    _fig.add_hline(y=50, line=dict(color="black", width=1, dash="dash"))
    _fig.update_layout(
        title="P(net load < 0 kW): likelihood the asset is exporting",
        yaxis=dict(title="%", range=[0, 100]), height=300, margin=dict(l=50, r=20, t=50, b=40),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M", dtick=4 * 3600 * 1000)
    _likely = [f"{t:%H:%M}" for t, p in zip(fc.index, _p_exp) if p > 0.5]
    mo.vstack([
        mo.md("## Sign flips: will the asset export?"),
        _fig,
        mo.md(f"Slots where export is more likely than not: **{_likely[0]}–{_likely[-1]}**" if _likely else "No slot where export is more likely than not."),
    ])
    return


@app.cell
def _(c, go, mo, s):
    _tbl = c.feature_table(s)
    _show = _tbl.drop(columns="unique_id").head(6).round(2)
    _last = _tbl.iloc[-96:]
    _fig = go.Figure()
    _fig.add_scatter(x=_last["ds"], y=_last["rolling_max_lag48_window_size48"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_last["ds"], y=_last["rolling_min_lag48_window_size48"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                     line=dict(width=0), name="yesterday's 24h min–max", hoverinfo="skip")
    _fig.add_scatter(x=_last["ds"], y=_last["y"], name="y (target)", line=dict(color="black", width=1.8))
    _fig.add_scatter(x=_last["ds"], y=_last["lag48"], name="lag48 = y shifted by 48 slots", line=dict(color="#d62728", dash="dash", width=1.5))
    _fig.update_layout(
        title="Features are just the target, shifted and summarised", yaxis_title="kW", hovermode="x unified",
        height=340, margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.25),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
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
