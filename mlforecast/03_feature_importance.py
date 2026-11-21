# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/03_feature_importance.py
#   uv run marimo run  mlforecast/03_feature_importance.py   (app mode, code hidden)

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
    from sklearn.inspection import partial_dependence, permutation_importance

    if mo.notebook_dir() is not None:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, mo, np, partial_dependence, permutation_importance, pd


@app.cell
def _(mo):
    noise = mo.ui.slider(1, 10, value=4, label="Demo-data noise (kW)")
    mo.vstack([
        mo.md(
            """
            # 3 · Feature importance: what is the model actually using?

            **Permutation importance:** shuffle one feature on *held-out* days and see how much the error gets worse.
            Big jump = the model leans on it. ~0 = it's ignorable (or redundant with another feature).

            The features are built by mlforecast (`lags`, `lag_transforms`, `date_features`), all known at
            midnight so valid for day-ahead. The fitted model is pulled straight out of `fcst.models_`
            and the feature table from `fcst.preprocess(...)`.
            """
        ),
        mo.hstack([noise], justify="start", gap=2),
    ])
    return (noise,)


@app.cell
def _(c, mo, noise):
    s, source = c.get_series(noise.value)
    df = c.to_long(s)
    ndays = len(s) // c.PERIODS_PER_DAY
    split_days = mo.ui.slider(5, max(6, ndays - 2), value=max(5, ndays - 4), label="Days used for training", show_value=True)
    reps = mo.ui.slider(5, 50, value=20, step=5, label="Shuffle repeats", show_value=True)
    mo.vstack([mo.md(f"**Data:** {source} · {ndays} days"), mo.hstack([split_days, reps], justify="start", gap=2)])
    return df, ndays, reps, split_days


@app.cell
def _(c, df, permutation_importance, pd, reps, split_days):
    _cut = df["ds"].min() + pd.Timedelta(days=split_days.value)
    fcst = c.make_forecaster([0.5])  # median models only (one per tier)
    fcst.fit(df[df["ds"] < _cut])  # fit on the training days only
    med_model = fcst.models_["gbm_q0.50"]  # the fitted LGBMRegressor
    feats = list(fcst.ts.features_order_)  # feature names mlforecast built
    _table = c.make_forecaster([0.5]).preprocess(df)  # (unique_id, ds, y, *features) for every usable row
    tr, te = _table[_table["ds"] < _cut], _table[_table["ds"] >= _cut]
    pi = permutation_importance(
        med_model, te[feats], te["y"], scoring="neg_mean_absolute_error",
        n_repeats=reps.value, random_state=0,
    )
    return feats, med_model, pi, te, tr


@app.cell
def _(c, feats, go, mo, pi, te, tr):
    _order = pi.importances_mean.argsort()
    _names = [feats[i] for i in _order]
    _fig = go.Figure()
    _fig.add_bar(
        y=_names, x=pi.importances_mean[_order], orientation="h", marker_color="#1f77b4", opacity=0.9,
        error_x=dict(type="data", array=pi.importances_std[_order]),
        hovertemplate="%{y}<br>MAE rises by %{x:.2f} kW when shuffled<extra></extra>",
    )
    _fig.add_vline(x=0, line=dict(color="black", width=1))
    _fig.update_layout(
        title="Permutation importance (held-out days)", xaxis_title="increase in MAE (kW) when shuffled",
        height=360, margin=dict(l=240, r=20, t=50, b=40),
    )
    _top = _names[-1]
    mo.vstack([
        mo.md(f"## What matters  \nTrained on **{tr['ds'].dt.normalize().nunique()}** days, importance measured on **{te['ds'].dt.normalize().nunique()}** unseen days."),
        _fig,
        mo.callout(
            mo.md(
                f"Most important: **`{_top}`**. Error bars show shuffle-to-shuffle noise only; with so few held-out days, "
                "treat small differences as a tie. Correlated features (e.g. `lag48` vs the rolling means built on it) share credit, "
                "so each can look weaker than the signal they carry together."
            ),
            kind="info",
        ),
    ])
    return


@app.cell
def _(feats, mo):
    feature_pick = mo.ui.dropdown(feats, value="slot", label="Show partial dependence for")
    mo.vstack([mo.md("## How does the forecast respond to one feature?"), feature_pick])
    return (feature_pick,)


@app.cell
def _(c, feats, feature_pick, go, med_model, mo, partial_dependence, tr):
    _pd = partial_dependence(med_model, tr[feats].astype(float), [feature_pick.value], grid_resolution=40)
    _fig = go.Figure()
    _fig.add_scatter(x=_pd["grid_values"][0], y=_pd["average"][0], mode="lines+markers", line=dict(color="#ff7f0e", width=2.5),
                     hovertemplate=feature_pick.value + " = %{x:.2f}<br>avg predicted %{y:.1f} kW<extra></extra>")
    _fig.update_layout(
        title=f"Partial dependence: {feature_pick.value}", xaxis_title=feature_pick.value,
        yaxis_title="average predicted kW", height=330, margin=dict(l=50, r=20, t=50, b=40),
    )
    mo.vstack([
        _fig,
        mo.md(
            "Average predicted median as this one feature is swept, all others left as observed. "
            "`slot` should trace the daily shape (evening peak, midday dip if there's solar). "
            "A flat line means the model barely uses it."
        ),
    ])
    return


@app.cell
def _(c, feats, mo, pd):
    _rows = pd.DataFrame({"feature": feats, "meaning": [c.FEATURE_NOTES.get(f, "") for f in feats]})
    mo.vstack([
        mo.md(
            """
            ## Reference: the features mlforecast built

            Names follow mlforecast's convention `<transform>_lag<k>_window_size<n>`. Read it as:
            "shift the target by `k`, then apply the transform over a window of `n`".
            Because every `k` is 48, nothing here can see today's actuals.
            """
        ),
        mo.ui.table(_rows, selection=None),
        mo.md(
            "**Try it:** in `common.make_forecaster`, add `lags=[48, 96]` or another `RollingMean`, re-run, and "
            "watch how importance is shared out between correlated features."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
