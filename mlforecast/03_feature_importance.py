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
    ])
    return


@app.cell
def _(c, mo):
    s, source = c.get_series()
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
    mo.vstack([
        mo.md(
            """
            ## How does the forecast respond to one feature? (partial dependence)

            Permutation importance above says **how much** the model leans on a feature. Partial dependence says **how**:
            which direction, and what shape.

            **How it is computed.** Take the training rows. Set one feature to a value (say `slot = 36`, i.e. 18:00)
            for *every* row, leave all other features as they were, and average the model's predictions.
            Repeat over a grid of values and plot the averages. The line answers:
            *"if only this feature changed, what would the forecast do on average?"*
            """
        ),
        feature_pick,
    ])
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
        mo.md(f"**`{feature_pick.value}`**: {c.FEATURE_NOTES.get(feature_pick.value, '')}."),
        mo.accordion({
            "What to look for, feature by feature": mo.md(
                """
                The synthetic data has a known truth, so you can check the model against it:
                demand peaks around 18:00 (and a smaller bump near 08:00), solar pulls net load down around midday,
                and weekends run about 15% lower.

                | Feature | What a sensible line looks like |
                |---|---|
                | `slot` | The daily shape: morning bump, midday dip (solar), evening peak. If you see this, the model has learned the main pattern. |
                | `weekend` | Two points only (0 and 1). The weekend point should be lower. |
                | `lag48` | Rising: a high same-slot-yesterday value predicts a high value today. A slope near 1 is "tomorrow looks like today". |
                | `rolling_mean/max/min_...48` | Rising, often gently: yesterday's overall level nudges today's forecast. |
                | any feature | **Flat** = the model ignores it. **Staircase** = normal for tree models, which predict in steps. |
                """
            ),
            "What you can learn from this notebook": mo.md(
                """
                1. **Did the model learn the right thing?** Compare the `slot` line with the known daily shape.
                   A model can score well for the wrong reason, and this is how you spot it.
                2. **Which inputs earn their place?** Features with ~0 importance and a flat line can be dropped from
                   `lags` / `lag_transforms` in `common.FEATURE_KWARGS`, which makes the pipeline simpler.
                3. **Where is the model brittle?** Steep or jagged regions show where a small change in an input swings the
                   forecast, a sign of overfitting on ~10 days of data.
                4. **What is missing?** If errors in notebook 04 are big but nothing here explains them, the missing driver
                   is something you have not given the model (here: cloud cover or weather).
                5. **How mlforecast helps:** the fitted model is `fcst.models_[...]` and the feature table is
                   `fcst.preprocess(...)`, so any sklearn inspection tool works on it directly.
                """
            ),
            "Cautions": mo.md(
                """
                - **Correlated features.** The rolling mean, max, min and `lag48` all describe yesterday. Sweeping one while the
                  others stay put creates combinations that never occur in real data (e.g. a rolling max below `lag48`),
                  so those lines can mislead. Read them as a hint, not a measurement.
                - **Averaging hides interactions.** If `slot` matters differently on weekends, the average blends both.
                - **Importance is not causation.** It describes this model, not the real world.
                - **Tiny data.** About 10 training days, so small wiggles are noise.
                """
            ),
        }),
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
