# Run from the project root (Time Series Project):
#   uv run marimo edit mlforecast/05_mlforecast_basics.py
#   uv run marimo run  mlforecast/05_mlforecast_basics.py   (app mode, code hidden)

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

    if str(mo.notebook_dir()) not in sys.path:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # mlforecast basics, day-ahead half-hourly

    mlforecast wraps *feature engineering from lags* + *any sklearn-style model* + *backtesting*.
    Mental model: you give it a long dataframe `(unique_id, ds, y)`; it builds lag/rolling/date
    features, fits one model on the stacked series, and predicts `h` steps ahead.

    **Day-ahead rule.** We forecast 48 slots at 00:00, so every lag must be >= 48. Then a single
    non-recursive pass is valid and nothing leaks.
    """)
    return


@app.cell
def _(c, pd):
    # 1. Data in mlforecast's long format. Two fake assets so you see multi-series behaviour.
    frames = []
    for i, seed in enumerate([0, 1]):
        s = c.make_synthetic(days=28, seed=seed)
        frames.append(
            pd.DataFrame({"unique_id": f"asset_{i}", "ds": s.index, "y": s.to_numpy()})
        )
    df = pd.concat(frames, ignore_index=True)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Build the forecaster

    - `lags=[48, 96, 336]`: same slot yesterday, 2 days ago, last week.
    - `lag_transforms`: rolling stats are computed *on top of* a lag, so `lag=48` means
      "shifted by a day first". That is what keeps them day-ahead-safe.
    - `date_features`: callables on the DatetimeIndex.
    """)
    return


@app.cell
def _():
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import ExpandingMean, RollingMean, RollingStd
    from mlforecast.utils import PredictionIntervals

    def slot(dates):  # half-hour of day, 0..47
        return dates.hour * 2 + dates.minute // 30

    def weekend(dates):
        return (dates.dayofweek >= 5).astype(int)

    fcst = MLForecast(
        models={"lgbm": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)},
        freq="30min",
        lags=[48, 96, 336],
        lag_transforms={
            48: [RollingMean(window_size=48), RollingStd(window_size=48), ExpandingMean()],
        },
        date_features=[slot, weekend],
    )
    return PredictionIntervals, fcst, lgb


@app.cell
def _(df, fcst):
    # 3. See exactly what the model trains on. Do this often, it is the best debugging tool.
    prep = fcst.preprocess(df)
    prep.head()
    return (prep,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Backtest with `cross_validation`

    `h=48, step_size=48` = one forecast per day, each refit on data up to that day (`refit=True`)
    or fit once (`refit=False`, much faster). Conformal intervals come from `PredictionIntervals`.
    """)
    return


@app.cell
def _(PredictionIntervals, df, fcst):
    cv = fcst.cross_validation(
        df,
        n_windows=7,
        h=48,
        step_size=48,
        refit=False,
        prediction_intervals=PredictionIntervals(n_windows=3, h=48),
        level=[80],
    )
    cv.head()
    return (cv,)


@app.cell
def _(cv, np, pd):
    # 5. Compare to the seasonal-naive baseline (lag_48 is already a column in the data).
    naive = cv.merge(
        cv[["unique_id", "ds", "y"]].assign(ds=lambda d: d["ds"] + pd.Timedelta(days=1)),
        on=["unique_id", "ds"],
        suffixes=("", "_naive"),
    )
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    cover = float(((cv["y"] >= cv["lgbm-lo-80"]) & (cv["y"] <= cv["lgbm-hi-80"])).mean())
    summary = pd.DataFrame(
        {
            "MAE lgbm": [mae(naive["y"], naive["lgbm"])],
            "MAE seasonal naive": [mae(naive["y"], naive["y_naive"])],
            "80% interval coverage": [cover],
        }
    )
    summary
    return


@app.cell
def _(cv, go, mo):
    _one = cv[cv["unique_id"] == "asset_0"]
    _fig = go.Figure()
    _fig.add_scatter(x=_one["ds"], y=_one["lgbm-hi-80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_one["ds"], y=_one["lgbm-lo-80"], fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                     line=dict(width=0), name="80% PI (conformal)", hoverinfo="skip")
    _fig.add_scatter(x=_one["ds"], y=_one["y"], name="actual", line=dict(color="black", width=1.2))
    _fig.add_scatter(x=_one["ds"], y=_one["lgbm"], name="lgbm", line=dict(color="#1f77b4", width=1.5))
    _fig.update_layout(
        title="Backtest on asset_0: forecast vs actual", yaxis_title="kW", hovermode="x unified", height=380,
        margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2),
    )
    _fig.update_xaxes(tickformat="%d %b<br>%H:%M")
    mo.vstack([_fig, mo.md("The grey-blue band is the conformal interval. Hover to compare forecast and actual at any half-hour.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Final forecast for tomorrow

    `fit` on everything, `predict(h=48)`. With lags >= 48 there is no recursion.
    """)
    return


@app.cell
def _(PredictionIntervals, df, fcst):
    fcst.fit(df, prediction_intervals=PredictionIntervals(n_windows=3, h=48))
    tomorrow = fcst.predict(h=48, level=[80])
    tomorrow.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try next
    1. Delete `RollingStd`, re-run the backtest. Does MAE move?
    2. Add `lags=[1]`. The backtest looks great, but recursive prediction over 48 steps is a
       different problem. Why?
    3. Swap in `LGBMRegressor(objective="quantile", alpha=0.1)` as extra models for native quantiles.
    4. Add `target_transforms=[Differences([48])]` (from `mlforecast.target_transforms`).
    5. Pass exogenous columns (weather) via `X_df` in `predict`.
    """)
    return


if __name__ == "__main__":
    app.run()
