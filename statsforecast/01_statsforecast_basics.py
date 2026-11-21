# Run from the project root (Time Series Project):
#   uv run marimo edit statsforecast/01_statsforecast_basics.py
#   uv run marimo run statsforecast/01_statsforecast_basics.py


import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    if str(mo.notebook_dir()) not in sys.path:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # statsforecast basics, day-ahead half-hourly

    statsforecast fits a **classical model per series** (ETS, ARIMA, MSTL, seasonal baselines). There is no
    feature table: the model sees only the history of `y`. Compare with mlforecast, which builds lag
    features and fits *one* ML model across series.

    | | statsforecast | mlforecast |
    |---|---|---|
    | Model scope | one fit per series | one fit on stacked series |
    | Inputs | history of `y` (plus optional exog) | engineered lag/date features |
    | Intervals | parametric (model maths) or conformal | quantile models or conformal |
    | Strength | strong seasonal baselines, fast, interpretable | exploits exogenous drivers |

    Input is the same long format `(unique_id, ds, y)`.
    """)
    return


@app.cell
def _(c):
    df = c.make_df(n_assets=2, days=28)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Smallest possible forecast

    `SeasonalNaive(season_length=48)` = "same half-hour yesterday". Every other model has to beat this.
    `StatsForecast.forecast` fits and predicts in one call (nothing is kept, which is why it is fast).
    """)
    return


@app.cell
def _(c, df):
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    naive_only = StatsForecast(models=[SeasonalNaive(season_length=c.SEASON)], freq="30min")
    naive_fc = naive_only.forecast(df=df, h=48, level=[80])
    naive_fc.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Use the model zoo

    `common.make_models()` returns six models. Why each is there:

    - `HistoricAverage`: the dumbest benchmark (one constant). Sanity floor.
    - `SeasonalNaive`: yesterday. Cheap and hard to beat for stable assets.
    - `SeasonalWindowAverage(window_size=7)`: average of the last 7 days at each slot. Smooths noise.
    - `AutoETS`: automatic exponential smoothing. With `season_length=48` it carries 48 seasonal states, so it is
      slow and (here) badly behaved. Worth seeing why.
    - `MSTL`: STL decomposition into several seasonalities, then a simple model on the deseasonalised remainder.
      This is the statsforecast answer to *high-frequency, multi-seasonal* data.
    - `MSTL_ARIMA`: same, with daily *and* weekly seasons and ARIMA on the trend.
    """)
    return


@app.cell
def _(c, df):
    sf = c.make_sf()
    fc = sf.forecast(df=df, h=48, level=[80])
    fc.head()
    return (fc,)


@app.cell
def _(c, df, fc, go, mo):
    _uid = "asset_0"
    _h = df[df["unique_id"] == _uid].tail(48 * 3)
    _f = fc[fc["unique_id"] == _uid]
    _fig = go.Figure()
    _fig.add_scatter(x=_h["ds"], y=_h["y"], name="history", line=dict(color="black", width=1.2))
    for _m in c.model_cols(fc):
        _fig.add_scatter(x=_f["ds"], y=_f[_m], name=_m, line=dict(color=c.COLORS[_m], width=1.5))
    _fig.update_layout(title="Tomorrow's forecast, asset_0", yaxis_title="kW", hovermode="x unified",
                       height=400, margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    mo.vstack([_fig, mo.md("Click legend entries to isolate models. Note how differently each treats the solar dip.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try
    1. Change `window_size` in `SeasonalWindowAverage` to 1, 3, 14. Window 1 *is* seasonal naive.
    2. Time `AutoETS` with `season_length=48` vs `season_length=1` (no seasonality). Why is the first so slow?
    3. Pass a second series with a very different shape. Models are fitted per `unique_id`, with no sharing.
    4. Add exogenous drivers: `AutoARIMA` and `MSTL` accept extra columns in `df` plus `X_df` for the future.
    """)
    return


if __name__ == "__main__":
    app.run()
