# Run from the project root (Time Series Project):
#   uv run marimo edit statsforecast/02_model_comparison.py

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
    # Do statistical models beat the seasonal baselines?

    Rolling-origin backtest with `StatsForecast.cross_validation`: each window is one day ahead
    (`h=48, step_size=48`), refit on data before that day. Metrics are MAE / RMSE and **skill vs seasonal
    naive** (positive = better than "same slot yesterday").

    Every window refits six models, so this takes a minute. Use fewer windows to go faster.
    """)
    return


@app.cell
def _(mo):
    n_windows = mo.ui.slider(3, 10, value=5, label="backtest days")
    noise = mo.ui.slider(1, 8, value=4, step=1, label="noise sd (kW)")
    mo.hstack([n_windows, noise])
    return n_windows, noise


@app.cell
def _(c, n_windows, noise):
    df = c.make_df(n_assets=2, days=28, noise=float(noise.value))
    cv = c.make_sf().cross_validation(df=df, h=48, step_size=48, n_windows=n_windows.value, level=[80])
    cv.head()
    return (cv,)


@app.cell
def _(c, cv, mo):
    mo.vstack([mo.md("## Overall"), c.metrics_table(cv).round(3)])
    return


@app.cell
def _(c, cv, go, mo):
    # Error by time of day: where does each model lose?
    _long = cv.melt(id_vars=["unique_id", "ds", "y"], value_vars=c.model_cols(cv), var_name="model", value_name="pred")
    _long["abs_err"] = (_long["y"] - _long["pred"]).abs()
    _long["slot"] = _long["ds"].dt.hour * 2 + _long["ds"].dt.minute // 30
    _prof = _long.groupby(["model", "slot"])["abs_err"].mean().reset_index()
    _fig = go.Figure()
    for _m, _g in _prof.groupby("model"):
        _fig.add_scatter(x=_g["slot"] / 2, y=_g["abs_err"], name=_m, line=dict(color=c.COLORS[_m], width=1.6))
    _fig.update_layout(title="MAE by hour of day", xaxis_title="hour", yaxis_title="MAE (kW)", height=380,
                       hovermode="x unified", margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    mo.vstack([_fig, mo.md("Errors cluster around the solar ramp (cloudiness varies by day), which no univariate model can anticipate.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the result

    - A univariate model can only extrapolate the *shape* of recent days. Tomorrow's cloud cover is
      information it does not have, so a floor on error remains. That is where mlforecast with weather
      features, or an exogenous `X_df`, earns its keep.
    - `AutoETS` with 48 seasonal states over ~3 weeks has little data per state: expect it to overfit or
      lose to simple averages. `MSTL` sidesteps that by estimating the seasonal profile separately.
    - A smoothed seasonal average (`SeasonalWindowAverage`) is often the model to beat. Check whether
      `MSTL_ARIMA` justifies its extra cost.
    """)
    return


if __name__ == "__main__":
    app.run()
