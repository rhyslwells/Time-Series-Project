# Run from the project root (Time Series Project):
#   uv run marimo edit statsforecast/03_intervals.py

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import plotly.graph_objects as go

    if str(mo.notebook_dir()) not in sys.path:
        sys.path.insert(0, str(mo.notebook_dir()))
    import common as c

    return c, go, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Prediction intervals: parametric vs conformal

    statsforecast offers two routes to an 80% band:

    - **Parametric** (`SeasonalNaive`, `HistoricAverage`, `AutoETS`, `AutoARIMA`): derived from the model's own
      error assumptions (Gaussian, constant variance, variance growing with horizon).
    - **Conformal** (`ConformalIntervals(h=48, n_windows=3)`, any model): residuals from past rolling forecasts set
      the band width. No distributional assumption, but it needs `n_windows * h` extra history.
      Some models (`MSTL`, `SeasonalWindowAverage`) have *only* this option.

    Question: which is better calibrated (coverage near 80%) for this half-hourly data, and at what width?
    """)
    return


@app.cell
def _(c):
    from statsforecast.models import AutoETS, SeasonalNaive
    from statsforecast.utils import ConformalIntervals

    def _ci():
        return ConformalIntervals(h=48, n_windows=3)

    models = [
        SeasonalNaive(season_length=48, alias="naive_param"),
        SeasonalNaive(season_length=48, alias="naive_conformal", prediction_intervals=_ci()),
        AutoETS(season_length=48, model="ZZN", alias="ets_param"),
        AutoETS(season_length=48, model="ZZN", alias="ets_conformal", prediction_intervals=_ci()),
    ]
    sf = c.make_sf(models=models)
    return (sf,)


@app.cell
def _(c, sf):
    # A model without prediction_intervals uses its parametric formula; with it, conformal.
    df = c.make_df(n_assets=2, days=28)
    cv = sf.cross_validation(df=df, h=48, step_size=48, n_windows=5, level=[80])
    cv.head()
    return (cv,)


@app.cell
def _(c, cv, mo):
    mo.vstack([mo.md("## Coverage and width (target coverage 0.80)"), c.metrics_table(cv).round(3)])
    return


@app.cell
def _(cv, go, mo):
    _one = cv[cv["unique_id"] == "asset_0"].sort_values("ds")
    _fig = go.Figure()
    for _m, _col in [("naive_param", "rgba(127,127,127,0.25)"), ("naive_conformal", "rgba(31,119,180,0.25)")]:
        _fig.add_scatter(x=_one["ds"], y=_one[f"{_m}-hi-80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        _fig.add_scatter(x=_one["ds"], y=_one[f"{_m}-lo-80"], fill="tonexty", fillcolor=_col, line=dict(width=0), name=f"{_m} 80%")
    _fig.add_scatter(x=_one["ds"], y=_one["y"], name="actual", line=dict(color="black", width=1.1))
    _fig.add_scatter(x=_one["ds"], y=_one["naive_conformal"], name="forecast", line=dict(color="#1f77b4", width=1.3))
    _fig.update_layout(title="SeasonalNaive 80% bands on asset_0", yaxis_title="kW", hovermode="x unified", height=400,
                       margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    mo.vstack([_fig, mo.md("Neither band adapts to the solar ramp: width depends on the horizon step, not the time of day. "
                           "Compare with the quantile models in `mlforecast/06_quantile_vs_conformal.py`, which can.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways to check against the table
    1. Parametric bands assume Gaussian, homoscedastic errors. Net load with solar breaks both, so expect
       under- or over-coverage depending on the hour.
    2. Conformal fixes *average* coverage but not *shape*.
    3. For hour-varying bands you need per-slot conformal scores or quantile models (see mlforecast).
    """)
    return


if __name__ == "__main__":
    app.run()
