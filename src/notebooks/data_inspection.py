"""
Time Series Data Inspection
============================

Purpose:
    Model-independent inspection of a single time series field, ahead of any
    forecasting work. Establishes distribution, trend, time-of-day and
    day-of-week patterns, autocorrelation structure, and rolling mean/variance
    for the field of interest.

Data:
    src/data/metering_data.parquet (metering_kwh), filtered to one asset_id.
    Uses only timestamp + one value column, so it does not depend on
    src/data/generate_metering_features.py - the checks apply to any dataset
    with a single field of interest.

Depends on:
    src/ts_data_checks.py - DataInspector

Flow (sections):
    1. Data Loading           - load, select asset, field of interest
    2. Distribution and Range - summary stats, histogram
    3. Series Over Time       - raw series + rolling mean (trend/regime changes)
    4. Time-of-Day Patterns   - boxplot by half-hour slot
    5. Day-of-Week Patterns   - boxplot by weekday
    6. Autocorrelation        - ACF and PACF
    7. Rolling Mean & Variance
    8. Stationarity Tests     - ADF and KPSS, combined read
    9. Seasonal Decomposition - STL trend/seasonal/residual, strength scores

Scope (see working_notes/8_gpt1/data_inspection.md):
    Missing values, timestamp validation, outlier detection/treatment, and
    production data-quality checks are deferred - this notebook is about
    temporal structure, not data integrity.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import marimo as mo
    import polars as pl

    from ts_data_checks import DataInspector

    return DataInspector, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Time Series Data Inspection

    Model-independent checks on one field of interest, before fitting any forecasting model.
    See [`working_notes/8_gpt1/data_inspection.md`](../../working_notes/8_gpt1/data_inspection.md)
    for the checklist this notebook implements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Data Loading

    Load `metering_data.parquet` and pick one asset and the field of interest.
    """)
    return


@app.cell
def _(pl):
    df = pl.read_parquet("../../src/data/metering_data.parquet")

    value_col = "metering_kwh"
    timestamp_col = "timestamp"
    asset_id = "ASSET_001"

    asset_df = df.filter(pl.col("asset_id") == asset_id).sort(timestamp_col)

    print("Data Summary:")
    print(f" Asset: {asset_id}")
    print(f" Type: {asset_df['asset_type'][0]}")
    print(f" Field of interest: {value_col}")
    print(f" Records: {asset_df.height}")
    print(f" Date range: {asset_df[timestamp_col].min()} to {asset_df[timestamp_col].max()}")
    return asset_df, timestamp_col, value_col


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Distribution and Range

    Overall shape of the target - central tendency, spread, and extremes.
    """)
    return


@app.cell
def _(DataInspector, asset_df, value_col):
    stats = DataInspector.summary_stats(asset_df, value_col)
    stats
    return


@app.cell
def _(DataInspector, asset_df, mo, value_col):
    fig_dist = DataInspector.distribution_plot(asset_df, value_col)
    mo.ui.plotly(fig_dist)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Series Over Time

    Raw series with a rolling mean overlay - look for trend, level shifts, or changes in behaviour.
    """)
    return


@app.cell
def _(DataInspector, asset_df, mo, timestamp_col, value_col):
    fig_series = DataInspector.series_plot(asset_df, value_col, timestamp_col, rolling_window=48)
    mo.ui.plotly(fig_series)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Time-of-Day Patterns

    Distribution of the target at each half-hour slot, across all days.
    """)
    return


@app.cell
def _(DataInspector, asset_df, mo, timestamp_col, value_col):
    fig_tod = DataInspector.time_of_day_plot(asset_df, value_col, timestamp_col)
    mo.ui.plotly(fig_tod)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Day-of-Week Patterns

    Distribution of the target by weekday - weekly seasonality, where observable.
    """)
    return


@app.cell
def _(DataInspector, asset_df, mo, timestamp_col, value_col):
    fig_dow = DataInspector.day_of_week_plot(asset_df, value_col, timestamp_col)
    mo.ui.plotly(fig_dow)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Autocorrelation

    ACF and PACF of the raw series (not residuals) - structure available to an autoregressive model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the ACF/PACF for ASSET_001's metering_kwh (672 points, 30-min data, so lag 48 = 24 hours), and it's telling you two things:

    Strong short-lag persistence. ACF starts at 1, drops to ~0.75 at lag 1, then decays smoothly to near zero by lag ~6-8. PACF is essentially all in that lag-1 spike (~0.75) with nothing else large until much later lags — meaning the short-range correlation is well explained by an AR(1)-like relationship, not a long AR chain. Consecutive half-hours are highly correlated, as expected for a smooth load curve.

    Daily seasonality at lag 48. ACF dips negative around lag 8-20, recovers to a local bump near lag 24 (~0.28), dips negative again, then climbs steeply back up to ~0.85 near lag 48. That's the signature of a 24-hour cycle: the same time-of-day tomorrow looks almost as similar as five minutes ago. The secondary bump at lag ~24 (12 hours) suggests a weaker sub-daily echo too, consistent with EV charging's two-peak (morning/evening) shape.

    Critically, PACF still shows meaningful spikes out near lag 44-48 (~0.35-0.4) even after the lag-1 term is accounted for. That means the daily seasonality is a genuine effect the short lags don't already explain, not just an artifact of lag-1 correlation compounding. Practically: this data wants a seasonal component at period 48, e.g. SARIMAModel(..., seasonal_order=(P, D, Q, 48)), or lag features at [1, 48] for LightGBMModel — matching what ts_model_explorer.py already configures (season_length=48).
    """)
    return


@app.cell
def _(DataInspector, asset_df, mo, value_col):
    y = asset_df[value_col].to_numpy()
    fig_acf_pacf = DataInspector.acf_pacf_plot(y, nlags=48, title=value_col)
    mo.ui.plotly(fig_acf_pacf)


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Rolling Mean & Variance

    A simple, visual stationarity check - flat rolling mean/variance suggests a stationary series.
    """)
    return


@app.cell
def _(DataInspector, asset_df, mo, timestamp_col, value_col):
    fig_rolling = DataInspector.rolling_stats_plot(asset_df, value_col, timestamp_col, window=48)
    mo.ui.plotly(fig_rolling)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 8: Stationarity Tests

    ADF and KPSS test different null hypotheses - agreement is conclusive, disagreement
    narrows down whether the series needs detrending or differencing.
    """)
    return


@app.cell
def _(DataInspector, asset_df, value_col):
    y_stationarity = asset_df[value_col].to_numpy()
    stationarity = DataInspector.stationarity_tests(y_stationarity)
    stationarity
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 9: Seasonal Decomposition

    STL splits the series into trend, seasonal, and residual components. Strength scores
    (Hyndman/Wang, each in [0, 1]) put a number on how much of the series each component
    explains, rather than reading it off the ACF/PACF plot.
    """)
    return


@app.cell
def _(DataInspector, asset_df, value_col):
    strength = DataInspector.seasonal_strength(asset_df, value_col, period=48)
    strength
    return


@app.cell
def _(DataInspector, asset_df, mo, value_col):
    fig_stl = DataInspector.seasonal_decomposition_plot(asset_df, value_col, period=48)
    mo.ui.plotly(fig_stl)
    return


if __name__ == "__main__":
    app.run()
