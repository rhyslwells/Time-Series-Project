""" """

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.insert(0, "../")

    import marimo as mo
    import polars as pl
    import numpy as np
    import warnings

    from ts_model_framework import SARIMAModel, ModelEvaluator, ResidualDiagnostics
    from ts_plots import TSPlotter

    warnings.filterwarnings("ignore")
    return ModelEvaluator, ResidualDiagnostics, SARIMAModel, TSPlotter, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    Numeric confirmation for SARIMA's residual diagnostics: does the Q-Q plot's normality
    read and the ACF's autocorrelation read actually hold up to a test, or is either "looks
    fine" / "looks off" call just eyeballing noise on a short test window?

    Fits the same SARIMA(1,1,1)x(1,1,1,48) pipeline as
    [`sarima.py`](sarima.py) on the same asset/split, then goes past the
    time-series + histogram view into:

    - **Q-Q plot** — is the normality assumption behind the prediction intervals reasonable?
    - **ACF / correlogram** — did the model leave autocorrelation on the table?
    - **Numeric confirmation** — `ResidualDiagnostics.ljung_box` and `.normality_stats`,
      checked against what the two plots show

    `TSPlotter.residuals_qq`/`residuals_acf` and `ResidualDiagnostics` work identically for
    ExponentialSmoothing, LightGBM and SeasonalNaive (they only need
    `y_test - forecast.prediction`) — this notebook uses SARIMA as the single worked example.
    Theory: [`docs_src/theory/diagnostics.md#confirming-diagnostics-numerically`](../../docs_src/theory/diagnostics.md#confirming-diagnostics-numerically).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Load, Split, Fit

    Same asset, split, and SARIMA order as `sarima.py` — this notebook picks up where that
    one's residual-diagnostic plot leaves off.
    """)
    return


@app.cell
def _(pl):
    df = pl.read_parquet("../../src/data/metering_data.parquet")
    asset_id = "ASSET_001"
    asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    y = asset_data.select("metering_kwh").to_numpy().flatten()

    test_split_idx = len(y) - (4 * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]

    print(f"Asset: {asset_id}")
    print(f"Train: {len(y_train)} obs ({len(y_train) / 48:.1f} days)")
    print(f"Test:  {len(y_test)} obs ({len(y_test) / 48:.1f} days)")
    return y_test, y_train


@app.cell
def _(SARIMAModel, y_train):
    # sarima = SARIMAModel(y_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 48)) #211
    sarima = SARIMAModel(y_train, order=(2, 0, 1), seasonal_order=(1, 1, 1, 48))
    # sarima = SARIMAModel(y_train, order=(0, 0, 1), seasonal_order=(1, 1, 1, 48))
    # sarima = SARIMAModel(y_train, order=(1, 0, 1), seasonal_order=(0, 1, 1, 48))

    sarima.fit()
    print(f"Fitted: {sarima.fitted}  |  AIC: {sarima.model.aic:.2f}")
    return (sarima,)


@app.cell
def _(ModelEvaluator, sarima, y_test):
    forecast_output = sarima.forecast(steps=len(y_test), confidence_level=0.80)
    metrics = ModelEvaluator.evaluate(y_test, forecast_output)
    residuals = y_test - forecast_output.prediction

    print(metrics)
    return forecast_output, residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Baseline Residuals Plot

    The existing time-series + histogram view, for reference before going numeric.
    See [Diagnostics — Residuals](../../docs_src/theory/diagnostics.md#residuals-diagnostic-time-series--histogram).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!-- The residuals-over-time plot confirms it. Residuals aren't bouncing around zero — they start around −1.5 to −2 early in the test window and drift up toward 0 by the end. That's diagnostics.md's "systematically biased" case, except it's a drifting bias rather than a constant one. Mean residual = −0.554 confirms it's off-center overall (actual < prediction more often than not — SARIMA is over-forecasting on average, though less so as the window progresses). The negative skew (−0.756) is the same drift showing up as a long left tail in the histogram. -->
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_baseline = TSPlotter.residuals_diagnostic(y_test, forecast_output, "SARIMA")
    mo.ui.plotly(fig_baseline)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Q-Q Plot — Is the Normality Assumption Reasonable?

    The framework's prediction intervals (`yhat +/- z * residual_std`) assume residuals are
    roughly normal. Points hugging the red line support that; a curved or S-shaped pattern
    is the visual form of "heavy tails" — real-world PI coverage runs below the reported
    target even when the residual mean is centered.

    See [Diagnostics — Confirming diagnostics numerically](../../docs_src/theory/diagnostics.md#confirming-diagnostics-numerically).
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_qq = TSPlotter.residuals_qq(y_test, forecast_output, "SARIMA")
    mo.ui.plotly(fig_qq)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: ACF / Correlogram — Did the Model Miss Structure?

    Bars outside the dashed 95% white-noise band ($\pm 1.96/\sqrt{n}$) flag autocorrelated
    residuals — the model left predictable structure on the table instead of reducing it to
    noise. For SARIMA specifically, a spike at lag 48 would mean the daily seasonal term
    (`P, D, Q`) isn't fully capturing the cycle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_acf = TSPlotter.residuals_acf(y_test, forecast_output, "SARIMA", nlags=20)
    mo.ui.plotly(fig_acf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Numeric Confirmation

    Ljung-Box (autocorrelation) and normality stats (skew, excess kurtosis, Jarque-Bera) for
    the same residuals — the threshold-based counterpart to eyeballing Sections 3-4.
    """)
    return


@app.cell
def _(ResidualDiagnostics, residuals):
    lb = ResidualDiagnostics.ljung_box(residuals)
    print("Ljung-Box (H0: residuals are white noise):")
    print(lb)
    return (lb,)


@app.cell
def _(ResidualDiagnostics, residuals):
    normality = ResidualDiagnostics.normality_stats(residuals)
    print("Normality stats:")
    for k, v in normality.items():
        print(f"  {k}: {v:.4f}")
    return (normality,)


@app.cell(hide_code=True)
def _(lb, mo, normality, residuals):
    lb_p = lb["lb_pvalue"][0]
    lb_verdict = (
        "fails to reject white noise (no significant autocorrelation)"
        if lb_p >= 0.05
        else "rejects white noise — residuals are autocorrelated at this lag"
    )

    jb_p = normality["jarque_bera_pvalue"]
    jb_verdict = (
        "fails to reject normality"
        if jb_p >= 0.05
        else "rejects normality"
    )

    mo.md(f"""
    ### Reading Sections 3-5 together

    - **Ljung-Box p-value = {lb_p:.4f}** — {lb_verdict}. Compare against Section 4: does a
      low p-value line up with a bar actually poking outside the ACF's confidence band, or
      is it a borderline call the plot doesn't clearly support (and vice versa)?
    - **Excess kurtosis = {normality["excess_kurtosis"]:.3f}**, **skew = {normality["skew"]:.3f}**,
      **Jarque-Bera p-value = {jb_p:.4f}** — {jb_verdict}. Compare against Section 3: do the
      Q-Q tails actually curve away from the line where kurtosis says they should?
    - Test window is {len(residuals)} points (4 days) — on a window this short, treat Jarque-Bera as a
      rough signal, not a firm verdict (see [Diagnostics](../../docs_src/theory/diagnostics.md#confirming-diagnostics-numerically)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Next Steps

    - If Ljung-Box rejects white noise at lag 48 specifically: revisit the seasonal order
      `(P,D,Q,48)` — see [Models — SARIMA](../../docs_src/theory/models.md#sarimapdqxpdqs).
    - If normality is rejected with real excess kurtosis: the z-score prediction interval is
      miscalibrated — check PI coverage falls short of 80%, and consider quantile regression
      (see [Metrics — proper scoring rules](../../docs_src/theory/metrics.md#proper-scoring-rules-for-intervals)).
    - To compare against ExponentialSmoothing/LightGBM/SeasonalNaive: swap the model in
      Section 1 — `TSPlotter.residuals_qq`/`residuals_acf` and `ResidualDiagnostics` take
      any `ForecastOutput`, unchanged.
    """)
    return


if __name__ == "__main__":
    app.run()
