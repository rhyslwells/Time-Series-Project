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
    import warnings

    from ts_models import LightGBMModel
    from ts_evaluation import ModelEvaluator, ResidualDiagnostics
    from ts_plots import TSPlotter

    warnings.filterwarnings("ignore")
    return LightGBMModel, ModelEvaluator, ResidualDiagnostics, TSPlotter, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    Numeric confirmation for LightGBM's residual diagnostics — same treatment as
    [`sarima_residual_diagnostics.py`](sarima_residual_diagnostics.py) (Q-Q plot, ACF
    correlogram, Ljung-Box, normality stats), same asset/split, same
    `metering_data.parquet` source, swapping in `ts_models.LightGBMModel`
    (lags `[1, 2, 48, 96]` + hour-of-day) for SARIMA.

    - **Q-Q plot** — is the normality assumption behind the prediction intervals reasonable?
    - **ACF / correlogram** — did the model leave autocorrelation on the table, or is it
      recursive-forecast error compounding (a LightGBM-specific caveat — see Section 4)?
    - **Numeric confirmation** — `ResidualDiagnostics.ljung_box` and `.normality_stats`,
      checked against what the two plots show

    `TSPlotter.residuals_qq`/`residuals_acf` and `ResidualDiagnostics` work identically for
    SARIMA, ExponentialSmoothing and SeasonalNaive (they only need
    `y_test - forecast.prediction`) — this notebook uses LightGBM as the worked example.
    Theory: [`docs_src/theory/diagnostics.md#confirming-diagnostics-numerically`](../../docs_src/theory/diagnostics.md#confirming-diagnostics-numerically).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Load, Split, Fit

    Same asset and split as `sarima_residual_diagnostics.py`.
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
def _(LightGBMModel, y_train):
    lgbm = LightGBMModel(y_train, lags=[1, 2, 48, 96])
    lgbm.fit()
    print(f"Fitted: {lgbm.fitted}")
    return (lgbm,)


@app.cell
def _(ModelEvaluator, lgbm, y_test):
    forecast_output = lgbm.forecast(steps=len(y_test), confidence_level=0.80)
    metrics = ModelEvaluator.evaluate(y_test, forecast_output)
    residuals = y_test - forecast_output.prediction

    print(metrics)
    return forecast_output, metrics, residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Baseline Residuals Plot

    The existing time-series + histogram view, for reference before going numeric.
    See [Diagnostics — Residuals](../../docs_src/theory/diagnostics.md#residuals-diagnostic-time-series--histogram).
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_baseline = TSPlotter.residuals_diagnostic(y_test, forecast_output, "LightGBM")
    mo.ui.plotly(fig_baseline)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Q-Q Plot — Is the Normality Assumption Reasonable?

    LightGBM's prediction interval (`yhat +/- z * residual_std`, constant width across the
    horizon — see [Diagnostics — Uncertainty width](../../docs_src/theory/diagnostics.md#uncertainty-width-over-time))
    assumes residuals are roughly normal. Points hugging the red line support that; a curved
    or S-shaped pattern is the visual form of "heavy tails."

    See [Diagnostics — Confirming diagnostics numerically](../../docs_src/theory/diagnostics.md#confirming-diagnostics-numerically).
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_qq = TSPlotter.residuals_qq(y_test, forecast_output, "LightGBM")
    mo.ui.plotly(fig_qq)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 4: ACF / Correlogram — Structure Missed, or Recursive-Forecast Compounding?

    Bars outside the dashed 95% white-noise band ($\pm 1.96/\sqrt{n}$) flag autocorrelated
    residuals. For LightGBM specifically, don't read this the way you would for SARIMA:
    [Models — LightGBM pitfall](../../docs_src/theory/models.md#lightgbm-gradient-boosting)
    notes that `forecast()` is recursive — past roughly step 96, every lag feature is itself
    a prior prediction, so errors compound with horizon by construction. Check whether
    autocorrelation *grows* past step 96 (consistent with compounding) rather than treating
    any nonzero bar as "the model missed structure."
    """)
    return


@app.cell
def _(TSPlotter, forecast_output, mo, y_test):
    fig_acf = TSPlotter.residuals_acf(y_test, forecast_output, "LightGBM", nlags=20)
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
    jb_verdict = "fails to reject normality" if jb_p >= 0.05 else "rejects normality"

    mo.md(f"""
    ### Reading Sections 3-5 together

    - **Ljung-Box p-value = {lb_p:.4f}** — {lb_verdict}. Before concluding the model missed
      structure, check Section 4: does autocorrelation *grow* past step 96 (consistent with
      recursive-forecast compounding), or is it flat/present from step 1 (a genuine fit
      problem instead)?
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
    ## Section 6: Feature Importance

    LightGBM-specific — no SARIMA equivalent. Which lag or the hour-of-day feature is the
    model actually relying on?
    """)
    return


@app.cell
def _(lgbm, pl):
    importances = pl.DataFrame(
        {
            "feature": lgbm.feature_names,
            "importance": lgbm.model.feature_importances_,
        }
    ).sort("importance", descending=True)

    print(importances)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Next Steps

    - If Ljung-Box rejects white noise and autocorrelation grows with horizon: that's the
      recursive-forecast compounding problem, not a fixable fit issue — the fix is on the
      forecasting strategy (direct multi-step, DirRec) or the interval (quantile regression),
      not on `num_leaves`/`learning_rate` — see
      [Models — LightGBM pitfall](../../docs_src/theory/models.md#lightgbm-gradient-boosting).
    - If normality is rejected with real excess kurtosis: consider quantile regression (see
      [Metrics — proper scoring rules](../../docs_src/theory/metrics.md#proper-scoring-rules-for-intervals)).
    - To compare against SARIMA/ExponentialSmoothing/SeasonalNaive: swap the model in
      Section 1 — `TSPlotter.residuals_qq`/`residuals_acf` and `ResidualDiagnostics` take
      any `ForecastOutput`, unchanged. See
      [`sarima_residual_diagnostics.py`](sarima_residual_diagnostics.py) for the SARIMA version.
    """)
    return


if __name__ == "__main__":
    app.run()
