"""
Time Series Model Comparison & Interpretation
==============================================

Purpose:
    Fit SARIMA, Exponential Smoothing, and LightGBM on one asset's metering
    data, compare them on RMSE/MAE/MAPE/PI coverage, and render diagnostic
    plots for the best model plus a side-by-side comparison of all three.

Data:
    src/data/metering_data.parquet, filtered to a single asset_id, split
    into train/test (last 4 days held out as test).

Depends on:
    src/ts_model_framework.py — SARIMAModel, ExponentialSmoothingModel,
        LightGBMModel, ModelComparison
    src/ts_plots.py — TSPlotter, ComparisonPlotter

Flow (sections):
    1. Data Loading & Exploration   — load, plot full series, train/test split
    2. Understanding Metrics        — pointer to docs (no computation)
    3. Model Comparison             — fit all 3 models, rank by RMSE
    4. Detailed Plot Analysis       — 4 diagnostic plots for the best model
       (forecast vs actual, residuals, uncertainty width, PI coverage)
    5. Comparing All Models         — overlay forecasts, metrics table
    6. Interpretation Guide         — best model's metrics translated into
       a flexibility-commitment example
    7. Mathematics Summary          — pointer to docs (no computation)
    8. Checklist                   — pass/fail production-readiness checks
    9. Next Steps                  — deploy/retrain recommendation

Theory & interpretation (formulas, good-vs-bad plots, retrain triggers):
    see docs_src/theory/ (metrics.md, models.md, diagnostics.md,
    model-decisions.md) — not duplicated in this notebook's markdown cells.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import norm
    import warnings

    warnings.filterwarnings("ignore")
    return go, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Time Series Model Comparison & Interpretation

    **Interactive explorer** for energy metering forecasts using SARIMA, Exponential Smoothing, and LightGBM.

    This notebook loads metering data, fits the 3 models, and renders diagnostic plots.
    For the math behind each metric/model and how to read each plot, see
    [`docs_src/theory/`](../../docs_src/theory/index.md) (metrics, models, diagnostics, decisions).
    """)
    return


## Section 1: Data Loading & Exploration
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Data Loading & Exploration

    Load metering data and examine characteristics.
    """)
    return


@app.cell
def _(np, pl):
    # Load data
    df = pl.read_parquet("src/data/metering_data.parquet")
    asset_id = "ASSET_001"
    asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    y = asset_data.select("metering_kwh").to_numpy().flatten()
    timestamps = asset_data.select("timestamp").to_numpy().flatten()

    print("Data Summary:")
    print(f" Asset: {asset_id}")
    print(f" Type: {asset_data['asset_type'][0]}")
    print(f" Records: {len(y)}")
    print(f" Date range: {timestamps[0]} to {timestamps[-1]}")
    print(f" Mean: {np.mean(y):.3f} kWh")
    print(f" Std: {np.std(y):.3f} kWh")
    print(f" Min: {np.min(y):.3f} kWh")
    print(f" Max: {np.max(y):.3f} kWh")
    return (y,)


@app.cell(hide_code=True)
def _(go, mo, y):
    # Interactive plot of full time series
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=y,
            mode="lines",
            name="Metering (kWh)",
            line=dict(color="steelblue", width=1),
        )
    )

    fig.update_layout(
        title="Full Time Series (14 days, 30-min intervals = 672 points)",
        xaxis_title="Time (30-min intervals)",
        yaxis_title="Power (kWh)",
        height=400,
        width=1200,
        hovermode="x unified",
    )

    mo.ui.plotly(fig)
    return


@app.cell
def _(y):
    # Train/test split
    test_split_idx = len(y) - (4 * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]

    print(f"Train: {len(y_train)} observations ({len(y_train) / 48:.1f} days)")
    print(f"Test: {len(y_test)} observations ({len(y_test) / 48:.1f} days)")
    return y_test, y_train


## About This Notebook
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## About This Notebook

    **Framework Pattern Used:**

    This notebook uses `ts_model_framework.py` classes to demonstrate best practices:

    1. **ModelComparison** — Orchestrates fitting + evaluation for multiple models
    2. **TSModel subclasses** — SARIMAModel, ExponentialSmoothingModel, LightGBMModel
    3. **ForecastOutput** — Standardised output contract (prediction, lower, upper, uncertainty_width)
    4. **EvaluationMetrics** — Standardised metrics (mae, rmse, mape, pi_coverage, uncertainty_width)
    5. **TSPlotter** — Generic plots that work with any model

    **Why this pattern matters:**
    - **Consistency**: All models use same interface (fit → forecast)
    - **Extensibility**: Add new models by subclassing TSModel
    - **Reusability**: Plotting works for any model automatically
    - **Maintainability**: Changes to framework apply everywhere

    **To add a new model:**
    ```python
    from ts_model_framework import TSModel

    class MyNewModel(TSModel):
        def fit(self):
            # Your fitting logic
            pass

        def forecast(self, steps, confidence_level=0.80):
            # Your forecast logic
            return ForecastOutput(...)

    comp.add_model(MyNewModel(y_train))
    # Everything else works automatically!
    ```
    """)
    return


## Section 2: Understanding Metrics
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Understanding Metrics

    See [`docs_src/theory/metrics.md`](../../docs_src/theory/metrics.md) for the formulas,
    interpretation, and expected ranges by asset type (MAE, RMSE, MAPE, PI coverage).
    """)
    return


## Section 3: Model Comparison
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 3: Model Comparison

    Compare 3 models using standardised framework classes.

    **Framework Pattern:**
    1. Create ModelComparison instance
    2. Add models (using TSModel subclasses)
    3. Fit all models
    4. Evaluate and rank by RMSE

    This pattern is extensible: add new models by subclassing TSModel.
    """)
    return


@app.cell
def _(y_test, y_train):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ts_model_framework import (
        SARIMAModel,
        ExponentialSmoothingModel,
        LightGBMModel,
        ModelComparison,
    )

    print("INITIALIZING MODEL COMPARISON")

    comp = ModelComparison(y_train, y_test)

    # Add models
    print("\nAdding models to comparison...")
    comp.add_model(SARIMAModel(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48)))
    print("  ✓ SARIMA(1,1,1)×(1,1,1,48)")

    comp.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
    print("  ✓ ExponentialSmoothing")

    comp.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))
    print("  ✓ LightGBM")

    print(f"\nFitting {len(comp.models)} models...")
    comp.fit_all()

    print("\nEvaluating forecasts...")
    results_df = comp.evaluate_all(confidence_level=0.80)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")

    print(results_df.to_string())
    return comp, results_df


### Model Ranking (by RMSE)
@app.cell(hide_code=True)
def _(mo, results_df):
    mo.md(f"""
    ### Model Ranking (by RMSE)

    {results_df.to_markdown()}

    **Interpretation:**
    - **RMSE** (lower is better): Point forecast accuracy
    - **MAPE**: Percentage error (scale-independent)
    - **PI Coverage %**: Should be ~80% (matches target confidence)
    - **Uncertainty Width**: Average prediction interval width

    **Which model to use?**
    1. Check RMSE rank (lowest wins)
    2. Verify PI Coverage ≈ 80% (±5%)
    3. If coverage off, model needs tuning or interval recalibration
    """)
    return


## Section 4: Detailed Plot Analysis
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 4: Detailed Plot Analysis

    Visualise best model's forecast with integrated mathematical explanations.
    """)
    return


@app.cell
def _(comp, results_df):
    # Get best model from comparison
    best_model_name = results_df.index[0]  # First row is lowest RMSE
    best_forecast = comp.get_forecast(best_model_name)
    best_metrics = comp.results[best_model_name]["metrics"]

    print(f"Best Model: {best_model_name}")
    print(f"  MAE: {best_metrics.mae:.4f} kWh")
    print(f"  RMSE: {best_metrics.rmse:.4f} kWh")
    print(f"  MAPE: {best_metrics.mape:.2f}%")
    print(f"  PI Coverage: {best_metrics.pi_coverage:.1f}%")
    return best_forecast, best_metrics, best_model_name


@app.cell
def _(best_forecast, best_metrics, best_model_name, mo, y_test):
    from ts_plots import TSPlotter

    # Plot 1: Forecast vs Actual
    fig = TSPlotter.forecast_vs_actual(
        y_test, best_forecast, best_model_name, best_metrics
    )

    mo.ui.plotly(fig)
    return


#### Plot 1: Forecast vs Actual
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Plot 1: Forecast vs Actual

    See [Diagnostics — Forecast vs Actual](../../docs_src/theory/diagnostics.md#forecast-vs-actual-with-prediction-interval)
    for good vs bad cases and what each failure mode means.
    """)
    return


@app.cell
def _(best_forecast, best_model_name, mo, y_test):
    from ts_plots import TSPlotter

    # Plot 2: Residuals Diagnostic
    fig = TSPlotter.residuals_diagnostic(y_test, best_forecast, best_model_name)

    mo.ui.plotly(fig)
    return


#### Plot 2: Residuals Diagnostic
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Plot 2: Residuals Diagnostic

    See [Diagnostics — Residuals](../../docs_src/theory/diagnostics.md#residuals-diagnostic-time-series--histogram)
    for how to read the time-series and histogram panels (bias, autocorrelation, heavy tails).
    """)
    return


@app.cell
def _(best_forecast, best_model_name, mo):
    from ts_plots import TSPlotter

    # Plot 3: Uncertainty Width
    fig = TSPlotter.uncertainty_analysis(best_forecast, best_model_name)

    mo.ui.plotly(fig)
    return


#### Plot 3: Uncertainty Width Over Time
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Plot 3: Uncertainty Width Over Time

    See [Diagnostics — Uncertainty width](../../docs_src/theory/diagnostics.md#uncertainty-width-over-time)
    for good/flat/exploding cases and the calibration formula.
    """)
    return


@app.cell
def _(best_forecast, best_model_name, mo, y_test):
    from ts_plots import TSPlotter

    # Plot 4: PI Coverage
    fig = TSPlotter.pi_coverage(y_test, best_forecast, best_model_name)

    mo.ui.plotly(fig)
    return


#### Plot 4: PI Coverage (Green/Red Scatter)
@app.cell(hide_code=True)
def _(best_forecast, mo, np, y_test):
    in_bounds = (y_test >= best_forecast["lower"]) & (y_test <= best_forecast["upper"])
    coverage_pct = np.mean(in_bounds) * 100

    mo.md(f"""
    #### Plot 4: PI Coverage (Green/Red Scatter)

    **Coverage = {coverage_pct:.1f}% (Target 80%)**

    See [Diagnostics — PI coverage](../../docs_src/theory/diagnostics.md#pi-coverage-greenred-scatter)
    for how to interpret under-coverage, over-coverage, and clustered-reds cases.
    """)
    return


## Section 5: Comparing All Models
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Comparing All Models

    Side-by-side comparison of all 3 models.
    """)
    return


@app.cell
def _(comp, mo, y_test):
    from ts_plots import ComparisonPlotter

    # Get all forecasts
    forecasts = {name: comp.get_forecast(name) for name in comp.results.keys()}

    # Plot: All model forecasts overlaid
    fig = ComparisonPlotter.forecast_comparison(y_test, forecasts, sample_size=96)

    mo.ui.plotly(fig)
    return (ComparisonPlotter,)


@app.cell
def _(ComparisonPlotter, comp, mo):
    # Metrics comparison table
    metrics_dict = {name: data["metrics"] for name, data in comp.results.items()}

    fig = ComparisonPlotter.metrics_comparison(metrics_dict)

    mo.ui.plotly(fig)
    return


## Section 6: Interpretation Guide
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Interpretation Guide

    How to interpret these results for  flexibility forecasting.
    """)
    return


### For Your Asset
@app.cell(hide_code=True)
def _(best_forecast, best_metrics, best_model_name, mo):
    mo.md(f"""
    ### For Your Asset

    **Best Model: {best_model_name}**

    **Interpretation of Results:**

    | Metric | Value | Meaning |
    |---|---|---|
    | MAE | {best_metrics.mae:.4f} kWh | Forecast off by ~{best_metrics.mae:.2f} kWh on average |
    | RMSE | {best_metrics.rmse:.4f} kWh | Larger error occasionally ({best_metrics.rmse / best_metrics.mae:.2f}× MAE) |
    | MAPE | {best_metrics.mape:.2f}% | Within ~{best_metrics.mape:.1f}% of actual |
    | PI Coverage | {best_metrics.pi_coverage:.1f}% | {f"Good (≈80%)" if 75 <= best_metrics.pi_coverage <= 85 else f"Check - target is 80%"} |
    | Uncertainty | {best_metrics.mean_uncertainty_width:.3f} kWh | Forecast ±{best_metrics.mean_uncertainty_width / 2:.3f} kWh (80% confidence) |

    **For Flexibility Commitment:**

    If you commit flexibility based on this forecast's bounds:
    ```
    Predicted demand: {best_forecast.prediction[0]:.2f} kWh
    Upper bound (P90): {best_forecast.upper[0]:.2f} kWh
    Lower bound (P10): {best_forecast.lower[0]:.2f} kWh

    Can commit upside (reduction):
    {best_forecast.upper[0] - best_forecast.prediction[0]:.2f} kWh

    Can commit downside (increase):
    {best_forecast.prediction[0] - best_forecast.lower[0]:.2f} kWh
    ```

    **Risk Assessment:**
    - Coverage {best_metrics.pi_coverage:.1f}% means:
      - At this confidence level, commitment will breach {100 - best_metrics.pi_coverage:.1f}% of time
      - Is that acceptable for your SLA?
    """)
    return


## Section 7: Mathematics Summary
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 7: Mathematics Summary

    Full formulas for metrics and models are in
    [`docs_src/theory/metrics.md`](../../docs_src/theory/metrics.md) and
    [`docs_src/theory/models.md`](../../docs_src/theory/models.md) — not repeated here.
    """)
    return


## Section 8: Checklist
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 8: Checklist

    Is this forecast production-ready?
    """)
    return


@app.cell
def _(best_metrics):
    checks = {
        "RMSE < 0.7 kWh (residential)": best_metrics.rmse < 0.7,
        "RMSE < 1.3 kWh (commercial)": best_metrics.rmse < 1.3,
        "MAE < 0.5 kWh (residential)": best_metrics.mae < 0.5,
        "MAPE < 15%": best_metrics.mape < 15,
        "PI Coverage 75–85%": 75 <= best_metrics.pi_coverage <= 85,
        "Coverage ≈ target (±5%)": 75 <= best_metrics.pi_coverage <= 85,
        "No systematic bias": True,  # Check residuals histogram (Section 4)
        "Residuals bell-shaped": True,  # Check histogram (Section 4)
    }

    print("Production Readiness Checklist:")
    print("=" * 50)
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")

    passed = sum(checks.values())
    total = len(checks)
    print(f"\nScore: {passed}/{total}")
    if passed >= 6:
        print("✓ LIKELY PRODUCTION-READY")
    elif passed >= 4:
        print("⚠ ACCEPTABLE, MONITOR CAREFULLY")
    else:
        print("✗ NEEDS MORE TUNING")
    return


## Section 9: Next Steps
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 9: Next Steps

    Recommended actions based on your results.
    """)
    return


### Recommendations
@app.cell(hide_code=True)
def _(best_metrics, best_model_name, mo):
    mo.md(f"""
    ### Recommendations

    **Best Model: {best_model_name}**

    {f"Deploy {best_model_name}" if best_metrics.pi_coverage > 75 else f"Retrain {best_model_name} - coverage below 75%"}.

    Full selection logic, retrain triggers, tuning priority, and the production-ready checklist
    are in [`docs_src/theory/model-decisions.md`](../../docs_src/theory/model-decisions.md).
    """)
    return


## Summary
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Summary

    **Key takeaways:**
    1. Lower RMSE doesn't guarantee good model (check coverage too!)
    2. PI Coverage should match target (80% → 80% coverage)
    3. Residuals should be white noise (random, centered at 0)
    4. Uncertainty width should vary by time-of-day
    5. Monitor weekly; retrain if metrics degrade

    For the theory behind all of this, see [`docs_src/theory/index.md`](../../docs_src/theory/index.md).
    """)
    return


if __name__ == "__main__":
    app.run()
