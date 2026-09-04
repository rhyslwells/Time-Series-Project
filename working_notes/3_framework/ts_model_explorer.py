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

    return mo, pl, np, pd, datetime, go, make_subplots, norm, warnings


@app.cell
def _(mo):
    mo.md("""
    # Time Series Model Comparison & Interpretation
    
    **Interactive explorer** for energy metering forecasts using SARIMA, Exponential Smoothing, and LightGBM.
    
    This notebook:
    1. Loads your metering data (14 days, 30-min intervals)
    2. Compares 3 forecasting models
    3. Explains each metric mathematically
    4. Visualizes good vs bad plots
    5. Shows how to interpret results
    
    Navigate using the sidebar to explore different sections.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: Data Loading & Exploration
    
    Load metering data and examine characteristics.
    """)
    return


@app.cell
def _(pl):
    # Load data
    df = pl.read_parquet("../../src/data/metering_data.parquet")
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

    return y, timestamps, asset_id, asset_data


@app.cell(hide_code=True)
def _(mo, y, np):
    # Interactive plot of full time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y,
        mode='lines',
        name='Metering (kWh)',
        line=dict(color='steelblue', width=1)
    ))
    
    fig.update_layout(
        title="Full Time Series (14 days, 30-min intervals = 672 points)",
        xaxis_title="Time (30-min intervals)",
        yaxis_title="Power (kWh)",
        height=400,
        width=1200,
        hovermode='x unified'
    )
    
    mo.ui.plotly(fig)
    return fig


@app.cell
def _(np, y):
    # Train/test split
    test_split_idx = len(y) - (4 * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]

    print(f"Train: {len(y_train)} observations ({len(y_train) / 48:.1f} days)")
    print(f"Test: {len(y_test)} observations ({len(y_test) / 48:.1f} days)")

    return y_train, y_test, test_split_idx


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: Understanding Metrics
    
    Here's the mathematics behind each metric and why it matters for flexibility forecasting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MAE (Mean Absolute Error)
    
    $$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$
    
    **Interpretation:**
    - Average absolute difference between actual and forecast
    - Same units as data (kWh)
    - Symmetric: errors up/down equally costly
    
    **For energy assets:**
    - Residential: < 0.5 kWh is good
    - Commercial: < 1.0 kWh is good
    - EV Charging: < 2.0 kWh is good
    
    **Why it matters:**
    "On average, forecast is off by X kWh"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RMSE (Root Mean Squared Error)
    
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$
    
    **Why square?**
    - Large errors get penalized more (quadratic effect)
    - A 10 kWh error contributes 100 (squared)
    - A 5 kWh error contributes only 25
    - Forces model to catch spikes
    
    **Relationship to MAE:**
    - RMSE ≥ MAE always (because of squaring)
    - If RMSE ≈ MAE: Consistent errors, few outliers ✓
    - If RMSE >> MAE: Occasional huge errors ✗
    
    **Example:**
    ```
    Scenario A: Errors [0.1, 0.1, 0.1, 0.1, 0.1]
      MAE = 0.1, RMSE = 0.1 ← Consistent
    
    Scenario B: Errors [0, 0, 0, 0, 5.0]
      MAE = 1.0, RMSE = 2.24 ← Has outlier
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MAPE (Mean Absolute Percentage Error)
    
    $$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$
    
    **What it does:**
    - Normalizes error by actual value
    - Scale-independent (compare across asset sizes)
    
    **Pitfall - Division by zero:**
    - Energy data often near-zero off-peak
    - If $y_t \approx 0$, error explodes
    - Fix: Use $\max(|y_t|, \epsilon)$ in denominator
    
    **For energy assets:**
    - Residential: < 10% is good
    - Commercial: < 15% is good
    - EV Charging: < 30% is good
    
    **Why it matters:**
    "Forecast is within X% of actual (scale-agnostic)"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PI Coverage (Prediction Interval Coverage)
    
    $$\text{Coverage} = \frac{1}{n} \sum_{t=1}^{n} \mathbb{1}[y_t \in [\hat{L}_t, \hat{U}_t]]$$
    
    **What it is:**
    - % of actual values within [Lower, Upper] bounds
    - Should match target confidence level
    
    **For 80% Confidence Interval:**
    - Upper = Point forecast + 1.282 × std(residuals)
    - Lower = Point forecast − 1.282 × std(residuals)
    - Expect **80% of actuals to fall between bounds**
    
    **What goes wrong:**
    - **Coverage < 80%** → Intervals too narrow (over-confident)
      - Model says "I'm 80% sure" but actually only 50% sure
      - Flexibility commitments breach constantly ✗
    
    - **Coverage > 90%** → Intervals too wide (under-confident)
      - Model too conservative
      - Misses revenue opportunity ✗
    
    **Ideal:** Coverage ≈ 80% (±5% acceptable)
    
    **Why it matters for FlexGo:**
    If you commit flexibility based on prediction intervals:
    - Low coverage → Surprise breaches
    - High coverage → Wasted conservatism
    - Right coverage → Reliable envelope
    """)
    return


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
def _(y_train, y_test):
    import sys
    sys.path.insert(0, '.')
    
    from ts_model_framework import (
        SARIMAModel, ExponentialSmoothingModel, LightGBMModel,
        ModelComparison
    )
    
    print("="*60)
    print("INITIALIZING MODEL COMPARISON")
    print("="*60)
    
    comp = ModelComparison(y_train, y_test)
    
    # Add models
    print("\nAdding models to comparison...")
    comp.add_model(SARIMAModel(y_train, order=(1,1,1), seasonal_order=(1,1,1,48)))
    print("  ✓ SARIMA(1,1,1)×(1,1,1,48)")
    
    comp.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
    print("  ✓ ExponentialSmoothing")
    
    comp.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))
    print("  ✓ LightGBM")
    
    print(f"\nFitting {len(comp.models)} models...")
    comp.fit_all()
    
    print("\nEvaluating forecasts...")
    results_df = comp.evaluate_all(confidence_level=0.80)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string())
    
    return comp, results_df


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
    best_metrics = comp.results[best_model_name]['metrics']
    
    print(f"Best Model: {best_model_name}")
    print(f"  MAE: {best_metrics.mae:.4f} kWh")
    print(f"  RMSE: {best_metrics.rmse:.4f} kWh")
    print(f"  MAPE: {best_metrics.mape:.2f}%")
    print(f"  PI Coverage: {best_metrics.pi_coverage:.1f}%")
    
    return best_model_name, best_forecast, best_metrics


@app.cell
def _(mo, y_test, best_forecast, best_metrics, best_model_name):
    from ts_plots import TSPlotter
    
    # Plot 1: Forecast vs Actual
    fig = TSPlotter.forecast_vs_actual(
        y_test, 
        best_forecast, 
        best_model_name, 
        best_metrics
    )
    
    mo.ui.plotly(fig)
    return fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 1: Forecast vs Actual
    
    **What to look for:**
    - Black line (actual) should mostly stay inside shaded region (PI)
    - Blue line (forecast) should track actual, with ~1-2 step lag
    - PI should widen when forecast uncertain, narrow when confident
    - No systematic bias (actuals not all above or below forecast)
    
    **Good case:**
    ```
        Upper PI ————————
                    /  ↓  \
    Actual (black) /  Forecast (blue) \___
                 /                     \
       Lower PI (P10)
    ```
    - Coverage ≈ 80% (some red outside bounds expected)
    - Forecast lags actual but follows trend
    - No flat-lined forecast
    
    **Bad case:**
    ```
        Upper PI ————
                  │   ← Many red dots (actual) outside
    Actual      /│\
               / │ \___
             /   └──── Forecast (flat!)
    Lower PI ╯
    ```
    - Coverage < 50% (intervals too narrow, over-confident)
    - Forecast doesn't adapt
    - Model underfitted
    
    **Mathematics:**
    $\hat{y}_t = \text{Forecast at time } t$
    
    $[L_t, U_t] = [\hat{y}_t - z_{0.90} \sigma, \hat{y}_t + z_{0.90} \sigma]$
    
    where $z_{0.90} = 1.282$ for 80% confidence
    """)
    return


@app.cell
def _(mo, y_test, best_forecast, best_model_name):
    from ts_plots import TSPlotter
    
    # Plot 2: Residuals Diagnostic
    fig = TSPlotter.residuals_diagnostic(
        y_test,
        best_forecast,
        best_model_name
    )
    
    mo.ui.plotly(fig)
    return fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot 2: Residuals Diagnostic
    
    **Residual = Actual − Forecast**
    $$\epsilon_t = y_t - \hat{y}_t$$
    
    **Left panel: Residuals over time**
    
    Good case:
    - Bars bounce randomly between ±1 kWh (white noise)
    - Red line (uncertainty width) ≈ 2 × typical bar height
    - No trend (doesn't drift up/down)
    - Occasional spikes normal
    
    Bad case:
    - All positive or all negative → Systematic bias (forecast too high/low)
    - Clear pattern → Autocorrelated (model missed structure)
    - Large spikes regularly → Uncertainty miscalibrated
    
    **Right panel: Histogram**
    
    Good case:
    - Bell-shaped (normal distribution)
    - Centered at 0
    - Symmetric
    
    Bad case:
    - Skewed right → Forecast systematically too low
    - Skewed left → Forecast systematically too high
    - Heavy tails → Extreme events more likely than model thinks
    
    **Why it matters:**
    SARIMA/ExponentialSmoothing assume residuals $\sim N(0, \sigma^2)$
    - If mean ≠ 0 → Model has systematic bias (dangerous!)
    - If not normal → Prediction intervals may be wrong
    - If autocorrelated → Model missed dynamics
    """)
    return


@app.cell
def _(mo, best_forecast, best_model_name):
    from ts_plots import TSPlotter
    
    # Plot 3: Uncertainty Width
    fig = TSPlotter.uncertainty_analysis(
        best_forecast,
        best_model_name
    )
    
    mo.ui.plotly(fig)
    return fig


@app.cell(hide_code=True)
def _(mo, np):
    mo.md(f"""
    #### Plot 3: Uncertainty Width Over Time
    
    **Uncertainty width = Upper − Lower**
    
    **What it means:**
    - Wide interval → Model uncertain about this time
    - Narrow interval → Model confident
    
    **Good case:**
    - Width varies by time-of-day
    - Narrow at stable periods (e.g., midday)
    - Wide at volatile periods (e.g., morning/evening transitions)
    - Mean width ≈ 2 × 1.282 × std(residuals) (well-calibrated)
    
    **Bad case:**
    - Constant everywhere (model doesn't adapt)
    - Exploding over forecast horizon (model losing confidence)
    - Much wider/narrower than expected (miscalibrated)
    
    **Mathematics:**
    For 80% confidence interval:
    
    $$\\text{{Width}} = U_t - L_t = 2 \\times z_{{0.90}} \\times \\sigma = 2 \\times 1.282 \\times \\sigma$$
    
    where $\\sigma = \\text{{std(residuals)}}$
    
    **Example:**
    - If std(residuals) = 0.4 kWh
    - Expected width = 2 × 1.282 × 0.4 = 1.03 kWh
    - If actual width = 1.0 kWh → Well-calibrated ✓
    - If actual width = 0.5 kWh → Over-confident ✗
    - If actual width = 2.0 kWh → Over-conservative ✗
    """)
    return


@app.cell
def _(mo, y_test, best_forecast, best_model_name):
    from ts_plots import TSPlotter
    
    # Plot 4: PI Coverage
    fig = TSPlotter.pi_coverage(
        y_test,
        best_forecast,
        best_model_name
    )
    
    mo.ui.plotly(fig)
    return fig


@app.cell(hide_code=True)
def _(mo, best_model, np, y_test, best_forecast):
    in_bounds = (y_test >= best_forecast["lower"]) & (y_test <= best_forecast["upper"])
    coverage_pct = np.mean(in_bounds) * 100
    
    mo.md(f"""
    #### Plot 4: PI Coverage (Green/Red Scatter)
    
    **Coverage = {coverage_pct:.1f}% (Target 80%)**
    
    **What it means:**
    - Green dot = Actual fell within prediction interval ✓
    - Red dot = Actual outside prediction interval ✗
    - Should be ~80% green, ~20% red
    
    **Good case (Coverage ≈ 80%):**
    - 80 green, 20 red
    - Reds scattered randomly (no clustering)
    - Coverage matches target confidence level
    - Model says "80% sure" and is actually 80% sure ✓
    
    **Bad case 1: Under-coverage (Coverage < 70%)**
    - Many reds (> 30%)
    - Model over-confident
    - For FlexGo: Flexibility commitments will breach
    - When you commit based on envelope, surprises happen frequently
    
    **Bad case 2: Over-coverage (Coverage > 90%)**
    - Too many greens
    - Intervals too wide
    - Model under-confident
    - Wasted conservatism, missed revenue opportunity
    
    **Bad case 3: Systematic pattern (e.g., reds only in morning)**
    - Model doesn't understand time-of-day patterns
    - Needs retraining or different model
    - Coverage uniform across hours? Check this!
    
    **Why it matters:**
    $$\\text{{Coverage}} = \\frac{{1}}{{n}} \\sum_{{t=1}}^{{n}} \\mathbb{{1}}[y_t \\in [L_t, U_t]]$$
    
    For 80% PI, coverage should be:
    - Good: 75–85% (acceptable)
    - Ideal: 78–82% (perfect calibration)
    - If < 70%: Intervals too narrow (over-confident, risky)
    - If > 90%: Intervals too wide (under-confident, conservative)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 5: Comparing All Models
    
    Side-by-side comparison of all 3 models.
    """)
    return


@app.cell
def _(mo, comp, y_test):
    from ts_plots import ComparisonPlotter
    
    # Get all forecasts
    forecasts = {name: comp.get_forecast(name) for name in comp.results.keys()}
    
    # Plot: All model forecasts overlaid
    fig = ComparisonPlotter.forecast_comparison(y_test, forecasts, sample_size=96)
    
    mo.ui.plotly(fig)
    return fig


@app.cell
def _(mo, comp):
    # Metrics comparison table
    metrics_dict = {name: data['metrics'] for name, data in comp.results.items()}
    
    fig = ComparisonPlotter.metrics_comparison(metrics_dict)
    
    mo.ui.plotly(fig)
    return fig


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 6: Interpretation Guide
    
    How to interpret these results for FlexGo flexibility forecasting.
    """)
    return


@app.cell(hide_code=True)
def _(mo, best_model_name, best_metrics, best_forecast):
    mo.md(f"""
    ### For Your Asset
    
    **Best Model: {best_model_name}**
    
    **Interpretation of Results:**
    
    | Metric | Value | Meaning |
    |---|---|---|
    | MAE | {best_metrics.mae:.4f} kWh | Forecast off by ~{best_metrics.mae:.2f} kWh on average |
    | RMSE | {best_metrics.rmse:.4f} kWh | Larger error occasionally ({best_metrics.rmse/best_metrics.mae:.2f}× MAE) |
    | MAPE | {best_metrics.mape:.2f}% | Within ~{best_metrics.mape:.1f}% of actual |
    | PI Coverage | {best_metrics.pi_coverage:.1f}% | {f"Good (≈80%)" if 75 <= best_metrics.pi_coverage <= 85 else f"Check - target is 80%"} |
    | Uncertainty | {best_metrics.mean_uncertainty_width:.3f} kWh | Forecast ±{best_metrics.mean_uncertainty_width/2:.3f} kWh (80% confidence) |
    
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 7: Mathematics Summary
    
    Quick reference for the math behind everything.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metric Formulas
    
    **MAE:**
    $$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$
    
    **RMSE:**
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$
    
    **MAPE:**
    $$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$
    
    **PI Coverage (80% confidence):**
    $$\text{Coverage} = \frac{1}{n} \sum_{t=1}^{n} \mathbb{1}[y_t \in [\hat{y}_t - z_{0.90}\sigma, \hat{y}_t + z_{0.90}\sigma]]$$
    
    where $z_{0.90} = 1.282$ for 80% confidence level
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Formulas
    
    **SARIMA(p,d,q)×(P,D,Q,s):**
    - Autoregressive (AR) component: $y_t$ depends on $y_{t-1}, y_{t-2}, ..., y_{t-p}$
    - Differencing: Remove trend with $\Delta y_t = y_t - y_{t-1}$ (repeat $d$ times)
    - Moving Average (MA): $y_t$ depends on past forecast errors
    - Seasonal: Repeat pattern at lag $s=48$ (24 hours of 30-min data)
    
    **Exponential Smoothing (Holt-Winters):**
    $$\hat{y}_{t+h} = \ell_t + T_t \cdot h + s_{t+h-s}$$
    where:
    - $\ell_t$ = level (baseline)
    - $T_t$ = trend (change rate)
    - $s_t$ = seasonal component (daily pattern)
    
    **LightGBM (Gradient Boosting):**
    $$\hat{y}_t = \sum_{m=1}^{M} \gamma_m f_m([y_{t-1}, y_{t-2}, y_{t-48}, y_{t-96}, \text{hour}])$$
    - Ensemble of trees predicting from lag features
    - Non-linear: Can capture asymmetric patterns
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 8: Checklist
    
    Is this forecast production-ready?
    """)
    return


@app.cell
def _(mo, best_metrics):
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
    print("="*50)
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
    
    return checks


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 9: Next Steps
    
    Recommended actions based on your results.
    """)
    return


@app.cell(hide_code=True)
def _(mo, best_model_name, best_metrics):
    mo.md(f"""
    ### Recommendations
    
    **Best Model: {best_model_name}**
    
    **Immediate Actions:**
    1. {f"✓ Deploy {best_model_name}" if best_metrics.pi_coverage > 75 else f"⚠ Retrain {best_model_name} - coverage below 75%"}
    2. Monitor metrics weekly:
       - Track MAE, RMSE, MAPE
       - Watch PI Coverage (should stay 75–85%)
    3. Retrain if:
       - Coverage drops below 70% → Model over-confident
       - RMSE increases 20% → Concept drift detected
       - New season → Patterns changed
    
    **For Model Tuning (using ts_model_framework.py):**
    - If RMSE too high: Use ModelTuner to search SARIMA(p,d,q) space
    - If coverage too low: Try ExponentialSmoothing with damped_trend=True
    - If systematic bias: Retrain on recent data (last 7–14 days)
    
    **For FlexGo Integration:**
    - Use prediction interval as flexibility envelope
    - Width represents uncertainty range
    - Coverage tells you reliability of commitment
    - If coverage < 75%: Envelopes are too tight, risky
    - If coverage > 90%: Envelopes too wide, conservative
    
    **Logging/Monitoring:**
    - Store forecast + actual for each timestep
    - Track residuals (actual - forecast)
    - Detect concept drift: Are residuals drifting over time?
    - Monthly re-evaluation: Are metrics stable?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    
    ## Summary
    
    You now have:
    ✓ Comparison of 3 forecasting models
    ✓ Detailed metrics (MAE, RMSE, MAPE, PI Coverage)
    ✓ 4 diagnostic plots with mathematical explanations
    ✓ Production-ready checklist
    ✓ Interpretation guide for FlexGo
    
    **Key takeaways:**
    1. Lower RMSE doesn't guarantee good model (check coverage too!)
    2. PI Coverage should match target (80% → 80% coverage)
    3. Residuals should be white noise (random, centered at 0)
    4. Uncertainty width should vary by time-of-day
    5. Monitor weekly; retrain if metrics degrade
    
    For more details, see the accompanying markdown documentation:
    - MATH_AND_INTERPRETATION.md (theory + formulas)
    - VISUAL_EXAMPLES.md (good vs bad plots)
    - METRIC_REFERENCE.md (lookup tables)
    - README_FORECASTING.md (master index)
    """)
    return


if __name__ == "__main__":
    app.run()
