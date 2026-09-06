# To build

**1. `ts_model_diagnostics.py`**
```
Create a Marimo notebook that analyzes residuals and model behavior across SARIMA, 
Exponential Smoothing, and LightGBM. For each model type, plot residual distributions, 
ACF/PACF of residuals, run ADF stationarity tests, and flag violations (e.g., 
non-stationary residuals, high autocorrelation). Use the framework's ModelEvaluator 
and actual metering data. Show which models have clean residuals and which need 
investigation. Include diagnostic checklists: what each plot tells you and when to 
worry.

Use the following for inspiration on structure: C:\Users\RhysL\Desktop\Time Series Project\src\notebooks\ts_model_explorer.py

If necessary add additional helper functions to the framework to support residual analysis and diagnostics in C:\Users\RhysL\Desktop\Time Series Project\src poissibly in a new script. These will be shared arcoss the repo.

on in  C:\Users\RhysL\Desktop\Time Series Project\src\ts_model_framework.py


```

**2. `ts_model_tuning.py`**
```
Create a Marimo notebook demonstrating hyperparameter tuning workflows using ModelTuner 
across all three model types. For SARIMA, show (p,d,q)(P,D,Q,s) sensitivity via grid search. 
For Exp Smoothing, vary smoothing coefficients. For LightGBM, adjust learning_rate, 
num_leaves, and regularization. Plot validation curves showing metric changes as parameters 
vary. Include walk-forward validation results. Show how to interpret tuning output and 
when you've hit diminishing returns.
```

**3. `ts_forecast_interpretation.py`**
```
Create a Marimo notebook showing what ForecastOutput and EvaluationMetrics reveal about 
model quality and forecast behavior. Visualize point forecasts with prediction intervals, 
highlight confidence degradation over horizon, flag anomalies (e.g., sharp trend breaks, 
seasonal pattern loss). Use MAPE/RMSE/PI coverage to diagnose forecast failure modes. 
Show examples: "This model drifts after day 3—why?" "PI coverage is poor—is the interval 
too narrow?" Tie diagnostics to actionable decisions (retrain, adjust parameters, reduce 
trust weight).
```

**4. `ts_lightgbm_feature_impact.py`**
```
Create a Marimo notebook comparing LightGBM performance with and without feature 
engineering (FE). Train both versions on the same asset(s), plot metrics side-by-side 
(MAPE, RMSE, PI coverage), show feature importance rankings from the FE model, and 
quantify the improvement delta. Include cost/benefit analysis: FE complexity vs. metric 
gain. Show which asset types or forecast horizons benefit most from FE. Visualize 
example forecasts from both versions to see the difference in practice.
```
