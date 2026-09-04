# Time Series Model Explorer — Marimo Interactive Notebook

## What Is This?

`ts_model_explorer.py` is an **interactive Marimo notebook** that:

1. **Loads your energy metering data** (14 days, 30-min intervals)
2. **Compares 3 forecasting models** side-by-side:
   - SARIMA(1,1,1)×(1,1,1,48)
   - Exponential Smoothing (Holt-Winters)
   - LightGBM (gradient boosting)
3. **Explains every metric mathematically** with formulas + intuition
4. **Visualizes 4 diagnostic plots** with integrated explanations
5. **Provides interpretation guidance** for FlexGo flexibility forecasting
6. **Includes production-ready checklist**

## How to Run

### Prerequisites
```bash
pip install marimo polars numpy pandas plotly scipy statsmodels scikit-learn lightgbm
```

### Launch Notebook
```bash
marimo run ts_model_explorer.py
```

Or if using VS Code with marimo extension:
```bash
marimo edit ts_model_explorer.py
```

## Notebook Structure

### Section 1: Data Loading
- Loads metering data from parquet file
- Shows summary statistics
- Visualizes full 14-day time series
- Splits train (10 days) / test (4 days)

### Section 2: Understanding Metrics
Each metric explained with:
- **Formula** (LaTeX math)
- **Interpretation** (plain language)
- **Good vs bad values** (by asset type)
- **Why it matters for FlexGo**

Metrics covered:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **PI Coverage** (Prediction Interval Coverage %)

### Section 3: Model Comparison
- Fits all 3 models
- Generates forecasts with 80% prediction intervals
- Evaluates on test set
- Ranks by RMSE, shows all metrics
- Interactive ranking table

### Section 4: Detailed Plot Analysis
**Plot 1: Forecast vs Actual**
- Black line = Actual demand
- Blue line = Model forecast
- Shaded region = 80% prediction interval
- Shows how well forecast tracks actual
- Interactive hover for values

**Plot 2: Residuals Diagnostic**
- Left panel: Residuals over time + uncertainty width
- Right panel: Histogram of residuals
- Diagnoses:
  - Systematic bias (residuals all positive/negative)
  - Autocorrelation (residuals in clusters)
  - Non-normality (histogram shape)

**Plot 3: Uncertainty Width**
- Shows prediction interval width over time
- Should vary by time-of-day (narrow when stable, wide when volatile)
- Overlaid with mean line

**Plot 4: PI Coverage**
- Green dots = Actual in bounds ✓
- Red dots = Actual outside bounds ✗
- Shows calibration of prediction intervals
- Coverage % = ratio of greens

### Section 5: Model Comparison
- All 3 forecast lines overlaid
- First 4 days of test period
- Metrics comparison table
- Visual ranking

### Section 6: Interpretation Guide
- Specific metrics for your asset
- Table: MAE/RMSE/MAPE/Coverage meanings
- Flexibility commitment interpretation
- Risk assessment (breach probability)

### Section 7: Mathematics Summary
- Quick reference for all formulas
- SARIMA model explanation
- Exponential Smoothing equations
- LightGBM ensemble approach

### Section 8: Production Checklist
- 8 criteria to evaluate
- Automatic scoring (0–8)
- Status: Production-ready / Monitor / Needs tuning

### Section 9: Next Steps
- Specific recommendations based on results
- Retraining triggers
- FlexGo integration notes
- Logging/monitoring guidelines

---

## Key Features

### ✓ Interactive
- Hover over plots for exact values
- Plots zoom, pan, download
- No context-switching (all in one notebook)

### ✓ Mathematical
Every plot includes LaTeX formulas explaining:
- What the metric measures
- Why it matters
- How it's calculated
- Good vs bad ranges

### ✓ Educational
Each section explains concepts before showing results:
- Why residuals should be white noise
- Why PI coverage matters for commitments
- How z-scores become prediction intervals
- Why RMSE > MAE indicates outliers

### ✓ Practical
- Asset-type specific thresholds (residential vs commercial vs EV charging)
- FlexGo-specific interpretation (flexibility envelope sizing)
- Production readiness checklist
- Monitoring recommendations

### ✓ Extensible
- Modify thresholds for your asset type
- Change model parameters in Section 3
- Add new plots by copying plot code
- Integrate with your forecasting pipeline

---

## Example Output

After running the notebook, you'll see:

```
Data Summary:
  Asset: ASSET_001
  Type: residential
  Records: 672
  Date range: 2025-01-01 to 2025-01-14
  Mean: 2.345 kWh
  Std: 1.234 kWh
  Min: 0.001 kWh
  Max: 8.921 kWh

...

============================================================
MODEL COMPARISON RESULTS
============================================================
             MAE    RMSE  MAPE  PI Coverage %  Uncertainty Width  Rank
SARIMA       0.35   0.52  9.0   80.2           1.6                1.0
ExpSmooth    0.40   0.58  11.0  76.1           1.4                2.0
LightGBM     0.32   0.48  8.5   72.3           1.5                3.0

Production Readiness Checklist:
✓ RMSE < 0.7 kWh (residential)
✓ MAE < 0.5 kWh (residential)
✓ MAPE < 15%
✓ PI Coverage 75–85%
✓ Coverage ≈ target (±5%)
...

Score: 6/8
✓ LIKELY PRODUCTION-READY
```

---

## Interpreting Results

### Best Model Selection

**By RMSE (lower = better):**
1. Check which model has lowest RMSE
2. Verify PI Coverage ≈ 80% (±5%)
3. If coverage off, model needs tuning or interval recalibration

**Trade-offs:**
- Better point forecast (RMSE) vs. better intervals (coverage)?
- For FlexGo: Coverage more important (reliability of commitment)

### Is This Production-Ready?

✓ All criteria pass → Deploy immediately  
⚠ 2+ criteria fail → Needs tuning before deployment  
✗ Coverage < 70% → High risk of commitment breach

---

## Common Questions

**Q: Why is RMSE > MAE?**
A: RMSE squares errors, so large mistakes get penalized more. This is by design—it forces the model to avoid huge spikes.

**Q: Why does coverage matter more than RMSE?**
A: For FlexGo, you commit flexibility based on prediction intervals. If coverage is low (e.g., 45%), your commitments will breach frequently, breaching SLAs.

**Q: What if coverage is 95% (too high)?**
A: Model is over-conservative. Intervals too wide. You're not selling enough flexibility. Retrain or use tighter bounds (90% instead of 80%).

**Q: When should I retrain?**
A: - Coverage drops < 70% (immediately!)
  - RMSE increases 20% (weekly check)
  - New season (seasonal patterns change)
  - Equipment change (asset behavior changed)

**Q: How do I improve RMSE?**
A: - Increase SARIMA complexity: Try SARIMA(2,1,1) or (1,2,1)
  - Add longer lags to LightGBM: [1, 2, 48, 96, 336]
  - Switch to different model if stuck

**Q: How do I improve coverage?**
A: - If coverage too low: Widen intervals (multiply by 1.2)
  - If coverage too high: Narrow intervals or retrain
  - For SARIMA/ExpSmoothing: Use damped_trend=True
  - Consider quantile regression (predict P10/P90 separately)

---

## Integration with Your Workflow

### Step 1: Explore with Notebook
```bash
marimo run ts_model_explorer.py
# Understand metrics and plots
# Pick best model
# Check production readiness
```

### Step 2: Use Code Framework
```python
# Use ts_model_framework.py for non-interactive use
from ts_model_framework import ModelComparison, SARIMAModel

comp = ModelComparison(y_train, y_test)
comp.add_model(SARIMAModel(y_train))
comp.fit_all()
ranking = comp.evaluate_all()
```

### Step 3: Reference Docs
- Unsure what metric means? → MATH_AND_INTERPRETATION.md
- Need to interpret plot? → VISUAL_EXAMPLES.md
- Quick lookup? → METRIC_REFERENCE.md

### Step 4: Monitor in Production
```python
# Weekly: Track metrics
mae, rmse, coverage = evaluate(y_test, forecast)

# If coverage drops: Retrain
if coverage < 70:
    retrain_model()

# Log residuals to detect drift
log_residuals(y_test - forecast)
```

---

## Customization

### Change Asset Type
In Section 6, update thresholds:
```python
# For commercial instead of residential:
if best_metrics["RMSE"] < 1.3:  # Commercial threshold
    print("Production-ready (commercial)")
```

### Add More Models
In Section 3, add another model:
```python
# Add Prophet or LSTM
from prophet import Prophet

prophet_model = Prophet()
# ... fit and predict
```

### Modify Plot Parameters
```python
# Change prediction interval to 90% instead of 80%
# In forecast generation:
z_score = norm.ppf(0.95)  # 90% confidence
upper = forecast + z_score * std_residuals
```

### Export Results
```python
# Save forecast output
forecast_df.to_csv("forecast_output.csv")

# Save model
import pickle

pickle.dump(best_model, open("model.pkl", "wb"))
```

---

## Troubleshooting

### "Data file not found"
Make sure `../../src/data/metering_data.parquet` exists relative to notebook location.

### "LightGBM not installed"
```bash
pip install lightgbm
```

### "Module not found errors"
Install all dependencies:
```bash
pip install marimo polars numpy pandas plotly scipy statsmodels scikit-learn lightgbm
```

### Plots not showing
Use VS Code Marimo extension or run:
```bash
marimo run ts_model_explorer.py --host localhost --port 8080
```

### Model fit fails
Check for:
- Duplicate timestamps
- Missing values (NaN)
- Very short data (< 48 points)
- Data quality issues (all zeros, constant values)

---

## Next Steps

1. **Run notebook:** `marimo run ts_model_explorer.py`
2. **Understand results:** Read Section 6 interpretation
3. **Check production readiness:** Review checklist in Section 8
4. **Use framework code:** Switch to `ts_model_framework.py` for automation
5. **Monitor weekly:** Track MAE/RMSE/coverage over time

---

## Files Reference

| File | Purpose |
|---|---|
| `ts_model_explorer.py` | This interactive notebook |
| `ts_model_framework.py` | Core models + tuning (for scripts) |
| `ts_plots.py` | Plotting module (for scripts) |
| `MATH_AND_INTERPRETATION.md` | Theory + formulas |
| `VISUAL_EXAMPLES.md` | Good vs bad plots |
| `METRIC_REFERENCE.md` | Lookup tables |
| `README_FORECASTING.md` | Master index |

---

**Version:** 1.0  
**Created:** January 2025  
**For:**  FlexGo  
**Data:** 14 days metering at 30-min intervals
