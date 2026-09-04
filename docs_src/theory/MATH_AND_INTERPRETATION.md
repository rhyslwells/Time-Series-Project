# Time Series Forecasting: Mathematics, Plots, and Interpretation

## Part 1: Metrics (What We're Optimizing)

### Mean Absolute Error (MAE)

**Formula:**
$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

**What it is:**
- Average absolute difference between actual and predicted values
- In same units as data (kWh)

**Interpretation:**
- **Good:** MAE ≈ 0.5 kWh (forecast off by ~0.5 kWh on average)
- **Bad:** MAE > 2 kWh (forecast systematically overestimates/underestimates by ~2 kWh)
- **Scale-dependent:** For 5 kWh assets, 0.1 kWh is excellent; for 0.1 kWh assets, it's terrible

**When to use:**
- Portfolio-level decisions (sum errors)
- Symmetric loss (errors up/down equally costly)

---

### Root Mean Squared Error (RMSE)

**Formula:**
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

**What it is:**
- Penalizes large errors more than MAE (because of squaring)
- Also in same units as data

**Why square?**
- A single 10 kWh error contributes 100 to the sum (squared)
- A 5 kWh error contributes only 25 (squared)
- Large outliers dominate optimization → forces model to catch spikes

**Interpretation:**
- **RMSE > MAE always** (because of squaring effect)
- **Good gap:** If MAE=0.5, RMSE≈0.7, model is consistent (no huge outliers)
- **Bad gap:** If MAE=0.5, RMSE=2.0, model occasionally makes large errors (inconsistent)

**When to use:**
- Default metric for time series
- When you care about rare but large errors (e.g., avoiding capacity violations)

**Example:**
```
Scenario A:
  Errors: [0.1, 0.1, 0.1, 0.1, 0.1]
  MAE = 0.1, RMSE = 0.1 ← Consistent, good

Scenario B:
  Errors: [0, 0, 0, 0, 5.0]
  MAE = 1.0, RMSE = 2.24 ← Has outlier, RMSE inflated
```

---

### Mean Absolute Percentage Error (MAPE)

**Formula:**
$$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

**What it is:**
- Percentage error, normalized by actual value
- Scale-independent (compare across assets of different sizes)

**Pitfall - Division by zero:**
- If $y_t \approx 0$, error explodes to infinity
- Common in energy data (off-peak periods have low/zero metering)
- Fix: Use $\max(|y_t|, \epsilon)$ in denominator, or exclude near-zero values

**Interpretation:**
- **Good:** MAPE < 10% (forecasts within 10% of actual)
- **Acceptable:** MAPE 10–20% (off by ~15%)
- **Poor:** MAPE > 30%

**When to use:**
- Comparing models across different assets/timescales
- Portfolio-level assessment (scale-agnostic)

**Example for energy:**
```
Off-peak (actual = 0.1 kWh):
  Forecast = 0.2 kWh
  Error % = |0.2 - 0.1| / 0.1 = 100% ← Misleading!
  
Peak (actual = 5 kWh):
  Forecast = 5.5 kWh
  Error % = |5.5 - 5| / 5 = 10% ← Good
```

---

### Prediction Interval (PI) Coverage

**Formula:**
$$\text{Coverage} = \frac{1}{n} \sum_{t=1}^{n} \mathbb{1}[y_t \in [\hat{L}_t, \hat{U}_t]]$$

where $\hat{L}_t$ = lower bound (P10), $\hat{U}_t$ = upper bound (P90)

**What it is:**
- Percentage of actual values that fall *within* the prediction interval
- If true, target confidence level = coverage %

**Theory:**
For 80% PI (P10 to P90):
- Upper = point forecast + 1.282 × std(residuals)
- Lower = point forecast − 1.282 × std(residuals)
- Expect **80% of actuals to fall between bounds**

**Interpretation:**
- **Ideal:** Coverage ≈ 80% (matches target confidence level)
- **Undercoverage (< 80%):** Intervals too narrow → model over-confident
  - Example: Coverage = 60% means 40% of actuals are outside predicted range
  - Consequence: Surprise failures (asset behaves outside predicted envelope)
- **Overcoverage (> 95%):** Intervals too wide → model under-confident
  - Example: Coverage = 95% means intervals are unnecessarily wide
  - Consequence: Wasted conservatism in planning

**Good case:**
```
Target 80% PI:
  Time 1: Actual = 3.5 kWh, Lower = 2.0, Upper = 5.0  ✓ In bounds
  Time 2: Actual = 1.8 kWh, Lower = 1.5, Upper = 4.2  ✓ In bounds
  Time 3: Actual = 6.2 kWh, Lower = 2.0, Upper = 5.0  ✗ Out of bounds (20% expected to be)
  
  Coverage = 2/3 ≈ 67% → Acceptable (close to 80% on small sample)
```

---

## Part 2: The Models

### SARIMA: Seasonal ARIMA

**Full name:** $\text{SARIMA}(p,d,q) \times (P,D,Q,s)$

**Parameters:**
- $(p,d,q)$ — Non-seasonal AR, differencing, MA
- $(P,D,Q,s)$ — Seasonal AR, differencing, MA, seasonality

**For 30-min energy data: SARIMA(1,1,1)×(1,1,1,48)**

### What each does:

#### Differencing: $d=1, D=1$
Makes series stationary by removing trend and seasonality.

**Original series (non-stationary):**
```
Hour 1: 1.0 kWh (morning ramp-up)
Hour 2: 2.0 kWh
Hour 3: 3.5 kWh
...
Hour 25: 1.2 kWh (next day morning, repeats)
```

**After non-seasonal differencing (d=1):**
```
$\Delta y_t = y_t - y_{t-1}$
Hour 2: 2.0 - 1.0 = 1.0
Hour 3: 3.5 - 2.0 = 1.5
```

**After seasonal differencing (D=1, s=48):**
```
$\Delta_s \Delta y_t = \Delta y_t - \Delta y_{t-48}$
Removes 24-hour patterns
```

**Why:** ARIMA requires stationarity (mean doesn't drift over time). Differencing achieves this.

#### AR (Autoregressive): $p=1, P=1$
Current value depends on past value(s).

$$y_t = \phi_0 + \phi_1 y_{t-1} + \phi_s y_{t-s} + \epsilon_t$$

**In words:**
- Today's metering ≈ constant + 0.8×(yesterday at this time) + noise
- Captures momentum: if asset was using power yesterday, likely using power today

**Expected coefficient:** $\phi_1 \approx 0.7$–$0.9$ (strong persistence in energy usage)

#### MA (Moving Average): $q=1, Q=1$
Current value depends on past forecast errors.

$$y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_s \epsilon_{t-s}$$

**In words:**
- If forecast was too high yesterday, adjust today's forecast downward
- Smooths out shocks quickly

**Expected coefficient:** $\theta_1 \approx 0.2$–$0.5$ (smaller than AR)

---

### Exponential Smoothing (Holt-Winters)

**Formula (additive):**
$$\hat{y}_{t+h} = \ell_t + T_t \cdot h + s_{t+h-s}$$

where:
- $\ell_t$ = level (baseline demand)
- $T_t$ = trend (how fast demand is changing)
- $s_t$ = seasonal component (daily pattern)

**Update equations:**
$$\ell_t = \alpha(y_t - s_{t-s}) + (1-\alpha)(\ell_{t-1} + T_{t-1})$$
$$T_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)T_{t-1}$$
$$s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-s}$$

**Parameters:**
- $\alpha \in [0,1]$ — Level smoothing (higher = react faster to changes)
- $\beta \in [0,1]$ — Trend smoothing
- $\gamma \in [0,1]$ — Seasonal smoothing

**Interpretation:**
- $\alpha = 0.1$ (smooth, stable) vs $\alpha = 0.9$ (noisy, responsive)
- Multiplicative vs additive:
  - **Additive:** Seasonality constant (e.g., +0.5 kWh every afternoon)
  - **Multiplicative:** Seasonality scales with level (e.g., 1.5x demand every afternoon)

**Good for energy:**
- Additive usually works (seasonal pattern is stable kWh offset)
- Damped trend useful if trends flatten (e.g., gradual efficiency improvements)

---

### LightGBM (Gradient Boosting)

**Core idea:** Build ensemble of weak decision trees, each correcting prior tree's mistakes.

**For time series:**
1. Create lag features: [y_{t-1}, y_{t-2}, y_{t-48}, y_{t-96}]
2. Add seasonal feature: hour-of-day ∈ [0, 1]
3. Train tree to predict y_t from these features
4. Recursively apply trained model to forecast horizon

**Update:**
$$\hat{y}_t = \sum_{m=1}^{M} \gamma_m f_m(\mathbf{x}_t)$$

where $f_m$ = weak learner (tree), $\gamma_m$ = step size (learning_rate)

**Hyperparameters:**
- `num_leaves` — Tree complexity (15 = conservative, 63 = complex)
- `learning_rate` — Step size (0.05 = moderate)
- `lags` — Which past values to use as features

**Good for:**
- Non-linear patterns (e.g., asymmetric ramp-up vs ramp-down)
- Multiple lag scales (1-hour, daily, weekly patterns)
- Portfolio-level (feed in multiple asset features)

**Pitfall:**
- Can overfit to training data (requires validation)
- Needs careful lag selection (garbage in = garbage out)

---

## Part 3: Understanding the Plots

### Plot 1: Forecast vs Actual (with PI)

**What it shows:**
```
        Upper PI (P90) ————————
                    /        \
    Actual (black) /  ↓       \___
                 /  Forecast   \
       Lower PI (P10) ___________\
                         /
```

**Good case:**
- Black line (actual) stays mostly inside shaded region (PI)
- Blue line (forecast) tracks actual but lags by ~1-2 time steps
- Forecast oscillates smoothly, not jumpy
- PI widens/narrows with model confidence

**Bad case:**
- Actual frequently spikes outside PI (undercoverage)
- Forecast completely misses events (e.g., sudden demand spikes)
- Forecast flat-lines (model gave up)
- PI always same width (model doesn't understand uncertainty)

**Example — Good SARIMA:**
```
Time     Actual  Forecast  Lower  Upper  In?
1        2.5     2.3       1.2    3.4    ✓
2        3.1     2.6       1.3    3.9    ✓
3        4.2     3.2       1.9    4.5    ✓
4        3.8     3.9       2.6    5.2    ✓
5        2.2     3.5       2.2    4.8    ✓
```

Coverage = 5/5 = 100% ✓ (slightly overconfident but acceptable)

**Example — Bad ExponentialSmoothing:**
```
Time     Actual  Forecast  Lower  Upper  In?
1        2.5     2.5       2.3    2.7    ✓
2        3.1     2.6       2.4    2.8    ✗ (actual jumps outside narrow PI)
3        4.2     2.7       2.5    2.9    ✗
4        3.8     2.8       2.6    3.0    ✗
5        2.2     2.9       2.7    3.1    ✗
```

Coverage = 1/5 = 20% ✗ (severely undercovered, model is over-confident)

---

### Plot 2: Residuals Diagnostic (Bar + Histogram)

**Residuals = Actual − Forecast**
$$\epsilon_t = y_t - \hat{y}_t$$

**Left panel: Residuals over time**
- Red line = uncertainty width (what model thinks is reasonable error)
- Blue bars = actual errors

**Good case:**
```
Residuals bounce randomly between ±1 kWh (white noise)
Uncertainty width ≈ 2 × typical |residuals| (model calibrated)
No trend in residuals (forecast not systematically biased)
```

**Bad case — Systematic bias:**
```
All residuals positive → forecast too low (systematic underestimation)
Model doesn't know it's wrong
```

**Bad case — Autocorrelated residuals:**
```
Residuals: +0.5, +0.4, +0.3, -0.1, -0.2, -0.3, ...
Pattern indicates model missed structure (e.g., slow drift)
Solution: Increase differencing (d or D)
```

**Right panel: Histogram**
- Should be bell-shaped (normal distribution) centered at 0
- Skewed left? Systematic overforecasting
- Skewed right? Systematic underforecasting
- Tall outlier bars? Model occasionally fails badly

**Mathematics:** SARIMA assumes residuals $\sim N(0, \sigma^2)$
- Mean should be ~0
- Std dev should match uncertainty width

---

### Plot 3: Uncertainty Width Over Time

**Uncertainty width = Upper − Lower**

**What it means:**
- Wide interval = model unsure about this prediction
- Narrow interval = model confident

**Good case:**
```
Hour 1 (morning): Width = 1.5 kWh (uncertain, complex patterns)
Hour 6 (noon):    Width = 0.8 kWh (confident, stable demand)
Hour 8 (evening): Width = 1.2 kWh (moderately uncertain)
Mean width ≈ 1.0 kWh (consistent)
```

Why narrower at noon?
- Noon demand is stable (everyone at work, A/C running)
- Morning/evening have variable behavior (people arriving/leaving)

**Bad case:**
```
Width is constant everywhere (0.5 kWh) — model doesn't adapt
Model learned fixed PI, not data-driven
Predictions at times of high volatility equally narrow as stable times
```

**Information value:**
- If width = 2 × residual std dev, model is well-calibrated
- Use width to set alerts: "Width > 3.0 means something unusual"

---

### Plot 4: PI Coverage

**Scatter plot with color coding:**
- Green dot = actual value fell within PI (correct prediction)
- Red dot = actual value outside PI (model surprised)

**Good case:**
```
80% green, 20% red → Coverage = 80%
Reds appear random (not clustered)
No systematic pattern
```

**Bad case:**
```
50% red, 50% green → Coverage = 50%
Model is overconfident
When model says "I'm 80% sure," it's actually only 50% sure

All reds in morning hours → Model doesn't understand morning variability
All reds on high-demand days → Model struggles with peak days
```

**Why it matters for FlexGo:**
```
Suppose you set flexibility envelope based on P10-P90:
- If coverage < 80%, actual demand will spike outside envelope frequently
  → Assets fail to meet commitments
- If coverage > 95%, envelope is unnecessarily wide
  → Miss revenue opportunity (could sell more flexibility)
  
Ideal: Coverage ≈ 80% for 80% PI
```

---

## Part 4: Model Outputs and What They Mean

### Standard Output Contract: ForecastOutput

```python
@dataclass
class ForecastOutput:
    prediction: np.ndarray       # Point forecast
    lower: np.ndarray            # Lower bound (P10 or P_alpha)
    upper: np.ndarray            # Upper bound (P90 or P_{1-alpha})
    uncertainty_width: np.ndarray  # upper - lower
```

**Example output:**
```
Time  Prediction  Lower  Upper  Width
1     3.2         2.1    4.3    2.2
2     3.5         2.2    4.8    2.6
3     3.1         1.9    4.3    2.4
4     2.8         1.7    4.0    2.3
```

**How to interpret:**
- **Prediction:** "I expect 3.2 kWh"
- **Lower/Upper:** "I'm 80% sure it will be between 2.1 and 4.3 kWh"
- **Width:** "My uncertainty is ±1.1 kWh"

**Using in decisions:**
```python
# Conservative planning (assume worst-case)
available_flexibility = prediction - lower  # 3.2 - 2.1 = 1.1 kWh upside

# Aggressive planning (assume best-case)
available_flexibility = upper - prediction  # 4.3 - 3.2 = 1.1 kWh downside

# Symmetric envelope
flexibility_envelope = (upper - lower) / 2  # ±1.1 kWh around forecast
```

---

### Evaluation Metrics Output: EvaluationMetrics

```python
@dataclass
class EvaluationMetrics:
    mae: float                      # Mean Absolute Error
    rmse: float                     # Root Mean Squared Error
    mape: float                     # Mean Absolute Percentage Error
    pi_coverage: float              # % of actuals in [lower, upper]
    mean_uncertainty_width: float   # Average interval width
```

**Example output:**
```
MAE: 0.4521 kWh
RMSE: 0.6234 kWh
MAPE: 12.34%
PI Coverage: 81.2%
Mean Uncertainty Width: 1.8 kWh
```

**Interpretation:**

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | 0.45 kWh | Forecast off by ~0.45 kWh on average |
| RMSE | 0.62 kWh | Occasional larger errors (~40% bigger than typical) |
| MAPE | 12% | Forecast within ~12% of actual (good for energy) |
| PI Coverage | 81% | Intervals calibrated well (target was 80%) |
| Uncertainty Width | 1.8 kWh | Model says forecast is uncertain by ±0.9 kWh |

**Relationship check:**
- RMSE > MAE? Yes (0.62 > 0.45) → Has outliers ✓
- Gap reasonable? 0.62/0.45 ≈ 1.38 → Moderate outliers ✓
- Coverage ≈ target? 81% ≈ 80% target ✓
- Width matches uncertainty? 1.8 ≈ 2 × std(residuals) ✓

---

## Part 5: Ranking Models — What Makes One "Best"?

### Scenario 1: SARIMA vs ExponentialSmoothing

**SARIMA results:**
```
MAE: 0.35 kWh
RMSE: 0.52 kWh
MAPE: 9%
PI Coverage: 80%
Uncertainty Width: 1.6 kWh
```

**ExponentialSmoothing results:**
```
MAE: 0.40 kWh
RMSE: 0.58 kWh
MAPE: 11%
PI Coverage: 75%
Uncertainty Width: 1.4 kWh
```

**Ranking by RMSE (framework default):**
1. SARIMA (0.52)
2. ExponentialSmoothing (0.58)

**Why SARIMA is better:**
- Lower error (RMSE 0.52 vs 0.58)
- Better calibrated (80% vs 75% coverage)
- Slightly more conservative uncertainty (1.6 vs 1.4)

**Tradeoff:** SARIMA takes longer to fit (ARIMA search harder than Holt-Winters)

---

### Scenario 2: SARIMA vs LightGBM

**SARIMA results:**
```
MAE: 0.35 kWh
RMSE: 0.52 kWh
MAPE: 9%
PI Coverage: 80%
```

**LightGBM results:**
```
MAE: 0.32 kWh
RMSE: 0.48 kWh
MAPE: 8.5%
PI Coverage: 72%
```

**Ranking by RMSE:**
1. LightGBM (0.48)
2. SARIMA (0.52)

**But wait — coverage is worse for LightGBM!**

**Decision logic:**
- If you care about **point forecast accuracy** → LightGBM wins
- If you care about **uncertainty quantification** → SARIMA wins
- If you care about **both** → Need to trade off

**Recommendation:**
```
For FlexGo flexibility forecasting:
  - Point forecast error (MAE/RMSE) determines "how much flexibility"
  - Prediction intervals (coverage) determine "confidence in that amount"
  - If coverage is bad, interval is too narrow
  - Problem: LightGBM estimates intervals from residuals
           If model systematically optimistic, residuals small
           → Predicted intervals too narrow

Solution: Retrain LightGBM with better hyperparameters
         or use ensemble (LightGBM point + SARIMA intervals)
```

---

## Part 6: Tuning Guide — When and How

### Problem 1: High RMSE (Model can't forecast well)

**Diagnosis:**
```python
mae: 1.2 kWh, rmse: 1.8 kWh
RMSE/MAE ratio ≈ 1.5 → Has outliers
```

**Root cause:**
- Model missed structure (e.g., didn't capture seasonality)
- Hyperparameters wrong (e.g., d=0 but series not stationary)
- Data quality issue (e.g., sensor outliers)

**Fix:**

For **SARIMA:**
```python
# Check differencing
# Original series drifts? Increase d from 0 to 1
# Series has repeating pattern? Increase D from 0 to 1

old: SARIMA(1,0,1) × (1,0,1,48)  # d=0, D=0 may be non-stationary
new: SARIMA(1,1,1) × (1,1,1,48)  # d=1, D=1 removes trend+seasonality
```

For **ExponentialSmoothing:**
```python
# Add trend if forecast is flat
trend="add" vs trend="mul"
# Make seasonal additive if multiplicative fails
seasonal="add" vs seasonal="mul"
```

For **LightGBM:**
```python
# Add longer lags if missing weekly pattern
old: lags=[1, 2, 48, 96]        # Up to 2 days
new: lags=[1, 2, 48, 96, 336]   # Up to 1 week
# Increase capacity if underfitting
old: num_leaves=31, learning_rate=0.05
new: num_leaves=63, learning_rate=0.1
```

---

### Problem 2: Low Coverage (Intervals too narrow)

**Diagnosis:**
```python
PI Coverage: 45%  (target 80%)
mean_uncertainty_width: 0.8 kWh
actual residual std: 0.6 kWh
```

**Root cause:**
- Model over-confident
- Interval calculation wrong (assumption of normality fails)
- True variability higher than training suggested

**Fix:**

For **SARIMA:**
```python
# Check residuals are white noise
# If autocorrelated, increase p, d, or q
# If non-normal, model assumptions violated

# Manual fix: Inflate intervals
upper_adjusted = upper * 1.3  # Make 30% wider
lower_adjusted = lower * 1.3
```

For **ExponentialSmoothing:**
```python
# Add damping to reduce over-confidence
damped_trend=True
# Increase seasonal smoothing
seasonal="mul"  # Usually less confident than additive
```

For **LightGBM:**
```python
# Use quantile regression instead of residual-based PI
# Current: PI = forecast ± z * std(residuals)
# Better: Fit separate models for P10 and P90
```

---

### Problem 3: Biased Forecast (Systematic error)

**Diagnosis:**
```python
Residuals histogram:
  All shifted right (positive residuals)
  → Forecasts systematically too low
Residuals histogram:
  All shifted left (negative residuals)
  → Forecasts systematically too high
```

**Root cause:**
- Model structure missing (e.g., trend not captured)
- Training data non-representative (e.g., trained on cool season, testing on hot season)

**Fix:**

For **SARIMA:**
```python
# Add constant term
# Check if trend present (increasing/decreasing pattern)
# If yes, ensure d >= 1 to difference it out

# Diagnosis plot:
residuals_over_time = y_test - forecast.prediction
if np.mean(residuals_over_time) > 0.2:
    print("Systematic overforecasting (forecast too high)")
    # Try increasing p (catch more AR dynamics)
```

For all models:
```python
# Retrain on recent data only (discard old seasonal patterns)
# If seasonal pattern changed, reestimate
# Add external features (temperature, holidays, etc.)
```

---

## Part 7: Expected Values by Asset Type

### Residential Building (Stable usage)

**Expect:**
- MAE: 0.3–0.5 kWh
- MAPE: 8–12%
- PI Coverage: 78–82%
- Uncertainty Width: 1.2–1.8 kWh

**Why small errors:**
- Predictable patterns (people same routine)
- Daily seasonality strong (morning ramp, evening peak)
- SARIMA/ExpSmoothing excel here

**Best model:** Exponential Smoothing (simple patterns, fast fit)

---

### Commercial Building (More variable)

**Expect:**
- MAE: 0.5–1.0 kWh
- MAPE: 12–18%
- PI Coverage: 75–82%
- Uncertainty Width: 2.0–3.5 kWh

**Why larger errors:**
- Occupancy variable (some days empty)
- HVAC behavior complex (non-linear responses to temp)
- Weekly patterns (weekday ≠ weekend)

**Best model:** SARIMA or LightGBM (need to capture more structure)

---

### EV Charging Station (Highly variable)

**Expect:**
- MAE: 1.0–3.0 kWh
- MAPE: 20–40%
- PI Coverage: 70–80%
- Uncertainty Width: 4.0–8.0 kWh

**Why errors large:**
- Arrival times random (Poisson process)
- Charging duration variable
- Weekly/hourly patterns weaker

**Best model:** LightGBM with external features (weather, time-of-day signals)
- Can capture non-linear charging curves
- Can incorporate exogenous variables (price, demand response signals)

---

## Part 8: Checklist — Is This Forecast Production-Ready?

- [ ] RMSE < X% of mean(y_train) — Point forecast accurate
- [ ] MAE < Y kWh — Systematic error acceptable
- [ ] MAPE < 15% — Percentage error reasonable
- [ ] PI Coverage between 75–85% — Intervals well-calibrated
- [ ] Mean Uncertainty Width < 2 × residual std — No over-confidence
- [ ] Residuals histogram bell-shaped, centered at 0
- [ ] No trend in residual_over_time plot
- [ ] Forecast doesn't flat-line (adapts to new data patterns)
- [ ] Coverage ≈ uniform across time (no periods of systematic under/overcoverage)
- [ ] Uncertainty width correlates with volatility (wide in variable periods, narrow in stable)

**If all checkboxes ✓:**
→ Ready to deploy for flexibility forecasting

**If 2+ checkboxes ✗:**
→ Needs tuning or model switch

---

## Part 9: Formulas Quick Reference

| Concept | Formula | Code |
|---------|---------|------|
| MAE | $\frac{1}{n}\sum\|y_t - \hat{y}_t\|$ | `mean_absolute_error(y, yhat)` |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$ | `np.sqrt(mean_squared_error(y, yhat))` |
| MAPE | $\frac{100}{n}\sum\frac{\|y_t - \hat{y}_t\|}{y_t}$ | `mean_absolute_percentage_error(y, yhat)` |
| Coverage | $\frac{1}{n}\sum \mathbb{1}[y_t \in [L_t, U_t]]$ | `np.mean((y >= L) & (y <= U))` |
| z-score (80% PI) | $z = 1.282$ | `norm.ppf(0.90)` |
| Upper PI | $\hat{y}_t + z \cdot \sigma$ | `forecast + 1.282 * std(residuals)` |
| Lower PI | $\hat{y}_t - z \cdot \sigma$ | `forecast - 1.282 * std(residuals)` |

---

## Part 10: Common Pitfalls

### Pitfall 1: "RMSE is small, so forecast is good"

**False!** RMSE ≈ 0.5 kWh looks good until you check:
- PI Coverage = 20% (model massively overconfident)
- All residuals positive (systematic bias)

**Fix:** Always check all four metrics together.

---

### Pitfall 2: "My model has zero error on training data"

**Red flag!** Means model overfitted.

```python
train_rmse = 0.01 kWh  ← Too perfect
test_rmse = 1.50 kWh   ← Huge gap
```

**Solution:** Use validation set, penalize complexity (AIC/BIC), reduce hyperparameters.

---

### Pitfall 3: "PI Coverage varies a lot day-to-day"

**Example:**
```
Monday: 95% coverage
Tuesday: 60% coverage
Wednesday: 88% coverage
```

**Problem:** Intervals not robust (over/underconfident depending on conditions)

**Solution:** Retrain model, check for missed patterns (day-of-week effect?)

---

### Pitfall 4: "I tuned SARIMA(2,3,2)×(2,3,2,48)"

**Warning!** Likely overfitted.

```python
# Too many parameters = fitting noise
SARIMA(2,3,2) × (2,3,2,48)  # 10 parameters
# Data: only 480 training points, 10 parameters
# Ratio: 480/10 = 48 obs per parameter (marginal)

# Better:
SARIMA(1,1,1) × (1,1,1,48)  # 4 parameters
# Ratio: 480/4 = 120 obs per parameter (safe)
```

**Rule of thumb:** Parameters << n/10

---

## Summary: Decision Tree

```
START
│
├─ "Which model is best?"
│  └─ Rank by RMSE (lower = better)
│     └─ If tied, check PI Coverage (closer to target = better)
│
├─ "Is forecast good enough?"
│  ├─ RMSE/MAE ratio reasonable? (gap ≤ 1.5)
│  ├─ MAPE < 15%?
│  ├─ Coverage 75–85%?
│  ├─ No trend in residuals?
│  └─ All YES? → Deploy
│     All NO? → Tune or switch model
│
├─ "How to tune?"
│  ├─ SARIMA: Check d, D, then vary p, q, P, Q
│  ├─ ExpSmoothing: Try trend (add/mul), seasonal (add/mul)
│  ├─ LightGBM: Add longer lags, increase num_leaves
│
└─ "Is it overfitted?"
   ├─ Train RMSE << Test RMSE? → Yes, reduce complexity
   ├─ Coverage too high (>95%)? → Yes, intervals too wide
   └─ Parameters > n/10? → Yes, reduce p, q, P, Q
```
