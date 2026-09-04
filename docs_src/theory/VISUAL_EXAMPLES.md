# Visual Examples: Good vs Bad Forecasts

NOTE: Reference the plot from C:\Users\RhysL\Desktop\Time Series Project\src\ts_plots.py

## Section 1: Forecast vs Actual Plot

### GOOD CASE: Well-Calibrated Model

NOTE: I dont like acsi diagrams just exaplin in words whats we see instead.

```
                     ↑ Upper PI (P90)
                    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
                   /                  \
Actual (black) ━━●━━━●━━━●━━━●━━━●━━━━ ← Mostly inside PI
               /  \              \    \
Forecast    ━━━━━●━━━●━━━●━━━●━━━●━━━ ← Tracks actual, lags slightly
            /    \               \    \
           /      \_______________\____ ← Lower PI (P10)
          /

Metrics:
  MAE: 0.35 kWh          ← Point forecast accurate
  RMSE: 0.52 kWh         ← Reasonable gap (some outliers)
  MAPE: 9%               ← Within 10% of actual
  PI Coverage: 80%       ← Matches target confidence level
  Width: 1.6 kWh         ← Proportional to uncertainty
```

**What's good here:**
- Black dots (actual) mostly within shaded region
- Blue line (forecast) follows trend, just lags
- PI narrows during stable periods (midday), widens during transitions
- No systematic bias (actuals not all above/below forecast)

---

### BAD CASE 1: Overconfident (Intervals Too Narrow)

```
                     ↑
                    /‾|‾‾\
                   /  |    \
Actual (black) ━━●━╱━━●━ ← Spikes outside narrow PI
               /  ╱    ╱ \
Forecast    ━━━━━━●━━●━ ← Forecast reasonable, but
             \   ╱   ╱   \  intervals don't capture volatility
              ╲_/_____/____\
               ↓

Metrics:
  MAE: 0.40 kWh          ← Forecast OK
  RMSE: 0.60 kWh         ← Similar to MAE (few outliers)
  MAPE: 11%              ← Reasonable
  PI Coverage: 45%       ← TOO LOW! (Target 80%, got 45%)
  Width: 0.8 kWh         ← Too narrow
```

**What's wrong:**
- Many red dots (actual values) fall outside PI bounds
- Model says "I'm 80% sure" but only 45% of actuals actually within bounds
- Consequence: When you plan based on this envelope, surprises happen 55% of the time
- Cause: Residual std estimated poorly, or model assumes normal distribution when it's not

**How to fix:**
- Retrain with `damped_trend=True` (for ExpSmoothing)
- Check if residuals are actually normal (plot histogram)
- Manually widen intervals: `upper *= 1.5, lower *= 0.67`
- Switch to quantile regression (predict P10 and P90 separately)

---

### BAD CASE 2: Underfitting (Forecast Flat-Lined)

```
                     ↑
                    /‾‾‾\
                   /     \
Actual (black) ━━●━━●━━●━ ← Sharp swings up and down
               /  \       \
Forecast    ━━━━━━●━━━●━ ← Flat, doesn't react
             \   /       \
              \_/         \_

Metrics:
  MAE: 1.5 kWh           ← High error!
  RMSE: 2.2 kWh          ← Much higher than MAE (big outliers)
  MAPE: 25%              ← Poor
  PI Coverage: 85%       ← Ironically OK (only because intervals wide)
  Width: 4.0 kWh         ← Very wide intervals
```

**What's wrong:**
- Blue line (forecast) barely moves, actual wildly varies
- Model gave up, predicting mean value every time
- Intervals are wide but only because of poor fit, not because model is uncertain

**Cause:**
- Differencing too aggressive (d=2 removed all signal)
- Wrong model class (SARIMA bad for this data, should use LightGBM)
- Hyperparameters wrong (p=0, q=0 means no dynamics captured)

**How to fix:**
- Reduce differencing: SARIMA(1,0,1) if d=1 too much
- Add AR: SARIMA(2,1,1) to capture past-value dependencies
- Switch to LightGBM with longer lags
- Verify data loading (maybe data is shuffled?)

---

### BAD CASE 3: Systematically Biased

```
                     ↑
                    /‾‾‾\
                   /     \
Actual (black) ━━●━━●━━●━
               /  /       \
Forecast    ━━●━━━●━━●━ ← Always below actual
             /  /         \
            ╱__╱___________\

Metrics:
  MAE: 0.50 kWh          ← Seems OK
  RMSE: 0.60 kWh         ← OK
  MAPE: 12%              ← OK individually
  PI Coverage: 88%       ← Higher than target (intervals wide)
  Width: 2.0 kWh         ← Wide

BUT: Look at residuals histogram
  All bars on right side (positive residuals)
  Mean(residuals) = +0.35 kWh (not centered at 0!)
```

**What's wrong:**
- Forecast consistently underestimates (forecast too low)
- Model doesn't know it's wrong
- Consequence: Plan for 5 kWh flexibility, actual demand is 5.35 kWh
  → Fail to deliver!

**Why it matters for FlexGo:**
```
If forecast is systematically low:
  Forecast says: "Asset will use 3 kWh ± 1 kWh"
  Upper bound: 4 kWh
  You commit 4 kWh flexibility
  But asset actually uses 5 kWh
  → Commitment breach!
```

**Cause:**
- Seasonal pattern changed (e.g., summer vs winter not accounted for)
- Training data not representative
- Missing external variable (temperature, holidays)

**How to fix:**
- Retrain on recent data only (last 30 days)
- Add trend term: Check if forecast should increase over time
- Decompose series: Separate level, trend, seasonality and inspect each
- Add external features: Temperature, occupancy, time-of-year

---

## Section 2: Residuals Diagnostic Plot

### GOOD CASE: White Noise Residuals

```
Left panel (residuals over time):
  Time  Residuals  Uncertainty
  1     +0.3 kWh   1.6 kWh  (residual within bounds ✓)
  2     -0.1 kWh   1.6 kWh  (✓)
  3     +0.5 kWh   1.7 kWh  (✓)
  4     -0.4 kWh   1.5 kWh  (✓)
  5     +0.1 kWh   1.6 kWh  (✓)
  
  ✓ Residuals bounce randomly (white noise)
  ✓ Bars mostly within ±2 uncertainty bands
  ✓ No trend (doesn't drift up or down over time)
  ✓ Occasional spikes normal (outliers)

Right panel (histogram):
  
   |
   |      ╭─╮
   |     ╭─┴─╮
   |    ╭─┴───╮
   |   ╭─┴──╱─╮
   |───┴──────┴──→ Residual Value
  -2  -1  0  +1  +2 kWh
  
  ✓ Bell-shaped (normal distribution)
  ✓ Centered at 0
  ✓ Most residuals within ±1 std dev
  ✓ No skew (not lopsided)
```

**Statistics:**
```
Mean(residuals) = 0.02 kWh  ← ≈ 0 ✓
Std(residuals) = 0.35 kWh
Uncertainty width = 1.6 kWh
Ratio: 1.6 / (2 × 0.35) = 2.3 ← Should be ≈ 1.28 (z-score for 80% PI)
                                 Slightly conservative, OK
```

---

### BAD CASE 1: Autocorrelated Residuals

```
Left panel (residuals over time):
  Time  Residuals
  1     +0.5 kWh   ↗
  2     +0.4 kWh   │
  3     +0.3 kWh   │
  4     -0.1 kWh   ↘
  5     -0.2 kWh   ↘
  6     -0.3 kWh   │
  
  ✗ Residuals not random! Clear pattern:
    - High values cluster together
    - Low values cluster together
    - Suggests model missed autocorrelation
  
  ✗ Implication: ARIMA assumptions violated
    (Residuals should be independent, but they're not)
```

**What it means:**
```
If residual(t) is positive, then residual(t+1) likely positive too
→ Forecast at t underestimates → Forecast at t+1 also underestimates
→ Errors cascade instead of self-correcting
```

**Cause:**
- Under-differenced: Set d=2 (or D=2 for seasonal)
- Too few AR terms: Increase p (or P for seasonal)
- Missed structure: Add longer lags

**Fix:**
```python
# Check ACF plot (would show significant bars)
# If autocorrelation > 0.2, increase differencing or AR

old: SARIMA(1,1,1) × (1,1,1,48)
new: SARIMA(2,1,1) × (1,1,1,48)  # More AR
or
new: SARIMA(1,2,1) × (1,2,1,48)  # More differencing
```

---

### BAD CASE 2: Systematic Bias in Residuals

```
Left panel (residuals over time):
  Time  Residuals
  1     +0.3 kWh   |
  2     +0.2 kWh   |
  3     +0.4 kWh   |  ← All positive!
  4     +0.1 kWh   |
  5     +0.35 kWh  |
  
  ✗ All residuals on same side (positive)
    → Forecast systematically too low
  
  Mean(residuals) = +0.27 kWh  ← Should be 0!

Right panel (histogram):
  
   |           ╭─╮
   |          ╭─┴─╮
   |          ╰───╮
   |              │
   |______________●──→ Residual Value
                  +0.27 (shifted right)
  
  ✗ Histogram skewed right (all mass on positive side)
  ✗ Not centered at 0
```

**Consequences:**
```
Forecast says: 3 kWh
Residual: +0.3 kWh
Actual: 3.3 kWh

Forecast says: 4 kWh
Residual: +0.2 kWh
Actual: 4.2 kWh

→ Model always underestimates by ~0.25 kWh
→ Plan for 5 kWh flexibility, actual is 5.25 kWh
→ Commitment breach!
```

**Cause:**
- Missing trend (forecast should increase over time)
- Seasonal pattern changed (trained on winter, testing on summer)
- External factor (temperature, holidays) not captured

**Fix:**
```python
# Check if trend exists
plt.plot(y_train)  # Does it drift up or down?

# If yes, ensure d=1 or d=2
SARIMA(1, 1, 1)  # d=1 removes trend

# Also check: Retrain on recent data only
y_recent = y[-480:]  # Last 10 days
model.fit(y_recent)  # Seasonal patterns may have shifted
```

---

### BAD CASE 3: Non-Normal Residuals (Heavy Tails)

```
Right panel (histogram):
  
   |
   |      ╭─╮
   |     ╭─┴─╮
   |    ╭─┴───╮
   |   ╭─┴────╱╲
   |   │       │  │
   |___|_______|__|___→ Residual
   -5 -2 0 +2 +5  (extreme outliers!)
  
  ✗ Tall bars at extremes (heavy tails)
    → Normal distribution assumption fails
    → True uncertainty wider than model thinks
  
  Implication: PI too narrow in reality
             But model thinks it's right width
             → Coverage will be < target
```

**What it means:**
```
Model assumes residuals ~ N(0, σ²)
But actual distribution has fatter tails
→ Extreme events more likely than model predicts
```

**Cause:**
- Outliers in training data (sensor glitches, maintenance events)
- Model can't capture rare events (e.g., equipment failure)
- Wrong model class (linear model can't capture non-linear shocks)

**Fix:**
```python
# Option 1: Remove/smooth outliers
residuals_cleaned = remove_outliers(residuals, threshold=3 * std)

# Option 2: Use robust model (LightGBM more resilient)
model = LightGBMModel(y_train)

# Option 3: Inflate PI manually
upper *= 1.3
lower *= 0.7

# Option 4: Use quantile regression (predict P10, P90 directly)
# Instead of: upper = forecast + 1.28*std
# Do: upper = model_P90.predict(X)
```

---

## Section 3: Uncertainty Width Plot

### GOOD CASE: Uncertainty Reflects Volatility

```
Uncertainty width over time:

    │                    Morning (high)
    │  ╭────╮       ╭────╮
2.0 │ ╱      ╲     ╱      ╲
    │╱        ╲   ╱        ╲
1.5 │          ╰─╯          ╰╮
    │ ↑ Transitions    ↓ Stable midday
1.0 │                 ¯¯¯¯¯¯¯¯¯¯    ↑ Evening ramp
    │                               ╱╲
  0 └───────────────────────────────
    0         6        12        18    24 (Hours)

✓ Width reflects true volatility:
  - Narrow (1.0 kWh) during stable noon hours
  - Wide (1.8 kWh) during morning/evening transitions
  - Adapts to conditions

Mean width: 1.4 kWh
Std of width: 0.4 kWh  ← Shows variation (good)
```

**Interpretation:**
```
At 10 AM: Width = 0.9 kWh → Forecast ±0.45 kWh (tight)
         Reason: Steady state (everyone working, HVAC stable)

At 8 AM: Width = 1.8 kWh → Forecast ±0.9 kWh (wide)
        Reason: People arriving, HVAC ramping up (uncertain)

This makes sense! Model knows when it's confident.
```

---

### BAD CASE 1: Flat Uncertainty (One-Size-Fits-All)

```
Uncertainty width over time:

2.0 │ ┌─────────────────────────────┐
    │ │  Width = constant 1.0 kWh   │ ← No variation!
1.0 │ │  everywhere                  │
    │ └─────────────────────────────┘
  0 └───────────────────────────────
    0         6        12        18    24 (Hours)

✗ Width same everywhere (morning, noon, evening)
✗ Doesn't adapt to conditions
✗ Says "I'm equally uncertain all the time"
  But actually midnight is predictable, peaks are chaotic

Mean width: 1.0 kWh
Std of width: 0.0 kWh  ← No variation (bad)
```

**Cause:**
- Model learned constant PI, not data-driven
- Or model class doesn't support adaptive intervals

**Consequence:**
```
At predictable noon: Intervals too wide (waste conservatism)
At chaotic morning: Intervals too narrow (under-confident)

Not optimal for either scenario.
```

**Fix:**
```python
# SARIMA and ExpSmoothing naturally adapt
# Make sure they're fitting residuals properly

# For LightGBM, explicitly model uncertainty:
# Instead of: PI = forecast ± constant
# Do: Use residual_std_model to predict uncertainty

# Or: Predict quantiles directly
upper_model = train_quantile_model(y_train, quantile=0.9)
lower_model = train_quantile_model(y_train, quantile=0.1)
```

---

### BAD CASE 2: Exploding Uncertainty (Unrealistic)

```
Uncertainty width over time:

10.0│                            ╱╲
    │                        ╱╲ ╱  ╲
5.0 │                    ╱╲╱  ╲      ╲
    │                ╱╲╱              ╲
2.0 │            ╱╱  ╲                 ╲╲
    │        ╱╱                         ╲╲
1.0 │    ╱╱                              ╲╲
    │╱╱_________________________           ╲╱
  0 └───────────────────────────────────────
    0         6        12        18    24

✗ Width explodes over forecast horizon
  (Forecast window is only 4 hours, width shouldn't grow that fast)
✗ Means model loses all confidence very quickly
```

**Cause:**
- Using naive error propagation
  (e.g., std grows as sqrt(horizon) in random walk)
- Not recalibrating to actual data
- LightGBM just repeating last few values

**Problem for FlexGo:**
```
Hour 0: Forecast = 3 kWh, Width = 1 kWh
Hour 1: Forecast = 3 kWh, Width = 1.5 kWh
Hour 2: Forecast = 3 kWh, Width = 3 kWh ← Nonsensical
Hour 3: Forecast = 3 kWh, Width = 6 kWh ← Giving up

By hour 3, "I'm 80% sure it's between -3 and 9 kWh"
That's useless for planning!
```

**Fix:**
```python
# For SARIMA/ExpSmoothing: This shouldn't happen
# (They respect seasonal patterns, width should plateau)

# For LightGBM: Use "recursive forecast with recent actuals"
# At each step, pretend latest forecast is actual
# This prevents std from exploding

# Or: Cap uncertainty width
upper = np.minimum(upper, forecast + 3 * std_train)
lower = np.maximum(lower, forecast - 3 * std_train)
```

---

## Section 4: PI Coverage Plot

### GOOD CASE: 80% Green, 20% Red

```
PI Coverage scatter plot:

Y
│
5 │     ● (green)    ✓
  │  ● (green)  ●(red)   ← Single red dot (expected)
4 │   ● (green)  ▲ Upper bound
  │  ● (green)  ●│(green)
3 │ ● (green) ──┼── Forecast
  │ ●(green) ●│(green)
2 │● (red) ───┘─ Lower bound  ← Only 1 red out of ~6 here (16%)
  │
1 │
  └────────────────────→ Time
  
✓ Mostly green (80%)
✓ Few red (20%) scattered randomly
✓ No pattern (reds not all in morning, all on weekends, etc.)
✓ Coverage = 80% matches target confidence level
```

**Statistics:**
```
Green dots: 80 / 100 = 80%  ← Perfect!
Red dots: 20 / 100 = 20%
Pattern: Random (no clustering)
```

**What it means:**
```
"When I say 80% confidence, actual actuals fall in 80% of the time"
Model is well-calibrated ✓
```

---

### BAD CASE 1: Too Many Reds (Under-Coverage)

```
PI Coverage scatter plot:

Y
│
5 │     ●(red)    ✗✗
  │  ●(red)  ●(red)   ← Too many reds!
4 │   ●(red)  ▲
  │  ●(red)  ●(red)
3 │ ●(red)  ──┼── 
  │ ●(red)  ●(red)
2 │● (red)  ──┘
  │ ●(green)
1 │ ●(green)
  └────────────────────→ Time
  
✗ Only 15% green (many reds)
✗ Coverage = 15% << 80% target
✗ Model WAY too confident
```

**Consequence:**
```
Model says: "I'm 80% confident forecast is 3 kWh ± 1 kWh"
           (Between 2 and 4 kWh)
Actual is: 4.5 kWh (outside bounds!)

This happens 85% of the time (catastrophic!)
→ Can't rely on this forecast for planning
→ Flexibility envelope commitments will breach constantly
```

**Cause:**
```
1. Residual std estimated too low
   (Maybe training data was calm, test data is volatile)
   
2. Normality assumption wrong
   (Residuals have heavy tails, extremes more likely)
   
3. Model systematically optimistic
   (Forecast variance doesn't match true variability)
```

**Fix:**
```python
# Step 1: Check residual distribution
import matplotlib.pyplot as plt

residuals = y_test - forecast.prediction
plt.hist(residuals)  # Is it really normal?

# Step 2: Manually inflate intervals
inflation_factor = 1.5
upper_new = upper * inflation_factor
lower_new = lower / inflation_factor

# Step 3: Use different confidence level
# If 80% PI gives 15% coverage, try 95% PI
# Upper = forecast + 1.96 * std (not 1.28)

# Step 4: Use quantile regression
# Train separate models for P10 and P90
# Don't assume residuals are normal
```

---

### BAD CASE 2: Systematic Pattern (Undercoverage at Peak Hours)

```
PI Coverage scatter plot (color coded by hour):

Y
│
5 │  ●(green) ✓  ●●(red)✗ (peak hours all red!)
  │ ●(green)    ●●(red)
4 │●(green)    ●●(red)
  │●(green)   ●●●(red)   ← Red cluster (8 AM - 5 PM)
3 │●(green) ●●●●(red)
  │●(green)●●●●(red)
2 │●●(green)●●(red)
  │●●●(green)
1 │
  └────────────────────→ Time
  
✗ Green (overnight): 95% coverage ✓ Over-confident
✗ Red (daytime): 30% coverage ✗ Under-confident
✗ Not uniform!

Coverage by hour:
  Midnight-6 AM:  92% ← Too high (intervals too wide)
  6 AM - 5 PM:    35% ← Too low (intervals too narrow)
  5 PM - Midnight: 88% ← Reasonable
```

**What it means:**
```
Model doesn't understand time-of-day effects
- Night is stable (actually needs narrow intervals, not wide)
- Day is chaotic (needs wide intervals, not narrow)
→ Model got it backwards!
```

**Cause:**
- Model didn't capture day/night cycle properly
- Or seasonal pattern shifted (winter vs summer)
- Or external factor (temperature, demand response) missing

**Fix:**
```python
# Ensure model captures seasonal component
# SARIMA: Make sure P > 0, D > 0 for seasonal differencing
SARIMA(1,1,1) × (1,1,1,48)  # ← Seasonal component present

# LightGBM: Add hour-of-day feature
# (Should already be there, check feature importance)

# Retrain on recent data matching current conditions
# If test is summer but trained on winter:
y_recent = y[-480:]  # Last 10 days (current season)
model.refit(y_recent)

# Add external variables
# Temperature, occupancy, demand response signals
X = np.column_stack([
    y_train,
    temperature_train,
    occupancy_train
])
```

---

### BAD CASE 3: All Greens (Over-Coverage)

```
PI Coverage scatter plot:

Y
│
5 │     ●(green)    ✓
  │  ●(green)  ●(green)   ← All green!
4 │   ●(green)  ▲
  │  ●(green)  ●(green)
3 │ ●(green)  ──┼── 
  │ ●(green)  ●(green)
2 │● (green)  ──┘
  │ ●(green)  (no reds!)
1 │ ●(green)
  └────────────────────→ Time
  
✗ 100% coverage
✗ But target was 80%!
✗ Intervals way too wide
```

**Consequence:**
```
Forecast: 3 kWh, Bounds: 0 to 6 kWh
That's useless! Too wide to plan on.

For flexibility commitment:
  Available upside = 6 - 3 = 3 kWh
  Available downside = 3 - 0 = 3 kWh
  That's huge, wouldn't commit this much.
  
Opportunity cost: Could have narrower intervals with 95% confidence
                 instead of 100% with useless width
```

**Cause:**
- Model panicked (can't find patterns, just widens bounds)
- Or intervals inflated to force high coverage
- Or residual std estimate way off

**Fix:**
```python
# Root cause: Model giving up
# 1. Check if forecast is actually moving (or flat?)
if forecast.std() < 0.1:
    print("Forecast not moving! Model underfitted")
    # Try more complex model (increase p, q, P, Q)
    # Or switch to LightGBM
    
# 2. If forecast OK but bounds too wide:
#    Might be correct (data is genuinely variable)
#    Check residual distribution
residual_std = np.std(y_test - forecast.prediction)
expected_width = 2 * 1.28 * residual_std
actual_width = np.mean(forecast.uncertainty_width)
print(f"Expected: {expected_width}, Actual: {actual_width}")

# 3. If actual >> expected:
#    Confidence level inflated or intervals calculated wrong
#    Verify: Should be forecast ± 1.28 * std for 80% PI
```

---

## Section 5: Comparison Across Models

### Good Ranking (SARIMA Best)

```
Model Ranking Table:
┌───────────────────┬───────┬───────┬───────────────┐
│ Model             │ RMSE  │ MAPE  │ PI Coverage % │
├───────────────────┼───────┼───────┼───────────────┤
│ SARIMA (1,1,1)    │ 0.45  │ 9%    │ 80%          │  ← Best
│ ExponentialSmooth │ 0.52  │ 11%   │ 76%          │
│ LightGBM          │ 0.48  │ 10%   │ 72%          │
└───────────────────┴───────┴───────┴───────────────┘

✓ SARIMA wins on RMSE (0.45 lowest)
✓ SARIMA also has best coverage (80% matches target)
✓ Clear winner
```

**Decision:**
```
Deploy SARIMA(1,1,1) × (1,1,1,48)
Expect: ±0.45 kWh error, 80% confidence in flexibility envelope
```

---

### Unclear Ranking (Trade-off Needed)

```
Model Ranking Table:
┌───────────────────┬───────┬───────┬───────────────┐
│ Model             │ RMSE  │ MAPE  │ PI Coverage % │
├───────────────────┼───────┼───────┼───────────────┤
│ LightGBM          │ 0.40  │ 8%    │ 68%          │  ← Best RMSE
│ SARIMA (1,1,1)    │ 0.48  │ 10%   │ 82%          │  ← Best coverage
│ ExponentialSmooth │ 0.55  │ 12%   │ 79%          │
└───────────────────┴───────┴───────┴───────────────┘

? LightGBM has lower error but poor calibration (68% < 80% target)
? SARIMA is balanced (good RMSE, good coverage)
```

**Decision logic:**
```
For FlexGo flexibility forecasting:
  Coverage (80%) matters more than RMSE
  
  Why? If coverage = 68%, flexibility envelope will breach 32% of time
       (Too risky for commitments)
       
  Even though LightGBM point forecast is better,
  can't rely on its prediction intervals.

Recommendation: Use SARIMA
  → Point forecast reasonable (0.48 vs 0.40)
  → Intervals reliable (82% ≈ 80%)
  → Small coverage offset (2%) easily corrected
  
Or: Use LightGBM for point forecast, SARIMA intervals
    (Ensemble approach)
```

---

## Section 6: Expected Metrics by Asset Type

### Type 1: Residential (Stable)

```
Good residential forecast:
┌─────────────────────────┐
│ MAE:     0.3-0.4 kWh    │
│ RMSE:    0.4-0.5 kWh    │
│ MAPE:    8-12%          │
│ Coverage: 78-82%        │
│ Width:   1.0-1.5 kWh    │
└─────────────────────────┘

Why small errors:
- Patterns stable (people same routine)
- Daily seasonality strong
- Fewer surprises

Best model: ExponentialSmoothing or SARIMA(1,1,1)×(1,1,1,48)
(Simple patterns, fast fit)
```

---

### Type 2: Commercial (Variable)

```
Good commercial forecast:
┌─────────────────────────┐
│ MAE:     0.6-1.0 kWh    │
│ RMSE:    0.8-1.2 kWh    │
│ MAPE:    12-18%         │
│ Coverage: 76-84%        │
│ Width:   2.0-3.0 kWh    │
└─────────────────────────┘

Why larger errors:
- Occupancy variable (sometimes empty)
- HVAC nonlinear
- Weekly patterns matter

Best model: SARIMA with P>0,Q>0 or LightGBM
(Captures more structure)
```

---

### Type 3: EV Charging (Chaotic)

```
Good EV charging forecast:
┌─────────────────────────┐
│ MAE:     1.0-3.0 kWh    │
│ RMSE:    1.5-4.0 kWh    │
│ MAPE:    25-40%         │
│ Coverage: 70-80%        │
│ Width:   4.0-8.0 kWh    │
└─────────────────────────┘

Why huge errors:
- Arrival times random (Poisson)
- Charging duration unknown
- Weekly patterns weak

Best model: LightGBM with external features
(Price signals, demand response indicators)
or just accept forecast is uncertain
```

---

## Summary: Checklist for "Production-Ready" Forecast

✓ MAE < 15% of mean(y_train)
✓ RMSE/MAE ratio < 1.5 (no huge outliers)
✓ MAPE < 15% (typical energy target)
✓ PI Coverage 75-85% (well-calibrated)
✓ Residuals bell-shaped, centered at 0
✓ No trend in residuals over time
✓ No autocorrelation in residuals
✓ Uncertainty width varies with time-of-day
✓ Coverage uniform across hours (no systematic under/overcoverage)
✓ Forecast adapts to new patterns (not flat-lined)

**If all ✓ → Deploy with confidence**
**If 3+ ✗ → Needs more tuning**
