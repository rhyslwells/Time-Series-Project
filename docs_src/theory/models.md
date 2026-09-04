# Models

## SARIMA(p,d,q) × (P,D,Q,s)

**Starting point for 30-min energy data:** `SARIMA(1,1,1) × (1,1,1,48)` — s=48 encodes daily seasonality.

**Differencing (d, D)** makes the series stationary, which ARIMA requires:

$$\Delta y_t = y_t - y_{t-1} \qquad \Delta_s \Delta y_t = \Delta y_t - \Delta y_{t-s}$$

Non-seasonal differencing (d) removes trend; seasonal differencing (D, s=48) removes the 24-hour repeating pattern.

**AR term (p, P)** — today depends on past values:

$$y_t = \phi_0 + \phi_1 y_{t-1} + \phi_s y_{t-s} + \epsilon_t$$

Captures momentum (if an asset drew power yesterday at this time, it likely will today). Expect $\phi_1 \approx 0.7\text{-}0.9$ — energy usage persists strongly.

**MA term (q, Q)** — today's forecast corrects for yesterday's error:

$$y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_s \epsilon_{t-s}$$

Expect $\theta_1 \approx 0.2\text{-}0.5$ (smaller than the AR coefficient).

**Complexity budget:** parameters should stay well under $n/10$. For a 480-point training set, keep $p+q < 48$, and prefer `(1,1,1)×(1,1,1,48)` (4 parameters, ~120 obs/parameter) over anything like `(2,3,2)×(2,3,2,48)` (10 parameters, ~48 obs/parameter — likely overfit).

**Good for:** stable patterns, strong daily cycle, fast fit (residential).
**Tuning direction:** RMSE high → increase p (more AR) or d (more differencing) or q (better error correction), in that order.

## Exponential Smoothing (Holt-Winters)

$$\hat{y}_{t+h} = \ell_t + T_t \cdot h + s_{t+h-s}$$

$$\ell_t = \alpha(y_t - s_{t-s}) + (1-\alpha)(\ell_{t-1} + T_{t-1}) \qquad T_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)T_{t-1} \qquad s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-s}$$

$\ell_t$ = level, $T_t$ = trend, $s_t$ = seasonal component. $\alpha, \beta, \gamma \in [0,1]$ control how fast each responds to new data (low = smooth/stable, high = noisy/responsive).

**Additive vs multiplicative seasonality:** additive assumes a constant kWh offset (e.g. +0.5 kWh every afternoon); multiplicative scales with the level (e.g. 1.5× demand every afternoon). Additive usually fits energy data, since the seasonal swing is closer to a fixed kWh amount than a fixed ratio.

**Good for:** simpler, stable seasonal patterns; fastest to fit; least prone to overfitting.
**Tuning direction:** coverage too low → try `seasonal="mul"` or add `damped_trend=True` to reduce over-confidence.

## LightGBM (Gradient Boosting)

Ensemble of trees, each correcting the previous ensemble's residual error:

$$\hat{y}_t = \sum_{m=1}^{M} \gamma_m f_m(\mathbf{x}_t)$$

For time series, $\mathbf{x}_t$ is built from lag features (e.g. `[1, 2, 48, 96]` → 30-min, 1-hour, 1-day, 2-day lags) plus an hour-of-day feature.

**Good for:** non-linear demand curves, multiple lag scales, portfolio-level forecasts with exogenous features (price, demand-response signals).
**Pitfall:** prone to overfitting the lag structure — validate on held-out data, and note that residual-based prediction intervals can come out too narrow if the point forecast is systematically optimistic (consider quantile regression, or pairing LightGBM's point forecast with SARIMA's intervals).
**Tuning direction:** RMSE high → add longer lags (e.g. `336` for weekly) or raise `num_leaves`/`learning_rate` for more capacity.

## Choosing between them

| | SARIMA | Exp. Smoothing | LightGBM |
|---|---|---|---|
| Best for | stable + daily cycle | simple + stable seasonality | non-linear, multi-scale lags |
| Fit speed | moderate | fast | moderate |
| Interval reliability | good (residual-based) | good | needs care — see pitfall above |
| Typical winner | residential | residential | EV charging, complex commercial |
