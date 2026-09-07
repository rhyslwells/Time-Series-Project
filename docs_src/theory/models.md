# Models

## Choosing between them

| | SARIMA | Exp. Smoothing | LightGBM |
|---|---|---|---|
| Best for | stable series with a single strong daily cycle | simple, stable seasonality | non-linear demand, multi-scale lags, exogenous features |
| Fit speed | moderate | fast | moderate |
| Interval behaviour | width grows with horizon (state-space CI) | constant width (single residual std) | constant width (single residual std) — see pitfall below |
| When to reach for it | you want calibrated day-ahead intervals and one seasonal period is enough | you want a fast, hard-to-overfit baseline | the relationship is non-linear or you have price/DR regressors to add |

No "typical winner" column: with 14 days and one fixed 10/4 split, the ranking between the
three models is largely noise. Rolling-origin evaluation is the fix and it is not yet
implemented anywhere.

## The weekly cycle no default configuration captures

`generate_raw_data.py` injects a 1.2/0.8 weekday/weekend factor for EV assets, and the data
docs report it as a headline characteristic. But:

- SARIMA takes a single seasonal period; the recommended `s=48` is daily only.
- `LightGBMModel._create_features` builds lags plus hour-of-day, with **no day-of-week feature**; default lags `[1, 2, 48, 96]` reach back two days.
- Holt-Winters takes one `seasonal_periods`.

So a known, deliberately injected weekly pattern is invisible to all three defaults. The
standard treatments are Fourier terms at both periods with ARIMA errors, MSTL, or simply a
day-of-week feature for the tree model. **The catch:** 14 days is only two weekly cycles, so
the weekly component is barely estimable — the honest conclusion is that this dataset cannot
support it, and `s=336` / lag `336` are not real options here.

## SARIMA(p,d,q) x (P,D,Q,s)

**Starting point for 30-min energy data:** `SARIMA(1,0,1) x (1,1,1,48)` — `s=48` encodes
daily seasonality. Note `d=0`: [synthetic_metering_data.md](../data/synthetic_metering_data.md)
states the data has no trend or drift, and load is mean-reverting in level anyway, so
non-seasonal differencing mostly inflates the innovation variance and induces spurious
negative MA structure. `D=1` (seasonal differencing) is still appropriate. If you do apply
`d=1`, expect the AR coefficient to be estimated on $\Delta y$, where values are small and
often negative — a correctly fitted model can look broken.

**Differencing (d, D)** makes the series stationary, which ARIMA requires. Test for it
rather than assuming: ADF/KPSS for `d`, a seasonal-strength or OCSB/Canova-Hansen criterion
for `D`.

$$\Delta y_t = y_t - y_{t-1} \qquad \Delta_s \Delta y_t = \Delta y_t - \Delta y_{t-s}$$

**AR term (p, P)** — today depends on past values:

$$y_t = \phi_0 + \phi_1 y_{t-1} + \phi_s y_{t-s} + \epsilon_t$$

On the level (`d=0`), expect $\phi_1 \approx 0.7\text{-}0.9$ — energy usage persists strongly.
On differenced data this no longer holds.

**MA term (q, Q)** — today's forecast corrects for yesterday's error:

$$y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_s \epsilon_{t-s}$$

**Order selection:** use AICc at these sample sizes (it is the standard instrument for ARIMA
order selection; `archive/.../Forecasting_AutoArima.py` has a starting point). As a rough
guard, keep the total parameter count well under $n/10$: for a ~480-point training set that
is about 4-5 parameters, so `(1,0,1)x(1,1,1,48)` (4 parameters) is near the ceiling and
anything like `(2,3,2)x(2,3,2,48)` is overfit.

**Good for:** stable patterns, one strong daily cycle, calibrated day-ahead intervals.
**Tuning direction:** RMSE high -> increase p (more AR) or q (better error correction); add
`d` only if a stationarity test says so.

## Exponential Smoothing (Holt-Winters)

$$\hat{y}_{t+h} = \ell_t + T_t \cdot h + s_{t+h-s}$$

$$\ell_t = \alpha(y_t - s_{t-s}) + (1-\alpha)(\ell_{t-1} + T_{t-1}) \qquad T_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)T_{t-1} \qquad s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-s}$$

$\ell_t$ = level, $T_t$ = trend, $s_t$ = seasonal component. $\alpha, \beta, \gamma \in [0,1]$
control how fast each responds to new data (low = smooth/stable, high = noisy/responsive).

**Additive vs multiplicative seasonality:** additive assumes a constant offset (e.g. +0.5
every afternoon); multiplicative scales with the level. Additive fits energy data better,
and **multiplicative requires strictly positive data** — it will fail or return nonsense for
the seven `solar_battery` assets, which are net exporters that cross zero. Do not use
`seasonal="mul"` on this dataset.

**Good for:** simpler, stable seasonal patterns; fastest to fit; least prone to overfitting.
**Tuning direction:** coverage too low -> add `damped_trend=True` to reduce over-confidence.
Note the interval width is constant across the horizon by construction.

## LightGBM (Gradient Boosting)

Ensemble of trees, each correcting the previous ensemble's residual error:

$$\hat{y}_t = \sum_{m=1}^{M} \gamma_m f_m(\mathbf{x}_t)$$

For time series, $\mathbf{x}_t$ is built from lag features (default `[1, 2, 48, 96]` -> 30-min,
1-hour, 1-day, 2-day lags) plus an hour-of-day feature. Add a **day-of-week feature** if you
want any chance of the weekday/weekend effect.

**Good for:** non-linear demand curves, multiple lag scales, portfolio-level forecasts with
exogenous features (price, demand-response signals).

**Pitfall — the intervals come out too narrow, and not because of bias.** `forecast()` is
recursive: past roughly step 96 every lag feature is itself a prediction, so errors compound
with horizon — while the interval width, a single one-step in-sample residual std, never
moves. The fix is on the forecasting strategy (direct multi-step, or DirRec) or the interval
(quantile regression), not on the point forecast.

**Tuning direction:** RMSE high -> raise `num_leaves`/`learning_rate` for more capacity, or
add a day-of-week feature. Longer lags like `336` (weekly) are not usefully estimable on 14 days.
