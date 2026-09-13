# Supervised vs statistical models

LightGBM (supervised/tree-based) and SARIMA (classical statistical) as a worked
comparison of the two model families available in the framework — see [Models](../theory/models.md)
for the full parameter-level treatment of each.

## When to choose LightGBM over SARIMA

For 30-minute short-term load forecasting, LightGBM is generally preferred over SARIMA once
any of the following hold:

- **Nested seasonality.** The data has both a daily (48-step) and weekly (336-step) cycle.
  SARIMA takes a single seasonal period, so fitting a large `s=336` term directly is
  computationally expensive and unstable — see [the weekly-cycle
  gap](../theory/models.md#the-weekly-cycle-no-default-configuration-captures) for why this
  project's defaults miss it either way.
- **Exogenous drivers.** Temperature, day-of-week, holiday indicators. LightGBM incorporates
  arbitrary numerical and categorical features alongside lags natively; SARIMA needs
  SARIMAX-style extensions.
- **Non-linear relationships.** Tree splits capture non-linear responses (e.g. heating/cooling
  demand under temperature extremes) without explicit transformations.

| | SARIMA | LightGBM |
|---|---|---|
| Advantages | Parsimonious and statistically transparent; no feature matrix — differencing and autocorrelation are modeled internally | Scales to large, high-dimensional feature sets; combines lags, calendar flags, Fourier terms, and weather forecasts freely |
| Limitations | Fails on high-frequency multi-seasonality; incorporating exogenous features requires SARIMAX complexity | Cannot extrapolate past the training range (trees output constant step functions — see [the interval pitfall](../theory/models.md#lightgbm-gradient-boosting)); needs manual lag/rolling-window engineering |

## Pitfalls when moving from ARIMA to a supervised model

- **Look-ahead bias.** Random k-fold CV shuffles rows, letting future data leak into training;
  fitting scalers or imputers on whole-dataset statistics does the same. Use walk-forward
  validation — see [Rolling-origin evaluation](../theory/rolling-origin-evaluation.md).
- **Time isn't automatic.** ARIMA models time dependence explicitly; a supervised model treats
  rows as independent ($X \to y$) unless lag, rolling-statistic, and calendar features are
  engineered by hand.
- **Unaddressed trend.** Tree models can't extrapolate a linear trend — remove it (differencing,
  or model-and-subtract) before training if one is present. This dataset has none; see
  [Forecasting approach](../theory/forecasting-approach.md).
- **Feature explosion.** Generating many lags and rolling windows without selection (PACF,
  information criteria) overloads the tree model and overfits.
- **Leading `NaN`s from lagging.** Drop them chronologically; never fill with future values.

## Comparing SARIMA and LightGBM fairly

1. **Walk-forward validation.** Score both models on a rolling or expanding window that respects
   chronological order — never a randomized split. See [Rolling-origin
   evaluation](../theory/rolling-origin-evaluation.md).
2. **Same test horizon.** Both models forecast the identical out-of-sample timestamps.
3. **Invert transformations before scoring.** Differencing, Box-Cox, or scaling must be undone
   before computing [RMSE, MAE, or MAPE](../theory/metrics.md) so both models are scored on the
   raw target scale.
4. **Equal information constraints.** LightGBM's features must use only observations available
   before forecast step $t$. If SARIMA gets no future exogenous data, LightGBM gets none either.

Once both are scored, [Decisions](../theory/models-decisions.md) covers the baseline check and
deploy criteria that apply regardless of model family.
