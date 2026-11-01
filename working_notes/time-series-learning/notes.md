# Time series forecasting: deeper learning map

Purpose: a focused list of areas to study beyond what `docs_src/` already covers, with the questions each one should let you answer. References are from memory, so confirm titles, editions and years when you look them up.

## What docs_src already covers (so you can skip the basics)

- Baselines, SARIMA, Holt-Winters, LightGBM, and when to choose each
- MAE / RMSE / MAPE / MASE, PI coverage, interval scoring
- Chronological split, expanding window, rolling-origin evaluation
- Residual diagnostics, quantile regression with LightGBM, forecast products

The five focus areas below are mostly about the theory under those pieces. Areas 6 to 10 are parked at the end for later.

## How to use this file

For each focus area: read the "why we care" paragraph, try to answer the questions in your own words before and after reading, then do the project exercise. If you cannot answer a question, that is the gap to read about. Write answers in "Notes while reading" at the bottom, or in a new file per area.

## Suggested reading order

1. Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., free at otexts.com/fpp3; a Python edition is also on otexts.com). Read it once end to end. It covers most of areas 1, 3 and 5 and part of 4.
2. Gneiting & Raftery (2007) on proper scoring rules, then Gneiting, Balabdaoui & Raftery (2007) on calibration and sharpness. These are the core of area 2.
3. Hewamalage, Bergmeir & Bandara (2023), "Forecast evaluation for data scientists: common pitfalls and best practices". Short and directly relevant to area 1.
4. Hong et al. (2016), "Probabilistic electric load forecasting: a tutorial review". Ties areas 2 and 4 to your energy domain.
5. Then dig into the specific references under each area.

---

## Focus area 1: Forecast evaluation theory

### Why we care

Every decision in this project (which model, which hyperparameters, when to retrain) rests on an error estimate. With 14 days of data, 15 assets and a handful of rolling origins, those estimates are noisy, and a model that looks 3% better on one split may just be luckier. Experts treat evaluation as the main source of risk in forecasting work: a flawed evaluation produces confident wrong conclusions, and no model improvement can compensate. The project's own rolling-origin notes already point this way; this area is the statistical theory behind them.

### Questions to investigate

- What does a rolling-origin error estimate actually estimate: performance of this fitted model, or of the training procedure on future data? Does the answer change how you read the results?
- How many origins and how long a test window do you need before differences between models are distinguishable from noise? Can you compute a rough standard error for MAE across origins?
- Origins overlap and errors are autocorrelated across horizons. How does that break the usual assumption that fold errors are independent?
- What does the Diebold-Mariano test assume, and when does it fail (small samples, nested models, heavily autocorrelated loss differentials)? What is the Model Confidence Set for, and how does it differ from a pairwise test?
- Why does MAPE explode or mislead for solar at night and for EV charging near zero? What do MASE and RMSSE fix, and what does MASE still hide?
- How should errors be aggregated across 15 assets: pooled, averaged per asset, or scaled per asset first? Which choice lets one large asset dominate the ranking?
- Does a model's ranking change with the horizon, the time of day, or the asset type? Is a single headline metric hiding that?
- Where can leakage enter this project? Check lags, rolling windows, and especially the daily metrics broadcast across each day's 48 rows in `metering_data_with_features.parquet`: are any of them computed using the day being forecast?
- Is tuning on `y_val` and reporting on `y_test` enough protection from selection bias when many configurations are tried? What happens to the test estimate if you peek twice?
- What is the difference between in-sample fit, one-step-ahead error and true multi-step out-of-sample error, and which one does each current metric report?

### Why this matters for the project

If the answers show your current evaluation cannot separate models, the correct conclusion is that the simpler model wins by default. That is a real finding, not a failure.

### References

- Hyndman & Koehler (2006), "Another look at measures of forecast accuracy" (introduces MASE)
- Diebold & Mariano (1995), "Comparing predictive accuracy"
- Hansen, Lunde & Nason (2011), "The Model Confidence Set"
- Bergmeir & Benitez (2012), "On the use of cross-validation for time series predictor evaluation"
- Cerqueira, Torgo & Mozetic (2020), "Evaluating time series forecasting models: an empirical study on performance estimation methods"
- Hewamalage, Bergmeir & Bandara (2023), "Forecast evaluation for data scientists"
- FPP3, chapters on evaluating forecast accuracy and time series cross-validation

### Exercise

Take the existing rolling-origin results for SeasonalWindowAverage, SARIMA, ExponentialSmoothing and LightGBM. Compute per-origin loss differences against the seasonal-naive baseline, apply a Diebold-Mariano test, and report which pairwise differences are distinguishable from noise.

---

## Focus area 2: Probabilistic forecasting and uncertainty

### Why we care

The forecast contract carries `lower`, `upper` and `uncertainty_width`, and the planned downstream layers (flexibility, optimisation) consume uncertainty as a decision input, not just a point value. A bidding or dispatch decision depends on how wrong the forecast could be, so interval quality can matter more than point accuracy. PI coverage alone is a weak check: an interval can hit 80% coverage by being enormous, and a model can be well calibrated on average and badly calibrated at the peaks where it matters.

### Questions to investigate

- What makes a scoring rule "proper", and why does that matter? Show with a small example that PI coverage alone can be gamed.
- What is the pinball (quantile) loss, how does it relate to CRPS, and what does the interval score (Winkler) penalise that coverage ignores?
- Calibration versus sharpness: how do you plot and read a PIT histogram or a reliability diagram for your forecasts? What does a U-shape or hump tell you?
- Is coverage the same at every time of day and for every asset, or only on average? Where does it fail (ramp periods, charging peaks, sunrise and sunset)?
- Where does interval width come from: parameter uncertainty, model uncertainty, irreducible noise? Which of these does each of your models include, and which does it ignore (SARIMA intervals assume Gaussian, constant-variance errors)?
- Is the residual variance constant, or does it scale with the level or time of day? What would heteroscedastic intervals look like for EV charging?
- Quantile crossing: how often do your LightGBM P10, P50, P90 outputs cross, and what are the standard fixes (rearrangement, monotone constraints)?
- What does conformal prediction guarantee, and what assumption does it need (exchangeability)? Why does that assumption fail for time series, and how do adaptive and rolling variants repair it?
- Compare quantile regression intervals, mean-plus-residual-variance intervals and conformal intervals on the same model. Which gives the sharpest intervals at equal coverage?
- How should a downstream consumer use a distribution rather than an interval (sampling scenarios, event probabilities)? What does that require of the forecaster?

### Why this matters for the project

Your forecast products notes describe event probability, scenario generation and uncertainty as a decision input. Each of those is only as good as the calibration of the underlying distribution, so scoring and checking that calibration is the groundwork for the whole upper half of the layer stack.

### References

- Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and estimation"
- Gneiting, Balabdaoui & Raftery (2007), "Probabilistic forecasts, calibration and sharpness"
- Koenker, *Quantile Regression*
- Angelopoulos & Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification"
- Romano, Patterson & Candes (2019), "Conformalized quantile regression"
- Gibbs & Candes (2021), "Adaptive conformal inference under distribution shift"
- Hong et al. (2016), "Probabilistic electric load forecasting: a tutorial review"

### Exercise

Add pinball loss, interval score and a reliability diagram to a notebook (not `src/`) and check whether the ranking of models changes versus PI coverage alone. Then wrap the LightGBM point forecast in conformal intervals and compare to the quantile-regression intervals at equal coverage.

---

## Focus area 3: Classical model theory

### Why we care

SARIMA and Holt-Winters are currently black boxes with tuned parameters. When they fail, for example on the weekly cycle that no default configuration captures, you need to know why: is it a stationarity problem, a seasonal-period limit, a trend specification, or a likelihood that cannot represent two seasonalities at once? Classical theory also explains what the supervised models are quietly replacing (differencing, lags, seasonal terms), which is how you judge whether a LightGBM feature set is complete. These models also remain the strongest baselines on short, clean, seasonal series like yours, so you need to be able to tune them properly before concluding anything beats them.

### Questions to investigate

- What does stationarity actually require, and why do ARIMA models need it? What do ADF and KPSS test, and why can they disagree?
- How do you read ACF and PACF plots to propose orders, and what does the Ljung-Box test on residuals rule in or out?
- Why is ARIMA a special case of a state space model, and what do you gain from the state space view (missing data, likelihood, forecast variance, Kalman filtering)?
- What is the ETS taxonomy (error, trend, seasonal; additive versus multiplicative), and how does AIC-based selection work over it? When is multiplicative seasonality right, for example solar that scales with irradiance?
- Why do statsmodels SARIMA and ExponentialSmoothing struggle with a seasonal period of 48 (and 336 for a weekly cycle)? What do Fourier terms, TBATS, MSTL and dynamic harmonic regression do instead?
- What is regression with ARIMA errors, and how would you bring exogenous drivers (calendar, weather, behaviour metrics) into a classical model correctly?
- Damped trend: why does it often win, and what does it say about extrapolating trends?
- How are prediction intervals derived analytically for ARIMA and ETS, and which assumptions do they depend on? How does that link to area 2?
- Which of these models' implicit assumptions does the synthetic data satisfy or violate, and so how much of the model ranking is an artefact of the generator?
- With only 14 days of data, how many seasonal cycles can a weekly pattern be estimated from? What is the minimum history needed for each model?

### Why this matters for the project

The project's current conclusions about SARIMA versus LightGBM may partly reflect configuration limits rather than model class limits. This area tells you whether a better-specified classical model would change the answer.

### References

- Hyndman & Athanasopoulos, FPP3 (ARIMA, ETS, dynamic regression, complex seasonality chapters)
- Hyndman, Koehler, Ord & Snyder, *Forecasting with Exponential Smoothing: The State Space Approach*
- Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*
- Shumway & Stoffer, *Time Series Analysis and Its Applications*
- Durbin & Koopman, *Time Series Analysis by State Space Methods*
- Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter*
- Hamilton, *Time Series Analysis* (a reference rather than a read-through)
- De Livera, Hyndman & Snyder (2011), TBATS paper

### Exercise

Reproduce the FPP3 complex-seasonality approach on one asset: decompose with MSTL, model the remainder with ETS or ARIMA, and add Fourier terms for the weekly cycle. Compare to the current SARIMA and the seasonal-naive baseline on MASE and a Diebold-Mariano test.

---

## Focus area 4: Machine learning for forecasting

### Why we care

LightGBM is the main supervised model and the vehicle for quantile forecasts, but supervised learning on time series has its own failure modes that do not exist in the tabular setting: leakage through features, error accumulation across horizons, no extrapolation, and the choice between one model per series and one pooled model. Most disappointing ML forecasting results come from these design choices rather than from the learning algorithm. There is also a strong literature showing simple baselines are hard to beat, so knowing what to expect keeps the comparison honest.

### Questions to investigate

- Direct, recursive and multi-output strategies for multi-step horizons: what errors does each accumulate, and which does the current feature table implicitly assume?
- Which features are known in the future (calendar, planned events) and which are only known in the past (lags, rolling stats)? How does each constrain the forecast horizon, and is the current feature table consistent about this?
- Global versus local models: when does one LightGBM across all 15 assets beat 15 separate models? What does it need (asset id feature, per-series scaling)?
- Why can tree models not extrapolate a trend or level shift beyond the training range, and what do detrending, differencing the target, or hybrid residual models do about it?
- Target transformation: should you forecast the level, the difference from last week, the ratio to a seasonal-naive forecast, or (for solar) a clear-sky index? How does each change what the trees must learn?
- How much of LightGBM's accuracy comes from lag features that simply copy the seasonal-naive forecast? What is the gain over that baseline, and is it meaningful (area 1)?
- Hyperparameters for time series: which matter (learning rate, trees, min child samples, feature fraction), and how do you tune them without leaking the future (nested or rolling validation)?
- Feature importance and SHAP on time series: how do you tell a causal driver from a proxy for time of day?
- With 10,080 rows, how data-hungry is the model? Plot a learning curve against the training window length.
- Landscape awareness: DeepAR, N-BEATS / N-HiTS, Temporal Fusion Transformer, PatchTST, and pretrained foundation models (Chronos, TimesFM, Moirai). What do they claim, what do the M4 / M5 results and the DLinear paper say about simple baselines, and where might a foundation model be worth a zero-shot comparison?

### Why this matters for the project

The architecture treats models as swappable. This area decides what to put in the swap slot and how to feed it, and it protects the comparison from the classic mistake of crediting the algorithm for what the features supplied.

### References

- Ben Taieb et al. (2012), "A review and comparison of strategies for multi-step ahead forecasting"
- Montero-Manso & Hyndman (2021), "Principles and algorithms for forecasting groups of time series: locality and globality"
- Makridakis et al., M4 and M5 competition papers (International Journal of Forecasting)
- Januschowski et al. (2020), "Criteria for classifying forecasting methods"
- Zeng et al. (2023), "Are transformers effective for time series forecasting?"
- Salinas et al. (2020), DeepAR; Lim et al. (2021), Temporal Fusion Transformer
- Petropoulos et al. (2022), "Forecasting: theory and practice" (International Journal of Forecasting), a broad survey of all of the above

### Exercise

Run an ablation on the LightGBM feature table: remove feature groups one at a time (lags, rolling, calendar, daily behaviour metrics) and report the change in MASE with a Diebold-Mariano test. Then compare a global model with an asset-id feature against per-asset models.

---

## Focus area 5: Forecast combination and model selection

### Why we care

Averaging forecasts from several decent models routinely beats the best single model, and it is cheap, robust to a single model's failure mode, and compatible with a swappable-model architecture. It also changes how to read the comparison work: if SARIMA, Holt-Winters and LightGBM make different errors, the question shifts from "which model wins" to "which set combines best". Combination also gives a principled way to build uncertainty from model disagreement, which links back to area 2. The related question of selection (information criteria versus out-of-sample error) governs how much to trust any automated "best model" pick.

### Questions to investigate

- Why does a simple average of forecasts so often beat sophisticated weighting (the forecast combination puzzle)? What does that say about estimating weights from short histories?
- When do two models' errors combine well? Compute the error correlation between your four models: which pair is most complementary?
- What is the gain from an equal-weight average of SARIMA, Holt-Winters and LightGBM versus the best single model, in MASE and in interval score?
- How do you combine probabilistic forecasts: average the quantiles (Vincentization) or average the distributions (mixture)? What does each do to sharpness and calibration?
- Stacking and learned weights: how do you fit weights without leaking the future, and how stable are the weights across rolling origins?
- Selection versus combination: how often does the per-origin "best model" change, and what does that instability say about picking one model?
- Information criteria (AIC, AICc, BIC) versus out-of-sample error for model choice: when do they agree, and when is in-sample AIC misleading, especially across model classes where likelihoods are not comparable?
- Does combining inflate the effective degrees of freedom you tuned, so that the evaluation protocol from area 1 must be applied to the combination as a whole?
- How would the contract represent a combined forecast (`model_version` for an ensemble, uncertainty from disagreement plus within-model variance)?

### Why this matters for the project

A combination of the existing models is nearly free given the current framework, and it is a likely candidate for a stronger baseline than any single model. It also tests whether the model-comparison machinery is really architecture-agnostic, since an ensemble should be just another `TSModel`.

### References

- Bates & Granger (1969), "The combination of forecasts"
- Timmermann (2006), "Forecast combinations" (Handbook of Economic Forecasting)
- Claeskens et al. (2016), "The forecast combination puzzle: a simple theoretical explanation"
- Wang, Hyndman, Li & Kang (2023), "Forecast combinations: an over 50-year review"
- FPP3, ensemble and forecast combination sections
- Makridakis et al., M4 competition results (top entries were combinations)

### Exercise

Add an equal-weight ensemble as a `TSModel` subclass in a notebook, run it through `ModelComparison`, and test whether it beats the best single model using the area 1 protocol. Then repeat with quantile averaging for the prediction interval and score with the area 2 metrics.

---

## Parked areas (later)

Brief notes only; expand when the five above are done.

- **Hierarchical and grouped forecasting.** Portfolio forecasting is an aggregation problem: summing 15 independent forecasts is neither coherent-optimal nor gives valid aggregate intervals. Look at bottom-up, top-down and MinT reconciliation, aggregating intervals with cross-asset correlation, and temporal hierarchies. References: Hyndman et al. (2011); Wickramasuriya, Athanasopoulos & Hyndman (2019); FPP3 hierarchical chapter. This is probably the next one to promote given the forecast products plan.
- **Non-stationarity, drift and monitoring.** Structural breaks, drift detectors (CUSUM, Page-Hinkley, ADWIN), retraining policies, forecast error as a control chart. References: Gama et al. (2014); Truong, Oudre & Vayatis (2020).
- **Energy and flexibility specifics.** Load forecasting conventions, GEFCom, EV session models, solar clear-sky index, customer baseline load methods for demand response. References: Hong et al. (2016); Weron (2014).
- **Intermittent and sparse series.** Croston, SBA, TSB, zero-inflated models, metrics for series with many zeros. References: Syntetos & Boylan (2005).
- **Anomaly detection and causal baselines.** Residual-based detection, matrix profile, CausalImpact, synthetic control for flexibility events. References: Brodersen et al. (2015); Schmidl, Wenig & Papenbrock (2022).

## Notes while reading

(Add per-source notes below as you go.)
