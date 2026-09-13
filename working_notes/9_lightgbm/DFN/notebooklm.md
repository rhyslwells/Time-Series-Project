## LIGHTGBM Notebook 

I do not feel i have much knowledge about this, but i have books about using ml for timeseries: what are some quetions i can ask to get the most out of this notebook? Using Noteebooklm?

C:\Users\RhysL\Desktop\Time Series Project\working_notes\8_gpt1\LIGHTGBM_PLAN.md


## Strategic NotebookLM Questions

### Phase 1: Supervised Learning Feature Engineering

#### Core concepts
- What is the difference between using lagged values vs. rolling statistics for time series features? When would you use each?
- How do I create lag features for 30-minute energy data without causing look-ahead leakage?


#### Temporal encoding
- How should I encode cyclical time (hour of day, day of week) as features for a tree-based model?
- Why do sin/cos encoding work better than simple integer encoding (0-23 for hours) in ML models?

#### Feature validation
- How do I detect feature leakage in a supervised learning time series model?
- What methods are recommended for selecting which features to keep (vs. using all of them)?

---

### Phase 2: Quantile Regression with LightGBM

#### Uncertainty quantification
- What is quantile regression and how does it differ from predicting mean + residual variance?
- Can you explain how to use tree-based models to estimate prediction intervals (P10, P50, P90)?
- What are the pros and cons of training one multi-objective model vs. three separate quantile models?

#### Model-specific guidance
- How does LightGBM's quantile loss function work? Are there gotchas with hyperparameters?
- Is it common for predicted quantiles to violate ordering (P10 > P50)? How do you fix it?
- How should I validate quantile regression models? Is standard RMSE sufficient, or do I need calibration tests?

---

### Phase 3: Overall Strategy

#### When to use tree models vs. statistical models
- For 30-minute energy load forecasting, when would you choose LightGBM over SARIMA? What are the trade-offs?
- What are common pitfalls when switching from ARIMA to machine learning for time series?
- How do you compare forecasts from different models (SARIMA vs. LightGBM) fairly?

#### Validation & evaluation
- What's the right way to split time series data for training and testing? (e.g., chronological vs. random)
- How do I validate that my prediction intervals are well-calibrated (cover the right % of actuals)?
- Are there evaluation metrics specific to energy load forecasting that I should use?

