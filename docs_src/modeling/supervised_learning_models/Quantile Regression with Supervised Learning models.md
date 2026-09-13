### Phase 2: Quantile Regression with Supervised Learning models


Standalone note to be integrated to docs 

Uwse lightgbm as an example of a supervised learning model that can be used for quantile regression..

#### Uncertainty quantification
- What is quantile regression and how does it differ from predicting mean + residual variance?
- Can you explain how to use tree-based models to estimate prediction intervals (P10, P50, P90)? Can you predict the distribution of the target variable using quantile regression with LightGBM? How do you interpret the results?
- What are the pros and cons of training one multi-objective model vs. three separate quantile models?



#### Model-specific guidance
- How does LightGBM's quantile loss function work? Are there gotchas with hyperparameters?
- How should I validate quantile regression models? Is standard RMSE sufficient, or do I need calibration tests?

---