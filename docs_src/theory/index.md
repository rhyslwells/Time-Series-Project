# Theory

Math, model behavior, and interpretation for the forecasting framework. Code usage lives in [Coding](../coding/index.md); this section covers the reasoning behind it.

- [Metrics](metrics.md) — MAE, RMSE, MAPE, PI coverage: formulas and expected ranges by asset type
- [Models](models.md) — SARIMA, Exponential Smoothing, LightGBM: the math and when to use each
- [Diagnostics](diagnostics.md) — how to read the four diagnostic plots, good vs bad cases
- [Decisions](decisions.md) — model selection, retrain triggers, tuning guide, deploy checklist

## Find an answer fast

| Question | Page |
|---|---|
| "What's a good MAE/RMSE for my asset type?" | [Metrics](metrics.md) |
| "Why does SARIMA use these parameters?" | [Models](models.md) |
| "How do I read this diagnostic plot?" | [Diagnostics](diagnostics.md) |
| "Which model should I pick?" | [Decisions](decisions.md) |
| "When should I retrain?" | [Decisions](decisions.md#when-to-retrain) |
| "Is this forecast ready to deploy?" | [Decisions](decisions.md#production-ready-checklist) |
| "How do I call the framework code?" | [Coding](../coding/framework_usage.md) |
