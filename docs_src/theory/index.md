# Theory

Math, model behavior, and interpretation for the forecasting framework. Code usage lives in [Coding](../coding/index.md); this section covers the reasoning behind it.

- [Architecture](architecture.md) — the layer model, each layer's inputs and outputs, the forecast contract, and which layers exist today
- [Metrics](metrics.md) — MAE, RMSE, MAPE, MASE, PI coverage: formulas, expected ranges, and why every number is read against a seasonal-naive baseline at a stated horizon
- [Models](models.md) — SARIMA, Exponential Smoothing, LightGBM: the math and when to use each
- [Diagnostics](diagnostics.md) — how to read the four diagnostic plots, good vs bad cases
- [Rolling-origin evaluation](rolling-origin-evaluation.md) — why one train/test split isn't enough, and how error-by-horizon confirms a model choice
- [Decisions](models-decisions.md) — model selection, retrain triggers, tuning guide, deploy checklist
- [Forecast products](forecast_products.md) — turning a forecast into derived signals: uncertainty, event probability, anomaly detection, asset classification, portfolio and flexibility forecasting (design, not yet built)

## Find an answer fast

| Question | Page |
|---|---|
| "How do the layers fit together / what's built?" | [Architecture](architecture.md) |
| "What's a good MAE/RMSE for my asset type?" | [Metrics](metrics.md) — but read it against a baseline, not an absolute threshold |
| "How do I compare across assets of different sizes?" | [Metrics — MASE](metrics.md#mase-mean-absolute-scaled-error) |
| "Why don't the metrics mean anything without a baseline?" | [Metrics](metrics.md#read-metrics-against-a-baseline) |
| "Why does SARIMA use these parameters?" | [Models](models.md) |
| "How do I read this diagnostic plot?" | [Diagnostics](diagnostics.md) |
| "Which model should I pick?" | [Decisions](models-decisions.md) |
| "Is one train/test split enough to trust a model?" | [Rolling-origin evaluation](rolling-origin-evaluation.md) |
| "How far ahead can I trust this forecast?" | [Rolling-origin evaluation](rolling-origin-evaluation.md#what-it-answers-that-the-single-split-doesnt) |
| "When should I retrain?" | [Decisions](models-decisions.md#when-to-retrain) |
| "Is this forecast ready to deploy?" | [Decisions](models-decisions.md#production-ready-checklist) |
| "How do I turn a forecast into a flexibility envelope?" | [Forecast Products](forecast_products.md#flexibility-forecasting) |
| "How do I classify assets by behaviour?" | [Forecast Products](forecast_products.md#cross-asset-comparison-and-asset-classification) |
| "How do I call the framework code?" | [Coding](../coding/framework_usage.md) |
