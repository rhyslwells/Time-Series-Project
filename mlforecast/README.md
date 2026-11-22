# Half-hourly forecasting demos (marimo)

Learning notebooks for the `mlforecast` package. Run from the project root:

```bash
uv run marimo edit mlforecast/05_mlforecast_basics.py   # edit mode
uv run marimo run  mlforecast/05_mlforecast_basics.py   # app mode (code hidden, widgets only)
```

Start with `05`, which walks through the API step by step. `01`-`04` are applications built on it.

| Notebook | Question it answers |
|---|---|
| `05_mlforecast_basics.py` | How do `MLForecast`, lags, `preprocess`, `cross_validation` and conformal intervals fit together? |
| `06_quantile_vs_conformal.py` | Quantile models or conformal intervals: which gives better-calibrated 80% bands, and why? |
| `07_beat_naive.py` | Does any model beat Naive / SeasonalNaive(48) / SeasonalNaive(336) on the real 15-asset data? (statsforecast baselines vs LightGBM, MASE, per-asset win rate, error by horizon) |
| `01_likelihoods.py` | How likely is demand above/below X at a given time tomorrow? Will the asset export? |
| `02_evaluation.py` | Does ML beat simple baselines? Are the percentiles calibrated? |
| `03_feature_importance.py` | Which inputs does the model rely on? |
| `04_residuals.py` | Where and how does it go wrong (bias, time-of-day, memory, calibration)? |

`make_long_data.py` generates `data/ev_long.parquet`: 30 heterogeneous EV charge points, 8 weeks (6-week training
window + 2-week holdout), session-like spiky load, with a peak-time shift in ~30% of assets from day 42 to test retraining.
Independent of `src/data/`. Run `uv run python mlforecast/make_long_data.py`.

`common.py` builds one `MLForecast` object (`make_forecaster`) holding three tiers of models (seasonal naive,
slot profile, quantile LightGBM), each at 7 quantiles. Backtests use `cross_validation`, tomorrow's forecast uses
`fit` + `predict`, and the baselines are tiny sklearn-style estimators, showing that mlforecast accepts any model
with `fit`/`predict`. All notebooks use synthetic data.
