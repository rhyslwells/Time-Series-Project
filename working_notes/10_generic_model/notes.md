# A generic "good enough" forecasting model (no per-asset understanding)

Goal: one ML approach that gives a decent forecast for any asset, without hand-tuning per asset.
Status: exploration. Section 1 comes from our own run. Section 2 is from web search snippets only
(papers and docs not read in full), so treat it as leads to check, not conclusions.

## 1. What `mlforecast/07_beat_naive.py` showed

Setup: 15 synthetic assets (EV charging + solar/battery), 14 days, 30-min data, day-ahead (h=48),
6 rolling-origin windows, scored with MAE / MASE (scale-free) / per-asset win rate. Data is synthetic
(see `mlforecast/README.md`), so absolute numbers do not transfer to real assets.

Mean over 15 assets (from the saved session output):

| model | MAE | MASE | skill vs best baseline |
|---|---|---|---|
| LGB_raw (global LightGBM, lags 48/96/144, hour + dayofweek) | 0.185 | 0.60 | +21.5% |
| Ens_SN336 + LGB_raw | 0.192 | 0.63 | +18.5% |
| SeasonalNaive(336) (best baseline) | 0.236 | 0.77 | 0 |
| LGB_diff48 | 0.249 | 0.79 | -5.6% |
| LGB_diff48 + LocalStandardScaler | 0.256 | 0.81 | -8.5% |
| SeasonalWindowAverage(48, 3) | 0.329 | 1.03 | -39.6% |
| SeasonalNaive(48) | 0.330 | 1.04 | -40.0% |
| Ens_generic (ETS + SN48 + LGB_diff48) | 0.477 | 1.50 | -102% |
| Naive / AutoETS(48) | 1.16 | 3.7 | -394% |

Lessons:

1. **The baseline is the bar, and it is seasonal naive, not naive.** Plain Naive and AutoETS with
   season 48 were ~3.5x worse than SeasonalNaive(48). Always include seasonal naive at every plausible period (day, week).
2. **Weekly seasonality mattered.** SN336 beat SN48 by 28%. A "generic" model must be allowed to see the
   weekly cycle, either through lag 336 or through `dayofweek` / `hour` features.
3. **A single global LightGBM across all assets won**, with `static_features` marking the asset type and no
   per-asset tuning. This is the closest thing in the notebook to the "apply broadly" goal.
4. **Day-ahead-safe lags only (>= horizon).** Avoids recursive forecasting and error compounding.
5. **The differenced / standardised target variants lost to the raw target here.** The usual advice
   (differencing helps trees on level-shifting series) did not hold on this stable synthetic data. Do not
   assume it; test it. It may matter on real assets with level changes.
6. **Naive ensembles can hurt.** The generic "ETS + SN48 + LGB" ensemble was worse than its best member
   because ETS was terrible. Ensemble only members that individually beat the baseline.
7. **Caveats:** 14 days and 6 windows is small, the winner was chosen on the same windows it was scored
   on (optimistic), and the data has no real-world weather / tariff / outage noise.

Not yet done: the per-asset win rate and per-type MASE tables exist in the notebook; I did not pull them
into this note. Check whether LGB_raw loses badly on any single asset.

## 2. Packages worth evaluating (leads from web search, unverified)

| Package | What it is | Why it fits "no asset understanding" |
|---|---|---|
| [statsforecast](https://github.com/Nixtla/statsforecast) | Fast local statistical models: AutoARIMA, AutoETS, AutoCES, AutoTheta, seasonal naive | Auto-fitted per series, cheap, good baselines. Already in use. |
| [mlforecast](https://github.com/Nixtla/mlforecast) | Global tree/ML models with lag, rolling and date features, cross-validation, conformal intervals | One model across many series. Already in use (our winner). |
| AutoGluon-TimeSeries | AutoML: ensembles statistical, ML and deep models with a few lines of code, probabilistic output | The most "automated" option. Reported strong on a 29-dataset benchmark (arXiv 2308.05566 and 2407.16445). Not yet tried here. |
| Chronos (Amazon, also runs inside AutoGluon) | Pretrained foundation model, zero-shot | No training or features. Needs a check on half-hourly energy data. Not yet tried here. |
| TimeGPT (Nixtla) | Hosted foundation model API | Closed, API key, and it sends data to a third party. Likely a non-starter for asset data without approval. |
| Darts / sktime | General toolkits wrapping many model families | More manual control, less automation. Lower priority. |

Search findings worth remembering (from snippets, not verified against the papers):

- On M4, no single local statistical method won everywhere (AutoTheta, AutoETS, AutoARIMA each led some
  frequencies, naive led daily in one third-party run). Argument for cross-validated model selection or
  combination instead of picking a model up front.
- The top M4 entries were ensembles/combinations. The ensemble lesson in 1.6 above is the caveat: members must be good.
- Locality vs globality (arXiv 2008.00444) is the relevant theory for why one global model across assets can beat
  per-asset fits when each series is short.

## 3. Candidate recipe (for review, not implemented)

1. Seasonal naive at every plausible period as the floor (SN48, SN336).
2. Global LightGBM via mlforecast: lags >= horizon, rolling means, hour + dayofweek, asset-type static feature.
3. Add AutoGluon-TimeSeries and Chronos as zero-shot / automated comparators on the same rolling-origin windows.
4. Score everything with MASE and per-asset win rate. An asset only gets the ML model if it beats the best
   baseline in backtest, otherwise fall back to seasonal naive. This fallback is what removes the need to understand each asset.
5. Re-run as history grows beyond 14 days; conclusions here will move.

## 3b. Operating constraints (stated later) and what they change

Constraints: no trend; daily and possibly weekly seasonality; many assets of the same type (e.g. EV charge
points); per-asset forecasts wanted; train on at most 6 weeks of 30-min data (about 2,000 rows per asset);
retrain every 6 weeks to pick up new behaviour.

1. **Lag 336 becomes usable.** With 14 days it was dropped because it deleted most training rows. With 6 weeks
   it leaves about 1,700 rows per asset, so the weekly lag can go into the global model.
2. **No trend means no differencing.** Consistent with the raw target winning in section 1. Skip
   `Differences` and start with the raw target.
3. **Pool by asset type, forecast per asset.** One global model per asset type, trained on all assets of that
   type, gives each asset hundreds of times more training signal than a local fit on 2,000 rows. Per-asset
   output comes from per-asset inputs, not per-asset models: the asset's own lags and rolling means,
   `LocalStandardScaler` (per-series scale), and optionally `unique_id` as a categorical or a per-asset profile
   feature (mean by slot-of-week computed on training data only). New assets with little history also benefit.
4. **Between retrains, update the features, not the model.** Each day, predict with the latest history so lags
   are fresh. The model is only refit every 6 weeks. Between refits, track rolling error versus seasonal naive
   per asset and allow an early retrain if the model falls behind.
5. **Capturing new behaviour with a 6-week window.** Train on the most recent 6 weeks only, or train on more
   and down-weight old data (`sample_weight`). Test both if longer history ever exists.
6. **Validation has to cover a full weekly cycle.** Hold out the last 1 to 2 weeks as rolling-origin windows
   (h=48, step 48) and train on the rest. With only 6 weeks, fewer than 7 validation days cannot show whether
   weekly seasonality is captured.
7. **Local statistical models are a fair comparator now.** The earlier AutoETS(48) was flat like Naive on 14
   days. With 6 weeks, test statsforecast `MSTL` (daily 48 + weekly 336) per asset. Needs a check that it fits.
8. **Foundation models fit this regime well.** Chronos needs no training, and 6 weeks is a reasonable context.
   AutoGluon retraining only every 6 weeks makes its compute cost acceptable. Both are worth benchmarking
   against the pooled LightGBM.
9. **EV charging is spiky and often zero.** Point-forecast MAE can hide this. Check calibration of quantile
   or conformal intervals as well (see `mlforecast/06_quantile_vs_conformal.py`).

Revised recipe: per asset type, seasonal naive (48, 336) as the floor, pooled LightGBM with lags 48/96/144/336
and a per-asset scale, MSTL and Chronos as comparators, all scored on the same 1 to 2 week rolling-origin
holdout, with per-asset fallback to seasonal naive when the ML model does not beat it.

## 3c. Longer synthetic data and AutoGluon (status)

- Data: `mlforecast/make_long_data.py` -> `mlforecast/data/ev_long.parquet` (30 EV assets, 56 days, per-asset
  scale / peak timing / weekend ratio / noise, session-like spikes, no trend, 30% of assets shift their evening
  peak by 2.5h from day 42). `src/data/` is untouched. The 14-day generator gave every asset of a type the same
  pattern, so it could not test per-asset forecasting.
- AutoGluon (`autogluon.timeseries`): dry-run install resolves on Python 3.13 but would downgrade
  `statsforecast` (2.1.1 -> 2.0.1) and `utilsforecast` (0.2.17 -> 0.2.11) and add torch, transformers and
  others. Not installed. Plan: run it in a separate environment so the project env is not affected.
  Not yet run, so there are no AutoGluon results.

## Open questions

- Does the LGB_raw win survive on real (non-synthetic) assets and longer history?
- Does AutoGluon or Chronos beat the global LightGBM on this data, and at what compute / dependency cost?
- Is a per-asset fallback rule robust, or does choosing on the backtest just overfit?
- New model types are a "ask before changing" item in CLAUDE.md, so confirm before adding AutoGluon or Chronos to `src/`.

## Sources

- [temporalcv M4 benchmark](https://temporalcv.readthedocs.io/en/latest/benchmarks.html)
- [Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality](https://arxiv.org/pdf/2008.00444)
- [Nixtla: automatic model selection with StatsForecast](https://www.nixtla.io/blog/statsforecast-automatic-model-selection.md)
- [Weighted Ensemble of Statistical Models](https://arxiv.org/pdf/1811.07761)
- [Can time series forecasting be automated? A benchmark and analysis](https://arxiv.org/html/2407.16445v2)
- [AutoGluon-TimeSeries paper](https://arxiv.org/abs/2308.05566)
