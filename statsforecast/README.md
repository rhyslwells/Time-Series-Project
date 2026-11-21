# statsforecast exploration (marimo)

Exploratory notebooks for the `statsforecast` package, the statistical-model counterpart to `../mlforecast`.
Same synthetic half-hourly net-load data, same day-ahead framing (`h=48`, `season_length=48`). Run from the project root:

```bash
uv run marimo edit statsforecast/01_statsforecast_basics.py
```

| Notebook | Question it answers |
|---|---|
| `01_statsforecast_basics.py` | How do `StatsForecast`, the model zoo and `forecast` work, and how does it differ from mlforecast? |
| `02_model_comparison.py` | Do ETS / MSTL / ARIMA beat the seasonal-naive and seasonal-average baselines in a rolling backtest? |
| `03_intervals.py` | Parametric vs conformal 80% intervals: which is better calibrated? |

`common.py` holds the data generator, the model list (`make_models`) and `metrics_table`. Backtests take about a
minute because every window refits every model. `AutoARIMA` is only used as MSTL's trend model, since fitting it
directly at `season_length=48` is impractical.
