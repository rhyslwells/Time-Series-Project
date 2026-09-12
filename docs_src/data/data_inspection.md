# Data inspection

## Why

Forecasting models encode assumptions: stationarity, a seasonal period, additive vs
multiplicative structure, an error distribution, that the data either satisfies or doesn't.
Fitting a model before checking those assumptions means any mismatch only shows up later, as
a bad metric or a model that looks wrong for no obvious reason, with no way to tell whether
the problem is the model choice, its configuration, or the data itself. Inspecting first
turns that guesswork into a decision made from evidence: what seasonal period to configure,
whether to difference or detrend, which model family is even plausible, and what a normal
residual plot should look like once a model is fit.

These checks also make model diagnostics interpretable later. A residual ACF only tells you a
model failed to capture structure if you already know the raw series had structure to
capture in the first place, and that's established here, before any model exists.

These checks are diagnostic, not a preprocessing gate: they surface structure (trend,
seasonality, non-stationarity) so a model choice is informed, but they don't mandate
differencing, detrending, or any other transformation of the raw series. See
[Forecasting approach](../theory/forecasting-approach.md).

## What

Ten checks, each aimed at one modeling assumption:

| Check | Output | What it's for |
|---|---|---|
| Distribution and range | Summary stats, histogram | Sanity-checks scale and units, flags degenerate series (constant, all-zero, heavy-tailed), and shows whether the series crosses zero (rules out metrics like MAPE) |
| Trend / regime changes | Raw series with a rolling-mean overlay | Flags drift or level shifts that violate stationarity assumptions, and signals whether differencing or detrending is needed |
| Time-of-day pattern | Boxplot by time-of-day slot | Surfaces intraday seasonality: informs the seasonal period to configure, or whether a time-of-day feature is needed |
| Day-of-week pattern | Boxplot by weekday | Surfaces weekly seasonality on top of, or instead of, daily seasonality, at a longer period |
| Autocorrelation (ACF) | Correlogram with a 95% white-noise band | Confirms the series has exploitable structure beyond noise, and identifies candidate seasonal periods or MA order |
| Partial autocorrelation (PACF) | Correlogram with a 95% white-noise band | Isolates direct lag dependence from structure already explained by shorter lags: candidate AR order |
| Rolling mean and variance | Rolling mean and rolling variance over time | A visual stationarity check: a flat rolling mean and variance support models that assume stationarity, drift in either argues against them |
| Stationarity tests (ADF, KPSS) | A two-row result table plus a combined conclusion | Replaces the rolling-mean/variance eyeball test with a statistical one. ADF's null is a unit root; KPSS's null is stationarity, so the two disagreeing (rather than just one failing) tells you whether the series needs differencing or detrending |
| Seasonal decomposition (STL) | Observed / trend / seasonal / residual panels | Splits the series into components explicitly, instead of inferring them from ACF bumps |
| Trend and seasonal strength | Two scores in [0, 1] | Puts a number on how much of the series each component (trend, seasonal) explains beyond noise, so "there's seasonality" becomes "seasonal strength is 0.92" |

## How

Each check is a function that takes a polars `DataFrame`, a `value_col` name, and an optional
`timestamp_col` (default `"timestamp"`). The trend and rolling-stats checks also take a
`window`; the ACF/PACF check takes `nlags`; the STL-based checks take a `period` (the number
of samples in one seasonal cycle). None of them assume any prior feature engineering: they
run the same way on a raw series or on any other dataset with one timestamp column and one
numeric field of interest, so the same checks apply regardless of what's being measured.

## Limitations

- These checks establish temporal structure, not data quality. Missing values, timestamp
  validation, and outlier detection or treatment are separate concerns and out of scope here.
- Boxplot-based seasonality checks (time-of-day, day-of-week) need enough repeated cycles to
  be more than anecdotal. A short series limits them to a qualitative read rather than a
  fitted estimate.
- ACF/PACF estimates get noisier as the lag approaches the length of the series. Keep `nlags`
  well under the series length, and treat correlations near that ceiling with suspicion.

---

**Source:** `src/ts_data_checks.py` (`DataInspector`); applied to this project's metering
data in `src/notebooks/data_inspection.py`.
