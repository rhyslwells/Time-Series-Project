# Forecast Products

The model outputs in [Models](models.md) are inputs to a larger system, not the end goal. This page covers the theory for turning a per-timestep forecast into the derived signals that system needs — see `working_notes/main_idea.md` for the full design rationale.

## The forecast as a distribution

Treat each forecast as a distribution, not a single number: $E[Y_t]$, $P_{10}(Y_t)$, $P_{50}(Y_t)$, $P_{90}(Y_t)$ (or whatever interval the model supports). A point forecast alone answers "what will happen"; the distribution also answers "how sure are we", which is what every signal below is built from.

## Asset-level derived signals

From a forecast series $\hat{y}_t$ over a horizon:

$$E_{\text{day}} = \sum_{t \in \text{day}} \hat{y}_t \qquad P_{\text{day}} = \max_t \hat{y}_t \qquad \Delta_t = \hat{y}_t - \hat{y}_{t-1}$$

Daily energy, daily peak, and stepwise ramp. Ramp is often more operationally relevant than the raw forecast — "expected increase of 2.4 MW between 16:00 and 16:30" is directly actionable in a way "42 kWh at 16:30" is not.

## Uncertainty as a decision input

The interval width is itself a signal:

$$U_t = P_{90,t} - P_{10,t}$$

Large $U_t$ means the forecast shouldn't be trusted at face value for that timestep; small $U_t$ means it can be. This is why a downstream system should consume `expected`, `conservative` (e.g. $P_{10}$), and `uncertainty` as three separate numbers rather than collapsing to one — it lets the optimisation layer decide how much of the uncertainty it wants to absorb, rather than baking that choice into the forecast.

## Event probability

Once the forecast is a distribution, questions like "will this asset exceed 5 MW?" become $P(Y_t > 5)$ rather than a hard yes/no on the point forecast. This generalises to any threshold — a battery reaching a target SOC, consumption falling below a floor, a violation of an operational limit — and is usually more useful downstream than the raw forecast value, since it's already framed as a decision input.

## Expected behaviour and schedule

Applying threshold/event logic across a horizon turns a forecast into an expected schedule: expected start/shutdown, expected peak/minimum, expected sustained high consumption or export, expected regime change. E.g. "expected operation 06:00-08:30 and 17:00-20:00" for a pump, or "expected generation peak 12:30-13:00" for solar.

## Baseline and response estimation

For demand response, the forecast (made *without* knowledge of the intervention) serves as the counterfactual baseline:

$$\text{Response} = \text{Baseline forecast} - \text{Observed metering}$$

e.g. expected 4.2 MW, observed 2.8 MW → estimated response 1.4 MW. This is how forecast quality translates directly into measurement quality for anything being evaluated against a baseline.

## Anomaly detection via residuals

The residual $e_t = y_t - \hat{y}_t$ (see [Diagnostics](diagnostics.md) for what it says about model fit) can also be read the other way: as a monitor on the *asset*, once the model itself is trusted. Unusually large or persistent residuals, or a step change in the residual distribution, can indicate an asset going offline, a meter fault, or a genuine behaviour change — a forecast of 4.8 MW against an actual of 0.2 MW is as likely to be a monitoring signal as a forecasting failure.

## Model health and concept drift

Beyond the aggregate MAE/RMSE in [Metrics](metrics.md), error is worth tracking segmented — by day of week, season, weather, or asset operating regime. A model that's accurate Monday-Friday but poor on weekends is telling you something about the model *and* the asset (a regime that regime it doesn't understand), not just producing a lower average score. This segmented view is what should trigger the retrain/switch decisions in [Decisions](decisions.md), rather than a single rolling aggregate.

## Cross-asset comparison and asset classification

With many asset/model pairs, error metrics become a table (asset × model × MAE/RMSE/bias/coverage) that raises a meta-question: what characteristics of an asset predict which model will work best for it? That points toward a behavioural fingerprint per asset — mean load, variance, coefficient of variation, autocorrelation, seasonality strength, intermittency, ramp frequency, peak-to-average ratio, plus the asset's own forecast error/uncertainty history — used to cluster assets and route each cluster to the model class suited to it, rather than grid-searching every asset individually. This is the theory behind `behavioral_fingerprints.parquet` in the data pipeline.

## Portfolio forecasting

Aggregating per-asset forecasts gives a portfolio forecast, $\hat{Y}_t = \sum_i \hat{Y}_{i,t}$, and the aggregate's expectation is just the sum of expectations. Its *uncertainty* is not: it depends on the correlation between assets' errors, not the sum of their individual uncertainties. Where asset errors aren't perfectly correlated, the portfolio forecast is more reliable than any individual asset forecast — which is the main reason aggregation is valuable for a multi-asset system, beyond convenience.

## Flexibility forecasting

Reframes "what will the asset do" as "how much flexibility will it probably have":

$$F_t = [F_t^{\min}, F_t^{\max}]$$

— potential reduction and increase around the expected operating point (for a battery: expected SOC plus available charge/discharge capacity). This is a closer match to what an optimisation layer actually consumes than a point forecast, since it's already expressed as a range of feasible actions rather than a single predicted value.

## Scenario generation

An alternative to a single point-plus-interval forecast is producing several plausible full trajectories (scenario 1: 4.2 MW, scenario 2: 4.7 MW, ...) rather than one path with bounds around it. A downstream optimiser can then ask "what performs well across most plausible futures" instead of optimising against a single best-guess path — this is the bridge from point/interval forecasting toward stochastic optimisation, and becomes more valuable as more of the flexibility/portfolio layers above are built out.
