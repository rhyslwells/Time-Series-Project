#forecasting #timeseries #ml #energy #optimisation #

If you treat the forecast as a first-class data product rather than simply "the next five days of meter values", there is quite a lot you can derive from it.

A useful mental model is:

> **Metering history → forecast distribution → derived signals → asset-level decisions → portfolio-level decisions**

The forecast itself becomes an input to a larger analytical system.

## 1. The basic forecast

For an asset with 30-minute metering data, you might produce:

| Timestamp    | Actual | Forecast | Lower | Upper |
| ------------ | -----: | -------: | ----: | ----: |
| Monday 09:00 | 42 kWh |       40 |    34 |    47 |
| Monday 09:30 | 38 kWh |       39 |    33 |    46 |
| Monday 10:00 | 35 kWh |       36 |    29 |    44 |

I would strongly recommend thinking in terms of a **forecast distribution**, rather than only a point forecast.

You could have:

* $E[Y_t]$: expected metering value
* $P_{10}(Y_t)$
* $P_{50}(Y_t)$
* $P_{90}(Y_t)$

or some other prediction intervals.

That immediately gives you information about both **expected behaviour** and **uncertainty**.

---

# 2. Derive an expected asset profile

The most obvious derived product is a forecast profile:

> "What do we expect this asset to do over the next 5–7 days?"

But you can aggregate this in useful ways.

### Daily energy

For example:

$$
E_{\text{day}} = \sum_{t \in \text{day}} \hat{y}_t
$$

You could therefore say:

> Asset A is expected to consume 1.8 MWh tomorrow.

Or:

> Asset A is expected to export 0.7 MWh tomorrow.

### Daily peak

$$
P_{\text{day}} = \max_t \hat{y}_t
$$

This can be more useful operationally than total energy.

### Expected ramp

You can calculate:

$$
\Delta_t = \hat{y}*t - \hat{y}*{t-1}
$$

and identify periods where the asset is expected to ramp rapidly.

For example:

> Expected increase of 2.4 MW between 16:00 and 16:30.

That starts becoming useful for trading and system constraints.

---

# 3. Forecast uncertainty

This is one of the most valuable things you can derive.

Suppose your forecast says:

$$
\hat{y}_{t} = 5.0\text{ MW}
$$

with a prediction interval:

$$
[3.5, 6.5]\text{ MW}
$$

You can derive a measure of **forecast confidence**.

For example:

$$
U_t = P_{90,t} - P_{10,t}
$$

A large $U_t$ means:

> "We don't know what this asset is going to do very reliably at this point."

A small $U_t$ means:

> "We have high confidence in this forecast."

This can feed directly into trading decisions.

For example, rather than:

> Forecast available volume = 5 MW

you could have:

> Expected volume = 5 MW
> Conservative volume = 3.5 MW
> Forecast uncertainty = 3 MW

That gives your optimisation system something much more useful to work with.

---

# 4. Forecast probability of an event

Once you have a forecast distribution, you can ask questions such as:

> What is the probability that the asset exceeds 5 MW?

$$
P(Y_t > 5)
$$

Or:

> What is the probability that consumption falls below 1 MW?

Or:

> What is the probability that the battery reaches 80% SOC tomorrow?

These are often more useful than the raw forecast.

You could create derived signals such as:

| Signal            | Meaning                             |
| ----------------- | ----------------------------------- |
| $P(Y > 5)$        | Probability of exceeding 5 MW       |
| $P(Y < 1)$        | Probability of falling below 1 MW   |
| $P(Y > X)$        | Probability of violating threshold  |
| $P(\text{event})$ | Probability of an operational event |

This becomes particularly interesting for batteries and flexible assets.

---

# 5. Detect expected behaviour changes

You can use forecasts to identify **expected events**.

For example:

* expected start of operation
* expected shutdown
* expected peak
* expected minimum
* expected ramp
* expected sustained high consumption
* expected sustained export
* expected change in operating regime

For a pump:

> Expected operation: 06:00–08:30 and 17:00–20:00.

For solar:

> Expected generation peak: 12:30–13:00.

For a battery:

> Expected SOC reaches 90% at approximately 14:00.

The forecast therefore becomes a way of producing an **expected schedule**.

---

# 6. Baseline estimation

This is particularly interesting in demand response.

You can estimate:

> "What would this asset have done if we had not intervened?"

That becomes your **baseline**.

You could then compare:

$$
\text{Response} =
\text{Baseline forecast}
------------------------

\text{Observed metering}
$$

For example:

```text
Expected consumption:  4.2 MW
Actual consumption:    2.8 MW
--------------------------------
Estimated response:    1.4 MW
```

This could be useful for measuring the performance of a trade or flexibility instruction.

It also connects naturally to your existing trade evaluation work.

---

# 7. Detect anomalies

The forecast provides an expected value against which reality can be compared.

Define the residual:

$$
e_t = y_t - \hat{y}_t
$$

Then you can monitor:

* unusually large residuals
* persistent residuals
* changes in residual distribution
* unexpected ramps
* unexpected outages
* sensor problems
* asset behaviour changes

For example:

```text
Forecast: 4.8 MW
Actual:   0.2 MW
```

That could indicate:

* asset offline
* meter failure
* unexpected customer behaviour
* communication problem
* model failure

So the forecast model can simultaneously become an **asset monitoring system**.

---

# 8. Model health and concept drift

This becomes particularly important when you have lots of assets and lots of models.

For each asset/model pair you can calculate:

### Forecast error

$$
MAE = \frac{1}{n}\sum |y_t-\hat{y}_t|
$$

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum(y_t-\hat{y}_t)^2
}
$$

But you can go further.

Track error over:

* time
* settlement period
* day of week
* season
* weather conditions
* asset operating regime

You might discover:

```text
Asset A
Monday–Friday: good
Weekend:       poor
Summer:        good
Winter:        poor
High load:     poor
```

That tells you something about the model **and potentially the asset**.

---

# 9. Compare models across assets

Once you have many models, you can construct a model-performance layer.

For example:

| Asset | Model    |  MAE | RMSE |  Bias | Coverage |
| ----- | -------- | ---: | ---: | ----: | -------: |
| A001  | LightGBM | 0.21 | 0.34 | -0.03 |      91% |
| A002  | SARIMA   | 0.42 | 0.61 | +0.12 |      87% |
| A003  | XGBoost  | 0.18 | 0.27 | -0.01 |      94% |

Then you can ask:

> Which model works best for this asset?

or more interestingly:

> **What characteristics of an asset determine which forecasting approach works best?**

That could become an interesting meta-modelling problem.

---

# 10. Automatically classify assets

The time series and forecasts can give you an **asset behavioural fingerprint**.

For example:

```text
Asset A
  High seasonality
  Low variance
  Strong weekday pattern
  Low forecast uncertainty

Asset B
  High variance
  Weak seasonality
  Strong autocorrelation
  High forecast uncertainty

Asset C
  Intermittent
  Strong solar correlation
  Highly weather dependent
```

You could cluster assets based on features such as:

* mean load
* variance
* coefficient of variation
* autocorrelation
* seasonality strength
* intermittency
* ramp frequency
* peak-to-average ratio
* forecast error
* forecast uncertainty

Then use the clusters to determine:

> Which forecasting strategy should we use for this type of asset?

That starts looking like an **automated model-selection system**.

---

# 11. Portfolio forecasting

This is where having lots of individual models becomes particularly interesting.

Suppose you have:

```text
Asset A → forecast
Asset B → forecast
Asset C → forecast
...
Asset N → forecast
```

You can aggregate:

$$
\hat{Y}*t =
\sum*{i=1}^{N}\hat{Y}_{i,t}
$$

giving you a portfolio forecast.

But there is an important distinction:

$$
E\left[\sum_i Y_i\right]
========================

\sum_i E[Y_i]
$$

while the uncertainty of the aggregate depends on the **correlation between assets**.

This means you can potentially get a much more reliable portfolio forecast than individual forecasts.

For example:

```text
Individual asset uncertainty: high

                 ↓ aggregation

Portfolio uncertainty:        relatively low
```

This is particularly valuable for an aggregator.

---

# 12. Flexibility forecasting

For your particular energy context, I think this is one of the more interesting directions.

Instead of forecasting:

> "What will the asset do?"

you can try to forecast:

> **"How much flexibility will the asset probably have?"**

For example:

```text
09:00
Expected consumption: 4 MW
Potential reduction:  1 MW
Potential increase:   0.5 MW

10:00
Expected consumption: 3 MW
Potential reduction:  0.2 MW
Potential increase:   1.5 MW
```

Now the forecast is becoming a **flexibility envelope**.

You could represent:

$$
F_t = [F_t^{min}, F_t^{max}]
$$

or, for a battery:

```text
Expected SOC
SOC minimum
SOC maximum
Available charge capacity
Available discharge capacity
```

This is much closer to what an optimisation system actually needs.

---

# 13. Forecast tradable volume

You could then transform forecasts into something like:

```text
Asset forecast
       ↓
Available flexibility
       ↓
Expected deliverable volume
       ↓
Confidence-adjusted volume
       ↓
Trade candidate
```

For example:

$$
V_{\text{trade}}
================

V_{\text{forecast}}
\times
P(\text{successful delivery})
$$

Or a conservative quantile:

$$
V_{\text{trade}} = Q_{10}(V)
$$

depending on whether you're forecasting available export, reduction, etc.

That gives you a principled bridge between **forecasting and trading**.

---

# 14. Forecast opportunity

You can also combine the forecast with market prices.

For example:

```text
             Forecast
                ↓
       Expected flexibility
                ↓
       Market price forecast
                ↓
       Expected opportunity
                ↓
        Trade recommendation
```

You might calculate something like:

$$
\text{Expected value}
=====================

\text{Expected volume}
\times
\text{Expected price}
\times
P(\text{delivery})
$$

Now you're no longer just forecasting the asset.

You're forecasting the **economic opportunity associated with the asset**.

---

# 15. Scenario generation

Instead of producing one future:

```text
Forecast
  5.0 MW
```

you can generate many plausible futures:

```text
Scenario 1: 4.2 MW
Scenario 2: 4.7 MW
Scenario 3: 5.1 MW
Scenario 4: 5.8 MW
Scenario 5: 6.3 MW
...
```

Then an optimiser can ask:

> "What trade performs well across most plausible scenarios?"

This moves you towards **stochastic optimisation**.

For an energy aggregator, that could ultimately be more useful than trying to find the single "best" point forecast.

---

# 16. A useful architecture

I would think about the system in layers:

```text
                    METERING DATA
                         │
                         ▼
                 ┌─────────────────┐
                 │ Forecast Models │
                 └────────┬────────┘
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
       Point Forecast   Uncertainty   Scenarios
            │             │              │
            └─────────────┼──────────────┘
                          ▼
                  DERIVED FEATURES
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  Asset Behaviour    Flexibility        Anomalies
       │                  │                  │
       ▼                  ▼                  ▼
  Asset Metrics       Available          Model
                      Volume             Health
       │                  │
       └────────────┬─────┘
                    ▼
             PORTFOLIO LAYER
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
       Forecast   Flexibility  Risk
       Portfolio  Portfolio    Metrics
          │         │          │
          └─────────┼──────────┘
                    ▼
              OPTIMISATION
                    │
                    ▼
              TRADE SUGGESTION
```

The important design decision is that **the forecast should probably not be the final output of your modelling system**.

It should be a reusable intermediate layer.

---

## The particularly interesting extension

Given the sort of system you're describing, I would investigate a hierarchy like this:

### Level 1 — What will the asset do?

**Metering forecast**

> $Y_{i,t}$

### Level 2 — How certain are we?

**Forecast distribution**

> $P(Y_{i,t})$

### Level 3 — What can the asset probably do?

**Flexibility forecast**

> $F_{i,t}^{min}, F_{i,t}^{max}$

### Level 4 — What will the portfolio probably do?

**Aggregated forecast**

> $\sum_i Y_{i,t}$

### Level 5 — What can the portfolio probably deliver?

**Aggregated flexibility**

> $\sum_i F_{i,t}$

### Level 6 — What is economically attractive?

**Expected value / risk**

> $\text{volume} \times \text{price} \times P(\text{delivery})$

### Level 7 — What should we actually do?

**Optimisation / trade recommendation**

That gives you a fairly clean progression from **raw time-series forecasting → decision intelligence**.

For a system with many assets, I would also make the forecast output a standard contract, e.g. `asset_id`, `forecast_ts`, `target_ts`, `yhat`, `lower`, `upper`, `model_version`, and then build all the downstream analytics from that. This would make it possible to swap forecasting models without rebuilding the analytical layer.
