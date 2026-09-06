C:\Users\RhysL\Desktop\Time Series Project\docs_src\theory\forecast_products.md


Use this as a base 

C:\Users\RhysL\Desktop\Time Series Project\src\notebooks\sarima.py

but apply to this direction.

Main objective:
Take a model does not matter which and explore what can be achieved with each of the forecast_products and whati tells use for certain examples.



Fixing a model, what can be done with the model.

between two timestamps.

## Section 8: Forecast Probability of Events

Calculate P(forecast > threshold) for operational decisions.

Example: What's the probability metering exceeds 5 kW?

# ============================================================================
# SECTION 8: FORECAST PROBABILITY OF EVENTS
# ============================================================================

# TODO: I will need help understanding the meaning of this.

# Define event thresholds
thresholds = [
    np.percentile(y_train, 25),
    np.percentile(y_train, 50),
    np.percentile(y_train, 75),
]


# Calculate probability of exceeding each threshold
from scipy.stats import norm

prob_data = []
for threshold in thresholds:
    forecast_std = (upper - lower) / (2 * 1.645)
    prob_exceed = 1 - norm.cdf(threshold, loc=yhat, scale=forecast_std)
    mean_prob = np.mean(prob_exceed)
    prob_data.append(
        {
            "Threshold": f"{threshold:.2f} kWh",
            "P(Forecast > Threshold)": f"{mean_prob:.1%}",
            "Min Probability": f"{np.min(prob_exceed):.1%}",
            "Max Probability": f"{np.max(prob_exceed):.1%}",
        }
    )

probability_events = pl.DataFrame(prob_data)
print("\nForecast Probability of Events:")
print(probability_events)
