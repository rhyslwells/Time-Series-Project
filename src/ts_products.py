"""
Forecast products: signals derived from a ForecastOutput.

Only the pieces that are easy to get wrong live here. One-liners (daily energy,
peak, ramp, baseline response) are plain numpy on `ForecastOutput.prediction`.
"""

import numpy as np
from scipy.stats import norm

from ts_contracts import ForecastOutput


def implied_std(forecast: ForecastOutput, confidence_level: float) -> np.ndarray:
    """Per-step standard deviation implied by a symmetric prediction interval.

    The z-score must match the interval's confidence level (80% -> 1.2816,
    90% -> 1.645); using the wrong one rescales every probability built on it.
    """
    z = norm.ppf((1 + confidence_level) / 2)
    return forecast.uncertainty_width / (2 * z)


def exceedance_probability(
    forecast: ForecastOutput, threshold: float, confidence_level: float
) -> np.ndarray:
    """P(Y_t > threshold) per step, treating the forecast as Normal(prediction, implied_std)."""
    sigma = implied_std(forecast, confidence_level)
    return 1 - norm.cdf(threshold, loc=forecast.prediction, scale=sigma)
