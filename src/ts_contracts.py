"""
Forecast data contracts shared across the forecasting framework.
"""

import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class ForecastOutput:
    """Standard forecast output contract"""
    prediction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    uncertainty_width: np.ndarray


@dataclass
class EvaluationMetrics:
    """Standardised evaluation metrics"""
    mae: float
    rmse: float
    mape: float  # percent (0-100)
    pi_coverage: float  # % of actuals within prediction interval
    mean_uncertainty_width: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        return (
            f"MAE: {self.mae:.4f} | RMSE: {self.rmse:.4f} | "
            f"MAPE: {self.mape:.2f}% | PI Coverage: {self.pi_coverage:.1f}%"
        )
