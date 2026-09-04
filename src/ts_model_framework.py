"""
Generalised Time Series Forecasting Framework
- Multiple model implementations (SARIMA, ExponentialSmoothing, LightGBM)
- Standardised evaluation metrics and plots
- Model tuning and hyperparameter search
- Comparison and ranking
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import norm


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
    mape: float
    pi_coverage: float  # % of actuals within prediction interval
    mean_uncertainty_width: float
    
    def to_dict(self):
        return asdict(self)
    
    def __repr__(self):
        return (
            f"MAE: {self.mae:.4f} | RMSE: {self.rmse:.4f} | "
            f"MAPE: {self.mape:.2f}% | PI Coverage: {self.pi_coverage:.1f}%"
        )


class TSModel(ABC):
    """Abstract base class for time series models"""
    
    def __init__(self, name: str, y_train: np.ndarray):
        self.name = name
        self.y_train = y_train
        self.model = None
        self.fitted = False
        
    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Fit model to training data"""
        pass
    
    @abstractmethod
    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        """Generate forecasts with prediction intervals"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Return current model hyperparameters"""
        pass
    
    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Set model hyperparameters"""
        pass


class SARIMAModel(TSModel):
    """SARIMA(p,d,q)×(P,D,Q,s) wrapper"""
    
    def __init__(self, y_train: np.ndarray, order: Tuple = (1,1,1), 
                 seasonal_order: Tuple = (1,1,1,48)):
        super().__init__("SARIMA", y_train)
        self.order = order
        self.seasonal_order = seasonal_order
        
    def fit(self, **kwargs) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        try:
            model = SARIMAX(
                self.y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = model.fit(disp=False, maxiter=kwargs.get("maxiter", 1000))
            self.fitted = True
        except Exception as e:
            raise RuntimeError(f"SARIMA fit failed: {e}")
    
    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        forecast = self.model.get_forecast(steps=steps)
        yhat = np.asarray(forecast.predicted_mean)
        ci = np.asarray(forecast.conf_int(alpha=1 - confidence_level))
        
        return ForecastOutput(
            prediction=yhat,
            lower=ci[:, 0],
            upper=ci[:, 1],
            uncertainty_width=ci[:, 1] - ci[:, 0]
        )
    
    def get_params(self) -> Dict:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }
    
    def set_params(self, **kwargs) -> None:
        if "order" in kwargs:
            self.order = kwargs["order"]
        if "seasonal_order" in kwargs:
            self.seasonal_order = kwargs["seasonal_order"]
        self.fitted = False


class ExponentialSmoothingModel(TSModel):
    """Exponential Smoothing wrapper (Holt-Winters)"""
    
    def __init__(self, y_train: np.ndarray, seasonal_periods: int = 48,
                 trend: str = "add", seasonal: str = "add", damped_trend: bool = False):
        super().__init__("ExponentialSmoothing", y_train)
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        
    def fit(self, **kwargs) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        try:
            model = ExponentialSmoothing(
                self.y_train,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend
            )
            self.model = model.fit(optimized=True)
            self.fitted = True
        except Exception as e:
            raise RuntimeError(f"ExponentialSmoothing fit failed: {e}")
    
    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        yhat = self.model.forecast(steps=steps)
        
        # Estimate prediction intervals from residuals
        residuals = self.model.resid
        residual_std = np.std(residuals)
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std
        
        return ForecastOutput(
            prediction=yhat.values,
            lower=yhat.values - margin,
            upper=yhat.values + margin,
            uncertainty_width=np.full_like(yhat.values, 2 * margin)
        )
    
    def get_params(self) -> Dict:
        return {
            "seasonal_periods": self.seasonal_periods,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend
        }
    
    def set_params(self, **kwargs) -> None:
        if "seasonal_periods" in kwargs:
            self.seasonal_periods = kwargs["seasonal_periods"]
        if "trend" in kwargs:
            self.trend = kwargs["trend"]
        if "seasonal" in kwargs:
            self.seasonal = kwargs["seasonal"]
        if "damped_trend" in kwargs:
            self.damped_trend = kwargs["damped_trend"]
        self.fitted = False


class LightGBMModel(TSModel):
    """LightGBM for time series (lag features + seasonality)"""
    
    def __init__(self, y_train: np.ndarray, lags: List[int] = None, 
                 num_leaves: int = 31, learning_rate: float = 0.05):
        super().__init__("LightGBM", y_train)
        self.lags = lags or [1, 2, 48, 96]  # 30min, 1hr, 1day, 2day
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.feature_names = None
        
    def _create_features(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lag features and seasonality"""
        max_lag = max(self.lags)
        X = []
        targets = []
        
        for i in range(max_lag, len(y)):
            features = [y[i - lag] for lag in self.lags]
            # Add seasonal component (hour of day)
            hour_of_day = (i % 48) / 48.0
            features.append(hour_of_day)
            X.append(features)
            targets.append(y[i])
        
        self.feature_names = [f"lag_{lag}" for lag in self.lags] + ["hour_of_day"]
        return np.array(X), np.array(targets)
    
    def fit(self, **kwargs) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("LightGBM not installed. pip install lightgbm")
        
        X, y = self._create_features(self.y_train)
        
        self.model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=kwargs.get("n_estimators", 100),
            verbose=-1
        )
        self.model.fit(X, y)
        self.fitted = True
    
    def forecast(self, steps: int, confidence_level: float = 0.80) -> ForecastOutput:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        yhat = []
        y_recent = self.y_train.copy()
        
        for _ in range(steps):
            features = [y_recent[-lag] if lag <= len(y_recent) else y_recent[0] 
                       for lag in self.lags]
            hour = (len(y_recent) % 48) / 48.0
            features.append(hour)
            
            pred = self.model.predict([features])[0]
            yhat.append(pred)
            y_recent = np.append(y_recent, pred)
        
        yhat = np.array(yhat)
        
        # Estimate prediction intervals from training residuals
        X_train, y_train = self._create_features(self.y_train)
        train_preds = self.model.predict(X_train)
        residual_std = np.std(y_train - train_preds)
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std
        
        return ForecastOutput(
            prediction=yhat,
            lower=yhat - margin,
            upper=yhat + margin,
            uncertainty_width=np.full_like(yhat, 2 * margin)
        )
    
    def get_params(self) -> Dict:
        return {
            "lags": self.lags,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate
        }
    
    def set_params(self, **kwargs) -> None:
        if "lags" in kwargs:
            self.lags = kwargs["lags"]
        if "num_leaves" in kwargs:
            self.num_leaves = kwargs["num_leaves"]
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]
        self.fitted = False


class ModelEvaluator:
    """Standardised evaluation for all models"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, forecast: ForecastOutput) -> EvaluationMetrics:
        """Compute all metrics"""
        mae = mean_absolute_error(y_true, forecast.prediction)
        rmse = np.sqrt(mean_squared_error(y_true, forecast.prediction))
        mape = mean_absolute_percentage_error(y_true, forecast.prediction)
        
        # PI coverage: % of actual values within [lower, upper]
        coverage = np.mean((y_true >= forecast.lower) & (y_true <= forecast.upper)) * 100
        
        # Mean uncertainty width
        mean_width = np.mean(forecast.uncertainty_width)
        
        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            pi_coverage=coverage,
            mean_uncertainty_width=mean_width
        )


class ModelComparison:
    """Compare and rank multiple models"""
    
    def __init__(self, y_train: np.ndarray, y_test: np.ndarray):
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def add_model(self, model: TSModel) -> None:
        """Register a model"""
        self.models[model.name] = model
        
    def fit_all(self) -> None:
        """Fit all registered models"""
        for name, model in self.models.items():
            try:
                model.fit()
                print(f"✓ {name} fitted")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    
    def evaluate_all(self, confidence_level: float = 0.80) -> pd.DataFrame:
        """Evaluate all models and return ranking"""
        results = {}
        
        for name, model in self.models.items():
            if not model.fitted:
                print(f"Skipping {name} (not fitted)")
                continue
            
            try:
                forecast = model.forecast(len(self.y_test), confidence_level)
                metrics = ModelEvaluator.evaluate(self.y_test, forecast)
                results[name] = {
                    'metrics': metrics,
                    'forecast': forecast,
                    'params': model.get_params()
                }
            except Exception as e:
                print(f"✗ {name} evaluation failed: {e}")
        
        self.results = results
        return self._ranking_table()
    
    def _ranking_table(self) -> pd.DataFrame:
        """Create ranking table with scores"""
        rows = []
        for name, data in self.results.items():
            metrics = data['metrics']
            rows.append({
                'Model': name,
                'MAE': metrics.mae,
                'RMSE': metrics.rmse,
                'MAPE': metrics.mape,
                'PI Coverage %': metrics.pi_coverage,
                'Uncertainty Width': metrics.mean_uncertainty_width,
            })
        
        df = pd.DataFrame(rows)
        # Rank by RMSE (lower is better)
        df['Rank'] = df['RMSE'].rank()
        return df.sort_values('Rank')
    
    def best_model(self) -> Tuple[str, EvaluationMetrics]:
        """Return best model name and metrics"""
        if not self.results:
            raise RuntimeError("No results yet. Call evaluate_all() first.")
        
        best_name = min(self.results, key=lambda x: self.results[x]['metrics'].rmse)
        return best_name, self.results[best_name]['metrics']
    
    def get_forecast(self, model_name: str) -> ForecastOutput:
        """Retrieve forecast from evaluated model"""
        return self.results[model_name]['forecast']


class ModelTuner:
    """Hyperparameter tuning for a single model"""
    
    def __init__(self, model_class, y_train: np.ndarray, y_test: np.ndarray):
        self.model_class = model_class
        self.y_train = y_train
        self.y_test = y_test
        self.trials = []
        
    def grid_search(self, param_grid: Dict[str, List], 
                   confidence_level: float = 0.80) -> pd.DataFrame:
        """Grid search over parameter combinations"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            
            try:
                model = self.model_class(self.y_train, **params)
                model.fit()
                forecast = model.forecast(len(self.y_test), confidence_level)
                metrics = ModelEvaluator.evaluate(self.y_test, forecast)
                
                self.trials.append({
                    'params': params,
                    'metrics': metrics,
                    'forecast': forecast
                })
            except Exception as e:
                print(f"✗ Params {params} failed: {e}")
        
        return self._trials_table()
    
    def _trials_table(self) -> pd.DataFrame:
        """Return trials as ranked dataframe"""
        rows = []
        for trial in self.trials:
            params = trial['params']
            metrics = trial['metrics']
            row = {
                'MAE': metrics.mae,
                'RMSE': metrics.rmse,
                'MAPE': metrics.mape,
                'PI Coverage %': metrics.pi_coverage,
            }
            row.update({f"param_{k}": v for k, v in params.items()})
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df['Rank'] = df['RMSE'].rank()
        return df.sort_values('Rank')
    
    def best_params(self) -> Dict:
        """Return best hyperparameters"""
        if not self.trials:
            raise RuntimeError("No trials yet. Call grid_search() first.")
        
        best_trial = min(self.trials, key=lambda x: x['metrics'].rmse)
        return best_trial['params']
