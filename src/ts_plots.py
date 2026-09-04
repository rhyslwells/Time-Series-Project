"""
Generic time series model plotting module
- Works with any model's ForecastOutput
- Diagnostic plots (residuals, coverage, uncertainty)
- Comparison plots across multiple models
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
from ts_model_framework import ForecastOutput, EvaluationMetrics


class TSPlotter:
    """Generic plotting for time series forecasts"""
    
    @staticmethod
    def forecast_vs_actual(
        y_test: np.ndarray,
        forecast: ForecastOutput,
        model_name: str = "Model",
        metrics: EvaluationMetrics = None
    ) -> go.Figure:
        """Plot: Forecast vs Actual with prediction intervals"""
        
        x_range = list(range(len(y_test)))
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=x_range, y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Point forecast
        fig.add_trace(go.Scatter(
            x=x_range, y=forecast.prediction,
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2),
            opacity=0.7
        ))
        
        # Prediction interval (shaded region)
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=forecast.upper.tolist() + forecast.lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% PI'
        ))
        
        title = f"{model_name}: Forecast vs Actual"
        if metrics:
            title += f" (RMSE: {metrics.rmse:.4f}, MAE: {metrics.mae:.4f})"
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Value (kWh)",
            height=500,
            width=1200,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def residuals_diagnostic(
        y_test: np.ndarray,
        forecast: ForecastOutput,
        model_name: str = "Model"
    ) -> go.Figure:
        """Plot: Residuals with uncertainty overlay"""
        
        residuals = y_test - forecast.prediction
        x_range = list(range(len(y_test)))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Residuals Over Time", "Residuals Distribution"),
            specs=[[{"secondary_y": True}], [{}]]
        )
        
        # Plot 1: Residuals vs Uncertainty
        fig.add_trace(
            go.Bar(
                x=x_range, y=residuals,
                name='Residuals',
                marker_color='steelblue',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=forecast.uncertainty_width,
                mode='lines',
                name='Uncertainty Width',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_hline(y=0, line_dash='solid', line_color='black', row=1, col=1)
        
        # Plot 2: Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=20,
                name='Distribution',
                marker_color='steelblue'
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Residual (kWh)", row=1, col=1)
        fig.update_yaxes(title_text="Uncertainty Width", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_xaxes(title_text="Residual Value", row=2, col=1)
        
        fig.update_layout(
            title_text=f"{model_name}: Residual Diagnostics",
            height=700,
            width=1200
        )
        
        return fig
    
    @staticmethod
    def uncertainty_analysis(
        forecast: ForecastOutput,
        model_name: str = "Model"
    ) -> go.Figure:
        """Plot: Uncertainty width over forecast horizon"""
        
        x_range = list(range(len(forecast.prediction)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_range, y=forecast.uncertainty_width,
            mode='lines+markers',
            name='Uncertainty Width',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        fig.add_hline(
            y=np.mean(forecast.uncertainty_width),
            line_dash='dash',
            line_color='green',
            annotation_text=f"Mean: {np.mean(forecast.uncertainty_width):.4f}"
        )
        
        fig.update_layout(
            title=f"{model_name}: Forecast Uncertainty Over Time",
            xaxis_title="Time Step",
            yaxis_title="Prediction Interval Width (kWh)",
            height=450,
            width=1200,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def pi_coverage(
        y_test: np.ndarray,
        forecast: ForecastOutput,
        model_name: str = "Model"
    ) -> go.Figure:
        """Plot: Prediction interval coverage (in/out of bounds)"""
        
        x_range = list(range(len(y_test)))
        
        in_bounds = (y_test >= forecast.lower) & (y_test <= forecast.upper)
        
        fig = go.Figure()
        
        # Color code: green if in bounds, red if out
        colors = ['green' if x else 'red' for x in in_bounds]
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_test,
            mode='markers',
            name='Actual',
            marker=dict(size=6, color=colors),
        ))
        
        # Add PI envelope
        fig.add_trace(go.Scatter(
            x=x_range, y=forecast.upper,
            mode='lines',
            name='Upper PI',
            line=dict(color='blue', width=1),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range, y=forecast.lower,
            mode='lines',
            name='Lower PI',
            line=dict(color='blue', width=1),
            opacity=0.5,
            fill='tonexty',
            fillcolor='rgba(0, 100, 200, 0.1)'
        ))
        
        coverage_pct = np.mean(in_bounds) * 100
        
        fig.update_layout(
            title=f"{model_name}: PI Coverage (actual: {coverage_pct:.1f}%, target: 80%)",
            xaxis_title="Time Step",
            yaxis_title="Value (kWh)",
            height=450,
            width=1200,
            hovermode='x unified'
        )
        
        return fig


class ComparisonPlotter:
    """Plots comparing multiple models"""
    
    @staticmethod
    def forecast_comparison(
        y_test: np.ndarray,
        forecasts: Dict[str, ForecastOutput],
        sample_size: int = 96  # Show first 4 days
    ) -> go.Figure:
        """Plot: All model forecasts overlaid"""
        
        sample_idx = min(sample_size, len(y_test))
        x_range = list(range(sample_idx))
        
        fig = go.Figure()
        
        # Actual
        fig.add_trace(go.Scatter(
            x=x_range, y=y_test[:sample_idx],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Model forecasts
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for (model_name, forecast), color in zip(forecasts.items(), colors):
            fig.add_trace(go.Scatter(
                x=x_range, y=forecast.prediction[:sample_idx],
                mode='lines',
                name=model_name,
                line=dict(color=color, width=2),
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Model Forecast Comparison (first 4 days)",
            xaxis_title="Time Step",
            yaxis_title="Value (kWh)",
            height=500,
            width=1200,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def metrics_comparison(metrics_dict: Dict[str, EvaluationMetrics]) -> go.Figure:
        """Plot: Bar chart comparing metrics across models"""
        
        models = list(metrics_dict.keys())
        mae_values = [metrics_dict[m].mae for m in models]
        rmse_values = [metrics_dict[m].rmse for m in models]
        mape_values = [metrics_dict[m].mape for m in models]
        coverage_values = [metrics_dict[m].pi_coverage for m in models]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("MAE", "RMSE", "MAPE %", "PI Coverage %"),
            specs=[[{}, {}], [{}, {}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color='steelblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightseagreen'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=models, y=mape_values, name='MAPE', marker_color='salmon'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=coverage_values, name='PI Coverage', marker_color='gold'),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="MAPE %", row=2, col=1)
        fig.update_yaxes(title_text="Coverage %", row=2, col=2)
        
        fig.update_layout(height=700, width=1200, title_text="Model Metrics Comparison")
        
        return fig
    
    @staticmethod
    def ranking_table(ranking_df: pd.DataFrame) -> str:
        """Pretty-print ranking table"""
        return ranking_df.to_string(index=False)
