"""
Example: Model Comparison & Tuning Workflow

1. Load data
2. Compare SARIMA, ExponentialSmoothing, LightGBM
3. Identify best model
4. Tune best model's hyperparameters
5. Generate diagnostic plots
"""

import numpy as np
import pandas as pd
from datetime import datetime
import polars as pl

from ts_model_framework import (
    SARIMAModel,
    ExponentialSmoothingModel,
    LightGBMModel,
    ModelComparison,
    ModelTuner,
    ModelEvaluator,
)
from ts_plots import TSPlotter, ComparisonPlotter


def load_data(asset_id: str, test_days: int = 4) -> tuple:
    """Load metering data and split train/test"""

    df = pl.read_parquet("../../src/data/metering_data.parquet")
    asset_data = df.filter(pl.col("asset_id") == asset_id).sort("timestamp")

    y = asset_data.select("metering_kwh").to_numpy().flatten()
    timestamps = asset_data.select("timestamp").to_numpy().flatten()

    test_split_idx = len(y) - (test_days * 48)
    y_train = y[:test_split_idx]
    y_test = y[test_split_idx:]

    return y_train, y_test, timestamps


def step1_compare_models(y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Compare three baseline models"""

    print("\n" + "=" * 60)
    print("STEP 1: Model Comparison (Baseline)")
    print("=" * 60)

    comparator = ModelComparison(y_train, y_test)

    # Add models
    comparator.add_model(
        SARIMAModel(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48))
    )
    comparator.add_model(ExponentialSmoothingModel(y_train, seasonal_periods=48))
    comparator.add_model(LightGBMModel(y_train, lags=[1, 2, 48, 96]))

    # Fit all
    comparator.fit_all()

    # Evaluate
    ranking = comparator.evaluate_all()
    print("\nRanking (by RMSE):")
    print(ranking.to_string(index=False))

    # Best model
    best_name, best_metrics = comparator.best_model()
    print(f"\n✓ Best Model: {best_name}")
    print(f"  {best_metrics}")

    # Save for next steps
    return ranking, comparator, best_name


def step2_diagnostic_plots(comparator: ModelComparison, best_name: str):
    """Generate diagnostic plots for best model"""

    print("\n" + "=" * 60)
    print(f"STEP 2: Diagnostics for {best_name}")
    print("=" * 60)

    forecast = comparator.get_forecast(best_name)
    metrics = comparator.results[best_name]["metrics"]

    # Plot 1: Forecast vs Actual
    fig1 = TSPlotter.forecast_vs_actual(comparator.y_test, forecast, best_name, metrics)
    fig1.write_html(f"{best_name}_forecast_vs_actual.html")
    print(f"✓ Saved: {best_name}_forecast_vs_actual.html")

    # Plot 2: Residuals
    fig2 = TSPlotter.residuals_diagnostic(comparator.y_test, forecast, best_name)
    fig2.write_html(f"{best_name}_residuals.html")
    print(f"✓ Saved: {best_name}_residuals.html")

    # Plot 3: Uncertainty
    fig3 = TSPlotter.uncertainty_analysis(forecast, best_name)
    fig3.write_html(f"{best_name}_uncertainty.html")
    print(f"✓ Saved: {best_name}_uncertainty.html")

    # Plot 4: PI Coverage
    fig4 = TSPlotter.pi_coverage(comparator.y_test, forecast, best_name)
    fig4.write_html(f"{best_name}_pi_coverage.html")
    print(f"✓ Saved: {best_name}_pi_coverage.html")


def step3_compare_all_forecasts(comparator: ModelComparison):
    """Plot all model forecasts side-by-side"""

    print("\n" + "=" * 60)
    print("STEP 3: Forecast Comparison (All Models)")
    print("=" * 60)

    forecasts = {name: data["forecast"] for name, data in comparator.results.items()}

    # Plot 1: Overlaid forecasts
    fig1 = ComparisonPlotter.forecast_comparison(comparator.y_test, forecasts)
    fig1.write_html("all_models_forecast_comparison.html")
    print("✓ Saved: all_models_forecast_comparison.html")

    # Plot 2: Metrics comparison
    metrics_dict = {name: data["metrics"] for name, data in comparator.results.items()}
    fig2 = ComparisonPlotter.metrics_comparison(metrics_dict)
    fig2.write_html("all_models_metrics_comparison.html")
    print("✓ Saved: all_models_metrics_comparison.html")


def step4_tune_best_model(y_train: np.ndarray, y_test: np.ndarray, best_name: str):
    """Hyperparameter tuning for best model"""

    print("\n" + "=" * 60)
    print(f"STEP 4: Hyperparameter Tuning for {best_name}")
    print("=" * 60)

    if best_name == "SARIMA":
        print("Grid searching SARIMA(p,d,q) × (P,D,Q,s)...")

        tuner = ModelTuner(SARIMAModel, y_train, y_test)

        # Define parameter grid (small for speed)
        param_grid = {
            "order": [(0, 1, 1), (1, 1, 0), (1, 1, 1), (1, 1, 2)],
            "seasonal_order": [(0, 0, 0, 48), (1, 0, 1, 48), (1, 1, 1, 48)],
        }

        trials = tuner.grid_search(param_grid)
        print("\nTuning Results (top 5):")
        print(trials.head(5).to_string(index=False))

        best_params = tuner.best_params()
        print(f"\n✓ Best params: {best_params}")

        return best_params

    elif best_name == "ExponentialSmoothing":
        print("Grid searching Exponential Smoothing (trend, seasonal, damped)...")

        tuner = ModelTuner(ExponentialSmoothingModel, y_train, y_test)

        param_grid = {
            "trend": ["add", "mul"],
            "seasonal": ["add", "mul"],
            "damped_trend": [True, False],
        }

        trials = tuner.grid_search(param_grid)
        print("\nTuning Results:")
        print(trials.head(5).to_string(index=False))

        best_params = tuner.best_params()
        print(f"\n✓ Best params: {best_params}")

        return best_params

    elif best_name == "LightGBM":
        print("Grid searching LightGBM (num_leaves, learning_rate)...")

        tuner = ModelTuner(LightGBMModel, y_train, y_test)

        param_grid = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.05, 0.1]}

        trials = tuner.grid_search(param_grid)
        print("\nTuning Results:")
        print(trials.head(5).to_string(index=False))

        best_params = tuner.best_params()
        print(f"\n✓ Best params: {best_params}")

        return best_params


def step5_final_model(
    y_train: np.ndarray, y_test: np.ndarray, best_name: str, best_params: dict
):
    """Refit best model with tuned params and generate final output"""

    print("\n" + "=" * 60)
    print(f"STEP 5: Final Model ({best_name} with tuned params)")
    print("=" * 60)

    if best_name == "SARIMA":
        model = SARIMAModel(y_train, **best_params)
    elif best_name == "ExponentialSmoothing":
        model = ExponentialSmoothingModel(y_train, **best_params)
    elif best_name == "LightGBM":
        model = LightGBMModel(y_train, **best_params)

    model.fit()
    forecast = model.forecast(len(y_test))
    metrics = ModelEvaluator.evaluate(y_test, forecast)

    print(f"\n✓ Final Model Performance:")
    print(f"  {metrics}")

    # Generate final plots
    fig1 = TSPlotter.forecast_vs_actual(
        y_test, forecast, f"{best_name} (Tuned)", metrics
    )
    fig1.write_html(f"{best_name}_final_forecast.html")
    print(f"\n✓ Saved: {best_name}_final_forecast.html")

    # Standard output contract
    forecast_df = pd.DataFrame(
        {
            "prediction": forecast.prediction,
            "lower": forecast.lower,
            "upper": forecast.upper,
            "uncertainty_width": forecast.uncertainty_width,
            "actual": y_test,
        }
    )

    forecast_df.to_csv(f"{best_name}_forecast_output.csv", index=False)
    print(f"✓ Saved: {best_name}_forecast_output.csv")

    return model, forecast, metrics


def main():
    """Full workflow: compare → diagnose → tune → finalize"""

    # Load data
    y_train, y_test, _ = load_data("ASSET_001", test_days=4)
    print(f"\nData loaded: {len(y_train)} train, {len(y_test)} test samples")

    # Step 1: Compare baseline models
    ranking, comparator, best_name = step1_compare_models(y_train, y_test)

    # Step 2: Diagnostics for best model
    step2_diagnostic_plots(comparator, best_name)

    # Step 3: Compare all forecasts
    step3_compare_all_forecasts(comparator)

    # Step 4: Tune best model
    best_params = step4_tune_best_model(y_train, y_test, best_name)

    # Step 5: Final model with tuned params
    final_model, final_forecast, final_metrics = step5_final_model(
        y_train, y_test, best_name, best_params
    )

    print("\n" + "=" * 60)
    print("✓ Workflow Complete")
    print("=" * 60)

    print(f"\nFinal Summary:")
    print(f"  Best Model: {best_name}")
    print(f"  Best Params: {best_params}")
    print(f"  Final Metrics: {final_metrics}")


if __name__ == "__main__":
    main()
