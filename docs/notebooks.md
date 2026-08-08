# Exploration Notebooks

Interactive marimo notebooks documenting ongoing investigations. These are living documents that evolve as we explore.

## How to Use

1. Each notebook is a `.md` file in the repository root
2. View summaries here, then open in marimo:

```bash
uv run marimo edit <notebook>.md
```

3. Notebooks contain executable Python cells, visualizations, and markdown explanations
4. When validated, findings move to the Findings section and methodology to Methodology

## Available Notebooks

### Foundational

- **[forecast_basics_exploration.md](../forecast_basics_exploration.md)**: Basic point forecasting with uncertainty intervals
- **[asset_classification_exploration.md](../asset_classification_exploration.md)**: Behavioral profiling and clustering

### Deep Dives

- **[asset_profiling_exploration.md](../asset_profiling_exploration.md)**: Detailed asset fingerprinting
  - Status: Validated
  - Finding: [Asset Profiling Analysis](findings/asset_profiling.md)

- **[anomaly_detection_exploration.md](../anomaly_detection_exploration.md)**: Detecting operational anomalies via residuals
  - Status: In progress
  - Related: [Anomaly Detection](findings/anomaly_detection.md)

- **[forecast_uncertainty_exploration.md](../forecast_uncertainty_exploration.md)**: Quantifying prediction interval quality
  - Status: Planning
  - Related: [Forecast Uncertainty](findings/forecast_uncertainty.md)

- **[flexibility_forecasting_exploration.md](../flexibility_forecasting_exploration.md)**: Capacity envelope estimation
  - Status: Planning

- **[portfolio_aggregation_exploration.md](../portfolio_aggregation_exploration.md)**: Multi-asset correlation and aggregation
  - Status: Planning

## Status Legend

- **Planning**: Not started, outline ready
- **In progress**: Active exploration
- **Validated**: Results documented in Findings
- **Production**: Implemented in src/

## Creating a New Notebook

```bash
uv run marimo edit new_topic_exploration.md
```

Follow this structure:

1. **Overview** section explaining the investigation
2. **Hypotheses** or questions being explored
3. **Data preparation** with examples
4. **Analysis** with visualizations
5. **Findings** section summarizing conclusions
6. **Next steps** for follow-up work

When validated, create a markdown file in `docs/findings/` linking back to the notebook.
