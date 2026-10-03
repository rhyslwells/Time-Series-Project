# Project Setup & Development Todos

## Notes



## Completed

## In Progress

## Next Steps 2
1. Use metering_data.parquet to train baseline forecasts
- [ ] **Baseline Forecasts:** Use metering_data.parquet to train models
- [ ] **Asset Profiling:** Use daily_metrics.parquet to study behavioral differences
- [ ] **Forecasting:** Evaluate models on held-out days


## Backlog: Architecture Implementation

Implementation follows main_idea.md layering. Tackle these in any order.

### Layer 1: Basic Forecasting
- [ ] Implement point forecast with uncertainty intervals
- [ ] Set up standard forecast output contract (asset_id, timestamp, yhat, lower, upper, model_version)
- [ ] Document findings in docs_src/findings/

### Layer 2-4: Derived Features & Asset Profiling
- [ ] Daily energy aggregation
- [ ] Peak and ramp rate calculations
- [ ] Asset behavioral fingerprinting (seasonality, variance, autocorrelation)
- [ ] Asset clustering by behavioral type
- [ ] Document methodology in docs_src/methodology/

### Layer 5-7: Uncertainty Quantification & Event Probability
- [ ] Forecast confidence quantification
- [ ] Event probability calculation
- [ ] Model health monitoring (MAE/RMSE by time/season/conditions)
- [ ] Document findings in docs_src/findings/

### Layer 8-10: Portfolio & Aggregation
- [ ] Multi-asset forecast aggregation
- [ ] Correlation analysis between assets
- [ ] Portfolio-level uncertainty estimation

## Backlog: Supporting Infrastructure

- [ ] Create data pipeline for loading metering data
- [ ] Build model training framework

## Backlog: Documentation

- [ ] Document each forecasting layer as it's implemented
- [ ] Create example notebooks showing usage
- [ ] Document assumptions and limitations

---

## Notes

- Start each new investigation in `working_notes/` with IPython
- Move validated findings to marimo notebooks
- Consolidate solid work into `docs_src/` and `src/`
- Archive experimental code to `archive/` if it might be useful later


## Completed

- [x] GitHub documentation page (README.md)
- [x] Set up marimo for exploration and investigation consolidation
- [x] Configure uv for setup
- [x] Update pyproject.toml (slimmed dependencies)
- [x] Update CLAUDE.md with project-specific guidance
- [x] Set up archive folder for non-core notes and scripts
- [x] Create BLUEPRINTS.md for workflow documentation
- [x] Create docs_src/ folder structure for tracked, solid notes
- [x] Create working_notes/ folder for local, untracked exploration
- [x] Create src/ folder for production-ready code
- [x] We will need example data for multple assets metering data. We can have synetics data, it will have noise, no missing data for 1 week of ts intervalas being 30mins, with valutes in metering_kwh. 
- [x] We will need to update Claude.md to reflect the structure I am using. and the bluepint.md
- [x] We will need an index in claude.md for the project, and a table of contents in the blueprint
