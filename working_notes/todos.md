# Project Setup & Development Todos

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

## In Progress

- [ ] Will mkdocs allow for ipynb still html output to be shown? or marimo?
- [x] We will need example data for multple assets metering data. We can have synetics data, it will have noise, no missing data for 1 week of ts intervalas being 30mins, with valutes in metering_kwh. 
- [x] We will need to update Claude.md to reflect the structure I am using. and the bluepint.md
- [x] We will need an index in claude.md for the project, and a table of contents in the blueprints.md. So we know the the structure of the project and where to find things and the decisions of choices made.




## Backlog: Architecture Implementation

Implementation follows main_idea.md layering. Tackle these in any order.

### Layer 1: Basic Forecasting
- [ ] Implement point forecast with uncertainty intervals
- [ ] Set up standard forecast output contract (asset_id, timestamp, yhat, lower, upper, model_version)
- [ ] Validate on sample time series data
- [ ] Document findings in docs_src/findings/

### Layer 2-4: Derived Features & Asset Profiling
- [ ] Daily energy aggregation
- [ ] Peak and ramp rate calculations
- [ ] Asset behavioral fingerprinting (seasonality, variance, autocorrelation)
- [ ] Asset clustering by behavioral type
- [ ] Document methodology in docs_src/methodology/

### Layer 5-7: Uncertainty & Anomaly Detection
- [ ] Forecast confidence quantification
- [ ] Event probability calculation
- [ ] Anomaly detection via forecast residuals
- [ ] Model health monitoring (MAE/RMSE by time/season/conditions)
- [ ] Document findings in docs_src/findings/

### Layer 8-10: Portfolio & Aggregation
- [ ] Multi-asset forecast aggregation
- [ ] Correlation analysis between assets
- [ ] Portfolio-level uncertainty estimation
- [ ] Asset classification system

### Layer 11-16: Advanced Features
- [ ] Flexibility forecasting (reduction/increase capacity)
- [ ] Tradable volume derivation
- [ ] Opportunity scoring (forecast + price)
- [ ] Scenario generation for stochastic optimization
- [ ] Automated model selection
- [ ] Trading recommendations

## Backlog: Supporting Infrastructure

- [ ] Create data pipeline for loading metering data
- [ ] Build model training framework

## Backlog: Testing & Quality

- [ ] Write unit tests for core forecasting functions

## Backlog: Documentation

- [ ] Document each forecasting layer as it's implemented
- [ ] Create example notebooks showing usage
- [ ] Build data quality reports
- [ ] Document assumptions and limitations

---

## Notes

- Start each new investigation in `working_notes/` with IPython
- Move validated findings to marimo notebooks
- Consolidate solid work into `docs_src/` and `src/`
- Archive experimental code to `archive/` if it might be useful later
