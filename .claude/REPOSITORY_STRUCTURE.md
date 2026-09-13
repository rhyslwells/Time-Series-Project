# Repository Structure & Surface Area Policy

## Directory Organization

Keep the core `src/` directory focused and clean. Use `archive/` for experimental work.

```
Time-Series-Project/
├── src/                          # Production code
│   ├── __init__.py
│   ├── ts_contracts.py            # Forecast data contracts (ForecastOutput, EvaluationMetrics)
│   ├── ts_models.py               # TSModel + SeasonalNaive/SARIMA/ExponentialSmoothing/LightGBM
│   ├── ts_evaluation.py           # ModelEvaluator, ResidualDiagnostics, ModelComparison, ModelTuner
│   ├── ts_plots.py                # Generic plotly diagnostics for any ForecastOutput
│   ├── notebooks/                 # Marimo notebooks kept alongside the framework
│   │   ├── ts_model_explorer.py
│   │   └── sarima.py
│   └── data/                      # Data generation pipeline + its output
│       ├── generate_raw_data.py
│       ├── generate_daily_metrics.py
│       ├── generate_metering_features.py
│       ├── metering_data_raw.csv
│       ├── metering_data.parquet
│       ├── daily_metrics.parquet
│       └── metering_data_with_features.parquet
│
├── working_notes/                 # Exploration & notes (staged, not a package)
│   ├── todos.md                   # Task tracking
│   └── <numbered topic dirs>/     # Churns freely - list here deliberately omitted
│
├── archive/                       # Experimental work (old code)
├── docs_src/                      # Documentation source (tracked): coding/, data/, theory/, findings/, notebooks/
├── docs/                          # Generated docs
├── .claude/                       # This directory
│   ├── CLAUDE.md                  # Routing index
│   ├── CODING_STANDARDS.md
│   ├── REPOSITORY_STRUCTURE.md
│   ├── WORKFLOW.md
│   ├── DOCUMENTATION.md
│   ├── ARCHITECTURE.md
│   ├── DATA_STACK.md
│   ├── RESPONSE_STYLE.md
│   └── skills/                    # scope / implement / review / test / docs-capture
│
└── pyproject.toml, mkdocs.yml, README.md
```

## Surface Area Policy

Before creating new files or directories:

1. **src/** → only stable, reusable production code (models, pipelines, utilities)
2. **archive/** → freely add exploratory work, reference implementations, scripts
3. **working_notes/** → staged exploration notes and quick iterations (committed, but low-ceremony; not import-safe)
4. **docs_src/** → solid, tracked documentation and methodology

Do not create supplementary scaffolding or examples unless explicitly requested.

The data generation scripts in `src/data/` are the one exception to "scripts live in
working_notes/": they were promoted to production code because their output
(`src/data/*.parquet`) is a production data contract, and keeping generator and
generated data in the same directory keeps that contract auditable in one place.

## Key Production Files

See [DATA_STACK.md](DATA_STACK.md) for the data pipeline scripts/outputs and
[CLAUDE.md](CLAUDE.md#forecasting-framework-quick-reference) for the forecasting framework files.
Models take and return numpy arrays; all dataframe construction and IO uses polars.

The runnable end-to-end walkthrough (load -> compare -> diagnose -> tune -> finalize) is
`working_notes/3_framework/example_model_comparison.py`, not in `src/`.
