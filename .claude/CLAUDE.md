# Claude Code Assistant — Time Series Forecasting Project

Embedded assistant for energy systems forecasting. Help with: model setup, time series analysis, feature engineering, marimo notebooks, IPython exploration, documentation, validation.

**Expertise:** time series forecasting, anomaly detection, energy flexibility, ML development, Python data science (polars, scikit-learn, statsmodels), marimo/Jupyter workflows, parquet data engineering.

**Primary objective:** Build a principled, layered forecasting system with clarity in analytical approach.

---

## Quick Navigation

### Getting Started
- [**CODING_STANDARDS.md**](CODING_STANDARDS.md) — Style, no emojis, polars for all data ops
- [**REPOSITORY_STRUCTURE.md**](REPOSITORY_STRUCTURE.md) — Directory layout, surface area policy, key files
- [**WORKFLOW.md**](WORKFLOW.md) — Exploration → Consolidation → Implementation flow, when to ask before implementing

### Detailed Guidance
- [**ARCHITECTURE.md**](ARCHITECTURE.md) — Forecasting layers, design principles, key decisions
- [**DATA_STACK.md**](DATA_STACK.md) — Polars API, parquet strategy, data generation pipeline
- [**DOCUMENTATION.md**](DOCUMENTATION.md) — Doc policy, what/where to document
- [**RESPONSE_STYLE.md**](RESPONSE_STYLE.md) — How to propose changes, implementation principles, repository first rule

---

## The System in 30 Seconds

**Layers (raw metering → forecasts → derived features → flexibility → optimization)**

- Layer-separated architecture ensures reusability and swappable components
- Standard forecast contracts: `asset_id`, `timestamp`, `prediction`, `uncertainty`, `model_version`
- Data: 14 days × 15 assets (EV + solar) × 30-min intervals → 3 parquet files in `src/data/`

**Repository structure:**
- `src/` — production-ready code (includes the data generation scripts in `src/data/`, since their output is a production data contract)
- `archive/` — experimental work freely added
- `working_notes/` — exploration (not committed)
- `docs_src/` — solid, tracked findings

---

## Key Architectural Decisions

**Ask before changing:**
- Splitting/combining forecast layers
- Forecast output contracts
- New model types or aggregation approaches
- Major src/ restructuring


---

## Data Pipeline (Quick Reference)

All three generation scripts live in `src/data/`, alongside their output:

1. `generate_raw_data.py` → `metering_data_raw.csv`
2. `generate_daily_metrics.py` → `metering_data.parquet` (raw, 10,080 rows), `daily_metrics.parquet` (daily aggregates + behavioral metrics, 210 rows)
3. `generate_metering_features.py` → `metering_data_with_features.parquet` (metering_data.parquet joined with daily_metrics.parquet, broadcast across each day's 48 half-hour rows, `feat_`-prefixed columns, 10,080 rows)

---

## Behavior When Uncertain

1. Analyze existing repository structure
2. Infer the most conservative change
3. Present assumptions explicitly
4. Ask for clarification before architectural changes
5. **Default to preserving existing patterns**

**The repository is the source of truth.** When code differs from best practices: follow the repository, not generic conventions.

---

## Tools & Environment

- **Python stack:** polars, scikit-learn, statsmodels, numpy
- **Notebooks:** marimo (consolidation), IPython (exploration)
- **Data I/O:** parquet and the intermediate raw CSV, both in `src/data/`
- **Documentation:** mkdocs, tracked in docs_src/

See individual docs for specific API details and implementation patterns.
