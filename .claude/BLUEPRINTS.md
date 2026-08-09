# Blueprints: Project Architecture & Workflow

Master index for project structure, phases, and development workflow.

**Last Updated:** 2026-08-09  
**Current Phase:** Data generation & inspection (Phase 1/3)

---

## Quick Navigation

| | |
|---|---|
| **Project Overview** | [Goal, Scope, Tech Stack](#project-overview) |
| **Architecture** | [7-Layer Forecasting System](#architecture-layers) |
| **Phases** | [Phase 1 (Done), Phase 2-3 (Planned)](#phase-roadmap) |
| **Directory Map** | [Complete structure](#directory-structure) |
| **Data Pipeline** | [Generation → Production](#data-pipeline) |
| **Development Workflow** | [Exploration → Consolidation → Implementation](#workflow-ipython--marimo--src) |
| **Getting Started** | [Run scripts & load data](#getting-started) |

---

## Project Overview

**Goal:** Build a layered time series forecasting system for energy assets (EV charging, solar+battery)

**Scope:**
- 15 synthetic assets with realistic behavioral patterns (8 EV charging, 7 solar+battery)
- 14 days of 30-minute metering data (2 measurements/hour = 1,344 records/asset, 10,080 total)
- Multi-layer forecasting: raw → forecast → derived → flexibility → optimization

**Technology Stack:**
- **Data:** Polars (not pandas), Parquet
- **Forecasting:** scikit-learn, statsmodels
- **Exploration:** IPython, Marimo notebooks
- **Documentation:** Markdown, MkDocs

---

## Architecture Layers

From `working_notes/main_idea.md` — 7-level hierarchy:

### Level 1: Raw Metering
- Input: 30-minute interval consumption/generation (kWh)
- Source: `src/data/metering_data.parquet`
- Schema: asset_id, timestamp, metering_kwh, asset_type

### Level 2: Point & Distribution Forecasts
- Output: yhat, P10, P50, P90 for next 3 days
- Models: SARIMA, LightGBM, Prophet, ensemble
- Contract: asset_id, forecast_ts, target_ts, yhat, lower, upper, model_version

### Level 3: Derived Features
- Daily energy, peak, ramp rates, profiles
- Source: `src/data/daily_metrics.parquet`, `src/data/ramp_rates.parquet`
- Enables: asset characterization, operational planning

### Level 4: Flexibility Forecasting
- Available reduction/increase capacity
- State-of-charge projections (batteries)
- Ramp limits from `ramp_rates.parquet`

### Level 5: Anomaly Detection
- Forecast residuals vs actuals
- Asset health monitoring
- Model degradation detection

### Level 6: Portfolio Aggregation
- Sum individual forecasts → portfolio forecast
- Account for correlation effects
- Reduce portfolio uncertainty

### Level 7: Optimization & Trading
- Stochastic optimization under uncertainty
- Trade candidate generation
- Risk-adjusted volume calculation

---

## Phase Roadmap

### Phase 1: Data Foundation (CURRENT — COMPLETE)
**Duration:** 1 week  
**Status:** Done  
**Deliverables:**
- [x] Synthetic data generation (15 assets, 2 weeks, 10,080 records)
- [x] Data inspection & analysis (metrics, fingerprints, clustering)
- [x] Parquet export to `src/data/` (4 files)
- [x] Project documentation (CLAUDE.md, BLUEPRINTS.md, notes.md)

**Files:**
- `working_notes/1_produce_data/generate_data.py`
- `working_notes/1_produce_data/inspect_data.py`
- `src/data/*.parquet` (metering_data, daily_metrics, ramp_rates, behavioral_fingerprints)

### Phase 2: Forecasting Models (PLANNED)
**Duration:** 2-3 weeks  
**Deliverables:**
- [ ] Baseline forecasts (SARIMA, exponential smoothing)
- [ ] Model validation framework
- [ ] Per-asset model selection
- [ ] Uncertainty quantification (prediction intervals)

**Location:** `working_notes/2_basic_forecast_model/`, later `src/forecasting/`

### Phase 3: Derived Analytics (PLANNED)
**Duration:** 2 weeks  
**Deliverables:**
- [ ] Daily profile generation
- [ ] Flexibility forecasting
- [ ] Anomaly detection
- [ ] Portfolio aggregation

**Location:** `src/analytics/`, `src/portfolio/`

---

## Directory Structure

```
Time-Series-Project/
│
├── src/                          # Production code & data
│   ├── __init__.py
│   ├── data/                      # Generated parquet files
│   │   ├── metering_data.parquet              # Raw 30-min (10,080 rows)
│   │   ├── daily_metrics.parquet              # Daily aggs (210 rows)
│   │   ├── ramp_rates.parquet                 # Ramp stats (15 rows)
│   │   └── behavioral_fingerprints.parquet    # Clustering features (15 rows)
│   │
│   ├── forecasting/               # Forecast models (Phase 2)
│   ├── analytics/                 # Derived features (Phase 3)
│   ├── portfolio/                 # Portfolio operations (Phase 3)
│   └── evaluation/                # Validation & testing
│
├── working_notes/                 # Exploration & reference
│   ├── main_idea.md               # Core architecture (7 layers)
│   ├── todos.md                   # Task tracking
│   │
│   ├── 1_produce_data/            # PHASE 1: Data generation
│   │   ├── notes.md               # Detailed workflow
│   │   ├── generate_data.py       # Synthetic data (10,080 records)
│   │   ├── inspect_data.py        # Analysis & parquet export
│   │   └── metering_data_raw.csv  # Working file (intermediate)
│   │
│   ├── 2_basic_forecast_model/    # PHASE 2: Forecasting (planned)
│   │   └── notes.md
│   │
│   └── 3_derived_analytics/       # PHASE 3: Analytics (planned)
│       └── notes.md
│
├── docs_src/                      # Documentation source
├── docs/                          # Generated docs (mkdocs)
│
├── .claude/
│   ├── CLAUDE.md                  # System instructions & index
│   ├── BLUEPRINTS.md              # This file: master plan
│   └── projects/[session]/memory/ # Session memory
│
├── archive/                       # Old/experimental code
├── .github/                       # CI/CD workflows
├── pyproject.toml                 # Dependencies & config
├── uv.lock                        # Locked dependencies
├── mkdocs.yml                     # Docs configuration
├── README.md                      # Project overview
└── .gitignore
```

### Directory Ownership

| Directory | Purpose | Committed | Notes |
|-----------|---------|-----------|-------|
| `src/` | Production code | Yes | Only stable, reusable modules |
| `src/data/` | Parquet data | No (.gitignored) | Generated by scripts, ~100KB |
| `working_notes/` | Exploration | No (ignored) | Scratch work, rapid iteration |
| `docs_src/` | Doc source | Yes | Tracked findings & methodology |
| `docs/` | Generated site | No (ignored) | Built by mkdocs, not committed |
| `archive/` | Old code | Yes | Reference implementations |
| `.claude/` | AI instructions | Yes | System role, plans, memory |

---

## Data Pipeline

### Generation Workflow

```
1. generate_data.py (1-2 min)
   ├─ Seed: 42 (reproducible)
   ├─ Assets: 15 (8 EV, 7 solar)
   ├─ Period: 14 days
   ├─ Interval: 30 minutes (2 measurements/hour)
   └─> metering_data_raw.csv (10,080 rows)

2. inspect_data.py (2-3 min)
   ├─ Compute: daily metrics, ramp rates, fingerprints
   ├─ Print: terminal report (8 sections)
   └─> Export to src/data/:
       ├─ metering_data.parquet
       ├─ daily_metrics.parquet
       ├─ ramp_rates.parquet
       └─ behavioral_fingerprints.parquet
```

### Production Data Schemas

**metering_data.parquet** (10,080 rows)
```
asset_id (str): ASSET_001 to ASSET_015
timestamp (datetime): 2025-01-01 00:00 to 2025-01-14 23:30
metering_kwh (float64): -2.5 to +4.0 (kWh per 30-min period)
asset_type (str): 'ev_charging' or 'solar_battery'
```

**daily_metrics.parquet** (210 rows)
```
asset_id, date, daily_energy_kwh, daily_peak_kw, daily_min_kw,
daily_avg_kw, daily_std_kw, n_samples (always 96)
```

**ramp_rates.parquet** (15 rows)
```
asset_id, mean_ramp_kw, std_ramp_kw, max_ramp_kw,
mean_abs_ramp_kw, max_abs_ramp_kw
```

**behavioral_fingerprints.parquet** (15 rows)
```
asset_id, asset_type, mean_load_kw, variance, coefficient_of_variation,
autocorrelation, intermittency_ratio, peak_to_avg_ratio
```

### Key Findings

**Asset Separation:**
- EV Charging (8 assets): mean_kw=1.83, cv=0.79 (predictable)
- Solar+Battery (7 assets): mean_kw=0.026, cv=45.5 (variable)

**Data Quality:**
- No missing values
- 2,767 negative values (18% = net export, expected)
- 30-minute intervals continuous

---

## Key Technical Decisions

### 1. Polars Over Pandas
- **Why:** Lazy evaluation, parquet-native, memory efficient, type preservation
- **Impact:** All scripts use polars (`.group_by()`, `.with_columns()`, `.write_parquet()`)

### 2. Synthetic Deterministic Data
- **Why:** Reproducible (seed=42), controllable, no privacy issues, fast iteration
- **Impact:** Data 100% predictable, ideal for testing

### 3. Parquet in Production
- **Why:** Efficient, discoverable, standard format, typed, versioned
- **Impact:** CSV working file only; parquet is source of truth

### 4. Two Contrasting Asset Types
- **Why:** Tests model selection (easy vs hard), realistic energy cases
- **Impact:** Clear clustering (ev vs solar), enables per-type models

### 5. 14-Day Window
- **Why:** Captures weekly patterns (Mon×2), sufficient for features, quick
- **Impact:** Shows weekday/weekend, good for seasonal decomposition

---

## The Three Zones

### Working Notes (`working_notes/`)
**Local, untracked, ephemeral**

- Quick exploratory scripts and IPython sessions
- Half-baked ideas and failed experiments
- Scratch calculations and test data
- Debug outputs and temporary analysis
- **Not committed to git** — this is your scratch space

Use `working_notes/` when you're:
- Testing a hypothesis quickly
- Debugging a model
- Exploring a new library or approach
- Running calculations you might not keep

### Documentation (`docs/` or root `.md` files)
**Tracked, stable, reference**

- Solid analysis findings and writeups
- Architecture and design decisions (BLUEPRINTS.md, CLAUDE.md)
- Methodology notes and validated approaches
- Data dictionaries and schemas
- **Committed to git** — this is your project knowledge base

Use tracked notes when you're:
- Recording findings worth keeping
- Documenting a validated approach
- Writing for future-you or collaborators
- Creating material for GitHub Pages

### Source Code (`src/`)
**Production-ready modules**

- Clean, reusable forecasting models and pipelines
- Tested utilities and data processing
- Model training scripts
- **Committed to git** — this is your codebase

### Archive (`archive/`)
**Experimental and reference**

- Experimental implementations that didn't make it
- Reference scripts from other projects
- Historical explorations
- **Can be committed** — useful for "remember how we did this?"

---

## Workflow: IPython → Marimo → src/

### Phase 1: Exploration (IPython)

```bash
uv run ipython
```

- Rapid iteration and prototyping
- Live interaction with data and models
- Test ideas before formalizing
- Keep notes in `working_notes/` as you go
- **No commitment to polish or documentation**

### Phase 2: Consolidation (Marimo)

Once you've validated an approach:

```bash
uv run marimo edit
```

- Create a marimo notebook (`.md` format)
- Clean up and document the validated work
- Make it reproducible and clear
- This becomes a form of documented analysis
- **Shared reference, not just personal scratch**

Naming: `<topic>_exploration.md` or `<layer>_analysis.md`

Example:
- `forecast_uncertainty_exploration.md`
- `asset_profiling_analysis.md`
- `anomaly_detection_validation.md`

### Phase 3: Implementation (src/)

When ready for production use:

- Extract validated logic into clean modules
- Write tests
- Add to the forecasting pipeline
- **Committed, versioned, reusable**

---

## Tackling notes.md

The architecture in notes.md describes 16 layers and concepts. Approach them as independent investigations:

1. **Pick one layer or concept** from notes.md
2. **Explore it**: IPython + `working_notes/`
3. **Document it**: Marimo notebook + tracked findings
4. **Implement it**: Move to src/ if it's core to the pipeline
5. **Archive reference**: Save the exploratory work to `archive/` if useful

Example progression:

```
Layer 1: Basic Forecast
  ↓
Layer 2: Forecast Distribution (investigate in ipython)
  ↓
Create marimo notebook: "forecast_distribution_validation.md"
  ↓
Move validated code to src/forecasting/
  ↓
Archive the working notebooks in archive/
```

You can return to other layers anytime. This is iterative.

---

## File Organization Summary

```
time-series-forecasting/
├── README.md                    # Project overview
├── BLUEPRINTS.md                # This file — workflow guide
├── CLAUDE.md                    # AI collaboration guidelines
├── notes.md                     # Architecture & strategy reference
├── pyproject.toml               # Dependencies
├── .gitignore                   # Git exclusions
│
├── working_notes/               # LOCAL, UNTRACKED
│   ├── forecast_experiment_1.py
│   └── asset_profiling_scratch.py
│
├── docs/                        # TRACKED, SHARED
│   ├── findings/
│   ├── methodology/
│   └── data_dictionary.md
│
├── src/                         # PRODUCTION CODE
│   ├── forecasting/
│   ├── features/
│   └── utils/
│
└── archive/                     # REFERENCE & EXPERIMENTAL
    └── TimeSeries/
```

---

## Getting Started

1. **Set up**: `uv sync`
2. **Explore**: Pick a layer from notes.md, start in IPython
3. **Document**: Create a marimo notebook when you find something worth keeping
4. **Clean up**: Move solid code to src/ later
5. **Reference**: Archive interesting experiments

No pressure to have everything in src/ immediately. The workflow supports rapid exploration with clear documentation.
