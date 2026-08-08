# Blueprints: Workflow & Exploration

How this project is structured and how work flows through it.

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
