# Repository Structure & Surface Area Policy

## Directory Organization

Keep the core `src/` directory focused and clean. Use `archive/` for experimental work.

```
Time-Series-Project/
├── src/                          # Production code
│   ├── __init__.py
│   └── data/                      # Generated parquet data
│
├── working_notes/                 # Exploration & documentation
│   ├── todos.md                   # Task tracking
│   └── 1_produce_data/            # Data generation
│       ├── notes.md
│       ├── generate_data.py
│       ├── inspect_data.py
│       └── metering_data_raw.csv
│
├── archive/                       # Experimental work (old code)
├── docs_src/                      # Documentation source (tracked)
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
│   └── projects/                  # Session memory
│
└── pyproject.toml, mkdocs.yml, README.md
```

## Surface Area Policy

Before creating new files or directories:

1. **src/** → only stable, reusable production code (models, pipelines, utilities)
2. **archive/** → freely add exploratory work, reference implementations, scripts
3. **working_notes/** → temporary exploration and quick iterations (not committed)
4. **docs_src/** → solid, tracked documentation and methodology

Do not create supplementary scaffolding or examples unless explicitly requested.

## Key Production Files

| File | Purpose |
|------|---------|
| `metering_data.parquet` | Raw 30-min metering (10,080 rows) |
| `daily_metrics.parquet` | Daily aggregates (210 rows) |
| `ramp_rates.parquet` | Asset ramp statistics (15 rows) |
| `behavioral_fingerprints.parquet` | Asset features (15 rows) |
