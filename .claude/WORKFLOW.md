# Workflow & Architectural Patterns

## Exploration → Consolidation → Implementation

1. **Exploration** (IPython)
   - Test hypotheses in `working_notes/`
   - Validate approaches with quick iterations
   - Write disposable scripts for rapid testing

2. **Consolidation** (Marimo)
   - Document validated work in reproducible notebooks
   - Source of truth: `docs_src/` and `working_notes/`
   - Create reusable reference materials

3. **Implementation** (src/)
   - Clean, reusable modules for production
   - Stable APIs and standardized contracts
   - Ready for integration into pipelines

## Forecasting System Layers

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full 8-layer model and diagram. In short:

- Maintain clear boundaries between forecasting layers
- Treat forecasts as reusable intermediate products (not final outputs)
- Use standard forecast contracts: `asset_id`, `timestamp`, `prediction`, `uncertainty`, `model_version`
- Support swappable forecasting models

### Architectural Decisions

See [CLAUDE.md](CLAUDE.md#key-architectural-decisions) for what to ask about before changing, and its "Behavior When Uncertain" section for how to proceed otherwise.
