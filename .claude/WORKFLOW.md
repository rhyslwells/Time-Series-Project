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

The system progresses through these layers:

```
Raw metering → Forecasts → Derived features → Flexibility/uncertainty → Optimization
```

### Layer Separation

- Maintain clear boundaries between forecasting layers
- Treat forecasts as reusable intermediate products (not final outputs)
- Use standard forecast contracts: `asset_id`, `timestamp`, `prediction`, `uncertainty`, `model_version`
- Support swappable forecasting models

### Architectural Decisions

Before implementing changes to:
- Splitting/combining forecast layers
- Forecast output contracts
- New model types or aggregation approaches
- Major restructuring of src/

**Always ask first.** Otherwise proceed with implementation based on framework in notes.md.

## Behavior When Uncertain

1. Analyze existing repository structure
2. Infer the most conservative change
3. Present assumptions explicitly
4. Ask for clarification before introducing architectural changes
5. Default to preserving existing patterns and conventions
