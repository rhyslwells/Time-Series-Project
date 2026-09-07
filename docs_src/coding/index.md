# Coding

Implementation reference for this project's code. For the math and reasoning behind it, see
[Theory](../theory/index.md).

## Find an answer fast

| Question | Page |
|---|---|
| "How do I load the parquet data into `y_train` / `y_test`?" | [Framework usage](framework_usage.md#from-parquet-to-numpy) |
| "How do I compare SARIMA / ExpSmoothing / LightGBM?" | [Framework usage](framework_usage.md#comparing-all-models) |
| "How do I grid-search hyperparameters?" | [Framework usage](framework_usage.md#tuning) |
| "What's in `ForecastOutput` / `EvaluationMetrics`?" | [Framework usage](framework_usage.md#output-contracts) |
| "How is this documentation site built and deployed?" | [MkDocs setup](mkdocs.md) |
| "How do I add or export a notebook page?" | [Notebooks](../notebooks/notebooks.md#authoring-a-notebook-page) |

## Pages

- [Framework usage](framework_usage.md) — `ts_model_framework.py` and `ts_plots.py`: loading data, comparison, tuning, plotting, output contracts, and the known caveats (test-set tuning, MAPE scale, constant-width intervals).
- [MkDocs setup](mkdocs.md) — the `docs_src/` → `docs/` split, the deploy workflow, local build/serve, and marimo export.

A runnable version of the full compare → diagnose → tune → finalise flow is in
`working_notes/3_framework/example_model_comparison.py`.
