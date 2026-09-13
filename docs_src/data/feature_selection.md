# Feature selection

!!! warning "Proposed — not implemented"
    No feature-selection step exists in the pipeline yet. This page records the intended
    approach for when one is added.

The goal is to keep the features that contribute to predictive power in
[`metering_data_with_features.parquet`](feature_engineering.md) and drop redundant or
irrelevant ones, rather than feeding every `feat_*` column to a model unfiltered.

## Planned approach

Start with the simplest automatic method — no manual/human-in-the-loop selection step — using
feature importance from [`LightGBMModel`](../theory/models.md#lightgbm-gradient-boosting),
the framework's tree-based model. More complex selection methods (e.g. permutation
importance, recursive elimination) are left for later notebooks once a baseline is in place.

See [Framework usage](../coding/framework_usage.md) for how `LightGBMModel` is instantiated
and fit; a feature-importance extraction would build on that existing entry point.
