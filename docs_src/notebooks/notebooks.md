# Exploration notebooks

Interactive marimo notebooks documenting ongoing investigations. They are living documents
that evolve as the analysis develops, and are stored in `src/notebooks/`.

## Available notebooks

| Notebook | What it covers |
|---|---|
| [SARIMA forecasting pipeline](sarima.md) | End-to-end SARIMA forecast for one asset, with prediction intervals and event probabilities |
| [Model comparison and interpretation](ts_model_explorer.md) | SARIMA vs Exponential Smoothing vs LightGBM on one split, with a diagnostic-plot reading guide |

## Running a notebook

```bash
uv run marimo edit src/notebooks/<notebook>.py
```

For the docs build and preview commands, see [MkDocs setup](../coding/mkdocs.md#build-locally).

---

## Authoring a notebook page

### Export the notebook

Once a notebook is ready to embed, export it as a static HTML artifact:

```bash
uv run marimo export html src/notebooks/<notebook>.py \
  -o docs_src/notebooks/<notebook>_export.html --include-code -f
```

The `*_export.html` files are committed. Regenerate the export after any notebook edit.

### Add the page

Create `docs_src/notebooks/<notebook>.md`. The built page lives at
`notebooks/<notebook>/index.html` and the export sits one level up at
`notebooks/<notebook>_export.html`, so **both the link and the iframe use `../`**:

~~~markdown
# Notebook title

A short preamble: the question, the setup, and what the notebook shows.

[Open the full-screen version](../<notebook>_export.html){ target="_blank" rel="noopener" }

<iframe
  src="../<notebook>_export.html"
  title="Notebook title"
  loading="lazy"
  style="width: 100%; height: 85vh; border: none;">
</iframe>

Run locally:
```bash
uv run marimo edit src/notebooks/<notebook>.py
```
~~~

Then add the page to `nav:` in `mkdocs.yml`.

### Notebook structure

1. Overview — the investigation
2. Questions or hypotheses
3. Analysis with visualisations
4. Findings — conclusions
5. Next steps

## Resources

- [marimo examples](https://github.com/marimo-team/marimo/tree/main/examples)
- [mkdocs-marimo getting started](https://github.com/marimo-team/mkdocs-marimo/tree/main/docs/getting-started)
- [marimo blocks reference](https://marimo-team.github.io/mkdocs-marimo/getting-started/blocks/)
