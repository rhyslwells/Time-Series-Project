# Exploration Notebooks

Interactive marimo notebooks documenting ongoing investigations. These are living documents that evolve as we explore.

Notebooks are stored in `src/notebooks/`.

## How to Use Notebooks

### Editing Locally

Open any marimo notebook for interactive editing:

```bash
uv run marimo edit src/notebooks/notebook_name.py
```

### Exporting to Documentation

Once a notebook is complete, export it as a static HTML artifact for embedding in documentation:

```bash
cd src/notebooks
uv run marimo export html notebook.py -o "../../docs_src/notebooks/notebook_export.html" --include-code -f
```

Then create a markdown file in `docs_src/notebooks/` that embeds it:

```markdown
# Notebook Title

[Open the full-screen version](notebook_export.html){ target="_blank" rel="noopener" }

<iframe
  src="notebook_export.html"
  title="Notebook title"
  loading="lazy"
  style="width: 100%; height: 85vh; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px;">
</iframe>

To run this notebook locally, use the following command:
```bash
cd src/notebooks
uv run marimo edit notebook.py
```
```

### Workflow After Export

After exporting, the HTML file becomes a build artifact — regenerate it after any notebook edits using the export command above.

## Notebook Structure

When creating a new notebook, follow this structure:

1. **Overview** section explaining the investigation
2. **Questions** or hypotheses being explored
3. **Analysis** with visualizations
4. **Findings** section summarizing conclusions
5. **Next steps** for follow-up work

## Resources

- [marimo Examples](https://github.com/marimo-team/marimo/tree/main/examples) — Starting point for building your own marimo scripts
- [mkdocs-marimo Getting Started](https://github.com/marimo-team/mkdocs-marimo/tree/main/docs/getting-started) — Integration with documentation sites
- [marimo Blocks Documentation](https://marimo-team.github.io/mkdocs-marimo/getting-started/blocks/) — Reference for marimo cell and block features

## Build & Serve Locally

Ensure docs dependencies are installed:

```bash
uv sync --group docs
uv run mkdocs serve
```
