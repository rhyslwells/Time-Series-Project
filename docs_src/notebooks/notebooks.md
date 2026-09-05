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

[Full notebook with code and outputs](../notebook_export.html)

<iframe src="../notebook_export.html" style="width:100%;height:600px;border:none;"></iframe>
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

## Available Notebooks in `src/notebooks/`

Add notebooks here with links and descriptions:


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
