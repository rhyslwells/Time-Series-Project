# MkDocs setup

How the documentation site is built and deployed. This covers the project-specific mechanics
only; for general MkDocs usage see the [Material docs](https://squidfunk.github.io/mkdocs-material/).

## The source / output split

- Source: `docs_src/` — the markdown you edit.
- Built output: `docs/` — auto-generated HTML. **Do not edit `docs/` by hand**; the next build overwrites it.
- Config: `mkdocs.yml` (nav, theme, markdown extensions).
- Published at <https://rhyslwells.github.io/time-series-project> from the `main` branch `/docs` folder.

## Deploy

`.github/workflows/deploy.yml` runs on push to `main`, **only when `docs_src/**` or
`mkdocs.yml` changed**. It installs mkdocs + plugins, runs `mkdocs build`, and commits the
regenerated `docs/` back to `main` with `[skip ci]`. No manual deploy step.

A push that changes only `src/` or notebooks will not rebuild the site — touch `mkdocs.yml`
or a `docs_src/` file if you need to force it.

## Build locally

```bash
uv sync --group docs        # matches the pyproject "docs" dependency group
uv run mkdocs build         # writes docs/
uv run mkdocs serve         # live-reload preview at http://localhost:8000
```

`mkdocs serve` is the command to use while writing — it is the single source for build/serve
instructions (the notebook page links here rather than repeating them).

## Marimo notebooks

The notebook pages embed marimo exports. The `mkdocs-marimo` plugin renders `.py` marimo
files, and the `*_export.html` artifacts under `docs_src/notebooks/` are committed alongside
them. Regenerate an export with `marimo export html <notebook>.py -o <notebook>_export.html`
before committing a notebook change. See [Notebooks](../notebooks/notebooks.md).

## Adding a page

1. Create the markdown file under `docs_src/`.
2. Add it to `nav:` in `mkdocs.yml`.
3. Commit and push `docs_src/` + `mkdocs.yml` together.

## Linking between pages

Use relative markdown links **to the `.md` file**, resolved from the current file's location:

```markdown
See [Decisions](../theory/models-decisions.md)
Explore the [SARIMA notebook](../notebooks/sarima.md)
```

`mkdocs.yml` sets `validation.links.anchors: warn`, so a broken anchor shows up in the build log.

## Version pin

`pyproject.toml` pins `mkdocs>=1.5.0,<2.0.0` and `mkdocs-material<10`. Material prints a
build-time warning about backward-incompatible changes planned for MkDocs 2.0; the `<2.0.0`
pin is what keeps the current build stable, and the pin should stay until Material ships a
2.0-compatible release.

## Troubleshooting

| Symptom | Check |
|---|---|
| Site doesn't update after push | repo → Actions → "Deploy Docs" — did it run, did it pass? A `src/`-only push won't trigger it. |
| Broken-link warnings in the build | relative path from the current file; target `.md` exists in `docs_src/` |
| GitHub Pages not serving | repo → Settings → Pages: source "Deploy from branch", branch `main`, folder `/docs` |
