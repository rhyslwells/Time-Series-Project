# MkDocs Documentation Setup 2

How the documentation site is built and deployed.

## Overview

This project uses MkDocs with Material theme to generate a documentation site published on GitHub Pages.

- Source: `docs_src/` (markdown files)
- Built output: `docs/` (HTML, CSS, JS)
- Published: https://rhyslwells.github.io/time-series-project
- Deployment: Automatic via GitHub Actions

## Directory Structure

```
project/
├── docs_src/                    Source markdown files
│   ├── index.md
│   ├── architecture.md
│   ├── findings/
│   ├── methodology/
│   ├── data/
│   ├── notebooks.md
│   └── coding/
│
├── docs/                        Built HTML (auto-generated)
│   ├── index.html
│   ├── assets/
│   └── [all built files]
│
├── mkdocs.yml                   MkDocs configuration
└── .github/workflows/deploy.yml Auto-build workflow
```

## Workflow: Edit -> Build -> Deploy

### 1. You Edit Source Files

Edit markdown files in `docs_src/`:

```bash
# Edit any markdown file
vim docs_src/findings/my_finding.md
```

### 2. Commit and Push

```bash
git add docs_src/
git commit -m "Document new finding"
git push origin main
```

### 3. GitHub Actions Builds Automatically

The workflow (`.github/workflows/deploy.yml`):
- Triggers on push to `main` branch
- Installs mkdocs and Material theme
- Builds from `docs_src/` to `docs/`
- Commits built files back to repo
- Pushes to `main`

### 4. GitHub Pages Deploys

GitHub Pages automatically serves the latest `docs/` folder at:
```
https://rhyslwells.github.io/time-series-project
```

No manual steps needed. Just edit, commit, push.

## Building Locally

To test changes before pushing:

```bash
# Install dependencies (if not already done)
uv sync --all-extras

# Build the site
uv run mkdocs build

# Or serve with live reload
uv run mkdocs serve
```

Then visit `http://localhost:8000`

## Configuration

### mkdocs.yml

Key settings:

```yaml
site_name: Time Series Forecasting
docs_dir: docs_src           # Read markdown from here
site_dir: docs               # Build to here
theme:
  name: material             # Material design theme

nav:                          # Navigation structure
  - Home: index.md
  - Findings: findings/index.md
  - etc...
```

### GitHub Actions Workflow

File: `.github/workflows/deploy.yml`

Triggers on:
- Push to `main` branch
- Changes to `docs_src/**` or `mkdocs.yml`

Does:
1. Checkout code
2. Install Python 3.10
3. Install mkdocs and mkdocs-material
4. Run `mkdocs build` (reads `docs_src/`, writes `docs/`)
5. Commit and push `docs/` folder

## Do NOT Manually Edit docs/

The `docs/` folder is auto-generated. Any manual changes will be overwritten on the next build.

Edit `docs_src/` instead.

## Adding New Pages

1. Create markdown file in `docs_src/`:
   ```bash
   # Example: new finding
   touch docs_src/findings/new_discovery.md
   ```

2. Add to navigation in `mkdocs.yml`:
   ```yaml
   nav:
     - Findings:
         - findings/index.md
         - New Discovery: findings/new_discovery.md
   ```

3. Write content, commit, push:
   ```bash
   git add docs_src/
   git commit -m "Add new discovery page"
   git push
   ```

4. Workflow auto-builds → site updates in ~1-2 minutes

## Linking Between Pages

Use relative markdown links:

```markdown
# From findings/my_finding.md to methodology page
See [Model Selection](../methodology/model_selection.md)

# From docs_src root to marimo notebook in repo root
Explore the [investigation](../asset_profiling_exploration.md)
```

## Troubleshooting

### Build fails: "unknown field `python-version`"

Your `pyproject.toml` has invalid `[tool.uv]` config. Remove it—`requires-python` in `[project]` is sufficient.

### Links show as broken in built site

Check that:
- File path is correct
- Using relative links from the markdown file location
- File exists in `docs_src/`

### Site doesn't update after push

Check GitHub Actions:
1. Go to repo → Actions tab
2. Look for "Deploy Docs" workflow
3. Check if it passed or failed
4. Logs show what went wrong

## Dependencies

Installed via `pyproject.toml`:

```
mkdocs>=1.5.0
mkdocs-material>=9.0.0
```

Install with:
```bash
uv sync --all-extras
```

### Important: MkDocs 2.0 Breaking Changes

MkDocs 2.0 (currently in development) will introduce backward-incompatible changes:

- Plugin system will be removed
- Theme overrides will break
- No migration path for existing projects
- Material for MkDocs will not be compatible

**Action needed**: Before MkDocs 2.0 is released, we need to either:
1. Lock to MkDocs 1.x
2. Migrate to a different documentation tool
3. Wait for Material for MkDocs to release a 2.0-compatible version

Currently using MkDocs 1.5.x, so there's time to plan. Monitor:
- https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/

## GitHub Pages Settings

Verify in repo Settings → Pages:
- Source: Deploy from branch
- Branch: `main`
- Folder: `/docs`

Everything else is automatic.
