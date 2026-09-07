# docs_src/

Source markdown for the documentation site. Edit files here; `mkdocs.yml` controls the nav.
The built HTML in `docs/` is generated — do not edit it by hand.

Sections: `data/`, `theory/`, `coding/`, `notebooks/`, `findings/`.

This file is excluded from the build (`exclude_docs` in `mkdocs.yml`). The documentation
policy — what belongs here versus in `working_notes/` — lives in `.claude/DOCUMENTATION.md`.
Build and deploy mechanics are in [coding/mkdocs.md](coding/mkdocs.md).
