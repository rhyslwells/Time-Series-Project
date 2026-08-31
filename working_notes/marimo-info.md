https://github.com/marimo-team/marimo/tree/main/examples

Look into this: https://marimo-team.github.io/mkdocs-marimo/getting-started/blocks/


Folder contains examples of marimo scripts and how to use them. It is a good starting point for building your own marimo scripts
https://github.com/marimo-team/marimo/tree/main/examples

Folder for mkdocs use of marimo 
https://github.com/marimo-team/mkdocs-marimo/tree/main/docs/getting-started


uv sync --group docs

uv run marimo export html working_notes/2_basic_forecasting/sarima_marimo.py -o docs_src/notebooks/sarima_marimo_export.html --no-include-code -f 2>&1

