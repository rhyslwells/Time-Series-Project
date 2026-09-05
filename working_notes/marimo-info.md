https://github.com/marimo-team/marimo/tree/main/examples

Look into this: https://marimo-team.github.io/mkdocs-marimo/getting-started/blocks/


Folder contains examples of marimo scripts and how to use them. It is a good starting point for building your own marimo scripts
https://github.com/marimo-team/marimo/tree/main/examples

Folder for mkdocs use of marimo 
https://github.com/marimo-team/mkdocs-marimo/tree/main/docs/getting-started


uv sync --group docs

uv run marimo export html working_notes/2_basic_forecasting/sarima_marimo.py -o docs_src/notebooks/sarima_marimo_export.html --no-include-code -f 2>&1

store the html here: C:\Users\RhysL\Desktop\Time Series Project\docs_src\notebooks

and a md file link it there too.


we store the files here: C:\Users\RhysL\Desktop\Time Series Project\src\notebooks

===
ld is clean. Summary of the final state:

working_notes/2_basic_forecasting/sarima_marimo.py — reverted to local parquet read (no network dependency for local dev), and the three plot cells now end with a bare fig reference instead of fig.show() so marimo's display system actually captures them (that omission was a real bug — outputs were being silently dropped even in normal marimo edit use, not just in export).
docs_src/notebooks/sarima_marimo_export.html — static export (marimo export html --include-code) containing full source + printed outputs + all three plotly figures. This is a build artifact, regenerate it after any notebook edit with:

cd working_notes/2_basic_forecasting
uv run marimo export html sarima_marimo.py -o "../../docs_src/notebooks/sarima_marimo_export.html" --include-code -f
docs_src/notebooks/sarima_forecasting.md — embeds that static file via a plain <iframe src="../sarima_marimo_export.html"> (path accounts for mkdocs-material's directory-URL nesting), with a note explaining why it's not a live embed.
Run uv run mkdocs serve and check the page — it should now show real code, print output, and the three plotly charts, fully static (no WASM loading spinner