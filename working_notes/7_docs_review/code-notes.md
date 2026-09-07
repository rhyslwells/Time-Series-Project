# Phased approach

Plan: docs_src review and correction
Phase 1 — Unbreak rendering (mkdocs.yml + one new file)
Change	Detail
Add pymdownx.arithmatex: generic: true	Plus extra_javascript for a new docs_src/javascripts/mathjax.js config shim and the MathJax v3 CDN. This is the standard Material recipe; pymdown-extensions already ships with mkdocs-material, so no deploy-workflow change is needed.
Fix the mermaid fence	Add format: !!python/name:pymdownx.superfences.fence_code_format so Material loads its bundled mermaid runtime.
Favicon	Remove theme.favicon (pointing at a nonexistent _static/) so Material falls back to its default.
Duplicate extension	Drop the second pymdownx.tasklist.
Orphan README	Add exclude_docs: README.md rather than a nav entry — see Phase 5.
I'll verify both fixes by running uv run mkdocs build and grepping the output for rendered math and a mermaid script tag, not just assuming.

Phase 2 — Correct the data docs
data/data_generation.md

mean_ramp_kw: correct to the signed mean of ramp_kw, matching generate_daily_metrics.py:35. Note the consequence — for a series that returns to a similar level daily, the signed mean sits near zero, so max_abs_ramp_kw is the ramp-magnitude column. Add a line noting this looks like a code bug and pointing to Phase 6.
Reclassify metering_data_raw.csv as a committed intermediate (stage 2's input), not "temporary".
Add a mermaid diagram of the three-stage pipeline, per the project's diagram policy — this page currently describes it only in prose while the downstream page has the diagram.
data/synthetic_metering_data.md — the most error-dense page:

Rewrite the EV pattern table from the generator: base load 0.5 across all hours, +2.5 for 06:00-09:00, +3.0 for 16:00-21:00, +1.5 for 22:00-02:00, all multiplied by the weekday factor. Present these as the additive components and the resulting value, since the current table's per-window "avg" figures are components read as outcomes.
Correct the noise description: Gaussian, unbounded, std = 10% of mean positive pattern (EV) and 8% of max(solar)+0.5 (solar) — not "±5%".
Fix the negative-value contradiction (2,767 of 10,080 is 27.5%, not 18.2%).
Add a documented caveat for generate_raw_data.py:44: (day_of_week < 5).all() collapses to a constant 0.85, so solar assets have no weekday/weekend variation. Correct the "Weekly Patterns … All assets" claim accordingly.
Replace ✓ glyphs with text; fix the Windows path.
Any remaining figure I cannot derive from the generator source (mean load, CV, peak-ratio, min/max) gets marked as an indicative summary rather than silently kept — per your instruction I won't recompute them.
Phase 3 — Realign the theory section to this project's assets
theory/metrics.md — rebuild the range table around ev_charging and solar_battery, at this dataset's kW scale rather than the current 0.5-5 kWh. Add the solar case explicitly, including why MAPE is unusable for a zero-crossing net exporter (the denominator passes through zero, not merely near it — a stronger caveat than the existing generic "blows up near zero"). Fix "ranges below" → above. Correct the coverage sanity check to the 0-100 scale the framework actually returns.

theory/models.md — replace the "typical winner" column, which no repo evidence supports, with a when to reach for this column. Fix "pitfall above" → below. Restate the complexity budget so the n/10 rule and the (1,1,1)x(1,1,1,48) example agree.

theory/diagnostics.md — add a note to "Flat width" that for ExponentialSmoothing and LightGBM a constant width is structural, not a fitting failure (ts_model_framework.py:175, :273), so only SARIMA's width is diagnostic here. This is the single most misleading gap in the theory section as it stands.

theory/forecast_products.md — repoint behavioral_fingerprints.parquet at daily_metrics.parquet; fix the "regime that regime" typo.

theory/models-decisions.md — state MAPE and coverage thresholds in units matching what the code returns, with a pointer to the Phase 6 bug list; retarget the "model-specific tuning directions" link to the per-model sections rather than the comparison table.

Phase 4 — Correct the coding docs
coding/framework_usage.md — replace pandas with polars and add scipy; say evaluate_all() returns a polars DataFrame; fix the deploy snippet's pi_coverage > 0.75 to the 0-100 scale; import ModelEvaluator in the refit example; add a short "what each model's interval actually is" note (SARIMA from the state-space CI, the other two from a single residual std).

coding/mkdocs.md — drop the stray "2" from the title; fill or remove the empty ## Configuration heading; fix both broken link examples in the very section that teaches linking; reconcile uv sync --all-extras with notebooks.md's uv sync --group docs; correct the dependency list; document that the workflow only fires on docs_src/** and mkdocs.yml. Remove the "MkDocs 2.0 Breaking Changes" section — I can't substantiate any of its four claims, and unsupported alarm in a setup guide is worse than silence. If you'd rather keep a pointer, I'll reduce it to one neutral line noting the version is pinned <2.0.0 in pyproject.

Phase 5 — Findings, notebooks, entry points
findings/asset_profiling.md — retitle without "DRAFTING:", open with an explicit status line stating no clusters have been computed and the table is a proposed scheme, and repair the three dangling references.
findings/index.md — repoint the SARIMA notebook to src/notebooks/sarima.py; remove or mark the two nonexistent exploration files; align the section's framing with the status markers above.
notebooks/sarima.md, ts_model_explorer.md — fix the full-screen links to ../*_export.html to match the working iframes and the verified build layout.
notebooks/notebooks.md — fix the nested-fence corruption (outer fence to ~~~~), and correct the iframe path it teaches.
index.md — keep the task-oriented spirit, but describe what the site actually contains. Move trend removal, deseasonalization, walk-forward validation, and anomaly detection out of "concepts explored" into the future list where they belong, since none are covered.
docs_src/README.md — excluded from the build in Phase 1; trimmed to a short orientation note, with the policy content left to .claude/DOCUMENTATION.md rather than duplicated.
Phase 6 — Report, no code changes
A summary listing the suspected code bugs for your decision: mean_ramp_kw signed vs absolute, MAPE stored as a fraction but printed with % (ts_model_framework.py:45), the (day_of_week < 5).all() collapse, and the unused n_assets parameter.

Not in scope unless you say otherwise: any edit under src/, regenerating parquet files, and .claude/CLAUDE.md — though note it points at src/example_model_comparison.py, which actually lives at working_notes/3_framework/example_model_comparison.py.

Say the word and I'll start with Phase 1, since a working build is what lets me verify the rest



# Initial ouput 


I've read all 23 files in docs_src/ and cross-checked them against src/, mkdocs.yml, pyproject.toml, the deploy workflow, and the built output in docs/. Here's what I found, then four questions before I write the plan.

Tier 1 — The site doesn't render what the docs are written in
These are the highest-impact issues and they're all in mkdocs.yml, not docs_src/:

All LaTeX renders as literal text. markdown_extensions has no pymdownx.arithmatex and there's no MathJax extra_javascript. Verified in the built site — docs/theory/metrics/index.html contains the raw string \text{MAE} = \frac{1}{n} \sum.... Every formula in metrics.md, models.md, and forecast_products.md is currently unreadable on the published site.
The mermaid diagram doesn't render. The superfences custom fence at mkdocs.yml:29-32 lacks format: !!python/name:pymdownx.superfences.fence_code_format, so Material never loads its mermaid bundle. The built page emits class="mermaid" with no mermaid runtime — the feature_engineering.md pipeline diagram shows as raw text.
theme.favicon: _static/favicon-32x32.png — docs_src/_static/ doesn't exist.
pymdownx.tasklist is listed twice.
docs_src/README.md isn't in nav — it builds as an orphan page.


Tier 2 — Docs contradict the code
data_generation.md:63 documents mean_ramp_kw = MEAN(|ramp_kw|), but generate_daily_metrics.py:35 computes pl.col('ramp_kw').mean() — the signed mean. Only max_abs_ramp_kw uses abs. The schema gloss "Average absolute change between intervals" is wrong for the same reason.
framework_usage.md:152 lists dependencies as "numpy, pandas" — the framework imports polars and scipy, never pandas. Directly contradicts the polars-only standard.
framework_usage.md:131 gates deployment on final_metrics.pi_coverage > 0.75, but ts_model_framework.py:304 returns coverage as a percentage 0-100 — so that condition is always true. metrics.md:20 has the same unit trap (assert 0.7 < coverage < 0.95).
MAPE scale is inconsistent in the code itself: ts_model_framework.py:301 stores sklearn's fraction (0-1) unscaled, while __repr__ prints it with a % suffix. Docs then quote thresholds like "MAPE < 15%".
Undocumented and material: only SARIMA produces a time-varying interval. ExponentialSmoothing (:175) and LightGBM (:273) return np.full_like — a constant width by construction. Yet diagnostics.md has a whole "Flat width" failure-mode section that will fire on two of three models no matter what.
forecast_products.md:51 cites behavioral_fingerprints.parquet "in the data pipeline" — no such file; data_generation.md:163 documents it being consolidated away.
synthetic_metering_data.md has several claims the generator doesn't support: night charging is 22:00-02:00 in code not 21:00-02:00; there's no distinct "02:00-06:00 overnight 0.1 kW" band (base load is 0.5 throughout); noise is Gaussian at 10%/8% std, not "±5%"; and the negative-value line says "2,767 records (18.2%)" when 2,767/10,080 is 27.5% — internally contradictory regardless of the data. Also, generate_raw_data.py:44 uses (day_of_week < 5).all() over the whole array, so it's always False and solar assets have no weekday/weekend variation — contradicting the doc's "Weekly Patterns … All assets".


Tier 3 — Broken links and stale paths
findings/index.md:10 points at working_notes/2_basic_forecasting/sarima_marimo.py; actual file is src/notebooks/sarima.py. Lines 17 and 24 reference asset_profiling_exploration.md and forecast_uncertainty_exploration.md — neither exists.
asset_profiling.md:41-43 — same missing notebook, plus bare "Methodology: Feature Engineering" / "See Model Selection" that aren't links to anything.
mkdocs.md:128 teaches ../theory/model-decisions.md (real file is models-decisions.md) and notebooks/sarima_forecasting.md (real file is notebooks/sarima.md) — a how-to-link section whose examples are both broken links.
sarima.md:3 and ts_model_explorer.md:3: the "Open the full-screen version" link uses sarima_export.html while the iframe correctly uses ../sarima_export.html. Built layout is docs/notebooks/sarima/index.html beside docs/notebooks/sarima_export.html, so the plain link 404s.
notebooks.md:28-45: the ```markdown example block contains a nested ```bash fence, which closes the outer fence early and mangles the rest of the section. It also teaches the broken notebook_export.html iframe path from #16.


Tier 4 — Accuracy, structure, tone
Asset-taxonomy mismatch runs through the theory section. metrics.md tabulates expected ranges by Residential / Commercial / EV Charging and models.md names "typical winner: residential" — but this project's assets are ev_charging and solar_battery. Solar is absent entirely, despite being the case where MAPE is meaningless (values cross zero) and where the ranges matter most. The kWh magnitudes (0.5-5) also sit well above this dataset's scale.
Two cross-references point the wrong direction: metrics.md says "the asset-type ranges below" (they're above), models.md says "see pitfall above" (it's below).
models.md:34: the complexity budget states parameters should stay under n/10, then says "keep p+q < 48" — a far weaker constraint than the 4-parameter example that follows implies. Confusing as written.
mkdocs.md: title has a stray "2"; ## Configuration is an empty heading; the dependency list shows two packages while pyproject's docs group has ~25; it says uv sync --all-extras while notebooks.md:72 says uv sync --group docs; and it doesn't mention the workflow only triggers on docs_src/** + mkdocs.yml paths.
mkdocs.md:169-184 — the "MkDocs 2.0 Breaking Changes" section asserts the plugin system will be removed, theme overrides will break, there's no migration path, and Material won't be compatible. I can't substantiate any of that from the repo, and it's prominent, alarming, and reads as speculation.
asset_profiling.md is titled "DRAFTING:" and presents a 3-cluster table with per-cluster "Best Model" assignments as settled results — but nothing in the repo computes clusters, and it lives under Findings, which findings/index.md:3 defines as "validated discoveries".
index.md frames the site as "a curated collection of production-ready code patterns… users land here, find their task, copy the pattern" and lists trend removal, deseasonalization, walk-forward validation, and anomaly detection as "concepts explored" — none of those appear anywhere in the docs or code.
Smaller: synthetic_metering_data.md uses ✓ glyphs (CODING_STANDARDS forbids emoji in docs) and a Windows path src\data\...; forecast_products.md:47 has "a regime that regime it doesn't understand"; data_generation.md:34 calls metering_data_raw.csv a "temporary working file" though it's committed and is stage 2's input; docs_src/README.md is author-facing meta-guidance that duplicates .claude/DOCUMENTATION.md.