
# docs review notes

Two greps confirm a pattern worth naming up front: zero admonitions, zero images, zero tabbed blocks across all 23 files — despite admonition, pymdownx.details, attr_list and tasklist all being enabled in mkdocs.yml. And units split cleanly by section: synthetic_metering_data.md says kW 19 times and kWh never; metrics.md says kWh 14 times and kW never. Same quantity, two sections, never reconciled.

Diagnosis: what's actually wrong with the content
The individual pages are mostly well-written — diagnostics.md and models-decisions.md are genuinely good, with a consistent good-case/failure-case structure that earns its length. The problems are between pages, not within them.

1. The two halves of the site never meet. The data docs end at a parquet file. framework_usage.md begins at y_train = np.array([...]). Nothing bridges them — no page shows how to load metering_data.parquet, pick an asset, split train/test, and hand numpy to a model. That is the one thing every reader needs first, and it's the one thing missing. A runnable walkthrough already exists at working_notes/3_framework/example_model_comparison.py and is undocumented.

2. The project's central organizing idea is absent. The layered model — raw metering → forecasts → derived features → flexibility → optimization — is the stated architecture, and no page in docs_src/ states it. forecast_products.md gestures at the upper layers without ever naming the layer model, so its fifteen sections read as a list of loosely related ideas rather than a map of a system.

3. Nothing marks what's built versus what's imagined. forecast_products.md is almost entirely unbuilt. asset_profiling.md is unbuilt. One of the three findings entries says "not yet implemented" and the other two point at files that don't exist. Read cold, the site claims far more than the repo contains — and the reader has no way to tell which parts are safe to rely on.

4. Nav order inverts the reading order. Findings comes first and is the thinnest, least-validated section; Coding comes last and is where the how-to lives. The home page links four of the five sections and omits Coding entirely.

5. Entry-point quality is wildly uneven. theory/index.md has an excellent "find an answer fast" question→page table. data/index.md and findings/index.md are bare link lists; coding/index.md is two lines.

6. Notebook pages are wrappers with no content. sarima.md and ts_model_explorer.md are ~15 lines each: a link, an iframe, a run command. A reader must load an 85vh iframe just to find out whether the notebook is relevant, and there's no record of what either one concluded — so the analysis is trapped in an HTML artifact instead of being in the docs.

7. The two data pages overlap by about half. data_generation.md Step 1 and synthetic_metering_data.md both describe asset types, charging windows, and behavioral metrics. The clean split is: generation = process and formulas, description = resulting characteristics and fitness-for-use. Right now both do both, which is why the two pattern tables have drifted out of agreement.

8. kW vs kWh is genuinely ambiguous, not just inconsistent. metering_kwh is energy per half-hour, but daily_peak_kw is the max of it — so a column named _kw holds a kWh-per-30-min value, which is half the kW figure. The docs inherit that ambiguity and then use different units per section. This needs deciding once and applying everywhere.

9. Voice and heading case split by section. Theory and framework_usage use sentence case and an impersonal register; data/, findings/, notebooks/, and mkdocs.md use Title Case, with mkdocs.md in second person ("You Edit Source Files") and notebooks.md in first person plural ("as we explore").

10. mkdocs.md is mostly a generic MkDocs tutorial. How to edit, commit, push, add a page, write a relative link — none of it project-specific, and roughly half the page. The project-specific parts (the docs_src/docs split, the do-not-edit warning, the marimo export step, the workflow triggers) are the parts worth keeping.

Content plan (C1-C6)
These are independent of the correctness Phases 1-6 and best done after them, so I'm editing pages that already say true things.

C1 — Establish conventions, then apply them
The cheapest high-leverage step, because everything after depends on it.

Status admonitions. Adopt one convention using the already-enabled admonition extension: !!! success "Implemented", !!! note "Partially implemented", !!! warning "Proposed — not implemented". Apply to every page that mixes built and planned material — forecast_products.md, asset_profiling.md, findings/index.md. This is what lets the site keep its aspirational content honestly rather than deleting it.
Units. Decide metering_kwh semantics once (energy per 30-min interval), add a short units note to data/index.md, and apply consistently. Flag in the Phase 6 bug report that the daily_*_kw column names are arguably mislabeled.
Heading case and voice. Standardise on sentence case and the impersonal register of the theory pages — that's the stronger half of the site, so the weaker half moves to match it.
C2 — Fix the entry points and the reading path
index.md — rewrite as an actual front door: what the system is, what exists today, and an explicit reading path (Data → Theory → Coding → Notebooks). Add the missing Coding link. Keep the task-oriented spirit, but let the task list reflect what's covered.
Nav reorder in mkdocs.yml: Home → Data → Theory → Coding → Notebooks → Findings. Dependency order, and it stops the thinnest section from holding the prime slot.
data/index.md — give it the "find an answer fast" treatment that works so well in theory: a question→page table, plus the three-file pipeline at a glance. Reorder to pipeline order and make nav and index agree (they currently disagree with each other).
coding/index.md and findings/index.md — same treatment, scaled to their size.
C3 — Build the missing bridge (the highest-value addition)
New page, coding/end_to_end.md — "From parquet to forecast." Loads metering_data.parquet with polars, filters to one asset, splits train/test, converts to numpy, runs ModelComparison, reads the ranking, plots diagnostics. This is a write-up of the existing example_model_comparison.py, so it documents real code rather than inventing an interface.
Link it from index.md, data/index.md, and the top of framework_usage.md, which becomes the reference page that this walkthrough feeds into.
C4 — Give the theory section its spine
New page, theory/architecture.md — the layer model, each layer's inputs/outputs, the forecast output contract, and which layers exist today (layers 1-2) versus which are theory (3-5). Flagging this one: it's the largest piece of new content and it promotes the layer model out of .claude/CLAUDE.md into public docs. I'd propose it as description-only, changing nothing about the layers themselves — but say if you'd rather it stay internal.
forecast_products.md — keep every section, but group the fifteen into three bands (asset-level signals → portfolio and flexibility → model health and monitoring), open with a sentence tying them to the layer model, and status-mark them. It's the most interesting page on the site and currently the most orphaned.
metrics.md — commit to reference-first ordering with a one-line orienting sentence, so the layout is deliberate rather than accidental.
models-decisions.md — fix the decision rule in "Which model is best?": with three checks, "3+ pass → deploy" and "2+ fail → tune" leave the one-failure case undefined. Small, but it's the page's core algorithm.
diagnostics.md — strong already. Add a "confirm with" line to each failure mode naming the metric or check that distinguishes it, so the page works without the reader having a plot in front of them. Longer term this page wants example figures; I'd note that rather than generate them now.
C5 — Make the notebook and findings sections carry content
sarima.md and ts_model_explorer.md — add a short preamble to each: the question it investigates, the setup, and what it concluded, above the iframe. Roughly a screen of text apiece, and it moves the analysis out of the HTML artifact and into the searchable, diffable docs.
notebooks.md — split its two jobs. Reader-facing content (what notebooks exist, how to run them) stays; the authoring template and export procedure move to a short authoring section, or out to .claude/ alongside the other authoring policy.
findings/index.md — restructure around what has actually been established, with status markers, so an honestly small Findings section reads as early-stage rather than broken.
C6 — Trim the redundancy
data_generation.md / synthetic_metering_data.md — enforce the split in point 7: process and formulas in one, resulting characteristics and fitness-for-use in the other, cross-linked once. Removes the duplicated pattern tables that drifted apart in the first place.
mkdocs.md — cut the generic MkDocs tutorial, keep the project-specific mechanics. Roughly halves the page.
Build/serve instructions — currently in both mkdocs.md and notebooks.md, with conflicting commands. mkdocs.md owns it; notebooks.md links to it.
docs_src/README.md — as in Phase 5, reduced to orientation with the policy left to .claude/DOCUMENTATION.md.
Two things I'd want your call on before starting: whether C4's architecture page should exist in public docs, and whether C5's notebook write-ups should be my summaries of the exported notebooks or something you'd rather draft, since they're the one place I'd be making claims about what the analysis actually found.

My suggested order is Phases 1-6 first, then C1 → C2 → C3, and treat C4-C6 as a second pass — C3 alone closes the biggest gap in the site.