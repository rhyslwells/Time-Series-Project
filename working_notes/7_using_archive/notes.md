

# Prompt 2:



# Case 1: DONE

 C:\Users\RhysL\Desktop\Time Series Project\archive\ML_Tools-TimeSeries\out-of-sample rolling forecast evaluation.py

Consider this script:

C:\Users\RhysL\Desktop\Time Series Project\archive\ML_Tools-TimeSeries\out-of-sample rolling forecast evaluation.py

Is there anything in script in here

C:\Users\RhysL\Desktop\Time Series Project\src


that is used throughout that could benifit from the reasoning contained withing it?


# Base Prompt

Consider this script:

<ARCHIVE_SCRIPT_PATH>

Compare its reasoning against `src/` and `docs_src/theory/`:

1. Does anything in `src/` do the same job as this script but miss reasoning it contains
   (an evaluation method, a metric, a diagnostic, a decision rule)? Name the specific
   file/function gap, not just the topic.
2. Is the gap real, or does `src/` already cover it under different naming — check before
   proposing anything.
3. **Sense-check the archive script itself as a time series expert, before porting
   anything.** It was written against different data (frequency, length, seasonality,
   noise) than this project's 30-min, 14-day, two-asset-type series. Identify what
   carries over unchanged, what needs adapting, and what doesn't apply at all - e.g.
   expanding-window refit cost for computationally heavy models (SARIMA) at 30-min
   granularity, whether there's enough history for the technique to be statistically
   meaningful here (metrics.md already flags 14 days as barely enough for weekly
   seasonal-naive), and whether the script's implicit assumptions (stationarity,
   single seasonal period, symmetric errors) hold for this data. Don't port the logic
   uncritically just because it's more rigorous-looking.
4. If real and sound: should it integrate into an existing class/function, or sit
   alongside as a new one? Prefer alongside when the existing code is a fast/cheap path
   and the archive reasoning is a slower/more-rigorous one for the same question - state
   which case this is and why, don't assume "more accurate" means "replaces".
5. Ask before touching anything on the "ask before changing" list in CLAUDE.md
   (forecast contracts, new model types, aggregation approaches, src/ restructuring).
6. Implement the code in `src/`, then **cross-check the new logic against what
   `docs_src/theory/` already claims** - metric definitions and scales (metrics.md),
   known structural quirks of specific models (e.g. diagnostics.md's note that only
   SARIMA produces horizon-varying interval width - a flat result from the others is
   expected, not a bug), and baseline conventions (seasonal naive, MASE). A new page
   that contradicts or duplicates an existing claim without cross-referencing it is a
   defect, not just a missing link. Verify behaviour with a small runnable smoke test
   before treating the implementation as done.
7. Add/extend a high-level page in `docs_src/theory/` explaining what the technique
   does and why it's needed, with minimal code (formulas/one short snippet, not the
   implementation) - point the reader at the `src/` code for the real implementation
   rather than duplicating it, and at the related theory pages checked in step 6. Wire
   the new page into `mkdocs.yml` nav and `docs_src/theory/index.md`'s summary list and
   question table.
8. Never cite `working_notes/` scripts as evidence of "the repo's workflow" - they're
   drafts, not the source of truth.

