# Skill: review

Check implementation against project patterns, standards, and design decisions.

**Prerequisites:** Testing must pass (via `/test`). If tests failed, fix implementation first.

---

## Workflow

1. **Code review (5 min)**
   - [ ] Follows CODING_STANDARDS.md (polars for data ops, type hints, brief comments)
   - [ ] Well-named functions/variables (explain logic via names, not comments)
   - [ ] No debug code, print statements, or TODOs left in production code
   - [ ] Location correct (src/ = production-ready, archive/ = exploratory)

2. **Architecture check (3 min)**
   - [ ] Doesn't violate layer separation (review ARCHITECTURE.md)
   - [ ] Data contracts match what's documented
   - [ ] Dependencies are clear (doesn't create circular deps)
   - [ ] If adding a layer, does it fit the forecasting pipeline logically?

3. **Documentation alignment (5 min)**
   - [ ] Code matches what's in docs (or docs are stale—note for `/docs-capture`)
   - [ ] New contracts documented or will be
   - [ ] Architectural changes flagged (if any)

4. **Edge cases (3 min)**
   - [ ] Handles empty input gracefully
   - [ ] Works on all 15 assets (not just test subset)
   - [ ] Time series edge cases covered (start/end of data, gaps)

---

## Quick Checklist

```
CODE REVIEW
──────────
Standards: ✓ polars / ✓ type hints / ✓ no cruft
Architecture: ✓ layer separation / ✓ contracts clear / ✓ no circular deps
Docs: ✓ matches current / ✓ notes stale docs for sync
Edge cases: ✓ empty input / ✓ all assets / ✓ time series bounds

Result: APPROVE / REQUEST CHANGES
──────────
```

---

## If Changes Needed

1. Note what doesn't match patterns
2. Go back to `/implement`, fix
3. Re-run `/test` to confirm tests still pass
4. Return to `/review`

---

## If All Checks Pass

You're done. Next steps:

- **Run `/docs-capture`** to sync documentation with changes
- **Commit** with message describing what changed and why
- **Note any follow-ups** (refactoring, optimization, future work) as separate scopes

---

## Red Flags

- ❌ Code style inconsistent with CODING_STANDARDS.md
- ❌ Contracts don't match ARCHITECTURE.md
- ❌ Undocumented breaking changes
- ❌ Layer structure unclear (too much in one place)
- ❌ Time series validation skipped

Any red flag → Request changes, fix, re-review.
