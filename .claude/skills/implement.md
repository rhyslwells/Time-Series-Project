# Skill: implement

Build to the locked scope. Validation is part of implementation, not a separate phase.

---

## Workflow

1. **Pre-check (1 min)**
   - [ ] Scope is locked (state it: "Goal is X, criteria are Y, Z, W")
   - [ ] Location decided (src/ or archive/?)
   - [ ] Relevant contracts understood (check DATA_STACK.md)

2. **Code path (2 min)**
   - **src/** → production code, follows CODING_STANDARDS.md, no debug cruft
   - **archive/working_notes/** → exploratory, can be rough

3. **Implement & validate inline (varies)**
   - Follow acceptance criteria as checklist
   - Validate data contracts as you build (don't leave it)
   - Test on actual data (full 2-week dataset), not mocks
   - For time series: validate against ramp_rates.parquet if relevant

4. **Acceptance criteria check (5 min)**
   - [ ] Criterion 1 passes—how did you verify?
   - [ ] Criterion 2 passes—how did you verify?
   - [ ] Criterion 3 passes—how did you verify?

5. **Done when:**
   - All criteria pass
   - Data contracts verified
   - Code is self-contained (no pending dependencies)

---

## What NOT to Do

- ❌ Update docs (that's `/docs-capture`)
- ❌ Refactor unrelated code
- ❌ Write comprehensive test suite (validation = verification)
- ❌ Optimize prematurely
- ❌ Change contracts (ask first)

---

## Output

```
IMPLEMENTATION COMPLETE
Criteria: [all pass / list which]
Location: [src/ or archive/]
Files: [list changed/created]
Next: /test or /docs-capture
```

---

## Troubleshooting

**Getting complicated?** → Scope was too broad. Split.  
**Scope unclear?** → Go back to `/scope`.  
**Need to refactor?** → That's a separate scope.
