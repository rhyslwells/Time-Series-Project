# Skill: scope

**Purpose:** Define a feature/fix scope precisely. Gate it so implementation stays focused.

---

## Workflow (Max 5 minutes)

### 1. State the Goal (1 min max)
**Lock in:** What is the ONE thing being built/fixed?

Write 1-2 sentences. If you can't fit it in 2 sentences, scope is too broad—split it.

Example ✓:
- "Add daily aggregation layer to forecast pipeline that sums 30-min predictions into daily totals"
- "Fix: Ramp rate validation rejects valid transitions under 50 kW/h"

Example ✗ (too broad):
- "Improve the forecasting system" — STOP, split this
- "Refactor and add features" — STOP, split this

### 2. Define Input/Output Contracts (1.5 min max)
**What data comes in? What leaves?**

For forecasting work, always specify:
- **Input:** Asset ID range, timestamp resolution, required fields
- **Output:** Same contract as defined in `DATA_STACK.md`, or new contract (document it)
- **Constraints:** Time limits, layer dependencies, validation rules

Example:
```
Input: metering_data.parquet (30-min intervals, asset_id, timestamp, value)
Output: daily_predictions (24 rows per asset, date, sum of 30-min forecasts, asset_id)
Constraint: Must validate output sums match ramp_rates bounds
```

### 3. Acceptance Criteria (1.5 min max)
**How do we know it's done?**

Write 3–5 specific, testable criteria. If they're vague, rewrite them.

✓ Testable:
- "Daily sums pass ramp rate validation on all 15 assets"
- "Notebook runs without errors on 2 weeks of data"
- "Output schema matches contract defined in ARCHITECTURE.md"

✗ Vague:
- "Works correctly" 
- "Is better than before"
- "Handles edge cases" (which ones?)

### 4. Constraints & Gotchas (1 min max)
**What could go wrong? What's off-limits?**

- **Size limits:** (e.g., "must not exceed 2GB in memory")
- **Dependency locks:** (e.g., "cannot refactor polars pipeline during this work")
- **Data bounds:** (e.g., "must work on all 15 assets, not just EV")
- **Location:** Where does code go? (src/ = production, archive/ = exploratory)

### 5. Estimated Effort (30 sec)
**Rough size:** tiny (< 30 min), small (< 2 hr), medium (< 4 hr), large (> 4 hr, consider splitting)

If large, go back to step 1 and split.

---

## Output Format (Lock & Gate)

Print this to confirm scope is locked:

```
SCOPE LOCKED
────────────
Goal: [1-2 sentence description]

Input/Output:
  In:  [contract name & key fields]
  Out: [contract name & key fields]
  
Acceptance Criteria:
  1. [specific, testable]
  2. [specific, testable]
  3. [specific, testable]

Constraints:
  - [location: src/ or archive/]
  - [size/dependency/data bounds]
  
Effort: [tiny/small/medium/large]

────────────
Ready to implement? (yes/no/split this)
```

---

## Guardrails Against Scope Creep

### Red Flags—STOP and Clarify:

❌ "What about also...?" — That's a new scope. Document separately.

❌ "While we're here, let's refactor..." — No. Refactoring is a separate scope.

❌ "Let me handle edge case X, Y, Z..." — Pick the critical one. Others become separate scopes.

❌ "Add docs/tests/comments while building..." — No. Build first, docs are a separate pass (handled by `/docs-capture`).

❌ "I'll also update ARCHITECTURE.md" — Nope. That happens in `/docs-capture` after.

### When to Reject Scope:

1. **Can't fit in 2 sentences** → Too broad, split
2. **More than 5 acceptance criteria** → Too broad, split
3. **Touches 3+ layers** → Probably too broad, split
4. **No clear output contract** → Vague, clarify first
5. **Effort is "large"** → Consider splitting into small chunks

---

## What This Unlocks

Once scope is locked:
- **Implement** follows the criteria—no guessing
- **Test** knows exactly what to verify
- **Review** checks against this contract
- You stay focused; no wasted exploration

---

## Examples

### Example 1: Feature
```
Goal: Add rolling window feature (7-day moving average) to daily aggregation layer

Input/Output:
  In:  daily_predictions (asset_id, date, sum_forecast)
  Out: daily_predictions_with_rolling (asset_id, date, sum_forecast, ma7_forecast)

Acceptance Criteria:
  1. Rolling avg computed correctly for 7+ day windows
  2. Handles edge case: first 6 days (no full window)
  3. Validation passes ramp rate checks on all 15 assets
  
Constraints:
  - Code in src/layers/
  - Cannot refactor daily aggregation
  
Effort: small
```

### Example 2: Bug Fix
```
Goal: Fix ramp rate validation rejecting valid 40 kW/h transitions under 50 kW/h threshold

Input/Output:
  In:  ramp_rates.parquet (asset_id, max_up_ramp, max_down_ramp)
  Out: Same structure, validation logic corrected

Acceptance Criteria:
  1. Transition of 40 kW/h passes validation when max is 50
  2. Transitions over 50 kW/h still rejected
  3. All 15 assets validate on full 2-week dataset

Constraints:
  - Validation only, no changes to data
  - Must maintain backward compat with existing ramp data

Effort: tiny
```

---

## Next Steps

- **Accepted scope?** → `/implement`
- **Need to split?** → Define new scopes, lock each one
- **Unclear?** → Clarify constraints/criteria here, don't implement yet
