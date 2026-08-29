# Skill: docs-capture

**Purpose:** Capture changes from the current session and suggest/apply documentation updates to `docs_src/`.

---

## Workflow

1. **Review session changes** — Identify modified files and the intent behind changes
2. **Map to relevant docs** — Determine which documentation files should be updated
3. **Draft updates** — Show proposed changes to docs_src/ files
4. **Apply or review** — Let user approve before updating, or apply directly

---

## Implementation Steps

### Step 1: Identify Changes (30 seconds)

Run:
```bash
git diff --name-only
git status
```

Categorize changes:
- **Architecture changes** → Update `ARCHITECTURE.md` or `REPOSITORY_STRUCTURE.md`
- **Data pipeline changes** → Update `DATA_STACK.md`
- **New features/layers** → Update `ARCHITECTURE.md` with layer details
- **Workflow/process changes** → Update `WORKFLOW.md`
- **Coding/style changes** → Update `CODING_STANDARDS.md`
- **Model/validation changes** → Update relevant doc in `docs_src/`

### Step 2: Extract Context from Chat (1-2 min)

Review the conversation to extract:
- What was the goal? (feature, fix, refactor, etc.)
- What changed conceptually? (not just "file X changed", but "why and what it means")
- What should users/future-you know about this?
- Are there contracts, constraints, or gotchas to document?

### Step 3: Propose Doc Updates (1-2 min)

For each relevant doc file:
- Show what currently exists
- Propose the addition/change with context
- Include: what changed, why, any new constraints or contracts

Example format:
```
FILE: .claude/ARCHITECTURE.md
CHANGE: Add new forecasting layer description
PROPOSED TEXT:
### Layer: [Name]
- Purpose: [what it does]
- Inputs: [data contracts]
- Outputs: [data contracts]
- Dependencies: [other layers]
```

### Step 4: Apply or Get Approval

- **Option A (Default):** Show proposed changes and ask for approval before applying
- **Option B (If user confirms):** Apply changes directly with summary

---

## When to Use This Skill

✅ After implementing a new feature or layer  
✅ After changing data pipeline or model structure  
✅ After architectural decisions  
✅ End of session when you know docs are stale  
✅ Before committing code (ensure docs match implementation)

---

## What Gets Updated

**Always check:**
- `.claude/ARCHITECTURE.md` — If layers or design changed
- `.claude/DATA_STACK.md` — If data contracts or pipeline changed
- `.claude/DOCUMENTATION.md` — If doc policy/approach changed
- `docs_src/` — Any user-facing or deeper documentation
- `.claude/WORKFLOW.md` — If process changed

**Never edit:**
- `.claude/CLAUDE.md` — This is system instructions, ask before changing
- Git history — Documentation is separate

---

## Constraints

1. Only update docs if changes actually warrant it (no busywork)
2. Keep writing consistent with existing style in each file
3. Use absolute dates in project memory, not relative ("2026-08-29", not "today")
4. Link related docs with cross-references where relevant
5. If proposing a major change to architecture or data contracts, flag it for review before applying

---

## Example Scenarios

### Scenario 1: Added New Data Validation
**Changed:** `src/` code for validating new metric  
**Update:** Add validation constraint to `DATA_STACK.md` contract section  
**Why:** Future readers need to know this validation exists and when it runs

### Scenario 2: Refactored Forecast Layer
**Changed:** Merged two forecast layers into one  
**Update:** Update `ARCHITECTURE.md` layer diagram and contract definitions  
**Why:** Layer structure is a core architectural decision others need to know

### Scenario 3: Added New Feature Engineering Step
**Changed:** New polars transformation in exploration notebook  
**Update:** If consolidating to src/, add to `ARCHITECTURE.md` or feature documentation  
**Why:** If it's becoming production code, future work needs to know what it does

---

## Quick Checklist Before Applying

- [ ] Changes match what was actually implemented
- [ ] All new contracts/constraints are documented
- [ ] Cross-references are included where relevant
- [ ] Writing is consistent with existing docs
- [ ] No sensitive info (keys, paths, etc.) in docs
- [ ] Relative dates converted to absolute (if using memory)
