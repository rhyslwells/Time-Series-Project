# Skill: test

Verify implementation against the locked acceptance criteria and data contracts.

---

## Workflow

1. **Restate criteria (1 min)**
   - List all acceptance criteria from `/scope`
   - For each one: "Criterion X means [specific, testable thing]"

2. **Test each criterion (3–10 min)**
   - **Criterion 1:** Run specific test/check. Pass or fail?
   - **Criterion 2:** Run specific test/check. Pass or fail?
   - **Criterion 3:** Run specific test/check. Pass or fail?

3. **Contract validation (5 min)**
   - **Input:** Does data match expected schema? (check DATA_STACK.md)
   - **Output:** Does result match contract? (check ARCHITECTURE.md)
   - **Edge cases:** Empty data? Single asset? Full 15 assets?

4. **Data sanity (5 min)**
   - Spot-check values (are they reasonable? No NaNs where shouldn't be?)
   - Time series alignment (no gaps, correct resolution?)
   - For forecasts: pass ramp rate validation?

---

## Formats

**Unit validation** (exploratory notebook):
```python
# Load output, verify
result = load_result()
assert result.shape[0] > 0, "Output empty"
assert set(result.columns) == {"expected", "cols"}, "Schema mismatch"
# Spot check
print(result.head())
```

**Integration check** (if touching layers):
```python
# Full pipeline: input → this layer → output
input_data = load_metering_data()
output = my_layer.predict(input_data)
assert output validates against ramp_rates.parquet
```

---

## Pass/Fail Criteria

**PASS if:**
- ✓ All acceptance criteria verified
- ✓ Input/output contracts match
- ✓ No silent failures (nulls, empty results, etc.)
- ✓ Data looks reasonable on spot-check

**FAIL if:**
- ❌ Any criterion doesn't pass
- ❌ Schema mismatch
- ❌ Output is malformed (NaNs, wrong types, etc.)
- ❌ Ramp validation fails (if applicable)

---

## Output

```
TEST COMPLETE
Criterion 1: PASS (verified by: [how])
Criterion 2: PASS (verified by: [how])
Criterion 3: PASS (verified by: [how])

Contracts: ✓ Input schema | ✓ Output schema | ✓ Ramp validation
Sanity: ✓ No NaNs | ✓ Full dataset | ✓ [other checks]

Result: PASS / FAIL
Next: /review (if PASS) or /implement (if FAIL, fix)
```

---

## If Tests Fail

1. Identify which criterion failed
2. Look at actual output vs. expected
3. Go back to `/implement`, fix
4. Re-run `/test`

Don't move forward to `/review` until all tests pass.
