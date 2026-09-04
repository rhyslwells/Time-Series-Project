Question                          → Document                  Section
─────────────────────────────────────────────────────────────────────────
"What's a good MAE?"             → METRIC_REFERENCE.md       Metrics at Glance
"How do I interpret this plot?"   → VISUAL_EXAMPLES.md        Good vs Bad Cases
"Why use this formula?"           → MATH_AND_INTERPRETATION   Part 1-5
"When should I retrain?"          → README_FORECASTING.md     When to Retrain
"How do I tune SARIMA?"           → MODEL_COMPARISON_GUIDE    Tune Best Model
"Is my forecast ready?"           → VISUAL_EXAMPLES.md        Section 6
"Which model is best?"            → README_FORECASTING        Decision Tree


================================================================================
INTERPRETATION GUIDELINES
================================================================================

"Is MAE=0.35 kWh good?"
  → Depends on asset type:
    Residential: 0.35 kWh is excellent (< 0.5 target)
    Commercial: 0.35 kWh is good (< 1.0 target)
    EV Charging: 0.35 kWh is great but unrealistic (expect 1-3 kWh)

"Is coverage=80% correct?"
  → Yes! Target 80% PI should have 80% coverage
  → If coverage < 75%: intervals too narrow (over-confident)
  → If coverage > 85%: intervals too wide (under-confident)
  → For FlexGo: low coverage means flexibility commitments will breach

"Is RMSE=0.52 kWh better than MAE=0.35 kWh?"
  → They measure different things:
    RMSE penalizes large errors more (squaring effect)
    MAE is average error
  → Relationship: RMSE/MAE ratio indicates outlier frequency
    Ratio 1.0–1.2: Consistent, few outliers ✓
    Ratio 1.3–1.5: Some outliers (normal)
    Ratio > 1.5: Frequent large errors ✗

"Which model should I use?"
  1. Rank by RMSE (lower = better)
  2. Check coverage (80% ± 5% = good)
  3. Check coverage uniformity (same across hours/days? ✓)
  4. Deploy best, monitor weekly

"When do I retrain?"
  → Coverage drops below 70% (immediately!)
  → RMSE increases 20% (weekly check)
  → New season (spring/summer/fall/winter)
  → Equipment changes (new HVAC, solar installed)

================================================================================
QUICK SANITY CHECKS (Before Deployment)
================================================================================

✓ RMSE < [residential: 0.7 | commercial: 1.3 | EV: 2.5] kWh
✓ MAE < [residential: 0.5 | commercial: 1.0 | EV: 2.0] kWh
✓ MAPE < [residential: 12% | commercial: 18% | EV: 40%]
✓ PI Coverage 75–85% (ideally 78–82%)
✓ RMSE/MAE < 1.5 (ratio too high = huge outliers)
✓ Residuals bell-shaped (histogram, not skewed)
✓ Mean(residuals) ≈ 0 (no systematic bias)
✓ No trend in residuals over time
✓ Uncertainty width varies by time-of-day (not flat)
✓ Coverage uniform across hours (no systematic under/overcoverage)

If all ✓ → Production-ready
If 2+ ✗ → Needs more tuning
