# Forecast products

What each [forecast product](../theory/forecast_products.md) tells you, with the model held fixed. A 7-day `SeasonalWindowAverage` forecast for one EV-charging asset (ASSET_001) supplies the point forecast and 80% interval, every product is computed from those alone.

[Open the full-screen version](../forecast_products_export.html){ target="_blank" rel="noopener" }

<iframe
  src="../forecast_products_export.html"
  title="Forecast products notebook"
  loading="lazy"
  style="width: 100%; height: 85vh; border: none;">
</iframe>

Run locally:
```bash
uv run marimo edit src/notebooks/forecast_products.py
```

The reusable pieces (`implied_std`, `exceedance_probability`) are in `src/ts_products.py`, everything else in the notebook is plain numpy on `ForecastOutput`.
