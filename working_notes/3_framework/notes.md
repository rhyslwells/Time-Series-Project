Explore and understand the content of

C:\Users\RhysL\Desktop\Time Series Project\src\notebooks\ts_model_explorer.py

```bash
uv run marimo edit src/notebooks/ts_model_explorer.py
```

# Notes


### src\notebooks\ts_model_explorer.py

- Loading data functions in the src for each of the datasets
- need the dates instead in the plots not the index, so we need to convert the timestamp to datetime and set it as the index for plotting.
- we need to build a seperate dataset for LightGBMModel with lagging features.
