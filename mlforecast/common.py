"""Shared helpers for the half-hourly forecasting demo notebooks, built on mlforecast.

Everything model-related goes through ONE `MLForecast` object (see `make_forecaster`):
  * features  : `lags`, `lag_transforms`, `date_features`   (mlforecast builds them)
  * models    : a dict of sklearn-style estimators, one per (tier, quantile)
  * backtest  : `MLForecast.cross_validation`
  * tomorrow  : `MLForecast.fit` + `MLForecast.predict`

Conventions
-----------
* Series is half-hourly, DatetimeIndex, values in kW (can be negative = export).
* "Day-ahead" means: forecast issued at 00:00 for the 48 periods of that day, using
  only data from the *previous* days. Every lag is >= 48, so one non-recursive
  `predict(h=48)` is valid and the backtest is not cheating.
* Synthetic data only (see `make_synthetic`).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMax, RollingMean, RollingMin
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
MODELS = ["Seasonal naive", "Slot profile", "Quantile GBM"]  # the tiers compared in the backtest notebooks
BASELINES = ["Seasonal naive", "Slot profile"]
ML_MODELS = ["Quantile GBM", "HistGB quantile", "Linear quantile"]  # real ML models you can pick between
TIER_COLORS = {
    "Seasonal naive": "#7f7f7f", "Slot profile": "#1f77b4",
    "Quantile GBM": "#ff7f0e", "HistGB quantile": "#2ca02c", "Linear quantile": "#9467bd",
}
PERIODS_PER_DAY = 48
UID = "asset"

# mlforecast names the feature columns it builds; notes for the notebooks.
LAG1D = "lag48"
FEATURE_NOTES = {
    "lag48": "same slot yesterday",
    "rolling_mean_lag48_window_size3": "mean of yesterday's slot and the 2 slots before it",
    "rolling_mean_lag48_window_size48": "mean of the 24h ending yesterday's slot",
    "rolling_max_lag48_window_size48": "max of the 24h ending yesterday's slot",
    "rolling_min_lag48_window_size48": "min of the 24h ending yesterday's slot",
    "slot": "half-hour of day (0-47)",
    "weekend": "1 if Sat/Sun",
}


# --------------------------------------------------------------------------- data
def make_synthetic(days: int = 14, noise: float = 4.0, seed: int = 0) -> pd.Series:
    """Fake net-load for one asset: evening demand peak, midday solar export, weekday effect."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-02", periods=days * PERIODS_PER_DAY, freq="30min")  # a Monday
    h = idx.hour + idx.minute / 60
    demand = 30 + 20 * np.exp(-(((h - 18) / 2.5) ** 2)) + 8 * np.exp(-(((h - 8) / 1.5) ** 2))
    demand = demand * np.where(idx.dayofweek >= 5, 0.85, 1.0)
    cloud = rng.uniform(0.3, 1.0, size=days)  # one cloudiness factor per day
    solar = 35 * np.clip(np.sin(np.pi * (h - 6) / 12), 0, None) ** 2 * cloud[(idx.normalize() - idx[0].normalize()).days]
    eps = np.zeros(len(idx))
    for i in range(1, len(idx)):  # AR(1) noise so errors are autocorrelated, like real meters
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, noise)
    return pd.Series(demand - solar + eps, index=idx, name="kw")


def get_series(noise: float = 4.0, seed: int = 0) -> tuple[pd.Series, str]:
    return make_synthetic(noise=noise, seed=seed), "synthetic demo data"


def to_long(s: pd.Series) -> pd.DataFrame:
    """mlforecast's input format: one row per (unique_id, ds) with the target in `y`."""
    return pd.DataFrame({"unique_id": UID, "ds": s.index, "y": s.to_numpy()})


# ------------------------------------------------------------------------- models
# mlforecast accepts ANY estimator with fit/predict, so the two baselines are tiny
# sklearn-style classes. They see the same engineered feature table as the GBM.
class SeasonalNaiveQuantile(BaseEstimator, RegressorMixin):
    """Yesterday's value (lag48) + the q-quantile of its historical error."""

    def __init__(self, q: float = 0.5):
        self.q = q

    def fit(self, X, y):
        self.offset_ = float(np.quantile(np.asarray(y) - X[LAG1D].to_numpy(), self.q))
        return self

    def predict(self, X):
        return X[LAG1D].to_numpy() + self.offset_


class SlotProfileQuantile(BaseEstimator, RegressorMixin):
    """Empirical q-quantile of each half-hour slot in the training window."""

    def __init__(self, q: float = 0.5):
        self.q = q

    def fit(self, X, y):
        self.table_ = pd.Series(np.asarray(y)).groupby(X["slot"].to_numpy()).quantile(self.q)
        self.fallback_ = float(np.quantile(y, self.q))
        return self

    def predict(self, X):
        return X["slot"].map(self.table_).fillna(self.fallback_).to_numpy()


def _gbm(q: float) -> lgb.LGBMRegressor:
    # Small + shallow on purpose: 2 weeks of data is ~670 rows.
    return lgb.LGBMRegressor(
        objective="quantile", alpha=q, max_depth=3, n_estimators=120, learning_rate=0.05,
        min_child_samples=10, random_state=0, verbose=-1,
    )


def _hist_gbm(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_depth=3, max_iter=120, learning_rate=0.05,
        min_samples_leaf=10, random_state=0,
    )


def _linear_quantile(q: float) -> QuantileRegressor:
    # Linear in the features: cannot bend around the evening peak. A useful contrast with the trees.
    return QuantileRegressor(quantile=q, alpha=0.0, solver="highs")


_TIERS = {
    "Seasonal naive": ("naive", SeasonalNaiveQuantile),
    "Slot profile": ("profile", SlotProfileQuantile),
    "Quantile GBM": ("gbm", _gbm),  # LightGBM
    "HistGB quantile": ("histgb", _hist_gbm),  # scikit-learn's histogram gradient boosting
    "Linear quantile": ("linq", _linear_quantile),  # scikit-learn linear quantile regression
}


def model_name(tier: str, q: float) -> str:
    return f"{_TIERS[tier][0]}_q{q:.2f}"


def slot(dates):  # date_features callable: half-hour of day, 0..47
    return dates.hour * 2 + dates.minute // 30


def weekend(dates):
    return (dates.dayofweek >= 5).astype(int)


# The feature recipe, shared by every MLForecast in this folder (also reused by 06).
FEATURE_KWARGS = dict(
    freq="30min",
    lags=[48],  # same slot yesterday
    lag_transforms={
        48: [RollingMean(window_size=3), RollingMean(window_size=48),
             RollingMax(window_size=48), RollingMin(window_size=48)],
    },
    date_features=[slot, weekend],
)


def make_forecaster(quantiles=QUANTILES, tiers=MODELS) -> MLForecast:
    """One MLForecast holding a model per (tier, quantile). Fresh object each call."""
    models = {model_name(t, q): _TIERS[t][1](q) for t in tiers for q in quantiles}
    return MLForecast(models=models, **FEATURE_KWARGS)


def _collect(frame: pd.DataFrame, quantiles, tiers=MODELS) -> dict[str, np.ndarray]:
    """{tier: (n, n_quantiles)} from an mlforecast output frame; sorted to stop quantile crossing."""
    return {t: np.sort(frame[[model_name(t, q) for q in quantiles]].to_numpy(), axis=1) for t in tiers}


# ----------------------------------------------------------------- forecast / backtest
def forecast_tomorrow(s: pd.Series, tier: str = "Quantile GBM", quantiles=QUANTILES) -> pd.DataFrame:
    """Fit on all of `s`, predict the next 48 slots. Index = timestamps, columns q0.05..q0.95."""
    fcst = make_forecaster(quantiles, tiers=[tier])
    fcst.fit(to_long(s))
    out = fcst.predict(h=PERIODS_PER_DAY)
    return pd.DataFrame(_collect(out, quantiles, [tier])[tier], columns=qcols(quantiles), index=pd.DatetimeIndex(out["ds"]))


def feature_table(s: pd.Series) -> pd.DataFrame:
    """What the models actually train on: `MLForecast.preprocess` output (unique_id, ds, y, *features)."""
    return make_forecaster([0.5]).preprocess(to_long(s))


def backtest(s: pd.Series, min_train_days: int = 8, quantiles=QUANTILES, tiers=MODELS) -> pd.DataFrame:
    """Rolling-origin, day-ahead via `MLForecast.cross_validation`: each test day is predicted by
    models refit on earlier days only. (The first ~2 days of any training set are lost to the lags.)
    Returns a long frame: one row per (model, timestamp) with y and a column per quantile."""
    n_days = len(s) // PERIODS_PER_DAY
    cv = make_forecaster(quantiles, tiers).cross_validation(
        to_long(s), n_windows=n_days - min_train_days, h=PERIODS_PER_DAY,
        step_size=PERIODS_PER_DAY, refit=True,
    )
    fold = cv["cutoff"].rank(method="dense").astype(int).to_numpy() - 1
    idx = pd.DatetimeIndex(cv["ds"])
    rows = []
    for name, pred in _collect(cv, quantiles, tiers).items():
        out = pd.DataFrame(pred, columns=qcols(quantiles), index=idx)
        out.insert(0, "y", cv["y"].to_numpy())
        out.insert(1, "hod", idx.hour * 2 + idx.minute // 30)
        out.insert(2, "fold", fold)
        out.insert(3, "model", name)
        rows.append(out)
    return pd.concat(rows)


# ----------------------------------------------------------------------- metrics
def qcols(quantiles=QUANTILES) -> list[str]:
    return [f"q{q:.2f}" for q in quantiles]


def pinball(y: np.ndarray, pred: np.ndarray, quantiles=QUANTILES) -> float:
    """Mean pinball (quantile) loss across all quantiles. Lower is better."""
    qs = np.asarray(quantiles)[None, :]
    diff = y[:, None] - pred
    return float(np.mean(np.maximum(qs * diff, (qs - 1) * diff)))


def metrics_table(bt: pd.DataFrame, quantiles=QUANTILES) -> pd.DataFrame:
    rows = []
    for name, g in bt.groupby("model"):
        y, pred = g["y"].to_numpy(), g[qcols(quantiles)].to_numpy()
        med = g["q0.50"].to_numpy()
        rows.append({
            "model": name,
            "MAE (kW)": np.mean(np.abs(y - med)),
            "RMSE (kW)": np.sqrt(np.mean((y - med) ** 2)),
            "pinball": pinball(y, pred, quantiles),
            "80% cover": np.mean((y >= g["q0.10"]) & (y <= g["q0.90"])),
            "90% cover": np.mean((y >= g["q0.05"]) & (y <= g["q0.95"])),
        })
    t = pd.DataFrame(rows).set_index("model").reindex(list(bt["model"].unique()))
    t["skill vs naive"] = 1 - t["pinball"] / t.loc["Seasonal naive", "pinball"]
    return t


# ------------------------------------------------------------------ likelihoods
def prob_exceed(levels, values, x: float) -> tuple[float, str]:
    """P(y > x) from a set of predicted quantiles (linear interpolation of the CDF).
    Beyond the outermost quantiles we can't resolve the tail, so we say so."""
    levels, values = np.asarray(levels), np.sort(np.asarray(values))
    values = values + np.arange(len(values)) * 1e-9  # break ties so interp is well-defined
    if x <= values[0]:
        return 1 - levels[0], f">{(1 - levels[0]) * 100:.0f}%"
    if x >= values[-1]:
        return 1 - levels[-1], f"<{(1 - levels[-1]) * 100:.0f}%"
    p = 1 - float(np.interp(x, values, levels))
    return p, f"{p * 100:.0f}%"


def prob_exceed_curve(pred: np.ndarray, x: float, levels=QUANTILES) -> np.ndarray:
    """P(y > x) for every row of a (n, n_quantiles) prediction array (tails clipped to outer level)."""
    return np.array([prob_exceed(levels, r, x)[0] for r in pred])


def slot_label(hod: int) -> str:
    return f"{hod // 2:02d}:{'30' if hod % 2 else '00'}"


def pit_values(bt_model: pd.DataFrame, quantiles=QUANTILES) -> np.ndarray:
    """Probability-integral-transform: where did the actual fall in the predicted CDF (0..1)?
    A well-calibrated model gives a flat histogram."""
    qs = np.asarray(quantiles)
    full_l = np.r_[0.0, qs, 1.0]
    out = []
    for _, r in bt_model.iterrows():
        v = np.sort(r[qcols(quantiles)].to_numpy(dtype=float))
        lo, hi = v[0] - (v[1] - v[0]), v[-1] + (v[-1] - v[-2])  # crude tail extension
        out.append(float(np.interp(r["y"], np.r_[lo, v, hi] + np.arange(len(v) + 2) * 1e-9, full_l)))
    return np.array(out)


def prob_event(levels, values, x: float, direction: str = "above") -> tuple[float, str]:
    """P(y > x) or P(y < x) with a human-readable label (handles unresolved tails)."""
    p, txt = prob_exceed(levels, values, x)
    if direction == "above":
        return p, txt
    q = 1 - p
    if txt.startswith(">"):
        return q, f"<{q * 100:.0f}%"
    if txt.startswith("<"):
        return q, f">{q * 100:.0f}%"
    return q, f"{q * 100:.0f}%"
