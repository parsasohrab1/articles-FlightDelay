from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CAT = ["Reporting_Airline", "Origin", "Dest"]
NUM_SCHEDULE = ["DayOfWeek", "hour", "Distance", "CRSElapsedTime"]


def design(d: pd.DataFrame, cat_maps: dict, with_prev: bool) -> pd.DataFrame:
    X = pd.DataFrame({c: d[c].map(cat_maps[c]).fillna(len(cat_maps[c])).astype(int) for c in CAT})
    for c in NUM_SCHEDULE:
        X[c] = d[c].to_numpy()
    if with_prev:
        X["prev_delay"] = d["prev_delay"].to_numpy()
    return X


def fit_cat_maps(d: pd.DataFrame) -> dict:
    """Category codes; at most the 250 most frequent values per column, everything else shares one 'other' code."""
    return {c: {v: i for i, v in enumerate(d[c].value_counts().index[:250])} for c in CAT}


def make_model(q: float | None = None, **kw) -> HistGradientBoostingRegressor:
    base = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "categorical_features": [0, 1, 2],
        "random_state": 0,
    }
    base.update(kw)
    if q is None:
        return HistGradientBoostingRegressor(**base)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **base)


def cqr_correction(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Conformalized quantile regression: score = max(lo - y, y - hi); returns the symmetric margin."""
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])


def split_conformal_margin(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    s = np.abs(y - pred)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])
