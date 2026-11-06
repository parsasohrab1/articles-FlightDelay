from __future__ import annotations

import numpy as np
import pandas as pd

COLS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "Distance",
    "CRSElapsedTime",
    "DayOfWeek",
]


def _minutes(hhmm: pd.Series) -> pd.Series:
    h = (hhmm // 100) % 24
    return h * 60 + hhmm % 100


def load_month(path: str, sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """BTS On-Time Performance month -> flights with schedule features and a pre-departure knock-on feature.

    Knock-on feature: the arrival delay of the same aircraft's previous flight (same day), used only if that
    flight had already landed before this flight's scheduled departure (else NaN).
    No departure-delay or other post-departure field is used.
    """
    d = pd.read_csv(path, usecols=COLS, low_memory=False)
    d = (
        d[(d.Cancelled == 0) & (d.Diverted == 0)]
        .dropna(subset=["ArrDelay", "CRSDepTime", "CRSArrTime", "CRSElapsedTime"])
        .copy()
    )
    d["date"] = pd.to_datetime(d.FlightDate)
    d["dep_min"] = _minutes(d.CRSDepTime.astype(int))
    d["arr_sched_min"] = d.dep_min + d.CRSElapsedTime
    d["t_dep"] = (d.date - d.date.min()).dt.days * 1440 + d.dep_min
    d["t_arr_actual"] = (d.date - d.date.min()).dt.days * 1440 + d.arr_sched_min + d.ArrDelay
    d = d.sort_values(["Tail_Number", "t_dep"])
    same = d.Tail_Number.eq(d.Tail_Number.shift(1)) & d.Tail_Number.notna()
    prev_delay = d.ArrDelay.shift(1)
    prev_landed = d.t_arr_actual.shift(1)
    ok = same & (prev_landed <= d.t_dep)
    d["prev_delay"] = np.where(ok, prev_delay, np.nan)
    d["hour"] = d.dep_min // 60
    d["day"] = d.date.dt.day
    d = d.sort_values("t_dep").reset_index(drop=True)
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=seed).sort_values("t_dep").reset_index(drop=True)
    return d
