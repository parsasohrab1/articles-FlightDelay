"""BTS On-Time Performance, US domestic flights (real): arrival-delay intervals from pre-departure information.

usage: python research/run_study.py <January 2023 csv> <July 2023 csv>
Features: airline, origin, destination, day of week, scheduled departure hour, distance, scheduled elapsed time and (model B)
the arrival delay of the aircraft's previous flight when it had landed before this departure. No departure delay is used.
Chronological split within January: train days 1-15, calibration days 16-22, test days 23-31 (150 000 flights sampled per month).
July 2023 tests a seasonal shift: coverage of the January-calibrated intervals, and after recalibration on 2 000 July flights.
Intervals: conformalized quantile regression (gradient-boosted 5 % / 95 % quantiles + conformal margin), alpha = 0.1.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from flightdelay.data import load_month
from flightdelay.models import (
    cqr_correction,
    design,
    fit_cat_maps,
    make_model,
    split_conformal_margin,
)

warnings.filterwarnings("ignore")
ALPHA = 0.1


def cover(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def run(jan: str, jul: str):
    dj = load_month(jan, sample=150000)
    dl = load_month(jul, sample=150000, seed=1)
    tr, ca, te = dj[dj.day <= 15], dj[(dj.day >= 16) & (dj.day <= 22)], dj[dj.day >= 23]
    maps = fit_cat_maps(tr)
    out = {
        "note": "real BTS On-Time data; pre-departure features only; alpha=0.1; 150k flights sampled per month",
        "n": {"train": len(tr), "cal": len(ca), "test": len(te), "july": len(dl)},
        "arr_delay_quantiles_jan_test": {
            str(q): float(np.quantile(te.ArrDelay, q)) for q in (0.05, 0.5, 0.9, 0.95)
        },
        "share_prev_delay_known": float(dj.prev_delay.notna().mean()),
        "models": {},
    }
    for name, wp in (("schedule_only", False), ("with_previous_leg", True)):
        Xtr, Xca, Xte, Xjl = (design(d, maps, wp) for d in (tr, ca, te, dl))
        mean = make_model().fit(Xtr, tr.ArrDelay)
        lo_m, hi_m = make_model(0.05).fit(Xtr, tr.ArrDelay), make_model(0.95).fit(Xtr, tr.ArrDelay)
        res = {
            "mae_test": float(np.mean(np.abs(te.ArrDelay - mean.predict(Xte)))),
            "mae_test_median_baseline": float(
                np.mean(np.abs(te.ArrDelay - np.median(tr.ArrDelay)))
            ),
            "rmse_test": float(np.sqrt(np.mean((te.ArrDelay - mean.predict(Xte)) ** 2))),
            "rmse_test_mean_baseline": float(
                np.sqrt(np.mean((te.ArrDelay - tr.ArrDelay.mean()) ** 2))
            ),
        }
        marg = split_conformal_margin(ca.ArrDelay.to_numpy(), mean.predict(Xca), ALPHA)
        m = cqr_correction(ca.ArrDelay.to_numpy(), lo_m.predict(Xca), hi_m.predict(Xca), ALPHA)
        for lab, (lo, hi) in {
            "split_conformal": (mean.predict(Xte) - marg, mean.predict(Xte) + marg),
            "cqr": (lo_m.predict(Xte) - m, hi_m.predict(Xte) + m),
        }.items():
            y = te.ArrDelay.to_numpy()
            res[lab] = {"coverage": cover(y, lo, hi), "mean_width_min": float(np.mean(hi - lo))}
            # coverage by prior-delay status and by hour block
            late = (te.prev_delay.fillna(0) > 15).to_numpy()
            res[lab]["coverage_prev_delay_gt15"] = (
                cover(y[late], lo[late], hi[late]) if late.any() else None
            )
            res[lab]["coverage_prev_delay_le15_or_unknown"] = cover(y[~late], lo[~late], hi[~late])
            eve = (te.hour >= 17).to_numpy()
            res[lab]["coverage_departures_after_17h"] = cover(y[eve], lo[eve], hi[eve])
            res[lab]["coverage_departures_before_9h"] = cover(
                y[te.hour.to_numpy() < 9], lo[te.hour.to_numpy() < 9], hi[te.hour.to_numpy() < 9]
            )
        # July shift with January-calibrated interval, and after recalibration on 2000 July flights
        yj = dl.ArrDelay.to_numpy()
        lo_j, hi_j = lo_m.predict(Xjl) - m, hi_m.predict(Xjl) + m
        res["july_cqr_january_calibration"] = {
            "coverage": cover(yj, lo_j, hi_j),
            "mean_width_min": float(np.mean(hi_j - lo_j)),
        }
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(dl))
        c, t = idx[:2000], idx[2000:]
        m_j = cqr_correction(yj[c], lo_m.predict(Xjl.iloc[c]), hi_m.predict(Xjl.iloc[c]), ALPHA)
        lo2, hi2 = lo_m.predict(Xjl.iloc[t]) - m_j, hi_m.predict(Xjl.iloc[t]) + m_j
        res["july_cqr_recalibrated_2000"] = {
            "coverage": cover(yj[t], lo2, hi2),
            "mean_width_min": float(np.mean(hi2 - lo2)),
            "margin_min": m_j,
            "january_margin_min": m,
        }
        out["models"][name] = res
        print(
            name,
            json.dumps(
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in res.items()
                    if not isinstance(v, dict)
                }
            ),
            {
                k: {a: (None if b is None else round(b, 3)) for a, b in v.items()}
                for k, v in res.items()
                if isinstance(v, dict)
            },
        )
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
