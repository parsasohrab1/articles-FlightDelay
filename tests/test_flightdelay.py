import numpy as np
import pandas as pd

from flightdelay.data import _minutes, load_month
from flightdelay.models import cqr_correction, split_conformal_margin


def test_minutes_conversion():
    assert list(_minutes(pd.Series([0, 59, 100, 2359, 2400]))) == [0, 59, 60, 1439, 0]


def test_cqr_gives_coverage_on_synthetic_heteroscedastic_data():
    rng = np.random.default_rng(0)

    def draw(n):
        x = rng.uniform(0, 1, n)
        y = rng.normal(0, 0.2 + 2 * x)
        return (
            y,
            -1.645 * (0.2 + 2 * x) * 0.6,
            1.645 * (0.2 + 2 * x) * 0.6,
        )  # deliberately too narrow quantiles

    y, lo, hi = draw(5000)
    m = cqr_correction(y, lo, hi, 0.1)
    y2, lo2, hi2 = draw(20000)
    cov = np.mean((y2 >= lo2 - m) & (y2 <= hi2 + m))
    assert abs(cov - 0.9) < 0.02
    assert split_conformal_margin(np.arange(1.0, 100.0), np.zeros(99), 0.1) == 90.0


def test_prev_delay_uses_only_landed_flights(tmp_path):
    rows = []
    # one aircraft, two flights the same day; first lands (sched 120 + delay 10) before the second departs at 300
    for dep, arr, delay in ((100, 120, 10), (300, 400, -5)):
        rows.append(
            {
                "FlightDate": "2023-01-01",
                "Reporting_Airline": "AA",
                "Tail_Number": "N1",
                "Origin": "A",
                "Dest": "B",
                "CRSDepTime": (dep // 60) * 100 + dep % 60,
                "CRSArrTime": (arr // 60) * 100 + arr % 60,
                "ArrDelay": delay,
                "Cancelled": 0,
                "Diverted": 0,
                "Distance": 100,
                "CRSElapsedTime": arr - dep,
                "DayOfWeek": 7,
            }
        )
    # a third flight later the same day (departs at 10:00, after the second has landed)
    rows.append(
        {**rows[1], "CRSDepTime": 600 // 60 * 100 + 0 + 0, "CRSArrTime": 1000, "ArrDelay": 0}
    )
    p = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    d = load_month(str(p))
    assert np.isnan(d.prev_delay.iloc[0]) and d.prev_delay.iloc[1] == 10
