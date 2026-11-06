# articles-FlightDelay

Arrival-delay forecasts for US domestic flights from **pre-departure information only**, with conformal prediction intervals, on real
data (BTS On-Time Performance, January and July 2023). TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: BTS "Reporting Carrier On-Time Performance" monthly files (transtats.bts.gov; not committed)
python research/run_study.py <January 2023 csv> <July 2023 csv>
```
