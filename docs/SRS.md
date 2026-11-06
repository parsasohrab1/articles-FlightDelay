# Software Requirements Specification – FlightDelay

- SR-1 `data.load_month`: removes cancelled and diverted flights; builds minute-level schedule times; `prev_delay` = arrival delay of the same tail number's previous flight if it had landed before this flight's scheduled departure, else missing (a unit test checks this). No departure delay or other post-departure field is used.
- SR-2 `models.design`: airline, origin and destination codes (250 most frequent values each, the rest share one code), day of week, scheduled departure hour, distance, scheduled elapsed time, optionally `prev_delay`.
- SR-3 `models.make_model`: histogram gradient boosting (mean, 5 % and 95 % quantile losses); `cqr_correction`: conformalized-quantile-regression margin; `split_conformal_margin`: margin around the mean prediction.
- SR-4 `research/run_study.py`: 150 000 flights sampled per month; January days 1–15 train, 16–22 calibration, 23–31 test; July as a seasonal shift; alpha 0.1.
- SR-5 Three unit tests.
