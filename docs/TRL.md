# TRL assessment – FlightDelay

**Claimed level: TRL 4** (laboratory validation on public real data). Not TRL 5: two months of one year, no weather or ATC data, no operational use.

Data: BTS On-Time Performance, 150 000 sampled non-cancelled flights per month. Sizes: train 71576, calibration 34854, test 43570 (January), July 150000. January test arrival delay quantiles (min): 5 % -28, median -5, 90 % 39, 95 % 71. Previous-leg delay known for 85 % of flights.

90 % intervals; "cov/width" = coverage / mean width in minutes.

| Model | RMSE (min) | MAE (min) | Split-conformal cov / width | CQR cov / width | CQR cov, departures after 17:00 | CQR cov, previous leg > 15 min late | July, January calibration | July, recalibrated on 2 000 flights |
|---|---|---|---|---|---|---|---|---|
| Schedule only | 56.5 | 29.9 | 0.895 / 112 | 0.885 / 115 | 0.868 | 0.901 | 0.874 | 0.911 |
| Schedule + previous-leg delay | 53.7 | 26.0 | 0.885 / 95 | 0.887 / 98 | 0.874 | 0.894 | 0.880 | 0.909 |

Reference: a constant (training mean) has RMSE 54.4 min on the January test days; a constant median has MAE 22.8 min.

## Findings and caveats
- **Pre-departure information predicts arrival delay poorly.** The schedule-only model has a *higher* RMSE than the constant mean (56.5 against 54.4 min), probably because the training days include the 11 January 2023 FAA system outage and are not representative of the test days. Adding the previous-leg delay lowers the RMSE only to 53.7 min (1.4 % better than the constant).
- The mean model's MAE (29.9 and 26.0 min) is above the constant median's (22.8) because the model predicts the mean of a heavily skewed distribution; this is a loss mismatch, not a better or worse model.
- Intervals are wide: 95–115 minutes on average. Coverage is slightly below the 0.90 target on the test days (0.885–0.895) and lowest for late-day departures (0.868–0.879 with CQR).
- Seasonal shift: January-calibrated CQR covers 0.874 (schedule only) and 0.880 of July flights; recalibration on 2 000 July flights gives 0.909 and 0.911 with margins 5–7 minutes wider.
- Conformal split ignores the time ordering; the calibration week and the test week are not exchangeable (coverage drifted between them). Only two months of one year were used.
- The previous-leg feature can only be used when that flight had landed before departure (85 % of flights here).
