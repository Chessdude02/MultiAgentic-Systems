# Improvements: volatility forecasting on 22 years of S&P 500 data

The first version of the pooled model (`backtest.py` in the project root) showed
that **volatility** is the forecastable quantity, but it was tested on
100 stocks, 3 years and a weak naive baseline. This folder re-tests the idea
properly:

| | Before (`backtest.py`) | Here |
|---|---|---|
| Universe | 100 of today's large caps | ~500 S&P 500 stocks, each included only **after the date it joined the index** |
| History | 6 years (3 out-of-sample) | 2000-2026, **22 years out-of-sample** (2008, 2011, 2020, 2022 included) |
| Baselines | trailing vol only | trailing vol, **HAR** (pooled log-HAR) and **GARCH(1,1)** per stock |
| Features | close-to-close vol, trend, VIX | + range-based (Garman-Klass + overnight) HAR components, downside/overnight share, vol-of-vol, beta/correlation, VIX-vs-realised, liquidity, **earnings cycle** |
| Statistics | point estimates | **Diebold-Mariano** tests (Newey-West), per-year and per-VIX-regime breakdowns, return-band **coverage** |
| Use case | inverse-vol weights | + **SPY volatility targeting** |

## Run

```bash
python -m improvements.vol_backtest               # ~6 min after the first download (~4 min)
python -m improvements.vol_backtest --target-vol 0.10 --retrain-every 126
```

Data is cached in `improvements/data/` (git-ignored); tables are written to
`improvements/results/summary.md`, plus `vol_targeting.png`.

## Files

| File | Purpose |
|---|---|
| `universe.py` | S&P 500 constituents + "date added", prices since 2000, earnings dates (cached) |
| `vol_features.py` | Feature/target panel, point-in-time universe filter |
| `garch.py` | Dependency-free GARCH(1,1) (Gaussian QMLE, `lfilter` recursion) |
| `vol_backtest.py` | Annual walk-forward with 20-day purge; all evaluation and portfolio tests |

## Results (out-of-sample Oct 2004 - Aug 2026, 1.86M stock-days)

Target: realised volatility over the next 20 trading days.

| Forecast | RMSE (log vol) | R² | QLIKE | Median abs % error |
|---|---|---|---|---|
| Trailing 22-day vol | 0.416 | 0.29 | 0.477 | 25.4% |
| HAR | 0.360 | 0.47 | 0.376 | 22.0% |
| GARCH(1,1) | 0.400 | 0.35 | 0.345 | 27.0% |
| **XGBoost** | **0.329** | **0.56** | 0.331 | **19.9%** |
| Combo (avg of HAR, GARCH, XGBoost) | 0.337 | 0.54 | **0.307** | 21.2% |

**What holds up**

- **XGBoost is the most accurate forecaster, and it's not luck.** Diebold-Mariano
  t-stats on squared error: 18.2 vs trailing, 9.4 vs HAR, 13.2 vs GARCH.
  It beats all three baselines in 22 of 23 years (the exception is 2020,
  where HAR is better).
- **The Combo is the most robust.** Best QLIKE (the loss that punishes
  under-predicting risk), significantly better than trailing, HAR and GARCH
  (t = 16.0, 12.7, 5.5), though not significantly better than XGBoost
  (t = 1.7). It has the lowest error in 8 of 23 years, including every
  stress year (2008, 2011, 2015, 2020); XGBoost is lowest in 14, and
  2022 is a tie.
- **XGBoost is weakest in crises:** with VIX > 30 its RMSE (0.394) falls
  behind the Combo (0.367) and HAR (0.376). Use the Combo when markets are
  stressed.
- **The bands are well calibrated:** XGBoost's 80% / 95% return bands contain
  the outcome 79.4% / 93.7% of the time (slightly narrow at 95% because of
  fat tails). GARCH's are wide (84% / 96%) because it over-predicts vol by
  about 13%.
- **Most useful features:** long- and medium-window volatility (`har_q`,
  `cc_66`, `har_m`) dominate, then drawdown and **days since earnings**.

**What does not**

- **SPY volatility targeting works whichever forecast drives it:**
  every variant cuts max drawdown from 54% to ~29%, caps the worst quarter's
  vol at ~21% (vs 48%) and lifts Sharpe from 0.75 to 0.80-0.88. But the
  *simplest* forecast (trailing vol, Sharpe 0.88) does best and XGBoost does
  worst (0.81). The reason is bias: the pooled model is trained on single
  stocks and over-predicts index vol by ~22% (log bias +0.20, vs +0.01 for
  trailing vol), so it sits at 0.76x average exposure and gives up return.
  Across all five forecasts, lower SPY bias goes with higher exposure and
  higher Sharpe.
- **Inverse-vol stock weighting adds nothing over 22 years:** Sharpe 0.88-0.89
  for every forecast vs 0.88 for equal weight. More accurate vol forecasts
  don't turn into better portfolio weights here.

**Verdict:** for **single-stock** risk (sizing a position, setting a stop,
quoting a 20-day price range), use XGBoost, or the Combo when VIX is high.
For **index-level** vol targeting, trailing or HAR vol is as good or better,
and this pooled stock model should not be used.

## Caveats

- **Survivorship bias is reduced, not removed.** Stocks enter the universe on
  their S&P 500 "date added", but companies later removed from the index are
  missing (Yahoo no longer serves most of them). This matters much less for
  volatility than for returns.
- Earnings dates come from Yahoo; after-close announcements are shifted to
  the next session. Only past earnings dates are used as features.
- Settings (horizon, target vol, leverage cap, model hyperparameters) were
  fixed before the first run and not tuned on these results.
