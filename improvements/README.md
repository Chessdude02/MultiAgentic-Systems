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
python -m improvements.vol_backtest               # ~17 min after the first download (~4 min)
python -m improvements.vol_backtest --target-vol 0.10 --retrain-every 126
```

Data is cached in `improvements/data/` (git-ignored); tables are written to
`improvements/results/summary.md`, plus `vol_targeting.png`. The v1 run
(stocks-only training, five forecasts) is archived in `improvements/results/v1/`.

## Files

| File | Purpose |
|---|---|
| `universe.py` | S&P 500 constituents + "date added", prices since 2000, earnings dates (cached) |
| `vol_features.py` | Feature/target panel, point-in-time universe filter |
| `garch.py` | Dependency-free GARCH(1,1) (Gaussian QMLE, `lfilter` recursion) |
| `vol_backtest.py` | Annual walk-forward with 20-day purge; all evaluation and portfolio tests |

## v1 results (out-of-sample Oct 2004 - Aug 2026, 1.86M stock-days)

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

## v2: attempts to fix the v1 weaknesses

The numbers above are v1 (tables in `results/v1/`). v2 (`results/summary.md`,
same data and walk-forward) tested four fixes, and only two of them worked.
All settings were fixed before the v2 run.

| Fix | Idea | Outcome |
|---|---|---|
| **Train on 13 index/sector ETFs too** (+ `is_fund` flag) | Remove the ~22% over-prediction of index vol | **Worked.** SPY forecast bias +0.200 → **+0.039** (log); stock accuracy unchanged (R² 0.558). |
| **XGBoost-QLIKE** (custom objective) | Penalise under-predicting risk | **Worked. Best risk model.** QLIKE 0.298 vs 0.331 for XGBoost (DM t = 3.3); best QLIKE in every VIX regime; forecasts >30% too low fall from 12.0% to **9.1%**; bands already honest (82.2% / 95.0%). Costs a little RMSE (R² 0.541 vs 0.558). |
| **Calibrated bands** (empirical past-OOS quantiles) | Fix the 95% band being too narrow | **Partly worked.** XGBoost 95% coverage 93.7% → 94.5%; still slightly narrow in calm markets (VIX < 15). |
| **Stacked** regime-aware blend (weights vary with VIX, learned on earlier folds' OOS forecasts) | Get XGBoost in calm markets and the Combo in crises | **Failed.** Worse than XGBoost on MSE (DM t = -2.8). It was trained on calm 2004-07 data and extrapolated badly into 2008-09 (RMSE 0.451 vs 0.406); still weakest with VIX > 30. It wins in many calm years after 2012. |
| **Bias correction** (subtract each ticker's recent realised error) | Remove persistent per-ticker bias | **Failed for stocks.** Adds noise (QLIKE 0.355 vs 0.331). Removes SPY bias (-0.004) but doesn't improve vol targeting. |

**v3: constrained Blend (specified before running).** After the stacked blend
failed, I declared one follow-up design before running it, to limit data
snooping: the same four forecasts with constant weights >= 0 that sum to 1,
no intercept and no VIX term, fit on earlier folds' out-of-sample forecasts.
Every other model reproduced exactly.

| | RMSE | R² | QLIKE | 2008 RMSE | VIX > 30 RMSE |
|---|---|---|---|---|---|
| XGBoost | 0.329 | 0.558 | 0.331 | 0.406 | 0.394 |
| XGBoost-QLIKE | 0.335 | 0.541 | **0.298** | 0.386 | 0.390 |
| Stacked (v2) | 0.333 | 0.546 | 0.331 | 0.451 | 0.422 |
| **Blend (v3)** | **0.328** | **0.561** | 0.319 | 0.397 | 0.378 |

- It fixed the stacking failure: no crisis blow-up (2008: 0.397 vs 0.451).
- It has the best point accuracy of any forecast, but this is a **statistical tie with
  XGBoost** on MSE (DM t = -0.02). It's marginally better on QLIKE (t = 2.0).
- It is still clearly behind XGBoost-QLIKE on QLIKE (t = -4.3), and it doesn't
  help SPY vol targeting (Sharpe 0.84).
- Weights in the last fold: XGBoost 0.61, HAR 0.21, XGBoost-QLIKE 0.12, GARCH 0.06.

**Vol targeting is still best with plain trailing vol.** Removing XGBoost's
index bias lifted its SPY vol-targeting Sharpe from 0.81 to 0.84 (exposure
0.76 → 0.92), but trailing vol (0.88) and the Stacked blend (0.87) still do
better, and XGBoost's worst quarter got worse (27% vs 23% vol). Vol targeting
rewards reacting quickly to vol spikes more than it rewards average
forecast accuracy.

**Updated recommendation**

- Single-stock risk (sizing, 20-day ranges, stop distances): **XGBoost-QLIKE**,
  with calibrated bands. It has the fewest dangerous under-predictions in
  every market regime.
- Point forecasts judged by squared error: the constrained Blend or plain XGBoost
  (statistically tied). The Blend holds up better in crises.
- Index vol targeting: trailing 22-day vol. It's simpler and still the best here.

## Production risk model and AI brief

`risk_model.py` packages the recommended model: XGBoost-QLIKE trained on the
full panel (stocks + ETFs). The band multipliers are calibrated on a 2-year
holdout before the final fit.

```bash
python -m improvements.risk_model --train     # -> models/risk/ (~1 min)
python -m improvements.risk_model AAPL SPY    # 20-day vol + calibrated 80% / 95% price ranges
```

In the latest training, the holdout (Aug 2024 - Aug 2026) had RMSE 0.320 (log vol)
and bias +0.04, with band multipliers of 1.25 (80%) and 1.99 (95%). The Streamlit
app shows this model and the Claude risk brief (`agents/risk_brief_agent.py`).

## Caveats

- **Survivorship bias is reduced, not removed.** Stocks enter the universe on
  their S&P 500 "date added", but companies later removed from the index are
  missing (Yahoo no longer serves most of them). This matters much less for
  volatility than for returns.
- Earnings dates come from Yahoo; after-close announcements are shifted to
  the next session. Only past earnings dates are used as features.
- Settings (horizon, target vol, leverage cap, model hyperparameters) were
  fixed before the first run and not tuned on these results.
