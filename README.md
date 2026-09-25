# MultiAgentic Systems — Neuro-Symbolic Stock Predictor

A multi-agent pipeline that forecasts the next-period closing price of a stock.
Three learned models (BiLSTM, Transformer, XGBoost) predict the next-period
**return** from a window of technical-indicator features; their ensemble is then
adjusted by a rule-based (RSI/MACD) symbolic agent and served through a
Streamlit app.

## How it works

```
yfinance ──► DataAgent ──► FeatureEngineerAgent ──► utils.pipeline ──► PredictionAgent ──► NeuroSymbolicAgent ──► prediction
             (OHLCV)       (RSI, SMA, EMA, MACD,    (scale-free         (BiLSTM +           (ensemble mean,
                            ROC, volatility)         features, 10-bar    Transformer +       + 30% symbolic
                                                     windows, return     XGBoost)            RSI/MACD override)
                                                     target)
```

Key design choice: models are trained on **stationary features** (prices
expressed relative to the current close, log volume changes, scaled
indicators) and predict the **next-period return**, which is converted back to
a price as `last_close * (1 + return)`. An earlier version fed raw prices and
volumes with an absolute price target, which caused the saved models to
collapse to meaningless predictions; `diagnose.py` reproduces that comparison.

## Project layout

| Path | Purpose |
| --- | --- |
| `agents/data_agent.py` | Downloads OHLCV data via yfinance |
| `agents/feature_engineer_agent.py` | Adds technical indicators |
| `agents/prediction_agent.py` | Trains, saves, loads and runs BiLSTM / Transformer / XGBoost |
| `agents/symbolic_override_agent.py` | RSI/MACD rule-based price adjustment |
| `agents/neuro_symbolic_agent.py` | Fuses the model ensemble with the symbolic rules and explains the result |
| `agents/ensemble_agent.py`, `regime_switching_agent.py`, `explainability_agent.py`, `tuning_agent.py` | Auxiliary agents (averaging, K-means regime detection, SHAP explanations, grid-search tuning) |
| `agents/forecast_agent.py` | Pooled 100-stock model: P(beat SPY over 5 days) + 20-day volatility forecast |
| `utils/pipeline.py` | Shared feature/windowing/scaling code used by training, evaluation and the app |
| `utils/panel.py` | Universe download and (Date, Ticker) feature/target panel for the pooled model |
| `backtest.py` | Walk-forward backtest of the pooled model with trading costs; writes `reports/` |
| `train.py` | Trains all models and saves them to `models/` |
| `evaluate.py` | Evaluates saved (or freshly retrained) models on a held-out split |
| `diagnose.py` | Experiment showing why scaling + return targets are needed |
| `app.py` | Streamlit UI |
| `cli.py` | Command-line entry point |
| `simulator.py` | Simple buy/sell trading simulator |
| `models/` | Trained models (`.keras`, `.pkl`), feature scaler and `meta.json` |

## Setup

Python 3.11 is recommended (TensorFlow compatibility).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# optional: install as a package with the `multiagent-cli` command
pip install -e .
```

## Usage

**Train** (defaults: AAPL, 2 years, daily bars):

```bash
python train.py
# or
python cli.py --train --symbol MSFT --period 2y --interval 1d
```

Training prints RMSE, MAE, direction accuracy and a ratio against the naive
"tomorrow = today" baseline (< 1 means the model beats it), then writes the
models, scaler and `meta.json` to `models/`.

**Evaluate** without overwriting saved models:

```bash
python evaluate.py                      # uses train.py defaults
python evaluate.py --symbol MSFT --period 1y
python evaluate.py --cache              # cache data in data/<symbol>_<period>_<interval>.csv
python evaluate.py --retrain --runs 3   # retrain in memory, average over 3 seeds
```

**Run the app:**

```bash
streamlit run app.py
```

Enter a ticker, period and interval and click **Predict** to see the engineered
features, the predicted next close (with change vs. last close), each model's
predicted return, the symbolic-rule explanation and a chart. The app warns when
the chosen interval differs from the one the models were trained on. If a
pooled model exists, the app also shows a 5-day "beats SPY" probability and a
20-day volatility forecast for the ticker.

## Pooled multi-stock model and walk-forward backtest

```bash
python backtest.py --save              # ~100 large caps, 6y history, saves models/pooled/
python backtest.py --quantile 0.3 --cost-bps 5
```

One gradient-boosted model is trained on ~100 large caps at once (≈130k rows)
using scale-free, market-relative features (momentum, volatility, beta, trend,
volume, SPY/VIX regime), so it also applies to tickers outside the universe.
It predicts **P(stock beats SPY over the next 5 days)** and **realised
volatility over the next 20 days**. Models are refit every quarter on all prior
data with a 20-day purge gap and scored only on the following quarter.

Out-of-sample results, Jul 2023 – Sep 2026, 10 bps per unit turnover:

| | Result |
| --- | --- |
| Volatility forecast | R² 0.44 vs 0.05 for trailing 20-day vol; median error 21% vs 27% |
| Inverse-vol weighting | Sharpe 1.25 with model forecast vs 1.09 with trailing vol, half the turnover |
| Direction signal | AUC 0.51, rank-IC 0.022; returns rise across the bottom nine deciles |
| Direction long/short | Sharpe ≈0.05 net, ≈0.3 gross (20% quantile, smoothed, 2x hold buffer) |

The volatility forecast is the useful part; the direction edge is real but too
small to survive trading costs. Caveats: the universe is today's large caps
(survivorship bias) and the test window is a strong bull market, where plain
equal weight (Sharpe 1.38) beat every model-driven portfolio. Full tables,
a Sharpe grid over quantile × cost, and an equity curve are written to
`reports/`.

## Disclaimer

This is a research/educational project. Short-horizon price prediction is
extremely noisy, and results against the naive baseline should be read with
that in mind. Not financial advice.

## License

MIT
