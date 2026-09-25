"""Production 20-day risk model: XGBoost trained on the QLIKE loss (the best
risk forecaster in the walk-forward study) with empirically calibrated
return bands.

    python -m improvements.risk_model --train     # fit on the full panel -> models/risk/
    python -m improvements.risk_model AAPL MSFT   # print forecasts

Calibration: before the final fit, a model trained on all data except the
last two years forecasts those two years; the 80% / 95% quantiles of
|20-day return| / forecast sigma become the band multipliers.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from improvements.universe import ETFS, MARKET, VIX, _earnings_one
from improvements.vol_features import FEATURES, H, market_frame, ticker_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "risk")
HOLDOUT = 504  # sessions (~2 years) used to calibrate the bands


def _qlike_objective(y, p):
    e = np.exp(np.clip(2 * (y - p), -8, 8))
    return 2 * (1 - e), 4 * np.maximum(e, 0.05)


def _fit(train):
    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
        n_jobs=os.cpu_count(), random_state=0, objective=_qlike_objective,
        base_score=float(train["fwd_logvol"].mean()))
    model.fit(train[FEATURES], train["fwd_logvol"])
    return model


class RiskModel:
    def __init__(self, model=None, meta=None):
        self.model = model
        self.meta = meta or {}

    # ------------------------------------------------------------------ fit
    @classmethod
    def train(cls, panel):
        dates = panel.index.get_level_values("Date")
        uniq = dates.unique().sort_values()
        every5 = set(uniq[::5])
        labelled = panel.dropna(subset=["fwd_logvol"])
        ld = labelled.index.get_level_values("Date")

        # 1) calibration fit: hold out the last HOLDOUT labelled sessions
        l_uniq = ld.unique().sort_values()
        hold_start = l_uniq[-HOLDOUT]
        cal_end = l_uniq[-HOLDOUT - H - 1]
        cal_model = _fit(labelled[(ld <= cal_end) & ld.isin(every5)])
        hold = labelled[ld >= hold_start].dropna(subset=["fwd_ret"])
        sigma = np.exp(cal_model.predict(hold[FEATURES])) * np.sqrt(H / 252)
        z = hold["fwd_ret"].abs() / sigma
        k80, k95 = float(z.quantile(0.8)), float(z.quantile(0.95))
        err = cal_model.predict(hold[FEATURES]) - hold["fwd_logvol"]

        # 2) final fit on everything
        model = _fit(labelled[ld.isin(every5)])
        meta = {
            "features": FEATURES, "horizon": H, "k80": k80, "k95": k95,
            "trained_through": str(l_uniq[-1].date()),
            "holdout": {"from": str(hold_start.date()), "rows": int(len(hold)),
                        "rmse_log_vol": float(np.sqrt((err ** 2).mean())),
                        "bias_log_vol": float(err.mean())},
        }
        return cls(model, meta)

    def save(self, path=MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, "vol_qlike.json"))
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

    @classmethod
    def load(cls, path=MODEL_DIR):
        p = os.path.join(path, "vol_qlike.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No risk model in '{path}'. Run `python -m improvements.risk_model --train`.")
        model = xgb.XGBRegressor()
        model.load_model(p)
        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)
        if meta.get("features") != FEATURES:
            raise ValueError("Saved risk model uses a different feature set; retrain it.")
        return cls(model, meta)

    # ------------------------------------------------------------- forecast
    def forecast(self, symbol, period="3y"):
        """20-day risk view for any ticker (stock or ETF)."""
        import yfinance as yf

        symbol = symbol.upper().strip()
        tickers = list(dict.fromkeys([symbol, MARKET, VIX]))
        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                          progress=False, threads=True)
        if raw is None or raw.empty or symbol not in raw["Close"].columns \
                or raw["Close"][symbol].notna().sum() < 300:
            raise ValueError(f"Not enough price history for '{symbol}' (need ~300 sessions).")
        ohlcv = lambda s: pd.DataFrame({k: raw[k][s] for k in ("Open", "High", "Low", "Close", "Volume")})
        mkt = market_frame(ohlcv(MARKET), raw["Close"][VIX])

        is_fund = symbol in ETFS
        earn = pd.DatetimeIndex([]) if is_fund else _earnings_one(symbol)[1]
        f = ticker_frame(ohlcv(symbol), mkt, earn)
        if f is None:
            raise ValueError(f"Could not build features for '{symbol}'.")
        f["is_fund"] = 1.0 if is_fund else 0.0
        f = f.dropna(subset=[c for c in FEATURES if c not in ("days_since_earn", "earn_react")])
        last = f.iloc[[-1]]

        logvol = float(self.model.predict(last[FEATURES])[0])
        vol = float(np.exp(logvol))
        price = float(raw["Close"][symbol].dropna().iloc[-1])
        s = vol * np.sqrt(H / 252)
        k80, k95 = self.meta["k80"], self.meta["k95"]
        hist = np.exp(f["cc_22"].astype("float64"))
        upcoming = [d for d in earn if d > f.index[-1]]
        return {
            "symbol": symbol,
            "as_of": str(f.index[-1].date()),
            "price": price,
            "horizon_days": H,
            "forecast_vol": vol,
            "trailing_vol": float(hist.iloc[-1]),
            "vol_percentile_3y": float((hist < vol).mean() * 100),
            "range_80": (price * np.exp(-k80 * s), price * np.exp(k80 * s)),
            "range_95": (price * np.exp(-k95 * s), price * np.exp(k95 * s)),
            "days_since_earnings": None if pd.isna(last["days_since_earn"].iloc[0]) else int(last["days_since_earn"].iloc[0]),
            "next_earnings": str(upcoming[0].date()) if upcoming else None,
            "vix": float(np.exp(last["log_vix"].iloc[0]) * 100),
            "model_trained_through": self.meta.get("trained_through"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        from improvements.vol_backtest import load_panel
        panel, _ = load_panel()
        rm = RiskModel.train(panel)
        rm.save()
        print(json.dumps(rm.meta | {"features": f"{len(FEATURES)} features"}, indent=2))
    if args.symbols:
        rm = RiskModel.load()
        for s in args.symbols:
            try:
                print(json.dumps(rm.forecast(s), indent=2, default=float))
            except ValueError as e:
                print(f"{s}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
