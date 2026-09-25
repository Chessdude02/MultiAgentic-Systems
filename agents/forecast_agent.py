"""Pooled multi-ticker forecaster.

Two gradient-boosted models trained on the (Date, Ticker) panel from
``utils.panel``:
    direction  P(stock beats SPY over the next HORIZON days)
    volatility log annualised realised volatility over the next VOL_HORIZON days
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.panel import FEATURES, HORIZON, VOL_HORIZON

POOLED_DIR = os.path.join("models", "pooled")


class ForecastAgent:
    def __init__(self, seed=0, n_jobs=-1):
        common = dict(tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, n_jobs=n_jobs)
        # Heavily regularised: the direction signal is tiny and noisy.
        self.direction = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=3,
            min_child_weight=200, reg_lambda=10.0, eval_metric="logloss", **common)
        self.volatility = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            min_child_weight=100, **common)
        self.meta = {}

    def fit(self, panel):
        d = panel.dropna(subset=["up"])
        self.direction.fit(d[FEATURES], d["up"].astype(int))
        v = panel.dropna(subset=["fwd_logvol"])
        self.volatility.fit(v[FEATURES], v["fwd_logvol"])
        last = panel.index.get_level_values("Date").max()
        self.meta = {"features": FEATURES, "horizon": HORIZON, "vol_horizon": VOL_HORIZON,
                     "trained_through": str(pd.Timestamp(last).date()),
                     "n_rows": int(len(d))}
        return self

    def predict(self, panel):
        X = panel[FEATURES]
        return pd.DataFrame({
            "p_up": self.direction.predict_proba(X)[:, 1],
            "pred_logvol": self.volatility.predict(X),
        }, index=panel.index)

    def feature_importance(self, top=10):
        imp = self.direction.get_booster().get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False).head(top)

    # ------------------------------------------------------------ persist
    def save(self, path=POOLED_DIR):
        os.makedirs(path, exist_ok=True)
        self.direction.save_model(os.path.join(path, "direction.json"))
        self.volatility.save_model(os.path.join(path, "volatility.json"))
        with open(os.path.join(path, "meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

    @classmethod
    def load(cls, path=POOLED_DIR):
        if not os.path.exists(os.path.join(path, "direction.json")):
            raise FileNotFoundError(f"No pooled model in '{path}/'. Run `python backtest.py --save` first.")
        agent = cls()
        agent.direction.load_model(os.path.join(path, "direction.json"))
        agent.volatility.load_model(os.path.join(path, "volatility.json"))
        with open(os.path.join(path, "meta.json")) as fh:
            agent.meta = json.load(fh)
        if agent.meta.get("features") != FEATURES:
            raise ValueError("Saved pooled model was trained on a different feature set; retrain it.")
        return agent


def outlook(symbol, agent=None, period="2y"):
    """Latest pooled-model view of ``symbol`` (works for tickers outside the universe)."""
    from utils.panel import single_ticker_panel

    agent = agent or ForecastAgent.load()
    rows = single_ticker_panel(symbol, period=period)
    last = rows.iloc[[-1]]
    pred = agent.predict(last).iloc[0]
    return {
        "as_of": str(last.index.get_level_values("Date")[0].date()),
        "horizon": agent.meta.get("horizon", HORIZON),
        "vol_horizon": agent.meta.get("vol_horizon", VOL_HORIZON),
        "p_beat_spy": float(pred["p_up"]),
        "forecast_vol": float(np.exp(pred["pred_logvol"])),
        "trailing_vol": float(np.exp(last["logvol_20"].iloc[0])),
        "trained_through": agent.meta.get("trained_through"),
    }
