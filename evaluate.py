"""Evaluate saved models in models/ on a held-out test split.

Reproduces train.py's data pipeline (utils.pipeline: stationary features,
next-period return target, same SEQUENCE_LENGTH and 80/20 chronological
split), then reports a full metric set on reconstructed prices. Does NOT
retrain or overwrite saved models.

Usage:
    python evaluate.py                      # AAPL, 2y, 1d (train.py defaults)
    python evaluate.py --symbol MSFT --period 1y
    python evaluate.py --cache              # reuse/save data/<symbol>_<period>_<interval>.csv
    python evaluate.py --retrain            # train fresh models in-memory (no overwrite)
    python evaluate.py --retrain --runs 3   # average metrics over N seeds
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from agents.data_agent import fetch_stock_data
from agents.feature_engineer_agent import FeatureEngineerAgent
from train import DEFAULT_PERIOD
from utils.pipeline import create_sequences

MODEL_DIR = "models"
DATA_DIR = "data"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(symbol, period, interval, use_cache):
    cache_path = os.path.join(DATA_DIR, f"{symbol}_{period}_{interval}.csv")
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = fetch_stock_data(symbol, period=period, interval=interval)
    if use_cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(cache_path)
        print(f"Cached data to {cache_path}")
    return df


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred, last_close=None):
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    out = {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(err))),
        "MedAE": float(np.median(np.abs(err))),
        "MaxErr": float(np.max(np.abs(err))),
        "MAPE %": float(np.mean(np.abs(err / y_true)) * 100),
        "sMAPE %": float(
            np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100
        ),
        "Bias": float(np.mean(err)),
    }

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    out["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # normalised RMSE, comparable across symbols / price levels
    out["RMSE %"] = out["RMSE"] / float(np.mean(y_true)) * 100

    if last_close is not None:
        last_close = np.asarray(last_close, dtype="float64").ravel()
        actual_dir = np.sign(y_true - last_close)
        pred_dir = np.sign(y_pred - last_close)
        mask = actual_dir != 0
        # a forecast that never implies a direction (e.g. pure persistence)
        # has no meaningful directional accuracy
        if not mask.any() or not np.any(pred_dir != 0):
            out["DirAcc %"] = float("nan")
        else:
            out["DirAcc %"] = float(
                np.mean(actual_dir[mask] == pred_dir[mask]) * 100
            )
        # Theil's U / MASE-style ratio vs persistence baseline
        naive_rmse = float(np.sqrt(np.mean((last_close - y_true) ** 2)))
        out["vs Naive"] = out["RMSE"] / naive_rmse if naive_rmse > 0 else float("nan")

    return out


def print_table(results):
    cols = [
        "RMSE", "MAE", "MedAE", "MAPE %", "sMAPE %",
        "R2", "RMSE %", "DirAcc %", "vs Naive", "Bias", "MaxErr",
    ]
    name_w = max(len(n) for n in results) + 2
    header = "Model".ljust(name_w) + "".join(c.rjust(10) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(name_w)
        for c in cols:
            v = m.get(c, float("nan"))
            row += ("n/a" if v != v else f"{v:,.3f}").rjust(10)
        print(row)
    print("-" * len(header))
    print(
        "\nRMSE/MAE/MedAE/MaxErr/Bias are in dollars. 'vs Naive' < 1.0 means the\n"
        "model beats a last-close persistence forecast; >= 1.0 means it does not.\n"
        "DirAcc % is up/down direction accuracy vs the previous close (50% = coin flip)."
    )


# ----------------------------------------------------------------------------
# model loading / prediction
# ----------------------------------------------------------------------------
def retrain_and_predict(X_train, y_train, X_test, seed):
    """Train fresh models on the train split only and predict test returns.
    Nothing is saved, so models/ is left untouched."""
    from agents.prediction_agent import PredictionAgent

    agent = PredictionAgent(seed=seed).fit_scaler(X_train)
    agent.train_bilstm(X_train, y_train)
    agent.train_transformer(X_train, y_train)
    agent.train_xgboost(X_train, y_train)
    return agent.predict(X_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--period", default=DEFAULT_PERIOD)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="train fresh models on the train split instead of loading models/",
    )
    ap.add_argument(
        "--runs", type=int, default=1, help="seeds to average over with --retrain"
    )
    args = ap.parse_args()

    df = load_data(args.symbol, args.period, args.interval, args.cache)
    df = FeatureEngineerAgent().add_indicators(df)

    X, y_ret, last_close, next_close, index = create_sequences(df)
    split = int(len(X) * 0.8)
    X_test, y_test, lc_test = X[split:], next_close[split:], last_close[split:]
    test_index = index[split:]
    print(f"Features: {X.shape[-1]} | Train windows: {split} | Test windows: {len(X_test)}")
    print(
        f"Test period: {test_index[0].date()} -> {test_index[-1].date()}  |  "
        f"price range ${y_test.min():.2f}-${y_test.max():.2f}"
    )

    results = {}
    preds = {}

    # --- baselines ------------------------------------------------------
    results["Naive (last close)"] = metrics(y_test, lc_test, lc_test)
    results["Always +1%"] = metrics(y_test, lc_test * 1.01, lc_test)

    if args.retrain:
        X_train, y_train = X[:split], y_ret[:split]
        runs = []
        for s in range(args.runs):
            print(f"Training fresh models (seed {s})...")
            runs.append(retrain_and_predict(X_train, y_train, X_test, s))
        for name in runs[0]:
            # average predicted prices across seeds for a stable point estimate
            prices = [lc_test * (1 + r[name]) for r in runs]
            preds[name] = np.mean(prices, axis=0)
            results[name] = metrics(y_test, preds[name], lc_test)
            if args.runs > 1:
                per_seed_rmse = [metrics(y_test, p)["RMSE"] for p in prices]
                results[name]["RMSE sd"] = float(np.std(per_seed_rmse))
                print(f"  {name}: per-seed RMSE " + ", ".join(f"{v:.2f}" for v in per_seed_rmse))
    else:
        from agents.prediction_agent import PredictionAgent
        try:
            agent = PredictionAgent().load_models(MODEL_DIR)
        except FileNotFoundError as e:
            print(f"  ! {e}")
            return
        trained = agent.meta
        if trained and (trained.get("symbol"), trained.get("period"), trained.get("interval")) != (
            args.symbol, args.period, args.interval
        ):
            print(f"  ! Saved models were trained on {trained}; this test split "
                  "may overlap their training data.")
        for name, ret in agent.predict(X_test).items():
            preds[name] = lc_test * (1 + ret)
            results[name] = metrics(y_test, preds[name], lc_test)

    # indicators as of each window's last bar, aligned to the predicted bar
    finalize(results, preds, y_test, lc_test, df.shift(1).loc[test_index])


def finalize(results, preds, y_test, lc_test, df_signal):
    # --- ensemble -------------------------------------------------------
    if len(preds) > 1:
        from agents.ensemble_agent import EnsembleAgent
        yp = EnsembleAgent().average(preds)
        results[f"Ensemble (avg of {len(preds)})"] = metrics(y_test, yp, lc_test)

    # --- symbolic override on best available model -----------------------
    # Rules use the indicators of the bar the prediction is made from, not
    # the bar being predicted (that would leak the future).
    if preds:
        from agents.symbolic_override_agent import SymbolicOverrideAgent
        base_name = min(preds, key=lambda k: results[k]["RMSE"])
        try:
            adj = SymbolicOverrideAgent().override(df_signal, preds[base_name])
            results[f"{base_name} + Symbolic"] = metrics(y_test, adj, lc_test)
        except Exception as e:
            print(f"  ! Symbolic override failed: {e}")

    print_table(results)


if __name__ == "__main__":
    main()
