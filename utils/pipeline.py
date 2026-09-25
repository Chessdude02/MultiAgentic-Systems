"""Shared data pipeline for training, evaluation and live prediction.

The models predict the *next-period return* from a window of *stationary*
features, then convert that back to a price. Feeding raw price levels and
raw volume (hundreds of millions) into a neural net with an absolute price
target made the saved models collapse to ~$7 predictions for a ~$300 stock.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEQUENCE_LENGTH = 10
TARGET = "Close"
SCALER_FILE = "scaler.pkl"
META_FILE = "meta.json"


def make_features(df):
    """Turn an indicator-enriched OHLCV frame into scale-free features.

    Every price-level column is expressed relative to the same bar's close, so
    the features look the same whether the stock trades at $30 or $300.
    """
    close = df[TARGET]
    feats = pd.DataFrame(index=df.index)
    feats["Return"] = close.pct_change()
    feats["LogVolChg"] = np.log(df["Volume"].replace(0, np.nan)).diff()
    feats["HighRel"] = df["High"] / close - 1
    feats["LowRel"] = df["Low"] / close - 1
    feats["OpenRel"] = df["Open"] / close - 1
    feats["RSI"] = df["RSI"] / 100
    feats["SMARel"] = df["SMA"] / close - 1
    feats["EMARel"] = df["EMA"] / close - 1
    feats["MACDRel"] = df["MACD"] / close
    feats["SignalRel"] = df["Signal"] / close
    feats["ROC"] = df["ROC"] / 100
    feats["VolatilityRel"] = df["Volatility"] / close
    return feats.replace([np.inf, -np.inf], np.nan)


def create_sequences(df, seq_len=SEQUENCE_LENGTH):
    """Window the features and build next-period return targets.

    Returns
        X           (n, seq_len, n_features) raw (unscaled) feature windows
        y_ret       (n,) return from the window's last close to the next close
        last_close  (n,) close at the end of each window
        next_close  (n,) actual close being predicted
        index       timestamps of the predicted closes
    """
    feats = make_features(df)
    valid = feats.notna().all(axis=1)
    feats, close = feats[valid], df.loc[valid, TARGET]

    f = feats.values.astype("float32")
    c = close.values.astype("float64")

    X, y_ret, last_close, next_close = [], [], [], []
    for i in range(len(feats) - seq_len):
        X.append(f[i:i + seq_len])
        last_close.append(c[i + seq_len - 1])
        next_close.append(c[i + seq_len])
        y_ret.append(c[i + seq_len] / c[i + seq_len - 1] - 1)
    return (
        np.array(X, dtype="float32").reshape(-1, seq_len, f.shape[1]),
        np.array(y_ret, dtype="float32"),
        np.array(last_close),
        np.array(next_close),
        close.index[seq_len:],
    )


def latest_window(df, seq_len=SEQUENCE_LENGTH):
    """Most recent feature window, for predicting the period after the data."""
    feats = make_features(df).dropna()
    if len(feats) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} complete feature rows to predict, got "
            f"{len(feats)}. Choose a longer period."
        )
    return feats.values[-seq_len:].astype("float32")[None, ...]


class WindowScaler:
    """StandardScaler applied per feature across all timesteps of a window."""

    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()

    def fit(self, X):
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        return self

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")

    def save(self, path):
        joblib.dump(self.scaler, os.path.join(path, SCALER_FILE))

    @classmethod
    def load(cls, path):
        return cls(joblib.load(os.path.join(path, SCALER_FILE)))


def save_meta(path, **meta):
    with open(os.path.join(path, META_FILE), "w") as fh:
        json.dump(meta, fh, indent=2)


def load_meta(path):
    p = os.path.join(path, META_FILE)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)
