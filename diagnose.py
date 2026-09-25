"""Diagnostic: isolate WHY the current pipeline produces unusable metrics.

Tests three variants of the same BiLSTM on the same split:
  A) as-is            - raw unscaled features, absolute price target (current code)
  B) scaled features  - StandardScaler on features, absolute price target
  C) scaled + returns - scaled features, next-day % return target, then
                        reconstructed back to a price for comparable RMSE

If B/C are dramatically better than A, the problem is preprocessing, not
model architecture.
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd
from sklearn.preprocessing import StandardScaler

from agents.feature_engineer_agent import FeatureEngineerAgent

SEQ = 10
SEED = 0


def build_bilstm(shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(SEED)
    m = Sequential([
        Bidirectional(LSTM(64), input_shape=shape),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m, EarlyStopping(patience=5, restore_best_weights=True)


def rmse(a, b):
    a, b = np.ravel(a).astype("float64"), np.ravel(b).astype("float64")
    return float(np.sqrt(np.mean((a - b) ** 2))), float(np.mean(np.abs(a - b)))


def main():
    df = pd.read_csv("data/AAPL_6mo_1d.csv", index_col=0, parse_dates=True)
    df = FeatureEngineerAgent().add_indicators(df).dropna()

    fcols = [c for c in df.columns if c != "Close"]
    feats = df[fcols].values.astype("float64")
    close = df["Close"].values.astype("float64")

    X, y, lc = [], [], []
    for i in range(len(df) - SEQ):
        X.append(feats[i:i + SEQ])
        y.append(close[i + SEQ])
        lc.append(close[i + SEQ - 1])
    X, y, lc = np.array(X), np.array(y), np.array(lc)

    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    lc_te = lc[split:]

    print(f"Train {len(Xtr)} / Test {len(Xte)}")
    print(f"Train price range ${ytr.min():.2f}-${ytr.max():.2f}")
    print(f"Test  price range ${yte.min():.2f}-${yte.max():.2f}  <-- note overlap")
    print(f"Raw feature magnitudes (max per column):")
    for c, v in zip(fcols, np.abs(feats).max(axis=0)):
        print(f"    {c:<12} {v:,.2f}")

    n_r, n_m = rmse(yte, lc_te)
    print(f"\nBaseline naive(last close)   RMSE {n_r:8.3f}  MAE {n_m:8.3f}")

    # --- A: as-is ------------------------------------------------------
    m, es = build_bilstm(Xtr.shape[1:])
    m.fit(Xtr, ytr, epochs=20, batch_size=16, callbacks=[es], verbose=0)
    pa = m.predict(Xte, verbose=0).ravel()
    r, a = rmse(yte, pa)
    print(f"A) raw features, price tgt   RMSE {r:8.3f}  MAE {a:8.3f}"
          f"   mean pred ${pa.mean():.2f}")

    # --- B: scaled features --------------------------------------------
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr_s = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xte_s = sc.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape)

    m, es = build_bilstm(Xtr_s.shape[1:])
    m.fit(Xtr_s, ytr, epochs=20, batch_size=16, callbacks=[es], verbose=0)
    pb = m.predict(Xte_s, verbose=0).ravel()
    r, a = rmse(yte, pb)
    print(f"B) scaled feats, price tgt   RMSE {r:8.3f}  MAE {a:8.3f}"
          f"   mean pred ${pb.mean():.2f}")

    # --- C: scaled + return target -------------------------------------
    lc_tr = lc[:split]
    ret_tr = (ytr - lc_tr) / lc_tr

    m, es = build_bilstm(Xtr_s.shape[1:])
    m.fit(Xtr_s, ret_tr, epochs=200, batch_size=16, callbacks=[es], verbose=0)
    pred_ret = m.predict(Xte_s, verbose=0).ravel()
    pc = lc_te * (1 + pred_ret)
    r, a = rmse(yte, pc)
    print(f"C) scaled feats, return tgt  RMSE {r:8.3f}  MAE {a:8.3f}"
          f"   mean pred ${pc.mean():.2f}")

    print(
        "\nInterpretation: if A collapses to a near-zero mean prediction while\n"
        "B/C land near the real price range, the pipeline is missing feature\n"
        "scaling and a stationary target - not a model-capacity problem."
    )


if __name__ == "__main__":
    main()
