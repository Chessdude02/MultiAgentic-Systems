import os

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D, Input,
    LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model, Sequential, load_model

from utils.pipeline import WindowScaler, load_meta, save_meta

# Daily returns are ~0.01; training on percent keeps the loss well-conditioned.
TARGET_SCALE = 100.0
VAL_FRACTION = 0.2

BILSTM_FILE = "bilstm_model.keras"
TRANSFORMER_FILE = "transformer_model.keras"
XGB_FILE = "xgb_model.pkl"


def _early_stopping():
    return EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


class PredictionAgent:
    """Trains and serves models that predict next-period returns.

    All ``train_*`` and ``predict`` methods take *unscaled* feature windows
    from ``utils.pipeline.create_sequences``; the agent owns the scaler.
    """

    def __init__(self, seed=None):
        self.bilstm_model = None
        self.xgb_model = None
        self.transformer_model = None
        self.scaler = None
        self.meta = {}
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

    # ------------------------------------------------------------------ fit
    def fit_scaler(self, X):
        self.scaler = WindowScaler().fit(X)
        return self

    def _prep(self, X):
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted/loaded; call fit_scaler() or load_models() first.")
        return self.scaler.transform(X)

    def train_bilstm(self, X, y):
        Xs = self._prep(X)
        model = Sequential([
            Input(shape=Xs.shape[1:]),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        # validation_split takes the *last* fraction, so this stays chronological
        model.fit(Xs, y * TARGET_SCALE, epochs=200, batch_size=16,
                  validation_split=VAL_FRACTION, shuffle=True,
                  callbacks=[_early_stopping()], verbose=0)
        self.bilstm_model = model

    def train_transformer(self, X, y):
        Xs = self._prep(X)
        inp = Input(shape=Xs.shape[1:])
        x = LayerNormalization()(inp)
        attn = MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(x, x)
        x = LayerNormalization()(x + attn)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="relu")(x)
        model = Model(inputs=inp, outputs=Dense(1)(x))
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xs, y * TARGET_SCALE, epochs=200, batch_size=16,
                  validation_split=VAL_FRACTION, shuffle=True,
                  callbacks=[_early_stopping()], verbose=0)
        self.transformer_model = model

    def train_xgboost(self, X, y):
        Xs = self._prep(X).reshape(len(X), -1)
        n_val = max(1, int(len(Xs) * VAL_FRACTION))
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            early_stopping_rounds=30,
        )
        model.fit(Xs[:-n_val], y[:-n_val] * TARGET_SCALE,
                  eval_set=[(Xs[-n_val:], y[-n_val:] * TARGET_SCALE)], verbose=False)
        self.xgb_model = model

    # -------------------------------------------------------------- predict
    def predict(self, X):
        """Predicted next-period returns (fractions) per available model."""
        Xs = self._prep(X)
        out = {}
        if self.bilstm_model is not None:
            out["BiLSTM"] = self.bilstm_model.predict(Xs, verbose=0).ravel()
        if self.transformer_model is not None:
            out["Transformer"] = self.transformer_model.predict(Xs, verbose=0).ravel()
        if self.xgb_model is not None:
            out["XGBoost"] = self.xgb_model.predict(Xs.reshape(len(Xs), -1)).ravel()
        return {k: v.astype("float64") / TARGET_SCALE for k, v in out.items()}

    # ------------------------------------------------------------ persist
    def save_models(self, path="models", **meta):
        os.makedirs(path, exist_ok=True)
        if self.bilstm_model:
            self.bilstm_model.save(os.path.join(path, BILSTM_FILE))
        if self.xgb_model:
            joblib.dump(self.xgb_model, os.path.join(path, XGB_FILE))
        if self.transformer_model:
            self.transformer_model.save(os.path.join(path, TRANSFORMER_FILE))
        if self.scaler:
            self.scaler.save(path)
        save_meta(path, **meta)

    def load_models(self, path="models"):
        try:
            self.scaler = WindowScaler.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained models in '{path}/'. Run `python train.py` first."
            ) from None
        self.meta = load_meta(path)

        p = os.path.join(path, BILSTM_FILE)
        if os.path.exists(p):
            self.bilstm_model = load_model(p)
            print("BiLSTM model loaded.")
        else:
            print("BiLSTM model not found.")

        p = os.path.join(path, XGB_FILE)
        if os.path.exists(p):
            self.xgb_model = joblib.load(p)
            print("XGBoost model loaded.")
        else:
            print("XGBoost model not found.")

        p = os.path.join(path, TRANSFORMER_FILE)
        if os.path.exists(p):
            self.transformer_model = load_model(p)
            print("Transformer model loaded.")
        else:
            print("Transformer model not found.")
        return self

    def has_models(self):
        return any(m is not None for m in (self.bilstm_model, self.xgb_model, self.transformer_model))
