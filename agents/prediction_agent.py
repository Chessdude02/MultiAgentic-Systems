import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib
import os

class PredictionAgent:
    def __init__(self):
        self.bilstm_model = None
        self.xgb_model = None
        self.transformer_model = None

    def train_bilstm(self, X, y):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=False), input_shape=(X.shape[1], X.shape[2])),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')  # use proper loss name
        model.fit(X, y, epochs=20, batch_size=16, callbacks=[EarlyStopping(patience=5)], verbose=0)
        self.bilstm_model = model

    def train_xgboost(self, X, y):
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        self.xgb_model = model

    def train_transformer(self, X, y):
        input_layer = Input(shape=(X.shape[1], X.shape[2]))
        x = LayerNormalization()(input_layer)
        x = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = Dropout(0.1)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')  # use proper loss name
        model.fit(X, y, epochs=20, batch_size=16, callbacks=[EarlyStopping(patience=5)], verbose=0)
        self.transformer_model = model

    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)
        if self.bilstm_model:
            self.bilstm_model.save(os.path.join(path, "bilstm_model.h5"))
        if self.xgb_model:
            joblib.dump(self.xgb_model, os.path.join(path, "xgb_model.pkl"))
        if self.transformer_model:
            self.transformer_model.save(os.path.join(path, "transformer_model.h5"))

    def load_models(self, path="models"):
        if os.path.exists(os.path.join(path, "bilstm_model.h5")):
            self.bilstm_model = load_model(os.path.join(path, "bilstm_model.h5"))
            print("✅ BiLSTM model loaded.")
        else:
            print("⚠️ BiLSTM model not found.")

        xgb_path = os.path.join(path, "xgb_model.pkl")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            print("✅ XGBoost model loaded.")
        else:
            print("⚠️ XGBoost model file not found.")

        if os.path.exists(os.path.join(path, "transformer_model.h5")):
            self.transformer_model = load_model(os.path.join(path, "transformer_model.h5"))
            print("✅ Transformer model loaded.")
        else:
            print("⚠️ Transformer model not found.")
