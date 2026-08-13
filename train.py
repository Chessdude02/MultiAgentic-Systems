from agents.data_agent import fetch_stock_data
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.prediction_agent import PredictionAgent
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

SEQUENCE_LENGTH = 10

def create_sequences(df, target_column="Close", seq_len=SEQUENCE_LENGTH):
    print("📦 Creating sequences...")
    features = df.drop(columns=[target_column]).values
    target = df[target_column].values

    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    print(f"📊 {name} RMSE: {round(rmse, 4)} | MAE: {round(mae, 4)}")

def train_pipeline():
    df = fetch_stock_data("AAPL", period="6mo", interval="1d")
    print("📋 Columns in fetched DataFrame:", df.columns.tolist())
    print("🔍 Initial df shape:", df.shape)

    engineer = FeatureEngineerAgent()
    df = engineer.add_indicators(df).dropna()
    print("✅ After feature engineering:", df.shape)

    X, y = create_sequences(df)
    print("📈 X shape:", X.shape, "| y shape:", y.shape)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    agent = PredictionAgent()

    print("🧠 Training BiLSTM...")
    agent.train_bilstm(X_train, y_train)
    y_pred_bilstm = agent.bilstm_model.predict(X_test).flatten()
    evaluate_model("BiLSTM", y_test, y_pred_bilstm)

    print("🧠 Training XGBoost...")
    X_flat = X.reshape(X.shape[0], -1)
    agent.train_xgboost(X_flat[:split], y_train)
    y_pred_xgb = agent.xgb_model.predict(X_flat[split:])
    evaluate_model("XGBoost", y_test, y_pred_xgb)

    print("🧠 Training Transformer...")
    agent.train_transformer(X_train, y_train)
    y_pred_transformer = agent.transformer_model.predict(X_test).flatten()
    evaluate_model("Transformer", y_test, y_pred_transformer)

    agent.save_models()
    print("💾 Models saved.")

if __name__ == "__main__":
    train_pipeline()
