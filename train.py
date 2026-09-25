import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from agents.data_agent import fetch_stock_data
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.prediction_agent import PredictionAgent
from utils.pipeline import SEQUENCE_LENGTH, create_sequences

DEFAULT_PERIOD = "2y"  # 6mo leaves only ~65 training windows


def evaluate_model(name, y_true, y_pred, last_close):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    naive_rmse = mean_squared_error(y_true, last_close) ** 0.5
    actual_dir = np.sign(y_true - last_close)
    pred_dir = np.sign(y_pred - last_close)
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    print(f"{name:<12} RMSE: {rmse:8.4f} | MAE: {mae:8.4f} | "
          f"vs naive: {rmse / naive_rmse:5.3f} | direction: {dir_acc:5.1f}%")
    return rmse


def train_pipeline(symbol="AAPL", period=DEFAULT_PERIOD, interval="1d", seed=0):
    df = fetch_stock_data(symbol, period=period, interval=interval)
    print("Initial df shape:", df.shape)

    df = FeatureEngineerAgent().add_indicators(df)
    X, y, last_close, next_close, _ = create_sequences(df)
    print("X shape:", X.shape, "| y shape:", y.shape)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]
    lc_test, price_test = last_close[split:], next_close[split:]

    agent = PredictionAgent(seed=seed).fit_scaler(X_train)

    print("Training BiLSTM...")
    agent.train_bilstm(X_train, y_train)
    print("Training XGBoost...")
    agent.train_xgboost(X_train, y_train)
    print("Training Transformer...")
    agent.train_transformer(X_train, y_train)

    print(f"\nTest set: {len(X_test)} windows. Prices reconstructed as last_close * (1 + predicted return).")
    print("'vs naive' < 1 beats the 'tomorrow = today' baseline.")
    evaluate_model("Naive", price_test, lc_test, lc_test)
    preds = agent.predict(X_test)
    for name, ret in preds.items():
        evaluate_model(name, price_test, lc_test * (1 + ret), lc_test)
    ensemble = np.mean(list(preds.values()), axis=0)
    evaluate_model("Ensemble", price_test, lc_test * (1 + ensemble), lc_test)

    agent.save_models(symbol=symbol, period=period, interval=interval,
                      sequence_length=SEQUENCE_LENGTH)
    print("Models saved.")
    return agent


if __name__ == "__main__":
    train_pipeline()
