import numpy as np

def analyze_strategy(predictions, prices):
    # Ensure input arrays are 1D and same length
    predictions = np.asarray(predictions).flatten()
    prices = np.asarray(prices).flatten()

    min_len = min(len(predictions), len(prices))
    predictions = predictions[:min_len]
    prices = prices[:min_len]

    # Avoid empty arrays
    if len(predictions) < 2 or len(prices) < 2:
        return {"Sharpe Ratio": 0.0, "Return %": 0.0}

    try:
        returns = np.diff(predictions) / predictions[:-1]
        strategy_returns = np.diff(prices) / prices[:-1]
        sharpe = (returns.mean() - strategy_returns.mean()) / (returns.std() + 1e-6)
        total_return = (prices[-1] - prices[0]) / prices[0]
        return {
            "Sharpe Ratio": round(sharpe, 2),
            "Return %": round(total_return * 100, 2)
        }
    except Exception as e:
        print("⚠️ Error in analyze_strategy:", e)
        return {"Sharpe Ratio": 0.0, "Return %": 0.0}
