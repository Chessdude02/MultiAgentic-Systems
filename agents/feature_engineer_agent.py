import ta
import pandas as pd
class FeatureEngineerAgent:
    def add_indicators(self, df):
        print("🔧 Engineering features...")

        if len(df) < 20:
            raise ValueError("❌ Not enough data to compute indicators. Minimum 20 rows required.")

        close = df["Close"]

        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["SMA"] = ta.trend.SMAIndicator(close=close, window=14).sma_indicator()
        df["EMA"] = ta.trend.EMAIndicator(close=close, window=14).ema_indicator()
        df["MACD"] = ta.trend.MACD(close=close).macd()
        df["Signal"] = ta.trend.MACD(close=close).macd_signal()
        df["ROC"] = ta.momentum.ROCIndicator(close=close, window=12).roc()
        df["Volatility"] = ta.volatility.AverageTrueRange(
            high=df["High"],
            low=df["Low"],
            close=close,
            window=14
        ).average_true_range()

        return df
