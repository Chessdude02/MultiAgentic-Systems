import ta
import pandas as pd

# MACD's slow EMA (26) is the longest lookback; SEQUENCE_LENGTH more rows are
# needed on top of that to build even one training window.
MIN_ROWS = 40


class FeatureEngineerAgent:
    def add_indicators(self, df):
        print("Engineering features...")

        if len(df) < MIN_ROWS:
            raise ValueError(f"Not enough data to compute indicators: {len(df)} rows, "
                             f"minimum {MIN_ROWS} required. Choose a longer period.")

        df = df.copy()
        close = df["Close"]
        macd = ta.trend.MACD(close=close)

        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        df["SMA"] = ta.trend.SMAIndicator(close=close, window=14).sma_indicator()
        df["EMA"] = ta.trend.EMAIndicator(close=close, window=14).ema_indicator()
        df["MACD"] = macd.macd()
        df["Signal"] = macd.macd_signal()
        df["ROC"] = ta.momentum.ROCIndicator(close=close, window=12).roc()
        df["Volatility"] = ta.volatility.AverageTrueRange(
            high=df["High"],
            low=df["Low"],
            close=close,
            window=14
        ).average_true_range()

        return df
