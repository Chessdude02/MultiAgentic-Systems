import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol="AAPL", period="6mo", interval="1d"):
    print("▶ Fetching data...")
    df = yf.download(tickers=symbol, period=period, interval=interval)

    # 🛠️ Drop multi-index if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print("📋 Columns in fetched DataFrame:", df.columns.tolist())
    return df
