import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol="AAPL", period="6mo", interval="1d"):
    print(f"Fetching {symbol} ({period}, {interval})...")
    df = yf.download(tickers=symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for '{symbol}' ({period}, {interval}). "
                         "Check the symbol and your network connection.")

    # Drop the ticker level yfinance adds to the columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None

    print("Columns in fetched DataFrame:", df.columns.tolist())
    return df
