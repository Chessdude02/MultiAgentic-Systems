"""Pooled multi-ticker dataset: one row per (Date, Ticker).

Features are scale-free and expressed relative to the market (SPY) where it
matters, so a single model can learn from ~100 stocks at once and still be
applied to any ticker, inside the universe or not.

Targets (all strictly forward-looking, used for training/evaluation only):
    fwd_ret      return over the next ``horizon`` days
    fwd_excess   fwd_ret minus SPY's return over the same days
    up           1 if the stock beats SPY over the horizon, else 0
    fwd_logvol   log annualised realised volatility over the next VOL_HORIZON days
"""

import hashlib
import os

import numpy as np
import pandas as pd
import ta
import yfinance as yf

HORIZON = 5
VOL_HORIZON = 20
ANN = np.sqrt(252)
MARKET = "SPY"
VIX = "^VIX"
DATA_DIR = "data"

# S&P 100-style large caps. NOTE: today's constituents, so backtests carry
# survivorship bias (companies that fell out of the index are missing).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "CVX", "LLY", "ABBV", "MRK",
    "PEP", "KO", "COST", "WMT", "BAC", "PFE", "TMO", "AVGO", "CSCO", "ACN",
    "MCD", "ABT", "DHR", "NKE", "DIS", "ADBE", "CRM", "NFLX", "TXN", "LIN",
    "NEE", "PM", "ORCL", "CMCSA", "VZ", "T", "INTC", "AMD", "QCOM", "UPS",
    "HON", "LOW", "IBM", "AMGN", "SBUX", "CAT", "GS", "MS", "BLK", "SPGI",
    "AXP", "DE", "LMT", "RTX", "BA", "GE", "MMM", "MDT", "BMY", "GILD",
    "CVS", "CI", "ISRG", "SYK", "INTU", "AMAT", "NOW", "BKNG", "ADP", "MDLZ",
    "MO", "CL", "USB", "PNC", "C", "WFC", "SCHW", "TGT", "F", "GM",
    "DUK", "SO", "COP", "EOG", "SLB", "OXY", "KHC", "ELV", "PYPL", "MU",
]

FEATURES = [
    # own momentum / reversal
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120",
    # relative to market
    "exret_5", "exret_20", "exret_60", "beta_60",
    # volatility
    "logvol_5", "logvol_20", "logvol_60", "vol_ratio", "atr_rel", "hl_range_5",
    # trend / oscillators
    "rsi_14", "macd_rel", "macd_hist_rel", "sma20_dist", "sma50_dist", "sma200_dist", "dist_high_252",
    # activity
    "volume_z",
    # market regime
    "mkt_ret_5", "mkt_ret_20", "mkt_logvol_20", "vix", "vix_chg_5",
]
TARGETS = ["fwd_ret", "fwd_excess", "up", "fwd_logvol"]


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------
def download(symbols, period="6y", cache=True):
    """OHLCV for ``symbols`` plus SPY and ^VIX as {field: DataFrame[date x ticker]}."""
    symbols = list(dict.fromkeys(list(symbols) + [MARKET, VIX]))
    key = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    path = os.path.join(DATA_DIR, f"panel_{period}_{key}.pkl")
    if cache and os.path.exists(path):
        print(f"Loading cached prices from {path}")
        return pd.read_pickle(path)

    print(f"Downloading {len(symbols)} tickers ({period})...")
    raw = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    prices = {f: raw[f] for f in ["Open", "High", "Low", "Close", "Volume"]}
    missing = [s for s in symbols if prices["Close"].get(s) is None or prices["Close"][s].notna().sum() == 0]
    if MARKET in missing or VIX in missing:
        raise ValueError(f"Could not download {MARKET}/{VIX}.")
    if missing:
        print(f"  ! no data for: {missing}")
    if cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.to_pickle(prices, path)
    return prices


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def market_frame(spy_close, vix_close, horizon=HORIZON):
    # The multi-ticker download aligns every symbol on one calendar, which
    # leaves NaN holes; work on SPY's own trading days so rolling windows and
    # forward shifts are measured in real sessions.
    spy_close = spy_close.dropna()
    vix_close = vix_close.reindex(spy_close.index).ffill()
    r = spy_close.pct_change()
    return pd.DataFrame({
        "mkt_close": spy_close,
        "mkt_ret_5": spy_close / spy_close.shift(5) - 1,
        "mkt_ret_20": spy_close / spy_close.shift(20) - 1,
        "mkt_ret_60": spy_close / spy_close.shift(60) - 1,
        "mkt_r": r,
        "mkt_logvol_20": np.log(r.rolling(20).std() * ANN),
        "vix": vix_close,
        "vix_chg_5": vix_close / vix_close.shift(5) - 1,
        "mkt_fwd_ret": spy_close.shift(-horizon) / spy_close - 1,
    })


def ticker_features(ohlcv, mkt, horizon=HORIZON):
    """Features + targets for one ticker. ``ohlcv`` has Open/High/Low/Close/Volume."""
    # restrict to sessions where both the stock and SPY traded
    df = ohlcv.dropna(subset=["Close"]).join(mkt, how="inner")
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    r = c.pct_change()

    f = pd.DataFrame(index=df.index)
    f["ret_1"] = r
    for n in (5, 20, 60, 120):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["exret_5"] = f["ret_5"] - df["mkt_ret_5"]
    f["exret_20"] = f["ret_20"] - df["mkt_ret_20"]
    f["exret_60"] = f["ret_60"] - df["mkt_ret_60"]
    f["beta_60"] = r.rolling(60).cov(df["mkt_r"]) / df["mkt_r"].rolling(60).var()

    vol = {n: r.rolling(n).std() * ANN for n in (5, 20, 60)}
    for n, s in vol.items():
        f[f"logvol_{n}"] = np.log(s.replace(0, np.nan))
    f["vol_ratio"] = vol[5] / vol[60]
    f["atr_rel"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c
    f["hl_range_5"] = ((h - l) / c).rolling(5).mean()

    macd = ta.trend.MACD(close=c)
    f["rsi_14"] = ta.momentum.RSIIndicator(close=c, window=14).rsi() / 100
    f["macd_rel"] = macd.macd() / c
    f["macd_hist_rel"] = macd.macd_diff() / c
    for n in (20, 50, 200):
        f[f"sma{n}_dist"] = c / c.rolling(n).mean() - 1
    f["dist_high_252"] = c / c.rolling(252, min_periods=200).max() - 1

    lv = np.log(v.replace(0, np.nan))
    f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    for col in ("mkt_ret_5", "mkt_ret_20", "mkt_logvol_20", "vix", "vix_chg_5"):
        f[col] = df[col]

    # targets -----------------------------------------------------------
    f["fwd_ret"] = c.shift(-horizon) / c - 1
    f["fwd_excess"] = f["fwd_ret"] - df["mkt_fwd_ret"]
    f["up"] = (f["fwd_excess"] > 0).astype("float64").where(f["fwd_excess"].notna())
    # std of r[t+1 .. t+VOL_HORIZON]
    fwd_vol = r.rolling(VOL_HORIZON).std().shift(-VOL_HORIZON) * ANN
    f["fwd_logvol"] = np.log(fwd_vol.replace(0, np.nan))
    f["close"] = c
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(prices, symbols=None, horizon=HORIZON):
    """Stack per-ticker features into a (Date, Ticker)-indexed frame.

    Rows keep NaN targets at the end of history (needed for live prediction);
    rows with any NaN feature are dropped.
    """
    close = prices["Close"]
    mkt = market_frame(close[MARKET], close[VIX], horizon)
    symbols = symbols or [s for s in close.columns if s not in (MARKET, VIX)]

    frames = []
    for s in symbols:
        if s not in close.columns or close[s].notna().sum() < 300:
            continue
        ohlcv = pd.DataFrame({k: prices[k][s] for k in ("Open", "High", "Low", "Close", "Volume")})
        f = ticker_features(ohlcv, mkt, horizon).dropna(subset=FEATURES)
        f["Ticker"] = s
        frames.append(f)
    if not frames:
        raise ValueError("No tickers had enough history to build features.")
    panel = pd.concat(frames)
    panel.index.name = "Date"
    return panel.set_index("Ticker", append=True).sort_index()


def single_ticker_panel(symbol, period="2y", horizon=HORIZON):
    """Features for one ticker (for live prediction in the app)."""
    prices = download([symbol], period=period, cache=False)
    return build_panel(prices, [symbol], horizon)
