"""S&P 500 universe with point-in-time entry dates, long price history and
earnings dates. Everything is cached under improvements/data/.

Survivorship handling (partial): each stock only enters the universe on the
date it was added to the S&P 500 (Wikipedia "Date added"), so the backtest
never holds a company before it became a large cap. Companies that have
since been *removed* are still missing - Yahoo no longer serves most of them.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
START = "2000-01-01"
MARKET, VIX = "SPY", "^VIX"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _cache(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def sp500_constituents(refresh=False):
    """DataFrame[Symbol, Sector, added] of current members (Yahoo-style symbols)."""
    path = _cache("sp500.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=["added"])
    html = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    out = pd.DataFrame({
        "Symbol": t["Symbol"].str.replace(".", "-", regex=False),
        "Sector": t["GICS Sector"],
        "added": pd.to_datetime(t["Date added"], errors="coerce"),
    })
    out.to_csv(path, index=False)
    return out


def download_prices(symbols, start=START, chunk=100, refresh=False):
    """{field: DataFrame[date x ticker]} for Open/High/Low/Close/Volume, float32."""
    path = _cache(f"prices_{start}.pkl")
    if os.path.exists(path) and not refresh:
        print(f"Loading cached prices from {path}")
        return pd.read_pickle(path)
    symbols = list(dict.fromkeys(list(symbols) + [MARKET, VIX]))
    parts = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(symbols)}")
        raw = yf.download(batch, start=start, interval="1d", auto_adjust=True,
                          progress=False, threads=True)
        parts.append(raw)
    raw = pd.concat(parts, axis=1)
    prices = {f: raw[f].astype("float32") for f in ["Open", "High", "Low", "Close", "Volume"]}
    pd.to_pickle(prices, path)
    return prices


def _earnings_one(sym):
    try:
        e = yf.Ticker(sym).get_earnings_dates(limit=100)
        if e is None or e.empty:
            return sym, pd.DatetimeIndex([])
        idx = pd.DatetimeIndex(e.index)
        # announcements after the close move the *next* session
        after_close = idx.hour >= 12
        idx = idx.tz_localize(None).normalize()
        idx = idx + pd.to_timedelta(np.where(after_close, 1, 0), unit="D")
        return sym, idx.sort_values().unique()
    except Exception:
        return sym, pd.DatetimeIndex([])


def earnings_dates(symbols, refresh=False, workers=6):
    """{symbol: DatetimeIndex of earnings reaction days}. Missing -> empty index."""
    path = _cache("earnings.pkl")
    cached = {} if refresh or not os.path.exists(path) else pd.read_pickle(path)
    todo = [s for s in symbols if s not in cached]
    if todo:
        print(f"  fetching earnings dates for {len(todo)} tickers...")
        with ThreadPoolExecutor(workers) as ex:
            for sym, idx in ex.map(_earnings_one, todo):
                cached[sym] = idx
        pd.to_pickle(cached, path)
    got = sum(len(cached.get(s, [])) > 0 for s in symbols)
    print(f"  earnings dates available for {got}/{len(symbols)} tickers")
    return cached
