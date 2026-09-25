"""Volatility features and targets for the (Date, Ticker) panel.

Daily variance proxy: Garman-Klass intraday range variance plus the squared
overnight gap - a much less noisy daily estimate than a squared close-to-close
return. HAR components are its day / week / month / quarter averages.

Target: log annualised realised volatility over the next H sessions,
sqrt(mean(r^2 over t+1..t+H) * 252) with close-to-close log returns r - the
quantity a risk model actually has to get right.
"""

import numpy as np
import pandas as pd

H = 20                    # forecast horizon, trading days
ANN = 252
EPS = 1e-10

FEATURES = [
    # HAR on range-based daily variance (log annualised vol)
    "har_d", "har_w", "har_m", "har_q",
    # close-to-close realised vol
    "cc_22", "cc_66", "cc_252",
    # shape of recent risk
    "down_share", "overnight_share", "vol_of_vol", "ret_5", "ret_22", "drawdown_252",
    # market / systematic
    "beta_66", "corr_66", "mkt_har_d", "mkt_har_w", "mkt_har_m", "log_vix", "vix_vs_rv", "vix_chg_5",
    # liquidity / size
    "log_dollar_vol", "volume_z",
    # earnings cycle (past dates only)
    "days_since_earn", "earn_react",
    # 1 for index/sector funds, 0 for single stocks
    "is_fund",
]
BASE_REQUIRED = [f for f in FEATURES if f not in ("days_since_earn", "earn_react")]
TARGETS = ["fwd_logvol", "fwd_ret"]


def _logvol(var):
    return np.log(np.sqrt(np.maximum(var, EPS) * ANN))


def daily_variance(o, h, l, c):
    """Garman-Klass + overnight daily variance proxy (log units)."""
    hl = np.log(h / l)
    co = np.log(c / o)
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    overnight = np.log(o / c.shift(1)) ** 2
    return (gk.clip(lower=0) + overnight).where(lambda s: s > 0)


def market_frame(spy, vix):
    """Market-level features on SPY's own trading days."""
    spy = spy.dropna(subset=["Close"])
    r = np.log(spy["Close"]).diff()
    dv = daily_variance(spy["Open"], spy["High"], spy["Low"], spy["Close"])
    vix = vix.reindex(spy.index).ffill()
    rv22 = _logvol(r.pow(2).rolling(22).mean())
    return pd.DataFrame({
        "mkt_r": r,
        "mkt_har_d": _logvol(dv),
        "mkt_har_w": _logvol(dv.rolling(5).mean()),
        "mkt_har_m": _logvol(dv.rolling(22).mean()),
        "log_vix": np.log(vix / 100),
        "vix_vs_rv": np.log(vix / 100) - rv22,     # variance-risk-premium proxy
        "vix_chg_5": np.log(vix / vix.shift(5)),
    })


def _event_sessions(index, events):
    """Positions in ``index`` of the first session on/after each event date."""
    ev_pos = np.unique(np.searchsorted(index.values, pd.DatetimeIndex(events).values))
    return ev_pos[ev_pos < len(index)]


def _days_since(index, ev_pos):
    """Trading sessions since the most recent event session (NaN before the first)."""
    pos = np.arange(len(index))
    last = np.searchsorted(ev_pos, pos, side="right") - 1
    out = np.where(last >= 0, pos - ev_pos[np.clip(last, 0, None)], np.nan)
    return pd.Series(out, index=index, dtype="float64")


def ticker_frame(ohlcv, mkt, earn_dates=None, horizon=H):
    df = ohlcv.dropna(subset=["Close", "Open", "High", "Low"]).join(mkt, how="inner")
    if len(df) < 300:
        return None
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    r = np.log(c).diff()
    r2 = r.pow(2)
    dv = daily_variance(o, h, l, c)

    f = pd.DataFrame(index=df.index)
    f["har_d"] = _logvol(dv)
    f["har_w"] = _logvol(dv.rolling(5).mean())
    f["har_m"] = _logvol(dv.rolling(22).mean())
    f["har_q"] = _logvol(dv.rolling(66).mean())
    f["cc_22"] = _logvol(r2.rolling(22).mean())
    f["cc_66"] = _logvol(r2.rolling(66).mean())
    f["cc_252"] = _logvol(r2.rolling(252, min_periods=200).mean())

    f["down_share"] = (r2 * (r < 0)).rolling(66).sum() / r2.rolling(66).sum()
    overnight = np.log(o / c.shift(1)) ** 2
    f["overnight_share"] = overnight.rolling(66).sum() / dv.rolling(66).sum()
    f["vol_of_vol"] = f["har_d"].rolling(22).std()
    f["ret_5"] = np.log(c / c.shift(5))
    f["ret_22"] = np.log(c / c.shift(22))
    f["drawdown_252"] = np.log(c / c.rolling(252, min_periods=200).max())

    m = df["mkt_r"]
    f["beta_66"] = r.rolling(66).cov(m) / m.rolling(66).var()
    f["corr_66"] = r.rolling(66).corr(m)
    for k in ("mkt_har_d", "mkt_har_w", "mkt_har_m", "log_vix", "vix_vs_rv", "vix_chg_5"):
        f[k] = df[k]

    f["log_dollar_vol"] = np.log((c * v).rolling(22).mean().replace(0, np.nan))
    lv = np.log(v.replace(0, np.nan))
    f["volume_z"] = (lv - lv.rolling(22).mean()) / lv.rolling(22).std()

    # earnings cycle - only announcements already made by date t
    ev_pos = _event_sessions(df.index, earn_dates) if earn_dates is not None and len(earn_dates) else []
    if len(ev_pos):
        # companies report ~every 63 sessions; a longer gap means missing data
        f["days_since_earn"] = _days_since(df.index, ev_pos).where(lambda s: s <= 100)
        # mean size of the last 4 earnings-day moves in units of normal daily
        # vol, known from the event session onward
        base = np.sqrt(r2.rolling(66).median()).shift(1)
        react = (r.abs() / base).iloc[ev_pos].rolling(4, min_periods=1).mean()
        f["earn_react"] = react.reindex(df.index).ffill()
    else:
        f["days_since_earn"] = np.nan
        f["earn_react"] = np.nan

    # targets --------------------------------------------------------------
    fwd_var = r2.rolling(horizon).mean().shift(-horizon)          # r[t+1..t+H]
    f["fwd_logvol"] = _logvol(fwd_var).where(fwd_var.notna())
    f["fwd_ret"] = np.log(c.shift(-horizon) / c)
    return f.replace([np.inf, -np.inf], np.nan)


def build_panel(prices, constituents, earnings, horizon=H, extra=("SPY",)):
    """Stack per-ticker frames. Rows before a stock joined the S&P 500 are
    dropped (point-in-time universe); ``extra`` tickers (index/sector funds)
    are kept for their whole history and flagged with ``is_fund``."""
    close = prices["Close"]
    ohlcv_of = lambda s: pd.DataFrame({k: prices[k][s] for k in ("Open", "High", "Low", "Close", "Volume")})
    spy = ohlcv_of("SPY")
    mkt = market_frame(spy, close["^VIX"])
    added = dict(zip(constituents["Symbol"], constituents["added"]))

    frames = []
    extra = list(extra)
    for s in [c for c in constituents["Symbol"] if c not in extra] + extra:
        if s not in close.columns or close[s].notna().sum() < 300:
            continue
        f = ticker_frame(ohlcv_of(s), mkt, earnings.get(s), horizon)
        if f is None:
            continue
        f["is_fund"] = 1.0 if s in extra else 0.0
        f = f.dropna(subset=BASE_REQUIRED)
        if s not in extra and pd.notna(added.get(s)):
            f = f[f.index >= added[s]]
        if f.empty:
            continue
        f = f.astype("float32")
        f["Ticker"] = s
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "Date"
    return panel.set_index("Ticker", append=True).sort_index()
