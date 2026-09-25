"""Walk-forward backtest of the pooled multi-ticker ForecastAgent.

Every ``--retrain-every`` trading days the models are refit on all history
available at that point (with a purge gap so no training label overlaps the
test period), then predict the next block out-of-sample. The stitched
out-of-sample predictions are scored three ways:

  1. signal quality   AUC / rank-IC of P(beat SPY) vs simple baselines
  2. volatility       forecast error vs the naive trailing-20d volatility
  3. trading          non-overlapping HORIZON-day rebalances, with costs

Usage:
    python backtest.py                    # 100 large caps, 6y, 5-day horizon
    python backtest.py --cost-bps 5 --quantile 0.2
    python backtest.py --save             # also fit on all data -> models/pooled/
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from agents.forecast_agent import ForecastAgent
from utils.panel import HORIZON, UNIVERSE, VOL_HORIZON, build_panel, download

REPORT_DIR = "reports"


# ----------------------------------------------------------------------------
# walk-forward
# ----------------------------------------------------------------------------
def walk_forward(panel, warmup=504, step=63, seed=0):
    dates = panel.index.get_level_values("Date").unique().sort_values()
    purge = max(HORIZON, VOL_HORIZON)
    date_lvl = panel.index.get_level_values("Date")
    out = []
    for i in range(warmup, len(dates), step):
        train_end = dates[i - purge - 1]
        test_dates = dates[i:i + step]
        train = panel[date_lvl <= train_end]
        test = panel[date_lvl.isin(test_dates)]
        t = time.time()
        agent = ForecastAgent(seed=seed).fit(train)
        out.append(agent.predict(test))
        print(f"  fold {test_dates[0].date()} -> {test_dates[-1].date()} | "
              f"train rows {len(train):,} | {time.time() - t:4.1f}s")
    preds = pd.concat(out)
    return panel.loc[preds.index].join(preds), agent


# ----------------------------------------------------------------------------
# signal / volatility quality
# ----------------------------------------------------------------------------
def rank_ic(df, signal, target="fwd_excess"):
    ic = df.groupby(level="Date").apply(
        lambda g: g[signal].rank().corr(g[target].rank()) if len(g) > 5 else np.nan)
    ic = ic.dropna()
    return float(ic.mean()), float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else np.nan


def signal_report(oos, signals):
    d = oos.dropna(subset=["up", "fwd_excess"])
    rows = {}
    for name, col in signals.items():
        ic, t = rank_ic(d, col)
        rows[name] = {"AUC": roc_auc_score(d["up"], d[col]), "Rank IC": ic, "IC t-stat": t}
    table = pd.DataFrame(rows).T

    p = d["p_up"]
    conf = {}
    for thr in (0.02, 0.05, 0.10):
        m = (p - 0.5).abs() >= thr
        acc = ((p[m] > 0.5) == (d.loc[m, "up"] > 0.5)).mean() if m.any() else np.nan
        conf[f"|p-0.5| >= {thr:.2f}"] = {"coverage %": m.mean() * 100, "accuracy %": acc * 100}
    return table, pd.DataFrame(conf).T, float(d["up"].mean())


def vol_report(oos):
    v = oos.dropna(subset=["fwd_logvol"])
    y = v["fwd_logvol"]
    rows = {}
    for name, col in {"Model": "pred_logvol", "Naive (trailing 20d)": "logvol_20"}.items():
        err = v[col] - y
        rows[name] = {
            "RMSE (log vol)": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "Corr": float(np.corrcoef(v[col], y)[0, 1]),
            "Median abs % err": float(np.median(np.abs(np.exp(err) - 1)) * 100),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# trading
# ----------------------------------------------------------------------------
def _select(ranked, held, n, buffer):
    """Top-n names of ``ranked`` (best first), keeping current holdings that
    are still inside the top ``n * buffer`` to cut turnover."""
    keep = [t for t in ranked.index[:int(n * buffer)] if t in held][:n]
    fill = [t for t in ranked.index if t not in keep][:n - len(keep)]
    return keep + fill


def portfolio_returns(oos, signal, quantile, cost_bps, mode, buffer=1.0):
    """Per-rebalance net returns. mode: 'long_short', 'long_only' or 'equal_weight'.

    ``buffer`` > 1 holds a name until it drops out of the top quantile*buffer.
    """
    d = oos.dropna(subset=["fwd_ret"])
    dates = d.index.get_level_values("Date").unique().sort_values()[::HORIZON]
    prev = pd.Series(dtype="float64")
    longs, shorts = set(), set()
    rets, turns = [], []
    for dt in dates:
        g = d.xs(dt, level="Date")
        n = max(1, int(len(g) * quantile))
        w = pd.Series(0.0, index=g.index)
        if mode == "equal_weight":
            w[:] = 1 / len(g)
        elif mode == "inverse_vol":
            # signal is a log-volatility forecast; weight each name by 1/vol
            inv = np.exp(-g[signal])
            w[:] = inv / inv.sum()
        else:
            ranked = g[signal].sort_values(ascending=False)
            longs = set(_select(ranked, longs, n, buffer))
            w[list(longs)] = 1 / n
            if mode == "long_short":
                shorts = set(_select(ranked[::-1], shorts, n, buffer))
                w[list(shorts)] = -1 / n
        turnover = w.subtract(prev, fill_value=0).abs().sum()
        gross = float((w * g["fwd_ret"]).sum())
        rets.append(gross - turnover * cost_bps / 1e4)
        turns.append(turnover)
        # weights drift with returns until the next rebalance
        prev = w * (1 + g["fwd_ret"])
        if mode != "long_short" and prev.sum() != 0:
            prev = prev / prev.sum()
    return pd.Series(rets, index=dates), float(np.mean(turns))


def spy_returns(oos):
    d = oos.dropna(subset=["fwd_ret"])
    dates = d.index.get_level_values("Date").unique().sort_values()[::HORIZON]
    first = d.groupby(level="Date").first().loc[dates]
    return first["fwd_ret"] - first["fwd_excess"]


def perf_stats(r, turnover=np.nan):
    ppy = 252 / HORIZON
    equity = (1 + r).cumprod()
    years = len(r) / ppy
    return {
        "Ann. return %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "Ann. vol %": r.std() * np.sqrt(ppy) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan,
        "Max DD %": ((equity / equity.cummax()) - 1).min() * 100,
        "Hit rate %": (r > 0).mean() * 100,
        "Turnover/reb": turnover,
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="6y")
    ap.add_argument("--quantile", type=float, default=0.2, help="fraction of names held long (and short)")
    ap.add_argument("--buffer", type=float, default=2.0,
                    help="keep a holding until it leaves the top quantile*buffer (turnover control)")
    ap.add_argument("--cost-bps", type=float, default=10, help="cost per unit turnover, basis points")
    ap.add_argument("--retrain-every", type=int, default=63)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save", action="store_true", help="fit on all data and save to models/pooled/")
    args = ap.parse_args()

    panel = build_panel(download(UNIVERSE, args.period, cache=not args.no_cache))
    n_tick = panel.index.get_level_values("Ticker").nunique()
    print(f"Panel: {len(panel):,} rows, {n_tick} tickers, "
          f"{panel.index.get_level_values('Date').min().date()} -> "
          f"{panel.index.get_level_values('Date').max().date()}")

    print("\nWalk-forward:")
    oos, last_agent = walk_forward(panel, step=args.retrain_every)
    oos["reversal"] = -oos["ret_5"]
    oos["momentum"] = oos["ret_120"]
    # average of each ticker's last HORIZON daily predictions (uses past only);
    # the raw daily signal is noisy and churns the book
    oos["p_up_smooth"] = oos.groupby(level="Ticker")["p_up"].transform(
        lambda s: s.rolling(HORIZON, min_periods=1).mean())
    oos_dates = oos.index.get_level_values("Date")
    print(f"Out-of-sample: {oos_dates.min().date()} -> {oos_dates.max().date()} ({len(oos):,} rows)")

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    signals = {"Model P(beat SPY)": "p_up", "Reversal (-5d ret)": "reversal", "Momentum (120d ret)": "momentum"}
    sig, conf, base_rate = signal_report(oos, signals)
    print(f"\n=== 1. {HORIZON}-day 'beat SPY' signal (base rate {base_rate:.1%}; AUC 0.5 / IC 0 = no skill)")
    print(sig)
    print("\nConfidence buckets (model):")
    print(conf)

    vol = vol_report(oos)
    print(f"\n=== 2. {VOL_HORIZON}-day volatility forecast")
    print(vol)

    print(f"\n=== 3. Trading, rebalance every {HORIZON} days, top/bottom {args.quantile:.0%} "
          f"(hold buffer {args.buffer:g}x), {args.cost_bps:g} bps per unit turnover")
    curves, stats = {}, {}
    for label, sig_col, mode, buf in [
        ("Model L/S (raw daily)", "p_up", "long_short", 1.0),
        ("Model L/S (smoothed)", "p_up_smooth", "long_short", args.buffer),
        ("Reversal L/S", "reversal", "long_short", args.buffer),
        ("Momentum L/S", "momentum", "long_short", args.buffer),
        ("Model long-only", "p_up_smooth", "long_only", args.buffer),
        ("Equal-weight universe", "p_up", "equal_weight", 1.0),
        ("Inverse-vol (trailing 20d)", "logvol_20", "inverse_vol", 1.0),
        ("Inverse-vol (model forecast)", "pred_logvol", "inverse_vol", 1.0),
    ]:
        r, turn = portfolio_returns(oos, sig_col, args.quantile, args.cost_bps, mode, buf)
        curves[label], stats[label] = r, perf_stats(r, turn)
    curves["SPY"] = spy_returns(oos)
    stats["SPY"] = perf_stats(curves["SPY"], 0.0)
    stats = pd.DataFrame(stats).T
    print(stats)

    # Robustness: the strategy settings were not optimised, so show the whole
    # grid rather than one flattering cell.
    grid = {}
    for q in (0.1, 0.2, 0.3):
        for c in (0, 5, 10):
            r, _ = portfolio_returns(oos, "p_up_smooth", q, c, "long_short", args.buffer)
            grid[(f"top/bottom {q:.0%}", f"{c:g} bps")] = perf_stats(r)["Sharpe"]
    grid = pd.Series(grid).unstack()
    print("\nModel L/S (smoothed) Sharpe by quantile x cost:")
    print(grid)

    print("\nTop features (last fold, direction model, by gain):")
    print(last_agent.feature_importance().to_string(float_format=lambda v: f"{v:.2f}"))

    # --- reports ----------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    equity = (1 + pd.DataFrame(curves).fillna(0)).cumprod()
    equity.to_csv(os.path.join(REPORT_DIR, "equity.csv"))
    oos.to_pickle(os.path.join(REPORT_DIR, "oos_predictions.pkl"))
    with open(os.path.join(REPORT_DIR, "backtest_summary.json"), "w") as fh:
        json.dump({"signal": sig.to_dict("index"), "confidence": conf.to_dict("index"),
                   "volatility": vol.to_dict("index"), "trading": stats.to_dict("index"),
                   "sharpe_grid": {f"{k[0]} | {k[1]}": v for k, v in grid.stack().items()},
                   "args": vars(args)}, fh, indent=2, default=float)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = equity.plot(figsize=(11, 5), title="Walk-forward equity (net of costs)")
        ax.set_ylabel("Growth of $1")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "equity.png"), dpi=110)
    except Exception as e:
        print(f"  ! plot failed: {e}")
    print(f"\nReports written to {REPORT_DIR}/ (equity.csv, equity.png, backtest_summary.json)")
    print("Caveat: universe is today's large caps -> survivorship bias flatters long-only results.")

    if args.save:
        agent = ForecastAgent().fit(panel)
        agent.save()
        print(f"Saved pooled model trained through {agent.meta['trained_through']} to models/pooled/")


if __name__ == "__main__":
    main()
