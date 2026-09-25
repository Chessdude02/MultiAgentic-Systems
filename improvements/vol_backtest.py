"""Walk-forward evaluation of 20-day volatility forecasts on the S&P 500 (v2).

    python -m improvements.vol_backtest
    python -m improvements.vol_backtest --retrain-every 126 --target-vol 0.10

Each fold refits every model on all history up to ``H`` sessions before the
fold (purge gap: no training target overlaps the test period) and forecasts
the next ``--retrain-every`` sessions. Training data = S&P 500 stocks
(point-in-time) + 13 index/sector ETFs. Competing forecasts:

    Trailing 22d    close-to-close realised vol of the last month (naive)
    HAR             pooled log-HAR regression on range-based day/week/month vol
    GARCH(1,1)      fitted per ticker
    XGBoost         pooled gradient boosting, squared error on log vol
    XGBoost-QLIKE   same features, trained on the QLIKE loss (penalises
                    under-predicting risk more than over-predicting it)
    XGBoost +BC     XGBoost minus each ticker's recent realised forecast error
                    (only errors already observable at forecast time)
    Combo           average of HAR, GARCH and XGBoost log forecasts
    Stacked         regime-aware blend of HAR/GARCH/XGBoost/XGBoost-QLIKE whose
                    weights vary with VIX, fitted on *earlier folds'*
                    out-of-sample forecasts

Return bands use normal quantiles, or (``calibrated``) the empirical quantiles
of past out-of-sample standardised returns.

v1 results (stocks-only training, no QLIKE/BC/stacking/calibration) are kept
in improvements/results/v1/. Outputs go to improvements/results/.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import norm

warnings.filterwarnings("ignore")

from improvements import garch
from improvements.universe import (DATA_DIR, ETFS, download_prices, earnings_dates,
                                   sp500_constituents, with_etfs)
from improvements.vol_features import FEATURES, H, build_panel

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
HAR_COLS = ["har_d", "har_w", "har_m"]
MODELS = ["Trailing 22d", "HAR", "GARCH(1,1)", "XGBoost", "XGBoost-QLIKE",
          "XGBoost +BC", "Combo", "Stacked"]
COL = {"Trailing 22d": "cc_22", "HAR": "p_har", "GARCH(1,1)": "p_garch",
       "XGBoost": "p_xgb", "XGBoost-QLIKE": "p_xgbq", "XGBoost +BC": "p_bc",
       "Combo": "p_combo", "Stacked": "p_stack"}
STACK_BASE = ["p_har", "p_garch", "p_xgb", "p_xgbq"]
CALIBRATE = ["XGBoost", "XGBoost-QLIKE", "Stacked"]
BC_HALFLIFE = 126
KEEP = ["cc_22", "log_vix", "is_fund", "fwd_logvol", "fwd_ret"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_panel(refresh=False):
    path = os.path.join(DATA_DIR, "vol_panel_v2.pkl")
    cons = sp500_constituents()
    prices = with_etfs(download_prices(cons["Symbol"].tolist()))
    if os.path.exists(path) and not refresh:
        print(f"Loading cached panel from {path}")
        return pd.read_pickle(path), prices
    earn = earnings_dates(cons["Symbol"].tolist())
    t = time.time()
    panel = build_panel(prices, cons, earn, extra=ETFS)
    print(f"Built panel in {time.time() - t:.0f}s")
    panel.to_pickle(path)
    return panel, prices


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def fit_har(train):
    X = np.column_stack([np.ones(len(train))] + [train[c].values for c in HAR_COLS])
    coef, *_ = np.linalg.lstsq(X, train["fwd_logvol"].values.astype("float64"), rcond=None)
    return coef


def predict_har(coef, df):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in HAR_COLS])
    return X @ coef


def qlike_objective(y, p):
    """XGBoost objective: QLIKE on variance, with p = predicted log vol.
    L = exp(2(y-p)) - 2(y-p) - 1. Hessian floored for stability when p >> y."""
    e = np.exp(np.clip(2 * (y - p), -8, 8))
    return 2 * (1 - e), 4 * np.maximum(e, 0.05)


def fit_xgb(train, loss="mse", seed=0):
    extra = {}
    if loss == "qlike":
        extra = {"objective": qlike_objective, "base_score": float(train["fwd_logvol"].mean())}
    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=200,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        tree_method="hist", n_jobs=os.cpu_count(), random_state=seed, **extra)
    model.fit(train[FEATURES], train["fwd_logvol"])
    return model


def _garch_job(sym, r, train_end, test_dates):
    p = garch.fit(r[r.index <= train_end].values)
    if p is None:
        return sym, None
    fc = pd.Series(garch.forecast_logvol(p, r.values, H), index=r.index)
    return sym, fc.reindex(test_dates)


def garch_fold(returns, tickers, train_end, test_dates, n_jobs):
    jobs = (delayed(_garch_job)(s, returns[s], train_end, test_dates)
            for s in tickers if s in returns)
    out = {sym: s for sym, s in Parallel(n_jobs=n_jobs)(jobs) if s is not None}
    if not out:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(out)
    df.index.name = "Date"
    s = df.stack()
    s.index.names = ["Date", "Ticker"]
    return s


def walk_forward(panel, prices, warmup_years, step, n_jobs):
    dates = panel.index.get_level_values("Date").unique().sort_values()
    d_lvl = panel.index.get_level_values("Date")
    close = prices["Close"]
    returns = {s: np.log(close[s].dropna()).diff().dropna()
               for s in panel.index.get_level_values("Ticker").unique() if s in close.columns}

    # training uses every 5th session: 20-day targets overlap heavily anyway
    train_dates = set(dates[::5])
    out, folds = [], []
    for i in range(warmup_years * 252, len(dates), step):
        t0 = time.time()
        train_end = dates[i - H - 1]
        test_dates = dates[i:i + step]
        train = panel[(d_lvl <= train_end) & d_lvl.isin(train_dates)].dropna(subset=["fwd_logvol"])
        test = panel[d_lvl.isin(test_dates)]

        har = fit_har(train)
        model = fit_xgb(train, "mse")
        model_q = fit_xgb(train, "qlike")
        pred = test[KEEP].astype("float64")
        pred["p_har"] = predict_har(har, test)
        pred["p_xgb"] = model.predict(test[FEATURES])
        pred["p_xgbq"] = model_q.predict(test[FEATURES])
        g = garch_fold(returns, test.index.get_level_values("Ticker").unique(),
                       train_end, test_dates, n_jobs)
        pred["p_garch"] = g.reindex(test.index).values
        out.append(pred)
        folds.append((test_dates[0], test_dates[-1], train_end))
        print(f"  fold {test_dates[0].date()} -> {test_dates[-1].date()} | "
              f"train {len(train):,} rows | test {len(test):,} | {time.time() - t0:.0f}s")
    oos = pd.concat(out)
    # GARCH can fail for a handful of short histories: fall back to HAR there
    oos["p_garch"] = oos["p_garch"].fillna(oos["p_har"])
    oos["p_combo"] = oos[["p_har", "p_garch", "p_xgb"]].mean(axis=1)
    return oos, model, folds


# ----------------------------------------------------------------------------
# post-hoc layers, all strictly using information available at forecast time
# ----------------------------------------------------------------------------
def bias_correct(oos, col="p_xgb"):
    """Subtract each ticker's EWMA of past forecast errors. The error of a
    forecast made at s is only known at s+H, hence the shift."""
    o = oos.sort_index(level=["Ticker", "Date"])
    err = (o[col] - o["fwd_logvol"]).groupby(level="Ticker").shift(H)
    corr = err.groupby(level="Ticker").transform(
        lambda s: s.ewm(halflife=BC_HALFLIFE, min_periods=20).mean())
    return (o[col] - corr.fillna(0)).reindex(oos.index)


def _stack_design(df, mu, sd):
    z = ((df["log_vix"] - mu) / sd).values[:, None]
    base = df[STACK_BASE].values
    return np.column_stack([np.ones(len(df)), base, base * z, z])


def stack(oos, folds):
    """Regime-aware linear blend refit per fold on earlier folds' OOS forecasts
    whose targets were already realised (date <= train_end)."""
    d = oos.index.get_level_values("Date")
    every5 = set(d.unique().sort_values()[::5])
    p = oos["p_combo"].copy()
    weights = None
    for start, end, train_end in folds:
        tr = oos[(d <= train_end) & d.isin(every5)].dropna(subset=["fwd_logvol"])
        te_mask = (d >= start) & (d <= end)
        if len(tr) < 20_000:          # first fold(s): no history yet -> Combo
            continue
        mu, sd = tr["log_vix"].mean(), tr["log_vix"].std()
        coef, *_ = np.linalg.lstsq(_stack_design(tr, mu, sd), tr["fwd_logvol"].values, rcond=None)
        p[te_mask] = _stack_design(oos[te_mask], mu, sd) @ coef
        weights = coef
    return p, weights


def calibrated_bands(oos, folds, col):
    """Band multipliers = empirical 80/95% quantiles of |ret| / (sigma*sqrt(H/252))
    over past OOS rows with known outcomes; normal quantiles before that."""
    d = oos.index.get_level_values("Date")
    k = pd.DataFrame({"k80": norm.ppf(0.9), "k95": norm.ppf(0.975)}, index=oos.index)
    z = oos["fwd_ret"].abs() / (np.exp(oos[col]) * np.sqrt(H / 252))
    last = None
    for start, end, train_end in folds:
        past = z[(d <= train_end)].dropna()
        if len(past) < 20_000:
            continue
        m = (d >= start) & (d <= end)
        last = (past.quantile(0.8), past.quantile(0.95))
        k.loc[m, "k80"], k.loc[m, "k95"] = last
    return k, last


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def qlike(true_logvol, pred_logvol):
    ratio = np.exp(2 * (true_logvol - pred_logvol))
    return ratio - np.log(ratio) - 1


def newey_west_t(d, lags):
    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    n, u = len(d), d - d.mean()
    var = u @ u / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    return d.mean() / np.sqrt(var / n)


def accuracy_table(ev):
    y = ev["fwd_logvol"]
    rows = {}
    for m in MODELS:
        e = ev[COL[m]] - y
        rows[m] = {
            "RMSE (log vol)": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "QLIKE": qlike(y, ev[COL[m]]).mean(),
            "Bias (log)": e.mean(),
            "Median abs % err": np.median(np.abs(np.exp(e) - 1)) * 100,
            "Under-pred >30% %": (e < np.log(0.7)).mean() * 100,
        }
    return pd.DataFrame(rows).T


def dm_table(ev, challenger):
    """Diebold-Mariano: daily cross-sectional mean loss differential,
    Newey-West t-stat with H lags. Positive t => challenger is better."""
    y = ev["fwd_logvol"]
    rows = {}
    for m in MODELS:
        if m == challenger:
            continue
        for name, loss in (("MSE", lambda p: (p - y) ** 2), ("QLIKE", lambda p: qlike(y, p))):
            diff = (loss(ev[COL[m]]) - loss(ev[COL[challenger]])).groupby(level="Date").mean()
            rows.setdefault(m, {})[f"t ({name})"] = newey_west_t(diff.values, H)
    return pd.DataFrame(rows).T


def breakdown(ev, key, loss="rmse"):
    y = ev["fwd_logvol"]
    out = {}
    for m in MODELS:
        if loss == "rmse":
            out[m] = np.sqrt(((ev[COL[m]] - y) ** 2).groupby(key).mean())
        else:
            out[m] = qlike(y, ev[COL[m]]).groupby(key).mean()
    return pd.DataFrame(out)


def coverage_table(ev, bands, regime):
    rows = {}
    for m in MODELS:
        sig = np.exp(ev[COL[m]]) * np.sqrt(H / 252)
        z = ev["fwd_ret"].abs() / sig
        rows[m] = {"80% band": (z <= norm.ppf(0.9)).mean() * 100,
                   "95% band": (z <= norm.ppf(0.975)).mean() * 100}
        if m in bands:
            k = bands[m].reindex(ev.index)
            rows[f"{m} (calibrated)"] = {"80% band": (z <= k["k80"]).mean() * 100,
                                         "95% band": (z <= k["k95"]).mean() * 100}
    by_regime = {}
    sig = np.exp(ev[COL["Stacked"]]) * np.sqrt(H / 252)
    z = ev["fwd_ret"].abs() / sig
    k = bands["Stacked"].reindex(ev.index)
    for name, hit in (("normal 95%", z <= norm.ppf(0.975)), ("calibrated 95%", z <= k["k95"])):
        by_regime[f"Stacked {name}"] = hit.groupby(regime.values).mean() * 100
    return pd.DataFrame(rows).T, pd.DataFrame(by_regime).T


def perf(r, periods_per_year):
    eq = (1 + r).cumprod()
    yrs = len(r) / periods_per_year
    return {
        "Ann. return %": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
        "Ann. vol %": r.std() * np.sqrt(periods_per_year) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(periods_per_year),
        "Max DD %": (eq / eq.cummax() - 1).min() * 100,
    }


def vol_targeting(ev_spy, prices, target, max_lev, cost_bps, step=5):
    """Scale SPY exposure to target/forecast vol, rebalanced every ``step`` days."""
    close = prices["Close"]["SPY"].dropna()
    fwd = (close.shift(-step) / close - 1).reindex(ev_spy.index.get_level_values("Date"))
    spy = ev_spy.reset_index(level="Ticker", drop=True)
    reb = spy.index[::step]
    fwd = fwd.loc[reb].dropna()
    reb = fwd.index
    curves, extra = {"Buy & hold SPY": fwd}, {}
    for m in MODELS:
        e = np.minimum(target / np.exp(spy.loc[reb, COL[m]]), max_lev)
        cost = e.diff().abs().fillna(e.iloc[0]) * cost_bps / 1e4
        key = f"Target {target:.0%} via {m}"
        curves[key] = e * fwd - cost
        # > 0 means the forecast over-predicts index risk and under-invests
        extra[key] = {"Avg exposure": e.mean(),
                      "SPY forecast bias": (spy[COL[m]] - spy["fwd_logvol"]).mean()}
    ppy = 252 / step
    table = {}
    for k, r in curves.items():
        row = perf(r, ppy)
        roll = r.rolling(13).std() * np.sqrt(ppy) * 100          # ~quarterly realised vol
        row["Worst qtr vol %"] = roll.max()
        row["Qtr vol dispersion"] = roll.std()
        row.update(extra.get(k, {"Avg exposure": 1.0, "SPY forecast bias": np.nan}))
        table[k] = row
    return pd.DataFrame(table).T, pd.DataFrame(curves)


def inverse_vol_portfolios(ev_stocks, cost_bps):
    """Monthly (H-day) rebalance: equal weight vs weights proportional to 1/forecast vol."""
    d = ev_stocks.dropna(subset=["fwd_ret"])
    dates = d.index.get_level_values("Date").unique().sort_values()[::H]
    res = {}
    for name, col in [("Equal weight", None)] + [(f"Inverse-vol ({m})", COL[m]) for m in MODELS]:
        prev, rets = pd.Series(dtype="float64"), []
        for dt in dates:
            g = d.xs(dt, level="Date")
            w = pd.Series(1.0, index=g.index) if col is None else np.exp(-g[col])
            w = w / w.sum()
            turn = w.subtract(prev, fill_value=0).abs().sum()
            gross = np.exp(g["fwd_ret"]) - 1
            rets.append(float((w * gross).sum()) - turn * cost_bps / 1e4)
            prev = w * (1 + gross)
            prev = prev / prev.sum()
        res[name] = perf(pd.Series(rets, index=dates), 252 / H)
    return pd.DataFrame(res).T


# ----------------------------------------------------------------------------
def md(df, fmt="{:,.3f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(
            "n/a" if pd.isna(v) else fmt.format(v) for v in row.values) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain-every", type=int, default=252)
    ap.add_argument("--warmup-years", type=int, default=4)
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--max-leverage", type=float, default=1.5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached feature panel")
    args = ap.parse_args()

    panel, prices = load_panel(args.refresh)
    tick = panel.index.get_level_values("Ticker")
    dts = panel.index.get_level_values("Date")
    print(f"Panel: {len(panel):,} rows, {tick.nunique()} tickers ({int(panel.groupby(level='Ticker')['is_fund'].first().sum())} funds), "
          f"{dts.min().date()} -> {dts.max().date()}")

    print("\nWalk-forward:")
    oos, last_model, folds = walk_forward(panel, prices, args.warmup_years, args.retrain_every, args.jobs)
    oos["p_bc"] = bias_correct(oos)
    oos["p_stack"], stack_w = stack(oos, folds)
    bands, band_k = {}, {}
    for m in CALIBRATE:
        bands[m], band_k[m] = calibrated_bands(oos, folds, COL[m])

    ev = oos.dropna(subset=["fwd_logvol"])
    ev_stocks = ev[ev["is_fund"] == 0]
    ev_funds = ev[ev["is_fund"] == 1]
    ev_spy = oos[oos.index.get_level_values("Ticker") == "SPY"]
    d = ev_stocks.index.get_level_values("Date")
    span = f"{d.min().date()} -> {d.max().date()}"
    print(f"Out-of-sample: {span}, {len(ev_stocks):,} stock-days, {len(ev_funds):,} fund-days")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    regime = pd.cut(np.exp(ev_stocks["log_vix"]) * 100, [0, 15, 20, 30, 200],
                    labels=["VIX<15", "15-20", "20-30", "VIX>30"])
    cov, cov_regime = coverage_table(ev_stocks, bands, regime)
    vt, vt_curves = vol_targeting(ev_spy, prices, args.target_vol, args.max_leverage, cost_bps=2)
    stack_names = (["const"] + [f"w_{c[2:]}" for c in STACK_BASE]
                   + [f"w_{c[2:]} x VIX" for c in STACK_BASE] + ["VIX"])
    sections = [
        ("1. Forecast accuracy - stocks", accuracy_table(ev_stocks)),
        ("1b. Forecast accuracy - index/sector ETFs", accuracy_table(ev_funds)),
        ("2a. Diebold-Mariano t-stats: Stacked vs each (t > 2 => Stacked significantly better)",
         dm_table(ev_stocks, "Stacked")),
        ("2b. Diebold-Mariano t-stats: XGBoost vs each", dm_table(ev_stocks, "XGBoost")),
        ("3a. RMSE by year - stocks", breakdown(ev_stocks, d.year)),
        ("3b. RMSE by VIX regime - stocks", breakdown(ev_stocks, regime.values)),
        ("3c. QLIKE by VIX regime - stocks", breakdown(ev_stocks, regime.values, "qlike")),
        ("4a. Return-band coverage - stocks (target 80 / 95)", cov),
        ("4b. Stacked 95% band coverage by VIX regime", cov_regime),
        ("4c. Calibrated band multipliers, last fold (normal: 1.282 / 1.960)",
         pd.DataFrame(band_k, index=["k80", "k95"]).T),
        (f"5. SPY vol targeting ({args.target_vol:.0%} target, max {args.max_leverage:g}x, 2 bps)", vt),
        ("6. Monthly inverse-vol stock portfolios (10 bps)", inverse_vol_portfolios(ev_stocks, 10)),
        ("7. Stacked blend coefficients, last fold (VIX standardised)",
         pd.DataFrame({"coef": stack_w}, index=stack_names)),
    ]
    for title, table in sections:
        print(f"\n=== {title}")
        print(table)
    imp = pd.Series(last_model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print("\nTop XGBoost features (last fold, gain):")
    print(imp.head(12).to_string(float_format=lambda v: f"{v:.1f}"))

    # --- write results ----------------------------------------------------
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# Volatility forecasting - walk-forward results (v2)\n\n")
        fh.write(f"Out-of-sample {span}; {len(ev_stocks):,} stock-days on "
                 f"{ev_stocks.index.get_level_values('Ticker').nunique()} S&P 500 stocks "
                 f"(point-in-time entry) plus {len(ev_funds):,} ETF-days; horizon {H} "
                 f"sessions; retrain every {args.retrain_every} sessions.\n\n")
        for title, table in sections:
            fh.write(f"## {title}\n\n{md(table)}\n\n")
        fh.write("## Top XGBoost features (last fold, gain)\n\n")
        fh.write(md(imp.head(12).to_frame("gain"), "{:,.1f}") + "\n")
    eq = (1 + vt_curves.fillna(0)).cumprod()
    eq.to_csv(os.path.join(RESULTS, "vol_targeting_equity.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["Buy & hold SPY"] + [c for c in eq.columns if any(k in c for k in ("Trailing", "Stacked", "+BC", "via XGBoost"))]
        ax = eq[cols].plot(figsize=(11, 5), logy=True,
                           title="SPY volatility targeting (out-of-sample, net of costs)")
        ax.set_ylabel("Growth of $1 (log scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "vol_targeting.png"), dpi=110)
    except Exception as e:
        print(f"  ! plot failed: {e}")
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
