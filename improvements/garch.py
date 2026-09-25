"""Minimal GARCH(1,1) (zero mean, Gaussian QMLE) - the textbook volatility
baseline - without the `arch` dependency.

    s2[t] = omega + alpha * r[t-1]^2 + beta * s2[t-1]

The recursion is a linear filter in r^2, so scipy.signal.lfilter makes the
likelihood fast enough to refit ~500 stocks per walk-forward fold.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

SCALE = 100.0  # work in percent returns for numerical stability


def _variance_path(params, r2, s2_0):
    """Conditional variance s2[t] for t = 0..n (one step past the data)."""
    omega, alpha, beta = params
    # s2[t+1] - beta*s2[t] = omega + alpha*r2[t],  t = 0..n-1
    s2, _ = lfilter([1.0], [1.0, -beta], omega + alpha * r2, zi=np.array([beta * s2_0]))
    return np.concatenate(([s2_0], s2))


def _nll(theta, r2, s2_0):
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    s2 = _variance_path(theta, r2, s2_0)[:-1]
    return 0.5 * np.sum(np.log(s2) + r2 / s2)


def fit(returns, max_obs=2500):
    """Fit on log returns (fractions). Returns (omega, alpha, beta) in pct^2 units."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)][-max_obs:] * SCALE
    if len(r) < 250:
        return None
    r2 = r ** 2
    var = r2.mean()
    best = None
    for a, b in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95)):
        x0 = np.array([var * (1 - a - b), a, b])
        res = minimize(_nll, x0, args=(r2, var), method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return tuple(best.x)


def forecast_logvol(params, returns, horizon, ann=252):
    """Walk the fitted filter over ``returns`` (fixed params) and return, for
    each date t, the log annualised vol forecast for t+1..t+horizon."""
    omega, alpha, beta = params
    r = np.nan_to_num(np.asarray(returns, dtype="float64")) * SCALE
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    s2 = _variance_path(params, r ** 2, long_run)[1:]   # s2[t+1 | t]
    # mean of E[s2[t+k]] for k=1..h under GARCH(1,1) mean reversion
    k = np.arange(horizon)
    avg_decay = np.mean(persistence ** k)
    avg_var = long_run + (s2 - long_run) * avg_decay
    return np.log(np.sqrt(np.maximum(avg_var, 1e-12) * ann) / SCALE)
