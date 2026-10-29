import numpy as np
import pandas as pd

from risk import DAYS


# how much each stock wiggles, alone and together, scaled to a year
def cov(returns):
    return returns.cov() * DAYS


# the whole basket's yearly wiggle
def vol(w, cov):
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(w @ cov.values @ w))


# who's actually bringing the risk. 10% of your money can be 30% of your risk
def risk_share(w, cov):
    w = np.asarray(w, dtype=float)
    s = cov.values
    v = np.sqrt(w @ s @ w)
    return pd.Series(w * (s @ w) / v / v, index=cov.index)


# risk parity: tweak weights till every stock brings the SAME risk. fair share for all
def risk_parity(cov, iters=10_000, tol=1e-12):
    s = cov.values
    n = s.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        before = w.copy()
        for i in range(n):
            c = s[i] @ w - s[i, i] * w[i]
            w[i] = (-c + np.sqrt(c * c + 4.0 * s[i, i] / n)) / (2.0 * s[i, i])
        if np.max(np.abs(w - before)) < tol * max(1.0, np.max(np.abs(w))):
            break  # stopped moving = done
    return w / w.sum()


# turn the whole basket up or down to hit a 10% yearly wiggle
def vol_target(w, cov, target=0.10):
    v = vol(w, cov)
    lev = target / v if v > 0 else 1.0
    return {"leverage": lev, "weights": np.asarray(w, dtype=float) * lev}


# days to sell out. you only trade 20% of a stock's normal daily dollar volume so the price doesn't tank
def days_to_exit(w, adv, book=1_000_000, part=0.20):
    pos = np.abs(np.asarray(w, dtype=float)) * book
    cap = part * adv.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.where(cap > 0, pos / cap, np.inf), index=adv.index)  # no volume data = inf, never faked
