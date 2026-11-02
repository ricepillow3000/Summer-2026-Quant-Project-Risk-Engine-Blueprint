import numpy as np
import pandas as pd
from scipy import special

DAYS = 252  # trading days in a year


# the whole basket's daily % move = each stock's move x how much you hold
def book(returns, weights):
    return returns @ weights


# VaR: the "bad day" line. 95% of days you lose less than this
def var(r, conf=0.95):
    return float(-np.percentile(r, (1 - conf) * 100))


# CVaR: when you DO cross that line, how bad it gets on average. the "how cooked are we" number
def cvar(r, conf=0.95):
    r = pd.Series(r).dropna()
    cut = np.percentile(r, (1 - conf) * 100)
    return float(-r[r <= cut].mean())


# grade the VaR line. each day's line uses only the 250 days BEFORE it (no peeking)
def backtest(r, conf=0.95, window=250):
    q = 1 - conf
    win = window if len(r) > window else max(60, len(r) // 2)
    line = r.rolling(win).quantile(q).shift(1)
    hit = (r < line)[line.notna()]
    n, x = len(hit), int(hit.sum())
    if n == 0:
        return {"breaches": 0, "n": 0, "lr": None, "passed": None}  # too few days to grade, so no fake grade
    lr = -2 * ((n - x) * np.log(1 - q) + x * np.log(q)
               - special.xlogy(n - x, 1 - x / n) - special.xlogy(x, x / n))
    return {"breaches": x, "n": n, "lr": round(float(lr), 2), "passed": bool(lr <= 3.841)}  # 3.841 = pass mark


# how long bad stretches stick together. big days come in bunches, not one at a time
def block_len(r, cap=40):
    x = np.abs(np.asarray(r, dtype=float))
    x = x[np.isfinite(x)]
    if x.size < 30 or x.std(ddof=1) == 0:
        return 2
    x = x - x.mean()
    tau = 1.0
    for lag in range(1, min(100, x.size // 2)):
        rho = float(x[lag:] @ x[:-lag]) / float(x @ x)
        if rho <= 0:
            break
        tau += 2 * rho
    return int(np.clip(round(tau), 2, cap))


# the scoreboard both simulators share
def summary(paths, conf, **extra):
    total = paths[:, -1] - 1  # where each fake year ends up
    cut = np.percentile(total, (1 - conf) * 100)
    return {"var": float(-cut), "cvar": float(-total[total <= cut].mean()),
            "median": float(np.median(total)), "prob_loss": float((total < 0).mean()), **extra}


# play 10,000 fake next-years by stitching real chunks of the past. crash weeks stay in one piece
def monte_carlo(returns, weights, paths=10_000, days=DAYS, conf=0.95):
    r = book(returns, weights).values
    rng = np.random.default_rng(42)  # same seed = same answer every run
    block = block_len(r) if r.size >= 30 else 1
    if block > 1:
        idx = np.empty((paths, days), dtype=np.int64)
        idx[:, 0] = rng.integers(0, r.size, size=paths)
        for t in range(1, days):
            hop = rng.random(paths) < 1.0 / block  # coin flip: keep walking this chunk or hop to a new spot
            idx[:, t] = np.where(hop, rng.integers(0, r.size, size=paths), (idx[:, t - 1] + 1) % r.size)
        sim = r[idx]
    else:
        sim = rng.choice(r, size=(paths, days), replace=True)  # tiny history? just pick random days
    return summary(np.cumprod(1 + sim, axis=1), conf, block_days=block)


# split history into chill days and jump days (moves 3x bigger than normal)
def calibrate_jumps(r, k=3.0):
    lr = np.log1p(np.asarray(r, dtype=float))
    m, s = lr.mean(), lr.std(ddof=1)
    if s == 0:
        return {"mu_d": m, "sigma_d": 0.0, "lam": 0.0, "mu_j": 0.0, "sigma_j": 0.0}
    jump = np.abs(lr - m) > k * s
    calm, big = lr[~jump], lr[jump]
    mu_d = calm.mean() if calm.size else m
    return {"mu_d": mu_d, "sigma_d": calm.std(ddof=1) if calm.size > 1 else s, "lam": jump.mean(),
            "mu_j": big.mean() - mu_d if big.size else 0.0,
            "sigma_j": big.std(ddof=1) if big.size > 1 else 0.0}


# Merton model: normal wiggles + surprise jumps. can dream up crashes worse than any we've seen
def jump_mc(returns, weights, paths=10_000, days=DAYS, conf=0.95):
    p = calibrate_jumps(book(returns, weights).values)
    rng = np.random.default_rng(42)
    shape = (paths, days)
    wiggle = p["mu_d"] + p["sigma_d"] * rng.standard_normal(shape)
    n = rng.poisson(p["lam"], size=shape)  # jumps per day (usually zero)
    jumps = n * p["mu_j"] + p["sigma_j"] * np.sqrt(n) * rng.standard_normal(shape)
    return summary(np.exp(np.cumsum(wiggle + jumps, axis=1)), conf, jumps=p)
