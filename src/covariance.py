"""
Covariance estimators - how the risk matrix itself is built.

The raw sample covariance is noisy and can be near-singular, which matters
because everything downstream INVERTS or leans on it: risk-parity weights,
vol-targeting, the Balance min-variance blend, Monte Carlo. Two professional
estimators stabilize or sharpen it. All return an ANNUALIZED covariance
DataFrame with the same asset labels, so they are drop-in swappable.

Quant Deep Dive
---------------
1. **Sample** - the plain historical covariance (analytics.covariance_matrix).
   Baseline; unbiased but high-variance, unstable as #assets approaches #days.

2. **Ledoit-Wolf shrinkage** (Ledoit & Wolf 2004, "A well-conditioned
   estimator for large-dimensional covariance matrices", J. Multivariate
   Analysis - the scaled-identity variant, as implemented by scikit-learn):
        Σ̂ = δ·F + (1−δ)·S
   S is the messy sample matrix, F = (avg variance)·I the scaled-identity
   target, and δ ∈ [0,1] the closed-form optimal intensity. Pulls the noisy
   estimate toward a stable target, guaranteeing a well-conditioned,
   invertible matrix. (The constant-correlation target is the other LW 2004
   paper, "Honey, I Shrunk the Sample Covariance Matrix"; sklearn ships the
   identity-target estimator, and that is what runs here.)

3. **EWMA** (RiskMetrics, λ=0.94 daily):
        Σ_t = λ·Σ_{t−1} + (1−λ)·rₜrₜᵀ
   Recent days weigh exponentially more, so the matrix REACTS to a volatility
   spike instead of averaging it away over a long window. Note: it reacts to a
   regime change already underway - it does not foresee one.

Honest limit: none of these predict a crash. Shrinkage trades a little bias for
much less variance; EWMA trades stability for responsiveness. They make the risk
number steadier or timelier, not clairvoyant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISKMETRICS_LAMBDA = 0.94


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain annualized sample covariance (the baseline estimator)."""
    return returns.cov() * TRADING_DAYS


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Annualized Ledoit-Wolf shrunk covariance and the chosen intensity δ.

    Requires scikit-learn (already a project dependency). Falls back to the
    sample covariance with δ=0 if unavailable, rather than crashing.
    """
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        return sample_covariance(returns), 0.0
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                       index=returns.columns, columns=returns.columns)
    return cov, float(lw.shrinkage_)


def ewma_covariance(returns: pd.DataFrame,
                    lam: float = RISKMETRICS_LAMBDA) -> pd.DataFrame:
    """Annualized EWMA covariance (RiskMetrics), recent days weighted more.

    Zero-mean convention (RiskMetrics): the terminal matrix of the recursion
    Σ_t = λΣ_{t−1} + (1−λ)rₜrₜᵀ equals a weighted sum of daily outer products;
    weights are normalized to sum to 1 so a finite sample gives a proper
    weighted-average estimate.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    r = returns.values
    t = len(r)
    if t == 0:
        raise ValueError("no returns to estimate from")
    # newest row gets the largest weight (1-λ); oldest fades as λ^age
    w = (1.0 - lam) * lam ** np.arange(t - 1, -1, -1)
    w /= w.sum()
    cov = (r * w[:, None]).T @ r          # Σ w_t rₜrₜᵀ  (zero-mean)
    cov *= TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def estimate_covariance(returns: pd.DataFrame, method: str = "sample",
                        lam: float = RISKMETRICS_LAMBDA):
    """Dispatch to an estimator. Returns (cov_df, info) where info is a short
    human string describing what was used (for the UI caption)."""
    if method == "Ledoit-Wolf":
        cov, delta = ledoit_wolf_covariance(returns)
        return cov, (f"Ledoit-Wolf shrinkage (δ = {delta:.2f} toward scaled "
                     "identity, avg-variance x I)")
    if method == "EWMA":
        return ewma_covariance(returns, lam), (
            f"EWMA reactive lens, RiskMetrics λ = {lam:.2f} "
            "(~11-day half-life - flinches at today, forgets the calm quarter)")
    return sample_covariance(returns), "Sample covariance (equal weight, full window)"


if __name__ == "__main__":  # smoke test
    idx = pd.bdate_range("2022-01-01", periods=500)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (500, 3))
    base[-20:] *= 4  # recent volatility spike - EWMA should catch it
    df = pd.DataFrame(base, columns=["A", "B", "C"], index=idx)
    for m in ("sample", "Ledoit-Wolf", "EWMA"):
        cov, info = estimate_covariance(df, m)
        print(f"{m:12s} A vol={np.sqrt(cov.loc['A','A']):.3f}  | {info}")
