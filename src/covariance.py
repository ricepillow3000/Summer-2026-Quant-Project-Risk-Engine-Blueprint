"""
covariance estimators - how the risk matrix itself gets built.

the raw sample covariance is noisy and can be close to singular, which hurts
because everything downstream inverts or leans on it (risk parity, vol
targeting, monte carlo). two pro estimators steady it or sharpen it. all
return an ANNUALIZED cov DataFrame with the same labels, so they swap freely.

how the math works
------------------
1. **Sample** - plain historical covariance. unbiased but noisy, and shaky
   once #assets gets close to #days.

2. **Ledoit-Wolf shrinkage** (Ledoit & Wolf 2004, "A well-conditioned
   estimator for large-dimensional covariance matrices", J. Multivariate
   Analysis - the scaled-identity version scikit-learn ships):
        Σ̂ = δ·F + (1−δ)·S
   S = the messy sample matrix, F = (avg variance)·I, δ ∈ [0,1] the
   closed-form best intensity. pulls the noisy estimate toward a stable
   target, so the result is always well-conditioned and invertible. (the
   constant-correlation target is their other 2004 paper, "Honey, I Shrunk
   the Sample Covariance Matrix" - not what runs here.)

3. **EWMA** (RiskMetrics, λ=0.94 daily):
        Σ_t = λ·Σ_{t−1} + (1−λ)·rₜrₜᵀ
   recent days count exponentially more, so the matrix REACTS to a vol spike
   instead of averaging it away. it reacts to a regime change already
   happening - it doesn't see one coming.

honest limit: none of these predict a crash. shrinkage trades a bit of bias
for way less noise; EWMA trades stability for speed. steadier or quicker,
not psychic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252   # the engine's one trading-days-per-year; every module imports it

# the published RiskMetrics decay for DAILY data (J.P. Morgan/Reuters,
# RiskMetrics Technical Document, 4th ed., 1996). an industry convention we
# take as-is, not fitted here - cited so it doesn't look hand-picked
RISKMETRICS_LAMBDA = 0.94


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """plain annualized sample covariance (the baseline)."""
    return returns.cov() * TRADING_DAYS


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """annualized Ledoit-Wolf shrunk covariance plus the intensity δ it picked.
    no scikit-learn? falls back to the sample cov with δ=0 instead of crashing."""
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
    """annualized EWMA covariance (RiskMetrics), recent days weigh more.

    zero-mean, like RiskMetrics: the end state of Σ_t = λΣ_{t−1} + (1−λ)rₜrₜᵀ
    is just a weighted sum of daily outer products. weights get normalized to
    sum to 1 so a finite sample is a proper weighted average.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    r = returns.values
    t = len(r)
    if t == 0:
        raise ValueError("no returns to estimate from")
    # newest day gets (1-λ), older ones fade as λ^age
    w = (1.0 - lam) * lam ** np.arange(t - 1, -1, -1)
    w /= w.sum()
    cov = (r * w[:, None]).T @ r          # Σ w_t rₜrₜᵀ  (zero-mean)
    cov *= TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def estimate_covariance(returns: pd.DataFrame, method: str = "sample",
                        lam: float = RISKMETRICS_LAMBDA):
    """pick an estimator. returns (cov_df, info), info being the short line
    the UI caption shows."""
    if method == "Ledoit-Wolf":
        cov, delta = ledoit_wolf_covariance(returns)
        return cov, (f"Ledoit-Wolf shrinkage (δ = {delta:.2f} toward scaled "
                     "identity, avg-variance x I)")
    if method == "EWMA":
        return ewma_covariance(returns, lam), (
            f"EWMA reactive lens, RiskMetrics λ = {lam:.2f} "
            "(~11-day half-life: it flinches at today and forgets the calm quarter)")
    return sample_covariance(returns), "Sample covariance (equal weight, full window)"


if __name__ == "__main__":  # smoke test
    idx = pd.bdate_range("2022-01-01", periods=500)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, (500, 3))
    base[-20:] *= 4  # recent vol spike - EWMA should catch it
    df = pd.DataFrame(base, columns=["A", "B", "C"], index=idx)
    for m in ("sample", "Ledoit-Wolf", "EWMA"):
        cov, info = estimate_covariance(df, m)
        print(f"{m:12s} A vol={np.sqrt(cov.loc['A','A']):.3f}  | {info}")
