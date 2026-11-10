"""
eigenrisk - eigendecomposition / pca on top of the covariance estimators.

a covariance matrix is a tangle: N assets -> N(N-1)/2 pairs, none of them
independent. eigendecomposition untangles it into risk factors that don't
overlap (orthogonal).

how the math works
------------------
picture returns as points on a rubber sheet the market stretches:
- eigenvectors = the straight directions it stretches along (risk pathways).
  the first is almost always the "market wave" (everything moves together);
  later ones are groups rotating against each other.
- eigenvalues = how hard it stretches along each one (the variance that
  factor carries). sorted big to small, they rank the main risk drivers.

honest label: these factors are STATISTICAL. calling PC2 "growth vs value"
is a human's story; the math only promises orthogonal + variance-ranked.

why this exists: optimizers invert the covariance, which turns every
eigenvalue into 1/λ - so the tiniest noise eigenvalues become the biggest
weights. two fixes:
1. Ledoit-Wolf shrinkage (src/covariance.py) pulls the matrix toward a
   stable target first.
2. eigenvalue clipping (Marcenko-Pastur style): anything under the pure-noise
   ceiling λ₊ = σ̄²(1+√(N/T))² gets averaged flat. total variance (the trace)
   stays the same, the 1/λ blow-up goes away. MP is an N,T -> ∞ result, so
   at 6-13 assets it's a sensible noise floor, not an exact law.

guardrails
----------
- condition number κ = λmax/λmin, checked BEFORE inverting; κ ≳ 1e8 = fragile.
- sign fix: eigh can hand back v or −v at random. we force each vector's
  biggest |entry| positive so a factor hedge never flips between reruns.
- singular matrix (two assets 100% correlated -> a zero eigenvalue) falls
  back to the Moore-Penrose pseudo-inverse instead of crashing.

plain np.linalg.eigh (symmetric solver), no sklearn wrapper, so every step
is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.covariance import TRADING_DAYS

# κ above this = too fragile to invert, use pinv. float64 has ~16 digits;
# 1e8 eats half and leaves half for the math on top.
CONDITION_LIMIT = 1e8


# ---------------------------------------------------------------- guards

def condition_number(cov: pd.DataFrame) -> float:
    """λmax/λmin (∞ if λmin ≤ 0). huge κ = 1/λmin will blow up noise and
    float error, so shrink, clip or pinv before inverting."""
    values = np.linalg.eigvalsh(cov.values)
    lam_min, lam_max = float(values[0]), float(values[-1])
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def align_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    """make each eigenvector's biggest |entry| positive (the v vs −v fix).
    otherwise the same book could report a flipped hedge on the next run."""
    anchor = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[anchor, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs


# ------------------------------------------------------------- denoising

def marcenko_pastur_bounds(n_assets: int, n_obs: int,
                           sigma2: float = 1.0) -> tuple[float, float]:
    """Marcenko-Pastur noise band λ± = σ²(1±√(N/T))².
    a pure-noise matrix's eigenvalues land in here (asymptotically), so under
    λ₊ = can't tell it from noise. at 6-13 assets that's weak, so it's drawn
    as a heuristic reference line, never used as a hard test."""
    if n_obs <= 0 or n_assets <= 0:
        raise ValueError("n_assets and n_obs must be positive")
    q = n_assets / n_obs
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def clip_eigenvalues(cov: pd.DataFrame, n_obs: int) -> tuple[pd.DataFrame, int]:
    """Marcenko-Pastur-style clipping -> (cleaned, n_clipped).

    eigenvalues under λ₊ = σ̄²(1+√(N/T))² get replaced by their own average
    (not zero), so the trace - total variance - is untouched and only the
    split across noise directions flattens. that kills the 1/λ blow-up.

    σ̄² = mean of every eigenvalue except the top one (assumes one real
    factor; a second real factor leaks into the noise average). MP really
    wants a correlation matrix (one common variance); on a covariance it's a
    standard adaptation, which is why it's only a heuristic. n_obs is T.

    if EVERY eigenvalue is under the ceiling there's no signal/noise split
    to act on, so the matrix comes back as-is and n_clipped = 0.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    values, vectors = np.linalg.eigh(cov.values)   # ascending
    n = len(values)
    # top eigenvalue = signal, the rest set the noise level
    sigma2 = values[:-1].mean() if n > 1 else values.mean()
    _, lam_plus = marcenko_pastur_bounds(n, n_obs, sigma2)
    noise = values < lam_plus
    n_clipped = int(noise.sum())
    if 0 < n_clipped < n:
        cleaned = values.copy()
        cleaned[noise] = values[noise].mean()      # flatten, keep the trace
    else:
        cleaned = values                            # none clipped, or all (no-op)
        n_clipped = 0
    rebuilt = (vectors * cleaned) @ vectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0           # exact symmetry
    return (pd.DataFrame(rebuilt, index=cov.index, columns=cov.columns),
            n_clipped)


# ---------------------------------------------------------- decomposition

def eigen_factors(cov: pd.DataFrame) -> dict:
    """full eigendecomposition with the guardrails on. returns:
      eigenvalues        - descending
      eigenvectors       - df (assets x PC1..PCn), signs fixed
      variance_explained - % of total variance per factor
      loadings           - eigenvectors x √λ: how much each asset rides each
                           factor, in return units (the heatmap)
      condition_number   - λmax/λmin of the input
    """
    values, vectors = np.linalg.eigh(cov.values)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    vectors = align_eigenvector_signs(vectors)

    total = values.sum()
    var_explained = values / total * 100.0 if total > 0 else np.zeros_like(values)
    cols = [f"PC{i + 1}" for i in range(len(values))]
    loadings = vectors * np.sqrt(np.clip(values, 0.0, None))
    return {
        "eigenvalues": values,
        "eigenvectors": pd.DataFrame(vectors, index=cov.index, columns=cols),
        "variance_explained": var_explained,
        "loadings": pd.DataFrame(loadings, index=cov.index, columns=cols),
        "condition_number": condition_number(cov),
    }


def project_returns(returns: pd.DataFrame, eigenvectors: pd.DataFrame,
                    k: int = 3) -> pd.DataFrame:
    """returns -> top-k factor time series: F = (R − R̄) · V[:, :k].
    PC1's column is what an eigen-hedge would try to neutralize."""
    k = max(1, min(k, eigenvectors.shape[1]))
    demeaned = returns[eigenvectors.index] - returns[eigenvectors.index].mean()
    projected = demeaned.values @ eigenvectors.values[:, :k]
    return pd.DataFrame(projected, index=returns.index,
                        columns=list(eigenvectors.columns[:k]))


def safe_inverse(cov: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """invert, or pinv if κ > CONDITION_LIMIT (or inv fails) -> (inverse, used_pinv).
    pinv zeroes the impossible 1/0 directions instead of crashing or handing
    back garbage weights."""
    if condition_number(cov) > CONDITION_LIMIT:
        return np.linalg.pinv(cov.values), True
    try:
        return np.linalg.inv(cov.values), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov.values), True


if __name__ == "__main__":  # smoke test on fixed synthetic data
    rng = np.random.default_rng(7)
    t, tickers = 500, ["AAPL", "MSFT", "GOOG", "NVDA"]
    market = rng.normal(0, 0.012, t)                       # shared wave
    r = np.column_stack([market + rng.normal(0, 0.006, t) for _ in tickers])
    df = pd.DataFrame(r, columns=tickers)

    cov = df.cov() * TRADING_DAYS
    cleaned, n_clip = clip_eigenvalues(cov, n_obs=t)
    fac = eigen_factors(cleaned)

    print(f"condition number (raw)     : {condition_number(cov):,.1f}")
    print(f"eigenvalues clipped        : {n_clip}")
    print(f"trace preserved            : "
          f"{np.isclose(np.trace(cov.values), np.trace(cleaned.values))}")
    for i, (lam, pct) in enumerate(zip(fac["eigenvalues"],
                                       fac["variance_explained"])):
        print(f"  PC{i+1}: λ={lam:.4f}  {pct:5.1f}% of variance")
    print("\nfactor loadings (√λ-scaled):")
    print(fac["loadings"].round(3))
    ts = project_returns(df, fac["eigenvectors"], k=2)
    print(f"\nfactor time series shape   : {ts.shape}")
    inv, used_pinv = safe_inverse(cleaned)
    print(f"inverse via pinv fallback  : {used_pinv}")
