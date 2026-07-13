"""
Eigenrisk — eigendecomposition / PCA layer over the covariance estimators.

The covariance matrix is a tangled web: N assets → N×(N−1)/2 pairwise
relationships, none independent. Eigendecomposition untangles it into
ORTHOGONAL risk factors — statistical directions that do not overlap.

Quant Deep Dive (the rubber-sheet intuition)
--------------------------------------------
Picture returns as points on a rubber sheet the market stretches:
- **Eigenvectors** are the straight, unbending directions of stretch —
  the risk pathways. The first one is almost always a "market wave"
  (everything moves together); later ones are rotations between groups.
- **Eigenvalues** are the stretch STRENGTH along each pathway — the
  variance that factor carries. Sorted descending, they rank the
  portfolio's leading risk drivers.

Honest labeling: these factors are STATISTICAL, not named. Calling PC2
"growth vs value" is an interpretation a human adds — the math only
guarantees the directions are orthogonal and variance-ranked.

Why this layer exists (the inversion trap)
------------------------------------------
Optimizers invert the covariance matrix. Inversion flips every
eigenvalue to 1/λ, so the TINIEST noise eigenvalues become the BIGGEST
weights — noise explodes, signal vanishes. Two defenses, both here:
1. **Ledoit-Wolf shrinkage** (src/covariance.py) pulls the whole matrix
   toward a stable target before decomposition.
2. **Eigenvalue clipping** (Marcenko-Pastur flavor): eigenvalues below
   the random-noise ceiling λ₊ = σ̄²(1+√(N/T))² are deemed
   indistinguishable from noise and averaged flat, preserving total
   variance (the trace) while killing the 1/λ blow-up.
   Honest limit: Marcenko-Pastur is an N,T→∞ asymptotic result; at this
   engine's N of 6–13 assets it is a principled, teachable noise floor,
   not an exact law.

Structural guardrails
---------------------
- **Condition number** κ = λmax/λmin measured BEFORE any inversion;
  κ ≳ 1e8 flags a numerically fragile matrix.
- **Deterministic sign alignment**: eigh may return v or −v arbitrarily
  between runs. The largest-|entry| in each eigenvector is forced
  positive so a factor hedge never silently inverts across reruns.
- **Degenerate fallback**: a singular matrix (two assets 100%
  correlated → a zero eigenvalue) routes inversion through the
  Moore-Penrose pseudo-inverse instead of crashing.

Uses np.linalg.eigh directly (symmetric-matrix solver) — no sklearn PCA
wrapper — so the linear algebra is transparent and interview-defensible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# κ above this → matrix treated as numerically fragile; inversion falls
# back to the pseudo-inverse. float64 has ~16 significant digits; 1e8
# leaves half of them for the arithmetic on top of the inversion.
CONDITION_LIMIT = 1e8


# ---------------------------------------------------------------- guards

def condition_number(cov: pd.DataFrame) -> float:
    """Structural stability ratio λmax/λmin (∞ when λmin ≤ 0).

    Measured BEFORE inversion: a huge κ means 1/λmin will amplify noise
    and float error, so downstream code should shrink, clip, or pinv.
    """
    values = np.linalg.eigvalsh(cov.values)
    lam_min, lam_max = float(values[0]), float(values[-1])
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def align_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    """Force each eigenvector's largest-|entry| positive (v vs −v fix).

    eigh is free to return v or −v; without this, the same portfolio on
    two runs could report a factor hedge with flipped sign — an
    inverted trade. Deterministic convention: anchor on the dominant
    asset in each factor and make its loading positive.
    """
    anchor = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[anchor, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs


# ------------------------------------------------------------- denoising

def marcenko_pastur_bounds(n_assets: int, n_obs: int,
                           sigma2: float = 1.0) -> tuple[float, float]:
    """Marcenko-Pastur noise band λ± = σ²(1±√(N/T))².

    Eigenvalues of a PURE-NOISE covariance matrix fall inside this band
    (asymptotically). λ₊ is the noise ceiling: an eigenvalue below it is
    statistically indistinguishable from noise for this N and T.
    Council ruling: at this engine's N of 6–13 the asymptotics are weak —
    use as a diagnostic overlay on the eigenvalue chart, captioned as a
    heuristic reference line, not a hard statistical test.
    """
    if n_obs <= 0 or n_assets <= 0:
        raise ValueError("n_assets and n_obs must be positive")
    q = n_assets / n_obs
    root = np.sqrt(q)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def clip_eigenvalues(cov: pd.DataFrame, n_obs: int) -> tuple[pd.DataFrame, int]:
    """Marcenko-Pastur-style eigenvalue clipping. Returns (cleaned, n_clipped).

    Eigenvalues below λ₊ = σ̄²(1+√(N/T))² — the ceiling a PURE-NOISE
    correlation matrix would produce — carry no distinguishable signal.
    They are replaced by their own average (not zero), which preserves
    the trace: total portfolio variance is untouched, only its split
    across noise directions is flattened. That kills the 1/λ explosion
    when the matrix is later inverted.

    σ̄² is estimated as the average variance not explained by the kept
    (signal) eigenvalues. n_obs is T, the number of return observations
    behind the covariance estimate.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    values, vectors = np.linalg.eigh(cov.values)   # ascending
    n = len(values)
    q = n / n_obs
    # iterate once: assume top eigenvalue is signal, estimate noise var
    # from the rest, then form the MP ceiling.
    sigma2 = values[:-1].mean() if n > 1 else values.mean()
    _, lam_plus = marcenko_pastur_bounds(n, n_obs, sigma2)
    noise = values < lam_plus
    n_clipped = int(noise.sum())
    if 0 < n_clipped < n:
        cleaned = values.copy()
        cleaned[noise] = values[noise].mean()      # flatten, preserve trace
    else:
        cleaned = values                            # nothing clipped (or all —
        n_clipped = 0                               # degenerate; leave as-is)
    rebuilt = (vectors * cleaned) @ vectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0           # enforce exact symmetry
    return (pd.DataFrame(rebuilt, index=cov.index, columns=cov.columns),
            n_clipped)


# ---------------------------------------------------------- decomposition

def eigen_factors(cov: pd.DataFrame) -> dict:
    """Full eigendecomposition with guardrails applied.

    Returns dict:
      eigenvalues          — descending np.ndarray
      eigenvectors         — DataFrame (assets × PC1..PCn), sign-aligned
      variance_explained   — % of total variance per factor (descending)
      loadings             — eigenvectors scaled by √λ: how much each
                             asset anchors onto each risk factor, in
                             return units (interpretable heatmap matrix)
      condition_number     — λmax/λmin of the input
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
    """Project raw returns onto the top-K factors → factor time series.

    F = (R − R̄) · V[:, :k]. Each column is the daily return of one
    statistical risk factor; PC1's series is what an eigen-hedge would
    aim to neutralize with index derivatives.
    """
    k = max(1, min(k, eigenvectors.shape[1]))
    demeaned = returns[eigenvectors.index] - returns[eigenvectors.index].mean()
    projected = demeaned.values @ eigenvectors.values[:, :k]
    return pd.DataFrame(projected, index=returns.index,
                        columns=list(eigenvectors.columns[:k]))


def pc1_exposure(weights: np.ndarray, eigen: dict) -> float:
    """Share of portfolio variance carried by PC1 alone, in [0, 1].

    Var(w) = Σ λᵢ (vᵢᵀw)² — the portfolio's variance split across the
    orthogonal factors. PC1's slice over the total is the macro-vs-
    idiosyncratic headline: how much of the book is ONE systematic wave.
    In a crisis this ratio spikes toward 1 — the diversification the
    calm-period matrix showed collapses into a single stretch direction.
    """
    v = eigen["eigenvectors"].values
    lam = np.clip(eigen["eigenvalues"], 0.0, None)
    contrib = lam * (v.T @ np.asarray(weights)) ** 2
    total = contrib.sum()
    return float(contrib[0] / total) if total > 0 else 0.0


def safe_inverse(cov: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Invert with a pseudo-inverse fallback. Returns (inverse, used_pinv).

    A singular or near-singular matrix (κ > CONDITION_LIMIT — e.g. two
    assets perfectly correlated) routes through np.linalg.pinv, which
    zeroes the impossible 1/0 directions instead of crashing or
    returning garbage weights.
    """
    if condition_number(cov) > CONDITION_LIMIT:
        return np.linalg.pinv(cov.values), True
    try:
        return np.linalg.inv(cov.values), False
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov.values), True


if __name__ == "__main__":  # smoke test — deterministic synthetic data
    rng = np.random.default_rng(7)
    t, tickers = 500, ["AAPL", "MSFT", "GOOG", "NVDA"]
    market = rng.normal(0, 0.012, t)                       # shared wave
    r = np.column_stack([market + rng.normal(0, 0.006, t) for _ in tickers])
    df = pd.DataFrame(r, columns=tickers)

    cov = df.cov() * 252
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
