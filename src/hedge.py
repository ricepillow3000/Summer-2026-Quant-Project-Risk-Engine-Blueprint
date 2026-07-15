"""
Hedge & Balance - the second solution.

Crisis Conviction is the *emotional* answer (don't panic-sell). This is the
*structural* one: pair a holding with a low- or negatively-correlated partner so
a move in one is partly offset by the other - classic Markowitz diversification,
computed from the real covariance matrix, never a narrative.

Quant Deep Dive
---------------
For two assets a (anchor) and b (hedge) with annualized variances σa², σb² and
covariance σab, the LONG-ONLY minimum-variance blend has closed-form weight

        w_a* = (σb² − σab) / (σa² + σb² − 2σab),   clipped to [0, 1]

and blended variance σp² = w_a²σa² + w_b²σb² + 2 w_a w_b σab. The lower σab (the
more the two move against each other), the more the blend's volatility drops
below the anchor's - that is the whole mechanism.

Honest limit (surfaced in the UI, not buried): correlations are historical and
UNSTABLE. In a real crash they converge toward +1 - everything falls together -
so a hedge softens ordinary risk but is not crisis-proof. Diversification fades
exactly when it is needed most. This is the counterweight to the Conviction tab,
not a contradiction of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_hedges(corr: pd.DataFrame, anchor: str) -> pd.Series:
    """Every other asset ranked by correlation to `anchor`, ascending.

    Most negative (best natural hedge) first; most positive (moves with the
    anchor, no diversification) last.
    """
    if anchor not in corr.columns:
        raise KeyError(f"{anchor!r} not in correlation matrix")
    return corr[anchor].drop(labels=[anchor]).sort_values()


def min_variance_pair(cov: pd.DataFrame, anchor: str, hedge: str) -> dict:
    """Long-only minimum-variance blend of `anchor` and `hedge`.

    `cov` is the annualized covariance matrix. Returns the blend weights, the
    blended volatility vs. holding the anchor alone, and the volatility
    reduction that buys.
    """
    va = float(cov.loc[anchor, anchor])
    vb = float(cov.loc[hedge, hedge])
    cab = float(cov.loc[anchor, hedge])

    denom = va + vb - 2.0 * cab
    w_a = 0.5 if denom == 0 else (vb - cab) / denom
    w_a = float(np.clip(w_a, 0.0, 1.0))
    w_b = 1.0 - w_a

    blended_var = w_a**2 * va + w_b**2 * vb + 2.0 * w_a * w_b * cab
    blended_vol = float(np.sqrt(max(blended_var, 0.0)))
    anchor_vol = float(np.sqrt(max(va, 0.0)))
    denom_corr = np.sqrt(va * vb)
    corr = float(cab / denom_corr) if denom_corr > 0 else 0.0

    return {
        "anchor": anchor,
        "hedge": hedge,
        "w_anchor": w_a,
        "w_hedge": w_b,
        "anchor_vol": anchor_vol,
        "blended_vol": blended_vol,
        "vol_reduction": 0.0 if anchor_vol == 0 else 1.0 - blended_vol / anchor_vol,
        "correlation": corr,
    }


if __name__ == "__main__":  # smoke test on synthetic data
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, 500)
    df = pd.DataFrame({"A": a, "B": -a + rng.normal(0, 0.001, 500),  # near mirror
                       "C": rng.normal(0, 0.01, 500)}, index=idx)      # independent
    try:
        from analytics import correlation_matrix, covariance_matrix  # type: ignore
    except ModuleNotFoundError:
        from src.analytics import correlation_matrix, covariance_matrix
    cov = covariance_matrix(df)
    corr = correlation_matrix(df)
    print("hedges for A (ascending corr):")
    print(rank_hedges(corr, "A"))
    print("\nA hedged by its mirror B:")
    print(min_variance_pair(cov, "A", "B"))
