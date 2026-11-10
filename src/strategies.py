"""
risk-management strategies - published methods, no secret data.

all of it runs on the same public returns the engine already pulls. these are
textbook frameworks (risk parity, risk budgeting, managed vol), not any
firm's real positions.

how the math works:
- risk contribution: "where is my risk actually coming from?" two names can
  have the same dollar weight but very different RISK weight if one is
  jumpier or moves more with the rest.
- risk parity (Bridgewater All-Weather style): reweight so every name adds
  the SAME risk - nobody dominates.
- vol targeting (AQR managed-vol style): scale the whole book up or down to
  hold a fixed target vol, levering up when markets are calm.
"""

import numpy as np
import pandas as pd


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame) -> pd.DataFrame:
    """
    split portfolio vol into each name's share.

    marginal  = (Sigma w)_i / sigma_p
    component = w_i * marginal_i   (sums to sigma_p)
    percent   = component / sigma_p (sums to 100%)

    returns weight and risk % per ticker - the gap between them is the
    concentration story.
    """
    w = np.asarray(weights, dtype=float)
    sigma = cov.values
    port_vol = float(np.sqrt(w @ sigma @ w))
    marginal = sigma @ w / port_vol
    pct = w * marginal / port_vol
    return pd.DataFrame({"weight": w, "risk_pct": pct}, index=cov.index)


def risk_parity_weights(cov: pd.DataFrame, iters: int = 10000,
                        tol: float = 1e-12) -> np.ndarray:
    """
    equal-risk-contribution (ERC) long-only weights, by cyclical coordinate
    descent on the strictly convex problem
        min 0.5 w'Sigma w - lam * sum_i b_i ln(w_i),
    whose first-order condition w_i (Sigma w)_i = lam b_i IS the ERC
    condition once normalized (Spinu 2013; Griveau-Billion, Richard &
    Roncalli 2013). converges for any positive-definite cov. the naive fixed
    point w <- b/(Sigma w) just bounces between two points on baskets with
    zero or negative correlation (bonds, gold, FX, futures) and quietly
    hands back equal weight.
    """
    sigma = cov.values
    n = sigma.shape[0]
    b = np.ones(n) / n
    lam = 1.0
    w = np.ones(n) / n
    for _ in range(iters):
        w_prev = w.copy()
        for i in range(n):
            c_i = sigma[i] @ w - sigma[i, i] * w[i]
            w[i] = (-c_i + np.sqrt(c_i * c_i + 4.0 * sigma[i, i] * lam * b[i])) \
                / (2.0 * sigma[i, i])
        if np.max(np.abs(w - w_prev)) < tol * max(1.0, np.max(np.abs(w))):
            break
    return w / w.sum()


def portfolio_vol(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """annual portfolio vol for a weight vector (cov is annualized)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov.values @ w))


def vol_target(weights: np.ndarray, cov: pd.DataFrame,
               target_vol: float = 0.10) -> dict:
    """
    managed-vol overlay: scale exposure to hit a target annual vol.

    leverage = target_vol / realized_vol
      < 1  -> de-risk (book is jumpier than the target)
      > 1  -> lever up (book is calmer than the target)

    returns the leverage, the scaled weights (they sum to leverage), and both vols.
    """
    realized = portfolio_vol(weights, cov)
    leverage = target_vol / realized if realized > 0 else 1.0
    return {
        "leverage": leverage,
        "scaled_weights": np.asarray(weights, dtype=float) * leverage,
        "realized_vol": realized,
        "target_vol": target_vol,
    }


if __name__ == "__main__":
    from src.ingestion import fetch_prices, get_returns
    from src.analytics import covariance_matrix

    returns = get_returns(fetch_prices())
    cov = covariance_matrix(returns)
    n = returns.shape[1]
    eq = np.ones(n) / n

    print("--- Equal-weight risk contributions ---")
    rc = risk_contributions(eq, cov)
    print((rc * 100).round(1))

    print("\n--- Risk-parity weights ---")
    rp = risk_parity_weights(cov)
    for t, w in zip(cov.index, rp):
        print(f"  {t:10s}: {w:6.1%}")

    print("\n--- Risk parity risk contributions (should be ~equal) ---")
    print((risk_contributions(rp, cov)["risk_pct"] * 100).round(1).to_dict())

    print("\n--- Vol target (10%) on equal weight ---")
    vt = vol_target(eq, cov, 0.10)
    print(f"  realized {vt['realized_vol']:.1%} -> leverage {vt['leverage']:.2f}x")
