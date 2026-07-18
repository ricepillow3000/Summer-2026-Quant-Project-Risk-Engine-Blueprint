"""
Risk-management strategies - institutional methodologies, no proprietary data.

Everything here runs on the same public price returns the engine already
fetches. These are published frameworks (risk parity, risk budgeting, managed
volatility), not any firm's actual positions - the legal, defensible way to
make a student project look like a real risk desk.

Quant Deep Dive:
- Risk contribution answers "where is my risk actually coming from?" Two assets
  can have equal dollar weight but wildly unequal *risk* weight if one is more
  volatile or more correlated with the rest.
- Risk parity (Bridgewater All-Weather style) re-weights so every asset
  contributes the SAME risk - no single name dominates.
- Volatility targeting (AQR managed-vol style) scales total exposure up or down
  to hold a constant target volatility, using leverage when markets are calm.
"""

import numpy as np
import pandas as pd


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose portfolio volatility into each asset's contribution.

    Marginal contribution = (Sigma w)_i / sigma_p
    Component contribution = w_i * marginal_i   (these sum to sigma_p)
    Percent               = component / sigma_p (these sum to 100%)

    Returns a DataFrame indexed by ticker with weight, risk %, and the gap
    between them - the gap is the concentration story.
    """
    w = np.asarray(weights, dtype=float)
    sigma = cov.values
    port_vol = float(np.sqrt(w @ sigma @ w))
    marginal = sigma @ w / port_vol
    component = w * marginal
    pct = component / port_vol

    return pd.DataFrame({
        "weight": w,
        "risk_pct": pct,
    }, index=cov.index)


def risk_parity_weights(cov: pd.DataFrame, iters: int = 10000,
                        tol: float = 1e-12) -> np.ndarray:
    """
    Equal-risk-contribution (ERC) long-only weights via cyclical coordinate
    descent on the strictly convex problem
        min 0.5 w'Sigma w - lam * sum_i b_i ln(w_i),
    whose first-order condition w_i (Sigma w)_i = lam b_i IS the ERC
    condition after normalization (Spinu 2012; Griveau-Billion, Richard &
    Roncalli 2013). Converges for any positive-definite covariance matrix,
    including baskets with bond, gold, FX or futures legs where the naive
    fixed point w <- b/(Sigma w) is a period-2 oscillator and silently
    returns equal weight (audit 2026-07-18: that failure hit any zero- or
    negative-correlation universe, e.g. the FX and futures presets).
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
    """Annualized portfolio volatility for a weight vector (cov is annualized)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov.values @ w))


def vol_target(weights: np.ndarray, cov: pd.DataFrame,
               target_vol: float = 0.10) -> dict:
    """
    Managed-volatility overlay: scale exposure to hit a target annual vol.

    leverage = target_vol / realized_vol
      < 1  -> de-risk (portfolio is more volatile than target)
      > 1  -> lever up (portfolio is calmer than target)

    Returns the leverage, the scaled weights (sum = leverage), and both vols.
    """
    realized = portfolio_vol(weights, cov)
    leverage = target_vol / realized if realized > 0 else 1.0
    return {
        "leverage": leverage,
        "scaled_weights": np.asarray(weights, dtype=float) * leverage,
        "realized_vol": realized,
        "target_vol": target_vol,
    }


# (A hand-tuned SCENARIOS dict of illustrative crisis magnitudes used to live
# here. Deleted: never imported by the app, and it contradicted the house
# doctrine that stress magnitudes are whatever actually happened - real
# replays live in src/scenarios.py HISTORICAL_REGIMES.)


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
