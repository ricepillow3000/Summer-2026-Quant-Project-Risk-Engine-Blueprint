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


def risk_parity_weights(cov: pd.DataFrame, iters: int = 5000, tol: float = 1e-10) -> np.ndarray:
    """
    Equal-risk-contribution (ERC) long-only weights via the standard fixed-point
    iteration: w_i <- (1/n) / (Sigma w)_i, renormalized. Converges for a
    positive-definite covariance matrix.
    """
    sigma = cov.values
    n = sigma.shape[0]
    budget = np.ones(n) / n
    w = np.ones(n) / n
    for _ in range(iters):
        marginal = sigma @ w
        marginal = np.where(marginal <= 0, 1e-12, marginal)
        w_new = budget / marginal
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


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


# Historical stress scenarios as (drawdown %, volatility shock %) magnitudes.
# Drawdown is the peak-to-trough equity move; vol shock scales daily dispersion.
# Calibrated to the rough character of each episode - illustrative, not exact.
SCENARIOS = {
    "2008 Global Financial Crisis": {"drawdown": -50, "vol": 150},
    "COVID-19 crash (Mar 2020)": {"drawdown": -34, "vol": 200},
    "2022 rate-shock selloff": {"drawdown": -25, "vol": 80},
    "1987 Black Monday": {"drawdown": -22, "vol": 250},
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
