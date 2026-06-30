"""
Risk Engine: CVaR and Monte Carlo simulation.

Quant Deep Dive:
- VaR (Value at Risk) answers: "What's the most I lose on a bad day (95th pctile)?"
- CVaR (Conditional VaR) answers: "When things ARE that bad, how bad on average?"
  CVaR is strictly better — VaR ignores what happens in the tail, CVaR measures it.
- Monte Carlo: instead of assuming returns are normally distributed (they aren't),
  we bootstrap from real historical returns. This captures actual fat tails —
  the real crash days that a normal distribution would say are "impossible."
"""

import numpy as np
import pandas as pd
from src.ingestion import fetch_prices, get_returns


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute daily portfolio returns for a given weight vector."""
    return returns @ weights


def var(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.
    The loss threshold you exceed only (1 - confidence)% of the time.

    Returns a positive number representing the loss (e.g. 0.032 = 3.2% loss).
    """
    return float(-np.percentile(port_returns, (1 - confidence) * 100))


def cvar(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).
    Average loss on the worst (1 - confidence)% of days.

    This is the number risk desks actually use — it captures tail severity,
    not just where the tail begins.
    """
    threshold = np.percentile(port_returns, (1 - confidence) * 100)
    tail = port_returns[port_returns <= threshold]
    return float(-tail.mean())


def monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    horizon_days: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap Monte Carlo simulation of portfolio returns.

    For each simulation: randomly sample `horizon_days` daily returns
    (with replacement) from history and compound them into a final value.
    No normality assumption — we use the real return distribution.

    Args:
        returns: historical daily returns DataFrame.
        weights: portfolio weight vector.
        n_simulations: number of simulated futures to run.
        horizon_days: trading days to simulate (252 = 1 year).
        confidence: CVaR confidence level.

    Returns:
        dict with simulation results and risk metrics.
    """
    port_returns = portfolio_daily_returns(returns, weights).values
    rng = np.random.default_rng(seed=42)

    # Each row = one simulated year of daily returns
    sampled = rng.choice(port_returns, size=(n_simulations, horizon_days), replace=True)

    # Compound daily returns into a final portfolio value (starting at $1)
    final_values = np.prod(1 + sampled, axis=1)
    total_returns = final_values - 1

    # Risk metrics on the simulated distribution
    sim_var = float(-np.percentile(total_returns, (1 - confidence) * 100))
    threshold = np.percentile(total_returns, (1 - confidence) * 100)
    sim_cvar = float(-total_returns[total_returns <= threshold].mean())

    return {
        "final_values": final_values,
        "total_returns": total_returns,
        "median_return": float(np.median(total_returns)),
        "mean_return": float(np.mean(total_returns)),
        "var": sim_var,
        "cvar": sim_cvar,
        "worst_case": float(total_returns.min()),
        "best_case": float(total_returns.max()),
        "prob_loss": float((total_returns < 0).mean()),
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "confidence": confidence,
    }


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)

    n = len(returns.columns)
    equal_weights = np.ones(n) / n
    port_returns = portfolio_daily_returns(returns, equal_weights)

    # Historical risk metrics
    h_var = var(port_returns)
    h_cvar = cvar(port_returns)
    print("--- Historical Risk (Equal-Weight Portfolio) ---")
    print(f"  Daily VaR  (95%): {h_var:.2%}  — on a bad day, expect to lose at least this")
    print(f"  Daily CVaR (95%): {h_cvar:.2%}  — when it's bad, this is the average loss")

    # Monte Carlo
    print("\n--- Running 10,000 Monte Carlo Simulations (1-Year Horizon) ---")
    mc = monte_carlo(returns, equal_weights)
    print(f"  Median 1-year return : {mc['median_return']:+.1%}")
    print(f"  Mean 1-year return   : {mc['mean_return']:+.1%}")
    print(f"  1-Year VaR  (95%)    : {mc['var']:.1%}  loss in worst 5% of years")
    print(f"  1-Year CVaR (95%)    : {mc['cvar']:.1%}  avg loss in worst 5% of years")
    print(f"  Worst simulated year : {mc['worst_case']:+.1%}")
    print(f"  Best simulated year  : {mc['best_case']:+.1%}")
    print(f"  Probability of loss  : {mc['prob_loss']:.1%}")