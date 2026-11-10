"""
analytics: covariance, correlation and eigen-decomposition.

how the math works:
- the covariance matrix says how each name moves vs every other. a big number
  between NVDA and TSLA means they crash together - that's concentration
  risk, owning both doesn't protect you.
- eigen-decomposition splits that matrix into hidden "risk factors". the
  first one is almost always the market (everything moving together), the
  next might be growth vs value. this is PCA's building block, and real risk
  desks run it every night to see where the risk actually lives.
"""

import numpy as np
import pandas as pd
from src.ingestion import fetch_prices, get_returns
from src.covariance import TRADING_DAYS


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """sample covariance of daily returns (tickers x tickers), annualized by
    TRADING_DAYS unless annualize=False."""
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """pearson correlation - covariance squeezed into [-1, 1].
    1 = moves in lockstep, 0 = unrelated."""
    return returns.corr()


def eigen_decomposition(cov: pd.DataFrame) -> dict:
    """eigenvalues/vectors of the cov, biggest factor first. returns
    eigenvalues, eigenvectors (columns), variance_explained (% of total risk
    per factor) and tickers."""
    values, vectors = np.linalg.eigh(cov.values)
    idx = np.argsort(values)[::-1]   # eigh sorts ascending, we want biggest first
    values = values[idx]
    vectors = vectors[:, idx]
    return {
        "eigenvalues": values,
        "eigenvectors": pd.DataFrame(vectors, index=cov.index),
        "variance_explained": values / values.sum() * 100,
        "tickers": list(cov.columns),
    }


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """annual portfolio vol, sqrt(w' Sigma w), from weights summing to 1 and
    an annualized cov. 0.18 = 18%."""
    return float(np.sqrt(weights @ cov.values @ weights))


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)

    cov = covariance_matrix(returns)
    corr = correlation_matrix(returns)
    eigen = eigen_decomposition(cov)

    print("--- Annualized Covariance Matrix ---")
    print(cov.round(4))

    print("\n--- Correlation Matrix ---")
    print(corr.round(2))

    print("\n--- Eigen-Decomposition: Variance Explained by Each Risk Factor ---")
    for i, (val, pct) in enumerate(zip(eigen["eigenvalues"], eigen["variance_explained"])):
        print(f"  Factor {i+1}: eigenvalue={val:.4f}  |  {pct:.1f}% of total risk")

    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
    vol = portfolio_volatility(equal_weights, cov)
    print(f"\n--- Equal-Weight Portfolio Volatility: {vol:.1%} annualized ---")
