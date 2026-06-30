"""
Analytics: covariance mapping and eigen-decomposition.

Quant Deep Dive:
- The covariance matrix tells you how each asset moves relative to every other.
  A high number between NVDA and TSLA means they tend to crash together — that
  is concentration risk. Owning both doesn't protect you.
- Eigen-decomposition breaks the covariance matrix into "risk factors" —
  hidden forces driving the portfolio. The first eigenvector almost always
  turns out to be a "market factor" (everything goes up/down together).
  The second might be "growth vs value." These are the building blocks of PCA.
- In a real hedge fund, risk systems run this decomposition every night to
  understand where the portfolio's risk is actually coming from.
"""

import numpy as np
import pandas as pd
from src.ingestion import fetch_prices, get_returns


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Compute the sample covariance matrix from daily returns.

    Args:
        returns: DataFrame of daily percent returns (assets as columns).
        annualize: If True, multiply by 252 (trading days/year) to express
                   risk in annual terms — standard for quant reporting.

    Returns:
        Covariance matrix as a DataFrame (tickers x tickers).
    """
    cov = returns.cov()
    if annualize:
        cov = cov * 252
    return cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix — covariance scaled to [-1, 1].
    Easier to read at a glance: 1.0 = move in lockstep, 0 = independent.
    """
    return returns.corr()


def eigen_decomposition(cov: pd.DataFrame) -> dict:
    """
    Decompose the covariance matrix into eigenvalues and eigenvectors.

    Returns a dict with:
      - eigenvalues: variance explained by each risk factor (descending order)
      - eigenvectors: the risk factors themselves (columns)
      - variance_explained: what % of total portfolio risk each factor drives
      - tickers: asset labels for reference
    """
    values, vectors = np.linalg.eigh(cov.values)

    # eigh returns ascending order — reverse so the biggest factor is first
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]

    variance_explained = values / values.sum() * 100

    return {
        "eigenvalues": values,
        "eigenvectors": pd.DataFrame(vectors, index=cov.index),
        "variance_explained": variance_explained,
        "tickers": list(cov.columns),
    }


def portfolio_volatility(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """
    Annualized portfolio volatility (standard deviation) for a given weight vector.
    Formula: sqrt(w^T * Sigma * w)

    Args:
        weights: array of asset weights that sum to 1.
        cov: annualized covariance matrix.

    Returns:
        Volatility as a decimal (e.g. 0.18 = 18% annualized vol).
    """
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