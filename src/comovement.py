"""
Co-movement - correlation as a moving picture, not a snapshot.

Quant Deep Dive:
Covariance and correlation carry the SAME directional information (the sign).
The difference is units. Covariance is in squared-return units - cov(A,B) =
0.0004 is unreadable on its own, because its size depends on each asset's
volatility. Correlation is covariance normalized by both vols:

    corr(A,B) = cov(A,B) / (sigma_A * sigma_B)

which forces it into [-1, +1]: +1 = lockstep, 0 = no linear relationship,
-1 = seesaw. In matrix form the same normalization is

    R = D^-1 · Sigma · D^-1

where Sigma is the covariance matrix and D = diag(sigma_1..sigma_n). This
module computes R from Sigma by that identity (not pandas .corr()) so the
math is visible, and - the part a static matrix can't show - tracks how a
pair's correlation MOVES through time with a rolling window. Diversification
is not a constant: two names that look independent on a 2-year average can
run correlation > 0.9 for months inside a stress regime. The monitor flags
when the loaded portfolio's key pair crosses a concentration threshold, and
the defensive simulator answers "what if I cut the pair and moved the
proceeds into the least-correlated name?" - honestly, in both directions:
the shift is measured through the same CVaR engine and reported whether it
helped or hurt.
"""

import numpy as np
import pandas as pd


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix from a covariance matrix by the linear-algebra
    identity R = D^-1 · Sigma · D^-1, with D = diag(volatilities).

    D is diagonal, so D^-1 is just 1/sigma on the diagonal - computed
    directly rather than via a matrix inversion. A zero-variance asset has
    no defined correlation; its row/column comes back NaN (flagged, not
    faked). The diagonal is clamped to exactly 1.0 against float drift.
    """
    sigma = cov.to_numpy(dtype=float)
    vols = np.sqrt(np.diag(sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(vols > 0, 1.0 / vols, np.nan)
    r = sigma * np.outer(inv, inv)
    np.fill_diagonal(r, np.where(vols > 0, 1.0, np.nan))
    return pd.DataFrame(r, index=cov.index, columns=cov.columns)


def rolling_correlation(returns: pd.DataFrame, a: str, b: str,
                        window: int = 21) -> pd.Series:
    """
    Rolling Pearson correlation between two return columns over `window`
    trading days (21 ≈ one month). The first `window-1` rows are NaN by
    construction - not enough history yet, excluded rather than padded.
    """
    return returns[a].rolling(window).corr(returns[b])


def most_correlated_pair(corr: pd.DataFrame) -> tuple[str, str, float]:
    """
    The pair with the highest off-diagonal correlation - the two names most
    likely to stop diversifying each other exactly when it matters.
    """
    m = corr.to_numpy(dtype=float, copy=True)
    m[~np.triu(np.ones_like(m, dtype=bool), k=1)] = np.nan  # keep upper triangle
    c = pd.DataFrame(m, index=corr.index, columns=corr.columns)
    a, b = c.stack().idxmax()
    return a, b, float(c.loc[a, b])


def defensive_shift(weights: np.ndarray, tickers: list[str],
                    pair: tuple[str, str], into: str,
                    cut: float = 0.15) -> np.ndarray:
    """
    Simulate cutting each name of a correlated pair by up to `cut` of book
    weight and moving the freed weight into `into` (the engine picks the
    least-correlated name, never a hardcoded ticker).

    Guards: a leg is cut by min(cut, its current weight) so no weight ever
    goes negative - cutting more than a position holds would silently open
    a short. Total exposure is preserved exactly: everything cut is added
    to `into`, nothing created or destroyed.

    This is a SIMULATION for comparison, not advice - the caller must run
    the result through the risk engine and report the before/after honestly,
    whichever direction it moves.
    """
    w = np.asarray(weights, dtype=float).copy()
    idx = {t: i for i, t in enumerate(tickers)}
    freed = 0.0
    for leg in pair:
        take = min(cut, w[idx[leg]])
        w[idx[leg]] -= take
        freed += take
    w[idx[into]] += freed
    return w


def least_correlated_to_pair(corr: pd.DataFrame,
                             pair: tuple[str, str]) -> tuple[str, float]:
    """
    The name whose AVERAGE correlation to both legs of the pair is lowest -
    the most independent destination for a defensive shift. Returns the
    ticker and that average correlation (disclosed so the UI can say
    honestly how independent the "independent" name actually is).
    """
    others = [t for t in corr.columns if t not in pair]
    if not others:
        raise ValueError("universe has no asset outside the pair")
    avg = corr.loc[others, list(pair)].mean(axis=1)
    best = avg.idxmin()
    return str(best), float(avg.loc[best])


if __name__ == "__main__":
    from src.ingestion import fetch_prices, get_returns
    from src.analytics import covariance_matrix

    prices = fetch_prices()
    returns = get_returns(prices)
    cov = covariance_matrix(returns)

    r = correlation_from_cov(cov)
    print("--- Correlation via R = D^-1 Sigma D^-1 (should match .corr()) ---")
    print(r.round(2))
    print("max |R - .corr()| =", float((r - returns.corr()).abs().max().max()))

    a, b, top = most_correlated_pair(r)
    print(f"\nMost correlated pair: {a}/{b} at {top:+.2f}")

    roll = rolling_correlation(returns, a, b).dropna()
    print(f"21d rolling corr - last: {roll.iloc[-1]:+.2f}, "
          f"min: {roll.min():+.2f}, max: {roll.max():+.2f}")

    dest, dcorr = least_correlated_to_pair(r, (a, b))
    print(f"Least-correlated destination: {dest} (avg corr to pair {dcorr:+.2f})")
