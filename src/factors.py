"""
Factor exposures: what systematic risks is this portfolio actually taking?

Quant Deep Dive:
- Eigen-decomposition (analytics.py) finds *statistical* factors - real but
  unnamed. This module finds *named* exposures a PM actually talks about:
  market beta, plus size, value, and momentum tilts.
- We build long-short style factors from liquid ETF proxies and run one OLS
  regression of the portfolio's daily returns on them. The betas are the tilts;
  R-squared says how much of the portfolio's movement these factors explain.
- Honest labelling: these are ETF-proxy factors, not the academic Fama-French
  research factors. They capture the same intuition with tradable instruments,
  which is the defensible version for a live tool.
"""

import numpy as np
import pandas as pd
from src.ingestion import fetch_prices, get_returns, fetch_risk_free_rate

# ETF proxies used to construct the factors.
#   Market   = SPY (broad market)
#   Size     = IWM - SPY  (small-cap minus market)
#   Value    = IWD - IWF  (value minus growth)
#   Momentum = MTUM - SPY (momentum minus market)
_PROXIES = ["SPY", "IWM", "IWD", "IWF", "MTUM"]


def _build_factors(period: str = "2y") -> pd.DataFrame:
    """Daily factor return series from ETF proxies."""
    px = fetch_prices(_PROXIES, period=period)
    r = get_returns(px)
    return pd.DataFrame({
        "Market": r["SPY"],
        "Size": r["IWM"] - r["SPY"],
        "Value": r["IWD"] - r["IWF"],
        "Momentum": r["MTUM"] - r["SPY"],
    }).dropna()


def factor_exposures(portfolio_returns: pd.Series, period: str = "2y") -> dict:
    """
    Regress portfolio daily returns on the named factors via OLS.

    Returns betas per factor, the regression R-squared, and the annualized
    alpha (intercept). Aligns on common dates so calendar mismatches can't
    distort the fit.

    Alpha is return over the RISK-FREE leg, so the portfolio and the market
    both enter in excess of the 13-week T-bill. Size/Value/Momentum are
    long-short and already self-financing, so they take no adjustment. If the
    T-bill fetch fails the intercept is a raw-return one and `alpha_basis`
    says so - the same doctrine as Sharpe: never quietly pass off an
    unadjusted number as the adjusted one.
    """
    factors = _build_factors(period)
    df = pd.concat([portfolio_returns.rename("port"), factors], axis=1).dropna()
    if len(df) < 30:
        raise RuntimeError("Not enough overlapping history to estimate factors.")

    rf_annual = fetch_risk_free_rate()
    rf_daily = (rf_annual / 252.0) if rf_annual is not None else 0.0

    y = df["port"].values - rf_daily
    X = df[["Market", "Size", "Value", "Momentum"]].values.astype(float).copy()
    X[:, 0] -= rf_daily                        # Market -> EXCESS market
    X = np.column_stack([np.ones(len(X)), X])  # intercept

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["Market", "Size", "Value", "Momentum"]
    return {
        "alpha_annual": float(beta[0] * 252),
        "risk_free_annual": rf_annual,
        "alpha_basis": (
            f"excess of the 13-week T-bill at {rf_annual:.2%} annual"
            if rf_annual is not None else
            "RAW returns - no T-bill rate available, so this is NOT "
            "risk-free-adjusted"),
        "betas": {name: float(b) for name, b in zip(names, beta[1:])},
        "r_squared": float(r2),
        "observations": len(df),
    }


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    port = returns @ weights

    res = factor_exposures(port)
    print("--- Factor exposures (equal-weight default universe) ---")
    print(f"  R-squared : {res['r_squared']:.1%}")
    print(f"  Alpha (ann): {res['alpha_annual']:+.2%}")
    for name, b in res["betas"].items():
        print(f"  {name:9s}: {b:+.2f}")
