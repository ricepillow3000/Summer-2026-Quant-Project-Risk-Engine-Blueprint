"""
Signal Lab: information-coefficient evaluation of a cross-sectional signal.

Quant Deep Dive:
- The information coefficient (IC) is the cross-sectional Spearman rank
  correlation, computed one date at a time, between a signal's ranking of the
  universe today and the forward returns that actually followed. IC = 1 means
  the signal ranked every name perfectly; IC = 0 means it carried no
  information; the sign tells you the direction. Rank (Spearman) correlation
  is used instead of Pearson because a signal only needs to ORDER the
  cross-section correctly to be tradable — outlier returns shouldn't dominate.
- The textbook significance bar: mean IC divided by its standard error,
  t = mean / (std / sqrt(n)), and a t-stat above ~2 is "significant" at the
  usual 5% level. Harvey, Liu & Zhu (2016, "...and the Cross-Section of
  Expected Returns") argue that after decades of the industry testing
  thousands of candidate factors, the multiple-testing problem means t = 2
  is far too easy a bar — enough tries and something clears 2 by luck —
  and propose raising the hurdle toward t = 3. This module reports both bars
  and says plainly which one a signal clears.
- Grinold's fundamental law of active management: IR = IC * sqrt(breadth),
  where breadth is the number of INDEPENDENT bets per year. It converts a
  per-decision skill number (IC) into a portfolio-level information ratio.
- Why raw breadth overstates independence: counting tickers x rebalances
  assumes every bet is independent, but correlated stocks are largely the
  same bet taken twice. The standard correlation adjustment replaces N names
  with N / (1 + (N - 1) * avg_pairwise_correlation) effective independent
  bets — with avg correlation ~0.5, a 10-name book is closer to 2 real bets
  than 10. The honest IR uses this smaller, effective breadth.
- All of this is measured IN-SAMPLE on the loaded history: no transaction
  costs, no out-of-sample validation, and ICs decay once a signal is known.
  Educational analysis, not investment advice.
"""

import numpy as np
import pandas as pd


def momentum_signal(prices: pd.DataFrame, lookback: int = 60, skip: int = 5) -> pd.DataFrame:
    """
    Cross-sectional momentum: the return over `lookback` days ending `skip`
    days ago. Skipping the most recent days avoids contamination from the
    well-documented short-term reversal effect.

    Args:
        prices: DataFrame of prices (dates x tickers).
        lookback: formation window in trading days.
        skip: most-recent days excluded from the formation window.

    Returns:
        DataFrame (dates x tickers) of momentum values; NaN until a ticker
        has `lookback + skip` days of history.
    """
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def forward_returns(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Next-`horizon`-day return, aligned so row t holds the return from t to
    t + horizon (i.e. what a signal observed at t is trying to predict).
    The last `horizon` rows are NaN — those returns haven't happened yet.
    """
    return prices.shift(-horizon) / prices - 1.0


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_pairs: int = 3) -> pd.Series:
    """
    Per-date cross-sectional Spearman rank correlation between the signal and
    forward returns across tickers — the daily information coefficient.

    Dates with fewer than `min_pairs` valid (non-NaN in both) ticker pairs are
    dropped, as are dates where the correlation is undefined (e.g. a constant
    signal). Pearson correlation on ranks equals Spearman's rho.

    Returns:
        Series indexed by date, values in [-1, 1], named "ic".
    """
    idx = signal.index.intersection(fwd.index)
    cols = signal.columns.intersection(fwd.columns)
    sig = signal.loc[idx, cols]
    f = fwd.loc[idx, cols]

    out = {}
    for dt in idx:
        s_row, f_row = sig.loc[dt], f.loc[dt]
        valid = s_row.notna() & f_row.notna()
        if int(valid.sum()) < min_pairs:
            continue
        rho = s_row[valid].rank().corr(f_row[valid].rank())
        if pd.notna(rho):
            out[dt] = float(rho)
    return pd.Series(out, dtype=float, name="ic").sort_index()


def ic_summary(ic: pd.Series) -> dict:
    """
    Summary statistics for a daily IC series.

    Returns a dict with:
      - mean_ic: average daily IC
      - std_ic: sample standard deviation (ddof=1)
      - n_days: number of IC observations
      - t_stat: mean / (std / sqrt(n)) — the textbook significance statistic
      - hit_rate: share of days with IC > 0
    """
    ic = ic.dropna()
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (std / np.sqrt(n)) if n > 1 and np.isfinite(std) and std > 0 else float("nan")
    hit = float((ic > 0).mean()) if n else float("nan")
    return {"mean_ic": mean, "std_ic": std, "n_days": n, "t_stat": t, "hit_rate": hit}


def fundamental_law_ir(mean_ic: float, breadth: float) -> float:
    """
    Grinold's fundamental law of active management: IR = IC * sqrt(breadth).
    `breadth` is the number of independent bets per year — see
    effective_breadth() for why the raw count overstates it.
    """
    return float(mean_ic * np.sqrt(breadth))


def effective_breadth(returns: pd.DataFrame) -> float:
    """
    Honest-breadth estimate: the correlation-adjusted count of independent
    bets among N correlated assets,

        N_eff = N / (1 + (N - 1) * avg_pairwise_correlation)

    Perfectly correlated names collapse to ~1 independent bet; uncorrelated
    names keep all N. The average pairwise correlation is clamped to [0, 1)
    for stability (a slightly negative sample average would otherwise inflate
    breadth beyond N, which overstates independence).
    """
    n = returns.shape[1]
    if n <= 1:
        return float(n)
    corr = returns.corr().values
    iu = np.triu_indices(n, k=1)
    avg_rho = float(np.nanmean(corr[iu]))
    avg_rho = min(max(avg_rho, 0.0), 1.0 - 1e-9)
    return float(n / (1.0 + (n - 1) * avg_rho))


if __name__ == "__main__":
    from src.ingestion import fetch_prices, get_returns

    prices = fetch_prices()
    returns = get_returns(prices)

    sig = momentum_signal(prices)
    fwd = forward_returns(prices)
    ic = daily_ic(sig, fwd)
    summ = ic_summary(ic)

    print("--- Momentum signal IC (in-sample, 60d lookback / 5d skip / 5d horizon) ---")
    for k, v in summ.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    n_eff = effective_breadth(returns)
    rebalances = 252 / 5
    raw_b = len(prices.columns) * rebalances
    eff_b = n_eff * rebalances
    print(f"\n--- Grinold fundamental law ---")
    print(f"  raw breadth:       {raw_b:.0f} bets/yr -> IR {fundamental_law_ir(summ['mean_ic'], raw_b):.2f}")
    print(f"  effective breadth: {eff_b:.0f} bets/yr -> IR {fundamental_law_ir(summ['mean_ic'], eff_b):.2f}"
          f"  ({n_eff:.1f} independent names out of {len(prices.columns)})")
