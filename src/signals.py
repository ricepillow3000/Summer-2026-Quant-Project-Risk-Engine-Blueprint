"""
Momentum signal, and how to tell if a signal is any good (information coefficient).

How it works:
- the IC is the Spearman rank correlation, one date at a time, between how a
  signal ranks the names today and the returns that actually came next. 1 =
  ranked everything perfectly, 0 = no info, the sign says which way. ranks
  instead of Pearson because a signal only has to get the ORDER right to be
  tradable - one wild return shouldn't run the show.
- the textbook bar: t = mean / (std / sqrt(n)), and t > ~2 is "significant"
  at 5%. Harvey, Liu & Zhu (2016, "...and the Cross-Section of Expected
  Returns") point out the industry has tested thousands of factors, so with
  enough tries something clears 2 by luck - they push the bar toward t = 3.
  we report both bars and say which one a signal clears.
- Grinold's fundamental law: IR = IC * sqrt(breadth), breadth = INDEPENDENT
  bets per year. turns a per-call skill number (IC) into a portfolio-level
  information ratio.
- raw breadth (tickers x rebalances) oversells it: correlated stocks are
  mostly the same bet twice. the usual fix swaps N names for
  N / (1 + (N - 1) * avg_pairwise_correlation) real bets - at avg corr ~0.5 a
  10-name book is closer to 2 bets than 10. the honest IR uses that.
- all of it is IN-SAMPLE on the loaded history: no costs, no out-of-sample
  check, and ICs fade once a signal is known. educational, not investment
  advice.
"""

import numpy as np
import pandas as pd

from src.covariance import TRADING_DAYS


# the momentum window - The Book's sentences read these instead of retyping them
MOMENTUM_LOOKBACK = 60   # formation window, trading days
MOMENTUM_SKIP = 5        # skip the latest days: dodges short-term reversal
IC_HORIZON = 5           # forward-return window the IC is scored on


def momentum_signal(prices: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK,
                    skip: int = MOMENTUM_SKIP) -> pd.DataFrame:
    """
    Momentum: the return over `lookback` days ending `skip` days ago. skipping
    the last few days dodges the well-known short-term reversal effect.

    NaN until a name has `lookback + skip` days of history.
    """
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def forward_returns(prices: pd.DataFrame, horizon: int = IC_HORIZON) -> pd.DataFrame:
    """
    Next-`horizon`-day return: row t holds the return from t to t + horizon
    (what a signal seen at t is trying to call). the last `horizon` rows are
    NaN - those days haven't happened yet.
    """
    return prices.shift(-horizon) / prices - 1.0


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_pairs: int = 3) -> pd.Series:
    """
    The daily IC: rank correlation of signal vs forward returns across names,
    one date at a time (Pearson on ranks = Spearman's rho).

    skips dates with fewer than `min_pairs` names that have both values, and
    dates where it's undefined (e.g. a flat signal). returns a Series named
    "ic", values in [-1, 1].
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
    Stats for a daily IC series: mean_ic, std_ic (ddof=1), n_days,
    t_stat = mean / (std / sqrt(n)), and hit_rate (share of days IC > 0).
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
    Grinold: IR = IC * sqrt(breadth), breadth = independent bets per year
    (see effective_breadth for why the raw count oversells it).
    """
    return float(mean_ic * np.sqrt(breadth))


def effective_breadth(returns: pd.DataFrame) -> float:
    """
    How many truly independent bets N correlated names add up to:

        N_eff = N / (1 + (N - 1) * avg_pairwise_correlation)

    perfectly correlated names shrink to ~1 bet; uncorrelated ones keep all N.
    avg correlation is clamped to [0, 1) - a slightly negative sample average
    would push breadth past N, which oversells independence.
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
    rebalances = TRADING_DAYS / IC_HORIZON
    raw_b = len(prices.columns) * rebalances
    eff_b = n_eff * rebalances
    print(f"\n--- Grinold fundamental law ---")
    print(f"  raw breadth:       {raw_b:.0f} bets/yr -> IR {fundamental_law_ir(summ['mean_ic'], raw_b):.2f}")
    print(f"  effective breadth: {eff_b:.0f} bets/yr -> IR {fundamental_law_ir(summ['mean_ic'], eff_b):.2f}"
          f"  ({n_eff:.1f} independent names out of {len(prices.columns)})")
