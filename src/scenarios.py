"""
Historical regime replication.

Quant Deep Dive:
- Parametric stress ("apply a -30% drawdown") is an approximation you impose.
- Regime *replication* is stronger: it replays the ACTUAL daily returns from a
  real crisis window, so the simulation inherits that period's true correlation
  breakdown, volatility spike, and drawdown path - jointly, not as separate
  knobs. This is how risk desks ask "what if 2008 happened to today's book?"
- We Monte-Carlo from the crisis window's empirical returns. Bootstrapping a
  1-year path from a short, severe window deliberately concentrates that
  regime - that is the stress.

Honest limits:
- An asset that didn't trade during the window (later IPO, new ETF, an FX pair
  Yahoo lacks history for) is excluded from that scenario, and we say so.
- Magnitudes are whatever actually happened; nothing is hand-tuned.
"""

import pandas as pd
from src.ingestion import fetch_prices

# (start, end) windows for notable risk regimes - major and minor.
HISTORICAL_REGIMES = {
    "Dot-com crash (2000-02)": ("2000-03-01", "2001-04-30"),
    "Global Financial Crisis (2008)": ("2008-09-01", "2009-03-31"),
    "Flash Crash (May 2010)": ("2010-05-03", "2010-05-21"),
    "Euro debt crisis (2011)": ("2011-07-01", "2011-10-31"),
    "China devaluation (Aug 2015)": ("2015-08-01", "2015-09-30"),
    "Volmageddon (Feb 2018)": ("2018-01-26", "2018-02-28"),
    "Q4 2018 selloff": ("2018-10-01", "2018-12-31"),
    "COVID-19 crash (2020)": ("2020-02-19", "2020-04-30"),
    "2022 bear market": ("2022-01-01", "2022-10-31"),
    "SVB banking stress (Mar 2023)": ("2023-03-01", "2023-03-31"),
}

# How stale the pre-window close may be and still count as "the previous
# session": a long weekend plus a holiday. Beyond this the lead-in is refused.
MAX_LEAD_GAP_DAYS = 7


def replay_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Daily returns for the requested universe over a historical window.

    Pulls full history WITHOUT cross-asset date alignment (so different IPO
    dates don't truncate everyone), slices the window, drops assets with no
    data in it, then aligns the surviving subset within the window.

    The window's OPENING day is included. pct_change() taken inside the slice
    has no prior close to measure the first bar against, so it used to drop
    that bar - and in a crash window the opening day is frequently the largest
    single move in the whole replay (COVID-19 opens 2020-02-19, the exact top).
    The last complete close BEFORE the window is re-attached, returns are
    taken, and the lead-in falls away with its own NaN row.
    """
    prices = fetch_prices(tickers, period="max", align=False)
    window = prices.loc[start:end]
    window = window.dropna(axis=1, how="all")  # drop assets absent in this window
    window = window.dropna()                    # align the survivors over the window
    if window.empty:
        return window

    first = window.index[0]
    lead = prices.loc[:first, window.columns]
    lead = lead[lead.index < first].dropna().tail(1)
    # Only usable if it really is the PREVIOUS session. A stale close from
    # weeks earlier - a name that had not traded, or a gap in the feed - would
    # manufacture a multi-week move and label it one day, which is worse than
    # losing the bar. Beyond the guard the old behaviour stands and the window
    # opens on day two, honestly short one observation.
    if not lead.empty and (first - lead.index[-1]).days <= MAX_LEAD_GAP_DAYS:
        window = pd.concat([lead, window])
    return window.pct_change().dropna()


if __name__ == "__main__":
    from src.ingestion import DEFAULT_UNIVERSE
    for name, (s, e) in HISTORICAL_REGIMES.items():
        r = replay_returns(DEFAULT_UNIVERSE, s, e)
        print(f"{name:32s}: {r.shape[0]:3d} days, {r.shape[1]} assets "
              f"({', '.join(r.columns)})")
