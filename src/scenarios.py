"""
Crisis replays: rerun real crashes on today's book.

How it works:
- a made-up shock ("say it drops 30%") is a guess you pick. a replay is better:
  it plays back the ACTUAL daily returns from a real crisis, so the book gets
  that period's real correlation blowup, vol spike and drawdown path all at
  once, not as separate knobs. it's how risk desks ask "what if 2008 hit us now?"
- the monte carlo draws from the crisis window's own returns. pulling a 1-year
  path out of a short, nasty window piles that regime up on purpose - that's
  the stress.

Honest limits:
- a name that didn't trade in the window (later IPO, new ETF, an FX pair yahoo
  has no history for) sits that scenario out, and we say so. same for one that
  lists more than MAX_LEAD_GAP_DAYS after the window opens (it would cut
  everyone else's window short) and one priced at or below zero in it (simple
  returns break across a sign change).
- sizes are whatever really happened; nothing is hand-tuned.
"""

import pandas as pd
from src.ingestion import fetch_prices

# (start, end) for big and small risk regimes
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

# how old the close before the window can be and still count as "yesterday":
# a long weekend plus a holiday. older than that and we don't use it
MAX_LEAD_GAP_DAYS = 7


def replay_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Daily returns for the basket over a crisis window.

    pulls full history WITHOUT lining dates up across names (so a late IPO
    doesn't cut everyone short), slices the window, drops names with no data,
    a price <= 0, or a first price more than MAX_LEAD_GAP_DAYS after the
    earliest one, then lines up the survivors.

    the window's first day counts too. pct_change() inside the slice has no
    close before day one, so it used to lose it - and in a crash that's often
    the biggest move (COVID opens on 2020-02-19, the exact top). so we glue
    on the last close before the window, take returns, and drop that row.
    """
    prices = fetch_prices(tickers, period="max", align=False)
    window = prices.loc[start:end]
    window = window.dropna(axis=1, how="all")  # name not around in this window
    # returns break across a sign change: WTI's -37.63 settle (2020) turned a
    # +47.64 $/bbl bounce into a fake -126.6% "loss"
    window = window.loc[:, (window > 0).sum() == window.notna().sum()]
    # a name that lists partway in would drag everyone's start to its listing
    # (IVV/IWM cut Mar-May 2000 off dot-com), so it counts as absent
    if not window.empty:
        firsts = window.notna().idxmax()
        window = window.loc[:, firsts <= firsts.min() + pd.Timedelta(days=MAX_LEAD_GAP_DAYS)]
    window = window.dropna()                    # line up the survivors
    if window.empty:
        return window

    first = window.index[0]
    lead = prices.loc[:first, window.columns]
    lead = lead[lead.index < first].dropna().tail(1)
    # only use it if it really is the day before. a close from weeks back would
    # fake a multi-week move as one day - worse than losing the bar, so then the
    # window just starts on day two. a lead close <= 0 is out too (sign change).
    if (not lead.empty and (first - lead.index[-1]).days <= MAX_LEAD_GAP_DAYS
            and (lead.to_numpy() > 0).all()):
        window = pd.concat([lead, window])
    return window.pct_change().dropna()


if __name__ == "__main__":
    from src.ingestion import DEFAULT_UNIVERSE
    for name, (s, e) in HISTORICAL_REGIMES.items():
        r = replay_returns(DEFAULT_UNIVERSE, s, e)
        print(f"{name:32s}: {r.shape[0]:3d} days, {r.shape[1]} assets "
              f"({', '.join(r.columns)})")
