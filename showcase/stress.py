import numpy as np
import pandas as pd

# real crash windows. we replay what actually happened, no made-up shocks
CRISES = {
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


# the real daily moves inside a crash window. stocks that weren't around sit it out
def replay(prices, start, end, max_gap=7):
    win = prices.loc[start:end].dropna(axis=1, how="all")
    win = win.loc[:, (win > 0).sum() == win.notna().sum()]  # price hit zero or below? % moves break, so it's out
    if not win.empty:
        first = win.notna().idxmax()
        win = win.loc[:, first <= first.min() + pd.Timedelta(days=max_gap)]  # showed up late = not invited
    win = win.dropna()
    if win.empty:
        return win
    lead = prices.loc[:win.index[0], win.columns]
    lead = lead[lead.index < win.index[0]].dropna().tail(1)  # the day before, so day 1's drop counts too
    if (not lead.empty and (win.index[0] - lead.index[-1]).days <= max_gap
            and (lead.to_numpy() > 0).all()):
        win = pd.concat([lead, win])
    return win.pct_change().dropna()


# how deep it fell inside the crash, and did it get back to its pre-crash price within 2 years
def crisis_dip(p, start, end, horizon=504):
    win = p.loc[start:end].dropna()
    if win.empty:
        return None  # wasn't trading yet
    after = p.loc[win.index[-1]:].iloc[1:1 + horizon].dropna()
    return float((win / win.cummax() - 1).min()), bool((after >= win.iloc[0]).any())


# every 5%+ dip: trading days from the bottom back to the old high. None = still underwater
def dips(p, threshold=0.05):
    v = p.to_numpy(dtype=float)
    out, down, low, low_i = [], False, None, None
    if v.size == 0:
        return out
    peak = v[0]
    for i in range(1, v.size):
        if down:
            if v[i] < low:
                low, low_i = v[i], i
            if v[i] >= peak:
                out.append(i - low_i)
                down, peak = False, v[i]
                continue
        if v[i] > peak:
            peak = v[i]
        elif not down and v[i] / peak - 1 <= -threshold:
            down, low, low_i = True, v[i], i
    return out + [None] if down else out


# rank 0..1 vs the rest of the basket. unknown = 0, no free points
def rank(x, up=True):
    x = x if up else -x
    ok = x.dropna()
    out = pd.Series(0.0, index=x.index)
    if len(ok) == 1:
        out[ok.index] = 1.0
    elif len(ok):
        out[ok.index] = ok.rank(pct=True)
    return out


# GRIT: does it bounce back? recovery + consistency + crash resilience, ranked vs your basket
def grit(prices, min_days=400, year=252):
    rows = {}
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < min_days:
            continue  # too new to judge
        d = dips(s)
        done = [x for x in d if x is not None]
        roll = (s / s.shift(year) - 1).dropna()
        hits = [c for c in (crisis_dip(prices[t], a, b) for a, b in CRISES.values()) if c]
        rows[t] = {
            "recovered": len(done) / len(d) if d else np.nan,
            "recover_days": float(np.median(done)) if done else np.nan,
            "consistency": float((roll > 0).mean()) if len(s) > year and len(roll) else np.nan,  # % of 1-yr holds that made money
            "crisis_dd": float(np.mean([h[0] for h in hits])) if hits else np.nan,
            "crisis_back": float(np.mean([h[1] for h in hits])) if hits else np.nan,
        }
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T.astype(float)
    rec = 0.6 * rank(df["recovered"]) + 0.4 * rank(df["recover_days"], up=False)
    res = 0.5 * rank(df["crisis_dd"]) + 0.5 * rank(df["crisis_back"])
    df["grit"] = ((rec + rank(df["consistency"]) + res) / 3 * 100).round(1)  # equal thirds, out of 100
    return df.sort_values("grit", ascending=False)
