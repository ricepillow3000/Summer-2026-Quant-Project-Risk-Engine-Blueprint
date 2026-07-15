"""
Grit Zone - how much perseverance a name has shown, not how cheap or hot it is.

Concept:
Fear & Greed indices measure market MOOD. Grit measures something different:
whether an asset, when it gets knocked down, actually gets back up - and does
so consistently, across real crises, over its own history. There is no such
thing as a perfect stock; every name in a basket has drawdowns. Grit is not
about avoiding setbacks, it's about what happens after one.

Quant Deep Dive - three real, computable components, no vibes:
  1. Recovery:    of this asset's own historical drawdowns (>=5%), how many
                  did it actually claw back from, and how fast?
  2. Consistency: what fraction of rolling 1-year holding periods ended
                  positive? (steady long-run grinding forward, not one lucky
                  run)
  3. Resilience:  across the REAL historical crisis windows this asset lived
                  through (src.scenarios.HISTORICAL_REGIMES), how shallow was
                  the drawdown and how fast did it reclaim its pre-crisis
                  price afterward?

Honest limits:
  - The Grit Score is RELATIVE to the universe you're analyzing right now -
    each component is percentile-ranked against the other assets you chose,
    not against some absolute universal scale. A different basket produces
    different scores for the same ticker. That's disclosed in the UI, not
    hidden.
  - Assets with too little price history (< MIN_HISTORY_DAYS) are excluded
    from ranking rather than scored on thin data.
  - An asset that never traded through a given historical regime (later IPO)
    simply contributes no observation for that regime - same "excluded, not
    estimated" honesty as src.scenarios and src.liquidity.
  - Nothing here is a claim about a company's character. It's a description
    of its OWN historical price path: how it has behaved after past setbacks.
"""

import numpy as np
import pandas as pd

from src.scenarios import HISTORICAL_REGIMES

# Need enough history to say anything about multi-year drawdown recovery and
# rolling-window consistency - much longer than the 60-row floor risk.py uses
# for day-to-day VaR/CVaR.
MIN_HISTORY_DAYS = 400

# Trading days of drawdown from a running peak before we call it a real
# "episode" worth timing a recovery from (vs. day-to-day noise).
DEFAULT_DD_THRESHOLD = 0.05

# Trading days of rolling window for the consistency metric (~1 year).
DEFAULT_CONSISTENCY_WINDOW = 252

# How far past a crisis window we'll look for the price to reclaim its
# pre-crisis level before calling it "not recovered" (~2 trading years).
REGIME_RECOVERY_HORIZON_DAYS = 504


def drawdown_episodes(prices: pd.Series, threshold: float = DEFAULT_DD_THRESHOLD) -> pd.DataFrame:
    """
    Segment a price series into distinct drawdown episodes: peak -> trough ->
    recovery (or still-open if the series ends before recovering).

    An episode starts the first day the price falls more than `threshold`
    below its running all-time high, and closes the day the price reclaims
    that prior high. `days_to_recover` is measured in trading days from the
    trough, not calendar days.
    """
    idx = prices.index
    vals = prices.to_numpy(dtype=float)
    n = len(vals)
    episodes = []
    if n == 0:
        return pd.DataFrame(episodes)

    peak_val, peak_pos = vals[0], 0
    in_dd = False
    trough_val = trough_pos = None

    for i in range(1, n):
        v = vals[i]
        if in_dd:
            if v < trough_val:
                trough_val, trough_pos = v, i
            if v >= peak_val:
                episodes.append({
                    "peak_date": idx[peak_pos], "peak_value": peak_val,
                    "trough_date": idx[trough_pos], "trough_value": trough_val,
                    "recovery_date": idx[i],
                    "depth": trough_val / peak_val - 1.0,
                    "days_to_trough": trough_pos - peak_pos,
                    "days_to_recover": i - trough_pos,
                })
                in_dd = False
                peak_val, peak_pos = v, i
                continue
        if v > peak_val:
            peak_val, peak_pos = v, i
            continue
        if not in_dd:
            dd = v / peak_val - 1.0
            if dd <= -threshold:
                in_dd = True
                trough_val, trough_pos = v, i

    if in_dd:
        episodes.append({
            "peak_date": idx[peak_pos], "peak_value": peak_val,
            "trough_date": idx[trough_pos], "trough_value": trough_val,
            "recovery_date": None,
            "depth": trough_val / peak_val - 1.0,
            "days_to_trough": trough_pos - peak_pos,
            "days_to_recover": None,
        })

    return pd.DataFrame(episodes)


def recovery_stats(prices: pd.Series, threshold: float = DEFAULT_DD_THRESHOLD) -> dict:
    """Summarize an asset's own drawdown/recovery track record."""
    episodes = drawdown_episodes(prices, threshold)
    current_dd = float(prices.iloc[-1] / prices.cummax().iloc[-1] - 1.0)

    if episodes.empty:
        return {
            "n_episodes": 0, "n_recovered": 0, "pct_recovered": 1.0,
            "median_recovery_days": float("nan"), "still_underwater": False,
            "current_drawdown": current_dd,
        }

    recovered = episodes[episodes["recovery_date"].notna()]
    still_open = episodes[episodes["recovery_date"].isna()]
    return {
        "n_episodes": len(episodes),
        "n_recovered": len(recovered),
        "pct_recovered": len(recovered) / len(episodes),
        "median_recovery_days": (float(recovered["days_to_recover"].median())
                                 if len(recovered) else float("nan")),
        "still_underwater": len(still_open) > 0,
        "current_drawdown": current_dd,
    }


def rolling_consistency(prices: pd.Series, window: int = DEFAULT_CONSISTENCY_WINDOW) -> float:
    """
    Fraction of rolling `window`-day holding periods (any start day) that
    ended in a positive total return. 1.0 = every 1-year stretch you could
    have bought made money; 0.0 = none did. NaN if there isn't yet a full
    window of history.
    """
    if len(prices) <= window:
        return float("nan")
    roll_ret = (prices / prices.shift(window) - 1.0).dropna()
    if roll_ret.empty:
        return float("nan")
    return float((roll_ret > 0).mean())


def regime_drawdown_and_recovery(prices_full: pd.Series, start: str, end: str,
                                 recovery_horizon_days: int = REGIME_RECOVERY_HORIZON_DAYS) -> dict | None:
    """
    Max drawdown an asset suffered DURING a named historical crisis window,
    and whether/how fast it reclaimed its pre-crisis price afterward.

    Returns None if the asset has no price history in the window at all
    (later IPO, etc.) - same "excluded, not estimated" convention as
    src.scenarios.replay_returns.
    """
    window = prices_full.loc[start:end].dropna()
    if window.empty:
        return None

    pre_crisis_price = float(window.iloc[0])
    running_max = window.cummax()
    dd = window / running_max - 1.0
    max_dd = float(dd.min())

    after = prices_full.loc[window.index[-1]:].iloc[1:1 + recovery_horizon_days].dropna()
    recovered_at = after[after >= pre_crisis_price]
    recovery_days = int(after.index.get_loc(recovered_at.index[0])) + 1 if len(recovered_at) else None

    return {
        "max_drawdown": max_dd,
        "recovery_days": recovery_days,
        "pre_crisis_price": pre_crisis_price,
        "window_days": len(window),
    }


def regime_resilience(prices_full: pd.Series) -> dict:
    """Aggregate an asset's drawdown/recovery behavior across every named
    historical crisis window it actually has price data for."""
    results = {}
    for name, (s, e) in HISTORICAL_REGIMES.items():
        r = regime_drawdown_and_recovery(prices_full, s, e)
        if r is not None:
            results[name] = r

    if not results:
        return {"n_regimes": 0, "avg_max_drawdown": float("nan"),
                "pct_recovered_in_horizon": float("nan"), "regimes": results}

    dds = [v["max_drawdown"] for v in results.values()]
    recovered = [v["recovery_days"] is not None for v in results.values()]
    return {
        "n_regimes": len(results),
        "avg_max_drawdown": float(np.mean(dds)),
        "pct_recovered_in_horizon": float(np.mean(recovered)),
        "regimes": results,
    }


def _score01(raw: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Percentile-rank a metric to [0, 1] across the peer group; 1.0 = grittiest
    on this metric. A NaN (couldn't be computed - e.g. no drawdown episodes
    yet to time a recovery from) always scores 0.0: an unknown is never
    rewarded as if it were resilient.
    """
    x = raw if higher_is_better else -raw
    valid = x.dropna()
    out = pd.Series(0.0, index=raw.index)
    if len(valid) == 0:
        return out
    if len(valid) == 1:
        out.loc[valid.index] = 1.0
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True)
    return out


def grit_scores(tickers: list[str], prices: pd.DataFrame | None = None,
                dd_threshold: float = DEFAULT_DD_THRESHOLD,
                consistency_window: int = DEFAULT_CONSISTENCY_WINDOW) -> dict:
    """
    Composite Grit Score per ticker, ranked RELATIVE to the other tickers in
    `tickers` (there's no universal "grit" scale - only relative perseverance
    within the peer group you're actually comparing).

    Args:
        tickers: universe to score.
        prices: optional pre-fetched full-history price DataFrame (unaligned,
            NaN before each asset's inception is fine). If omitted, pulls
            full history via src.ingestion.fetch_prices(period="max",
            align=False) - the same approach src.scenarios uses for regime
            replay, so different IPO dates don't truncate everyone.

    Returns:
        {"scores": DataFrame indexed by ticker, sorted by grit_score desc,
         "excluded": tickers dropped for having < MIN_HISTORY_DAYS of data}
    """
    if prices is None:
        from src.ingestion import fetch_prices
        prices = fetch_prices(tickers, period="max", align=False)

    rows = {}
    excluded = []
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < MIN_HISTORY_DAYS:
            excluded.append(t)
            continue
        rec = recovery_stats(s, dd_threshold)
        cons = rolling_consistency(s, consistency_window)
        res = regime_resilience(prices[t])
        rows[t] = {
            "history_days": len(s),
            "pct_recovered": rec["pct_recovered"],
            "median_recovery_days": rec["median_recovery_days"],
            "still_underwater": rec["still_underwater"],
            "current_drawdown": rec["current_drawdown"],
            "consistency": cons,
            "n_regimes_survived": res["n_regimes"],
            "avg_regime_drawdown": res["avg_max_drawdown"],
            "pct_regime_recovered": res["pct_recovered_in_horizon"],
        }

    if not rows:
        return {"scores": pd.DataFrame(), "excluded": excluded}

    df = pd.DataFrame(rows).T

    recovery_component = (
        0.6 * _score01(df["pct_recovered"], higher_is_better=True)
        + 0.4 * _score01(df["median_recovery_days"], higher_is_better=False)
    )
    consistency_component = _score01(df["consistency"], higher_is_better=True)
    resilience_component = (
        0.5 * _score01(df["avg_regime_drawdown"], higher_is_better=True)   # less negative = better
        + 0.5 * _score01(df["pct_regime_recovered"], higher_is_better=True)
    )

    df["recovery_score"] = (recovery_component * 100).round(1)
    df["consistency_score"] = (consistency_component * 100).round(1)
    df["resilience_score"] = (resilience_component * 100).round(1)
    df["grit_score"] = (
        (recovery_component + consistency_component + resilience_component) / 3 * 100
    ).round(1)

    return {"scores": df.sort_values("grit_score", ascending=False), "excluded": excluded}


if __name__ == "__main__":
    from src.ingestion import DEFAULT_UNIVERSE

    result = grit_scores(DEFAULT_UNIVERSE)
    scores = result["scores"]
    print("--- Grit Zone (relative to this universe) ---")
    cols = ["grit_score", "recovery_score", "consistency_score", "resilience_score",
           "history_days", "n_regimes_survived", "pct_recovered", "consistency",
           "current_drawdown"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(scores[cols])
    if result["excluded"]:
        print(f"\nExcluded (insufficient history < {MIN_HISTORY_DAYS}d): "
              f"{', '.join(result['excluded'])}")
