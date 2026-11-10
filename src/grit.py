"""
grit zone - when a name gets knocked down, does it get back up?

fear & greed measures mood. grit measures what happens after a fall. every
stock has drawdowns, so this is about the bounce, not dodging the hit.

how the math works (three parts, all from the name's own price history):
  1. recovery:    of its drawdowns of 5%+, how many did it win back, and how
                  fast (median trading days from the low back to the old high)?
  2. consistency: share of rolling 1-year holds that ended positive.
  3. resilience:  in each real crisis it traded through
                  (src.scenarios.HISTORICAL_REGIMES), how deep was the fall and
                  did it reclaim its pre-crisis price within ~2 years?
each part is percentile-ranked against the other names in the basket, then
averaged into one score out of 100.

honest limits:
  - scores are relative to the basket you picked: swap the basket and the
    same ticker scores differently. the ui says so.
  - under MIN_HISTORY_DAYS of data = excluded, not scored on thin data.
  - a crisis a name never traded through (later ipo) just doesn't count for
    it - excluded, not estimated, same as src.scenarios and src.liquidity.
  - it describes a price path, not a company's character.
"""

import numpy as np
import pandas as pd

from src.scenarios import HISTORICAL_REGIMES
from src.covariance import TRADING_DAYS

# need years of data to judge recoveries and rolling 1y windows
# (way more than the 60 rows risk.py needs for var/cvar)
MIN_HISTORY_DAYS = 400

# a dip has to sit 5% under the high to count as a drawdown, not noise
DEFAULT_DD_THRESHOLD = 0.05

# rolling window for consistency (~1 trading year)
DEFAULT_CONSISTENCY_WINDOW = TRADING_DAYS

# how long after a crisis we wait for the pre-crisis price (~2 trading years)
REGIME_RECOVERY_HORIZON_DAYS = 504


def drawdown_episodes(prices: pd.Series, threshold: float = DEFAULT_DD_THRESHOLD) -> pd.DataFrame:
    """
    split a price series into drawdowns: peak -> trough -> recovery (or still
    open if the data ends first). one starts the first day price is more than
    `threshold` under its running high and ends the day it gets that high
    back. days are trading days; days_to_recover counts from the trough.
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

    def episode(i):  # i = recovery day, None if still underwater
        return {
            "peak_date": idx[peak_pos], "peak_value": peak_val,
            "trough_date": idx[trough_pos], "trough_value": trough_val,
            "recovery_date": None if i is None else idx[i],
            "depth": trough_val / peak_val - 1.0,
            "days_to_trough": trough_pos - peak_pos,
            "days_to_recover": None if i is None else i - trough_pos,
        }

    for i in range(1, n):
        v = vals[i]
        if in_dd:
            if v < trough_val:
                trough_val, trough_pos = v, i
            if v >= peak_val:
                episodes.append(episode(i))
                in_dd = False
                peak_val, peak_pos = v, i
        elif v > peak_val:
            peak_val, peak_pos = v, i
        elif v / peak_val - 1.0 <= -threshold:
            in_dd = True
            trough_val, trough_pos = v, i

    if in_dd:
        episodes.append(episode(None))
    return pd.DataFrame(episodes)


def recovery_stats(prices: pd.Series, threshold: float = DEFAULT_DD_THRESHOLD) -> dict:
    """a name's own drawdown/recovery track record."""
    episodes = drawdown_episodes(prices, threshold)
    current_dd = float(prices.iloc[-1] / prices.cummax().iloc[-1] - 1.0)

    if episodes.empty:
        # no drawdowns = unknown record, not a perfect one. nan -> _score01 gives 0
        return {
            "n_episodes": 0, "n_recovered": 0, "pct_recovered": float("nan"),
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
    """share of every `window`-day hold (any start day) that ended up positive.
    1.0 = every 1y stretch made money, 0.0 = none did, nan = not a full window yet."""
    if len(prices) <= window:
        return float("nan")
    roll_ret = (prices / prices.shift(window) - 1.0).dropna()
    if roll_ret.empty:
        return float("nan")
    return float((roll_ret > 0).mean())


def regime_drawdown_and_recovery(prices_full: pd.Series, start: str, end: str,
                                 recovery_horizon_days: int = REGIME_RECOVERY_HORIZON_DAYS) -> dict | None:
    """worst drawdown inside one crisis window, and how many trading days after
    it the price took to get back to where the window started. None if the
    name has no prices in the window at all (excluded, not estimated)."""
    window = prices_full.loc[start:end].dropna()
    if window.empty:
        return None

    pre_crisis_price = float(window.iloc[0])
    max_dd = float((window / window.cummax() - 1.0).min())

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
    """roll up drawdown/recovery over every crisis the name has data for."""
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
    """percentile-rank a metric to [0, 1] across the basket, 1.0 = grittiest.
    nan (couldn't be measured) always gets 0 - an unknown never gets rewarded."""
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
    grit score per ticker, ranked against the other `tickers` (there's no
    universal grit scale, only who held up best in this group).

    `prices`: optional full-history closes, unaligned (nan before each ipo is
    fine). if missing we pull fetch_prices(period="max", align=False), same as
    src.scenarios, so a young name doesn't cut everyone's history short.

    returns {"scores": df sorted by grit_score desc,
             "excluded": tickers with < MIN_HISTORY_DAYS of data}
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

    # these weights are a stated judgement call, nothing fits or tunes them.
    # recovery leans 60/40 to "did it come back at all" over "how fast" (never
    # coming back is a different animal). resilience splits 50/50.
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
