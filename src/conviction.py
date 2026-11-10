"""
Crisis Conviction - the hardest trade, in numbers.

The idea:
buying in a crisis is brutal - your brain reads a money loss like a physical
threat, so the scariest day is exactly when acting feels impossible. we don't
argue with that, we price it. for every crisis the engine already replays
(src.scenarios) we check what ACTUALLY happened next, from real prices: if you
bought the benchmark at the pre-crisis PEAK (worst timing possible) or at the
TROUGH (scariest day), where were you 1 and 3 years later?

How it works:
- entries are picked mechanically: trough = the window's low, peak = the high
  before that low. no judgement, no curve fitting.
- forward returns are plain point-to-point returns on adjusted closes
  (dividends/splits already in via auto_adjust), counted in TRADING days
  (252 ~ 1y, 756 ~ 3y).

Honest limits:
- nobody times the exact bottom. the trough row shows which WAY the edge
  points, not a trade you could run - that's why the peak row (worst timing)
  sits right next to it.
- a crisis too recent for a full horizon is left out of that horizon, not
  guessed.
- it's all hindsight on one benchmark. educational, not investment advice.
"""

import pandas as pd

from src.ingestion import fetch_prices
from src.covariance import TRADING_DAYS
from src.scenarios import HISTORICAL_REGIMES

BENCHMARK = "SPY"
HORIZONS = {"1y later": TRADING_DAYS, "3y later": 3 * TRADING_DAYS}

# how far past the trough we look for the old peak before calling it
# "not recovered" (~3 trading years)
RECOVERY_HORIZON_DAYS = 3 * TRADING_DAYS

MIN_WINDOW_ROWS = 5  # fewer rows than this and the window is useless


def _peak_trough(window: pd.Series):
    """trough = the window's low; peak = the high BEFORE it (crashes fall
    peak first, trough second).

    (None, None) if the window never fell - its low is its first bar, so
    there's no fall to measure. "never fell" and "fell and bounced back in 0
    days" aren't the same thing, and callers must not show one as the other.
    """
    trough_date = window.idxmin()
    peak_date = window.loc[:trough_date].idxmax()
    if peak_date == trough_date:
        return None, None
    return peak_date, trough_date


def _forward_return(prices: pd.Series, date: pd.Timestamp, horizon: int):
    """Return `horizon` TRADING days after `date`, or None if the data runs
    out first (left out, not guessed)."""
    pos = prices.index.get_loc(date)
    if pos + horizon >= len(prices):
        return None
    return float(prices.iloc[pos + horizon] / prices.iloc[pos] - 1.0)


def _days_to_reclaim(prices: pd.Series, trough_date: pd.Timestamp,
                     level: float, horizon: int = RECOVERY_HORIZON_DAYS):
    """Trading days from the trough to the first close >= `level`, or None
    if that doesn't happen within `horizon` days (or the data ends)."""
    pos = prices.index.get_loc(trough_date)
    after = prices.iloc[pos:pos + horizon + 1]
    hit = after[after >= level]
    if hit.empty:
        return None
    return int(after.index.get_loc(hit.index[0]))


def crisis_forward_returns(prices: pd.Series) -> pd.DataFrame:
    """
    One row per crisis the series traded through: how deep it fell, trading
    days from the trough back to the old peak, and forward returns from the
    peak (worst entry) and the trough (scariest day) at each HORIZONS step.
    """
    prices = prices.dropna()
    rows = []
    for name, (s, e) in HISTORICAL_REGIMES.items():
        window = prices.loc[s:e]
        if len(window) < MIN_WINDOW_ROWS:
            continue
        peak_date, trough_date = _peak_trough(window)
        if peak_date is None:
            continue            # never fell in this window: no crash
        peak_val = float(window.loc[peak_date])
        trough_val = float(window.loc[trough_date])
        # the way back: days from trough to the first close at/above the old
        # peak. None = not back in time (or data ends), shown as-is, never
        # guessed. the page's case studies read these.
        back = _days_to_reclaim(prices, trough_date, peak_val)
        row = {
            "crisis": name,
            "trough_date": trough_date.date(),
            "depth": trough_val / peak_val - 1.0,
            "recovery_days": back,
            "recovery_date": (None if back is None else
                              prices.index[prices.index.get_loc(trough_date) + back].date()),
        }
        for label, h in HORIZONS.items():
            row[f"peak_{label}"] = _forward_return(prices, peak_date, h)
            row[f"trough_{label}"] = _forward_return(prices, trough_date, h)
        rows.append(row)
    return pd.DataFrame(rows)


def conviction_summary(table: pd.DataFrame) -> dict:
    """Headline numbers: over crises with a full horizon, how often was a
    buyer in the green, and by how much at the median?"""
    out = {"n_crises": int(len(table))}
    for entry in ("trough", "peak"):
        for label in HORIZONS:
            col = table.get(f"{entry}_{label}")
            vals = col.dropna() if col is not None else pd.Series(dtype=float)
            key = f"{entry}_{label}".replace(" ", "_")
            out[key] = {
                "n": int(len(vals)),
                "pct_positive": float((vals > 0).mean()) if len(vals) else None,
                "median": float(vals.median()) if len(vals) else None,
            }
    return out


def load_conviction(benchmark: str = BENCHMARK) -> dict:
    """Everything the UI needs: the benchmark's table + summary, from live
    yahoo data."""
    bench_px = fetch_prices([benchmark], period="max", align=False)[benchmark]
    table = crisis_forward_returns(bench_px)
    return {
        "benchmark": benchmark,
        "table": table,
        "summary": conviction_summary(table),
    }


if __name__ == "__main__":
    bundle = load_conviction()
    print(bundle["table"].to_string(index=False))
    print(bundle["summary"])
