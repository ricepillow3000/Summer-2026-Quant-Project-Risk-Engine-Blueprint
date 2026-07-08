"""
Crisis Conviction — the hardest trade, quantified.

Concept:
Buying during a crisis is emotionally brutal: the brain treats financial loss
like a physical threat, so the scariest day is precisely when acting feels
impossible. This module doesn't argue with that feeling — it prices it. For
every named crisis window the engine already replays (src.scenarios), it
computes what ACTUALLY happened next, from real prices:

  1. Forward returns — if you had bought the benchmark at the pre-crisis PEAK
     (the worst-timed entry possible) or at the TROUGH (the scariest single
     day), where did you stand 1 and 3 years later?
  2. Recovery race — did a basket of heavy compute/AI-infrastructure
     investors reclaim its pre-crisis level faster or slower than the broad
     market, crisis by crisis?

Quant Deep Dive:
- Entry points are defined mechanically: the peak is the window's running
  maximum before its minimum; the trough is the window minimum. No
  discretion, no curve fitting.
- Forward returns are simple point-to-point total returns on adjusted closes
  (dividends/splits already folded in by auto_adjust), measured in TRADING
  days (252 ≈ 1y, 756 ≈ 3y).
- The recovery race normalizes each surviving basket member to 1.0 at the
  window start and averages — an equal-weight composite, same construction
  as the benchmark's own normalized path, so the two race from the same line.

Honest limits:
- Nobody can time the exact trough. The trough row quantifies the DIRECTION
  of the edge, not an executable strategy — that's why the peak row (the
  worst possible timing) is shown alongside it.
- A crisis too recent to have a full forward horizon is excluded from that
  horizon, not extrapolated.
- The "AI capex" basket is today's label. In 2008 these same tickers were
  simply large-cap tech; the record shown is theirs regardless of the label.
  The thesis that AI investment speeds recovery is a HYPOTHESIS — this
  module shows the historical record and lets the viewer judge.
- All of it is hindsight on one benchmark and one basket. Educational
  analysis, not investment advice.
"""

import numpy as np
import pandas as pd

from src.ingestion import fetch_prices
from src.scenarios import HISTORICAL_REGIMES

BENCHMARK = "SPY"
HORIZONS = {"1y later": 252, "3y later": 756}

# Names whose capital spending is dominated by compute / AI infrastructure
# today. Fixed list, disclosed in the UI; their PAST record is shown as-is.
AI_CAPEX_BASKET = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO"]

# How far past the trough we search for a reclaim of the pre-crisis peak
# before calling it "not recovered" within the horizon (~3 trading years).
RECOVERY_HORIZON_DAYS = 756

MIN_WINDOW_ROWS = 5  # fewer rows than this and the window isn't usable


def _peak_trough(window: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Mechanical crash anatomy: trough = window minimum; peak = the running
    maximum BEFORE that trough (a crash falls peak-first, trough-second)."""
    trough_date = window.idxmin()
    peak_date = window.loc[:trough_date].idxmax()
    return peak_date, trough_date


def _forward_return(prices: pd.Series, date: pd.Timestamp, horizon: int):
    """Point-to-point return `horizon` TRADING days after `date`, or None if
    the series ends first (excluded, not extrapolated)."""
    pos = prices.index.get_loc(date)
    if pos + horizon >= len(prices):
        return None
    return float(prices.iloc[pos + horizon] / prices.iloc[pos] - 1.0)


def _days_to_reclaim(prices: pd.Series, trough_date: pd.Timestamp,
                     level: float, horizon: int = RECOVERY_HORIZON_DAYS):
    """Trading days from the trough until the price first closes at or above
    `level`, or None if it doesn't within `horizon` days (or data ends)."""
    pos = prices.index.get_loc(trough_date)
    after = prices.iloc[pos:pos + horizon + 1]
    hit = after[after >= level]
    if hit.empty:
        return None
    return int(after.index.get_loc(hit.index[0]))


def crisis_forward_returns(prices: pd.Series) -> pd.DataFrame:
    """
    One row per named crisis the series traded through: crash depth, plus
    forward returns from the pre-crisis peak (worst-timed entry) and from
    the trough (scariest day), at each horizon in HORIZONS.
    """
    prices = prices.dropna()
    rows = []
    for name, (s, e) in HISTORICAL_REGIMES.items():
        window = prices.loc[s:e]
        if len(window) < MIN_WINDOW_ROWS:
            continue
        peak_date, trough_date = _peak_trough(window)
        peak_val = float(window.loc[peak_date])
        trough_val = float(window.loc[trough_date])
        row = {
            "crisis": name,
            "trough_date": trough_date.date(),
            "depth": trough_val / peak_val - 1.0,
        }
        for label, h in HORIZONS.items():
            row[f"peak_{label}"] = _forward_return(prices, peak_date, h)
            row[f"trough_{label}"] = _forward_return(prices, trough_date, h)
        rows.append(row)
    return pd.DataFrame(rows)


def conviction_summary(table: pd.DataFrame) -> dict:
    """Headline numbers: across the crises with a full horizon, how often was
    a buyer whole — and by how much, at the median?"""
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


def _composite(prices: pd.DataFrame, start) -> pd.Series:
    """Equal-weight composite: each surviving column normalized to 1.0 at its
    first close on/after `start`, then averaged. NaN columns are dropped —
    excluded, not estimated."""
    sliced = prices.loc[start:]
    if sliced.empty:
        return pd.Series(dtype=float)
    # A member must already be trading AT the window start to join the race —
    # names that IPO later are excluded from this crisis, not back-filled.
    alive = sliced.columns[sliced.iloc[:MIN_WINDOW_ROWS].notna().all()]
    sliced = sliced[alive].dropna()
    if sliced.empty or sliced.shape[1] == 0:
        return pd.Series(dtype=float)
    return (sliced / sliced.iloc[0]).mean(axis=1)


def recovery_race(basket: list[str] | None = None,
                  benchmark: str = BENCHMARK) -> pd.DataFrame:
    """
    Crisis by crisis: trading days for the equal-weight basket composite vs
    the benchmark to reclaim their own pre-crisis peaks after the trough.
    None = did not recover within RECOVERY_HORIZON_DAYS (shown, not hidden).
    """
    basket = basket or AI_CAPEX_BASKET
    px = fetch_prices(basket + [benchmark], period="max", align=False)
    bench = px[benchmark].dropna()
    members = px[[c for c in px.columns if c != benchmark]]

    rows = []
    for name, (s, e) in HISTORICAL_REGIMES.items():
        bwin = bench.loc[s:e]
        if len(bwin) < MIN_WINDOW_ROWS:
            continue
        comp = _composite(members, s)
        cwin = comp.loc[s:e]
        row = {"crisis": name}

        # Benchmark leg
        b_peak, b_trough = _peak_trough(bwin)
        row["bench_days"] = _days_to_reclaim(
            bench, b_trough, float(bwin.loc[b_peak]))

        # Basket leg — needs enough members trading through the window
        if len(cwin) >= MIN_WINDOW_ROWS:
            c_peak, c_trough = _peak_trough(cwin)
            row["basket_days"] = _days_to_reclaim(
                comp, c_trough, float(cwin.loc[c_peak]))
            # Same aliveness rule as _composite: trading at the window start.
            mwin = members.loc[s:e]
            row["basket_members"] = int(
                mwin.iloc[:MIN_WINDOW_ROWS].notna().all().sum())
        else:
            row["basket_days"] = None
            row["basket_members"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def load_conviction(benchmark: str = BENCHMARK) -> dict:
    """Convenience bundle for the UI: benchmark forward-return table + summary
    + the AI-capex recovery race, all from live Yahoo data."""
    bench_px = fetch_prices([benchmark], period="max", align=False)[benchmark]
    table = crisis_forward_returns(bench_px)
    return {
        "benchmark": benchmark,
        "table": table,
        "summary": conviction_summary(table),
        "race": recovery_race(benchmark=benchmark),
    }


if __name__ == "__main__":
    bundle = load_conviction()
    print(bundle["table"].to_string(index=False))
    print(bundle["summary"])
    print(bundle["race"].to_string(index=False))
