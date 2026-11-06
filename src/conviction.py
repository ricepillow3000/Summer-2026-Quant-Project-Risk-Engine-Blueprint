"""
Crisis Conviction - the hardest trade, quantified.

Concept:
Buying during a crisis is emotionally brutal: the brain treats financial loss
like a physical threat, so the scariest day is precisely when acting feels
impossible. This module doesn't argue with that feeling - it prices it. For
every named crisis window the engine already replays (src.scenarios), it
computes what ACTUALLY happened next, from real prices: if you had bought
the benchmark at the pre-crisis PEAK (the worst-timed entry possible) or at
the TROUGH (the scariest single day), where did you stand 1 and 3 years later?

Quant Deep Dive:
- Entry points are defined mechanically: the peak is the window's running
  maximum before its minimum; the trough is the window minimum. No
  discretion, no curve fitting.
- Forward returns are simple point-to-point total returns on adjusted closes
  (dividends/splits already folded in by auto_adjust), measured in TRADING
  days (252 ≈ 1y, 756 ≈ 3y).

Honest limits:
- Nobody can time the exact trough. The trough row quantifies the DIRECTION
  of the edge, not an executable strategy - that's why the peak row (the
  worst possible timing) is shown alongside it.
- A crisis too recent to have a full forward horizon is excluded from that
  horizon, not extrapolated.
- All of it is hindsight on one benchmark. Educational analysis, not
  investment advice.
"""

import pandas as pd

from src.ingestion import fetch_prices
from src.covariance import TRADING_DAYS
from src.scenarios import HISTORICAL_REGIMES

BENCHMARK = "SPY"
HORIZONS = {"1y later": TRADING_DAYS, "3y later": 3 * TRADING_DAYS}

# How far past the trough we search for a reclaim of the pre-crisis peak
# before calling it "not recovered" within the horizon (~3 trading years).
RECOVERY_HORIZON_DAYS = 3 * TRADING_DAYS

MIN_WINDOW_ROWS = 5  # fewer rows than this and the window isn't usable


def _peak_trough(window: pd.Series):
    """Mechanical crash anatomy: trough = window minimum; peak = the running
    maximum BEFORE that trough (a crash falls peak-first, trough-second).

    Returns (None, None) when the window never declined - its minimum IS its
    first bar, so there is no peak-to-trough fall to measure. "Never fell" and
    "fell and was reclaimed in zero days" are different statements and the
    callers must not render one as the other.
    """
    trough_date = window.idxmin()
    peak_date = window.loc[:trough_date].idxmax()
    if peak_date == trough_date:
        return None, None
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
    One row per named crisis the series traded through: crash depth, the
    trading days from the trough back to the pre-crisis peak, plus forward
    returns from the pre-crisis peak (worst-timed entry) and from the trough
    (scariest day), at each horizon in HORIZONS.
    """
    prices = prices.dropna()
    rows = []
    for name, (s, e) in HISTORICAL_REGIMES.items():
        window = prices.loc[s:e]
        if len(window) < MIN_WINDOW_ROWS:
            continue
        peak_date, trough_date = _peak_trough(window)
        if peak_date is None:
            continue            # never declined inside this window: no crash
        peak_val = float(window.loc[peak_date])
        trough_val = float(window.loc[trough_date])
        # The way back: trading days from the trough to the first close at or
        # above the peak before the fall, within RECOVERY_HORIZON_DAYS - None
        # (not back within that span, or the data ends first) is shown as
        # such, never extrapolated. The page's case studies read these.
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
    """Headline numbers: across the crises with a full horizon, how often was
    a buyer whole - and by how much, at the median?"""
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
    """Convenience bundle for the UI: benchmark forward-return table + summary,
    from live Yahoo data."""
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
