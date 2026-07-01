"""
Data quality — automated validation gate on ingest, not a one-time eyeball check.

Quant Deep Dive:
A risk number computed on bad data is worse than no risk number — it's a wrong
answer with a confident-looking chart attached. Production data-engineering
teams don't trust a feed just because it returned 200 OK; they run a schema
and sanity gate on every pull and refuse to silently proceed if it fails. This
module is that gate, hand-rolled (no new dependency) to match this project's
existing style of transparent, from-scratch statistical checks.

Checks run (each PASS / WARN / FAIL):
  - Schema: numeric dtype, DatetimeIndex, no duplicate dates, dates sorted.
  - Positivity: no zero or negative prices (a data-feed corruption signal).
  - Coverage: enough rows to trust downstream risk math (reuses the same
    MIN_ROWS floor as src.ingestion).
  - Staleness: last row not too old (reuses src.ingestion.data_health logic).
  - Extreme moves: single-day returns beyond a sanity threshold are FLAGGED
    for review (WARN), not silently dropped or silently trusted — a real
    circuit breaker day (e.g. COVID crash) SHOULD warn; that's the point.
  - Gaps: trading-day coverage vs. the full business-day calendar for the
    window, flagging unexplained holes distinct from normal weekends/holidays.

Honest limits: this validates STRUCTURE and PLAUSIBILITY, not truth — it
cannot detect a wrong-but-plausible number, only a malformed or implausible
one.
"""

import numpy as np
import pandas as pd

from src.ingestion import MIN_ROWS

EXTREME_MOVE_THRESHOLD = 0.50  # single-day |return| beyond this gets flagged


def _check(name: str, status: str, message: str) -> dict:
    return {"check": name, "status": status, "message": message}


def validate_prices(prices: pd.DataFrame) -> dict:
    """
    Run the full validation gate on a price DataFrame (dates x tickers).
    Returns {"checks": [...], "passed": bool} — passed is False only on a
    hard FAIL; WARN checks surface issues without blocking the app.
    """
    checks = []

    # --- Schema ---
    if not isinstance(prices.index, pd.DatetimeIndex):
        checks.append(_check("schema.index", "FAIL", "Index is not a DatetimeIndex."))
    else:
        checks.append(_check("schema.index", "PASS", "DatetimeIndex confirmed."))

    dup = prices.index.duplicated().sum()
    checks.append(_check(
        "schema.duplicate_dates",
        "FAIL" if dup else "PASS",
        f"{dup} duplicate date(s) in the index." if dup else "No duplicate dates.",
    ))

    is_sorted = prices.index.is_monotonic_increasing
    checks.append(_check(
        "schema.sorted",
        "PASS" if is_sorted else "FAIL",
        "Dates are sorted ascending." if is_sorted else "Dates are NOT sorted ascending.",
    ))

    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    checks.append(_check(
        "schema.dtype",
        "FAIL" if non_numeric else "PASS",
        f"Non-numeric column(s): {', '.join(non_numeric)}." if non_numeric
        else "All columns numeric.",
    ))

    # --- Positivity ---
    non_positive = prices.le(0).sum().sum()
    checks.append(_check(
        "positivity.non_positive_prices",
        "FAIL" if non_positive else "PASS",
        f"{int(non_positive)} zero/negative price cell(s) found." if non_positive
        else "All prices strictly positive.",
    ))

    # --- Coverage ---
    rows = len(prices)
    checks.append(_check(
        "coverage.min_rows",
        "PASS" if rows >= MIN_ROWS else "FAIL",
        f"{rows} trading days (floor: {MIN_ROWS}).",
    ))

    # --- Staleness ---
    if rows:
        last_date = prices.index[-1]
        today = pd.Timestamp.now().normalize()
        staleness_days = int(np.busday_count(last_date.date(), today.date()))
        checks.append(_check(
            "freshness.staleness",
            "PASS" if staleness_days <= 1 else "WARN",
            f"Last row is {staleness_days} business day(s) old.",
        ))

    # --- Gaps vs. full business-day calendar ---
    if rows >= 2:
        full_span = pd.bdate_range(prices.index[0], prices.index[-1])
        gap_days = max(0, len(full_span) - rows)
        gap_pct = gap_days / len(full_span)
        checks.append(_check(
            "coverage.calendar_gaps",
            "WARN" if gap_pct > 0.05 else "PASS",
            f"{gap_days} business day(s) missing from the window "
            f"({gap_pct:.1%} of the span).",
        ))

    # --- Extreme single-day moves (flagged, not blocking) ---
    if rows >= 2:
        rets = prices.pct_change().dropna()
        extreme = (rets.abs() > EXTREME_MOVE_THRESHOLD)
        n_extreme = int(extreme.to_numpy().sum())
        if n_extreme:
            worst = rets.abs().to_numpy()
            worst_val = float(np.nanmax(worst))
            checks.append(_check(
                "sanity.extreme_moves",
                "WARN",
                f"{n_extreme} single-day move(s) beyond ±{EXTREME_MOVE_THRESHOLD:.0%} "
                f"(worst: {worst_val:.0%}) — could be a real crash day or a data glitch; "
                "verify before trusting.",
            ))
        else:
            checks.append(_check(
                "sanity.extreme_moves", "PASS",
                f"No single-day move beyond ±{EXTREME_MOVE_THRESHOLD:.0%}.",
            ))

    passed = not any(c["status"] == "FAIL" for c in checks)
    return {"checks": checks, "passed": passed}


if __name__ == "__main__":
    from src.ingestion import fetch_prices

    prices = fetch_prices()
    report = validate_prices(prices)
    print("--- Data Quality Report ---")
    for c in report["checks"]:
        print(f"[{c['status']:4s}] {c['check']:28s} {c['message']}")
    print(f"\nOverall: {'PASS' if report['passed'] else 'FAIL'}")
