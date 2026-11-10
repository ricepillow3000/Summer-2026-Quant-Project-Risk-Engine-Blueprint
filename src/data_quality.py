"""
data quality: a sanity gate that runs on every price pull, not a one-off eyeball.

Quant Deep Dive:
a risk number on bad data is worse than none - it's a wrong answer with a
confident chart on top. a feed returning 200 OK doesn't mean the data is fine,
so every pull goes through these checks (hand-rolled, no extra dependency).
each one says PASS / WARN / FAIL:
  - schema: numeric columns, DatetimeIndex, no duplicate dates, sorted.
  - positivity: no zero or negative prices (that's a broken feed).
  - coverage: at least MIN_ROWS rows (same floor as src.ingestion).
  - staleness: last row isn't too old (same count as ingestion.data_health).
  - extreme moves: a day beyond the threshold gets a WARN, not dropped and not
    trusted blindly - a real crash day (covid) SHOULD warn, that's the point.
  - gaps: missing business days beyond weekends and us holidays.

limits: this checks shape and plausibility, not truth. a wrong number that
looks normal gets through.
"""

import numpy as np
import pandas as pd

from src.ingestion import MIN_ROWS, data_health

EXTREME_MOVE_THRESHOLD = 0.50  # a one-day |return| past this gets flagged


def _check(name: str, status: str, message: str) -> dict:
    return {"check": name, "status": status, "message": message}


def validate_prices(prices: pd.DataFrame) -> dict:
    """
    run every check on a price frame (dates x tickers).
    returns {"checks": [...], "passed": bool}. only a FAIL makes passed False;
    a WARN shows the issue but doesn't block the app.
    """
    checks = []

    # --- schema ---
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
        "Dates are sorted ascending." if is_sorted else "Dates are not sorted ascending.",
    ))

    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    checks.append(_check(
        "schema.dtype",
        "FAIL" if non_numeric else "PASS",
        f"Non-numeric column(s): {', '.join(non_numeric)}." if non_numeric
        else "All columns numeric.",
    ))

    # --- positivity ---
    non_positive = prices.le(0).sum().sum()
    checks.append(_check(
        "positivity.non_positive_prices",
        "FAIL" if non_positive else "PASS",
        f"{int(non_positive)} zero/negative price cell(s) found." if non_positive
        else "All prices strictly positive.",
    ))

    # --- coverage ---
    rows = len(prices)
    checks.append(_check(
        "coverage.min_rows",
        "PASS" if rows >= MIN_ROWS else "FAIL",
        f"{rows} trading days (floor: {MIN_ROWS}).",
    ))

    # --- staleness (same count as the page's "Nd old" line) ---
    if rows:
        staleness_days = data_health(prices)["staleness_days"]
        checks.append(_check(
            "freshness.staleness",
            "PASS" if staleness_days <= 1 else "WARN",
            f"Last row is {staleness_days} business day(s) old.",
        ))

    if rows >= 2:
        # --- gaps vs the business-day calendar ---
        # not data_health's gap_days on purpose: that one counts holidays as
        # gaps, this one asks what's missing beyond them. bdate_range only drops
        # weekends, so we take out us federal holidays too (ships with pandas).
        # it's not the nyse calendar (no good friday), so a gap is only ever a WARN.
        from pandas.tseries.holiday import USFederalHolidayCalendar
        full_span = pd.bdate_range(prices.index[0], prices.index[-1])
        full_span = full_span.difference(
            USFederalHolidayCalendar().holidays(full_span[0], full_span[-1]))
        gap_days = max(0, len(full_span) - rows)
        gap_pct = gap_days / len(full_span)
        checks.append(_check(
            "coverage.calendar_gaps",
            "WARN" if gap_pct > 0.05 else "PASS",
            f"{gap_days} business day(s) missing from the window "
            f"({gap_pct:.1%} of the span).",
        ))

        # --- extreme one-day moves (flagged, never blocking) ---
        rets = prices.pct_change().dropna()
        n_extreme = int((rets.abs() > EXTREME_MOVE_THRESHOLD).to_numpy().sum())
        if n_extreme:
            worst_val = float(np.nanmax(rets.abs().to_numpy()))
            checks.append(_check(
                "sanity.extreme_moves",
                "WARN",
                f"{n_extreme} single-day move(s) beyond ±{EXTREME_MOVE_THRESHOLD:.0%} "
                f"(worst: {worst_val:.0%}). It could be a real crash day or a data "
                "glitch, so verify it before trusting it.",
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
