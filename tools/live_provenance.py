"""
Prove the engine's numbers come from Yahoo Finance - not a cache, not a model.

Three independent cross-checks against the LIVE feed (needs Yahoo access):

1. Same-source re-fetch. Every price the engine loads (`fetch_prices`, via
   yfinance) must match Yahoo's raw chart API - fetched separately here, no
   yfinance in the path - to 1e-6 relative on every common date.
2. Second-source check. The risk-free rate the Sharpe ratio uses (^IRX from
   Yahoo) must agree with the US Treasury 3-month bill rate published by the
   Federal Reserve Bank of St. Louis (FRED series DTB3) to within 0.25 points.
3. Identity checks. Instruments that track the same thing must co-move the
   way they have to: SPY vs IVV (both S&P 500), GLD vs gold futures, QQQ vs
   SPY. A mislabelled or synthetic series fails these.

    python -m tools.live_provenance
"""

import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.ingestion import MAX_UNIVERSE, PRESETS, fetch_prices, fetch_risk_free_rate
from src.netguard import guarded_session

SYMBOLS = ["AAPL", "MSFT", "SPY", "IVV", "QQQ", "GLD", "GC=F", "EURUSD=X", "^GSPC"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def raw_chart(symbol: str) -> pd.Series:
    """Adjusted closes straight from Yahoo's chart endpoint, keyed by date."""
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
           "?range=3mo&interval=1d&events=div%2Csplit")
    res = guarded_session().get(url, timeout=30).json()["chart"]["result"][0]
    off = res["meta"].get("gmtoffset", 0)
    days = pd.to_datetime(np.array(res["timestamp"]) + off, unit="s").normalize()
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") \
        or res["indicators"]["quote"][0]["close"]
    raw = pd.Series(adj, index=days, dtype=float).dropna()
    # Futures and FX can carry two bars on one calendar date (session roll).
    return raw[~raw.index.duplicated(keep="last")]


def main() -> int:
    eng = fetch_prices(SYMBOLS, period="3mo", use_cache=False, align=False)
    eng.index = pd.to_datetime(eng.index).tz_localize(None).normalize()
    eng = eng[~eng.index.duplicated(keep="last")]
    for sym in SYMBOLS:
        if sym not in eng.columns:
            check(f"{sym} loaded", False, "engine returned no column")
            continue
        raw = raw_chart(sym)
        # The newest bar can still be moving (FX trades round the clock), so
        # compare completed bars only.
        both = pd.concat([eng[sym].dropna(), raw], axis=1, join="inner").dropna().iloc[:-1]
        if both.empty:
            check(f"{sym} engine price == raw Yahoo chart", False, "no common dates")
            continue
        gap = (both.iloc[:, 0] / both.iloc[:, 1] - 1).abs()
        worst, day = float(gap.max()), gap.idxmax()
        check(f"{sym} engine price == raw Yahoo chart", len(both) >= 40 and worst < 1e-6,
              f"{len(both)} common days, worst relative gap {worst:.2e} on {day.date()} "
              f"(engine {both.loc[day].iloc[0]:.6f} vs raw {both.loc[day].iloc[1]:.6f}), "
              f"{both.index[-1].date()} close {both.iloc[-1, 0]:.4f}")

    last = eng.dropna(how="all").index[-1]
    age = np.busday_count(last.date(), datetime.now(timezone.utc).date())
    check("data is current", age <= 4, f"latest bar {last.date()}, {age} business days old")

    r = eng.pct_change(fill_method=None)
    for a, b, floor in (("SPY", "IVV", 0.99), ("GLD", "GC=F", 0.80), ("QQQ", "SPY", 0.80)):
        c = float(r[[a, b]].dropna().corr().iloc[0, 1])
        check(f"corr({a}, {b}) >= {floor}", c >= floor, f"{c:.4f}")

    irx = fetch_risk_free_rate(use_cache=False)
    req = urllib.request.Request("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3",
                                 headers={"User-Agent": "meleona-audit"})
    fred = None
    for _ in range(3):                      # FRED is slow at times; an
        try:                                # unreachable source fails the
            fred = pd.read_csv(urllib.request.urlopen(req, timeout=60))  # check
            break
        except OSError as exc:
            err = exc
    if fred is None:
        check("^IRX (Yahoo) vs DTB3 (FRED)", False, f"FRED unreachable, not verified ({err!r})")
    else:
        fred = fred[pd.to_numeric(fred.iloc[:, 1], errors="coerce").notna()]
        dtb3 = float(fred.iloc[-1, 1]) / 100
        check("^IRX (Yahoo) vs DTB3 (FRED)", irx is not None and abs(irx - dtb3) <= 0.0025,
              f"Yahoo {irx if irx is None else f'{irx:.4%}'} vs FRED {dtb3:.4%} "
              f"({fred.iloc[-1, 0]})")

    # 4. Every preset ticker still trades on Yahoo with a full 2-year history.
    # A delisted, merged or renamed symbol silently shrinks a one-click basket
    # to the days all members share - this names the culprit.
    every = sorted({t for basket in PRESETS.values() for t in basket})
    # fetch_prices caps a request at MAX_UNIVERSE symbols, so go in batches.
    cover = pd.concat([fetch_prices(every[i:i + MAX_UNIVERSE], period="2y",
                                    use_cache=False, align=False)
                       for i in range(0, len(every), MAX_UNIVERSE)], axis=1)
    today = datetime.now(timezone.utc).date()
    for t in every:
        col = cover[t].dropna() if t in cover.columns else pd.Series(dtype=float)
        if col.empty:
            check(f"preset ticker {t} on Yahoo", False, "no data returned")
            continue
        stale = np.busday_count(col.index[-1].date(), today)
        check(f"preset ticker {t} on Yahoo", len(col) >= 400 and stale <= 4,
              f"{len(col)} days, {col.index[0].date()} -> {col.index[-1].date()}")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} provenance checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
