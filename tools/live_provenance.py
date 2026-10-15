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

from src.ingestion import fetch_prices, fetch_risk_free_rate
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
    return pd.Series(adj, index=days, dtype=float).dropna()


def main() -> int:
    eng = fetch_prices(SYMBOLS, period="3mo", use_cache=False, align=False)
    eng.index = pd.to_datetime(eng.index).tz_localize(None).normalize()
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
        worst = float((both.iloc[:, 0] / both.iloc[:, 1] - 1).abs().max())
        check(f"{sym} engine price == raw Yahoo chart", len(both) >= 40 and worst < 1e-6,
              f"{len(both)} common days, worst relative gap {worst:.2e}, "
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
    fred = pd.read_csv(urllib.request.urlopen(req, timeout=30))
    fred = fred[pd.to_numeric(fred.iloc[:, 1], errors="coerce").notna()]
    dtb3 = float(fred.iloc[-1, 1]) / 100
    check("^IRX (Yahoo) vs DTB3 (FRED)", irx is not None and abs(irx - dtb3) <= 0.0025,
          f"Yahoo {irx if irx is None else f'{irx:.4%}'} vs FRED {dtb3:.4%} "
          f"({fred.iloc[-1, 0]})")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} provenance checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
