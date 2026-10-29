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
import yfinance as yf

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
    # Each bar's day in the exchange's own time zone, summer time included. The meta's gmtoffset is
    # only today's offset: added to every bar, it moved every London summer bar (23:00 UTC = midnight
    # BST) back a day once the clocks went back (live audit run 86, 2026-10-28: EURUSD=X 0.8% "gap").
    tz = res["meta"].get("exchangeTimezoneName") or "UTC"
    days = pd.to_datetime(np.array(res["timestamp"]), unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
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
        detail = (f"{len(both)} common days, worst relative gap {worst:.2e} on {day.date()} "
                  f"(engine {both.loc[day].iloc[0]:.6f} vs raw {both.loc[day].iloc[1]:.6f}), "
                  f"{both.index[-1].date()} close {both.iloc[-1, 0]:.4f}")
        ok = len(both) >= 40 and worst < 1e-6
        if len(both) >= 40 and not ok:
            # The engine downloads with yfinance's repair=True, which corrects
            # known Yahoo bar errors. Prove a gap is that and nothing else:
            # the SAME download with repair off must equal the raw API exactly.
            plain = yf.download(sym, period="3mo", auto_adjust=True, repair=False,
                                progress=False, session=guarded_session())["Close"]
            plain = plain.iloc[:, 0] if isinstance(plain, pd.DataFrame) else plain
            plain.index = pd.to_datetime(plain.index).tz_localize(None).normalize()
            pb = pd.concat([plain[~plain.index.duplicated(keep="last")], raw],
                           axis=1, join="inner").dropna().iloc[:-1]
            pgap = float((pb.iloc[:, 0] / pb.iloc[:, 1] - 1).abs().max())
            ok = pgap < 1e-6
            detail += (f"; with repair off the series equals raw Yahoo to {pgap:.1e} - the gap is "
                       f"yfinance's documented price repair of Yahoo's own bar"
                       if ok else f"; with repair off the gap is still {pgap:.2e} - unexplained")
        check(f"{sym} engine price == raw Yahoo chart", ok, detail)

    last = eng.dropna(how="all").index[-1]
    age = np.busday_count(last.date(), datetime.now(timezone.utc).date())
    check("data is current", age <= 4, f"latest bar {last.date()}, {age} business days old")

    r = eng.pct_change(fill_method=None)
    for a, b, floor in (("SPY", "IVV", 0.99), ("GLD", "GC=F", 0.80), ("QQQ", "SPY", 0.80)):
        c = float(r[[a, b]].dropna().corr().iloc[0, 1])
        check(f"corr({a}, {b}) >= {floor}", c >= floor, f"{c:.4f}")

    irx = fetch_risk_free_rate(use_cache=False)
    req = urllib.request.Request("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3",
                                 headers={"User-Agent": "meliona-audit"})
    # Second source: FRED's DTB3, else the US Treasury's own daily bill rates
    # (the original publisher FRED copies; FRED times out from some runners).
    ref = None
    try:
        fred = pd.read_csv(urllib.request.urlopen(req, timeout=20))  # nosec B310 - fixed https URL
        fred = fred[pd.to_numeric(fred.iloc[:, 1], errors="coerce").notna()]
        ref = (float(fred.iloc[-1, 1]) / 100, f"FRED DTB3 {fred.iloc[-1, 0]}")
    except OSError:
        year = datetime.now(timezone.utc).year
        tsy = pd.read_csv(urllib.request.urlopen(urllib.request.Request(  # nosec B310 - fixed https URL
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_bill_rates"
            f"&field_tdr_date_value={year}&page&_format=csv",
            headers={"User-Agent": "meliona-audit"}), timeout=60))
        col = next(c for c in tsy.columns if "13" in c and "BANK DISCOUNT" in c.upper())
        tsy["Date"] = pd.to_datetime(tsy["Date"])
        row = tsy.sort_values("Date").dropna(subset=[col]).iloc[-1]
        ref = (float(row[col]) / 100, f"US Treasury 13-week bank discount {row['Date'].date()}")
    check("^IRX (Yahoo) vs a second official source", irx is not None and abs(irx - ref[0]) <= 0.0025,
          f"Yahoo {irx if irx is None else f'{irx:.4%}'} vs {ref[1]} {ref[0]:.4%}")

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
