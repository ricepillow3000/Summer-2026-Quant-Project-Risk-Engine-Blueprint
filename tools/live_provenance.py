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
4. Every preset ticker still trades on Yahoo with a full 2-year history.
5. The four crisis-record boxes under the hero ("Bought the scariest day",
   "Three years on", "Worst possible timing", "The AI-capex race"), worked out
   a second time from Yahoo's raw chart API with this file's own loop - no
   yfinance, no src.conviction - and compared as the page prints them.

    python -m tools.live_provenance
"""

import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from src.conviction import AI_CAPEX_BASKET, BENCHMARK, load_conviction, race_days, race_record
from src.ingestion import MAX_UNIVERSE, PRESETS, fetch_prices, fetch_risk_free_rate
from src.netguard import guarded_session
from src.scenarios import HISTORICAL_REGIMES

SYMBOLS = ["AAPL", "MSFT", "SPY", "IVV", "QQQ", "GLD", "GC=F", "EURUSD=X", "^GSPC"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def raw_chart(symbol: str, span: str = "3mo") -> pd.Series:
    """Adjusted closes straight from Yahoo's chart endpoint, keyed by date.

    span="max" asks for every day since 1970 by dates: Yahoo answers
    range=max with MONTHLY bars whatever the interval (Live audit run 105,
    2026-10-30: the crisis boxes read 252 "days" on as 21 years, +475%)."""
    span_q = (f"period1=0&period2={int(datetime.now(timezone.utc).timestamp())}"
              if span == "max" else f"range={span}")
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?{span_q}&interval=1d&events=div%2Csplit")
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


def crisis_boxes(bench: pd.Series, members: pd.DataFrame) -> dict:
    """The four boxes, from closes alone. Crash anatomy: the trough is the
    window's lowest close, the peak the highest close before it. A buyer's
    return is counted 252 and 756 trading days on (a crisis without that much
    history after it is left out). The race: each side's trading days from its
    own trough back to its own peak (within 756 days); the basket is the equal
    average of the members already trading at the window's start, each set to
    1 there; a side that never fell scores 0, one that never got back last."""
    bench = bench.dropna()
    fwd = {"t1": [], "t3": [], "p3": []}
    wins = raced = 0
    scores = []

    def anatomy(w):
        lo = w.idxmin()
        hi = w.loc[:lo].idxmax()
        return (hi, lo) if hi != lo else (None, None)

    def ahead(series, day, n):
        k = series.index.get_loc(day)
        return series.iloc[k + n] / series.iloc[k] - 1 if k + n < len(series) else None

    def back(series, lo, level):
        after = series.iloc[series.index.get_loc(lo):][:757]
        hit = np.flatnonzero(after.to_numpy() >= level)
        return int(hit[0]) if hit.size else None

    for start, end in HISTORICAL_REGIMES.values():
        w = bench.loc[start:end]
        if len(w) < 5:
            continue
        hi, lo = anatomy(w)
        if hi is not None:
            for key, day, n in (("t1", lo, 252), ("t3", lo, 756), ("p3", hi, 756)):
                v = ahead(bench, day, n)
                if v is not None:
                    fwd[key].append(v)
        b_score = 0.0 if hi is None else (lambda d: np.inf if d is None else d)(back(bench, lo, w.loc[hi]))
        m = members.loc[start:]
        alive = [c for c in m.columns if m[c].iloc[:5].notna().all()]
        comp = m[alive].dropna()
        comp = (comp / comp.iloc[0]).mean(axis=1) if alive and len(comp) else pd.Series(dtype=float)
        cw = comp.loc[start:end]
        if len(cw) < 5:
            continue
        chi, clo = anatomy(cw)
        if hi is None and chi is None:
            continue
        c_score = 0.0 if chi is None else (lambda d: np.inf if d is None else d)(back(comp, clo, cw.loc[chi]))
        raced += 1
        wins += c_score < b_score
        scores.append((float(c_score), float(b_score)))
    med = {k: float(np.median(v)) if v else None for k, v in fwd.items()}
    pos = {k: sum(x > 0 for x in v) for k, v in fwd.items()}
    return {"t1": f"{pos['t1']} of {len(fwd['t1'])}", "t1_median": med["t1"],
            "t3_median": med["t3"], "t3": f"{pos['t3']} of {len(fwd['t3'])}",
            "p3": f"{pos['p3']} of {len(fwd['p3'])}", "p3_median": med["p3"],
            "race": f"{wins} of {raced}", "race_days": scores}


def page_boxes(conv: dict) -> dict:
    """The same four boxes as main.py prints them from src.conviction."""
    s = conv["summary"]
    t1, t3, p3 = s["trough_1y_later"], s["trough_3y_later"], s["peak_3y_later"]
    rec = race_record(conv["race"])
    return {"t1": f"{round(t1['pct_positive'] * t1['n'])} of {t1['n']}", "t1_median": t1["median"],
            "t3_median": t3["median"], "t3": f"{round(t3['pct_positive'] * t3['n'])} of {t3['n']}",
            "p3": f"{round(p3['pct_positive'] * p3['n'])} of {p3['n']}", "p3_median": p3["median"],
            "race": f"{rec['wins']} of {rec['n']}",
            "race_days": [(race_days(r["basket_days"], r["basket_fell"], np.inf),
                           race_days(r["bench_days"], r["bench_fell"], np.inf))
                          for _, r in rec["decided"].iterrows()]}


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

    # 5. The crisis-record boxes, a second time from the raw chart API.
    raw = {t: raw_chart(t, "max") for t in [BENCHMARK, *AI_CAPEX_BASKET]}
    gap = int(np.median(np.diff(raw[BENCHMARK].index.values).astype("timedelta64[D]").astype(int)))
    check("crisis record source is daily bars", gap <= 3 and len(raw[BENCHMARK]) > 252 * 20,
          f"{len(raw[BENCHMARK])} {BENCHMARK} bars, median gap {gap} days, "
          f"from {raw[BENCHMARK].index[0].date()}")
    mine = crisis_boxes(raw[BENCHMARK], pd.DataFrame({t: raw[t] for t in AI_CAPEX_BASKET}))
    page = page_boxes(load_conviction())
    for key, label in (("t1", "Bought the scariest day: crises positive a year on"),
                       ("t1_median", "Bought the scariest day: median return a year on"),
                       ("t3_median", "Three years on: median return"),
                       ("t3", "Three years on: crises positive"),
                       ("p3", "Worst possible timing: crises ahead three years on"),
                       ("p3_median", "Worst possible timing: median return"),
                       ("race", "The AI-capex race: crises the basket reclaimed first"),
                       ("race_days", "The AI-capex race: (basket, S&P 500) days back, crisis by crisis")):
        a, b = page[key], mine[key]
        ok = (a == b) if isinstance(a, (str, list)) else (
            a is not None and b is not None and f"{a:+.0%}" == f"{b:+.0%}" and abs(a - b) < 1e-4)
        def shown(v):
            return v if isinstance(v, (str, list)) or v is None else f"{v:+.4%}"
        check(f"crisis record box - {label}", ok, f"page {shown(a)} vs raw Yahoo {shown(b)}")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} provenance checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
