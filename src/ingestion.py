"""
pulls live prices from yahoo for any basket and caches them on disk.

where the numbers come from (read this before trusting one):
- every price is from yahoo finance via `yfinance`, fetched at runtime. nothing
  here comes from a language model or a static fixture.
- every download writes a provenance record (source, utc time, symbols,
  coverage), so any number on the page traces back to where and when it came from.

caching:
- one disk cache per basket, keyed by the ticker set, so baskets don't collide
  and repeat views are instant (and don't get us rate-limited).
- it goes stale after `max_age_hours` and re-pulls, so the live app never
  serves old data forever. `python -m tools.warm_cache --drill` forces a
  re-pull - that's an operator thing, not a visitor thing.

any yahoo symbol works - stocks, etfs, fx (EURUSD=X), futures (GC=F).
"""

import os
import re
import json
import hashlib
import threading
import time
import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

from src.netguard import BudgetExhausted, check_budget, guarded_session

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DEFAULT_UNIVERSE = ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

MIN_ROWS = 60               # fewer rows than this is too thin for risk stats
CACHE_MAX_AGE_HOURS = 6     # cache older than this gets re-pulled

# --- complexity budget ------------------------------------------------------
# no login, no queue: one visitor's request is the whole server's work, and
# every ticker costs a download, a covariance row, a full-history pull and a
# map unit. so basket size is our "query complexity" and it's capped at the
# same funnel that checks ticker shape - stops one visitor from eating the box.
MAX_UNIVERSE = 25

# the cache is re-fetchable data, but anyone can mint a new cache key by typing
# a new basket. capped so a stream of new baskets can't fill the disk; oldest
# goes first. a warm pass writes ~720 files, so 1500 leaves room (at 240 a
# pass evicted its own first presets).
CACHE_MAX_FILES = 1500
CACHE_MAX_MB = 200

# one-click starter baskets. illustrative groupings, NOT anyone's real holdings.
PRESETS = {
    "Hedge-fund favorites (13F-popular)": [
        "MSFT", "AMZN", "META", "GOOGL", "NVDA", "AAPL",
        "BRK-B", "UNH", "V", "MA", "LLY", "AVGO",
    ],
    "Mega-cap tech (Mag 7 + QQQ)": DEFAULT_UNIVERSE,
    "Multi-sector blue chips (institutional-style)": [
        "JPM", "XOM", "UNH", "WMT", "PG", "JNJ", "V", "HD", "CAT", "LMT",
        "KO", "MCD",
    ],
    "Index core (S&P 500 & broad market ETFs)": [
        "SPY", "VOO", "IVV", "RSP", "DIA", "IWM", "VTI", "QQQ",
    ],
    "S&P sector ETFs": [
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
    ],
    "FX majors": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    ],
    "Futures (index & commodity)": [
        "ES=F", "NQ=F", "CL=F", "GC=F", "SI=F", "ZN=F",
    ],
    # what a rates desk, a commodities book or a macro fund watches, via etfs
    "Rates & credit (ETF proxies)": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "TIP",
    ],
    "Commodities & real assets": [
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC",
    ],
    "Global macro (multi-asset ETFs)": [
        "SPY", "EFA", "EEM", "TLT", "GLD", "UUP",
    ],
    # picked for THIN volume, not as a view on the companies. every other
    # basket is mega-cap or etf, so liquidity always read 0% and looked broken;
    # here the participation cap actually bites.
    "Small-cap liquidity stress (thin volume)": [
        "CRWS", "WEYS", "BSET", "VRA", "HBB",
        "FLWS", "LAKE", "JOUT", "UNTY", "SMP",
    ],
    # sector baskets so visitors can spot their own names (NFLX, AMD, BAC, PFE
    # were in no basket, so the picker never suggested them)
    "Semiconductors": [
        "NVDA", "AMD", "AVGO", "TSM", "MU", "QCOM", "TXN", "INTC", "AMAT", "ASML",
    ],
    "Big banks & brokers": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "COF",
    ],
    "Healthcare & pharma": [
        "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "AMGN", "TMO", "ABT", "ISRG",
    ],
    "Consumer & retail": [
        "COST", "WMT", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "PEP", "KO",
    ],
    "Streaming, media & platforms": [
        # no EA (went private, ~20 usable days) and no WBD (yahoo had 4 days
        # for it, which cut the whole basket to 4). add back once history is.
        "NFLX", "DIS", "SPOT", "META", "GOOGL", "RBLX", "CMCSA", "TTWO", "PARA",
    ],
    "Software & cloud": [
        "MSFT", "ORCL", "CRM", "ADBE", "NOW", "SNOW", "PLTR", "SHOP", "INTU", "PANW",
    ],
    "Industrials & transport": [
        "CAT", "DE", "BA", "GE", "HON", "LMT", "UPS", "FDX", "UNP", "RTX",
    ],
    "Energy & oil majors": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "KMI",
    ],
    "High-volatility movers": [
        "TSLA", "PLTR", "COIN", "MSTR", "RIVN", "UBER", "ABNB", "DKNG", "SOFI", "AFRM",
    ],
}


# the picker's suggestion list: every preset name plus more familiar ones, so
# you can find a company by its first letters. tests/pair_audit.py checks every
# name live (alone and in pairs) - BK and K got "quote not found" and were cut.
SUGGESTIONS = sorted({t for basket in PRESETS.values() for t in basket} | {
    "T", "VZ", "TMUS", "CMCSA", "CVS", "CI", "ELV", "HUM", "MDT", "BMY",
    "GILD", "REGN", "VRTX", "ZTS", "SYK", "BSX", "DHR", "MCK",
    "SPGI", "ICE", "CME", "MCO", "PYPL", "V", "MA", "USB", "PNC", "TFC",
    "STT", "AIG", "MET", "PRU", "ALL", "TRV", "PGR", "CB",
    "F", "GM", "DAL", "UAL", "LUV", "CSX", "NSC", "EMR", "ETN", "PH", "ITW",
    "MMM", "GD", "NOC", "LHX", "TDG", "PCAR", "CMI",
    "PG", "CL", "KMB", "GIS", "HSY", "STZ", "MO", "PM", "KDP", "MNST",
    "YUM", "CMG", "DPZ", "ROST", "TJX", "DG", "DLTR", "BBY", "EBAY", "ETSY",
    "AMD", "ARM", "LRCX", "KLAC", "ADI", "NXPI", "ON", "MRVL", "SMCI", "DELL",
    "IBM", "ACN", "CSCO", "ANET", "HPQ", "WDC", "STX", "ZS", "CRWD", "DDOG",
    "NET", "MDB", "TEAM", "WDAY", "VEEV", "HUBS", "TWLO", "OKTA", "DOCU",
    "BABA", "NIO", "PDD", "JD", "SE", "MELI", "TCEHY", "SONY", "TM", "SAP",
    "SHEL", "BP", "TTE", "RIO", "BHP", "VALE", "NEM", "FCX", "AA",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG",
    "AMT", "PLD", "SPG", "O", "PSA", "EQIX", "CCI", "WELL",
    "LIN", "APD", "SHW", "ECL", "DOW", "DD", "NUE", "STLD",
    "ARKK", "SMH", "SOXX", "XBI", "IBB", "ITB", "XRT", "KRE", "JETS", "TAN",
    "VUG", "VTV", "IWF", "IWD", "MDY", "IJR", "VEA", "VWO", "IEFA", "AGG", "BND",
    "GDX", "GDXJ", "PPLT", "PALL", "CPER", "WEAT", "CORN", "SOYB",
    "BTC-USD", "ETH-USD",
})


# yahoo symbols are letters, digits and a bit of punctuation: BRK-B, EURUSD=X,
# GC=F, ^IRX. nothing else is a ticker.
#
# SECURITY: this is the main control. the ticker box takes any text, and ticker
# names end up in html (unsafe_allow_html), file paths and fetch urls. without
# this, a "ticker" like <IMG SRC=X ONERROR=...> is stored html injection
# (upper-casing doesn't help, tags are case-insensitive). checking once here,
# where every ticker list passes, closes every sink at once instead of
# escaping at 30+ render sites and hoping none got missed.
VALID_TICKER = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


class NoDataError(RuntimeError):
    """yahoo gave back nothing usable. the message is written for visitors (no
    paths, no internals), so the ui can show it as-is - unlike every other
    exception (see safe_err in main.py)."""


NO_DATA_MSG = ("Yahoo Finance returned no price history for these symbols. "
               "Check the spelling (e.g. BRK-B, EURUSD=X, GC=F). The feed may "
               "also be briefly unreachable, so try again in a minute.")
NO_OVERLAP_MSG = ("These symbols share no common trading days. Remove the "
                  "newest or a delisted name and try again.")


def valid_ticker(symbol: str) -> bool:
    """true if `symbol` looks like a yahoo ticker."""
    return bool(VALID_TICKER.match(str(symbol).strip().upper()))


def _clean(tickers: list[str] | None) -> list[str]:
    cleaned = {t.strip().upper() for t in (tickers or DEFAULT_UNIVERSE) if t.strip()}
    valid = sorted(t for t in cleaned if VALID_TICKER.match(t))
    # hard cap: even a caller that ignores MAX_UNIVERSE can't make unbounded work
    return valid[:MAX_UNIVERSE]


def _hash(text: str) -> str:
    # full sha256: a short name could be searched for offline, and a basket
    # whose name matched a preset's would be served to that preset's visitors
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_path(tickers: list[str], period: str, align: bool = True) -> str:
    """one cache file per basket so different baskets don't collide."""
    key = _hash(",".join(tickers) + "|" + period + "|" + ("a" if align else "r"))
    return os.path.join(DATA_DIR, f"prices_{key}.parquet")


def _dv_cache_path(tickers: list[str], period: str) -> str:
    """cache file for a basket's daily dollar volume."""
    key = _hash(",".join(tickers) + "|" + period + "|dv")
    return os.path.join(DATA_DIR, f"dollarvol_{key}.parquet")


def _raw_path(ticker: str, period: str) -> str:
    """one name's own (Close, Volume) download for one period."""
    return os.path.join(DATA_DIR, f"raw_{_hash(ticker + '|' + period + '|raw')}.parquet")


def _meta_path(cache_path: str) -> str:
    return cache_path.replace(".parquet", ".meta.json")


# end of day, on new york's clock. a daily bar is final once its day's trading
# is over: the us close plus 30 min for the closing auction to print, or 8pm for
# markets that trade round the clock (fx, futures, crypto), whose yahoo day has
# ended by then. before that, yahoo's "today" row is the live price, not a close,
# so it's dropped - the page says end-of-day and means it.
NEW_YORK = ZoneInfo("America/New_York")
CLOSE_FINAL = datetime.time(16, 30)
ALL_DAY_FINAL = datetime.time(20, 0)
ALL_DAY = ("=X", "=F", "-USD")


def _now() -> datetime.datetime:
    """the clock (tests pin it)."""
    return datetime.datetime.now(datetime.timezone.utc)


def _last_final_day(ticker: str) -> datetime.date:
    """the newest day whose bar for `ticker` is a real close by now."""
    now = _now().astimezone(NEW_YORK)
    cut = ALL_DAY_FINAL if ticker.endswith(ALL_DAY) else CLOSE_FINAL
    return now.date() if now.time() >= cut else now.date() - datetime.timedelta(days=1)


def _finished_rows(own: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """drop bars whose trading day isn't over yet (dates are exchange-local)."""
    days = pd.DatetimeIndex(own.index)
    if days.tz is not None:
        days = days.tz_localize(None)
    return own[days.normalize() <= pd.Timestamp(_last_final_day(ticker))]


def _close_boundaries(now: datetime.datetime) -> list[datetime.datetime]:
    """weekday moments a new bar becomes final, a week either side of now."""
    today = now.astimezone(NEW_YORK).date()
    days = [today + datetime.timedelta(days=d) for d in range(-7, 8)]
    return sorted(datetime.datetime.combine(d, t, NEW_YORK)
                  for d in days if d.weekday() < 5 for t in (CLOSE_FINAL, ALL_DAY_FINAL))


def last_close_boundary() -> datetime.datetime:
    """the latest moment a new close came in."""
    now = _now()
    return max(b for b in _close_boundaries(now) if b <= now)


def next_close_boundary() -> datetime.datetime:
    """the next moment a new close comes in (tools/warm_cache wakes for it)."""
    now = _now()
    return min(b for b in _close_boundaries(now) if b > now)


def _is_fresh(stamp: str, max_age_hours: float) -> bool:
    """true if an iso utc stamp is at most max_age_hours old AND no market has
    closed since - a pull from before the close must not hide the close."""
    fetched = datetime.datetime.fromisoformat(stamp)
    if fetched < last_close_boundary():
        return False
    age = _now() - fetched
    return age <= datetime.timedelta(hours=max_age_hours)


def _cache_is_fresh(meta_path: str, max_age_hours: float) -> bool:
    """true if a provenance record exists and was written recently enough."""
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as fh:
            return _is_fresh(json.load(fh)["fetched_at_utc"], max_age_hours)
    except (ValueError, KeyError, json.JSONDecodeError):
        return False


def _prune_cache(max_files: int = CACHE_MAX_FILES,
                 max_mb: float = CACHE_MAX_MB) -> int:
    """
    keep the cache under budget, oldest first. returns how many files got
    dropped. it's all re-fetchable, so dropping costs a download, never data.

    typed-in baskets go before the presets' own downloads, so a stranger typing
    new baskets can't push every preset out and make the next visitor wait.
    presets only go if they alone blow the budget. the anchor shelf every page
    pulls counts as a preset.
    """
    from src.pairing import DEFENSIVE_ANCHOR_TICKERS   # here: pairing -> risk -> ingestion
    names = ({t for basket in PRESETS.values() for t in basket} | {"SPY"}
             | set(DEFENSIVE_ANCHOR_TICKERS))
    keep = {_raw_path(t, p) for t in names for p in ("2y", "max")}   # periods a preset page pulls
    try:
        paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                 if f.endswith((".parquet", ".meta.json"))]
        entries = [(p in keep, os.path.getmtime(p), os.path.getsize(p), p) for p in paths]
    except OSError:
        return 0
    entries.sort()                       # typed-in first, then oldest first
    total_mb = sum(e[2] for e in entries) / 1e6
    dropped = 0
    while entries and (len(entries) > max_files or total_mb > max_mb):
        _, _, size, path = entries.pop(0)
        try:
            os.unlink(path)
        except OSError:
            continue
        total_mb -= size / 1e6
        dropped += 1
    return dropped


def _read_raw(ticker: str, period: str, max_age_hours: float) -> pd.DataFrame | None:
    """a name's cached download if its RECORDED pull time is fresh (a copied or
    touched file keeps its old stamp), else None."""
    try:
        frame = pd.read_parquet(_raw_path(ticker, period))
        return frame if _is_fresh(frame.attrs["fetched_at_utc"], max_age_hours) else None
    except (OSError, ValueError, KeyError):
        return None


def _own_rows(raw: pd.DataFrame, ticker: str, alone: bool) -> pd.DataFrame:
    """one name's rows out of a yf.download frame. yfinance fills missing volume
    with 0 and drops all-empty rows, so a row with no Volume is one it only
    added to line up with another name."""
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(1):
            return pd.DataFrame()
        own = raw.xs(ticker, axis=1, level=1)
    elif alone:                     # a flat frame is one name's download
        own = raw
    else:
        return pd.DataFrame()
    own = own[["Close", "Volume"]]
    return own[own["Volume"].notna()]


# yahoo sometimes drops a name (yfinance's tz cache hits "database is locked"),
# so an empty name we know is real (suggested, or an anchor) is asked for again,
# alone, up to 2 more times. nothing else is retried; a name still empty after
# that is just left out.
RETRIES = 2
RETRY_PAUSE = 0.5

# a made-up symbol is never cached, so asking for it again and again spent the
# day's yahoo budget (security review, 2026-11-12). a name that came back empty
# isn't asked for again for 15 minutes - except suggested names, so an outage
# can't lock out a real one. made-up names that change every time can still
# spend it; presets are warmed and cached, so they keep working.
NO_DATA_TTL = 900
_NO_DATA: dict[str, float] = {}     # "name|period" -> when it came back empty


def _download_close_volume(tickers: list[str], period: str,
                           max_age_hours: float = CACHE_MAX_AGE_HOURS,
                           use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (close, volume) frames, tickers as columns. the one chokepoint that feeds
    both the price cache and the dollar-volume cache.

    each name's download is cached per period and only missing names get
    downloaded, in ONE yf.download - otherwise 3 new names re-pulled all 15,
    and yfinance's repair pass costs seconds of cpu per name. yf.download is
    per name anyway, so a basket stitched from names equals one download
    exactly (test_a_basket_stitched_from_single_names_matches_one_download).
    names can be pulled at different times, so close.attrs keeps the oldest
    and newest pull for provenance.
    """
    # yfinance relabels an isin with its ticker, so a cached ticker next to its
    # isin would be the same series twice: those baskets skip the cache
    has_isin = any(yf.utils.is_isin(t) for t in tickers)
    use_cache = use_cache and not has_isin
    frames = {}
    if use_cache:
        for t in tickers:
            cached = _read_raw(t, period, max_age_hours)
            if cached is not None:
                frames[t] = cached
    missing = [t for t in tickers if t not in frames]
    if use_cache:
        clock = time.monotonic()
        missing = [t for t in missing
                   if clock - _NO_DATA.get(f"{t}|{period}", -NO_DATA_TTL) >= NO_DATA_TTL]
    if missing:
        # spend cap checked up front, so a spent cap reads as a cap and not as
        # yfinance's confusing "possibly delisted". every call below goes
        # through the egress allowlist (guarded_session).
        check_budget()
        from src.pairing import DEFENSIVE_ANCHOR_TICKERS   # here: pairing -> risk -> ingestion
        known = set(SUGGESTIONS) | set(DEFENSIVE_ANCHOR_TICKERS)
        ask = missing
        # no retry next to an isin: the name may be "empty" because yfinance
        # filed it under the isin, and asking again would add it twice
        for attempt in range(1 + (0 if has_isin else RETRIES)):
            if attempt:
                # only for names that came back empty, and only with budget left
                try:
                    check_budget()
                except BudgetExhausted:
                    break
                time.sleep(RETRY_PAUSE * attempt)
            raw = yf.download(ask, period=period, auto_adjust=True, progress=False,
                              repair=True, timeout=30, session=guarded_session())
            now = _now().isoformat(timespec="seconds")
            for t in ask:
                own = (pd.DataFrame() if raw is None or raw.empty
                       else _finished_rows(_own_rows(raw, t, alone=len(ask) == 1), t))
                if own.empty:
                    continue        # never cached
                own.attrs["fetched_at_utc"] = now
                if use_cache:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    _write_parquet(own, _raw_path(t, period))
                frames[t] = own
            came_back = any(t in frames for t in ask)
            missing = [t for t in missing if t not in frames]
            # only a name we know is real is asked again: a made-up one beside a
            # real one was asked three times, tripling its cost to the budget
            ask = [t for t in missing if t in known]
            # nothing at all came back: an outage or all made-up - retrying won't help
            if not ask or not came_back and not attempt:
                break
        if use_cache:
            if len(_NO_DATA) > 10_000:
                _NO_DATA.clear()
            for t in set(missing) - set(SUGGESTIONS):
                _NO_DATA[f"{t}|{period}"] = time.monotonic()
    if not frames:
        raise NoDataError(NO_DATA_MSG)
    # a name with no data stays an all-NaN column, same as one multi download
    names = sorted(set(tickers))
    close = pd.concat({t: f["Close"] for t, f in frames.items()}, axis=1,
                      sort=True).reindex(columns=names)
    volume = pd.concat({t: f["Volume"] for t, f in frames.items()}, axis=1,
                       sort=True).reindex(columns=names)
    stamps = sorted(f.attrs["fetched_at_utc"] for f in frames.values())
    # SECURITY: VALID_TICKER covers what visitors type, but these column names
    # are the FEED's (yfinance can label one after a search hit) and they reach
    # the map, captions and tables - so only ticker-shaped columns get out.
    keep = [c for c in close.columns if valid_ticker(c)]
    close = close[keep].rename_axis(columns="Ticker")
    close.attrs = {"fetched_at_utc": stamps[0], "fetched_newest_utc": stamps[-1]}
    return close, volume.reindex(columns=keep).rename_axis(columns="Ticker")


def _pulled(close: pd.DataFrame) -> tuple[str, str]:
    """(oldest, newest) pull time behind a frame from _download_close_volume;
    'now' for a frame with no stamps (tests/pair_audit)."""
    now = _now().isoformat(timespec="seconds")
    return (close.attrs.get("fetched_at_utc", now),
            close.attrs.get("fetched_newest_utc", now))


def _clean_price_frame(prices: pd.DataFrame, align: bool) -> pd.DataFrame:
    """
    clean up a raw price frame. fx, futures and stocks trade on different
    calendars and a naive join leaves NaN holes that quietly wreck covariance.
    drop empty assets, ffill tiny holiday gaps, and (if aligning) keep only
    the days every asset traded.
    """
    prices = prices.dropna(axis=1, how="all")
    # drop weekends BEFORE filling: crypto trades on weekends, and filling first
    # kept every sat/sun with stale stock prices (SPY vol read 16% low, a
    # 252-row "year" was 0.69 years). crypto's fri->mon move stays whole.
    prices = prices[prices.index.dayofweek < 5].ffill(limit=3)
    if align:
        prices = prices.dropna()
    return prices


def _dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """close x volume, except futures. yahoo's =F volume is contracts and close
    is per unit (oz, bbl, point), and with no contract multiplier that's not
    dollars (SI=F said 2,240 days to exit a book that clears in half a day).
    so futures get zero, like fx, and the page says they're excluded."""
    dv = close * volume
    fut = [c for c in dv.columns if str(c).endswith("=F")]
    dv[fut] = dv[fut] * 0.0
    return dv


def _write_parquet(frame: pd.DataFrame, path: str) -> None:
    """write to a temp file, then rename into place. the app re-warms these
    files while visitors read them, and a rename is atomic, so a reader gets
    the old file or the new one, never a half-written one. thread id is in the
    temp name so two threads pulling the same name don't share a temp file."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def _write_json(record: dict, path: str) -> None:
    """same deal as _write_parquet for the little json records: temp file, then
    rename. writing in place let a reader grab half a file and crash the page."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)


def _write_dollar_volume_cache(tickers: list[str], period: str, dv: pd.DataFrame,
                               pulled: tuple[str, str]) -> None:
    """save a daily dollar-volume frame + its provenance record."""
    dv = dv.dropna(axis=1, how="all")
    if dv.empty:
        return
    cache_path = _dv_cache_path(tickers, period)
    _write_parquet(dv, cache_path)
    _prune_cache()          # stay inside the disk budget
    _write_json({
        "source": "Yahoo Finance (via yfinance)",
        "metric": "daily dollar volume (adj close x volume)",
        "fetched_at_utc": pulled[0],
        "fetched_newest_utc": pulled[1],
        "symbols": list(dv.columns),
        "period": period,
        "rows": len(dv),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
    }, _meta_path(cache_path))


def fetch_prices(tickers: list[str] | None = None, period: str = "2y",
                 use_cache: bool = True,
                 max_age_hours: float = CACHE_MAX_AGE_HOURS,
                 align: bool = True) -> pd.DataFrame:
    """
    daily adjusted closes for a basket, from yahoo.

    reuses a cached pull younger than `max_age_hours`, else re-downloads and
    re-stamps. the same download also fills the dollar-volume cache (see
    `average_dollar_volume`), so a cold page load is one round-trip, not two.

    Args:
        tickers: yahoo symbols. None means DEFAULT_UNIVERSE.
        period:  yfinance period string - "1y", "2y", "5y", "max", etc.
        use_cache: read/write the disk cache.
        max_age_hours: how old the cache can get before re-pulling.
        align: True keeps only days every asset traded (right for current
            risk). False is for historical replay: assets start at different
            dates, and forcing common dates would cut everything to the
            youngest asset's ipo.

    Returns:
        dates x tickers. symbols with no data are dropped, so check the columns.
    """
    tickers = _clean(tickers)
    if not tickers:
        raise ValueError("No tickers supplied.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = _cache_path(tickers, period, align)
    meta_path = _meta_path(cache_path)

    if use_cache and os.path.exists(cache_path) and _cache_is_fresh(meta_path, max_age_hours):
        return pd.read_parquet(cache_path)

    try:
        close, volume = _download_close_volume(tickers, period, max_age_hours, use_cache)
    except BudgetExhausted:
        # spend cap is spent: serve the stale cache instead of hammering a feed
        # that's already limiting us. the old .meta.json stays, so the page
        # still says how old it is. no cache at all = nothing honest to show.
        if use_cache and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        raise
    prices = _clean_price_frame(close, align)
    if prices.empty:
        # a typo and a yahoo outage look the same from here (yfinance swallows
        # the error and returns NaNs), so the message covers both
        raise NoDataError(NO_DATA_MSG if close.dropna(how="all").empty
                          else NO_OVERLAP_MSG)

    if use_cache:
        _write_parquet(prices, cache_path)
        _prune_cache()      # stay inside the disk budget
        # provenance: where and when this came from. names are pulled separately,
        # so the oldest pull is the basket's age (drives freshness and "pulled N
        # min ago") and the newest shows how far apart they are.
        pulled = _pulled(close)
        _write_json({
            "source": "Yahoo Finance (via yfinance)",
            "fetched_at_utc": pulled[0],
            "fetched_newest_utc": pulled[1],
            "symbols": list(prices.columns),
            "period": period,
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "rows": len(prices),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }, meta_path)
        # free bonus from the same download: cache dollar volume on the aligned
        # days so average_dollar_volume() hits cache. a volume hiccup must
        # never break the price fetch.
        if align:
            try:
                dv = _dollar_volume(close, volume).reindex(index=prices.index)[prices.columns]
                _write_dollar_volume_cache(tickers, period, dv, pulled)
            except Exception:  # noqa: BLE001
                pass
    return prices


def provenance(tickers: list[str] | None = None, period: str = "2y") -> dict | None:
    """the provenance record of a cached basket (source, time, etc.)."""
    try:
        with open(_meta_path(_cache_path(_clean(tickers), period)), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):   # no record yet, or a broken one: same thing to the page
        return None


def fetch_dollar_volume(tickers: list[str] | None = None, period: str = "2y",
                        use_cache: bool = True,
                        max_age_hours: float = CACHE_MAX_AGE_HOURS) -> pd.DataFrame:
    """
    daily DOLLAR volume (adjusted close x shares) per asset, from yahoo.

    dollars, not shares: a million shares of a $5 stock and of a $500 stock
    are very different amounts of money. usually this just reads what
    `fetch_prices` already cached; it only hits the network standalone or
    once the cache is stale.

    assets with no volume (yahoo says 0 for fx) come back as zeros - shown
    honestly downstream, never faked.
    """
    tickers = _clean(tickers)
    if not tickers:
        raise ValueError("No tickers supplied.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = _dv_cache_path(tickers, period)
    meta_path = _meta_path(cache_path)

    if use_cache and os.path.exists(cache_path) and _cache_is_fresh(meta_path, max_age_hours):
        return pd.read_parquet(cache_path)

    close, volume = _download_close_volume(tickers, period, max_age_hours, use_cache)
    # use the same days fetch_prices caches volume on (aligned weekdays), so the
    # 63-day ADV is one number whichever ran first. on the raw calendar crypto's
    # weekends counted as sessions and SPY's ADV came out 3% different.
    days = _clean_price_frame(close, align=True)
    dollar_vol = _dollar_volume(close, volume)
    if not days.empty:
        dollar_vol = dollar_vol.reindex(index=days.index)[days.columns]
    dollar_vol = dollar_vol.dropna(axis=1, how="all")
    if dollar_vol.empty:
        raise RuntimeError("No volume history for that universe.")
    if use_cache:
        _write_dollar_volume_cache(tickers, period, dollar_vol, _pulled(close))
    return dollar_vol


def average_dollar_volume(tickers: list[str] | None = None, period: str = "2y",
                          lookback_days: int = 63, **kwargs) -> pd.Series:
    """
    mean daily dollar volume over the last `lookback_days` (~3 months), per
    ticker. recent volume, not a multi-year average, is what a name can absorb
    *today*. `period` matches `fetch_prices` so both share one download.

    no volume data (yahoo fx) gives 0.0 - the caller reads that as "can't
    liquidate from this feed", never as a made-up fill.
    """
    dv = fetch_dollar_volume(tickers, period=period, **kwargs)
    return dv.tail(lookback_days).mean().fillna(0.0)


def fetch_risk_free_rate(use_cache: bool = True,
                         max_age_hours: float = CACHE_MAX_AGE_HOURS) -> float | None:
    """
    latest us 13-week t-bill yield (^IRX) as a decimal, e.g. 0.0525 for 5.25%
    (yahoo quotes it in percent, so /100).

    it's the risk-free leg of the sharpe ratio. returns None if the fetch
    fails - we never make up a rate, sharpe just hides. cached in a small json
    with the same freshness window as prices.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "riskfree.json")

    if use_cache and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            if _is_fresh(rec["fetched_at_utc"], max_age_hours):
                return rec["rate"]
        except (ValueError, KeyError, json.JSONDecodeError):
            pass

    try:
        raw = yf.download("^IRX", period="5d", auto_adjust=True,
                          progress=False, timeout=20, session=guarded_session())
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rate = float(_finished_rows(close.dropna(), "^IRX").iloc[-1]) / 100.0
    except Exception:  # noqa: BLE001 - no rate beats a fake one
        return None
    if not np.isfinite(rate):
        return None

    if use_cache:
        _write_json({
            "rate": rate,
            "source": "^IRX (US 13-week T-bill) via Yahoo Finance",
            "fetched_at_utc": _now().isoformat(timespec="seconds"),
        }, path)
    return rate


def data_health(prices: pd.DataFrame) -> dict:
    """
    pipeline health for the page: catches the stuff that quietly breaks a
    risk model - too little history, a stale feed, gaps after aligning.
    """
    last_date = prices.index[-1]
    today = _now().astimezone(NEW_YORK)
    staleness_days = int(np.busday_count(last_date.date(), today.date()))

    # weekday holes INCLUDING holidays; data_quality's calendar check asks the
    # other question (what's missing beyond us federal holidays)
    full_span = pd.bdate_range(prices.index[0], prices.index[-1])
    gap_days = int(len(full_span) - len(prices))

    return {
        "rows": len(prices),
        "assets": prices.shape[1],
        "start": prices.index[0].date().isoformat(),
        "end": last_date.date().isoformat(),
        "staleness_days": staleness_days,
        "gap_days": max(0, gap_days),
        "sufficient": len(prices) >= MIN_ROWS,
    }


def flat_columns(returns: pd.DataFrame) -> list[str]:
    """assets whose price never moved (zero variance), like a $1.00 money-market
    fund. nothing to correlate, and the risk-parity solver divides by its
    variance, so the app drops it - and says so - before any math."""
    return [c for c in returns.columns if not float(returns[c].std()) > 1e-12]


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """prices to daily returns."""
    return prices.pct_change().dropna()


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)
    print("\n--- Price Data (last 5 rows) ---")
    print(prices.tail())
    print("\n--- Provenance ---")
    print(json.dumps(provenance(), indent=2))
    print(f"\nShape: {returns.shape[0]} trading days x {returns.shape[1]} assets")
