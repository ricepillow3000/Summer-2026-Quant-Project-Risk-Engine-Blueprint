"""
DataEngine: pulls and caches live price data for an arbitrary universe.

Data sourcing (read this before trusting a number):
- Every price comes from Yahoo Finance via the `yfinance` library, fetched at
  runtime. Nothing here originates from a language model or static fixture.
- Each download writes a provenance record (source, UTC timestamp, symbols,
  coverage) so any figure on the dashboard traces back to where and when it
  was pulled.

Caching strategy for a live web app:
- Per-universe disk cache keyed by ticker-set, so different baskets don't
  collide and repeat views are instant (no Yahoo rate-limiting on refresh).
- The cache is FRESHNESS-AWARE: it re-pulls once it ages past `max_age_hours`,
  so a deployed app never serves indefinitely-stale data. `clear_cache` (or
  `python -m tools.warm_cache --drill`) forces a re-pull - an operator action,
  not a visitor one.

Universe is fully configurable - equities, ETFs, FX (EURUSD=X), or futures
(GC=F) - so the engine speaks to any audience, not just one watchlist.
"""

import os
import re
import json
import hashlib
import threading
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from src.netguard import BudgetExhausted, check_budget, guarded_session

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Default universe (used when none is supplied). Kept as TICKERS for backwards
# compatibility with the analytics/risk modules' __main__ smoke tests.
DEFAULT_UNIVERSE = ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
TICKERS = DEFAULT_UNIVERSE

# Minimum history we consider statistically usable for risk estimates.
MIN_ROWS = 60

# How long a disk cache stays valid before we re-pull from Yahoo.
CACHE_MAX_AGE_HOURS = 6

# --- complexity budget ------------------------------------------------------
# The public app has no login and no queue: one visitor's request is the whole
# server's work. Cost scales with the number of tickers (each one is a Yahoo
# leg, a covariance row, a Grit Zone full-history pull and a map unit), so the
# universe size IS this engine's query-complexity dimension - the analogue of
# a depth/complexity limit on a query language. Bounded here, at the same
# funnel that validates ticker shape, so no caller can exceed it; the UI
# enforces it visibly rather than letting the funnel truncate in silence.
MAX_UNIVERSE = 25

# Cache is derived data (re-fetchable, no secrets), but the disk it sits on is
# a small ephemeral volume and any visitor can mint a new cache key by typing
# a new ticker combination. Bound it so a stream of novel baskets cannot fill
# the volume; oldest entries go first. 1500: a warm pass writes ~720 files
# (~270 basket files + ~450 per-name files, see _raw_path); at 240 a pass
# already evicted its own first presets
# (test_a_warm_pass_fits_inside_the_cache_budget).
CACHE_MAX_FILES = 1500
CACHE_MAX_MB = 200

# One-click starting baskets for a wider audience. These are representative,
# illustrative groupings - NOT any firm's actual holdings.
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
    # Institutional staples - what a rates/credit desk, a commodities book,
    # or a global-macro fund actually watches, via liquid ETF proxies.
    "Rates & credit (ETF proxies)": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "TIP",
    ],
    "Commodities & real assets": [
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC",
    ],
    "Global macro (multi-asset ETFs)": [
        "SPY", "EFA", "EEM", "TLT", "GLD", "UUP",
    ],
    # Small/micro-cap names, deliberately chosen for THIN volume rather than any
    # view on the companies. Every basket above is mega-cap or ETF, so days-to-
    # liquidate rounds to zero and the Liquidity tab shows a 0% adjustment no
    # matter where the sliders go - the model looks broken when it is merely
    # unstressed. On this basket the participation-rate cap actually binds.
    "Small-cap liquidity stress (thin volume)": [
        "CRWS", "WEYS", "BSET", "VRA", "HBB",
        "FLWS", "LAKE", "JOUT", "UNTY", "SMP",
    ],
    # Sector and theme baskets, added 2026-08-26 after a walkthrough found that
    # a visitor's own holding - NFLX, AMD, BAC, PFE - was in none of the
    # starter baskets, so the picker's suggestion list never offered it. The
    # box has always accepted any Yahoo symbol typed in full; these exist so a
    # visitor can RECOGNISE their names rather than having to know the ticker.
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
        # EA was dropped 2026-08-27: it went private and its last print was
        # 2026-08-13, leaving 20 usable days. The short-history guard in
        # main.py catches that and says so, but a one-click preset should not
        # need rescuing - TTWO and RBLX already carry the gaming exposure.
        # WBD was dropped 2026-10-15: the live audit found Yahoo returns only
        # 4 days for it (2026-10-05 -> 2026-10-08), which cut this whole
        # preset to 4 common days. Re-add only once its history is back.
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


# Names offered as suggestions in the ticker picker. A superset of every preset
# above: the box accepts any Yahoo symbol, but a visitor should be able to find
# a familiar company by typing the first letters of its ticker instead of
# needing to know it by heart. Every name here is checked live, alone and in
# every pair, by tests/pair_audit.py: on 2026-10-17 Yahoo answered "Quote not
# found" for BK and K (Kellanova, taken private), so a visitor picking either
# got no data - both left this list, and the audit fails on the next one.
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


# Yahoo Finance symbols are letters, digits and a small punctuation set:
# BRK-B, EURUSD=X, GC=F, ^IRX. Nothing else is a ticker.
#
# This is a SECURITY boundary, not just tidiness. The ticker box accepts
# arbitrary user text (accept_new_options=True), and the app renders several
# strings containing ticker names through `unsafe_allow_html=True` - the
# headline verdict interpolates the excluded-ticker list straight into a
# <div>. Without this filter a "ticker" of <IMG SRC=X ONERROR=...> is stored
# HTML injection, and upper-casing does not help because HTML tags are
# case-insensitive. Validating here, at the one funnel every ticker list
# passes through, closes every downstream sink at once - far safer than
# escaping at 32 separate render sites and hoping none is missed.
VALID_TICKER = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


class NoDataError(RuntimeError):
    """Yahoo returned nothing usable for the request. Its message is written
    for a visitor - no paths, no internals - so the UI may show it verbatim,
    unlike every other exception (see safe_err in main.py)."""


NO_DATA_MSG = ("Yahoo Finance returned no price history for these symbols. "
               "Check the spelling (e.g. BRK-B, EURUSD=X, GC=F) - or the feed "
               "may be briefly unreachable, so try again in a minute.")
NO_OVERLAP_MSG = ("These symbols share no common trading days. Remove the "
                  "newest or a delisted name and try again.")


def valid_ticker(symbol: str) -> bool:
    """True if `symbol` is shaped like a Yahoo Finance ticker."""
    return bool(VALID_TICKER.match(str(symbol).strip().upper()))


def _clean(tickers: list[str] | None) -> list[str]:
    cleaned = {t.strip().upper() for t in (tickers or DEFAULT_UNIVERSE) if t.strip()}
    valid = sorted(t for t in cleaned if VALID_TICKER.match(t))
    # Hard complexity bound: a caller that ignores MAX_UNIVERSE still cannot
    # make the server do unbounded work (see MAX_UNIVERSE above).
    return valid[:MAX_UNIVERSE]


def _cache_path(tickers: list[str], period: str, align: bool = True) -> str:
    """Stable per-universe cache filename so different baskets don't collide."""
    flag = "a" if align else "r"
    # A filename, not a security control: usedforsecurity=False says so.
    key = hashlib.md5((",".join(tickers) + "|" + period + "|" + flag).encode(),
                      usedforsecurity=False).hexdigest()[:12]
    return os.path.join(DATA_DIR, f"prices_{key}.parquet")


def _meta_path(cache_path: str) -> str:
    return cache_path.replace(".parquet", ".meta.json")


def _cache_is_fresh(meta_path: str, max_age_hours: float) -> bool:
    """True if a provenance record exists and was written within max_age_hours."""
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as fh:
            fetched = datetime.datetime.fromisoformat(json.load(fh)["fetched_at_utc"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return False
    age = datetime.datetime.now(datetime.timezone.utc) - fetched
    return age <= datetime.timedelta(hours=max_age_hours)


def _dv_cache_path(tickers: list[str], period: str) -> str:
    """Cache filename for a universe's daily dollar-volume history."""
    key = hashlib.md5((",".join(tickers) + "|" + period + "|dv").encode(),
                      usedforsecurity=False).hexdigest()[:12]
    return os.path.join(DATA_DIR, f"dollarvol_{key}.parquet")


def clear_cache(tickers: list[str] | None = None, period: str = "2y") -> None:
    """Delete the disk cache + provenance for a universe, forcing a fresh pull."""
    cleaned = _clean(tickers)
    price_cache = _cache_path(cleaned, period)
    paths = [price_cache, _meta_path(price_cache)]
    # Prices and dollar volume now share one download; clear both, plus the
    # legacy "6mo" dollar-volume cache from before they were unified.
    for dv_period in (period, "6mo"):
        dv_cache = _dv_cache_path(cleaned, dv_period)
        paths += [dv_cache, _meta_path(dv_cache)]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _prune_cache(max_files: int = CACHE_MAX_FILES,
                 max_mb: float = CACHE_MAX_MB) -> int:
    """
    Keep the parquet cache inside its budget, oldest entry first. Returns how
    many files were dropped. Cache entries are derived market data - dropping
    one costs a re-fetch, never information.
    """
    try:
        names = [f for f in os.listdir(DATA_DIR)
                 if f.endswith(".parquet") or f.endswith(".meta.json")]
        entries = [(os.path.getmtime(os.path.join(DATA_DIR, f)),
                    os.path.getsize(os.path.join(DATA_DIR, f)),
                    os.path.join(DATA_DIR, f)) for f in names]
    except OSError:
        return 0
    entries.sort()                                   # oldest first
    total_mb = sum(size for _, size, _ in entries) / 1e6
    dropped = 0
    while entries and (len(entries) > max_files or total_mb > max_mb):
        _, size, path = entries.pop(0)
        try:
            os.unlink(path)
        except OSError:
            continue
        total_mb -= size / 1e6
        dropped += 1
    return dropped


def _raw_path(ticker: str, period: str) -> str:
    """One name's own (Close, Volume) download for one period."""
    key = hashlib.md5((ticker + "|" + period + "|raw").encode(),
                      usedforsecurity=False).hexdigest()[:12]
    return os.path.join(DATA_DIR, f"raw_{key}.parquet")


def _read_raw(ticker: str, period: str, max_age_hours: float) -> pd.DataFrame | None:
    """A name's cached download if its RECORDED pull time is fresh (a copied or
    touched file keeps its old stamp), else None."""
    try:
        frame = pd.read_parquet(_raw_path(ticker, period))
        fetched = datetime.datetime.fromisoformat(frame.attrs["fetched_at_utc"])
    except (OSError, ValueError, KeyError):
        return None
    age = datetime.datetime.now(datetime.timezone.utc) - fetched
    return frame if age <= datetime.timedelta(hours=max_age_hours) else None


def _own_rows(raw: pd.DataFrame, ticker: str, alone: bool) -> pd.DataFrame:
    """One name's rows out of a yf.download frame. yfinance fills a name's
    missing volume with 0 and drops its all-empty rows, so a row with no
    Volume is one the download added only to line up with another name."""
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


# A name that comes back empty is asked for again, alone, at most RETRIES more
# times, RETRY_PAUSE * attempt seconds apart. Pair audit run 31 (2026-10-22)
# lost BA to "OperationalError('database is locked')" - yfinance's timezone
# cache, under its own download threads - and run 28 lost BAC; the same lock
# can drop a name from a visitor's basket here. Nothing else is retried, and a
# name still empty is left out exactly as before.
RETRIES = 2
RETRY_PAUSE = 0.5


def _download_close_volume(tickers: list[str], period: str,
                           max_age_hours: float = CACHE_MAX_AGE_HOURS,
                           use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalized (close, volume) frames, tickers-as-columns: the single chokepoint
    that feeds BOTH the price cache and the dollar-volume cache.

    Each name's own download is kept per period, and only names not already
    held are downloaded - in ONE yf.download. John, 2026-10-20: a typed-in
    basket was slow because the cache kept whole baskets, so three new names
    re-downloaded all fifteen under each of the page's five keys, and
    yfinance's repair pass costs seconds of CPU per name per download.
    yf.download is itself per name (Ticker.history, timezone strip, union of
    dates, sorted columns), so a basket stitched from names equals one download
    of it exactly (test_a_basket_stitched_from_single_names_matches_one_download).
    Names may now be pulled at different moments inside the freshness window;
    close.attrs carries the oldest and newest pull so provenance says so.
    """
    # yfinance relabels an ISIN with the ticker it resolves to, so a cached
    # ticker beside its ISIN would be one series twice: those take one download.
    has_isin = any(yf.utils.is_isin(t) for t in tickers)
    use_cache = use_cache and not has_isin
    frames = {}
    if use_cache:
        for t in tickers:
            cached = _read_raw(t, period, max_age_hours)
            if cached is not None:
                frames[t] = cached
    missing = [t for t in tickers if t not in frames]
    if missing:
        # Every outbound call goes through the egress allowlist (src/netguard.py):
        # the ticker funnel is the primary control, this is the second line.
        # The budget is checked up front so an exhausted cap reads as a cap, not
        # as a mysterious "possibly delisted" from inside yfinance.
        check_budget()
        # No retry beside an ISIN: there a name can come back "empty" because
        # yfinance filed its series under the ISIN, and asking for it again
        # would put the same series in twice (a singular covariance).
        for attempt in range(1 + (0 if has_isin else RETRIES)):
            if attempt:
                # Only for names that came back empty, and only with budget
                # left: the normal path never gets here.
                try:
                    check_budget()
                except BudgetExhausted:
                    break
                time.sleep(RETRY_PAUSE * attempt)
            raw = yf.download(missing, period=period, auto_adjust=True, progress=False,
                              repair=True, timeout=30, session=guarded_session())
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            for t in missing:
                own = (pd.DataFrame() if raw is None or raw.empty
                       else _own_rows(raw, t, alone=len(missing) == 1))
                if own.empty:
                    continue        # a typo or an outage is retried, never remembered
                own.attrs["fetched_at_utc"] = now
                if use_cache:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    _write_parquet(own, _raw_path(t, period))
                frames[t] = own
            missing = [t for t in missing if t not in frames]
            if not missing:
                break
    if not frames:
        raise NoDataError(NO_DATA_MSG)
    # A name with no data stays an all-NaN column, as in one multi download.
    names = sorted(set(tickers))
    close = pd.concat({t: f["Close"] for t, f in frames.items()}, axis=1,
                      sort=True).reindex(columns=names)
    volume = pd.concat({t: f["Volume"] for t, f in frames.items()}, axis=1,
                       sort=True).reindex(columns=names)
    stamps = sorted(f.attrs["fetched_at_utc"] for f in frames.values())
    # VALID_TICKER guards what a visitor types; these names are the FEED's, and
    # yfinance can label a column after a search result. 2026-10-15 security
    # audit: they flow into the map, captions and tables, so only ticker-shaped
    # columns leave this chokepoint (it feeds prices and dollar volume alike).
    keep = [c for c in close.columns if valid_ticker(c)]
    close = close[keep].rename_axis(columns="Ticker")
    close.attrs = {"fetched_at_utc": stamps[0], "fetched_newest_utc": stamps[-1]}
    return close, volume.reindex(columns=keep).rename_axis(columns="Ticker")


def _pulled(close: pd.DataFrame) -> tuple[str, str]:
    """(oldest, newest) pull behind a frame from _download_close_volume; now
    for a frame served without stamps (tests/pair_audit)."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return (close.attrs.get("fetched_at_utc", now),
            close.attrs.get("fetched_newest_utc", now))


def _clean_price_frame(prices: pd.DataFrame, align: bool) -> pd.DataFrame:
    """
    Data-integrity pass on a raw price frame. FX, futures, and equities trade on
    different calendars; a naive join leaves NaN holes that silently corrupt
    covariance/returns. Drop empty assets, forward-fill only tiny holiday gaps,
    then (when aligning) keep the common trading days so every asset is measured
    over the same dates.
    """
    prices = prices.dropna(axis=1, how="all")
    # Crypto trades on weekends. Filled before the align, every Saturday and
    # Sunday survived with stale stock prices: SPY vol read 16% low and the
    # 252-row "year" spanned 0.69 calendar years. The engine's year is 252
    # trading days, so weekends go; crypto's Friday->Monday move stays whole.
    prices = prices[prices.index.dayofweek < 5].ffill(limit=3)
    if align:
        prices = prices.dropna()
    return prices


def _dollar_volume(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Close x volume, except futures. Yahoo's =F Volume counts contracts and
    Close is per unit (oz, bbl, index point); with no contract multiplier the
    product is not dollars - SI=F read 2,240 days to exit a $5B book that
    clears in half a day. Zero is what an FX pair reports, so futures take the
    same no-volume path the page discloses as excluded, not estimated."""
    dv = close * volume
    fut = [c for c in dv.columns if str(c).endswith("=F")]
    dv[fut] = dv[fut] * 0.0
    return dv


def _write_parquet(frame: pd.DataFrame, path: str) -> None:
    """Write to a temporary name, then rename into place: the web process
    re-warms these files while visitors read them (Procfile), and a rename is
    atomic, so a reader gets the old file or the new one, never a torn one.
    The thread is in the name too: two visitors' threads that pull the same
    new name at once must not write one temp file."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def _write_dollar_volume_cache(tickers: list[str], period: str, dv: pd.DataFrame,
                               pulled: tuple[str, str]) -> None:
    """Persist a daily dollar-volume frame + its provenance record."""
    dv = dv.dropna(axis=1, how="all")
    if dv.empty:
        return
    cache_path = _dv_cache_path(tickers, period)
    _write_parquet(dv, cache_path)
    _prune_cache()          # keep the cache inside its disk budget
    meta = {
        "source": "Yahoo Finance (via yfinance)",
        "metric": "daily dollar volume (adj close x volume)",
        "fetched_at_utc": pulled[0],
        "fetched_newest_utc": pulled[1],
        "symbols": list(dv.columns),
        "period": period,
        "rows": len(dv),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
    }
    with open(_meta_path(cache_path), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def fetch_prices(tickers: list[str] | None = None, period: str = "2y",
                 use_cache: bool = True,
                 max_age_hours: float = CACHE_MAX_AGE_HOURS,
                 align: bool = True) -> pd.DataFrame:
    """
    Return daily adjusted closing prices for the given universe, from Yahoo.

    Uses a freshness-aware disk cache: a cached pull newer than `max_age_hours`
    is reused; otherwise the data is re-downloaded and re-stamped. The SAME
    download also populates the dollar-volume cache (see `average_dollar_volume`),
    so the dashboard's cold load costs one network round-trip, not two.

    Args:
        tickers: Yahoo Finance symbols. Defaults to DEFAULT_UNIVERSE if None.
        period:  yfinance period string - "1y", "2y", "5y", "max", etc.
        use_cache: read/write the disk cache.
        max_age_hours: how old a cache may be before re-pulling.
        align: if True, keep only common trading days across all assets (the
            right choice for current risk). Set False for historical replay,
            where assets have different inception dates and you slice a window
            later - forcing common dates over full history would truncate
            everything to the youngest asset's IPO.

    Returns:
        DataFrame with dates as index and tickers as columns. Symbols that
        return no data are dropped - check the columns to see what loaded.
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
        # The spend cap is spent. Degrade to stale cache rather than hammering
        # a feed that is already rate-limiting us - and let the provenance
        # panel keep reading the old .meta.json, so the page says how old the
        # data is instead of pretending it is live. With no cache at all there
        # is nothing honest to show, so the error stands.
        if use_cache and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        raise
    prices = _clean_price_frame(close, align)
    if prices.empty:
        # A typo'd symbol and a Yahoo outage look identical from here (yfinance
        # swallows the error and hands back NaNs), so the message names both.
        raise NoDataError(NO_DATA_MSG if close.dropna(how="all").empty
                          else NO_OVERLAP_MSG)

    if use_cache:
        _write_parquet(prices, cache_path)
        _prune_cache()      # keep the cache inside its disk budget
        # Provenance: stamp exactly where this data came from and when, so every
        # downstream number is traceable to a real source - not the model.
        # Names are pulled separately: the oldest pull is the basket's age
        # (it drives freshness and the "pulled N min ago" line), the newest
        # says how far apart they are.
        pulled = _pulled(close)
        meta = {
            "source": "Yahoo Finance (via yfinance)",
            "fetched_at_utc": pulled[0],
            "fetched_newest_utc": pulled[1],
            "symbols": list(prices.columns),
            "period": period,
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "rows": len(prices),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        # Free bonus from the same download: cache dollar volume over the aligned
        # trading days, so a later average_dollar_volume() call hits cache instead
        # of re-downloading. Never let a volume hiccup break the price fetch.
        if align:
            try:
                dv = _dollar_volume(close, volume).reindex(index=prices.index)[prices.columns]
                _write_dollar_volume_cache(tickers, period, dv, pulled)
            except Exception:  # noqa: BLE001
                pass
    return prices


def provenance(tickers: list[str] | None = None, period: str = "2y") -> dict | None:
    """Read the provenance record for a cached universe (source, timestamp, etc.)."""
    meta_path = _meta_path(_cache_path(_clean(tickers), period))
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def fetch_dollar_volume(tickers: list[str] | None = None, period: str = "2y",
                        use_cache: bool = True,
                        max_age_hours: float = CACHE_MAX_AGE_HOURS) -> pd.DataFrame:
    """
    Daily DOLLAR volume (adjusted close x share volume) per asset, from Yahoo.

    Dollar volume - not share count - is the liquidity metric that matters: a
    million shares of a $5 stock and a million shares of a $500 stock absorb
    wildly different amounts of capital. Normally this reads the cache that
    `fetch_prices` already populated from a shared download; it only hits the
    network if called standalone or the cache has aged out.

    Assets that report no volume (Yahoo returns 0 for FX pairs, for instance)
    come back as columns of zeros - surfaced honestly downstream, never faked.
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
    dollar_vol = _dollar_volume(close, volume).dropna(axis=1, how="all")
    if dollar_vol.empty:
        raise RuntimeError("No volume history for that universe.")
    if use_cache:
        _write_dollar_volume_cache(tickers, period, dollar_vol, _pulled(close))
    return dollar_vol


def average_dollar_volume(tickers: list[str] | None = None, period: str = "2y",
                          lookback_days: int = 63, **kwargs) -> pd.Series:
    """
    Mean daily dollar volume over the last `lookback_days` (default ~3 months of
    trading), as a Series indexed by ticker. Recent volume, not a multi-year
    average, is what tells you how much a name can absorb *today*. The `period`
    matches `fetch_prices` so both share one cached download.

    Assets with no volume data (e.g. Yahoo FX pairs) return 0.0 - a flag for the
    caller to treat as "not liquidatable from this feed," never a fabricated fill.
    """
    dv = fetch_dollar_volume(tickers, period=period, **kwargs)
    return dv.tail(lookback_days).mean().fillna(0.0)


def fetch_risk_free_rate(use_cache: bool = True,
                         max_age_hours: float = CACHE_MAX_AGE_HOURS) -> float | None:
    """
    Latest US 13-week Treasury-bill yield (^IRX) as an annual decimal - e.g.
    0.0525 for 5.25%. Yahoo quotes ^IRX in percent, so we divide by 100.

    This is the risk-free leg of the Sharpe ratio. Returns None if the fetch
    fails - we never fabricate a rate, so Sharpe hides rather than lying.
    Cached in a small JSON with the same freshness window as prices.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "riskfree.json")

    if use_cache and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            fetched = datetime.datetime.fromisoformat(rec["fetched_at_utc"])
            age = datetime.datetime.now(datetime.timezone.utc) - fetched
            if age <= datetime.timedelta(hours=max_age_hours):
                return rec["rate"]
        except (ValueError, KeyError, json.JSONDecodeError):
            pass

    try:
        raw = yf.download("^IRX", period="5d", auto_adjust=True,
                          progress=False, timeout=20, session=guarded_session())
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rate = float(close.dropna().iloc[-1]) / 100.0
    except Exception:  # noqa: BLE001 - no rate is better than a fake one
        return None
    if not np.isfinite(rate):
        return None

    if use_cache:
        rec = {
            "rate": rate,
            "source": "^IRX (US 13-week T-bill) via Yahoo Finance",
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc)
                              .isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
    return rate


def data_health(prices: pd.DataFrame) -> dict:
    """
    Summarize data-pipeline integrity for display. Catches the failure modes
    that quietly break a risk model: too little history, stale feeds, or
    calendar gaps left after alignment.
    """
    last_date = prices.index[-1]
    today = pd.Timestamp.now().normalize()
    staleness_days = int(np.busday_count(last_date.date(), today.date()))

    # Weekday holes INCLUDING holidays; data_quality's calendar check asks the
    # other question (what is missing beyond US federal holidays).
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
    """Assets whose price never moved in the window (zero return variance).

    A cash or money-market fund holding $1.00 looks like this. It has no
    correlation to estimate and no risk-parity weight - the ERC solver divides
    by its variance - so the app excludes it, and says so, before any math.
    """
    return [c for c in returns.columns if not float(returns[c].std()) > 1e-12]


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily returns - standard in quant finance."""
    return prices.pct_change().dropna()


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)
    print("\n--- Price Data (last 5 rows) ---")
    print(prices.tail())
    print("\n--- Provenance ---")
    print(json.dumps(provenance(), indent=2))
    print(f"\nShape: {returns.shape[0]} trading days x {returns.shape[1]} assets")
