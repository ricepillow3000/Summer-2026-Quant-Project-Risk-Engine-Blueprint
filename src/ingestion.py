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
  so a deployed app never serves indefinitely-stale data. Manual refresh clears
  it on demand.

Universe is fully configurable — equities, ETFs, FX (EURUSD=X), or futures
(GC=F) — so the engine speaks to any audience, not just one watchlist.
"""

import os
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Default universe (used when none is supplied). Kept as TICKERS for backwards
# compatibility with the analytics/risk modules' __main__ smoke tests.
DEFAULT_UNIVERSE = ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
TICKERS = DEFAULT_UNIVERSE

# Minimum history we consider statistically usable for risk estimates.
MIN_ROWS = 60

# How long a disk cache stays valid before we re-pull from Yahoo.
CACHE_MAX_AGE_HOURS = 6

# One-click starting baskets for a wider audience. These are representative,
# illustrative groupings — NOT any firm's actual holdings.
PRESETS = {
    "Hedge-fund favorites (13F-popular)": [
        "MSFT", "AMZN", "META", "GOOGL", "NVDA", "AAPL",
        "BRK-B", "UNH", "V", "MA", "LLY", "AVGO",
    ],
    "Mega-cap tech (Mag 7 + QQQ)": DEFAULT_UNIVERSE,
    "Multi-sector blue chips (institutional-style)": [
        "JPM", "XOM", "UNH", "WMT", "PG", "JNJ", "V", "HD", "CAT", "LMT",
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
}


def _clean(tickers: list[str] | None) -> list[str]:
    return sorted({t.strip().upper() for t in (tickers or DEFAULT_UNIVERSE) if t.strip()})


def _cache_path(tickers: list[str], period: str, align: bool = True) -> str:
    """Stable per-universe cache filename so different baskets don't collide."""
    flag = "a" if align else "r"
    key = hashlib.md5((",".join(tickers) + "|" + period + "|" + flag).encode()).hexdigest()[:12]
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
    key = hashlib.md5((",".join(tickers) + "|" + period + "|dv").encode()).hexdigest()[:12]
    return os.path.join(DATA_DIR, f"dollarvol_{key}.parquet")


def clear_cache(tickers: list[str] | None = None, period: str = "2y") -> None:
    """Delete the disk cache + provenance for a universe, forcing a fresh pull."""
    cleaned = _clean(tickers)
    price_cache = _cache_path(cleaned, period)
    # Clear both the price cache and the (independently-fetched) dollar-volume
    # cache so a Refresh re-pulls everything the dashboard shows.
    for path in (price_cache, _meta_path(price_cache)):
        if os.path.exists(path):
            os.remove(path)
    for dv_period in ("6mo",):
        dv_cache = _dv_cache_path(cleaned, dv_period)
        for path in (dv_cache, _meta_path(dv_cache)):
            if os.path.exists(path):
                os.remove(path)


def fetch_prices(tickers: list[str] | None = None, period: str = "2y",
                 use_cache: bool = True,
                 max_age_hours: float = CACHE_MAX_AGE_HOURS,
                 align: bool = True) -> pd.DataFrame:
    """
    Return daily adjusted closing prices for the given universe, from Yahoo.

    Uses a freshness-aware disk cache: a cached pull newer than `max_age_hours`
    is reused; otherwise the data is re-downloaded and re-stamped.

    Args:
        tickers: Yahoo Finance symbols. Defaults to DEFAULT_UNIVERSE if None.
        period:  yfinance period string — "1y", "2y", "5y", "max", etc.
        use_cache: read/write the disk cache.
        max_age_hours: how old a cache may be before re-pulling.
        align: if True, keep only common trading days across all assets (the
            right choice for current risk). Set False for historical replay,
            where assets have different inception dates and you slice a window
            later — forcing common dates over full history would truncate
            everything to the youngest asset's IPO.

    Returns:
        DataFrame with dates as index and tickers as columns. Symbols that
        return no data are dropped — check the columns to see what loaded.
    """
    tickers = _clean(tickers)
    if not tickers:
        raise ValueError("No tickers supplied.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = _cache_path(tickers, period, align)
    meta_path = _meta_path(cache_path)

    if use_cache and os.path.exists(cache_path) and _cache_is_fresh(meta_path, max_age_hours):
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                      repair=True, timeout=30)

    # yfinance returns a column MultiIndex for multiple tickers, a flat one
    # for a single ticker. Normalize both to a tickers-as-columns frame.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    # --- Data integrity: align calendars across asset classes ---
    # FX, futures, and equities trade on different calendars. A naive join
    # leaves NaN holes that silently corrupt covariance/returns. We drop assets
    # with no data, forward-fill only tiny gaps (holiday mismatches), then keep
    # the common trading days so every asset is measured over the same dates.
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill(limit=3)
    if align:
        prices = prices.dropna()
    if prices.empty:
        raise RuntimeError("No overlapping price history for that universe.")

    if use_cache:
        prices.to_parquet(cache_path)
        # Provenance: stamp exactly where this data came from and when, so every
        # downstream number is traceable to a real source — not the model.
        meta = {
            "source": "Yahoo Finance (via yfinance)",
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc)
                              .isoformat(timespec="seconds"),
            "symbols": list(prices.columns),
            "period": period,
            "start": prices.index[0].date().isoformat(),
            "end": prices.index[-1].date().isoformat(),
            "rows": len(prices),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    return prices


def provenance(tickers: list[str] | None = None, period: str = "2y") -> dict | None:
    """Read the provenance record for a cached universe (source, timestamp, etc.)."""
    meta_path = _meta_path(_cache_path(_clean(tickers), period))
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def fetch_dollar_volume(tickers: list[str] | None = None, period: str = "6mo",
                        use_cache: bool = True,
                        max_age_hours: float = CACHE_MAX_AGE_HOURS) -> pd.DataFrame:
    """
    Daily DOLLAR volume (adjusted close x share volume) per asset, from Yahoo.

    Dollar volume — not share count — is the liquidity metric that matters: a
    million shares of a $5 stock and a million shares of a $500 stock absorb
    wildly different amounts of capital. Same freshness-aware cache + provenance
    as prices, so a deployed app never rate-limits or serves stale figures.

    Assets that report no volume (Yahoo returns 0 for FX pairs, for instance)
    come back as columns of zeros — we surface that honestly downstream rather
    than inventing a liquidity number.
    """
    tickers = _clean(tickers)
    if not tickers:
        raise ValueError("No tickers supplied.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = _dv_cache_path(tickers, period)
    meta_path = _meta_path(cache_path)

    if use_cache and os.path.exists(cache_path) and _cache_is_fresh(meta_path, max_age_hours):
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                      repair=True, timeout=30)

    # Normalize the multi- vs single-ticker column shapes, same as fetch_prices.
    if isinstance(raw.columns, pd.MultiIndex):
        close, volume = raw["Close"], raw["Volume"]
    else:
        close = raw[["Close"]].copy(); close.columns = tickers[:1]
        volume = raw[["Volume"]].copy(); volume.columns = tickers[:1]

    dollar_vol = (close * volume).dropna(axis=1, how="all")
    if dollar_vol.empty:
        raise RuntimeError("No volume history for that universe.")

    if use_cache:
        dollar_vol.to_parquet(cache_path)
        meta = {
            "source": "Yahoo Finance (via yfinance)",
            "metric": "daily dollar volume (adj close x volume)",
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc)
                              .isoformat(timespec="seconds"),
            "symbols": list(dollar_vol.columns),
            "period": period,
            "rows": len(dollar_vol),
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    return dollar_vol


def average_dollar_volume(tickers: list[str] | None = None, period: str = "6mo",
                          lookback_days: int = 63, **kwargs) -> pd.Series:
    """
    Mean daily dollar volume over the last `lookback_days` (default ~3 months of
    trading), as a Series indexed by ticker. Recent volume, not a multi-year
    average, is what tells you how much a name can absorb *today*.

    Assets with no volume data (e.g. Yahoo FX pairs) return 0.0 — a flag for the
    caller to treat as "not liquidatable from this feed," never a fabricated fill.
    """
    dv = fetch_dollar_volume(tickers, period=period, **kwargs)
    return dv.tail(lookback_days).mean().fillna(0.0)


def data_health(prices: pd.DataFrame) -> dict:
    """
    Summarize data-pipeline integrity for display. Catches the failure modes
    that quietly break a risk model: too little history, stale feeds, or
    calendar gaps left after alignment.
    """
    last_date = prices.index[-1]
    today = pd.Timestamp.now().normalize()
    staleness_days = int(np.busday_count(last_date.date(), today.date()))

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


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily returns — standard in quant finance."""
    return prices.pct_change().dropna()


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)
    print("\n--- Price Data (last 5 rows) ---")
    print(prices.tail())
    print("\n--- Provenance ---")
    print(json.dumps(provenance(), indent=2))
    print(f"\nShape: {returns.shape[0]} trading days x {returns.shape[1]} assets")
