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
    paths = [price_cache, _meta_path(price_cache)]
    # Prices and dollar volume now share one download; clear both, plus the
    # legacy "6mo" dollar-volume cache from before they were unified.
    for dv_period in (period, "6mo"):
        dv_cache = _dv_cache_path(cleaned, dv_period)
        paths += [dv_cache, _meta_path(dv_cache)]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _download_close_volume(tickers: list[str], period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One Yahoo download → normalized (close, volume) frames, tickers-as-columns.

    This is the single network round-trip that now feeds BOTH the price cache
    and the dollar-volume cache. yfinance returns a column MultiIndex for
    multiple tickers and a flat index for one — we normalize both shapes here.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                      repair=True, timeout=30)
    if isinstance(raw.columns, pd.MultiIndex):
        close, volume = raw["Close"], raw["Volume"]
    else:
        close = raw[["Close"]].copy(); close.columns = tickers[:1]
        volume = raw[["Volume"]].copy(); volume.columns = tickers[:1]
    return close, volume


def _clean_price_frame(prices: pd.DataFrame, align: bool) -> pd.DataFrame:
    """
    Data-integrity pass on a raw price frame. FX, futures, and equities trade on
    different calendars; a naive join leaves NaN holes that silently corrupt
    covariance/returns. Drop empty assets, forward-fill only tiny holiday gaps,
    then (when aligning) keep the common trading days so every asset is measured
    over the same dates.
    """
    prices = prices.dropna(axis=1, how="all").ffill(limit=3)
    if align:
        prices = prices.dropna()
    return prices


def _write_dollar_volume_cache(tickers: list[str], period: str, dv: pd.DataFrame) -> None:
    """Persist a daily dollar-volume frame + its provenance record."""
    dv = dv.dropna(axis=1, how="all")
    if dv.empty:
        return
    cache_path = _dv_cache_path(tickers, period)
    dv.to_parquet(cache_path)
    meta = {
        "source": "Yahoo Finance (via yfinance)",
        "metric": "daily dollar volume (adj close x volume)",
        "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc)
                          .isoformat(timespec="seconds"),
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

    close, volume = _download_close_volume(tickers, period)
    prices = _clean_price_frame(close, align)
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
        # Free bonus from the same download: cache dollar volume over the aligned
        # trading days, so a later average_dollar_volume() call hits cache instead
        # of re-downloading. Never let a volume hiccup break the price fetch.
        if align:
            try:
                dv = (close * volume).reindex(index=prices.index)[prices.columns]
                _write_dollar_volume_cache(tickers, period, dv)
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

    Dollar volume — not share count — is the liquidity metric that matters: a
    million shares of a $5 stock and a million shares of a $500 stock absorb
    wildly different amounts of capital. Normally this reads the cache that
    `fetch_prices` already populated from a shared download; it only hits the
    network if called standalone or the cache has aged out.

    Assets that report no volume (Yahoo returns 0 for FX pairs, for instance)
    come back as columns of zeros — surfaced honestly downstream, never faked.
    """
    tickers = _clean(tickers)
    if not tickers:
        raise ValueError("No tickers supplied.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = _dv_cache_path(tickers, period)
    meta_path = _meta_path(cache_path)

    if use_cache and os.path.exists(cache_path) and _cache_is_fresh(meta_path, max_age_hours):
        return pd.read_parquet(cache_path)

    close, volume = _download_close_volume(tickers, period)
    dollar_vol = (close * volume).dropna(axis=1, how="all")
    if dollar_vol.empty:
        raise RuntimeError("No volume history for that universe.")
    if use_cache:
        _write_dollar_volume_cache(tickers, period, dollar_vol)
    return dollar_vol


def average_dollar_volume(tickers: list[str] | None = None, period: str = "2y",
                          lookback_days: int = 63, **kwargs) -> pd.Series:
    """
    Mean daily dollar volume over the last `lookback_days` (default ~3 months of
    trading), as a Series indexed by ticker. Recent volume, not a multi-year
    average, is what tells you how much a name can absorb *today*. The `period`
    matches `fetch_prices` so both share one cached download.

    Assets with no volume data (e.g. Yahoo FX pairs) return 0.0 — a flag for the
    caller to treat as "not liquidatable from this feed," never a fabricated fill.
    """
    dv = fetch_dollar_volume(tickers, period=period, **kwargs)
    return dv.tail(lookback_days).mean().fillna(0.0)


def fetch_risk_free_rate(use_cache: bool = True,
                         max_age_hours: float = CACHE_MAX_AGE_HOURS) -> float | None:
    """
    Latest US 13-week Treasury-bill yield (^IRX) as an annual decimal — e.g.
    0.0525 for 5.25%. Yahoo quotes ^IRX in percent, so we divide by 100.

    This is the risk-free leg of the Sharpe ratio. Returns None if the fetch
    fails — we never fabricate a rate, so Sharpe hides rather than lying.
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
                          progress=False, timeout=20)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rate = float(close.dropna().iloc[-1]) / 100.0
    except Exception:  # noqa: BLE001 — no rate is better than a fake one
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
