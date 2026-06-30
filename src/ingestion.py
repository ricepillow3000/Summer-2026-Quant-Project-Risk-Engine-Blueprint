"""
DataEngine: pulls and caches price data for QQQ + Magnificent Seven.

Why caching matters for a live product:
- yfinance hits Yahoo's servers on every call. In production, if a recruiter
  refreshes your dashboard 3x in a minute, Yahoo rate-limits you and your app
  crashes. The cache writes data to disk so repeat calls are instant and free.
"""

import os
import pandas as pd
import yfinance as yf

TICKERS = ["QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "prices.parquet")


def fetch_prices(period: str = "2y") -> pd.DataFrame:
    """
    Return daily adjusted closing prices for all tickers.
    Reads from disk cache if available; otherwise downloads and saves.

    Args:
        period: yfinance period string — "1y", "2y", "5y", etc.

    Returns:
        DataFrame with dates as index and tickers as columns.
    """
    if os.path.exists(CACHE_PATH):
        prices = pd.read_parquet(CACHE_PATH)
        print(f"[DataEngine] Loaded {len(prices)} rows from cache.")
        return prices

    print(f"[DataEngine] Cache miss — downloading {TICKERS} ...")
    raw = yf.download(TICKERS, period=period, auto_adjust=True, progress=False,
                      repair=True, timeout=30)
    prices = raw["Close"].dropna(how="all")
    if prices.empty:
        raise RuntimeError("Download returned no data. Check your internet connection.")
    prices.to_parquet(CACHE_PATH)
    print(f"[DataEngine] Saved {len(prices)} rows to cache.")
    return prices


def refresh_cache(period: str = "2y") -> pd.DataFrame:
    """Force a fresh download, overwriting the cache."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("[DataEngine] Cache cleared.")
    return fetch_prices(period)


def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily log returns — standard in quant finance."""
    return prices.pct_change().dropna()


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)
    print("\n--- Price Data (last 5 rows) ---")
    print(prices.tail())
    print("\n--- Daily Returns (last 5 rows) ---")
    print(returns.tail())
    print(f"\nShape: {returns.shape[0]} trading days x {returns.shape[1]} assets")