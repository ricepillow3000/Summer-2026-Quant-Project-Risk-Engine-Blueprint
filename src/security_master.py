"""
Security master & corporate actions - free-tier identifier and event data.

Quant Deep Dive:
A security master is the reference-data backbone of a real risk desk: it maps
a ticker to its stable identifiers (ISIN, CUSIP, SEDOL) so a position survives
a ticker rename, and it tracks corporate actions (splits, dividends, mergers)
so a price series doesn't silently misrepresent history. Full commercial
security masters (Bloomberg, Refinitiv) are paid products; this module builds
the same CONCEPT from what's actually available for free:
  - ISIN: yfinance exposes it directly for many (not all) tickers.
  - SEDOL / CUSIP / full merger history: not available without a paid
    reference-data vendor. We say so explicitly rather than fabricate one.
  - Dividends & splits: yfinance's per-ticker `.dividends` / `.splits` are
    real corporate-action event histories, not derived/estimated data.

Honest limits:
  - `auto_adjust=True` (used everywhere else in this engine) already folds
    splits/dividends into the adjusted close used for returns - this module
    doesn't change any risk number, it makes the underlying events VISIBLE
    and auditable instead of silently absorbed.
  - ISIN comes back as "-" or empty for some tickers (ETFs, some non-US
    listings) on the free yfinance feed. Flagged as unavailable, not guessed.
"""

import pandas as pd
import yfinance as yf


def corporate_actions(ticker: str, lookback_years: int = 5) -> dict:
    """
    Real dividend and split events for one ticker, most recent first.

    Returns:
        {"dividends": DataFrame[date, amount], "splits": DataFrame[date, ratio],
         "n_dividends": int, "n_splits": int, "last_split": dict | None}
    """
    tk = yf.Ticker(ticker)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=lookback_years)

    divs = tk.dividends
    divs = divs[divs.index >= cutoff] if len(divs) else divs
    div_df = divs.sort_index(ascending=False).rename("amount").reset_index()
    div_df.columns = ["date", "amount"]

    splits = tk.splits
    splits = splits[splits.index >= cutoff] if len(splits) else splits
    split_df = splits.sort_index(ascending=False).rename("ratio").reset_index()
    split_df.columns = ["date", "ratio"]

    last_split = None
    if not split_df.empty:
        row = split_df.iloc[0]
        last_split = {"date": row["date"], "ratio": float(row["ratio"])}

    return {
        "dividends": div_df,
        "splits": split_df,
        "n_dividends": len(div_df),
        "n_splits": len(split_df),
        "last_split": last_split,
    }


def identifier_lookup(ticker: str) -> dict:
    """
    Stable identifiers for a ticker. Only ISIN is available on the free
    yfinance feed; SEDOL/CUSIP/FIGI cross-referencing needs a paid
    reference-data vendor (OpenFIGI's free tier covers some names but isn't
    reliable enough to present as ground truth) -- flagged, not faked.
    """
    tk = yf.Ticker(ticker)
    try:
        isin = tk.isin
    except Exception:  # noqa: BLE001 - treat any lookup failure as "unavailable"
        isin = None
    isin = isin if isin and isin != "-" else None
    return {"ticker": ticker, "isin": isin}


def security_master(tickers: list[str], lookback_years: int = 5) -> pd.DataFrame:
    """
    One row per ticker: identifiers + a corporate-action summary. This is the
    reference-data view a risk desk checks BEFORE trusting a price series --
    "is this really the same security the whole way through?"
    """
    rows = {}
    for t in tickers:
        ident = identifier_lookup(t)
        ca = corporate_actions(t, lookback_years=lookback_years)
        rows[t] = {
            "isin": ident["isin"] or "unavailable",
            "dividends_paid": ca["n_dividends"],
            "total_dividends": float(ca["dividends"]["amount"].sum()) if ca["n_dividends"] else 0.0,
            "splits": ca["n_splits"],
            "last_split_date": ca["last_split"]["date"].date().isoformat() if ca["last_split"] else None,
            "last_split_ratio": ca["last_split"]["ratio"] if ca["last_split"] else None,
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.ingestion import DEFAULT_UNIVERSE

    sm = security_master(DEFAULT_UNIVERSE)
    print("--- Security Master (5y corporate-action lookback) ---")
    print(sm)
    missing_isin = sm[sm["isin"] == "unavailable"].index.tolist()
    if missing_isin:
        print(f"\nISIN unavailable on the free feed for: {', '.join(missing_isin)}")
