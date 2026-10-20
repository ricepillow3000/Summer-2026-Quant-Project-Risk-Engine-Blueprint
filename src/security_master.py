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

import json
import os
import time

import pandas as pd
import yfinance as yf

from src.netguard import guarded_session

# Reference data changes rarely (an ISIN never; a dividend a quarter), but two
# Yahoo calls per name, one name after another, cost 20 of the 44.8s an
# offline basket switch took (John, 2026-10-20). A complete row is kept a day
# in ONE capped file under DATA_DIR (the parquet budget in src.ingestion does
# not count it, so it carries its own cap); a row with a failed or missing
# lookup is kept an hour only - the page promises "Retried within the hour",
# and an outage must not blank a name for a day.
SM_MAX_AGE_HOURS = 24.0
SM_RETRY_HOURS = 1.0
SM_MAX_ENTRIES = 400
SM_FILE = "security_master.json"


def corporate_actions(ticker: str, lookback_years: int = 5) -> dict:
    """
    Real dividend and split events for one ticker, most recent first.

    Returns:
        {"dividends": DataFrame[date, amount], "splits": DataFrame[date, ratio],
         "n_dividends": int, "n_splits": int, "last_split": dict | None}
    """
    tk = yf.Ticker(ticker, session=guarded_session())
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=lookback_years)

    divs, splits = tk.dividends, tk.splits
    if divs is None or splits is None:
        # yfinance 1.7 returns None (not an empty Series) when the history
        # fetch behind these fails - a rate limit, a timeout, a symbol with no
        # history. That used to raise on len(None) and take the WHOLE table
        # down for every ticker. Report "unknown" (None), never "zero events":
        # a failed fetch is not evidence that nothing happened.
        empty = pd.DataFrame(columns=["date", "amount"])
        return {"dividends": empty, "splits": empty.rename(columns={"amount": "ratio"}),
                "n_dividends": None, "n_splits": None, "last_split": None}

    divs = divs[divs.index >= cutoff] if len(divs) else divs
    div_df = divs.sort_index(ascending=False).rename("amount").reset_index()
    div_df.columns = ["date", "amount"]

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
    tk = yf.Ticker(ticker, session=guarded_session())
    try:
        isin = tk.isin
    except Exception:  # noqa: BLE001 - treat any lookup failure as "unavailable"
        isin = None
    isin = isin if isin and isin != "-" else None
    return {"ticker": ticker, "isin": isin}


def _row(ticker: str, lookback_years: int) -> dict:
    """One ticker's reference row, from Yahoo."""
    ident = identifier_lookup(ticker)
    ca = corporate_actions(ticker, lookback_years=lookback_years)
    return {
        "isin": ident["isin"] or "unavailable",
        "dividends_paid": ca["n_dividends"],
        "total_dividends": (None if ca["n_dividends"] is None
                            else float(ca["dividends"]["amount"].sum()) if ca["n_dividends"]
                            else 0.0),
        "splits": ca["n_splits"],
        "last_split_date": ca["last_split"]["date"].date().isoformat() if ca["last_split"] else None,
        "last_split_ratio": ca["last_split"]["ratio"] if ca["last_split"] else None,
    }


def _store_path() -> str:
    from src.ingestion import DATA_DIR      # read at call time: tests and audits move it
    return os.path.join(DATA_DIR, SM_FILE)


def _load_store() -> dict:
    try:
        with open(_store_path(), encoding="utf-8") as fh:
            store = json.load(fh)
        return store if isinstance(store, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_store(store: dict) -> None:
    """Newest SM_MAX_ENTRIES rows, written to a temporary name and renamed into
    place, so a reader (the web process or the warmer) never sees half a file."""
    keep = sorted(store.items(), key=lambda kv: kv[1].get("fetched_at", 0))[-SM_MAX_ENTRIES:]
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(keep), fh)
    os.replace(tmp, path)


def _complete(row: dict) -> bool:
    return row.get("dividends_paid") is not None and row.get("isin") != "unavailable"


def security_master(tickers: list[str], lookback_years: int = 5, use_cache: bool = True,
                    max_age_hours: float = SM_MAX_AGE_HOURS) -> pd.DataFrame:
    """
    One row per ticker: identifiers + a corporate-action summary. This is the
    reference-data view a risk desk checks BEFORE trusting a price series --
    "is this really the same security the whole way through?"

    With use_cache, a row fetched within `max_age_hours` (SM_RETRY_HOURS for an
    incomplete one) is read from disk instead of Yahoo.
    """
    store = _load_store() if use_cache else {}
    now, fetched = time.time(), False
    rows = {}
    for t in tickers:
        hit = store.get(t)
        if hit and hit.get("years") == lookback_years:
            limit = max_age_hours if _complete(hit["row"]) else min(max_age_hours, SM_RETRY_HOURS)
            if now - hit.get("fetched_at", 0) < limit * 3600:
                rows[t] = hit["row"]
                continue
        rows[t] = _row(t, lookback_years)
        store[t] = {"row": rows[t], "fetched_at": time.time(), "years": lookback_years}
        fetched = True
    if use_cache and fetched:
        try:
            _save_store(store)
        except OSError:
            pass                            # a full or read-only disk never breaks the tab
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.ingestion import DEFAULT_UNIVERSE

    sm = security_master(DEFAULT_UNIVERSE)
    print("--- Security Master (5y corporate-action lookback) ---")
    print(sm)
    missing_isin = sm[sm["isin"] == "unavailable"].index.tolist()
    if missing_isin:
        print(f"\nISIN unavailable on the free feed for: {', '.join(missing_isin)}")
