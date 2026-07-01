"""
Liquidity modeling — how many days to unwind the book, from real volume.

Quant Deep Dive:
Volatility is not the only risk. A position you cannot exit is one that keeps
hurting you long after the model says "sell." Liquidity risk asks a different
question: if you HAD to unwind this portfolio, how long would it take — without
becoming the market and moving the price against yourself?

Method (participation-rate model, the industry standard):
- ADV_i = average daily DOLLAR volume of asset i (price x shares, averaged over
  a recent lookback window). Comes straight from Yahoo volume data — never
  estimated.
- You can realistically trade only a fraction of ADV per day before your own
  order flow moves the price. That cap is the PARTICIPATION RATE (typically
  10-20% on a risk desk).
- Position_i          = book_value x weight_i
- days_to_liquidate_i = Position_i / (participation_rate x ADV_i)

The portfolio's liquidation horizon is the SLOWEST leg — you are not flat until
the last share is sold. The dollar-weighted average describes the typical name,
and the share exitable in a single day is the number a risk memo leads with.
"""

import numpy as np
import pandas as pd


def days_to_liquidate(weights, adv: pd.Series, book_value: float = 1_000_000.0,
                      participation_rate: float = 0.20) -> pd.DataFrame:
    """
    Days to unwind each position under a participation-rate cap.

    Args:
        weights: portfolio weights, aligned to `adv`'s index order. May sum to
            more than 1 under a leverage overlay — that is intentional, since a
            levered book has more notional to sell.
        adv: average daily DOLLAR volume per asset, indexed by ticker.
        book_value: total dollars invested (position sizes scale from this).
        participation_rate: max fraction of a name's ADV traded per day before
            your own flow moves the price.

    Returns a DataFrame indexed by ticker with weight, position_value, adv,
    daily_capacity, and days. Names with no volume (ADV <= 0, e.g. FX pairs)
    get days = inf — flagged, never faked.
    """
    w = np.asarray(weights, dtype=float)
    adv_v = adv.values.astype(float)
    position = np.abs(w) * book_value               # notional to unwind per name
    capacity = participation_rate * adv_v           # dollars tradable per day
    with np.errstate(divide="ignore", invalid="ignore"):
        days = np.where(capacity > 0, position / capacity, np.inf)

    return pd.DataFrame({
        "weight": w,
        "position_value": position,
        "adv": adv_v,
        "daily_capacity": capacity,
        "days": days,
    }, index=adv.index)


def liquidity_profile(dtl: pd.DataFrame) -> dict:
    """
    Collapse a days-to-liquidate table into the handful of numbers a risk memo
    leads with: full-exit horizon, dollar-weighted average, one-day exitable
    share, the least-liquid name, and any assets with no volume feed.
    """
    finite = dtl[np.isfinite(dtl["days"])]
    no_volume = list(dtl.index[~np.isfinite(dtl["days"])])

    # Share of the book (by weight) that can be fully sold within one day.
    total_weight = float(np.abs(dtl["weight"]).sum()) or 1.0
    pct_1day = float(np.abs(dtl.loc[dtl["days"] <= 1.0, "weight"]).sum()) / total_weight

    if not finite.empty:
        full_exit_days = float(finite["days"].max())   # flat only when slowest leg is done
        least_liquid = finite["days"].idxmax()
        w = np.abs(finite["weight"])
        weighted_avg_days = float((finite["days"] * w).sum() / w.sum())
    else:
        full_exit_days = float("inf")
        least_liquid = None
        weighted_avg_days = float("inf")

    return {
        "full_exit_days": full_exit_days,
        "weighted_avg_days": weighted_avg_days,
        "pct_exitable_1day": pct_1day,
        "least_liquid": least_liquid,
        "no_volume": no_volume,
    }


if __name__ == "__main__":
    from src.ingestion import fetch_prices, get_returns, average_dollar_volume

    prices = fetch_prices()
    loaded = list(prices.columns)
    n = len(loaded)
    weights = np.ones(n) / n

    adv = average_dollar_volume(loaded).reindex(loaded).fillna(0.0)
    dtl = days_to_liquidate(weights, adv, book_value=10_000_000, participation_rate=0.20)
    print("--- Days to liquidate a $10M equal-weight book (20% participation) ---")
    print(dtl[["position_value", "adv", "days"]].round(1))

    prof = liquidity_profile(dtl)
    print("\n--- Profile ---")
    print(f"  Full-exit horizon:   {prof['full_exit_days']:.1f} days")
    print(f"  Weighted-avg horizon:{prof['weighted_avg_days']:.1f} days")
    print(f"  Exitable in 1 day:   {prof['pct_exitable_1day']:.0%}")
    print(f"  Least liquid:        {prof['least_liquid']}")
    if prof["no_volume"]:
        print(f"  No volume feed:      {', '.join(prof['no_volume'])}")
