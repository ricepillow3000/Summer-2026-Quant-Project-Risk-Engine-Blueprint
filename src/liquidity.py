"""
liquidity - how many days it takes to sell the book, from real volume.

vol isn't the only risk: a position you can't get out of keeps hurting after
the model says "sell". so: if you HAD to unwind, how long would it take
without moving the price against yourself?

quant deep dive (participation-rate model, the desk standard):
- ADV_i = average daily DOLLAR volume of name i (price x shares over a recent
  window), straight from Yahoo - never estimated.
- you can only trade a slice of ADV a day before your own orders move the
  price. that slice is the PARTICIPATION RATE (usually 10-20%).
- position_i          = book_value x weight_i
- days_to_liquidate_i = position_i / (participation_rate x ADV_i)

the book's exit horizon is its SLOWEST leg - you're not flat until the last
share is gone. we also report the dollar-weighted average and the share you
can sell in one day.
"""

import numpy as np
import pandas as pd

from src.covariance import TRADING_DAYS


def days_to_liquidate(weights, adv: pd.Series, book_value: float = 1_000_000.0,
                      participation_rate: float = 0.20) -> pd.DataFrame:
    """
    days to unwind each position under a participation cap.

    weights: in `adv`'s order. can sum past 1 under leverage - on purpose,
        a levered book has more to sell.
    adv: average daily DOLLAR volume per ticker.
    book_value: total dollars in the book.
    participation_rate: max slice of a name's ADV you trade per day.

    returns weight, position_value, adv, daily_capacity and days per ticker.
    no volume (ADV <= 0, e.g. FX) means days = inf - flagged, never faked.
    """
    w = np.asarray(weights, dtype=float)
    adv_v = adv.values.astype(float)
    position = np.abs(w) * book_value               # dollars to sell per name
    capacity = participation_rate * adv_v           # dollars you can sell per day
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
    the few numbers worth leading with: full-exit days, weighted-average days,
    share sellable in one day, the slowest name, and names with no volume.
    """
    finite = dtl[np.isfinite(dtl["days"])]
    no_volume = list(dtl.index[~np.isfinite(dtl["days"])])

    # share of the book (by weight) you can fully sell in one day
    total_weight = float(np.abs(dtl["weight"]).sum()) or 1.0
    pct_1day = float(np.abs(dtl.loc[dtl["days"] <= 1.0, "weight"]).sum()) / total_weight

    if not finite.empty:
        full_exit_days = float(finite["days"].max())   # flat only when the slowest leg is done
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


def liquidity_adjusted_cvar(cvar: float, full_exit_days: float,
                            base_horizon_days: int = TRADING_DAYS) -> dict:
    """
    CVaR stretched to cover the extra days you're stuck holding the book.

    the headline CVaR assumes you're out at the horizon. if selling everything
    takes `full_exit_days` MORE days, you're exposed for base + full_exit.
    square-root-of-time scaling (the Basel liquidity-horizon rule) says risk
    grows with the sqrt of the window:

        multiplier = sqrt((base + full_exit) / base) = sqrt(1 + full_exit/base)
        LVaR       = CVaR * multiplier

    a book you can dump in a day barely moves (~1.0x); one that takes weeks
    gets a fatter tail. it only goes up with full_exit_days, so a less liquid
    book can never look safer than a more liquid one with the same CVaR.

    limits: sqrt-of-time assumes iid returns (no vol clustering), and this
    prices how far the market can move while you're stuck - NOT the bid/ask
    cost of getting out. that spread cost is Bangia, Diebold, Schuermann and
    Stroughair (1999), who add a bid/ask term instead; this isn't their model.
    """
    finite = np.isfinite(full_exit_days)
    if cvar <= 0:
        # worst 5% of years still gain: nothing to widen (scaling a gain made
        # a stuck book look better)
        mult = 1.0
    elif not finite or full_exit_days <= 0:
        # no volume = can't exit at all -> inf; already liquid -> no penalty
        mult = float("inf") if not finite else 1.0
    else:
        mult = float(np.sqrt(1.0 + full_exit_days / base_horizon_days))
    return {
        "cvar": cvar,
        "full_exit_days": full_exit_days,
        "multiplier": mult,
        "lvar": cvar * mult if np.isfinite(mult) else float("inf"),
        "base_horizon_days": base_horizon_days,
    }


if __name__ == "__main__":
    from src.ingestion import fetch_prices, average_dollar_volume

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
