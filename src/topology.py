"""
the Risk Topology map's payload - the ONE place its numbers get built.

`main.py` splices this dict into `prototypes/war_room.html`, and
`tests/batch_audit.py` re-derives every field from raw prices. living here
(not inline in main.py) means the audited payload and the shipped one are the
same function call, so they can't drift.

where each field comes from:
  base/assets      - rolling beta (63d Cov/Var) and realized vol (21d std,
                     annualized) from yfinance daily returns.
  cal/muV/muB      - `src.state_calibration.calibrate_state_dynamics`
                     (AR(1) -> OU, Glasserman ch. 3).
  pairs            - `src.pairing.anchor_rank` / `pair_weights` /
                     `backtest_pair`: ES(97.5) and cushion replayed on real
                     overlapping history, monthly rebalance, no lookahead.
  hazard           - DISCLOSED POLICY, not an estimate (HAZARD_* below).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion import fetch_prices
from src.pairing import DEFENSIVE_ANCHOR_TICKERS, anchor_rank, pair_weights, backtest_pair
from src.risk import portfolio_daily_returns
from src.state_calibration import calibrate_state_dynamics
from src.covariance import TRADING_DAYS


# hazard perimeter = a risk-limit rule, not a fit. the footnote says so and the
# audit checks the drawn line against exactly these.
HAZARD_VOL_MULT = 3.0
HAZARD_VOL_FLOOR = 0.35
HAZARD_VOL_CAP = 0.90
HAZARD_BETA_ADD = 0.8
HAZARD_BETA_CAP = 1.9

BETA_WINDOW = 63          # matches state_calibration.rolling_state_series
VOL_WINDOW = 21


def anchor_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """
    the universe's returns plus the defensive anchors it doesn't already hold,
    on common days. a held anchor stays once, as a holding (joining the full
    pool crashed on the shared column and dropped every anchor). offline, or
    when every anchor is held: just the universe.
    """
    pool = [t for t in DEFENSIVE_ANCHOR_TICKERS if t not in returns.columns]
    try:
        etf = fetch_prices(pool, period="2y")
        return returns.join(etf.pct_change().dropna(), how="inner").dropna(axis=1)
    except Exception:  # noqa: BLE001 - offline: universe only
        return returns


def build_map_payload(returns: pd.DataFrame, weights: pd.Series,
                      loaded: list[str], bearish: bool = False,
                      days: int = 30) -> dict:
    """
    build the map payload from live returns. raises if there's too little
    history to fit - the map never shows made-up numbers.
    """
    try:
        market = fetch_prices(["SPY"], period="2y").pct_change().dropna()["SPY"]
        proxy = "S&P 500 via SPY"
    except Exception:  # noqa: BLE001 - offline: basket proxies the market
        market = returns.mean(axis=1)
        proxy = "equal-weight basket (SPY unavailable)"

    port = portfolio_daily_returns(returns, weights)
    if bearish:
        port = -port
    cal = calibrate_state_dynamics(port, market)

    # every dot on the map: universe + anchor pool
    frame = anchor_frame(returns)
    common = frame.index.intersection(market.index)
    # known, disclosed mismatch: dots are full-sample (2y) stats but the book's
    # mark is 63d beta / 21d vol. batch_audit MAP-02 re-derives this exact
    # definition, so change both together or not at all.
    mvar = float(market.loc[common].var())
    assets = []
    for t in frame.columns:
        b = float(frame[t].loc[common].cov(market.loc[common]) / mvar)
        v = float(frame[t].std() * np.sqrt(TRADING_DAYS))
        if bearish and t in loaded:
            # short book = flipped holdings, so their betas flip too (vol
            # doesn't care about sign). anchor-pool names aren't held - they're
            # the hedge, bought long - so they keep their own beta.
            b = -b
        if np.isfinite(b) and np.isfinite(v):
            assets.append({"t": t, "b": round(b, 3), "v": round(v, 4),
                           "f": "Universe" if t in loaded else "Anchor pool"})

    # the flyer vs its top anchor picks, each pair backtested on real history
    pairs = []
    try:
        vols = returns.std() * np.sqrt(TRADING_DAYS)
        flyer = str(vols.idxmax())
        bt_frame = frame.copy()
        if bearish:
            bt_frame[flyer] = -bt_frame[flyer]
        rank = anchor_rank(bt_frame, flyer, direction="short" if bearish else "long")
        corr = bt_frame.corr()
        for an in rank.index[:4]:
            pw = pair_weights(bt_frame[flyer], bt_frame[an])
            bt = backtest_pair(bt_frame[flyer], bt_frame[an], pw["w_a"])
            pairs.append({
                "fl": flyer, "an": str(an),
                "rho": round(float(corr.loc[flyer, an]), 3),
                "wFlyer": round(float(pw["w_a"]), 4),
                "real": {
                    "esSolo": round(float(bt["es_solo"]), 5),
                    "esPair": round(float(bt["es_pair"]), 5),
                    "cushion": round(float(bt["cushion"]), 5),
                    "volPair": round(float(bt["ann_vol_pair"]), 4),
                    "nDays": int(bt["n_days"]),
                },
            })
    except Exception:  # noqa: BLE001 - map still works without linkages
        pairs = []

    beta0 = round(float(cal["beta_now"]), 3)
    vol0 = round(float(cal["vol_now"]), 4)
    hvol = round(float(min(HAZARD_VOL_CAP,
                           max(HAZARD_VOL_FLOOR, HAZARD_VOL_MULT * vol0))), 2)
    # beta line = "+0.8, capped at 1.9" in the direction the book carries risk.
    # for a short that's more NEGATIVE beta - the long formula put the line on
    # the wrong side, so we apply it to |beta| and flip the sign back.
    # the map tests `betaSign * b > betaSign * betaMax` the same way.
    sign = -1 if bearish else 1
    hbeta = round(float(sign * min(HAZARD_BETA_CAP, sign * beta0 + HAZARD_BETA_ADD)), 2)
    bs = [a["b"] for a in assets]
    # hbeta goes in the min too: for a short it's the lowest thing on the
    # plane, and MAP-04 needs the domain to contain the perimeter
    bmax, bmin = max(bs + [hbeta, beta0]), min(bs + [beta0, hbeta])
    vmax = max([a["v"] for a in assets] + [hvol, vol0])
    # a short sits left of zero, so mirror the minimum window too
    b_lo, b_hi = (-2.0, 1.0) if bearish else (-1.0, 2.0)
    domain = {
        "b0": float(min(b_lo, np.floor((bmin - 0.25) * 2) / 2)),
        "b1": float(max(b_hi, np.ceil((bmax + 0.25) * 2) / 2)),
        "v0": 0.0,
        "v1": float(max(1.0, np.ceil((vmax + 0.1) * 10) / 10)),
    }
    base_cal = cal["cal"]["base"]
    return {
        "base": {"beta": beta0, "vol": vol0},
        "hazard": {"volMax": hvol, "betaMax": hbeta},
        "betaSign": sign,
        "assets": assets,
        "pairs": pairs,
        "muFlyer": 0.0, "muAnchor": 0.0, "muBook": 0.0,
        "cal": cal["cal"], "muV": round(float(cal["muV"]), 4),
        "muB": round(float(cal["muB"]), 3),
        "days": days,
        "domain": domain,
        "live": True,
        "provenance": f"Live · Yahoo daily closes · {cal['n_obs']} days fitted",
        # John, 2026-11-09: "simplify this for non technical users". A
        # plain-words lead anyone can read; the full model statement, the
        # disclosure an interviewer checks, stays one click away.
        "footnote": (
            "<b>How it is drawn:</b> fitted to this book's own past "
            f"({cal['n_obs']} days of it). "
            + (f"Tested on that past, the rings were too tight, so they are "
               f"widened x{cal['dispersion']['k']:.2f} until they held what "
               "really happened. "
               if cal["dispersion"].get("measured") and cal["dispersion"]["k"] > 1.0
               else "")
            + "Stressed makes every shock 1.4x bigger and Calm 0.7x: a set "
            "rule, not data. The danger line is a set rule too: three times "
            "today's swings, or a beta 0.8 further out."
            "<details><summary>The maths, for specialists</summary>"
            "<b>State model:</b> OU beta and log-OU volatility, calibrated "
            f"by AR(1) on rolling windows of the book's own daily returns "
            f"(market: {proxy}; {cal['n_obs']} state observations; "
            "overlapping windows smooth the mean-reversion estimate). "
            f"Shock corr {base_cal['rho']:+.2f}, leverage "
            f"{base_cal['lev']:+.2f}. "
            + (f"<b>Shocks widened x{cal['dispersion']['k']:.2f}</b>, the factor "
               "measured by walking this book's own history forward: fitted on "
               "prior data only, the nominal 68.3% ring held "
               f"{cal['dispersion']['coverage_raw']:.0%} of realized 30-day "
               f"states over {cal['dispersion']['n_dates']} dates. "
               "It prices plug-in estimation error and the overlapping-window "
               "smoothing together, and never narrows the terrain. "
               if cal["dispersion"].get("measured") and cal["dispersion"]["k"] > 1.0
               else "")
            + "Stressed = calibrated shocks x 1.4 "
            "plus +0.10 shock corr, calm = x 0.7 (disclosed policy, not "
            "data). Zero-drift risk view. Hazard: 3x today's vol (floored "
            "at 35%, capped at 90%) and beta "
            + ("-0.8 (capped at -1.9); the short book's holdings have their "
               "betas negated (anchor-pool names are not held and keep their "
               "own), so the perimeter is mirrored and danger is more negative "
               "beta"
               if bearish else "+0.8 (capped at 1.9)")
            + ". Pair numbers replay real history: monthly rebalance, no "
            "lookahead."
            + "</details>"
            + (" <b>Calibration clamps hit:</b> "
               + "; ".join(cal["clamp_flags"]) + "."
               if cal.get("clamp_flags") else "")
        ),
    }


WAR_ROOM_FILE = Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html"


def war_room_html(payload: dict) -> str:
    """
    the map page with the live payload spliced between its __PAYLOAD__
    markers. the file's demo block gets replaced whole, so no demo number can
    leak onto the real page.
    """
    html = WAR_ROOM_FILE.read_text(encoding="utf-8")
    begin, end = "/* __PAYLOAD_BEGIN__ */", "/* __PAYLOAD_END__ */"
    i = html.index(begin) + len(begin)
    j = html.index(end)
    # this lands inside a <script>, and json.dumps doesn't escape "<", so a
    # "</script>" in any string would end the block early. escape here instead
    # of trusting the ticker funnel. json.dumps' default ascii output already
    # escapes \u2028/\u2029, which would break the JS literal raw.
    blob = (json.dumps(payload, allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))
    return html[:i] + "\nconst DEMO = " + blob + ";\n" + html[j:]
