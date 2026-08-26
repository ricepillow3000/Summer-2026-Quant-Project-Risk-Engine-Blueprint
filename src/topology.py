"""
Risk Topology map payload - the ONE place the map's numbers are assembled.

`main.py` renders `prototypes/war_room.html` with this dict spliced in, and
`tests/batch_audit.py` re-derives every field of it from raw prices to prove
the map is measurement, not decoration. Keeping the builder here (instead of
inline in the Streamlit script) is what makes that audit possible: the audited
payload and the shipped payload are byte-identical because they are the same
function call.

Provenance of each field:
  base/assets      - rolling beta (63d Cov/Var) and realized vol (21d std,
                     annualized) computed from yfinance daily returns.
  cal/muV/muB      - `src.state_calibration.calibrate_state_dynamics`
                     (AR(1) -> OU inversion, Glasserman ch. 3).
  pairs            - `src.pairing.anchor_rank` / `pair_weights` /
                     `backtest_pair`: ES(97.5) and cushion replayed on actual
                     overlapping history, monthly rebalance, no lookahead.
  hazard           - DISCLOSED POLICY, not an estimate (see HAZARD_* below).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion import fetch_prices
from src.pairing import DEFENSIVE_ANCHOR_TICKERS, anchor_rank, pair_weights, backtest_pair
from src.risk import portfolio_daily_returns
from src.state_calibration import calibrate_state_dynamics

# Hazard perimeter policy - disclosed in the map footnote, deliberately NOT
# calibrated: it is a risk-limit convention, and the audit checks the drawn
# perimeter against exactly these constants.
HAZARD_VOL_MULT = 3.0
HAZARD_VOL_FLOOR = 0.35
HAZARD_VOL_CAP = 0.90
HAZARD_BETA_ADD = 0.8
HAZARD_BETA_CAP = 1.9

BETA_WINDOW = 63          # matches state_calibration.rolling_state_series
VOL_WINDOW = 21
TRADING_DAYS = 252


def build_map_payload(returns: pd.DataFrame, weights: pd.Series,
                      loaded: list[str], bearish: bool = False,
                      days: int = 30) -> dict:
    """
    Assemble the Risk Topology payload from live returns. Raises if the
    history is too short to calibrate - the map is never shown with
    fabricated numbers.
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

    # every unit on the map: universe + defensive anchor pool
    try:
        etf = fetch_prices(DEFENSIVE_ANCHOR_TICKERS, period="2y")
        frame = returns.join(etf.pct_change().dropna(), how="inner").dropna(axis=1)
    except Exception:  # noqa: BLE001 - offline: universe only
        frame = returns
    common = frame.index.intersection(market.index)
    mvar = float(market.loc[common].var())
    assets = []
    for t in frame.columns:
        b = float(frame[t].loc[common].cov(market.loc[common]) / mvar)
        v = float(frame[t].std() * np.sqrt(TRADING_DAYS))
        if np.isfinite(b) and np.isfinite(v):
            assets.append({"t": t, "b": round(b, 3), "v": round(v, 4),
                           "f": "Universe" if t in loaded else "Anchor pool"})

    # Bon Voyage tournament: the flyer against the top anchor candidates,
    # each linkage backtested on actual overlapping history
    pairs = []
    try:
        vols = returns.std() * np.sqrt(TRADING_DAYS)
        flyer = str(vols.idxmax())
        bt_frame = frame.copy()
        if bearish:
            bt_frame[flyer] = -bt_frame[flyer]
        rank = anchor_rank(bt_frame, flyer, direction="short" if bearish else "long")
        corr = bt_frame.corr()
        for an in list(rank.index[:4]):
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
    hbeta = round(float(min(HAZARD_BETA_CAP, beta0 + HAZARD_BETA_ADD)), 2)
    bmax = max([a["b"] for a in assets] + [hbeta, beta0])
    bmin = min([a["b"] for a in assets] + [beta0])
    vmax = max([a["v"] for a in assets] + [hvol, vol0])
    domain = {
        "b0": float(min(-1.0, np.floor((bmin - 0.25) * 2) / 2)),
        "b1": float(max(2.0, np.ceil((bmax + 0.25) * 2) / 2)),
        "v0": 0.0,
        "v1": float(max(1.0, np.ceil((vmax + 0.1) * 10) / 10)),
    }
    base_cal = cal["cal"]["base"]
    return {
        "base": {"beta": beta0, "vol": vol0},
        "hazard": {"volMax": hvol, "betaMax": hbeta},
        "assets": assets,
        "pairs": pairs,
        "muFlyer": 0.0, "muAnchor": 0.0, "muBook": 0.0,
        "cal": cal["cal"], "muV": round(float(cal["muV"]), 4),
        "muB": round(float(cal["muB"]), 3),
        "days": days,
        "domain": domain,
        "live": True,
        "provenance": f"Live · yfinance EOD · {cal['n_obs']} state obs",
        "footnote": (
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
               f"states over {cal['dispersion']['n_dates']} dates, "
               f"{cal['dispersion']['coverage_corrected']:.0%} after widening. "
               "It prices plug-in estimation error and the overlapping-window "
               "smoothing together, and never narrows the terrain. "
               if cal["dispersion"].get("measured") and cal["dispersion"]["k"] > 1.0
               else "")
            + "Stressed = calibrated shocks x 1.4 "
            "plus +0.10 shock corr, calm = x 0.7 (disclosed policy, not "
            "data). Zero-drift risk view. Hazard: 3x today's vol (floored "
            "at 35%, capped at 90%) and beta +0.8 (capped at 1.9). Pair "
            "numbers replay real history: monthly rebalance, no lookahead."
            + (" <b>Calibration clamps hit:</b> "
               + "; ".join(cal["clamp_flags"]) + "."
               if cal.get("clamp_flags") else "")
        ),
    }


WAR_ROOM_FILE = Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html"


def war_room_html(payload: dict) -> str:
    """
    The Risk Topology map (prototypes/war_room.html) with the LIVE engine
    payload spliced between its __PAYLOAD__ markers. The page renders only
    what this payload carries: every number traces back to the yfinance
    returns and the calibration/backtest code in src/. The file's built-in
    demo block is replaced wholesale here, so no demo number can leak into
    the product.
    """
    html = WAR_ROOM_FILE.read_text(encoding="utf-8")
    begin, end = "/* __PAYLOAD_BEGIN__ */", "/* __PAYLOAD_END__ */"
    i = html.index(begin) + len(begin)
    j = html.index(end)
    # The payload lands INSIDE a <script> block, and json.dumps does not escape
    # "<": a string containing "</script>" would close the block early and
    # everything after it would parse as markup. Ticker names cannot carry "<"
    # (ingestion.VALID_TICKER), but this payload also carries engine prose and
    # third-party strings, so escape at the splice rather than trusting the
    # funnel. \u003c / \u003e / \u0026 are the same JSON string to the parser;
    # \u2028 / \u2029 are JS line terminators that would break the literal.
    blob = (json.dumps(payload, allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return html[:i] + "\nconst DEMO = " + blob + ";\n" + html[j:]
