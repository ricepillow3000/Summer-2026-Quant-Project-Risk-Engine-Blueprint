"""
bon voyage - long-only defensive pairing.

how the math works:
- tie a high-flyer (circle A) to a defensive anchor (circle B) so a two-asset,
  long-only blend cushions A's falls. NOT pairs trading (no short leg, no
  cointegration) - it's core-satellite with periodic rebalancing.
- tail metric: expected shortfall at 97.5% (the Basel FRTB level). on ~2y of
  daily data that averages ~12 tail days; cvar@99 on 252 days is ~2.5 days,
  which is just noise. the "safety line" between the circles is the tail
  gap: ES(A) - ES(B).
- anchor screen: rank candidates by low correlation to the basket's main
  factor (PC1), low vol and a shallow ES. we rank, we never promise. in an
  all-tech basket everything loads on PC1, so a real near-zero usually needs
  another asset class (long treasuries, gold, staples/utilities/min-vol etfs).
- backtest: replays ACTUAL daily returns (fake data only lives in the
  tests). fixed target weights, periodic rebalance, no lookahead: day t's
  weight was set at most one rebalance period before t.

honest limits:
- long-only stocks have no floor above zero. diversifying CUSHIONS a fall,
  nothing here CAPS it. in a crisis correlations head to +1, so the cushion
  shrinks right when you need it - the crisis table measures that per regime.
- backtests describe one history, they don't forecast.
"""

import numpy as np
import pandas as pd

from src.eigenrisk import align_eigenvector_signs
from src.risk import cvar
from src.covariance import TRADING_DAYS

TAIL_CONFIDENCE = 0.975          # ES level (Basel FRTB)
DEFAULT_W_A = 0.60               # high-flyer weight in the pair
REBALANCE_DAYS = 21              # ~monthly

# defensive picks from OUTSIDE a stock basket (an all-tech basket can't hedge
# itself). screened like any other candidate; prices come live from yahoo.
DEFENSIVE_ANCHOR_TICKERS = ["TLT", "GLD", "XLP", "XLU", "USMV"]


def expected_shortfall(port_returns: pd.Series,
                       confidence: float = TAIL_CONFIDENCE) -> float:
    """ES: average loss on the worst (1-confidence) share of days, as a
    positive number. just risk.cvar at the FRTB level - one estimator, not two."""
    return cvar(port_returns, confidence)


def pc1_factor_correlations(returns: pd.DataFrame) -> pd.Series:
    """each asset's correlation with the basket's main risk factor (PC1 of
    the correlation matrix, signs fixed so it points the "market" way)."""
    corr = returns.corr()
    vals, vecs = np.linalg.eigh(corr.values)
    vecs = align_eigenvector_signs(vecs)
    v1 = vecs[:, -1]                                   # eigh: ascending order
    # a correlation-matrix eigenvector lives on STANDARDIZED returns - raw
    # ones would let high-vol names hijack the factor (Jolliffe, PCA)
    standardized = (returns - returns.mean()) / returns.std()
    factor = pd.Series(standardized.values @ v1, index=returns.index)
    return returns.corrwith(factor).rename("pc1_corr")


def _rank01(raw: pd.Series, higher_is_better: bool) -> pd.Series:
    """percentile rank in [0, 1] within the candidates (same idea as grit.py)."""
    ranked = raw.rank(pct=True)
    return ranked if higher_is_better else 1.0 - ranked + 1.0 / len(raw)


def anchor_rank(returns: pd.DataFrame, high_flyer: str,
                grit: pd.Series | None = None,
                direction: str = "long") -> pd.DataFrame:
    """
    rank every OTHER column as an anchor for `high_flyer`. score = average of
    percentile ranks for:
      - low |PC1 correlation| (independent of the main factor)
      - low annual vol        (steady)
      - shallow ES@97.5       (thin own tail)
      - high grit score       (only if given)
    direction="short": you're SHORT the flyer (pass its column already
    negated). a short's worst day is a squeeze - a sharp rally - and squeezes
    ride sector rallies, when defensives don't help. so the first part flips:
    the cushion is a LONG that moves WITH the shorted name (a classic
    sector-hedged short). returns best anchor first. ranks, never promises.
    """
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    if high_flyer not in returns.columns:
        raise ValueError(f"{high_flyer} not in returns")
    cands = returns.drop(columns=[high_flyer])
    if cands.shape[1] == 0:
        raise ValueError("no anchor candidates")

    pc1 = pc1_factor_correlations(returns).drop(high_flyer)
    vol = cands.std() * np.sqrt(TRADING_DAYS)
    es = cands.apply(expected_shortfall)
    corr_a = cands.corrwith(returns[high_flyer])

    df = pd.DataFrame({
        "pc1_corr": pc1, "ann_vol": vol, "es_975": es,
        "corr_to_flyer": corr_a,
    })
    if direction == "short":
        # flyer column is negated, so corr_to_flyer < 0 = moves WITH the real
        # name = the squeeze cushion. it counts 3x: a steady but unrelated
        # defensive fails exactly when the squeeze hits, so vol and ES are
        # just tie-breakers here.
        first = _rank01(df["corr_to_flyer"], higher_is_better=False)
        parts = [first, first, first,
                 _rank01(df["ann_vol"], higher_is_better=False),
                 _rank01(df["es_975"], higher_is_better=False)]
    else:
        parts = [
            _rank01(df["pc1_corr"].abs(), higher_is_better=False),
            _rank01(df["ann_vol"], higher_is_better=False),
            _rank01(df["es_975"], higher_is_better=False),
        ]
    if grit is not None:
        g = grit.reindex(df.index).dropna()
        if len(g) >= 2:
            parts.append(_rank01(g.reindex(df.index).fillna(g.min()),
                                 higher_is_better=True))
    df["anchor_score"] = (sum(parts) / len(parts) * 100).round(1)
    return df.sort_values("anchor_score", ascending=False)


def pair_weights(returns_a: pd.Series, returns_b: pd.Series) -> dict:
    """
    risk-parity split: each leg carries EQUAL RISK, not equal dollars. for two
    assets that's exact inverse-vol:
        w_a * sigma_a = w_b * sigma_b  =>  w_a = sigma_b / (sigma_a + sigma_b)
    a flyer with ~3x the anchor's vol lands near 25/75 - the anchor needs ~3x
    the dollars to carry the same risk. long-only by construction.
    """
    sa = float(pd.Series(returns_a).dropna().std())
    sb = float(pd.Series(returns_b).dropna().std())
    if sa <= 0 or sb <= 0:
        raise ValueError("both legs need positive volatility")
    w_a = sb / (sa + sb)
    return {"w_a": w_a, "w_b": 1.0 - w_a, "vol_a": sa * np.sqrt(TRADING_DAYS),
            "vol_b": sb * np.sqrt(TRADING_DAYS)}


def tail_gap(returns: pd.DataFrame, a: str, b: str,
             confidence: float = TAIL_CONFIDENCE) -> dict:
    """the safety line: how far apart the circles are in tail-loss units."""
    es_a = expected_shortfall(returns[a], confidence)
    es_b = expected_shortfall(returns[b], confidence)
    return {"es_a": es_a, "es_b": es_b, "gap": es_a - es_b}


def _max_drawdown(path: pd.Series) -> float:
    """worst peak-to-trough fall of a value path (negative number).
    the paths start AFTER day one (they're cumprods of returns), so the 1.0
    of starting money isn't in them and cummax can't see it. clipping the
    peak up to 1.0 puts it back - otherwise a day-one loss vanishes and every
    cushion looks smaller. pass a path normalised to 1.0 of capital."""
    return float((path / path.cummax().clip(lower=1.0) - 1.0).min())


def backtest_pair(returns_a: pd.Series, returns_b: pd.Series,
                  w_a: float = DEFAULT_W_A,
                  rebalance_days: int = REBALANCE_DAYS) -> dict:
    """
    real-history backtest of the long-only pair vs holding A alone.
    start at target weights, let each leg drift with its ACTUAL daily
    returns, reset to target every `rebalance_days` trading days. day t's
    weight only depends on returns before t (no lookahead).
    """
    if not 0.0 <= w_a <= 1.0:
        raise ValueError("w_a must be in [0, 1] (long-only)")
    joined = pd.concat([returns_a.rename("a"), returns_b.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        raise ValueError("not enough overlapping history")

    val_a, val_b = w_a, 1.0 - w_a          # dollars in each leg
    pair_path, weights_a, pair_rets = [], [], []
    for i, (ra, rb) in enumerate(zip(joined["a"], joined["b"])):
        w_t = val_a / (val_a + val_b)
        weights_a.append(w_t)
        pair_rets.append(w_t * ra + (1.0 - w_t) * rb)   # day-t blend return
        val_a *= (1.0 + ra)
        val_b *= (1.0 + rb)
        pair_path.append(val_a + val_b)
        if rebalance_days and (i + 1) % rebalance_days == 0:
            total = val_a + val_b
            val_a, val_b = total * w_a, total * (1.0 - w_a)

    pair = pd.Series(pair_path, index=joined.index, name="pair")
    solo = (1.0 + joined["a"]).cumprod().rename("solo")
    pair_ret = pd.Series(pair_rets, index=joined.index)

    dd_solo, dd_pair = _max_drawdown(solo), _max_drawdown(pair)
    return {
        "pair_path": pair, "solo_path": solo,
        "weights_a": pd.Series(weights_a, index=joined.index),
        "total_return_pair": float(pair.iloc[-1] - 1.0),
        "total_return_solo": float(solo.iloc[-1] - 1.0),
        "max_dd_pair": dd_pair, "max_dd_solo": dd_solo,
        "cushion": dd_pair - dd_solo,            # positive = shallower fall
        "ann_vol_pair": float(pair_ret.std() * np.sqrt(TRADING_DAYS)),
        "ann_vol_solo": float(joined["a"].std() * np.sqrt(TRADING_DAYS)),
        "es_pair": expected_shortfall(pair_ret),
        "es_solo": expected_shortfall(joined["a"]),
        "n_days": len(joined),
    }


def crisis_cushion(a: str, b: str, w_a: float = DEFAULT_W_A,
                   short_a: bool = False) -> pd.DataFrame:
    """per-crisis cushion on REPLAYED real returns: in every historical regime
    both names traded, A's worst fall alone vs the rebalanced pair. shows the
    crises where the cushion was ~0 too (correlations converged)."""
    from src.scenarios import HISTORICAL_REGIMES, replay_returns
    rows = {}
    for name, (start, end) in HISTORICAL_REGIMES.items():
        r = replay_returns([a, b], start, end)
        if a not in r.columns or b not in r.columns or len(r) < 5:
            continue                       # name missing in this window - the ui discloses it
        bt = backtest_pair(-r[a] if short_a else r[a], r[b], w_a)
        rows[name] = {"solo_dd": bt["max_dd_solo"], "pair_dd": bt["max_dd_pair"],
                      "cushion": bt["cushion"], "days": bt["n_days"]}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.ingestion import fetch_prices
    tickers = ["PLTR", "NVDA", "TLT", "GLD", "XLP"]
    prices = fetch_prices(tickers, period="2y")
    rets = prices.pct_change().dropna()
    print("PC1 correlations:\n", pc1_factor_correlations(rets).round(2))
    ranked = anchor_rank(rets, "PLTR")
    print("\nAnchor ranking for PLTR:\n", ranked.round(3))
    best = ranked.index[0]
    tg = tail_gap(rets, "PLTR", best)
    print(f"\nTail gap PLTR vs {best}: {tg['gap']:.2%} "
          f"(ES {tg['es_a']:.2%} vs {tg['es_b']:.2%})")
    bt = backtest_pair(rets["PLTR"], rets[best])
    print(f"Backtest {bt['n_days']}d: solo dd {bt['max_dd_solo']:.1%} vs pair "
          f"dd {bt['max_dd_pair']:.1%} (cushion {bt['cushion'] * 100:+.1f} points)")
    print("\nCrisis cushion:\n", crisis_cushion("PLTR", best).round(3))
