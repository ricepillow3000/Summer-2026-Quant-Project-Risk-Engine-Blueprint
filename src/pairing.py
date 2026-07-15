"""
Bon Voyage - long-only defensive pairing.

Quant Deep Dive:
- The idea: tether a high-flying asset (Circle A) to a defensive anchor
  (Circle B) so a two-asset, long-only blend cushions A's drawdowns. This is
  NOT pairs trading (no short leg, no cointegration spread) - it is a
  core-satellite defensive pairing with periodic rebalancing.
- Tail metric: Expected Shortfall at 97.5% (Basel FRTB standard). ES@97.5 on
  ~2y of daily data averages ~12 tail observations - defensible, where
  CVaR@99 on 252 days averages ~2.5 observations (noise). The "safety line"
  between the circles is the TAIL GAP: ES(A) - ES(B).
- Anchor screen: rank candidates by LOW correlation to the basket's dominant
  eigenvector (PC1), low realized vol, and shallow ES. We rank - we never
  promise. In an all-tech universe every name loads positively on PC1 by
  construction; a true near-zero loading usually needs another asset class
  (long Treasuries, gold, staples/utilities/min-vol ETFs).
- Backtest: replays ACTUAL daily returns (no simulated paths in the product;
  synthetic data lives only in the test suite). Fixed weights, periodic
  rebalancing, no lookahead: the weight applied on day t was set at most one
  rebalance period before t.

Honest limits:
- Long-only equities have no floor above zero. Diversification CUSHIONS a
  fall; nothing here CAPS it. Crisis correlations converge toward +1, so the
  cushion shrinks exactly when it matters most - measured, not hidden: the
  crisis table reports the realized cushion per replayed regime.
- Backtests are in-sample descriptions of one history, not forecasts.
"""

import numpy as np
import pandas as pd

from src.eigenrisk import align_eigenvector_signs

TAIL_CONFIDENCE = 0.975          # ES level (Basel FRTB)
DEFAULT_W_A = 0.60               # high-flyer weight in the pair
REBALANCE_DAYS = 21              # ~monthly

# Defensive candidates from OUTSIDE an equity basket. Screened like any
# other candidate - listed here only so the UI can offer real anchor classes
# (an all-tech universe cannot hedge itself). Data comes live from Yahoo.
DEFENSIVE_ANCHOR_TICKERS = ["TLT", "GLD", "XLP", "XLU", "USMV"]


def expected_shortfall(port_returns: pd.Series,
                       confidence: float = TAIL_CONFIDENCE) -> float:
    """ES: average loss on the worst (1-confidence) share of days (positive number)."""
    r = pd.Series(port_returns).dropna()
    threshold = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= threshold]
    return float(-tail.mean())


def es_confidence_interval(port_returns: pd.Series,
                           confidence: float = TAIL_CONFIDENCE,
                           n_boot: int = 500, seed: int = 7) -> tuple[float, float]:
    """Bootstrap 90% CI for ES - honesty about tail sampling error."""
    r = pd.Series(port_returns).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    stats = np.array([expected_shortfall(pd.Series(r[row]), confidence)
                      for row in idx])
    lo, hi = np.percentile(stats, [5, 95])
    return float(lo), float(hi)


def pc1_factor_correlations(returns: pd.DataFrame) -> pd.Series:
    """
    Correlation of each asset's daily returns with the universe's dominant
    risk factor (PC1 of the correlation matrix, sign-aligned so the factor
    represents the common "market" direction).
    """
    corr = returns.corr()
    vals, vecs = np.linalg.eigh(corr.values)
    vecs = align_eigenvector_signs(vecs)
    v1 = vecs[:, -1]                                   # eigh: ascending order
    # A correlation-matrix eigenvector is defined on STANDARDIZED returns -
    # projecting raw returns would let high-vol names hijack the factor
    # (audit pass 8, trace: Jolliffe, Principal Component Analysis).
    standardized = (returns - returns.mean()) / returns.std()
    factor = pd.Series(standardized.values @ v1, index=returns.index)
    return returns.corrwith(factor).rename("pc1_corr")


def _rank01(raw: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile rank in [0, 1] within the candidate set (grit.py pattern)."""
    ranked = raw.rank(pct=True)
    return ranked if higher_is_better else 1.0 - ranked + 1.0 / len(raw)


def anchor_rank(returns: pd.DataFrame, high_flyer: str,
                grit: pd.Series | None = None,
                direction: str = "long") -> pd.DataFrame:
    """
    Rank every OTHER column of `returns` as an anchor candidate for
    `high_flyer`. Components (equal-weight composite of percentile ranks):
      - low |PC1 correlation|  (independence from the dominant factor)
      - low annualized vol     (steadiness)
      - shallow ES@97.5        (thin own-tail)
      - high grit score        (only if provided - resilience record)
    direction="short": the caller is SHORT the flyer (pass the flyer column
    already negated). A short's worst day is a squeeze - a sharp rally - and
    squeezes ride sector rallies, when defensives don't help. So the first
    component flips: the cushion for a short is a LONG that moves WITH the
    shorted name (high correlation to the raw asset = low correlation to the
    negated flyer column), rallying alongside the squeeze. Classic
    sector-hedged short. Other components unchanged.
    Returns a DataFrame sorted best anchor first. Ranks, never promises.
    """
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    if high_flyer not in returns.columns:
        raise ValueError(f"{high_flyer} not in returns")
    cands = returns.drop(columns=[high_flyer])
    if cands.shape[1] == 0:
        raise ValueError("no anchor candidates")

    pc1 = pc1_factor_correlations(returns).drop(high_flyer)
    vol = cands.std() * np.sqrt(252)
    es = cands.apply(expected_shortfall)
    corr_a = cands.corrwith(returns[high_flyer])

    df = pd.DataFrame({
        "pc1_corr": pc1, "ann_vol": vol, "es_975": es,
        "corr_to_flyer": corr_a,
    })
    if direction == "short":
        # flyer column arrives negated: corr_to_flyer < 0 means the candidate
        # moves WITH the underlying asset - exactly the squeeze cushion. The
        # cushion DOMINATES this screen (weight 3x): a steady but uncorrelated
        # defensive fails precisely when the squeeze hits, so steadiness and
        # tail-depth are tie-breakers here, not co-equal votes.
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
    Risk-parity split for the two circles: each leg contributes EQUAL RISK,
    not equal dollars. For two assets this is exact inverse-volatility:
        w_a * sigma_a = w_b * sigma_b  =>  w_a = sigma_b / (sigma_a + sigma_b)
    A high-flyer at ~3x the anchor's vol naturally lands near 25/75 - the
    anchor holds most of the CAPITAL precisely because the flyer holds most
    of the RISK. Long-only by construction.
    """
    sa = float(pd.Series(returns_a).dropna().std())
    sb = float(pd.Series(returns_b).dropna().std())
    if sa <= 0 or sb <= 0:
        raise ValueError("both legs need positive volatility")
    w_a = sb / (sa + sb)
    return {"w_a": w_a, "w_b": 1.0 - w_a, "vol_a": sa * np.sqrt(252),
            "vol_b": sb * np.sqrt(252)}


def tail_gap(returns: pd.DataFrame, a: str, b: str,
             confidence: float = TAIL_CONFIDENCE) -> dict:
    """The safety line: distance between the circles in tail-loss units."""
    es_a = expected_shortfall(returns[a], confidence)
    es_b = expected_shortfall(returns[b], confidence)
    return {"es_a": es_a, "es_b": es_b, "gap": es_a - es_b}


def _max_drawdown(path: pd.Series) -> float:
    """Max peak-to-trough drawdown of a value path (negative number)."""
    return float((path / path.cummax() - 1.0).min())


def backtest_pair(returns_a: pd.Series, returns_b: pd.Series,
                  w_a: float = DEFAULT_W_A,
                  rebalance_days: int = REBALANCE_DAYS) -> dict:
    """
    Real-history backtest of the long-only pair vs holding A alone.

    Mechanics (no lookahead): start at target weights; legs drift with their
    ACTUAL daily returns; every `rebalance_days` trading days the blend is
    reset to target. Day-t weights depend only on returns before t.
    """
    if not 0.0 <= w_a <= 1.0:
        raise ValueError("w_a must be in [0, 1] (long-only)")
    joined = pd.concat([returns_a.rename("a"), returns_b.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        raise ValueError("not enough overlapping history")

    val_a, val_b = w_a, 1.0 - w_a          # dollar value of each leg
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
        "ann_vol_pair": float(pair_ret.std() * np.sqrt(252)),
        "ann_vol_solo": float(joined["a"].std() * np.sqrt(252)),
        "es_pair": expected_shortfall(pair_ret),
        "es_solo": expected_shortfall(joined["a"]),
        "n_days": len(joined),
    }


def regime_labels(prices_a: pd.Series, gap: float) -> pd.Series:
    """
    Tether / Descent / Rotation labels over A's REAL price history.
    Descriptive regime study, not a signal. No lookahead: the label at t uses
    only prices up to t.
      Tether   - drawdown from running peak shallower than the tail gap
      Descent  - drawdown deeper than the tail gap (the fall is 'live')
      Rotation - back within half the gap of the running peak after a Descent
    """
    dd = prices_a / prices_a.cummax() - 1.0
    labels, in_descent = [], False
    for d in dd:
        if d <= -abs(gap):
            in_descent = True
            labels.append("Descent")
        elif in_descent and d > -abs(gap) / 2:
            in_descent = False
            labels.append("Rotation")
        else:
            labels.append("Descent" if in_descent else "Tether")
    return pd.Series(labels, index=prices_a.index, name="phase")


def crisis_cushion(a: str, b: str, w_a: float = DEFAULT_W_A,
                   short_a: bool = False) -> pd.DataFrame:
    """
    Per-crisis cushion record on REPLAYED real returns: for each historical
    regime where both assets traded, max drawdown of A alone vs the
    rebalanced pair. The honest evidence - including any crisis where the
    cushion was ~0 because correlations converged.
    """
    from src.scenarios import HISTORICAL_REGIMES, replay_returns
    rows = {}
    for name, (start, end) in HISTORICAL_REGIMES.items():
        r = replay_returns([a, b], start, end)
        if a not in r.columns or b not in r.columns or len(r) < 5:
            continue                       # asset absent in window - UI discloses omissions
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
          f"dd {bt['max_dd_pair']:.1%} (cushion {bt['cushion']:+.1%})")
    print("\nCrisis cushion:\n", crisis_cushion("PLTR", best).round(3))
