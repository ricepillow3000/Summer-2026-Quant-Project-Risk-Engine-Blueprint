"""Receipts: every showcase function gives the SAME numbers as the full engine in src/.
Offline, on made-up practice prices (only for this test). Run: python showcase/test_showcase.py"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import data, portfolio as pf, risk, stress  # noqa: E401,E402
from src import analytics, grit as egrit, liquidity, risk as erisk, scenarios, strategies  # noqa: E402
from src.ingestion import get_returns  # noqa: E402


# fake practice market: bumpy days that come in bunches, one stock lists late, one has a gap
def practice_market():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("1999-01-04", "2024-12-31")
    vol = np.exp(np.convolve(rng.normal(0, 0.35, len(days)), np.ones(20) / 20, "same")) * 0.012
    cols = {}
    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        move = rng.standard_t(4, len(days)) * vol * (0.7 + 0.2 * i) + 0.0003
        cols[t] = 50 * np.exp(np.cumsum(move))
    p = pd.DataFrame(cols, index=days)
    p.loc[:"2009-06-30", "DDD"] = np.nan  # lists late
    p.iloc[300:303, 1] = np.nan           # a gap in the feed
    return p


FULL = practice_market()
PRICES = FULL.dropna().iloc[-500:]
R = data.returns(PRICES)
W = np.ones(R.shape[1]) / R.shape[1]
DAY = risk.book(R, W)


def same(a, b):
    assert np.allclose(a, b, rtol=1e-12, atol=1e-15, equal_nan=True), (a, b)


def test_returns_match():
    same(R.to_numpy(), get_returns(PRICES).to_numpy())


def test_var_cvar_and_backtest_match():
    same(risk.var(DAY), erisk.var(DAY))
    same(risk.cvar(DAY), erisk.cvar(DAY))
    mine, real = risk.backtest(DAY), erisk.var_backtest(DAY)
    assert (mine["breaches"], mine["n"], mine["lr"], mine["passed"]) == \
           (real["breaches"], real["n"], real["kupiec_lr"], real["passed"])


def test_both_simulators_match():
    assert risk.block_len(DAY.values) == erisk.mean_block_length(DAY.values)
    for mine, real in [(risk.monte_carlo(R, W, 2000), erisk.monte_carlo(R, W, 2000)),
                       (risk.monte_carlo(R.iloc[:20], W, 500, 21), erisk.monte_carlo(R.iloc[:20], W, 500, 21)),
                       (risk.jump_mc(R, W, 2000), erisk.jump_diffusion_mc(R, W, 2000))]:
        for k_mine, k_real in [("var", "var"), ("cvar", "cvar"), ("median", "median_return"), ("prob_loss", "prob_loss")]:
            same(mine[k_mine], real[k_real])


def test_portfolio_math_matches():
    c = pf.cov(R)
    same(c.to_numpy(), analytics.covariance_matrix(R).to_numpy())
    same(pf.risk_share(W, c).to_numpy(), strategies.risk_contributions(W, c)["risk_pct"].to_numpy())
    same(pf.risk_parity(c), strategies.risk_parity_weights(c))
    same(pf.vol_target(W, c)["leverage"], strategies.vol_target(W, c)["leverage"])
    adv = pd.Series([2e6, 5e5, 0.0, 1e7], index=R.columns)
    same(pf.days_to_exit(W, adv).to_numpy(), liquidity.days_to_liquidate(W, adv)["days"].to_numpy())


def test_crash_replay_and_grit_match():
    real_fetch = scenarios.fetch_prices
    scenarios.fetch_prices = lambda *a, **k: FULL
    try:
        for a, b in stress.CRISES.values():
            mine, real = stress.replay(FULL, a, b), scenarios.replay_returns(list(FULL), a, b)
            assert list(mine.columns) == list(real.columns)
            same(mine.to_numpy(), real.to_numpy())
    finally:
        scenarios.fetch_prices = real_fetch
    assert stress.CRISES == scenarios.HISTORICAL_REGIMES
    mine = stress.grit(FULL)["grit"].sort_index()
    real = egrit.grit_scores(list(FULL), prices=FULL)["scores"]["grit_score"].astype(float).sort_index()
    same(mine.to_numpy(), real.to_numpy())


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for t in tests:
        t()
        print("PASS ", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
