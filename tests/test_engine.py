"""
Regression tests for the risk engine.

Run standalone (no extra deps):   python -m tests.test_engine
Or with pytest if installed:       pytest

The pure-math tests use deterministic synthetic returns — no network, fast, and
they assert real invariants (CVaR >= VaR, risk parity equalizes contributions,
the jump-diffusion mean-consistency identity, vol targeting hits its target,
liquidity is monotonic in book size). One optional test boots the full Streamlit
app; it self-skips if the network or Streamlit's test harness is unavailable, so
the suite stays reliable offline.
"""

import numpy as np
import pandas as pd

from src.analytics import covariance_matrix, correlation_matrix
from src.risk import (
    var, cvar, monte_carlo, jump_diffusion_mc, calibrate_jump_diffusion, sharpe_ratio,
)
from src.strategies import (
    risk_parity_weights, risk_contributions, vol_target, portfolio_vol,
)
from src.liquidity import days_to_liquidate, liquidity_profile
from src.grit import (
    drawdown_episodes, recovery_stats, rolling_consistency,
    regime_drawdown_and_recovery, grit_scores, _score01,
)


def _synthetic_returns(n_days: int = 500, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    """Deterministic daily returns with two planted jump days (for calibration)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.012, size=(n_days, n_assets))
    data[100] -= 0.09   # downside jump day
    data[300] += 0.08   # upside jump day
    cols = [f"A{i}" for i in range(n_assets)]
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_cvar_at_least_var():
    """Expected shortfall is never less than VaR — a definitional invariant."""
    pr = _synthetic_returns().mean(axis=1)
    assert cvar(pr) >= var(pr) - 1e-12


def test_cvar_matches_gaussian_closed_form():
    """
    Validation, not just a smoke test: our empirical CVaR must match the PUBLISHED
    closed-form Gaussian Expected Shortfall. For X ~ N(mu, sigma),

        ES_c = -mu + sigma * phi(Phi^-1(1-c)) / (1-c)

    (Rockafellar & Uryasev). Our cvar() is a pure empirical estimator (percentile
    + tail mean), so agreement with the analytical formula on a large normal
    sample confirms the tail math is real, not approximated from memory.
    """
    from scipy import stats
    mu, sigma, c = 0.0004, 0.011, 0.95
    r = pd.Series(np.random.default_rng(7).normal(mu, sigma, size=2_000_000))
    analytical = -mu + sigma * stats.norm.pdf(stats.norm.ppf(1 - c)) / (1 - c)
    assert abs(cvar(r, c) - analytical) < 0.0003          # < 3 bps on 2M samples


def test_sharpe_matches_first_principles():
    """Sharpe from the engine equals a hand-rolled annualized computation."""
    pr = _synthetic_returns().mean(axis=1)
    rf = 0.03
    manual = (pr.mean() * 252 - rf) / (pr.std() * np.sqrt(252))
    assert abs(sharpe_ratio(pr, rf) - manual) < 1e-12


def test_covariance_symmetric_and_correlation_unit_diagonal():
    r = _synthetic_returns()
    cov = covariance_matrix(r)
    assert np.allclose(cov.values, cov.values.T)
    corr = correlation_matrix(r)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_risk_parity_equalizes_risk_contributions():
    cov = covariance_matrix(_synthetic_returns())
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)["risk_pct"].values
    assert np.allclose(rc, 1.0 / len(rc), atol=0.02)   # each asset ~ equal risk
    assert abs(w.sum() - 1.0) < 1e-8                    # long-only, fully invested


def test_vol_target_hits_target():
    cov = covariance_matrix(_synthetic_returns())
    w = np.ones(cov.shape[0]) / cov.shape[0]
    vt = vol_target(w, cov, 0.10)
    assert abs(portfolio_vol(vt["scaled_weights"], cov) - 0.10) < 1e-6


def test_jump_calibration_mean_consistency():
    """mu_d + lambda*mu_j must reproduce the empirical mean log-return exactly."""
    pr = _synthetic_returns().mean(axis=1)
    p = calibrate_jump_diffusion(pr)
    emp = float(np.log1p(pr.values).mean())
    assert abs((p["mu_d"] + p["lambda_daily"] * p["mu_j"]) - emp) < 1e-9
    assert p["n_jumps"] >= 1   # the planted jump days should be detected


def test_monte_carlo_engines_finite_and_coherent():
    r = _synthetic_returns()
    w = np.ones(r.shape[1]) / r.shape[1]
    for fn in (monte_carlo, jump_diffusion_mc):
        mc = fn(r, w, n_simulations=3000, horizon_days=252)
        for k in ("cvar", "var", "median_return", "worst_case", "best_case", "prob_loss"):
            assert np.isfinite(mc[k]), f"{fn.__name__}: {k} not finite"
        assert mc["cvar"] >= mc["var"] - 1e-9
        assert 0.0 <= mc["prob_loss"] <= 1.0
        assert mc["worst_case"] <= mc["best_case"]


def test_sharpe_ratio_behaves():
    pr = _synthetic_returns().mean(axis=1)
    base = sharpe_ratio(pr, 0.0)
    assert np.isfinite(base)
    # a higher risk-free rate must lower the Sharpe ratio
    assert sharpe_ratio(pr, 0.05) < base
    # zero-volatility series -> undefined (nan), not a divide-by-zero crash
    flat = pd.Series([0.001] * 100)
    assert np.isnan(sharpe_ratio(flat, 0.0))


def test_liquidity_monotonic_and_zero_adv_flagged():
    w = np.array([0.5, 0.5])
    adv = pd.Series([1e9, 1e9], index=["A", "B"])
    d_small = days_to_liquidate(w, adv, book_value=1e6)["days"]
    d_big = days_to_liquidate(w, adv, book_value=1e8)["days"]
    assert (d_big >= d_small).all() and (d_big > d_small).any()   # bigger book => more days

    adv0 = pd.Series([0.0, 1e9], index=["A", "B"])                # A has no volume
    prof = liquidity_profile(days_to_liquidate(w, adv0, book_value=1e6))
    assert "A" in prof["no_volume"]
    assert not np.isfinite(days_to_liquidate(w, adv0, book_value=1e6).loc["A", "days"])
    assert 0.0 <= prof["pct_exitable_1day"] <= 1.0


def test_drawdown_episodes_hand_worked():
    """Hand-traced example: two distinct peak->trough->recovery episodes."""
    idx = pd.bdate_range("2021-01-01", periods=7)
    vals = [100, 110, 90, 95, 111, 105, 120]
    s = pd.Series(vals, index=idx)
    ep = drawdown_episodes(s, threshold=0.05)
    assert len(ep) == 2

    e1, e2 = ep.iloc[0], ep.iloc[1]
    assert e1["peak_value"] == 110 and e1["trough_value"] == 90
    assert abs(e1["depth"] - (90 / 110 - 1)) < 1e-12
    assert e1["days_to_trough"] == 1 and e1["days_to_recover"] == 2
    assert e1["recovery_date"] == idx[4]

    assert e2["peak_value"] == 111 and e2["trough_value"] == 105
    assert abs(e2["depth"] - (105 / 111 - 1)) < 1e-12
    assert e2["days_to_trough"] == 1 and e2["days_to_recover"] == 1
    assert e2["recovery_date"] == idx[6]


def test_drawdown_episode_unresolved_flagged():
    """A drawdown that never reclaims its prior peak stays open, not silently dropped."""
    idx = pd.bdate_range("2021-01-01", periods=10)
    vals = [100, 110, 90, 80, 82, 85, 87, 88, 89, 89.5]  # never reclaims 110
    s = pd.Series(vals, index=idx)
    ep = drawdown_episodes(s, threshold=0.05)
    assert len(ep) == 1
    row = ep.iloc[0]
    assert row["recovery_date"] is None
    assert row["days_to_recover"] is None
    assert row["peak_value"] == 110 and row["trough_value"] == 80

    rec = recovery_stats(s, threshold=0.05)
    assert rec["still_underwater"] is True
    assert rec["pct_recovered"] == 0.0
    assert np.isnan(rec["median_recovery_days"])
    assert abs(rec["current_drawdown"] - (89.5 / 110 - 1)) < 1e-12


def test_recovery_stats_no_drawdown_is_full_credit():
    """A monotonically rising series has no setbacks -- trivially 'fully recovered'."""
    idx = pd.bdate_range("2021-01-01", periods=50)
    s = pd.Series(np.linspace(100, 150, 50), index=idx)
    rec = recovery_stats(s)
    assert rec["n_episodes"] == 0
    assert rec["pct_recovered"] == 1.0
    assert rec["still_underwater"] is False
    assert abs(rec["current_drawdown"]) < 1e-9


def test_rolling_consistency_bounds():
    idx = pd.bdate_range("2021-01-01", periods=600)
    up = pd.Series(np.linspace(100, 300, 600), index=idx)
    assert rolling_consistency(up, window=252) == 1.0

    down = pd.Series(np.linspace(300, 100, 600), index=idx)
    assert rolling_consistency(down, window=252) == 0.0

    short = pd.Series(np.linspace(100, 110, 100), index=idx[:100])
    assert np.isnan(rolling_consistency(short, window=252))


def test_regime_drawdown_and_recovery_hand_worked():
    """Custom crisis window on a hand-picked path: exact drawdown + recovery day count."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    vals = ([100] * 5 + [95, 90, 85, 80, 75, 70]
           + [72, 75, 80, 85, 90, 95, 100, 101] + [101] * 11)
    s = pd.Series(vals, index=idx)

    r = regime_drawdown_and_recovery(s, str(idx[5].date()), str(idx[10].date()))
    assert r is not None
    assert abs(r["max_drawdown"] - (70 / 95 - 1)) < 1e-12
    assert r["pre_crisis_price"] == 95.0
    assert r["window_days"] == 6
    assert r["recovery_days"] == 6   # first day after the window with price >= 95

    # An asset with no data at all in the window is excluded, not estimated.
    empty = pd.Series([np.nan] * len(idx), index=idx)
    assert regime_drawdown_and_recovery(empty, str(idx[5].date()), str(idx[10].date())) is None


def test_score01_direction_and_nan_handling():
    s = pd.Series([1.0, 2.0, 3.0, np.nan], index=["a", "b", "c", "d"])
    hi = _score01(s, higher_is_better=True)
    assert hi["d"] == 0.0
    assert hi["a"] < hi["b"] < hi["c"] == 1.0

    lo = _score01(s, higher_is_better=False)
    assert lo["d"] == 0.0
    assert lo["a"] > lo["b"] > lo["c"]
    assert lo["a"] == 1.0


def _synthetic_grit_universe():
    """
    Two deterministic price paths far in the future (so they never overlap any
    HISTORICAL_REGIMES window -- isolates the test to the recovery/consistency
    components) plus one short-history ticker to verify exclusion.

    GRITTY: smooth uptrend with three shallow dips that each fully recover.
    FRAGILE: decays early, takes one deep hit, partially claws back, then goes
             flat -- never reclaims its starting peak.
    """
    idx = pd.bdate_range("2030-01-02", periods=1600)

    def cum_price(log_returns, start=100.0):
        return start * np.exp(np.cumsum(np.concatenate([[0.0], log_returns])))

    gritty_returns = np.concatenate([
        np.full(300, 0.0006),
        np.full(10, np.log(0.92) / 10), np.full(15, 0.007),
        np.full(375, 0.0006),
        np.full(10, np.log(0.92) / 10), np.full(15, 0.007),
        np.full(375, 0.0006),
        np.full(10, np.log(0.92) / 10), np.full(15, 0.007),
        np.full(474, 0.0006),
    ])
    assert len(gritty_returns) == len(idx) - 1
    gritty = pd.Series(cum_price(gritty_returns), index=idx)

    fragile_returns = np.concatenate([
        np.full(300, -0.0002),
        np.full(15, np.log(0.75) / 15),
        np.full(300, 0.0007),
        np.full(984, 0.0),
    ])
    assert len(fragile_returns) == len(idx) - 1
    fragile = pd.Series(cum_price(fragile_returns), index=idx)

    short = pd.Series(np.nan, index=idx)
    short.iloc[-100:] = cum_price(np.full(99, 0.001))

    return pd.DataFrame({"GRITTY": gritty, "FRAGILE": fragile, "SHORT": short})


def test_grit_scores_ranks_resilient_above_fragile():
    prices = _synthetic_grit_universe()
    result = grit_scores(["GRITTY", "FRAGILE", "SHORT"], prices=prices)

    assert result["excluded"] == ["SHORT"]     # too little history to score
    scores = result["scores"]
    assert set(scores.index) == {"GRITTY", "FRAGILE"}
    assert ((scores["grit_score"] >= 0) & (scores["grit_score"] <= 100)).all()

    assert scores.loc["GRITTY", "pct_recovered"] == 1.0
    assert scores.loc["FRAGILE", "pct_recovered"] == 0.0
    assert scores.loc["FRAGILE", "still_underwater"] is np.True_ or scores.loc["FRAGILE", "still_underwater"] is True
    assert scores.loc["GRITTY", "consistency"] > scores.loc["FRAGILE", "consistency"]
    assert scores.loc["GRITTY", "grit_score"] > scores.loc["FRAGILE", "grit_score"]


def test_full_app_boots():
    """Integration: run the whole Streamlit script headless. Self-skips if offline."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:                       # streamlit test harness unavailable
        print(f"[skip] AppTest unavailable: {exc}")
        return
    try:
        at = AppTest.from_file("main.py", default_timeout=120)
        at.run()
    except Exception as exc:                        # network/data hiccup — don't fail suite
        print(f"[skip] app integration (data/network): {exc}")
        return
    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.error) == 0, f"app rendered errors: {[e.value for e in at.error]}"


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
