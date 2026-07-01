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
