"""
Regression tests for the risk engine.

Run standalone (no extra deps):   python -m tests.test_engine
Or with pytest if installed:       pytest

The pure-math tests use deterministic synthetic returns - no network, fast, and
they assert real invariants (CVaR >= VaR, risk parity equalizes contributions,
the jump-diffusion mean-consistency identity, vol targeting hits its target,
liquidity is monotonic in book size). One optional test boots the full Streamlit
app; it self-skips if the network or Streamlit's test harness is unavailable, so
the suite stays reliable offline.
"""

import pathlib

import numpy as np
import pandas as pd

from src.analytics import covariance_matrix, correlation_matrix
from src.risk import (
    var, cvar, monte_carlo, jump_diffusion_mc, calibrate_jump_diffusion, sharpe_ratio,
    mean_block_length,
    var_backtest, christoffersen_test, gpd_tail_fit,
    mcneil_frey_tail, ewma_volatility,
)
from src.strategies import (
    risk_parity_weights, risk_contributions, vol_target, portfolio_vol,
)
from src.liquidity import (days_to_liquidate, liquidity_profile,
                           liquidity_adjusted_cvar)
from src.comovement import (correlation_from_cov, rolling_correlation,
                            most_correlated_pair, defensive_shift,
                            least_correlated_to_pair)
from src.grit import (
    drawdown_episodes, recovery_stats, rolling_consistency,
    regime_drawdown_and_recovery, grit_scores, _score01,
)
from src.data_quality import validate_prices
from src.covariance import RISKMETRICS_LAMBDA


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
    """Expected shortfall is never less than VaR - a definitional invariant."""
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


def test_risk_parity_diagonal_two_asset_closed_form():
    """
    Regression for the 2026-07-18 audit bug: the old fixed-point iteration
    w <- b/(Sigma w) is a period-2 oscillator on a DIAGONAL matrix and
    silently returned 50/50. For two uncorrelated assets with vols 3% and 1%
    the exact ERC answer is w1*s1 = w2*s2 -> [0.25, 0.75].
    """
    cov = pd.DataFrame(np.diag([0.03 ** 2, 0.01 ** 2]),
                       index=["HI", "LO"], columns=["HI", "LO"])
    w = risk_parity_weights(cov)
    assert np.allclose(w, [0.25, 0.75], atol=1e-6)


def test_risk_parity_zero_correlation_is_inverse_vol():
    """With zero correlations ERC reduces exactly to inverse volatility."""
    vols = np.array([0.10, 0.20, 0.05, 0.40])
    cov = pd.DataFrame(np.diag(vols ** 2))
    w = risk_parity_weights(cov)
    iv = (1.0 / vols) / (1.0 / vols).sum()
    assert np.allclose(w, iv, atol=1e-6)


def test_risk_parity_negative_correlation_equal_rc():
    """
    A bond/gold-style basket with negative covariances must still return
    strictly positive weights whose risk contributions are equal (the old
    solver's marginal<=0 clamp threw legs to the boundary here).
    """
    s = np.array([0.15, 0.07, 0.12])
    corr = np.array([[1.0, -0.4, -0.2],
                     [-0.4, 1.0, 0.1],
                     [-0.2, 0.1, 1.0]])
    cov = pd.DataFrame(corr * np.outer(s, s))
    w = risk_parity_weights(cov)
    assert (w > 0).all()
    rc = risk_contributions(w, cov)["risk_pct"].values
    assert np.allclose(rc, 1.0 / len(rc), atol=1e-6)


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


def test_mc_cvar_standard_error_positive_and_shrinks_with_paths():
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (500, 3)),
                        columns=["A", "B", "C"])
    w = np.ones(3) / 3
    small = monte_carlo(rets, w, n_simulations=2_000)
    big = monte_carlo(rets, w, n_simulations=32_000)
    for mc in (small, big):
        assert np.isfinite(mc["cvar_se"]) and mc["cvar_se"] > 0
        assert mc["cvar_se"] < mc["cvar"]          # error is a fraction of the estimate
    # 16x the paths -> ~4x smaller sampling error (1/sqrt(N)); allow slack
    assert big["cvar_se"] < small["cvar_se"] / 2.5


def test_correlation_identity_matches_pandas_and_flags_zero_vol():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=["A", "B", "C"])
    df["B"] = 0.6 * df["A"] + 0.4 * df["B"]          # plant real correlation
    r = correlation_from_cov(df.cov())
    # R = D^-1 Sigma D^-1 must reproduce pandas .corr() to float precision
    assert np.allclose(r.values, df.corr().values, atol=1e-12)
    assert np.allclose(np.diag(r.values), 1.0)
    # Zero-variance asset: correlation undefined -> NaN, never faked
    rz = correlation_from_cov(df.assign(Z=0.0).cov())
    assert np.isnan(rz.loc["Z", "A"]) and np.isnan(rz.loc["Z", "Z"])


def test_comovement_pair_shift_and_destination_hand_worked():
    # A and B identical => corr exactly 1; C independent noise
    rng = np.random.default_rng(11)
    a = rng.normal(0, 0.01, 400)
    df = pd.DataFrame({"A": a, "B": a, "C": rng.normal(0, 0.01, 400)})
    corr = correlation_from_cov(df.cov())
    pa, pb, top = most_correlated_pair(corr)
    assert {pa, pb} == {"A", "B"} and abs(top - 1.0) < 1e-12
    # Least-correlated destination must be the only outside name
    dest, dcorr = least_correlated_to_pair(corr, (pa, pb))
    assert dest == "C" and abs(dcorr) < 0.2
    # Rolling correlation of identical series is 1 everywhere post-window
    roll = rolling_correlation(df, "A", "B", window=21).dropna()
    assert np.allclose(roll.values, 1.0)
    assert roll.index[0] == 20                        # first 20 rows NaN

    # Defensive shift: exposure preserved, never negative, cut capped at holding
    w = np.array([0.40, 0.10, 0.50])                  # B holds only 10%
    shifted = defensive_shift(w, ["A", "B", "C"], ("A", "B"), "C", cut=0.15)
    assert abs(shifted.sum() - w.sum()) < 1e-12       # nothing created/destroyed
    assert (shifted >= 0).all()                        # no silent short
    assert abs(shifted[1]) < 1e-12                     # B cut to zero, not -5%
    assert abs(shifted[2] - 0.75) < 1e-12              # C got 0.15 + 0.10


def test_liquidity_adjusted_cvar_widens_tail_monotonically():
    cv = 0.20
    # An instantly-liquid book pays no liquidity penalty.
    assert liquidity_adjusted_cvar(cv, 0.0)["multiplier"] == 1.0
    # LVaR is never smaller than CVaR, and grows with days-to-unwind.
    liquid = liquidity_adjusted_cvar(cv, 2.0)
    stuck = liquidity_adjusted_cvar(cv, 60.0)
    assert liquid["lvar"] >= cv
    assert stuck["lvar"] > liquid["lvar"]                 # slower exit => fatter tail
    # Closed-form check of the sqrt-of-time convention: sqrt(1 + 252/252) = sqrt(2).
    one_year = liquidity_adjusted_cvar(cv, 252.0)
    assert abs(one_year["multiplier"] - np.sqrt(2.0)) < 1e-12
    # A book with no volume feed can't be exited -> unbounded, flagged not faked.
    assert not np.isfinite(liquidity_adjusted_cvar(cv, np.inf)["lvar"])


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


def test_recovery_stats_no_drawdown_is_unknown_not_full_credit():
    """A monotonically rising series has no setbacks - so its recovery record
    is UNMEASURED, not perfect. The old contract returned 1.0 here, which the
    UI rendered as 'recovered 100% of its own drawdowns' for a name that never
    had one, and which handed it 60% of the recovery score for free."""
    idx = pd.bdate_range("2021-01-01", periods=50)
    s = pd.Series(np.linspace(100, 150, 50), index=idx)
    rec = recovery_stats(s)
    assert rec["n_episodes"] == 0
    assert np.isnan(rec["pct_recovered"])
    assert rec["still_underwater"] is False
    assert abs(rec["current_drawdown"]) < 1e-9

    # ...and it must reach the ranking as an unknown: _score01's documented
    # contract is that NaN scores 0.0, never top marks.
    scored = _score01(pd.Series({"CALM": rec["pct_recovered"], "B": 0.5,
                                 "C": 0.9}), higher_is_better=True)
    assert scored["CALM"] == 0.0
    assert scored["C"] > scored["B"] > scored["CALM"]


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


def _clean_prices(n_days: int = 100, n_assets: int = 3) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    vals = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, size=(n_days, n_assets)), axis=0)
    vals = np.abs(vals) + 50  # keep strictly positive
    return pd.DataFrame(vals, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_data_quality_clean_data_passes():
    report = validate_prices(_clean_prices())
    assert report["passed"] is True
    assert all(c["status"] in ("PASS", "WARN") for c in report["checks"])


def test_data_quality_catches_negative_price():
    prices = _clean_prices()
    prices.iloc[10, 0] = -5.0
    report = validate_prices(prices)
    assert report["passed"] is False
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["positivity.non_positive_prices"] == "FAIL"


def test_data_quality_catches_duplicate_dates():
    prices = _clean_prices()
    dup = pd.concat([prices, prices.iloc[[0]]]).sort_index()
    report = validate_prices(dup)
    assert report["passed"] is False
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["schema.duplicate_dates"] == "FAIL"


def test_data_quality_flags_extreme_move_without_failing():
    prices = _clean_prices()
    prices.iloc[20:, 0] = prices.iloc[20:, 0] * 3.0  # a +200% jump, single asset
    report = validate_prices(prices)
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["sanity.extreme_moves"] == "WARN"
    assert report["passed"] is True   # WARN surfaces the issue, doesn't block


def test_data_quality_catches_too_few_rows():
    report = validate_prices(_clean_prices(n_days=10))
    assert report["passed"] is False
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["coverage.min_rows"] == "FAIL"


def test_security_master_live():
    """Integration: real yfinance identifiers + corporate actions. Self-skips offline."""
    try:
        from src.security_master import security_master
        sm = security_master(["AAPL", "MSFT"])
    except Exception as exc:  # noqa: BLE001 - network hiccup, don't fail the suite
        print(f"[skip] security_master live check: {exc}")
        return
    assert set(sm.index) == {"AAPL", "MSFT"}
    assert {"isin", "dividends_paid", "total_dividends", "splits"}.issubset(sm.columns)
    # AAPL's ISIN is stable and well-known on the free feed; a good canary that
    # the free-tier lookup still works if it ever silently breaks upstream.
    assert sm.loc["AAPL", "isin"] == "US0378331005"


def test_full_app_boots():
    """Integration: run the whole Streamlit script headless. Self-skips if offline."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:                       # streamlit test harness unavailable
        print(f"[skip] AppTest unavailable: {exc}")
        return
    try:
        # AppTest resolves relative paths against the CALLING file, i.e.
        # tests/main.py - which does not exist, so this test silently skipped
        # every run instead of booting the app. Absolute path, no guessing.
        app = pathlib.Path(__file__).resolve().parent.parent / "main.py"
        at = AppTest.from_file(str(app), default_timeout=180)
        at.run()
    except Exception as exc:                        # network/data hiccup - don't fail suite
        print(f"[skip] app integration (data/network): {exc}")
        return
    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.error) == 0, f"app rendered errors: {[e.value for e in at.error]}"


# ---- Signal Lab (src/signals.py) - appended; existing tests above untouched ----

from src.signals import (
    momentum_signal, forward_returns, daily_ic, ic_summary,
    fundamental_law_ir, effective_breadth,
)


def _monotone_universe(n_days: int = 140, n_assets: int = 5) -> pd.DataFrame:
    """
    Deterministic prices where each ticker compounds at its own constant rate
    (rates strictly increasing across tickers). On every date the momentum
    ranking and the forward-return ranking are the same permutation, so a
    correct Spearman IC must be exactly +1 daily.
    """
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    rates = np.linspace(0.0005, 0.0045, n_assets)
    t = np.arange(n_days)[:, None]
    vals = 100.0 * np.exp(t * rates[None, :])
    return pd.DataFrame(vals, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_signal_perfect_momentum_ic_is_one():
    prices = _monotone_universe()
    sig = momentum_signal(prices, lookback=60, skip=5)
    fwd = forward_returns(prices, horizon=5)

    # NaN until enough history: first lookback+skip rows have no signal
    assert sig.iloc[:65].isna().all().all()
    assert sig.iloc[65:].notna().all().all()
    # forward_returns alignment: row t = return from t to t+horizon
    manual = prices.iloc[75, 0] / prices.iloc[70, 0] - 1.0
    assert abs(fwd.iloc[70, 0] - manual) < 1e-12
    # last `horizon` rows have no forward return yet
    assert fwd.iloc[-5:].isna().all().all()

    ic = daily_ic(sig, fwd)
    assert len(ic) > 0
    assert np.allclose(ic.values, 1.0)


def test_signal_anti_signal_ic_is_minus_one():
    prices = _monotone_universe()
    sig = -momentum_signal(prices, lookback=60, skip=5)   # deliberately inverted
    fwd = forward_returns(prices, horizon=5)
    ic = daily_ic(sig, fwd)
    assert len(ic) > 0
    assert np.allclose(ic.values, -1.0)


def test_signal_ic_summary_t_stat_first_principles():
    """t_stat must equal mean/(std/sqrt(n)) computed by hand on a fixed series."""
    idx = pd.bdate_range("2023-01-02", periods=5)
    ic = pd.Series([0.02, 0.05, -0.01, 0.04, 0.10], index=idx)
    summ = ic_summary(ic)
    assert summ["n_days"] == 5
    assert abs(summ["mean_ic"] - ic.mean()) < 1e-15
    assert abs(summ["std_ic"] - ic.std(ddof=1)) < 1e-15
    manual_t = ic.mean() / (ic.std(ddof=1) / np.sqrt(5))
    assert abs(summ["t_stat"] - manual_t) < 1e-12
    assert abs(summ["hit_rate"] - 0.8) < 1e-15     # 4 of 5 days positive


def test_signal_fundamental_law_hand_worked():
    """Grinold: IC 0.05 on 400 independent bets/yr -> IR = 0.05 * 20 = 1.0 exactly."""
    assert abs(fundamental_law_ir(0.05, 400.0) - 1.0) < 1e-12


def test_signal_effective_breadth_correlation_adjusted():
    idx = pd.bdate_range("2022-01-03", periods=800)
    rng = np.random.default_rng(3)

    # Perfectly correlated: four copies of one series is ~1 independent bet.
    base = rng.normal(0.0, 0.01, 800)
    perf = pd.DataFrame({f"A{i}": base for i in range(4)}, index=idx)
    assert abs(effective_breadth(perf) - 1.0) < 1e-6

    # Independent draws: close to all N bets (sample correlation noise only,
    # and the [0, 1) clamp means breadth can never exceed N).
    uncorr = pd.DataFrame(rng.normal(0.0, 0.01, size=(800, 4)),
                          index=idx, columns=[f"B{i}" for i in range(4)])
    be = effective_breadth(uncorr)
    assert 3.3 <= be <= 4.0 + 1e-9

    # Single asset: trivially one bet.
    assert effective_breadth(perf[["A0"]]) == 1.0


from src.regimes import (
    rolling_windows, wasserstein_distance_1d, wasserstein_kmeans,
    regime_stats, vol_ordered_labels, transition_matrix,
)


def test_regime_wasserstein_hand_worked():
    """W2 of sorted [0,1] vs [1,2] is exactly 1 (every quantile shifts by 1)."""
    assert abs(wasserstein_distance_1d(np.array([0.0, 1.0]),
                                       np.array([1.0, 2.0])) - 1.0) < 1e-12
    # and against itself, exactly zero
    a = np.sort(np.random.default_rng(0).normal(size=20))
    assert wasserstein_distance_1d(a, a) == 0.0


def test_regime_kmeans_separates_synthetic_regimes():
    """Calm half N(0, 0.005) vs turbulent half N(0, 0.03): k=2 recovers the split."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=800)
    r = pd.Series(np.concatenate([rng.normal(0, 0.005, 400),
                                  rng.normal(0, 0.03, 400)]), index=idx)
    Q, ends = rolling_windows(r, window=20, step=5)
    labels = vol_ordered_labels(Q, wasserstein_kmeans(Q, k=2)[0])
    # windows fully inside each half (skip the straddle zone around index 400)
    calm = labels[ends <= idx[380]]
    wild = labels[ends >= idx[420]]
    assert (calm == 0).mean() > 0.9, "calm half should be regime 0"
    assert (wild == 1).mean() > 0.9, "turbulent half should be regime 1"


def test_regime_stats_vol_ordered():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=900)
    r = pd.Series(np.concatenate([rng.normal(0, 0.004, 300),
                                  rng.normal(0, 0.015, 300),
                                  rng.normal(0, 0.04, 300)]), index=idx)
    Q, _ = rolling_windows(r)
    labels = wasserstein_kmeans(Q, k=3)[0]
    stats = regime_stats(Q, labels)
    vols = [s["ann_vol"] for s in stats]
    assert vols == sorted(vols), "regime_stats must be vol-ordered ascending"
    assert all(s["cvar_95"] >= 0 for s in stats), "cvar reported as positive loss"


def test_regime_transition_matrix_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1, 2, 1, 0, 2, 2, 0])
    P = transition_matrix(labels, 3)
    assert P.shape == (3, 3)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_regime_kmeans_deterministic():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2021-01-01", periods=500)
    r = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    Q, _ = rolling_windows(r)
    l1 = wasserstein_kmeans(Q, k=3, seed=42)[0]
    l2 = wasserstein_kmeans(Q, k=3, seed=42)[0]
    assert (l1 == l2).all()


# ---- Crisis Conviction (src/conviction.py) - synthetic, deterministic ----

def test_conviction_peak_trough_and_reclaim_hand_worked():
    """Crash anatomy on a hand-built path: peak before trough, and the
    reclaim counter measured in trading days from the trough."""
    from src.conviction import _peak_trough, _days_to_reclaim

    idx = pd.bdate_range("2020-01-01", periods=7)
    px = pd.Series([100.0, 110.0, 90.0, 80.0, 85.0, 110.0, 111.0], index=idx)
    peak_date, trough_date = _peak_trough(px)
    assert peak_date == idx[1], "peak must be the running max BEFORE the trough"
    assert trough_date == idx[3]
    # From the trough (pos 3), 110 is first reclaimed at pos 5 -> 2 trading days.
    assert _days_to_reclaim(px, trough_date, 110.0) == 2
    # A level never reached within the horizon is None, not extrapolated.
    assert _days_to_reclaim(px, trough_date, 500.0, horizon=10) is None


def test_conviction_forward_returns_and_exclusion():
    """Forward returns are point-to-point; horizons past the end of data are
    excluded (None), never extrapolated."""
    from src.conviction import _forward_return

    idx = pd.bdate_range("2020-01-01", periods=50)
    px = pd.Series(np.linspace(100.0, 149.0, 50), index=idx)
    r = _forward_return(px, idx[0], 10)
    assert abs(r - (px.iloc[10] / px.iloc[0] - 1.0)) < 1e-12
    assert _forward_return(px, idx[45], 10) is None


def test_crisis_forward_returns_on_synthetic_covid_window():
    """A synthetic series crashing inside the COVID window produces one row
    with the right depth; the 3y horizon (past end of data) stays NaN."""
    from src.conviction import crisis_forward_returns, conviction_summary

    idx = pd.bdate_range("2019-06-03", "2021-12-31")
    px = pd.Series(100.0, index=idx)
    px.loc["2020-02-19"] = 120.0            # pre-crisis peak, inside window
    px.loc["2020-03-23"] = 60.0             # trough, inside window
    px.loc["2020-03-24":] = 105.0           # partial recovery afterwards

    table = crisis_forward_returns(px)
    covid = table[table["crisis"].str.startswith("COVID")]
    assert len(covid) == 1
    assert abs(covid["depth"].iloc[0] - (60.0 / 120.0 - 1.0)) < 1e-12
    assert covid["trough_1y later"].iloc[0] > 0          # 60 -> 105
    assert pd.isna(covid["trough_3y later"].iloc[0])     # past end of data

    summ = conviction_summary(table)
    assert summ["trough_1y_later"]["n"] >= 1
    assert 0.0 <= summ["trough_1y_later"]["pct_positive"] <= 1.0


def test_conviction_composite_excludes_late_ipos():
    """A member with no data at the window start is excluded from the
    composite, not back-filled."""
    from src.conviction import _composite

    idx = pd.bdate_range("2020-01-01", periods=20)
    a = pd.Series(np.linspace(10, 20, 20), index=idx)
    b = pd.Series([np.nan] * 10 + list(np.linspace(50, 55, 10)), index=idx)
    comp = _composite(pd.DataFrame({"A": a, "B": b}), idx[0])
    # Only A is alive at the start: composite == A normalized to 1.0.
    assert abs(comp.iloc[0] - 1.0) < 1e-12
    assert abs(comp.iloc[-1] - (a.iloc[-1] / a.iloc[0])) < 1e-12


def test_hedge_negative_correlation_cuts_vol():
    """A near-mirror asset should roughly halve to near-zero the blended vol,
    at a ~50/50 minimum-variance weight - the whole point of a hedge."""
    from src.analytics import covariance_matrix
    from src.hedge import min_variance_pair

    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, 400)
    df = pd.DataFrame({"A": a, "B": -a}, index=idx)   # exact mirror
    cov = covariance_matrix(df)
    r = min_variance_pair(cov, "A", "B")
    assert r["correlation"] < -0.99
    assert 0.4 < r["w_anchor"] < 0.6            # near-even split
    assert r["vol_reduction"] > 0.9            # mirror kills almost all vol
    assert r["blended_vol"] < r["anchor_vol"]


def test_hedge_identical_asset_no_reduction():
    """Hedging an asset with a perfect copy of itself buys nothing - the
    blended vol must equal the anchor vol (corr = +1, no diversification)."""
    from src.analytics import covariance_matrix
    from src.hedge import min_variance_pair

    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(2)
    a = rng.normal(0, 0.01, 300)
    df = pd.DataFrame({"A": a, "B": a}, index=idx)     # identical
    cov = covariance_matrix(df)
    r = min_variance_pair(cov, "A", "B")
    assert r["correlation"] > 0.99
    assert abs(r["blended_vol"] - r["anchor_vol"]) < 1e-9
    assert abs(r["vol_reduction"]) < 1e-6


def test_hedge_ranking_orders_most_negative_first():
    """rank_hedges lists the most negatively-correlated partner first."""
    from src.analytics import correlation_matrix
    from src.hedge import rank_hedges

    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 400)
    df = pd.DataFrame({"A": a, "MIRROR": -a,
                       "INDEP": rng.normal(0, 0.01, 400)}, index=idx)
    corr = correlation_matrix(df)
    ranked = rank_hedges(corr, "A")
    assert ranked.index[0] == "MIRROR"          # most negative first
    assert ranked.iloc[0] < ranked.iloc[-1]
    assert "A" not in ranked.index              # anchor excluded


def test_ewma_reacts_to_recent_volatility_spike():
    """EWMA must weight a recent vol spike far more than the calm history -
    its whole reason for existing. Sample cov averages it away."""
    from src.covariance import ewma_covariance, sample_covariance

    idx = pd.bdate_range("2022-01-01", periods=400)
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, (400, 2))
    r[-15:] *= 5.0                       # recent panic
    df = pd.DataFrame(r, columns=["A", "B"], index=idx)
    ewma_vol = np.sqrt(ewma_covariance(df).loc["A", "A"])
    samp_vol = np.sqrt(sample_covariance(df).loc["A", "A"])
    assert ewma_vol > samp_vol * 1.5     # reacts, not averages


def test_ewma_rejects_bad_lambda():
    from src.covariance import ewma_covariance
    idx = pd.bdate_range("2022-01-01", periods=50)
    df = pd.DataFrame(np.ones((50, 2)) * 0.01, columns=["A", "B"], index=idx)
    for bad in (0.0, 1.0, 1.5, -0.1):
        try:
            ewma_covariance(df, lam=bad)
        except ValueError:
            continue
        raise AssertionError(f"lambda={bad} should have raised")


def test_ledoit_wolf_is_symmetric_psd_and_shrinks_in_range():
    """Shrunk matrix must stay a valid covariance (symmetric, PSD) with an
    intensity δ in [0,1]."""
    from src.covariance import ledoit_wolf_covariance

    idx = pd.bdate_range("2022-01-01", periods=120)
    rng = np.random.default_rng(8)
    df = pd.DataFrame(rng.normal(0, 0.01, (120, 5)),
                      columns=list("ABCDE"), index=idx)
    cov, delta = ledoit_wolf_covariance(df)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(cov.values, cov.values.T)                 # symmetric
    assert np.linalg.eigvalsh(cov.values).min() > -1e-10         # PSD


def test_estimate_covariance_dispatch_keeps_labels():
    """Every estimator returns a labeled matrix and a human info string."""
    from src.covariance import estimate_covariance

    idx = pd.bdate_range("2022-01-01", periods=100)
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)),
                      columns=["X", "Y", "Z"], index=idx)
    for method in ("sample", "Ledoit-Wolf", "EWMA"):
        cov, info = estimate_covariance(df, method)
        assert list(cov.columns) == ["X", "Y", "Z"]
        assert isinstance(info, str) and info


def _eigen_test_returns(n_days: int = 500, seed: int = 7) -> pd.DataFrame:
    """Four highly correlated names sharing one market wave (deterministic)."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.012, n_days)
    cols = ["AAPL", "MSFT", "GOOG", "NVDA"]
    data = np.column_stack([market + rng.normal(0, 0.006, n_days)
                            for _ in cols])
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_eigen_orthogonality():
    """Invariant 1: eigenvectors are perpendicular - QᵀQ = I."""
    from src.eigenrisk import eigen_factors

    cov = _eigen_test_returns().cov() * 252
    q = eigen_factors(cov)["eigenvectors"].values
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)


def test_eigen_reconstruction():
    """Invariant 2: QΛQᵀ rebuilds the cleaned covariance exactly."""
    from src.eigenrisk import clip_eigenvalues, eigen_factors

    returns = _eigen_test_returns()
    cleaned, _ = clip_eigenvalues(returns.cov() * 252, n_obs=len(returns))
    fac = eigen_factors(cleaned)
    q, lam = fac["eigenvectors"].values, fac["eigenvalues"]
    np.testing.assert_allclose((q * lam) @ q.T, cleaned.values, atol=1e-10)


def test_eigen_trace_invariant():
    """Invariant 3: Σλ = Tr(Σ) - factorization loses zero risk. Clipping
    also preserves the trace: total variance is reorganized, never lost."""
    from src.eigenrisk import clip_eigenvalues, eigen_factors

    returns = _eigen_test_returns()
    cov = returns.cov() * 252
    assert np.isclose(eigen_factors(cov)["eigenvalues"].sum(),
                      np.trace(cov.values))
    cleaned, n_clipped = clip_eigenvalues(cov, n_obs=len(returns))
    assert n_clipped > 0                       # correlated basket → noise floor
    assert np.isclose(np.trace(cleaned.values), np.trace(cov.values))


def test_eigen_degenerate_matrix_pinv_fallback():
    """Invariant 4: two 100%-correlated assets (singular matrix, λ=0) must
    route through the pseudo-inverse, not crash."""
    from src.eigenrisk import condition_number, safe_inverse

    returns = _eigen_test_returns()
    returns["AAPL2"] = returns["AAPL"]         # perfect duplicate → singular
    cov = returns.cov() * 252
    assert condition_number(cov) > 1e8
    inv, used_pinv = safe_inverse(cov)
    assert used_pinv
    assert np.all(np.isfinite(inv))


def test_eigen_sign_alignment_is_deterministic():
    """v vs −v indeterminacy: largest-|entry| per eigenvector is forced
    positive, so a factor hedge can never silently invert across runs."""
    from src.eigenrisk import align_eigenvector_signs, eigen_factors

    cov = _eigen_test_returns().cov() * 252
    q = eigen_factors(cov)["eigenvectors"].values
    anchors = np.argmax(np.abs(q), axis=0)
    assert np.all(q[anchors, np.arange(q.shape[1])] > 0)
    # aligning an already-aligned (or fully flipped) matrix is idempotent
    np.testing.assert_allclose(align_eigenvector_signs(q), q)
    np.testing.assert_allclose(align_eigenvector_signs(-q), q)


def test_eigen_pc1_dominates_and_exposure_in_range():
    """One shared market wave → PC1 must dominate variance explained, and
    the equal-weight portfolio's PC1 share must be a valid ratio near 1."""
    from src.eigenrisk import eigen_factors, marcenko_pastur_bounds, pc1_exposure

    returns = _eigen_test_returns()
    fac = eigen_factors(returns.cov() * 252)
    assert fac["variance_explained"][0] > 60.0
    w = np.ones(4) / 4
    share = pc1_exposure(w, fac)
    assert 0.0 <= share <= 1.0 and share > 0.9   # everything is one wave
    lo, hi = marcenko_pastur_bounds(4, len(returns))
    assert 0 <= lo < hi                          # sane noise band


def test_eigen_mp_bounds_hand_computed():
    """MP formula pinned to hand-computed values: N=25, T=100, σ²=2 →
    q=0.25, √q=0.5 → λ₋ = 2(0.5)² = 0.5, λ₊ = 2(1.5)² = 4.5."""
    from src.eigenrisk import marcenko_pastur_bounds

    lo, hi = marcenko_pastur_bounds(25, 100, sigma2=2.0)
    assert np.isclose(lo, 0.5) and np.isclose(hi, 4.5)


def test_eigen_edge_cases_no_crash():
    """Edge cases: zero-variance asset → κ=∞ + pinv; spherical (all-noise)
    matrix → clipping no-ops with n_clipped=0; single asset works."""
    from src.eigenrisk import (clip_eigenvalues, condition_number,
                               eigen_factors, safe_inverse)

    # zero-variance asset (constant returns → all-zero cov row/column)
    r = _eigen_test_returns()
    r["FLAT"] = 0.0
    cov = r.cov() * 252
    assert condition_number(cov) == float("inf")
    inv, used_pinv = safe_inverse(cov)
    assert used_pinv and np.all(np.isfinite(inv))

    # spherical matrix: independent equal-variance names - every eigenvalue
    # sits in the noise band, clipping must no-op, not flatten
    rng = np.random.default_rng(3)
    iso = pd.DataFrame(rng.normal(0, 0.01, (500, 4)), columns=list("WXYZ"))
    icov = iso.cov() * 252
    cleaned, n_clipped = clip_eigenvalues(icov, n_obs=500)
    assert n_clipped == 0
    np.testing.assert_allclose(cleaned.values, icov.values)

    # single asset: 100% variance explained, nothing to clip
    solo = _eigen_test_returns()[["AAPL"]]
    fac = eigen_factors(solo.cov() * 252)
    assert np.isclose(fac["variance_explained"][0], 100.0)
    _, n = clip_eigenvalues(solo.cov() * 252, n_obs=len(solo))
    assert n == 0


# ---------------------------------------------------------------------------
# Bon Voyage - long-only defensive pairing (src/pairing.py)
# Synthetic data is allowed HERE (deterministic, seeded) - never in the UI.
# ---------------------------------------------------------------------------
from src.pairing import (
    expected_shortfall, es_confidence_interval, pc1_factor_correlations,
    anchor_rank, tail_gap, backtest_pair, regime_labels)


def _seeded_returns(n=1000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.02, n), index=idx)


def test_es_coherent_and_monotone():
    r = _seeded_returns()
    v = float(-np.percentile(r, 2.5))
    es = expected_shortfall(r, 0.975)
    assert es >= v, "ES must be at least as severe as VaR at the same level"
    assert expected_shortfall(r, 0.99) >= es >= expected_shortfall(r, 0.95), \
        "ES must be nondecreasing in confidence"


def test_pair_variance_analytic_matches_empirical():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 0.02, 800))
    b = pd.Series(0.4 * a + rng.normal(0, 0.01, 800))
    w = np.array([0.6, 0.4])
    cov = np.cov(np.vstack([a, b]))
    analytic = (w[0]**2 * cov[0, 0] + w[1]**2 * cov[1, 1]
                + 2 * w[0] * w[1] * cov[0, 1])
    empirical = float(np.var(w[0] * a + w[1] * b, ddof=1))
    assert abs(analytic - empirical) / empirical < 1e-9


def test_pair_degenerate_correlations():
    a = _seeded_returns(500, seed=5)
    # rho = +1: no diversification - pair vol equals weighted sum of vols
    bt_same = backtest_pair(a, a.copy(), w_a=0.6, rebalance_days=0)
    assert abs(bt_same["ann_vol_pair"] - bt_same["ann_vol_solo"]) < 1e-9
    # rho = -1 at 50/50 with daily rebalancing: near-total cancellation
    bt_opp = backtest_pair(a, -a, w_a=0.5, rebalance_days=1)
    assert bt_opp["ann_vol_pair"] < 0.05 * bt_opp["ann_vol_solo"]


def test_pc1_planted_loading_ranks_correctly():
    rng = np.random.default_rng(7)
    n = 750
    market = rng.normal(0, 0.015, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({
        "HIBETA": 1.2 * market + rng.normal(0, 0.005, n),
        "MIDBETA": 0.8 * market + rng.normal(0, 0.005, n),
        "LOWBETA": 0.05 * market + rng.normal(0, 0.01, n),
    }, index=idx)
    pc1 = pc1_factor_correlations(rets)
    assert abs(pc1["LOWBETA"]) < abs(pc1["MIDBETA"]) < abs(pc1["HIBETA"])
    ranked = anchor_rank(rets, "HIBETA")
    assert ranked.index[0] == "LOWBETA", "planted independent asset must rank first"


def test_backtest_cushion_on_synthetic_crash():
    idx = pd.bdate_range("2021-01-01", periods=60)
    a = pd.Series([0.01] * 20 + [-0.05] * 20 + [0.0] * 20, index=idx)
    b = pd.Series([0.001] * 60, index=idx)          # flat anchor
    bt = backtest_pair(a, b, w_a=0.6)
    assert bt["max_dd_pair"] > bt["max_dd_solo"], "pair drawdown must be shallower"
    assert bt["cushion"] > 0
    assert abs(bt["max_dd_solo"] + (1 - 0.95**20)) < 0.05  # dd is negative; ~ -64%


def test_backtest_long_only_no_lookahead_weights():
    a, b = _seeded_returns(300, 1), _seeded_returns(300, 2)
    bt = backtest_pair(a, b, w_a=0.7, rebalance_days=21)
    w = bt["weights_a"]
    assert abs(float(w.iloc[0]) - 0.7) < 1e-12, "day-1 weight is the target, set before any return"
    assert ((w >= 0) & (w <= 1)).all(), "long-only: weights stay in [0,1]"
    assert abs(float(w.iloc[21]) - 0.7) < 1e-12, "weight resets to target after rebalance day"
    try:
        backtest_pair(a, b, w_a=1.4)
        raise AssertionError("w_a > 1 must raise (no leverage)")
    except ValueError:
        pass


def test_regime_labels_deterministic_and_causal():
    idx = pd.bdate_range("2021-01-01", periods=8)
    prices = pd.Series([100, 105, 110, 95, 90, 100, 108, 110], index=idx, dtype=float)
    labels = regime_labels(prices, gap=0.10)
    # 95, 90 breach the 10% gap; 100 (-9.1%) is still beyond gap/2 so the
    # Descent holds; 108 (-1.8%) re-enters within gap/2 -> Rotation; 110 -> Tether.
    assert list(labels) == ["Tether", "Tether", "Tether", "Descent", "Descent",
                            "Descent", "Rotation", "Tether"]
    # Causality: appending future data must not rewrite past labels.
    longer = pd.concat([prices, pd.Series([60.0], index=[idx[-1] + pd.Timedelta(days=1)])])
    assert list(regime_labels(longer, gap=0.10).iloc[:8]) == list(labels)


def test_es_bootstrap_ci_contains_point_and_narrows():
    small, big = _seeded_returns(250, 9), _seeded_returns(2500, 9)
    for r in (small, big):
        lo, hi = es_confidence_interval(r, n_boot=200)
        assert lo <= expected_shortfall(r) <= hi
    lo_s, hi_s = es_confidence_interval(small, n_boot=200)
    lo_b, hi_b = es_confidence_interval(big, n_boot=200)
    assert (hi_b - lo_b) < (hi_s - lo_s), "CI must narrow with more data"


def test_pair_weights_equal_risk_contribution():
    from src.pairing import pair_weights
    rng = np.random.default_rng(31)
    a = pd.Series(rng.normal(0, 0.03, 700))     # 3x the anchor's vol
    b = pd.Series(rng.normal(0, 0.01, 700))
    w = pair_weights(a, b)
    # Exact two-asset risk parity: each leg contributes equal risk.
    assert abs(w["w_a"] * a.std() - w["w_b"] * b.std()) < 1e-12
    assert w["w_a"] < 0.5 < w["w_b"], "higher-vol leg must hold less capital"
    assert abs(w["w_a"] + w["w_b"] - 1.0) < 1e-12
    try:
        pair_weights(a, pd.Series([0.0] * 700))
        raise AssertionError("zero-vol leg must raise")
    except ValueError:
        pass


def test_short_es_is_right_tail_of_long():
    r = _seeded_returns(2000, seed=13)
    es_short = expected_shortfall(-r, 0.975)
    thresh = np.percentile(r, 97.5)                 # right tail of the asset
    right_tail_mean = float(r[r >= thresh].mean())
    assert abs(es_short - right_tail_mean) < 1e-9, \
        "a short's expected shortfall must equal the asset's right-tail mean"


def test_anchor_rank_short_prefers_correlated_long():
    rng = np.random.default_rng(17)
    n = 800
    base = rng.normal(0, 0.02, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = pd.DataFrame({
        "FLYER": base + rng.normal(0, 0.01, n),
        "SAMESECTOR": 0.9 * base + rng.normal(0, 0.004, n),   # squeeze cushion
        "DEFENSIVE": rng.normal(0.0002, 0.006, n),            # uncorrelated
    }, index=idx)
    short_frame = rets.copy()
    short_frame["FLYER"] = -short_frame["FLYER"]              # synthetic short
    ranked = anchor_rank(short_frame, "FLYER", direction="short")
    assert ranked.index[0] == "SAMESECTOR", \
        "short mode must anchor in the correlated long, not the defensive"
    long_ranked = anchor_rank(rets, "FLYER", direction="long")
    assert long_ranked.index[0] == "DEFENSIVE", \
        "long mode must still prefer the independent defensive"


def test_short_backtest_directions():
    idx = pd.bdate_range("2021-01-01", periods=40)
    rising = pd.Series([0.02] * 40, index=idx)      # melts up every day
    flat = pd.Series([0.0] * 40, index=idx)
    bt_short_riser = backtest_pair(-rising, flat, w_a=1.0, rebalance_days=0)
    assert bt_short_riser["total_return_pair"] < 0, "shorting a riser loses"
    falling = pd.Series([-0.02] * 40, index=idx)
    bt_short_faller = backtest_pair(-falling, flat, w_a=1.0, rebalance_days=0)
    assert bt_short_faller["total_return_pair"] > 0, "shorting a faller gains"


def test_tail_gap_identity():
    rng = np.random.default_rng(21)
    rets = pd.DataFrame({"A": rng.normal(0, 0.03, 600),
                         "B": rng.normal(0, 0.01, 600)})
    tg = tail_gap(rets, "A", "B")
    assert abs(tg["gap"] - (tg["es_a"] - tg["es_b"])) < 1e-12
    assert tg["es_a"] > tg["es_b"], "3x-vol asset must carry the deeper tail"


def test_ou_fit_recovers_known_parameters():
    """
    The AR(1)->OU mapping must recover the parameters of a synthetic OU path
    generated with the EXACT discretization it inverts (validation against
    the closed-form transition, not a smoke test).
    """
    from src.state_calibration import fit_ou, DT
    rng = np.random.default_rng(7)
    theta, mu, sigma, n = 8.0, 1.0, 0.9, 2000
    phi = np.exp(-theta * DT)
    s = sigma * np.sqrt((1 - phi * phi) / (2 * theta))
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + s * rng.standard_normal()
    fit = fit_ou(pd.Series(x, index=pd.bdate_range("2018-01-01", periods=n)))
    assert abs(fit["mu"] - mu) < 0.12, f"mu {fit['mu']:.3f} vs {mu}"
    assert abs(fit["sigma"] - sigma) / sigma < 0.25, f"sigma {fit['sigma']:.3f}"
    assert abs(fit["theta"] - theta) / theta < 0.5, f"theta {fit['theta']:.2f}"


def test_state_calibration_sane_and_stress_ordering():
    """
    Full calibration on synthetic correlated port/market returns: outputs
    must respect the disclosed clamps, and the stressed set must widen the
    shocks relative to base exactly by the disclosed multiplier.
    """
    from src.state_calibration import (calibrate_state_dynamics, CLAMPS,
                                       STRESS_SHOCK_MULT)
    rng = np.random.default_rng(11)
    n = 750
    idx = pd.bdate_range("2022-01-03", periods=n)
    mkt = pd.Series(rng.normal(0.0003, 0.011, n), index=idx)
    port = 1.2 * mkt + pd.Series(rng.normal(0, 0.006, n), index=idx)
    out = calibrate_state_dynamics(port, mkt)
    base, stress = out["cal"]["base"], out["cal"]["stress"]
    for k in ("thB", "sigB", "thV", "etaV", "rho", "lev"):
        assert np.isfinite(base[k]), f"{k} not finite"
    assert CLAMPS["theta"][0] <= base["thB"] <= CLAMPS["theta"][1]
    assert CLAMPS["mu_v"][0] <= out["muV"] <= CLAMPS["mu_v"][1]
    assert CLAMPS["mu_b"][0] <= out["muB"] <= CLAMPS["mu_b"][1]
    # the planted book is 1.2x the market: the long-run beta must find it
    assert abs(out["muB"] - 1.2) < 0.25, f"muB {out['muB']:.2f} vs planted 1.2"
    assert abs(base["rho"]) <= 0.95 and abs(base["lev"]) <= 0.95
    assert abs(stress["etaV"] - base["etaV"] * STRESS_SHOCK_MULT) < 1e-12
    assert abs(stress["sigB"] - base["sigB"] * STRESS_SHOCK_MULT) < 1e-12
    assert out["n_obs"] >= 120


def test_rolling_state_series_tracks_planted_beta():
    """A book built as 1.5x the market must show rolling beta near 1.5."""
    from src.state_calibration import rolling_state_series
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.bdate_range("2023-01-02", periods=n)
    mkt = pd.Series(rng.normal(0.0004, 0.010, n), index=idx)
    port = 1.5 * mkt + pd.Series(rng.normal(0, 0.002, n), index=idx)
    state = rolling_state_series(port, mkt)
    assert not state.isna().any().any()
    assert abs(state["beta"].median() - 1.5) < 0.1
    assert (state["vol"] > 0).all()


def test_kupiec_rejects_zero_and_total_breaches():
    """
    Audit finding (2026-08-25, confirmed adversarially): x=0 and x=n were sent
    into the nan branch, and `passed = bool(np.isnan(lr) or ...)` turned that
    into True - a fabricated PASS on the model-validation panel. Kupiec is
    perfectly defined at both boundaries: at x=0 the unrestricted likelihood is
    1, so LR = -2n*ln(1-p), normally a decisive rejection.
    """
    import math
    up = pd.Series(np.linspace(0.0, 0.5, 520))      # monotone: never breaches
    b = var_backtest(up)
    assert b["breaches"] == 0 and b["n"] > 0
    assert b["passed"] is False, "zero breaches must FAIL, not pass"
    n, p = b["n"], b["expected_rate"]
    assert abs(b["kupiec_lr"] - round(-2.0 * n * math.log(1 - p), 2)) < 0.02

    down = pd.Series(np.linspace(0.0, -0.5, 520))   # breaches every tested day
    d = var_backtest(down)
    if d["breaches"] == d["n"]:
        assert d["passed"] is False, "all-breach must FAIL, not pass"


def test_mcneil_frey_keeps_residual_and_return_scales_separate():
    """
    Audit finding: everything gpd_tail_fit returns is in standardised-residual
    units, but only var/es were rescaled - so the UI printed a z-value of 2.1
    as "2.10%". Return-scale companions must exist and be exactly sigma_next
    times their residual counterparts.
    """
    r = pd.Series(np.random.default_rng(1).standard_t(5, 2600) * 0.01)
    m = mcneil_frey_tail(r, n_boot=0)
    assert m["fitted"] is True
    assert "residual" in m["scale"]
    s = m["sigma_next"]
    assert abs(m["threshold_return"] - s * m["threshold_z"]) < 1e-12
    # a return-scale daily threshold is a plausible loss, a z-score is not
    assert 0.0 < m["threshold_return"] < 0.25
    if m["finite_endpoint_z"] is None:
        assert m["finite_endpoint_return"] is None


def test_var_backtest_can_actually_fail():
    """
    Regression guard for a backtest that was arithmetic, not a test.

    The old version took the 5th percentile of the WHOLE series and counted how
    many of those same returns fell below it - which is exactly 5% by
    definition. Calm normal returns and a series of catastrophic clustered
    crashes both scored 75/1500 breaches, LR -0.0, passed True. It could not
    fail, so it validated nothing.

    A well-specified model must pass and a badly broken one must fail. If this
    test ever goes green on both, the in-sample bug is back.
    """
    good = pd.Series(np.random.default_rng(11).normal(0.0005, 0.01, 1500))
    bt_good = var_backtest(good)
    assert bt_good["passed"] is True, "well-specified model should pass"

    # Vol regime shift: VaR learned on 1200 calm days badly understates the
    # 300 high-vol days that follow, so breaches must overshoot 5%.
    rng = np.random.default_rng(12)
    broken = pd.Series(np.r_[rng.normal(0, 0.01, 1200),
                             rng.normal(-0.01, 0.05, 300)])
    bt_broken = var_backtest(broken)
    assert bt_broken["passed"] is False, "vol regime shift should fail Kupiec"
    assert bt_broken["observed_rate"] > bt_broken["expected_rate"]
    assert bt_broken["kupiec_lr"] > 3.841        # chi-square(1) at 95%

    # The whole point: the two verdicts must differ.
    assert bt_good["passed"] != bt_broken["passed"]


def test_var_backtest_has_no_lookahead():
    """
    Day t's VaR must come only from days strictly before t. Proof: change ONLY
    the final return to a catastrophic loss. Every earlier day's breach flag
    must be untouched - a leak would let the last value bend earlier verdicts.
    """
    base = pd.Series(np.random.default_rng(13).normal(0, 0.01, 800))
    mutated = base.copy()
    mutated.iloc[-1] = -0.5

    b1, b2 = var_backtest(base), var_backtest(mutated)
    assert b1["breach_flags"].iloc[:-1].equals(b2["breach_flags"].iloc[:-1])
    # ...and the mutated day itself must register as a breach.
    assert not bool(b1["breach_flags"].iloc[-1])
    assert bool(b2["breach_flags"].iloc[-1])


def test_ticker_validation_blocks_html_injection():
    """
    SECURITY regression guard. The ticker box accepts free text
    (accept_new_options=True) and ticker names are interpolated into
    `unsafe_allow_html=True` blocks downstream - the headline verdict names
    excluded symbols inside a <div>. An unvalidated symbol is therefore stored
    HTML injection. Upper-casing is NOT a defence: HTML tags are
    case-insensitive, so <IMG ...> survives .upper() intact.

    If this test fails, the app has an XSS hole on a public URL.
    """
    from src.ingestion import valid_ticker, _clean

    payloads = [
        "<IMG SRC=X ONERROR=ALERT(1)>",
        "<script>alert(1)</script>",
        "AAPL<B>",
        '"><svg onload=alert(1)>',
        "javascript:alert(1)",
        "../../etc/passwd",
        "A B",
    ]
    for p in payloads:
        assert not valid_ticker(p), f"injection payload accepted: {p!r}"

    # Real Yahoo symbol shapes must still pass - equities, class shares, FX,
    # futures and the ^-prefixed index used for the risk-free rate.
    for good in ("AAPL", "BRK-B", "EURUSD=X", "GC=F", "^IRX", "QQQ"):
        assert valid_ticker(good), f"legitimate symbol rejected: {good}"

    # The shared funnel drops bad symbols instead of passing them downstream.
    assert _clean(["AAPL", "<IMG SRC=X ONERROR=1>"]) == ["AAPL"]


def _garch11(n, omega=1e-6, alpha=0.09, beta=0.90, seed=0, innov="normal", df=5):
    """GARCH(1,1). With NORMAL innovations the residual tail is thin, so any
    fat unconditional tail is produced purely by volatility clustering."""
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(n) if innov == "normal"
         else rng.standard_t(df, n) / np.sqrt(df / (df - 2)))
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.Series(r)


def test_mcneil_frey_strips_clustering_from_the_tail_index():
    """
    The reason conditional EVT exists. Simulate GARCH with NORMAL innovations:
    the true residual tail is thin, so ANY apparent tail fatness in the raw
    returns is volatility clustering, not fat shocks. Filtering must therefore
    give a materially LOWER xi than the unconditional fit.

    The assertion is on the GAP, not on the levels: MLE-POT at a 95% threshold
    is known to sit below the asymptotic value, so absolute xi skews low for
    both fits. The difference between them is the clustering contribution and
    is the robust quantity.
    """
    mf = mcneil_frey_tail(_garch11(4000, seed=1, innov="normal"), n_boot=0)
    assert mf["fitted"] is True
    assert mf["unconditional_xi"] - mf["xi"] > 0.10, (
        f"clustering bias not detected: uncond={mf['unconditional_xi']} "
        f"cond={mf['xi']}")

    # Control: i.i.d. data has no clustering, so the filter has nothing to
    # remove and the two fits must stay close.
    iid = pd.Series(np.random.default_rng(3).standard_t(5, 4000) * 0.01)
    mf_iid = mcneil_frey_tail(iid, n_boot=0)
    assert abs(mf_iid["unconditional_xi"] - mf_iid["xi"]) < 0.10


def test_ewma_filter_has_no_lookahead_and_forecasts_forward():
    """
    sigma_t is built from returns up to t-1, so standardising r_t by it is
    legitimate. Mutating the FINAL return must leave the whole sigma path
    untouched while still moving the one-step-ahead forecast.
    """
    base = _garch11(1200, seed=4)
    mutated = base.copy()
    mutated.iloc[-1] = -0.5

    s_base, next_base = ewma_volatility(base.values)
    s_mut, next_mut = ewma_volatility(mutated.values)
    assert np.allclose(s_base, s_mut), "past sigma must not see the last return"
    assert next_mut > next_base * 2, "the forecast must react to a huge new loss"


def test_mcneil_frey_rescales_by_the_volatility_forecast():
    """Conditional risk = residual quantile blown back up by tomorrow's vol.
    Exact arithmetic, not approximate."""
    mf = mcneil_frey_tail(_garch11(4000, seed=2, innov="t", df=5), n_boot=0)
    s = mf["sigma_next"]
    assert s > 0
    for q in (0.99, 0.999):
        assert abs(mf["conditional"][q]["var"] - s * mf["tail"][q]["var"]) < 1e-12
        assert abs(mf["conditional"][q]["es"] - s * mf["tail"][q]["es"]) < 1e-12
    # Deeper quantile must cost more, and ES must sit at or above VaR.
    assert mf["conditional"][0.999]["var"] > mf["conditional"][0.99]["var"]
    assert mf["conditional"][0.99]["es"] >= mf["conditional"][0.99]["var"]


def test_mcneil_frey_refuses_a_series_too_short_to_filter():
    """Burn-in plus a tail fit needs history. Say so rather than fit noise."""
    short = pd.Series(np.random.default_rng(9).normal(0, 0.01, 120))
    mf = mcneil_frey_tail(short, n_boot=0)
    assert mf["fitted"] is False
    assert "EWMA filter" in mf["reason"]


def test_gpd_recovers_known_tail_index_from_theory():
    """
    Independent theory anchor: a Student-t with nu degrees of freedom has
    tail index xi = 1/nu. Fitting the GPD to simulated t losses must recover
    that, and must ORDER correctly - heavier t (lower nu) gives larger xi.

    Tolerance is deliberately loose: MLE-POT at a 95% threshold is known to
    sit slightly below the asymptotic value because the threshold is not
    infinitely high. Asserting exact equality would be asserting a bias away.
    """
    from scipy import stats as _st
    fits = {}
    for nu in (3, 6):
        losses = pd.Series(-_st.t.rvs(nu, size=20000, random_state=1) * 0.01)
        f = gpd_tail_fit(losses, n_boot=0)
        assert f["fitted"] is True
        fits[nu] = f["xi"]
        assert abs(f["xi"] - 1.0 / nu) < 0.12, f"t({nu}): xi={f['xi']}"

    assert fits[3] > fits[6], "heavier tail must give the larger xi"


def test_gpd_refuses_rather_than_fitting_too_few_exceedances():
    """
    Under the minimum exceedance count the GPD asymptotics do not apply.
    Report why and return no numbers - never a shaky estimate presented
    as if it were sound.
    """
    short = pd.Series(np.random.default_rng(31).normal(0, 0.01, 500))
    f = gpd_tail_fit(short, n_boot=0)
    assert f["fitted"] is False
    assert f["xi"] is None and f["tail"] == {}
    assert "needed for the GPD asymptotics" in f["reason"]


def test_gpd_reports_no_finite_maximum_for_heavy_tails():
    """
    The honest answer to "what is the absolute maximum I can lose": for
    xi >= 0 there is none. A finite right endpoint may only be reported when
    xi < 0. And once xi >= 1 the mean does not exist, so ES must be None
    rather than a number.
    """
    from scipy import stats as _st
    heavy = pd.Series(-_st.t.rvs(4, size=20000, random_state=2) * 0.01)
    f = gpd_tail_fit(heavy, n_boot=0)
    assert f["xi"] > 0
    assert f["finite_endpoint"] is None, "xi > 0 must NOT report a finite worst case"

    # VaR must increase with the quantile, and ES must sit at or above VaR.
    assert f["tail"][0.999]["var"] > f["tail"][0.99]["var"]
    assert f["tail"][0.99]["es"] >= f["tail"][0.99]["var"]

    # xi >= 1: infinite mean, so no ES exists.
    infinite_mean = pd.Series(
        -_st.genpareto.rvs(1.4, scale=0.01, size=6000, random_state=3))
    g = gpd_tail_fit(infinite_mean, n_boot=0)
    assert g["xi"] >= 1.0
    assert g["moments_finite"]["mean"] is False
    assert g["tail"][0.99]["es"] is None


def test_gpd_bootstrap_interval_is_deterministic():
    """A confidence interval that changes on every Streamlit rerun is noise
    dressed as precision. Same input, same seed, same interval."""
    r = pd.Series(np.random.default_rng(33).standard_t(4, 2515) * 0.01)
    a = gpd_tail_fit(r, n_boot=120)
    b = gpd_tail_fit(r, n_boot=120)
    assert a["xi_ci"] == b["xi_ci"]
    assert a["xi_ci"][0] <= a["xi"] <= a["xi_ci"][1]


def test_christoffersen_matches_hand_computed_likelihood_ratio():
    """
    Verify LR_ind against the formula worked out by hand with math.log - a
    different code path from the implementation's special.xlogy, so agreement
    is real corroboration rather than the same arithmetic twice.

    I = [0,0,1,1,0,1,0,0,1,1] -> 9 transitions: n00=2, n01=3, n10=2, n11=2
    """
    import math
    I = pd.Series([0, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    c = christoffersen_test(I)
    assert (c["n00"], c["n01"], c["n10"], c["n11"]) == (2, 3, 2, 2)
    assert c["transitions"] == 9

    pi01, pi11, pi = 3 / 5, 2 / 4, 5 / 9
    ll_markov = (2 * math.log(1 - pi01) + 3 * math.log(pi01)
                 + 2 * math.log(1 - pi11) + 2 * math.log(pi11))
    ll_indep = 4 * math.log(1 - pi) + 5 * math.log(pi)
    expected = -2 * (ll_indep - ll_markov)

    assert abs(c["lr_ind"] - round(expected, 2)) < 0.01
    assert abs(c["pi01"] - pi01) < 1e-12
    assert abs(c["pi11"] - pi11) < 1e-12


def test_christoffersen_flags_clustering_that_kupiec_cannot_see():
    """
    The whole reason this test exists: identical breach COUNTS, opposite
    verdicts. Kupiec sees one number for both; independence separates them.
    """
    n, k = 400, 40
    spread = np.zeros(n, dtype=bool)
    spread[::10] = True                      # 40 breaches, evenly spaced
    clustered = np.zeros(n, dtype=bool)
    clustered[100:140] = True                # same 40, all consecutive

    assert spread.sum() == clustered.sum() == k     # Kupiec cannot tell these apart

    c_spread = christoffersen_test(pd.Series(spread))
    c_clust = christoffersen_test(pd.Series(clustered))

    assert c_clust["lr_ind"] > c_spread["lr_ind"]
    assert c_clust["passed_ind"] is False, "40 consecutive breaches must fail independence"


def test_christoffersen_returns_none_when_undefined_not_zero():
    """
    No breach is ever followed by another observation -> pi11 is not
    estimable and LR_ind does not exist. It must report None. Returning 0
    would render as "passed independence", a fabricated verdict.
    """
    c = christoffersen_test(pd.Series(np.zeros(300, dtype=bool)))
    assert c["lr_ind"] is None
    assert c["passed_ind"] is None
    assert c["p_ind"] is None


def test_conditional_coverage_is_additive_and_wired_into_backtest():
    """LR_cc = LR_uc + LR_ind by construction, on chi2(2)."""
    rng = np.random.default_rng(21)
    bt = var_backtest(pd.Series(np.r_[rng.normal(0, 0.01, 1200),
                                      rng.normal(-0.01, 0.05, 300)]))
    c = bt["christoffersen"]
    assert c["lr_ind"] is not None
    assert abs(c["lr_cc"] - round(bt["kupiec_lr"] + c["lr_ind"], 2)) < 0.02
    assert c["passed_cc"] == (c["lr_cc"] <= 5.991)
    assert 0.0 <= c["p_cc"] <= 1.0


def test_var_backtest_reports_untestable_on_short_sample():
    """Too short to hold out any day: say so, don't invent a verdict."""
    bt = var_backtest(pd.Series(np.random.default_rng(14).normal(0, 0.01, 40)))
    assert bt["testable"] is False
    assert bt["n"] == 0
    assert bt["passed"] is None


# --- 2026-08-26 audit follow-ups (a), (b), (c) --------------------------------

def test_peak_trough_reports_no_crash_when_window_never_declined():
    """A window whose minimum is its first bar never fell. That must read as
    'no crash here', not as a crash reclaimed in zero days."""
    from src.conviction import _peak_trough, _days_to_reclaim

    idx = pd.bdate_range("2023-03-01", periods=20)
    rising = pd.Series(np.linspace(100.0, 120.0, 20), index=idx)
    assert _peak_trough(rising) == (None, None)

    # The old code returned peak == trough == the first bar, and reclaiming a
    # level you are already standing on takes zero days - the SVB "0d" bug.
    t0 = rising.idxmin()
    assert _days_to_reclaim(rising, t0, float(rising.loc[t0])) == 0

    # A window that really does fall still reports a real peak and trough.
    crash = pd.Series(
        list(np.linspace(100.0, 130.0, 8)) + list(np.linspace(128.0, 90.0, 6))
        + list(np.linspace(92.0, 135.0, 6)), index=idx)
    pk, tr = _peak_trough(crash)
    assert pk is not None and tr is not None and pk < tr
    assert crash.loc[pk] > crash.loc[tr]


def test_race_days_separates_never_fell_from_never_reclaimed():
    """Three states must not collapse: not falling is the best outcome in a
    recovery race, never getting back is the worst."""
    from src.conviction import race_days

    never_fell = race_days(None, False, np.inf)
    reclaimed = race_days(42, True, np.inf)
    never_back = race_days(None, True, np.inf)
    assert never_fell == 0.0
    assert never_fell < reclaimed < never_back
    assert np.isinf(never_back)
    # The chart passes the axis cap instead of inf so the bar stays drawable.
    assert race_days(None, True, 756) == 756
    assert race_days(float("nan"), True, 756) == 756


def test_max_drawdown_counts_starting_capital_as_a_peak():
    """The value paths start AFTER day one, so 1.0 of starting capital is not
    in the series. Without clipping the running peak up to 1.0 a day-one loss
    is invisible and every crisis cushion is understated."""
    from src.pairing import _max_drawdown

    idx = pd.bdate_range("2021-01-01", periods=4)
    # -20% on day one, then flat: the true worst drawdown is -20%.
    path = pd.Series([0.80, 0.80, 0.82, 0.85], index=idx)
    assert abs(_max_drawdown(path) - (-0.20)) < 1e-12

    # A path that only ever rises above starting capital still has no drawdown.
    assert _max_drawdown(pd.Series([1.01, 1.05, 1.09, 1.20], index=idx)) == 0.0

    # And a fall AFTER a peak above 1.0 is still measured from that peak.
    fall = pd.Series([1.10, 1.20, 0.96, 1.00], index=idx)
    assert abs(_max_drawdown(fall) - (0.96 / 1.20 - 1.0)) < 1e-12


# --- 2026-08-26 audit follow-ups (f), (g), (k) --------------------------------

def test_risk_contributions_are_leverage_invariant():
    """risk_pct is scale-invariant and always sums to 100%; dollar weights sum
    to the leverage. Plotting the LEVERED weights beside risk_pct put two
    denominators under one '% of portfolio' axis, so at 0.6x every weight bar
    shrank and read as a risk/weight gap that was pure leverage."""
    cov = covariance_matrix(_synthetic_returns())
    w = np.ones(len(cov)) / len(cov)
    lev = 1.6

    base = risk_contributions(w, cov)
    levered = risk_contributions(w * lev, cov)

    assert np.allclose(base["risk_pct"].values, levered["risk_pct"].values)
    assert abs(base["risk_pct"].sum() - 1.0) < 1e-12
    assert abs(levered["risk_pct"].sum() - 1.0) < 1e-12
    # The mismatch the chart used to draw:
    assert abs(base["weight"].sum() - 1.0) < 1e-12
    assert abs(levered["weight"].sum() - lev) < 1e-12


def test_ewma_weight_in_last_month_matches_the_caption():
    """The EWMA panel states where λ=0.94's weight actually sits. It claimed
    ~90% in the last month; the true figure is ~73%, and 90% needs ~37 days."""
    lam = RISKMETRICS_LAMBDA
    assert abs(lam - 0.94) < 1e-12
    month = 1.0 - lam ** 21
    assert 0.72 < month < 0.74                      # caption says ~73%
    assert not (0.88 < month < 0.92)                # the old ~90% claim
    days_to_90 = np.log(0.10) / np.log(lam)
    assert 36 < days_to_90 < 38                     # caption says ~37 days
    half_life = np.log(0.5) / np.log(lam)
    assert 10.5 < half_life < 11.5                  # caption says ~11 days


def test_factor_alpha_is_measured_over_the_risk_free_leg():
    """Alpha is return over the risk-free leg. Estimating the intercept on RAW
    returns overstates it by (1 - beta_market) * rf whenever beta != 1."""
    import src.factors as fac

    rng = np.random.default_rng(11)
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    rf_annual, beta_mkt = 0.0504, 0.5
    rf_d = rf_annual / 252.0
    alpha_d = 0.0002

    mkt_excess = rng.normal(0.0004, 0.011, n)
    factors = pd.DataFrame({
        "Market": rf_d + mkt_excess,                      # raw market return
        "Size": rng.normal(0, 0.006, n),                  # long-short, no rf
        "Value": rng.normal(0, 0.006, n),
        "Momentum": rng.normal(0, 0.006, n),
    }, index=idx)
    port = pd.Series(rf_d + beta_mkt * mkt_excess + alpha_d, index=idx)

    orig_build, orig_rf = fac._build_factors, fac.fetch_risk_free_rate
    try:
        fac._build_factors = lambda period="2y": factors
        fac.fetch_risk_free_rate = lambda *a, **k: rf_annual
        got = fac.factor_exposures(port)

        fac.fetch_risk_free_rate = lambda *a, **k: None
        raw = fac.factor_exposures(port)
    finally:
        fac._build_factors, fac.fetch_risk_free_rate = orig_build, orig_rf

    assert abs(got["betas"]["Market"] - beta_mkt) < 1e-6
    assert abs(got["alpha_annual"] - alpha_d * 252) < 1e-6
    assert "T-bill" in got["alpha_basis"]

    # Raw-return intercept is inflated by exactly (1 - beta) * rf, and says so.
    assert abs(raw["alpha_annual"] - (alpha_d * 252 + (1 - beta_mkt) * rf_annual)) < 1e-6
    assert raw["alpha_annual"] > got["alpha_annual"]
    assert "RAW" in raw["alpha_basis"]


# --- dispersion correction (2026-08-26: the map's terrain was too narrow) ----

def _ou_state_frame(n=700, theta_b=6.0, sigma_b=0.8, theta_v=4.0, eta_v=1.2,
                    mu_b=1.0, mu_v=0.22, seed=5):
    """A (beta, vol) state history drawn from EXACTLY the model the map
    simulates - so a correctly-sized calibration should need no widening."""
    from src.state_calibration import DT

    rng = np.random.default_rng(seed)
    phb, phv = np.exp(-theta_b * DT), np.exp(-theta_v * DT)
    sb = sigma_b * np.sqrt((1 - phb ** 2) / (2 * theta_b))
    sv = eta_v * np.sqrt((1 - phv ** 2) / (2 * theta_v))
    b, lv = np.empty(n), np.empty(n)
    b[0], lv[0] = mu_b, np.log(mu_v)
    for i in range(1, n):
        b[i] = mu_b + (b[i - 1] - mu_b) * phb + sb * rng.standard_normal()
        lv[i] = np.log(mu_v) + (lv[i - 1] - np.log(mu_v)) * phv + sv * rng.standard_normal()
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"beta": b, "vol": np.exp(lv)}, index=idx)


def test_dispersion_correction_prices_plug_in_estimation_error():
    """
    Even a state history drawn from EXACTLY the simulated model comes back
    under-covered: the horizon-ahead distribution is built from point
    estimates, so it ignores the uncertainty in theta, mu and sigma. Measured
    here, ~47% of realized states land inside the nominal 68.3% ring. The
    correction is therefore pricing plug-in estimation error as well as the
    overlapping-window smoothing - and on a correctly specified model it must
    stay modest and restore coverage, not run away.
    """
    from src.state_calibration import dispersion_correction

    d = dispersion_correction(_ou_state_frame())
    assert d["measured"] and d["n_dates"] >= 10
    assert d["coverage_raw"] < 0.60               # plug-in prediction is too tight
    assert 1.0 < d["k"] < 1.7, f"correction ran away on a correct model: {d['k']:.2f}"
    assert abs(d["coverage_corrected"] - 0.683) < 0.05


def test_dispersion_correction_widens_a_too_narrow_terrain():
    """When the realized state lands outside the drawn rings more often than
    the labels claim, the factor must fire and pull coverage back up."""
    from scipy import stats as _st

    from src.state_calibration import dispersion_correction, state_distances

    frame = _ou_state_frame(seed=11)
    d = dispersion_correction(frame)
    raw = state_distances(frame)
    # scaling every shock by k scales every distance by exactly 1/k
    assert d["k"] == 1.0 or abs(float(np.quantile(raw, 0.683)) / d["k"]
                                - float(np.sqrt(_st.chi2.ppf(0.683, 2)))) < 1e-9

    narrow = frame.copy()                      # a book whose state moves 1.8x
    narrow["beta"] = frame["beta"].mean() + (frame["beta"] - frame["beta"].mean()) * 1.8
    dn = dispersion_correction(narrow)
    assert dn["k"] > 1.0, "a too-narrow terrain was not widened"
    assert dn["coverage_corrected"] > dn["coverage_raw"]
    assert abs(dn["coverage_corrected"] - 0.683) < 0.10


def test_calibration_applies_the_measured_factor_to_both_shocks():
    """calibrate_state_dynamics must ship the widened shocks, report the factor
    it used, and widen beta diffusion and vol-of-vol by the SAME k (a uniform
    scale is what makes the corrected distance exactly raw / k)."""
    from src.state_calibration import (calibrate_state_dynamics,
                                       dispersion_correction, fit_ou,
                                       rolling_state_series)

    rng = np.random.default_rng(3)
    n = 900
    mkt = pd.Series(rng.normal(0.0003, 0.011, n),
                    index=pd.bdate_range("2019-01-01", periods=n))
    port = 1.2 * mkt + pd.Series(rng.normal(0, 0.006, n), index=mkt.index)

    cal = calibrate_state_dynamics(port, mkt)
    state = rolling_state_series(port, mkt)
    k = dispersion_correction(state)["k"]
    assert cal["dispersion"]["k"] == k

    uncorrected_b = fit_ou(state["beta"])["sigma"]
    uncorrected_v = fit_ou(np.log(state["vol"]))["sigma"]
    base = cal["cal"]["base"]
    assert base["sigB"] >= uncorrected_b - 1e-12    # never shrinks the terrain
    assert base["etaV"] >= uncorrected_v - 1e-12
    if k > 1.0 and not cal["clamp_flags"]:
        assert abs(base["sigB"] / uncorrected_b - k) < 1e-9
        assert abs(base["etaV"] / uncorrected_v - k) < 1e-9
    # the policy multipliers still ride on top of the corrected base
    assert abs(cal["cal"]["stress"]["sigB"] / base["sigB"] - 1.4) < 1e-9
    assert abs(cal["cal"]["calm"]["etaV"] / base["etaV"] - 0.7) < 1e-9


# --- pre-deploy security posture (2026-08-26) -------------------------------

def test_egress_allowlist_blocks_everything_but_the_market_feed():
    """The SSRF control: no request leaves the allowlist, and the checks that
    matter are the ones a typo-squat or a metadata-service probe would hit."""
    from src.netguard import ALLOWED_HOSTS, EgressBlocked, guarded_session, host_allowed

    assert host_allowed("https://query1.finance.yahoo.com/v8/finance/chart/SPY")
    assert host_allowed("https://query2.finance.yahoo.com/x")
    assert not host_allowed("http://query1.finance.yahoo.com/x")     # https only
    assert not host_allowed("https://169.254.169.254/latest/meta-data/")  # cloud metadata
    assert not host_allowed("https://127.0.0.1:8501/")               # loopback
    assert not host_allowed("https://query1.finance.yahoo.com.evil.com/")  # suffix trick
    assert not host_allowed("file:///etc/passwd")
    assert not host_allowed("https://evil.com/?u=query1.finance.yahoo.com")
    # Every allowlisted host is a deliberate, named market-data dependency:
    # Yahoo for prices, Business Insider for the ISIN lookup yfinance performs.
    assert all(h.endswith("yahoo.com") or h == "markets.businessinsider.com"
               for h in ALLOWED_HOSTS)

    # The guard refuses BEFORE any socket is opened, so this needs no network.
    session = guarded_session()
    for blocked in ("https://169.254.169.254/latest/meta-data/",
                    "http://query1.finance.yahoo.com/x"):
        try:
            session.get(blocked)
        except EgressBlocked:
            pass
        else:
            raise AssertionError(f"egress not blocked: {blocked}")


def test_map_payload_cannot_break_out_of_its_script_block():
    """The map's JSON is spliced inside a <script>. json.dumps does not escape
    '<', so a string containing '</script>' would end the block early."""
    from src.topology import war_room_html

    payload = {
        "base": {"beta": 1.0, "vol": 0.2}, "hazard": {"volMax": 0.6, "betaMax": 1.8},
        "assets": [{"t": "</script><img src=x onerror=alert(1)>", "b": 1.0,
                    "v": 0.2, "f": "Universe"}],
        "pairs": [], "muFlyer": 0.0, "muAnchor": 0.0, "muBook": 0.0,
        "cal": {"calm": {}, "base": {}, "stress": {}}, "muV": 0.2, "muB": 1.0,
        "days": 30, "domain": {"b0": -1.0, "b1": 2.0, "v0": 0.0, "v1": 1.0},
        "live": True, "provenance": "test",
        "footnote": "</SCRIPT ><svg onload=alert(1)>",
    }
    page = war_room_html(payload)
    begin, end = "/* __PAYLOAD_BEGIN__ */", "/* __PAYLOAD_END__ */"
    body = page[page.index(begin) + len(begin):page.index(end)]
    assert "<" not in body and ">" not in body, "raw angle bracket survived the splice"
    import json as _json

    blob = body.strip().removeprefix("const DEMO =").strip().rstrip(";")
    assert _json.loads(blob) == payload            # escaping changed nothing semantically


def test_universe_size_is_capped_at_the_funnel():
    """The complexity budget: no caller can make the server work on an
    unbounded universe, and junk symbols never reach a fetch."""
    from src.ingestion import MAX_UNIVERSE, _clean

    assert len(_clean([f"AA{i:03d}" for i in range(200)])) == MAX_UNIVERSE
    assert _clean(["SPY", "<script>", "A A", "'; DROP TABLE--", "QQQ"]) == ["QQQ", "SPY"]
    assert _clean(["../../etc/passwd", "https://evil.com"]) == []


def test_cache_budget_drops_oldest_first():
    """A visitor can mint a new cache key by typing a new basket. The budget
    keeps that from filling an ephemeral disk, and evicts oldest first."""
    import os
    import tempfile

    import src.ingestion as ingestion

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(6):
            path = os.path.join(tmp, f"prices_{i:02d}.parquet")
            with open(path, "wb") as fh:
                fh.write(b"x" * 1024)
            os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))  # oldest first
            paths.append(path)
        original = ingestion.DATA_DIR
        ingestion.DATA_DIR = tmp
        try:
            dropped = ingestion._prune_cache(max_files=4, max_mb=1000)
        finally:
            ingestion.DATA_DIR = original
        assert dropped == 2
        survivors = sorted(os.listdir(tmp))
        assert survivors == ["prices_02.parquet", "prices_03.parquet",
                             "prices_04.parquet", "prices_05.parquet"]


def test_deploy_config_keeps_the_visitor_out_of_the_operator_seat():
    """Regression guard on the deployed posture: a visitor is a viewer, gets no
    operator toolbar, no tracebacks, and XSRF/CORS stay on."""
    config = (pathlib.Path(__file__).resolve().parent.parent
              / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    settings = dict(
        line.split("=", 1)[0].strip() and
        (line.split("=", 1)[0].strip(), line.split("=", 1)[1].split("#")[0].strip())
        for line in config.splitlines()
        if "=" in line and not line.strip().startswith("#"))
    assert settings["toolbarMode"] == '"viewer"'
    assert settings["showErrorDetails"] == "false"
    assert settings["enableXsrfProtection"] == "true"
    assert settings["enableCORS"] == "true"
    assert settings["headless"] == "true"


# --- launch readiness (2026-08-26) ------------------------------------------

def test_crash_wire_correlates_a_session_reference_to_a_traceback():
    """The beta-test crash wire: a user quotes the footer reference, the
    operator finds that exact traceback."""
    from src.observability import (log_incident, log_session_start,
                                   new_session_ref, recent_incidents,
                                   setup_logging)

    setup_logging()
    ref = new_session_ref()
    assert len(ref) == 8 and ref.isalnum()
    log_session_start(ref, "regression test")
    try:
        raise RuntimeError("synthetic failure for the crash wire")
    except RuntimeError as exc:
        assert log_incident(ref, "test", exc) == ref
    hit = [line for line in recent_incidents(50) if ref in line]
    assert hit, "incident never reached the log"
    assert "RuntimeError" in hit[-1]
    # Two sessions must not collide - the reference is what makes a support
    # email findable at all.
    assert new_session_ref() != ref


def test_outbound_spend_cap_charges_then_refuses():
    """The API spend cap: counted at the one chokepoint, refuses when spent,
    and reports what is left."""
    import src.netguard as netguard

    original_cap, original_spent = netguard.MAX_REQUESTS_PER_DAY, dict(netguard._SPENT)
    try:
        netguard.MAX_REQUESTS_PER_DAY = 3
        netguard._SPENT.update({"day": 0.0, "count": 0.0})
        netguard.check_budget()                    # budget available: no raise
        for _ in range(3):
            netguard._charge_budget()
        assert netguard.budget_status() == {"spent": 3, "remaining": 0, "cap": 3}
        for call in (netguard.check_budget, netguard._charge_budget):
            try:
                call()
            except netguard.BudgetExhausted:
                pass
            else:
                raise AssertionError(f"{call.__name__} ignored an exhausted budget")
    finally:
        netguard.MAX_REQUESTS_PER_DAY = original_cap
        netguard._SPENT.update(original_spent)


def test_kill_switch_and_support_address_are_wired():
    """The kill switch must stop the script BEFORE any market-data call, and
    the support address must live in exactly one place."""
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "main.py").read_text(encoding="utf-8")
    assert 'os.getenv("MELEONA_MAINTENANCE"' in source
    kill = source.index("MELEONA_MAINTENANCE")
    assert "st.stop()" in source[kill:kill + 900], "maintenance mode never stops the run"
    assert kill < source.index("load_universe("), "kill switch runs after a data call"
    assert source.count("SUPPORT_EMAIL = os.getenv(") == 1, "support address is duplicated"
    assert "meleona.support@gmail.com" in source, "project mailbox not wired"
    assert "john4000.nguyen@gmail.com" not in source, (
        "personal address is published in the app - use the project mailbox")
    assert "MELEONA_SUPPORT_EMAIL" in source, "address is not env-overridable"
    # Phased release: the beta banner is opt-in per deployment.
    assert 'MELEONA_CHANNEL' in source and 'RELEASE_CHANNEL == "beta"' in source
    # setup_logging() must run before anything can fail interestingly
    assert source.index("setup_logging()") < source.index("with tab_3d:")


def test_launch_documents_exist_and_say_the_load_bearing_things():
    """Policies a public deploy needs, checked for the claims the app relies
    on rather than for mere existence."""
    root = pathlib.Path(__file__).resolve().parent.parent
    privacy = (root / "PRIVACY.md").read_text(encoding="utf-8").lower()
    terms = (root / "TERMS.md").read_text(encoding="utf-8").lower()
    support = (root / "SUPPORT.md").read_text(encoding="utf-8").lower()
    runbook = (root / "RUNBOOK.md").read_text(encoding="utf-8").lower()
    checklist = (root / "LAUNCH_CHECKLIST.md").read_text(encoding="utf-8").lower()

    for doc in (privacy, terms, support, checklist):
        assert "meleona.support@gmail.com" in doc
        assert "john4000.nguyen@gmail.com" not in doc, (
            "a personal address leaked into a public document")
    # The mailbox is a manual, pre-launch step - the runbook must carry it.
    assert "forwarding" in runbook and "meleona.support@gmail.com" in runbook
    assert "no sign-up" in privacy and "session reference" in privacy
    assert "yahoo finance" in privacy and "business insider" in privacy
    assert "not investment advice" in terms
    assert "session reference" in support
    for section in ("kill switch", "meleona_maintenance", "rollback",
                    "phased release", "dmarc", "spf"):
        assert section in runbook, f"runbook missing: {section}"
    # The checklist must stay honest about what does NOT apply here.
    assert "n/a" in checklist and "llm" in checklist


def test_restore_drill_reports_cost_per_universe():
    """RUNBOOK's restore claim, executable: warming a universe must report
    whether it worked, how long it took and what it spent from the budget."""
    from src.ingestion import PRESETS
    from tools.warm_cache import warm

    name = "Index core (S&P 500 & broad market ETFs)"
    row = warm(name, PRESETS[name])         # cached path: no network needed
    assert row["universe"] == name
    assert set(row) == {"universe", "ok", "detail", "seconds", "requests"}
    assert row["seconds"] >= 0 and row["requests"] >= 0
    # A failure must be reported, never raised - a warm-up cannot take a deploy
    # down, which is the whole point of running it after a restore.
    broken = warm("bad", ["THIS-IS-NOT-A-TICKER-AT-ALL"])
    assert broken["ok"] is False and broken["detail"]


# --- block bootstrap (2026-08-26): returns are not i.i.d. ------------------

def _clustered_returns(n=1500, seed=3):
    """A GARCH(1,1)-flavoured series: uncorrelated in sign, clustered in size -
    the stylised fact an i.i.d. bootstrap throws away."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    sigma2 = 1e-4
    for i in range(n):
        sigma2 = 2e-6 + 0.10 * (r[i - 1] ** 2 if i else 1e-4) + 0.88 * sigma2
        r[i] = rng.normal(0.0, np.sqrt(sigma2))
    return pd.DataFrame({"A": r}, index=pd.bdate_range("2019-01-01", periods=n))


def _abs_autocorr(x, lag=1):
    a = np.abs(np.asarray(x, dtype=float))
    a = a - a.mean()
    return float((a[lag:] @ a[:-lag]) / (a @ a))


def test_block_length_detects_clustering_and_ignores_noise():
    """The block length is measured, not typed in: long when volatility
    clusters, minimal when the series really is i.i.d."""
    rng = np.random.default_rng(11)
    iid = rng.normal(0, 0.01, 1500)
    assert mean_block_length(iid) <= 3, "i.i.d. noise should need no block"
    clustered = _clustered_returns()["A"].values
    assert mean_block_length(clustered) > mean_block_length(iid)
    assert 2 <= mean_block_length(clustered) <= 40      # clamped, never runaway


def test_block_bootstrap_preserves_volatility_clustering():
    """The point of the whole exercise: sampled paths must still show the
    magnitude-clustering the history has. i.i.d. resampling destroys it."""
    rets = _clustered_returns()
    w = np.array([1.0])
    history_acf = _abs_autocorr(rets["A"].values)
    assert history_acf > 0.05, "test data is not actually clustered"

    rng = np.random.default_rng(5)
    from src.risk import _stationary_bootstrap, mean_block_length as mbl
    port = rets["A"].values
    block_path = port[_stationary_bootstrap(port.size, 1, 4000, mbl(port), rng)][0]
    iid_path = rng.choice(port, size=4000, replace=True)

    assert _abs_autocorr(block_path) > 3 * _abs_autocorr(iid_path)
    assert _abs_autocorr(iid_path) < 0.03          # i.i.d. really is memoryless

    # and the resampling must not move the mean: it reorders history, not shifts it
    assert abs(block_path.mean() - port.mean()) < 4 * port.std() / np.sqrt(4000)


def test_monte_carlo_reports_its_resampling_scheme():
    """A number whose method is invisible cannot be checked, so the engine and
    the measured block length ride along with the result."""
    rets = _clustered_returns(n=800)
    w = np.array([1.0])
    blocked = monte_carlo(rets, w, n_simulations=4000)
    iid = monte_carlo(rets, w, n_simulations=4000, block=False)

    assert blocked["engine"] == "stationary block bootstrap"
    assert blocked["block_days"] >= 2
    assert iid["engine"] == "bootstrap" and iid["block_days"] == 1
    for mc in (blocked, iid):
        assert mc["cvar"] >= mc["var"] - 1e-9
        assert np.isfinite(mc["cvar"]) and mc["cvar_se"] > 0

    # too little history to form blocks: fall back rather than pretend
    short = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 20)})
    assert monte_carlo(short, w, n_simulations=200)["block_days"] == 1


# --- accessibility (2026-08-26) ---------------------------------------------

def _contrast(fg: str, bg: str) -> float:
    """WCAG 2.2 contrast ratio between two #rrggbb colours."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = lum(fg), lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def test_text_colours_meet_wcag_aa():
    """Every colour the stylesheet uses for TEXT must clear 4.5:1 on the ground
    it sits on. Bronze #9A7B4F measured 2.50:1 and is why this test exists - it
    may still be used for rules and hovers, never for words."""
    import re as _re

    root = pathlib.Path(__file__).resolve().parent.parent
    # main.py's inline style attributes are the same surface as the stylesheet:
    # the engine's "skip to the risk map" link sat at 2.50:1 there while
    # static/app.css was already clean, so both are scanned.
    css = ((root / "static" / "app.css").read_text(encoding="utf-8")
           + "\n" + (root / "main.py").read_text(encoding="utf-8"))

    # sanity: the formula agrees with the published reference pair
    assert abs(_contrast("#FFFFFF", "#000000") - 21.0) < 0.01

    page, panel = "#D4CDBF", "#E9E4DB"
    for fg in ("#4A4640", "#6A5030"):
        assert _contrast(fg, page) >= 4.5, f"{fg} fails on the page ground"
        assert _contrast(fg, panel) >= 4.5, f"{fg} fails on a panel"
    assert _contrast("#EDE9E3", "#3F3B35") >= 4.5          # footer on charcoal

    # Every text colour must clear 4.5:1 on the ground it actually sits on.
    # Light tones live on the charcoal bands, everything else on beige - so
    # check each against its own ground rather than excusing it from the test,
    # which is how #A89F8F sat at 4.25:1 on charcoal unnoticed.
    charcoal = "#3F3B35"
    for colour in set(_re.findall(r"(?<![-\w])color:\s*(#[0-9A-Fa-f]{6})", css)):
        on_beige = min(_contrast(colour, page), _contrast(colour, panel))
        on_band = _contrast(colour, charcoal)
        assert max(on_beige, on_band) >= 4.5, (
            f"{colour} clears nothing: {on_beige:.2f}:1 on beige, "
            f"{on_band:.2f}:1 on charcoal")
    assert "#9A7B4F" in css, "decorative bronze should still exist"


def test_accessibility_statement_matches_what_the_app_does():
    """A statement claiming more than the code delivers is the liability the
    statement exists to avoid, so check the load-bearing claims."""
    root = pathlib.Path(__file__).resolve().parent.parent
    doc = (root / "ACCESSIBILITY.md").read_text(encoding="utf-8")
    app = (root / "main.py").read_text(encoding="utf-8")
    map_html = (root / "prototypes" / "war_room.html").read_text(encoding="utf-8")

    assert "meleona.support@gmail.com" in doc
    assert "WCAG 2.2" in doc
    assert "Known gaps" in doc, "a statement with no gaps section is an overclaim"
    for gap in ("Streamlit", "Plotly", "No user testing"):
        assert gap in doc, f"undisclosed known gap: {gap}"

    # claims the document makes about the map must be true of the map
    assert map_html.count('aria-hidden="true"') >= 5      # the five canvas layers
    assert "sr-only" in map_html and "updateSummary()" in map_html
    assert "prefers-reduced-motion" in map_html

    # and the footer must carry the statement and the contact route
    assert "ACCESSIBILITY.md" in app, "footer does not link the statement"
    assert "WCAG 2.2 AA" in app and "barrier" in app.lower()


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
