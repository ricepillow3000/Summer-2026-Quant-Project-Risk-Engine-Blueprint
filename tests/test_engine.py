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
                            least_correlated_to_pair, cluster_order)
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


def test_correlation_tree_order_puts_each_planted_block_together():
    # Two planted factors, columns deliberately interleaved A1, B1, A2, B2...
    # In input order the heatmap scatters each group across the grid; the
    # correlation tree must pull each group into one unbroken run.
    rng = np.random.default_rng(23)
    n = 600
    fa, fb = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    cols = {}
    for i in (1, 2, 3):
        cols[f"A{i}"] = fa + rng.normal(0, 0.004, n)
        cols[f"B{i}"] = fb + rng.normal(0, 0.004, n)
    df = pd.DataFrame(cols)
    assert list(df.columns) == ["A1", "B1", "A2", "B2", "A3", "B3"]
    corr = df.corr()

    order = cluster_order(corr)
    assert sorted(order) == sorted(corr.columns)       # same names, once each
    groups = "".join(name[0] for name in order)
    assert groups in ("AAABBB", "BBBAAA"), order       # each block contiguous

    # A permutation only: every pair keeps exactly its own number.
    re = corr.loc[order, order]
    for a in corr.columns:
        for b in corr.columns:
            assert re.loc[a, b] == corr.loc[a, b]
    assert sorted(re.to_numpy().ravel()) == sorted(corr.to_numpy().ravel())

    # A NaN (a zero-vol name) gives no honest distance: keep the input order.
    bad = corr.copy()
    bad.loc["A1", "B2"] = bad.loc["B2", "A1"] = np.nan
    assert cluster_order(bad) == list(corr.columns)
    # Fewer than 3 names: nothing to group, input order back.
    assert cluster_order(corr.iloc[:2, :2]) == ["A1", "B1"]


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


def _yahoo_unreachable() -> str | None:
    """None if Yahoo answers, else the reason - lets live tests skip offline
    without swallowing real failures once the feed is up."""
    try:
        from src.ingestion import fetch_prices
        fetch_prices(["SPY", "QQQ"], period="5d", use_cache=False)
        return None
    except Exception as exc:                        # noqa: BLE001
        return type(exc).__name__


def test_security_master_survives_a_failed_fetch():
    """yfinance 1.7 returns None for .dividends/.splits when the history
    fetch fails. One such name used to raise len(None) and blank the whole
    Security Master tab; now that row reads unknown (None), never zero."""
    import src.security_master as sm_mod

    class _Failed:
        isin = "-"
        dividends = None
        splits = None

    class _Ok:
        isin = "US0000000001"
        dividends = pd.Series([0.5, 0.5], index=pd.to_datetime(
            ["2025-01-02", "2025-07-01"]).tz_localize("UTC"))
        splits = pd.Series(dtype=float)

    real = sm_mod.yf.Ticker
    sm_mod.yf.Ticker = lambda t, session=None: _Failed() if t == "BAD" else _Ok()
    try:
        sm = sm_mod.security_master(["GOOD", "BAD"], use_cache=False)
    finally:
        sm_mod.yf.Ticker = real
    assert sm.loc["GOOD", "dividends_paid"] == 2
    assert sm.loc["GOOD", "total_dividends"] == 1.0
    assert pd.isna(sm.loc["BAD", "dividends_paid"]), "failed fetch reported as a count"
    assert pd.isna(sm.loc["BAD", "total_dividends"]), "failed fetch reported as 0.0"
    assert pd.isna(sm.loc["BAD", "splits"]), "failed fetch reported as 0 splits"
    assert sm.loc["BAD", "isin"] == "unavailable"


def test_security_master_reads_its_disk_cache_and_never_keeps_a_failure():
    """John, 2026-10-20: the slow first basket switch. The security master
    made two Yahoo calls per name, one name after another, on every new basket
    (20 of the 44.8s an offline switch took) and kept nothing on disk.
    Identifiers and corporate actions change rarely: a complete row is now
    kept a day, under DATA_DIR, in one capped file. A row with a failed or
    missing lookup is kept an hour only, so the page's "Retried within the
    hour" stays true and an outage cannot blank a name for a day."""
    import json
    import tempfile
    import time as _time
    import src.ingestion as ing
    import src.security_master as sm_mod

    class _Failed:
        isin = "-"
        dividends = None
        splits = None

    class _Ok:
        isin = "US0000000001"
        dividends = pd.Series([0.5, 0.5], index=pd.to_datetime(
            ["2025-01-02", "2025-07-01"]).tz_localize("UTC"))
        splits = pd.Series(dtype=float)

    calls = []

    def fake_ticker(t, session=None):
        calls.append(t)
        return _Failed() if t == "BAD" else _Ok()

    real_ticker, real_dir = sm_mod.yf.Ticker, ing.DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        sm_mod.yf.Ticker, ing.DATA_DIR = fake_ticker, tmp
        try:
            fresh = sm_mod.security_master(["GOOD", "BAD"])
            assert sorted(set(calls)) == ["BAD", "GOOD"]
            calls.clear()
            again = sm_mod.security_master(["GOOD", "BAD"])
            assert calls == [], f"a second look within the hour went to Yahoo: {calls}"
            assert again.equals(fresh), "a cached row differs from the fetched one"
            path = pathlib.Path(tmp) / "security_master.json"
            assert path.is_file() and [f.name for f in pathlib.Path(tmp).iterdir()] == [path.name]

            def age_all(hours):
                store = json.loads(path.read_text())
                for row in store.values():
                    row["fetched_at"] = _time.time() - hours * 3600
                path.write_text(json.dumps(store))

            age_all(2)                       # a failure is retried after an hour...
            sm_mod.security_master(["GOOD", "BAD"])
            assert set(calls) == {"BAD"}, calls
            calls.clear()
            age_all(25)                      # ...a complete row after a day
            sm_mod.security_master(["GOOD"])
            assert set(calls) == {"GOOD"}, calls
            calls.clear()
            sm_mod.security_master(["GOOD"], use_cache=False)
            assert set(calls) == {"GOOD"}, "use_cache=False still read the disk"

            many = [f"T{i}" for i in range(sm_mod.SM_MAX_ENTRIES + 25)]
            sm_mod.security_master(many)
            assert len(json.loads(path.read_text())) <= sm_mod.SM_MAX_ENTRIES
        finally:
            sm_mod.yf.Ticker, ing.DATA_DIR = real_ticker, real_dir


def _fake_history(t, years=2.0):
    """One name's daily history the way yfinance's Ticker.history hands it
    over: tz-aware, on that name's own calendar. Offline and deterministic."""
    import yfinance.utils as yu
    if t == "DEAD":
        raise RuntimeError("possibly delisted")
    if t == "EMPTY":
        return yu.empty_df()
    end, days = pd.Timestamp("2026-10-16"), int(365 * years)
    if t.endswith("-USD"):                      # crypto: seven days a week
        idx = pd.date_range(end=end, periods=days, freq="D", tz="UTC")
    elif t.endswith("=X"):                      # FX: London weekdays
        idx = pd.bdate_range(end=end, periods=days * 5 // 7, tz="Europe/London")
    else:                                       # equities: each name its own holiday
        idx = pd.bdate_range(end=end, periods=days * 5 // 7, tz="America/New_York")
        idx = idx[(np.arange(len(idx)) % 37) != (sum(map(ord, t)) % 37)]
    idx = pd.DatetimeIndex(list(idx), name="Date")
    rng = np.random.default_rng(sum(map(ord, t)))
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))
    vol = (np.zeros(len(idx), dtype="int64") if t.endswith("=X")
           else rng.integers(1, 10**7, len(idx)))
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": vol}, index=idx)


class _FakeYahoo:
    """Runs yfinance's REAL yf.download (its threading, timezone strip, union
    and column sort) on fake per-name histories, records every download's
    names, and points the cache at a temp folder. Nothing leaves the machine."""

    def __init__(self, years=2.0, flaky=None):
        # flaky: {name: how many of its requests fail} - the way yfinance's
        # locked timezone cache dropped BA for one request (pair audit run 31)
        self.years, self.downloads, self.flaky = years, [], dict(flaky or {})

    def __enter__(self):
        import tempfile
        import yfinance.multi as multi
        import yfinance.utils as yu
        import src.ingestion as ing
        fake = self

        class _Ticker:
            def __init__(self, t, session=None):
                self.ticker, self._price_history = t, None

            def history(self, **kw):
                if fake.flaky.get(self.ticker, 0) > 0:
                    fake.flaky[self.ticker] -= 1
                    raise RuntimeError("OperationalError('database is locked')")
                return _fake_history(self.ticker, fake.years)

        self._tmp = tempfile.TemporaryDirectory()
        self._real = (multi.Ticker, yu.get_ticker_by_isin, ing.yf.download,
                      ing.check_budget, ing.guarded_session, ing.DATA_DIR,
                      getattr(ing, "RETRY_PAUSE", None))
        ing.RETRY_PAUSE = 0.0                   # no real waiting in a unit test
        real_download = ing.yf.download

        def recording(tickers, *a, **k):
            fake.downloads.append(sorted(tickers) if isinstance(tickers, list) else [tickers])
            return real_download(tickers, *a, **k)

        multi.Ticker = _Ticker
        yu.get_ticker_by_isin = lambda isin: {"US0000000AA1": "AAA"}.get(isin, "")
        ing.yf.download, ing.check_budget = recording, (lambda *a, **k: None)
        ing.guarded_session, ing.DATA_DIR = (lambda: None), self._tmp.name
        self.dir = pathlib.Path(self._tmp.name)
        return self

    def reference(self, tickers, period="2y"):
        """What ONE yf.download of the whole basket returns, unrecorded."""
        return self._real[2](tickers, period=period, auto_adjust=True, progress=False,
                             repair=True, timeout=30, session=None)

    def __exit__(self, *exc):
        import yfinance.multi as multi
        import yfinance.utils as yu
        import src.ingestion as ing
        (multi.Ticker, yu.get_ticker_by_isin, ing.yf.download,
         ing.check_budget, ing.guarded_session, ing.DATA_DIR, pause) = self._real
        if pause is None:
            del ing.RETRY_PAUSE
        else:
            ing.RETRY_PAUSE = pause
        self._tmp.cleanup()


def test_a_basket_stitched_from_single_names_matches_one_download():
    """John, 2026-10-20: a typed-in basket was slow because the cache kept
    whole baskets - three new names re-downloaded all fifteen, under each of
    the page's five keys, and yfinance's repair pass costs seconds of CPU per
    name per download. Each name is now kept on its own and a basket is
    stitched from them. yf.download is itself per name (Ticker.history,
    timezone strip, union of dates, sorted columns), so the stitch must equal
    ONE download of the whole basket exactly: raw, and after the cleaning pass
    with both aligns. Mixed calendars, a name that fails, one that is empty."""
    import src.ingestion as ing

    basket = ["AAA", "BBB", "BTC-USD", "DEAD", "EMPTY", "EURUSD=X"]
    with _FakeYahoo() as fy:
        ing._download_close_volume(["AAA", "BTC-USD"], "2y")    # two names known
        close, volume = ing._download_close_volume(basket, "2y")
        # only the names not held, then (ME-14) only the ones that came back empty
        assert fy.downloads[1:] == [["BBB", "DEAD", "EMPTY", "EURUSD=X"],
                                    ["DEAD", "EMPTY"], ["DEAD", "EMPTY"]], fy.downloads
        ref = fy.reference(basket)
        pd.testing.assert_frame_equal(close, ref["Close"], check_exact=True, check_freq=False)
        pd.testing.assert_frame_equal(volume, ref["Volume"], check_exact=True,
                                      check_freq=False, check_dtype=False)
        for align in (True, False):
            pd.testing.assert_frame_equal(
                ing.fetch_prices(basket, align=align),
                ing._clean_price_frame(ref["Close"], align), check_exact=True, check_freq=False)
        one, _ = ing._download_close_volume(["BBB"], "2y")       # cached single name
        pd.testing.assert_frame_equal(one, fy.reference(["BBB"])["Close"],
                                      check_exact=True, check_freq=False)
        try:
            ing._download_close_volume(["DEAD", "EMPTY"], "2y")
            raise AssertionError("an all-failed basket returned data")
        except ing.NoDataError as e:
            assert str(e) == ing.NO_DATA_MSG


def test_adding_names_to_a_known_basket_downloads_only_the_new_names():
    """The point of the per-name cache: a visitor who adds three names to a
    basket the server already holds costs three downloads, not fifteen. A
    name past its freshness is pulled again, a failed name is never
    remembered (a typo or an outage is retried), and the basket's provenance
    states its OLDEST pull - the age the page shows is the honest one - and
    its newest."""
    import datetime as _dt
    import json
    import src.ingestion as ing

    with _FakeYahoo() as fy:
        ing.fetch_prices(["AAA", "BBB"])
        ing.fetch_prices(["AAA", "BBB", "CCC", "DDD", "DEAD"])
        # DEAD is never served: asked again twice (ME-14), then left out
        assert fy.downloads == [["AAA", "BBB"], ["CCC", "DDD", "DEAD"], ["DEAD"], ["DEAD"]], fy.downloads
        ing.fetch_prices(["CCC", "DEAD"])
        assert fy.downloads[-1] == ["DEAD"], f"a failed name was remembered: {fy.downloads}"

        def stamp(t, hours_ago):
            path = ing._raw_path(t, "2y")
            f = pd.read_parquet(path)
            when = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours_ago)
            f.attrs["fetched_at_utc"] = when.isoformat(timespec="seconds")
            f.to_parquet(path)
            return f.attrs["fetched_at_utc"]

        stamp("AAA", 7)                          # past the 6 h window
        held = stamp("DDD", 2)                   # still fresh
        n = len(fy.downloads)
        ing.fetch_prices(["AAA", "DDD"])
        assert fy.downloads[n:] == [["AAA"]], fy.downloads[n:]
        meta = json.loads(pathlib.Path(ing._meta_path(
            ing._cache_path(["AAA", "DDD"], "2y"))).read_text())
        assert meta["fetched_at_utc"] == held, meta
        assert meta["fetched_newest_utc"] > held, meta


def test_use_cache_false_never_reads_or_writes_a_name_file():
    """Live checks (tools/live_provenance, the live tests) pass use_cache=False
    to compare the engine with Yahoo. A per-name file read there would make
    the check compare the cache with itself, so it must go to the feed for
    every name and leave nothing behind."""
    import src.ingestion as ing

    with _FakeYahoo() as fy:
        ing._download_close_volume(["AAA"], "2y")
        path = pathlib.Path(ing._raw_path("AAA", "2y"))
        poisoned = pd.read_parquet(path)
        poisoned["Close"] = 1.0
        poisoned.to_parquet(path)
        before = sorted(p.name for p in fy.dir.iterdir())
        px = ing.fetch_prices(["AAA", "BBB"], use_cache=False)
        assert fy.downloads[-1] == ["AAA", "BBB"], fy.downloads
        assert (px["AAA"] != 1.0).all(), "use_cache=False read a cached name"
        assert sorted(p.name for p in fy.dir.iterdir()) == before, "use_cache=False wrote a file"


def test_an_isin_basket_loads_as_one_download_would():
    """VALID_TICKER lets an ISIN through and yfinance relabels it: in one
    download US0000000AA1 and AAA are one series under the ISIN's name. A
    cached AAA beside a downloaded ISIN would be the same series twice (a
    singular covariance), so a basket with an ISIN skips the per-name cache."""
    import src.ingestion as ing

    basket = ["AAA", "BBB", "US0000000AA1"]
    with _FakeYahoo() as fy:
        ing.fetch_prices(["AAA"])
        px = ing.fetch_prices(basket)
        assert fy.downloads[-1] == basket, fy.downloads
        pd.testing.assert_frame_equal(
            px, ing._clean_price_frame(fy.reference(basket)["Close"], True),
            check_exact=True, check_freq=False)


def test_a_warm_pass_fits_inside_the_cache_budget():
    """The disk cache evicts its oldest files past CACHE_MAX_FILES. At 240 a
    warm pass already wrote more than that, so the pass evicted its own first
    presets; the per-name files add ~450 more. A full pass, on fake data, must
    leave room to spare and evict nothing."""
    import src.ingestion as ing
    import src.security_master as sm_mod
    import tools.warm_cache as wc

    class _Ref:
        isin = "US0000000001"
        dividends = pd.Series(dtype=float)
        splits = pd.Series(dtype=float)

    dropped = []
    real = (ing._prune_cache, sm_mod.yf.Ticker, wc.PAUSE_SECONDS)

    def counting_prune(*a, **k):
        dropped.append(real[0](*a, **k))
        return dropped[-1]

    with _FakeYahoo(years=3.0) as fy:
        ing._prune_cache, sm_mod.yf.Ticker, wc.PAUSE_SECONDS = (
            counting_prune, (lambda t, session=None: _Ref()), 0)
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                wc.run_pass()
        finally:
            ing._prune_cache, sm_mod.yf.Ticker, wc.PAUSE_SECONDS = real
        files = len([p for p in fy.dir.iterdir()
                     if p.suffix == ".parquet" or p.name.endswith(".meta.json")])
    assert sum(dropped) == 0, f"a warm pass evicted {sum(dropped)} of its own files"
    assert files < ing.CACHE_MAX_FILES / 2, f"{files} files against a cap of {ing.CACHE_MAX_FILES}"


def test_requirements_pin_yfinance():
    """The per-name cache splits and stitches downloads the way yfinance
    1.7.0's multi.py assembles them (the stitch test proves it on that
    version). Unpinned, the host could resolve a version the test never saw."""
    root = pathlib.Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = [ln.split("#")[0].strip() for ln in lines if ln.startswith("yfinance")]
    assert pins == ["yfinance==1.7.0"], pins


def test_the_walks_typed_basket_pool_is_in_no_preset():
    """The deployed walk times a typed-in basket. Its names must stay out of
    every preset (presets are warmed, so the step would time a warm read) and
    inside SUGGESTIONS (names the page offers)."""
    import re
    from src.ingestion import PRESETS, SUGGESTIONS
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "tools" / "recruiter_walk.mjs").read_text(encoding="utf-8")
    m = re.search(r"const TYPED_POOL = \[([^\]]*)\]", src)
    assert m, "recruiter_walk.mjs has no TYPED_POOL"
    pool = re.findall(r"'([^']+)'", m.group(1))
    in_presets = set().union(*PRESETS.values())
    assert len(pool) >= 20, pool
    assert not set(pool) & in_presets, sorted(set(pool) & in_presets)
    assert set(pool) <= set(SUGGESTIONS), sorted(set(pool) - set(SUGGESTIONS))


def test_unknown_symbols_get_a_readable_error():
    """A visitor who mistypes every symbol used to see "Couldn't load market
    data: RuntimeError - quote <ref>", i.e. a crash report for their own typo.
    Yahoo answers an unknown symbol with NaNs (or an empty frame), so both
    shapes must raise NoDataError, whose message the UI shows verbatim."""
    import src.ingestion as ing

    idx = pd.bdate_range("2025-01-02", periods=30)
    nan_frame = pd.concat({"Close": pd.DataFrame(np.nan, idx, ["XQZQ", "QZXQ"]),
                           "Volume": pd.DataFrame(np.nan, idx, ["XQZQ", "QZXQ"])},
                          axis=1)
    real = ing.yf.download
    try:
        for fake in (nan_frame, pd.DataFrame()):
            ing.yf.download = lambda *a, _f=fake, **k: _f
            try:
                ing.fetch_prices(["XQZQ", "QZXQ"], use_cache=False)
            except ing.NoDataError as exc:
                msg = str(exc)
            else:
                raise AssertionError("no error for symbols with no data")
            assert msg == ing.NO_DATA_MSG
            assert "/" not in msg and "\\" not in msg, "a path reached the page"
    finally:
        ing.yf.download = real


def test_security_master_live():
    """Integration: real yfinance identifiers + corporate actions. Self-skips offline."""
    why = _yahoo_unreachable()
    if why:
        print(f"[skip] security_master live check: Yahoo unreachable ({why})")
        return
    try:
        from src.security_master import security_master
        sm = security_master(["AAPL", "MSFT"], use_cache=False)   # the feed, not the disk
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
    # "Offline" means Yahoo is unreachable - probe it directly. The app catches
    # a failed load and renders st.error instead of raising, so without this
    # probe an offline run FAILED with "Couldn't load market data" instead of
    # skipping, as the docstring promises. Probing first (rather than ignoring
    # that error) keeps a real data-path bug a failure when the feed is up.
    why = _yahoo_unreachable()
    if why:
        print(f"[skip] app integration: Yahoo unreachable ({why})")
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
    finally:
        # Live prices and ^IRX must not outlive this test: they were read by
        # the next offline boot, half-expired (Live audit 2026-10-20).
        import streamlit as st
        st.cache_data.clear()
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


def test_the_lean_tail_objective_equals_scipys_bit_for_bit():
    """The tail fit hands scipy a lean copy of its own objective (3x faster).
    It is only allowed because every value is the same float scipy computes,
    on the edges a fit can reach too: c = 0 and -0.0, points past a negative c's endpoint,
    scale <= 0, NaN/inf parameters, and the 0.3-1.5 scales a residual fit
    lands on (math.log there differs from numpy's in the last bit). Not a
    one-point sample: scipy keeps an out-of-support lone point there, but
    the fit never sees fewer than min_exceedances (50)."""
    from scipy import stats
    from src import risk
    rng = np.random.default_rng(3)
    _x0, func, _restore, _args = stats.genpareto._reduce_func(
        (0.1, 0.0, 1.0), {"floc": 0.0})
    data = [rng.pareto(4.0, 160) * 0.4, np.abs(rng.standard_t(4, 130)) * 0.01,
            np.array([0.0, 0.5, 1.0, 3.0])]
    thetas = [(c, s) for c in (0.0, -0.0, 0.2, 0.7, -0.3, -0.9, 1e-9, -1e-9)
              for s in (0.01, 0.3, 0.77, 1.0, 1.5, 4.0, 0.0, -1.0)]
    thetas += [(np.nan, 1.0), (0.1, np.nan), (np.inf, 1.0), (0.1, np.inf)]
    thetas += [(rng.uniform(-0.5, 0.8), rng.uniform(0.3, 1.5)) for _ in range(3000)]
    for x in data:
        for c, s in thetas:
            a = func(np.array([c, s]), x)
            b = risk._lean_nnlf(np.array([c, s]), x)
            assert (np.isnan(a) and np.isnan(b)) or a == b, (c, s, a, b)


def test_the_local_logxmax_equals_scipys():
    """The penalty constant is computed here rather than imported from a
    private scipy module; it must be the very same float."""
    from scipy.stats._constants import _LOGXMAX
    from src import risk
    assert risk._LOGXMAX == _LOGXMAX


def test_the_fast_tail_fit_gives_the_same_numbers_as_scipys_own():
    """With scipy's stock optimizer put back, every published number of the
    tail fit - xi, beta, the bootstrap interval, the tail VaR/ES - must not
    move by one bit, on raw returns and on standardised residuals."""
    from scipy import optimize
    from src import risk
    raw = _garch11(3000, seed=4, innov="t", df=4)
    sig, _ = ewma_volatility(raw.values)
    series = [raw, pd.Series(raw.values[60:] / sig[60:])]
    lean = [gpd_tail_fit(s, n_boot=60) for s in series]
    keep = risk._lean_fmin
    try:
        risk._lean_fmin = optimize.fmin      # what genpareto.fit uses by default
        stock = [gpd_tail_fit(s, n_boot=60) for s in series]
    finally:
        risk._lean_fmin = keep
    for a, b in zip(lean, stock):
        assert a["fitted"] and a["xi_ci"] is not None
        for k in ("xi", "beta", "xi_ci", "tail", "finite_endpoint"):
            assert a[k] == b[k], (k, a[k], b[k])


def test_the_conditional_tail_still_reports_the_unconditional_xi():
    """The raw-return fit is only there for its xi (the side-by-side that shows
    the clustering bias); its bootstrap interval was read nowhere and cost half
    the panel's time, so it is not computed."""
    r = _garch11(3000, seed=5, innov="t", df=5)
    mf = mcneil_frey_tail(r, n_boot=60)
    assert mf["unconditional_xi"] == gpd_tail_fit(r, n_boot=0)["xi"]
    assert "unconditional_xi_ci" not in mf
    assert mf["xi_ci"] is not None, "the conditional fit keeps its interval"


def test_requirements_pin_scipy():
    """The tail fit's lean objective mirrors scipy 1.18.1's genpareto
    likelihood (the bit-for-bit test proves it on that version). Unpinned,
    the host could resolve a version the test never saw."""
    root = pathlib.Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = [ln.split("#")[0].strip() for ln in lines if ln.startswith("scipy")]
    assert pins == ["scipy==1.18.1"], pins


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


def test_dispersion_correction_widens_a_too_narrow_terrain():
    """When the realized state lands outside the drawn rings more often than
    the labels claim, the factor must fire and widen the terrain.

    The obvious construction does NOT work, and this test used to use it:
    scaling beta by a constant about its mean changes nothing, because the OU
    fit is scale-equivariant - the fitted sigma scales with the data and every
    Mahalanobis distance is left exactly where it was (k and coverage came back
    byte-identical). A terrain is only genuinely too narrow when the model,
    fitted on earlier data, under-predicts LATER dispersion. So shift the
    regime: inflate the second half's innovations and leave the first half as
    the thing the fit sees.
    """
    from scipy import stats as _st

    from src.state_calibration import dispersion_correction, state_distances

    frame = _ou_state_frame(seed=11)
    d = dispersion_correction(frame)
    raw = state_distances(frame)
    # scaling every shock by k scales every distance by exactly 1/k
    assert d["k"] == 1.0 or abs(float(np.quantile(raw, 0.683)) / d["k"]
                                - float(np.sqrt(_st.chi2.ppf(0.683, 2)))) < 1e-9

    narrow = frame.copy()
    steps = frame["beta"].diff().fillna(0.0).to_numpy().copy()
    steps[len(steps) // 2:] *= 3.0             # regime break the fit cannot see
    narrow["beta"] = float(frame["beta"].iloc[0]) + np.cumsum(steps)
    dn = dispersion_correction(narrow)

    assert dn["k"] > 1.0, "a too-narrow terrain was not widened"
    assert dn["coverage_raw"] < d["coverage_raw"] - 0.05, (
        "the regime break did not actually make the terrain too narrow "
        f"({dn['coverage_raw']:.3f} vs {d['coverage_raw']:.3f})")
    assert dn["k"] > d["k"] + 0.15, (
        f"a narrower terrain did not demand a larger factor "
        f"({dn['k']:.3f} vs {d['k']:.3f})")
    assert dn["k"] <= 3.0, "the correction escaped its clamp"


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


def test_map_renders_ticker_names_as_text_not_markup():
    """2026-10-15 security audit: the pairing inspector and standings built
    HTML strings with the flyer and anchor names (pr.fl / pr.an) raw, then
    handed them to innerHTML - so a name carrying markup would run in the
    map's frame. The splice escape above only stops a name ending the <script>
    block; once the page reads it back as a JS string it is live markup again.
    Every statement that builds markup must pass payload names through esc()."""
    import re
    import shutil
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    page = (root / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    script = page[page.index("<script>"):]
    m = re.search(r"function esc\(s\) \{.*?\n\}", script, re.S)
    assert m, "the map lost its esc() helper"

    offenders = []
    for stmt in re.split(r";[ \t]*\n", script):
        if "innerHTML" not in stmt and "'<" not in stmt:
            continue                                  # not building markup
        for name in re.finditer(r"[\w.]*\b(?:pr|[aAB])\.(?:fl|an|t)\b", stmt):
            if not stmt[:name.start()].endswith("esc("):
                line = page[:page.index(stmt)].count("\n") + 1
                offenders.append(f"war_room.html:~{line}: {name.group(0)}")
    assert not offenders, "payload names reach innerHTML unescaped: " + str(offenders)

    node = shutil.which("node")
    if node:   # and esc() really neutralises markup, run as the page runs it
        probe = m.group(0) + "\nprocess.stdout.write(esc(`<img src=x onerror='a(1)'>&\"`));"
        out = subprocess.run([node, "-e", probe], capture_output=True, text=True,
                             timeout=30).stdout  # nosec B603 - fixed argv
        assert out == "&lt;img src=x onerror=&#39;a(1)&#39;&gt;&amp;&quot;", out


def test_a_feed_column_that_is_not_ticker_shaped_never_reaches_the_page():
    """VALID_TICKER guards what a visitor TYPES, but the column names that come
    back are the feed's: yfinance can label a column after a search result.
    2026-10-15 security audit: those names flow into the map, captions and
    tables, so the download keeps only ticker-shaped columns. Offline: the
    downloader is replaced, nothing leaves the machine."""
    import src.ingestion as ing

    idx = pd.bdate_range("2025-01-02", periods=30)
    bad = "<img src=x onerror=alert(1)>"
    cols = ["AAPL", bad]
    frame = pd.concat({"Close": pd.DataFrame(100.0, idx, cols),
                       "Volume": pd.DataFrame(1e6, idx, cols)}, axis=1)
    real = (ing.yf.download, ing.check_budget, ing.guarded_session)
    try:
        ing.yf.download = lambda *a, **k: frame
        ing.check_budget = lambda *a, **k: None
        ing.guarded_session = lambda: None
        prices = ing.fetch_prices(["AAPL", "MSFT"], use_cache=False)
        dv = ing.fetch_dollar_volume(["AAPL", "MSFT"], use_cache=False)
    finally:
        ing.yf.download, ing.check_budget, ing.guarded_session = real
    assert list(prices.columns) == ["AAPL"], list(prices.columns)
    assert list(dv.columns) == ["AAPL"], list(dv.columns)


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


def _blend(fg: str, bg: str, opacity: float) -> str:
    """The colour a visitor sees: fg drawn at `opacity` over bg, as #rrggbb."""
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    g = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(opacity * a + (1 - opacity) * b):02X}" for a, b in zip(f, g))


# Streamlit 1.62 draws st.caption at opacity 0.6 (computed on the page,
# 2026-10-22); app.css must override it or the blend below is what ships.
STREAMLIT_CAPTION_OPACITY = 0.6


def test_captions_and_small_labels_clear_aa_after_their_opacity():
    """ME-13 (2026-10-22): the stylesheet said captions were #524E47 - 6.5:1 on
    the panel - and test_text_colours_meet_wcag_aa agreed. The page drew them
    at Streamlit's 60% opacity: measured from screenshot pixels, every caption
    read 2.74-2.77:1 at 1440px and 390px (tools/caption_contrast_check.mjs).
    The metric labels (#ABA499) read 2.14-2.17:1 on their beige tiles; the
    older test excused them because the same colour clears 4.5:1 on charcoal,
    a ground they never sit on. Text sizes here are body or smaller, so 4.5:1,
    after opacity, on every beige ground."""
    import re as _re
    css = (pathlib.Path(__file__).resolve().parent.parent / "static" / "app.css").read_text(encoding="utf-8")
    grounds = ("#D4CDBF", "#E9E4DB", "#EDE9E3", "#F3EFE6")   # page, panel, column, metric tile

    def rules(token):
        return [(s, b) for s, b in _re.findall(r"([^{}]+)\{([^}]*)\}", css) if token in s]

    def colour(token):
        found = [m.group(1) for _, b in rules(token)
                 for m in [_re.search(r"(?<![-\w])color:\s*(#[0-9A-Fa-f]{6})", b)] if m]
        assert found, f"no colour set for {token}"
        return found[-1]

    def opacity(token, default):
        found = [float(m.group(1)) for _, b in rules(token)
                 for m in [_re.search(r"(?<![-\w])opacity:\s*([\d.]+)", b)] if m]
        return found[-1] if found else default

    checks = {
        "caption": (colour("stCaptionContainer"), opacity("stCaptionContainer", STREAMLIT_CAPTION_OPACITY)),
        "metric label": (colour('[data-testid="stMetricLabel"] p'), opacity('[data-testid="stMetricLabel"]', 1.0)),
    }
    # any other rule that dims a coloured text with opacity is judged the same way
    for sel, body in _re.findall(r"([^{}]+)\{([^}]*)\}", css):
        c = _re.search(r"(?<![-\w])color:\s*(#[0-9A-Fa-f]{6})", body)
        o = _re.search(r"(?<![-\w])opacity:\s*(0?\.\d+)", body)
        if c and o:
            checks[sel.strip()[-60:]] = (c.group(1), float(o.group(1)))
    for name, (fg, op) in checks.items():
        for g in grounds:
            seen = _blend(fg, g, op)
            assert _contrast(seen, g) >= 4.5, (
                f"{name}: {fg} at opacity {op} reads {seen} on {g} - {_contrast(seen, g):.2f}:1")


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

    # The Risk Topology map is its own dark surface, and was never scanned:
    # the 2026-10-19 council review measured --dim at 3.86:1, --bronze 4.38:1,
    # --emberhi 3.99:1 and the canvas labels down to 2.16:1 there. Words sit on
    # the glass panel (#1F1A13; #282117 where it floats over the bright map),
    # and the selected pairing row tints that to #362C1E - the lightest ground
    # any word on the map gets. The canvas labels sit on the darker terrain, so
    # clearing these three grounds covers them too.
    html = (root / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    tokens = dict(_re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", html))
    grounds = ("#1F1A13", "#282117", "#362C1E")

    def over(colour, ground):
        """An rgba() canvas fill as it lands on the ground, as #rrggbb."""
        m = _re.fullmatch(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)", colour)
        if not m:
            return colour
        a = float(m.group(4))
        bg = [int(ground[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(a * int(f) + (1 - a) * b):02X}"
                             for f, b in zip(m.groups()[:3], bg))

    words = {}
    for name in _re.findall(r"(?<![-\w])color:\s*var\(--([\w-]+)\)", html):
        words.setdefault(tokens[name], f"var(--{name})")
    for colour in _re.findall(r"(?<![-\w])color:\s*(#[0-9A-Fa-f]{6})", html):
        words.setdefault(colour, "css color")
    for colour in _re.findall(r"<text[^>]*?fill=\"(#[0-9A-Fa-f]{6})\"", html):
        words.setdefault(colour, "svg <text>")
    # canvas: a fillStyle that a fillText/spacedText draws with before the
    # next fillStyle is set (dots, bars and fog fills are not words)
    canvas = 0
    for expr, then in _re.findall(r"fillStyle\s*=\s*([^;]+);(.*?)(?=fillStyle\s*=|$)",
                                  html, _re.S):
        if "fillText(" in then or "spacedText(" in then:
            for colour in _re.findall(r"'(#[0-9A-Fa-f]{6}|rgba\([\d.,\s]+\))'", expr):
                words.setdefault(colour.replace(" ", ""), "canvas label")
                canvas += 1
    assert canvas >= 8, "the canvas label scan found too few labels - regex drift?"

    for colour, where in words.items():
        worst = min(_contrast(over(colour, g), g) for g in grounds)
        assert worst >= 4.5, f"map {where} {colour} is {worst:.2f}:1 on its ground"

    # the Run button's hover fill carries the beige label
    hover = _re.search(r"\.runbtn:hover\s*\{[^}]*?background:\s*([^;]+);", html).group(1)
    hover = _re.sub(r"var\(--([\w-]+)\)", lambda m: tokens[m.group(1)], hover).strip()
    assert _contrast(tokens["beige"], hover) >= 4.5, f"Run on hover: {hover}"
    assert "var(--bronze)" in html and "var(--dim)" in html, \
        "decorative bronze/dim rules and bars should stay as they are"


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


# --- pre-deploy red-team findings, 2026-08-27 --------------------------------


def test_most_independent_ranks_on_distance_from_zero():
    """least_correlated_to_pair calls its winner the most INDEPENDENT name.
    Ranking the signed average handed that title to the most strongly inverse
    candidate: -0.60 to both legs beat +0.05, though -0.60 is a name that moves
    hard with the pair and merely backwards. Independence is distance from
    zero."""
    names = ["A", "B", "INVERSE", "FLAT"]
    c = pd.DataFrame(np.eye(4), index=names, columns=names)
    for leg in ("A", "B"):
        c.loc["INVERSE", leg] = c.loc[leg, "INVERSE"] = -0.60
        c.loc["FLAT", leg] = c.loc[leg, "FLAT"] = 0.05
    pick, avg = least_correlated_to_pair(c, ("A", "B"))
    assert pick == "FLAT", pick
    assert abs(avg - 0.05) < 1e-12          # the SIGNED value is still returned


def test_replay_includes_the_windows_opening_day():
    """pct_change() taken inside the slice has no prior close for the first
    bar, so every crisis replay silently lost its opening day - often the
    largest single move in a crash. The lead-in close is re-attached, but only
    when it is genuinely the previous session."""
    import src.scenarios as sc

    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.DataFrame({"A": np.linspace(100.0, 140.0, 40)}, index=idx)
    start, end = idx[10], idx[20]
    orig = sc.fetch_prices
    try:
        sc.fetch_prices = lambda *a, **k: px
        r = sc.replay_returns(["A"], str(start.date()), str(end.date()))
        assert r.index[0] == start, r.index[0]        # the opening day survives
        assert len(r) == 11                            # every bar in the window
        expected = px.loc[start, "A"] / px.loc[idx[9], "A"] - 1.0
        assert abs(r["A"].iloc[0] - expected) < 1e-12

        # A stale lead-in must be REFUSED rather than turned into a one-day move.
        gapped = px.drop(index=idx[4:10])
        sc.fetch_prices = lambda *a, **k: gapped
        r2 = sc.replay_returns(["A"], str(start.date()), str(end.date()))
        assert r2.index[0] > start, "a stale close was used as a daily return"
    finally:
        sc.fetch_prices = orig


def test_map_asset_marks_flip_with_a_short_book():
    """build_map_payload negates the portfolio return series when the book
    is short, but computed every asset mark from the raw frame - so the
    portfolio dot was the short book while each holding stayed its long self,
    on one shared (beta, vol) plane. A short of a name carries the opposite
    beta; volatility is sign-invariant. batch_audit never passes bearish=True,
    so this is the only thing checking it."""
    import src.topology as tp

    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2023-01-02", periods=400)
    mkt = rng.normal(0.0004, 0.010, 400)
    # Held names only, so weights align; with no SPY the builder falls back to
    # its equal-weight market proxy, which is all this invariant needs.
    rets = pd.DataFrame({"AAA": 0.8 * mkt + rng.normal(0, 0.006, 400),
                         "BBB": 1.2 * mkt + rng.normal(0, 0.006, 400)},
                        index=idx)
    w = pd.Series([0.5, 0.5], index=["AAA", "BBB"])

    orig = tp.fetch_prices
    try:
        tp.fetch_prices = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        lng = tp.build_map_payload(rets, w, ["AAA", "BBB"], False)
        sht = tp.build_map_payload(rets, w, ["AAA", "BBB"], True)
    finally:
        tp.fetch_prices = orig

    lb = {a["t"]: a for a in lng["assets"]}
    sb = {a["t"]: a for a in sht["assets"]}
    assert lb and set(lb) == set(sb)
    for k in lb:
        assert abs(sb[k]["b"] + lb[k]["b"]) < 1e-9, k    # beta flips
        assert sb[k]["v"] == lb[k]["v"], k               # vol does not


def test_map_short_book_perimeter_domain_and_drift_are_mirrored():
    """The short book's whole (beta, vol) plane is the long one reflected in
    beta, so the three things drawn ON that plane must reflect with it.

    All three were wrong before, and all three are invisible to batch_audit,
    which never passes bearish=True:

      1. HAZARD PERIMETER. `min(1.9, beta0 + 0.8)` on a negated beta put the
         line between the book and zero - it marked DE-RISKING as the hazard -
         and min() picks the more negative number, so the 1.9 magnitude cap was
         bypassed entirely (a -2.86 short got a -2.06 perimeter).
      2. DRAWN DOMAIN. The [-1, +2] minimum window is a long-book default; a
         short book's content all sits left of zero, and the perimeter could
         fall outside the plot altogether.
      3. LONG-RUN BETA. The mu_b clamp band (-0.5, 2.0) is a long-book prior.
         Applied to a short fit it dragged the OU mean up to -0.5, inventing a
         de-risking drift over the 30 days that the data never showed.
    """
    import src.topology as tp

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=520)
    mkt = rng.normal(0.0003, 0.010, 520)
    # Deliberately high-beta, so beta0 + 0.8 clears the 1.9 cap and the cap is
    # actually exercised on both sides.
    rets = pd.DataFrame({"AAA": 1.7 * mkt + rng.normal(0, 0.004, 520),
                         "BBB": 2.1 * mkt + rng.normal(0, 0.004, 520)},
                        index=idx)
    w = pd.Series([0.5, 0.5], index=["AAA", "BBB"])

    orig = tp.fetch_prices
    try:
        tp.fetch_prices = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        lng = tp.build_map_payload(rets, w, ["AAA", "BBB"], False)
        sht = tp.build_map_payload(rets, w, ["AAA", "BBB"], True)
    finally:
        tp.fetch_prices = orig

    assert lng["betaSign"] == 1 and sht["betaSign"] == -1

    lb, sb = lng["base"]["beta"], sht["base"]["beta"]
    lh, sh = lng["hazard"]["betaMax"], sht["hazard"]["betaMax"]
    assert abs(sb + lb) < 1e-9                                  # book reflects
    assert sht["base"]["vol"] == lng["base"]["vol"]              # vol does not
    assert lng["hazard"]["volMax"] == sht["hazard"]["volMax"]    # nor its cap

    # The disclosed policy, applied to |beta| and handed back its sign.
    assert lh == round(min(tp.HAZARD_BETA_CAP, lb + tp.HAZARD_BETA_ADD), 2)
    assert sh == round(-min(tp.HAZARD_BETA_CAP, -sb + tp.HAZARD_BETA_ADD), 2)
    assert abs(sh) <= tp.HAZARD_BETA_CAP + 1e-9                 # cap survives
    assert abs(lh + sh) < 1e-9                                  # exact mirror
    # The line sits on the side danger is on: past it, |market exposure| grew.
    assert abs(lh) >= abs(lb) or lh == tp.HAZARD_BETA_CAP
    assert abs(sh) >= abs(sb) or sh == -tp.HAZARD_BETA_CAP

    # Everything claimed is inside the plot, in BOTH directions (MAP-04's
    # invariant, which batch_audit only ever checks on a long book).
    for p in (lng, sht):
        d = p["domain"]
        assert d["b0"] <= p["hazard"]["betaMax"] <= d["b1"]
        assert d["b0"] <= p["base"]["beta"] <= d["b1"]
        assert p["hazard"]["volMax"] <= d["v1"]
        for a in p["assets"]:
            assert d["b0"] <= a["b"] <= d["b1"] and d["v0"] <= a["v"] <= d["v1"]

    # The OU beta target reflects too: fit_ou on a negated series returns the
    # negated mean with the same theta/sigma, so a mirrored clamp band leaves
    # muB an exact reflection. Under the long-only band this read -0.5.
    assert abs(sht["muB"] + lng["muB"]) < 1e-9
    assert sht["muV"] == lng["muV"]                 # vol target is unsigned
    assert sht["muB"] < sb + tp.HAZARD_BETA_ADD     # no invented drift to zero


def test_map_short_book_keeps_anchor_funds_long():
    """On a short book the HOLDINGS flip (a short of a name carries the opposite
    beta), but the anchor pool (TLT, GLD, XLP, XLU, USMV) is not held: it is the
    candidate hedge, and anchor_rank(direction="short") documents that hedge as
    a LONG. build_map_payload negated every dot, so each anchor - and the
    defensive link drawn to it - sat on the wrong side of zero."""
    import src.topology as tp
    from src.pairing import DEFENSIVE_ANCHOR_TICKERS

    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2023-01-02", periods=420)
    mkt = rng.normal(0.0004, 0.010, 420)
    rets = pd.DataFrame({"AAA": 0.9 * mkt + rng.normal(0, 0.006, 420),
                         "BBB": 1.4 * mkt + rng.normal(0, 0.012, 420)}, index=idx)
    anchors = pd.DataFrame({t: 100 * np.cumprod(1 + (0.3 + 0.1 * k) * mkt + rng.normal(0, 0.004, 420))
                            for k, t in enumerate(DEFENSIVE_ANCHOR_TICKERS)}, index=idx)
    w = pd.Series([0.5, 0.5], index=["AAA", "BBB"])

    def fake_fetch(tickers, *a, **k):
        if list(tickers) == list(DEFENSIVE_ANCHOR_TICKERS):
            return anchors
        raise RuntimeError("offline")       # no SPY: basket proxies the market
    orig = tp.fetch_prices
    try:
        tp.fetch_prices = fake_fetch
        lng = tp.build_map_payload(rets, w, ["AAA", "BBB"], False)
        sht = tp.build_map_payload(rets, w, ["AAA", "BBB"], True)
    finally:
        tp.fetch_prices = orig

    lb = {a["t"]: a for a in lng["assets"]}
    sb = {a["t"]: a for a in sht["assets"]}
    assert set(DEFENSIVE_ANCHOR_TICKERS) <= set(sb)
    for t in DEFENSIVE_ANCHOR_TICKERS:
        assert sb[t]["b"] == lb[t]["b"] and sb[t]["f"] == "Anchor pool", t   # held long
    for t in ("AAA", "BBB"):
        assert abs(sb[t]["b"] + lb[t]["b"]) < 1e-9, t                         # shorted
    # and the footnote says so, instead of "the short book's betas are negated"
    assert "keep their own" in sht["footnote"], sht["footnote"]


def test_a_universe_holding_an_anchor_keeps_the_rest_of_the_anchor_pool():
    """The anchor pool was joined to the universe with DataFrame.join, which
    raises on a shared column; the except fell back to the universe alone. So
    any universe holding TLT, GLD, XLP, XLU or USMV - four of the twenty
    presets (sector ETFs, rates & credit, commodities, global macro) and any
    visitor pair with one of them - drew no anchor pool at all, and the
    linkage tournament never saw the anchors it did not hold. Found by the
    every-pair audit, 2026-10-17. A held anchor is drawn once, as a holding."""
    import src.topology as tp
    from src.pairing import DEFENSIVE_ANCHOR_TICKERS

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-01-02", periods=420)
    mkt = rng.normal(0.0004, 0.010, 420)
    px = pd.DataFrame({t: 100 * np.cumprod(1 + (0.3 + 0.1 * k) * mkt + rng.normal(0, 0.004, 420))
                       for k, t in enumerate(DEFENSIVE_ANCHOR_TICKERS)}, index=idx)
    rets = pd.DataFrame({"AAA": 1.3 * mkt + rng.normal(0, 0.012, 420)}, index=idx)
    rets["TLT"] = px["TLT"].pct_change().fillna(0)
    rets = rets.iloc[1:]
    w = pd.Series([0.5, 0.5], index=["AAA", "TLT"])

    def fake_fetch(tickers, *a, **k):
        if set(tickers) <= set(DEFENSIVE_ANCHOR_TICKERS):
            return px[list(tickers)]
        raise RuntimeError("offline")
    orig = tp.fetch_prices
    try:
        tp.fetch_prices = fake_fetch
        pl = tp.build_map_payload(rets, w, ["AAA", "TLT"], False)
    finally:
        tp.fetch_prices = orig

    units = [a["t"] for a in pl["assets"]]
    assert sorted(units) == sorted(["AAA", *DEFENSIVE_ANCHOR_TICKERS]), units   # TLT once
    assert {a["t"]: a["f"] for a in pl["assets"]}["TLT"] == "Universe"
    assert {a["t"] for a in pl["assets"] if a["f"] == "Anchor pool"} == {"GLD", "XLP", "XLU", "USMV"}
    assert pl["pairs"] and {p["an"] for p in pl["pairs"]} - {"AAA", "TLT"}, pl["pairs"]
    # the Balance tab reads the same pool, not a second hand-written join
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "main.py").read_text(encoding="utf-8")
    assert "anchor_frame(returns)" in source
    assert "fetch_prices(DEFENSIVE_ANCHOR_TICKERS" not in source


def test_every_calibration_clamp_is_named_in_the_footnote():
    """The footnote's "Calibration clamps hit" list is the disclosure for every
    number the map draws that was NOT estimated. Three ways it was wrong:
      - fit_ou clipped the AR(1) coefficient to 0.995 silently. It binds on
        live data (2 of 4 universes in the 2026-09-10 audit read phi 0.99500),
        and then the drawn mean-reversion speed is the clip, not the data;
      - the leverage clamp was reported as "rho", the shock correlation's name,
        and both theta clamps as a bare "theta";
      - k was capped at 3.0 with no flag while the footnote called it measured.
    Every band is forced to bind here, so each must announce itself by name."""
    import src.state_calibration as sc

    rng = np.random.default_rng(31)
    idx = pd.bdate_range("2022-01-03", periods=520)
    mkt = pd.Series(rng.normal(0.0003, 0.010, 520), index=idx)
    port = 1.1 * mkt + pd.Series(rng.normal(0, 0.006, 520), index=idx)
    saved = dict(sc.CLAMPS), sc.K_CLAMP
    try:
        sc.CLAMPS.update(theta=(55.0, 55.0), rho=(0.5, 0.5))   # point bands: always clip
        sc.K_CLAMP = (1.0, 1.0)
        cal = sc.calibrate_state_dynamics(port, mkt)
    finally:
        sc.CLAMPS.clear(); sc.CLAMPS.update(saved[0]); sc.K_CLAMP = saved[1]
    names = {f.split(" ")[0] for f in cal["clamp_flags"]}
    assert {"theta_b", "theta_v", "rho", "lev"} <= names, cal["clamp_flags"]
    assert "theta" not in names
    if cal["dispersion"].get("k_uncapped", 1.0) > 1.0:
        assert "k" in names, cal["clamp_flags"]
    # the AR(1) clip: a smooth rolling-beta series is persistent enough to hit it
    state = sc.rolling_state_series(port, mkt)
    fb = sc.fit_ou(state["beta"])
    if fb["phi_raw"] != fb["phi"]:
        assert "phi_b" in names, cal["clamp_flags"]


def _map_function(name: str) -> str:
    """One top-level function's source from the shipped map page."""
    import re as _re
    html = (pathlib.Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    m = _re.search(r"^function " + name + r"\(.*?^}$", html, flags=_re.S | _re.M)
    assert m, f"{name} not found in war_room.html"
    return m.group(0)


def test_map_names_the_dot_under_the_finger_not_the_first_one_near_it():
    """On a phone's 260-300px plot AAPL and MSFT sit 8px apart: hover and
    explain took the FIRST asset within 9-10px, so a finger exactly on MSFT
    read "AAPL". The nearest asset in range wins."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        print("[skip] node not on PATH - map hit-test check not run")
        return
    js = _map_function("nearestAsset") + """
const DEMO = { assets: [{ t: 'AAPL', b: 0, v: 0 }, { t: 'MSFT', b: 8, v: 0 }, { t: 'XOM', b: 100, v: 0 }] };
const toX = b => b, toY = v => v;
const out = [nearestAsset(8, 0, 9), nearestAsset(0, 0, 9), nearestAsset(5, 0, 9), nearestAsset(50, 0, 9)];
console.log(JSON.stringify(out.map(a => a && a.t)));"""
    res = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == '["MSFT","AAPL","MSFT",null]', res.stdout
    html = (pathlib.Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    assert "{ hov = a.t; break; }" not in html, "hover still takes the first asset in range"


def test_map_hover_reads_risk_direction_on_a_short_book():
    """The hover's scenario sentence compared raw beta: on a short book danger
    is MORE NEGATIVE beta, so a point toward the hazard perimeter was read as
    "beta sheds ... de-risking" and a point away from it as "drifting toward
    index risk". Runs the page's own scenarioText under node."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        print("[skip] node not on PATH - map hover check not run")
        return
    js = _map_function("scenarioText") + """
const S = { cloud: { maha: () => 0, r68: 1, r95: 2, r997: 3 } };
const out = {};
for (const [sign, b0] of [[1, 1.2], [-1, -1.2]]) {
  globalThis.BSGN = sign; globalThis.DEMO = { base: { beta: b0, vol: 0.2 } };
  out[sign] = { toward: scenarioText(b0 + sign * 0.6, 0.2), away: scenarioText(b0 - sign * 0.6, 0.2) };
}
/* a book at beta ~0 (FX majors): any move away from zero ADDS market exposure, either sign */
for (const [sign, b0] of [[1, -0.002], [-1, 0.1]]) {
  globalThis.BSGN = sign; globalThis.DEMO = { base: { beta: b0, vol: 0.2 } };
  out['zero' + sign] = { toward: scenarioText(-0.8, 0.2), away: scenarioText(0.9, 0.2) };
}
console.log(JSON.stringify(out));"""
    js = js.replace("function scenarioText", "var BSGN, DEMO;\nfunction scenarioText", 1)
    r = subprocess.run([node, "-e", js], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    out = __import__("json").loads(r.stdout)
    for sign in ("1", "-1"):
        assert "builds" in out[sign]["toward"] and "sheds" not in out[sign]["toward"], out[sign]
        assert "sheds" in out[sign]["away"], out[sign]
    assert out["1"] == out["-1"], "long and short read the same move the same way"
    for k in ("zero1", "zero-1"):   # |beta| grows from ~0 either way: never "sheds"
        assert "builds" in out[k]["toward"] and "builds" in out[k]["away"], out[k]


def test_map_explanations_match_the_maths():
    """Text the map shows next to its numbers, checked against the maths:
      - the pair weights are two-asset risk parity, so each leg carries HALF the
        risk; "the flyer holds most of the risk" said otherwise;
      - for a short flyer the loss is a squeeze, not a crash;
      - the perimeter hover gave the rule without its floor and caps, so it
        read "beta up 0.8" where the drawn line was capped at 1.9."""
    html = (pathlib.Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    assert "holds most of the risk" not in html
    assert "equal share of the risk" in html
    assert "squeeze" in html
    hover = html[html.index("perimeter: () =>"):html.index("iso: pct =>")]
    assert "floored at 35%" in hover and "capped at" in hover
    # an anchor-pool dot is a candidate hedge, not something the book holds
    assert "candidate hedge" in html


def test_map_inline_scripts_parse_under_node():
    """A JavaScript syntax error inside prototypes/war_room.html kills the whole
    map: it is one inline <script>, so the frame renders and nothing draws -
    tiles read "--", the canvas stays 300x150, zero painted pixels. Exactly
    this shipped in d4ea1f4 (an unescaped apostrophe in a tooltip string) and
    141 tests plus the 692-check audit were green while the flagship visual was
    blank, because nothing here executed that script: tests/map_probe.mjs
    slices only the math region, and the boot test never runs iframe JS.

    This parses every inline block with `node --check`. Self-skips without
    node on PATH, the same way batch_audit's map checks do."""
    import re as _re
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        print("[skip] node not on PATH - map script parse check not run")
        return

    root = pathlib.Path(__file__).resolve().parent.parent
    html = (root / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    blocks = _re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html,
                         flags=_re.S)
    assert blocks, "no inline <script> in war_room.html - the map has moved"

    for i, body in enumerate(blocks, 1):
        body = body.replace("__PAYLOAD__", "{}")   # the JSON splice marker
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(body)
            path = pathlib.Path(fh.name)
        try:
            r = subprocess.run([node, "--check", str(path)],
                               capture_output=True, text=True)
        finally:
            path.unlink(missing_ok=True)
        assert r.returncode == 0, (
            f"war_room.html inline script {i} does not parse - the map will "
            f"render a blank frame:\n{r.stderr.strip()[:600]}")


def test_map_layout_holds_at_every_width():
    """The map's glass panels float over the terrain, positioned for a wide
    screen. Measured in Chromium at the iframe's 820px height (2026-10-15):
      - at EVERY width the bottom identity chips were drawn over the beta tick
        labels, so the scale under the dots could not be read;
      - at 1024px the HUD tiles spilled under the pairing inspector;
      - at 1000px and below the panels covered each other: Run / Explain /
        Calibration were painted but could not be clicked, the plot shrank to
        180px at 790px (the width the app embeds it at) and 18px on a phone,
        and under 640px the canvas was stretched to fit.
    tests/map_layout.mjs renders the SHIPPED page (war_room_html on a real
    payload, long and short) and reports each of those; this asserts none
    remain. Self-skips without node or playwright, like the parse check."""
    import json as _json
    import shutil
    import subprocess
    import tempfile
    import src.topology as tp

    node = shutil.which("node")
    if not node:
        print("[skip] node not on PATH - map layout check not run")
        return
    rng = np.random.default_rng(15)
    idx = pd.bdate_range("2023-01-02", periods=420)
    mkt = rng.normal(0.0004, 0.010, 420)
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rets = pd.DataFrame({n: (0.5 + 0.25 * k) * mkt + rng.normal(0, 0.006, 420)
                         for k, n in enumerate(names)}, index=idx)
    w = pd.Series(1 / len(names), index=names)
    root = pathlib.Path(__file__).resolve().parent.parent
    orig = tp.fetch_prices
    try:
        tp.fetch_prices = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        pages = {side: tp.war_room_html(tp.build_map_payload(rets, w, names, side == "short"))
                 for side in ("long", "short")}
    finally:
        tp.fetch_prices = orig
    for side, html in pages.items():
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(html)
            path = pathlib.Path(fh.name)
        try:
            r = subprocess.run([node, str(root / "tests" / "map_layout.mjs"), str(path)],
                               capture_output=True, text=True, timeout=600)
        finally:
            path.unlink(missing_ok=True)
        if r.returncode == 3:
            print("[skip] playwright not installed - map layout check not run")
            return
        assert r.returncode == 0, r.stderr[-600:]
        for row in _json.loads(r.stdout):
            assert not row["found"] and not row["errors"], (side, row)


def test_streamlit_telemetry_is_disabled():
    """Streamlit gatherUsageStats defaults to True and phones home on every
    run. PRIVACY.md tells visitors there is no analytics, advertising,
    attribution or crash-reporting SDK, so the default made the published
    policy false - and it leaves the host by a route src/netguard.py never
    sees, because the allowlist only wraps this engine own data calls."""
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    raw = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    cfg = tomllib.loads(raw)
    assert cfg["browser"]["gatherUsageStats"] is False, "telemetry is back on"

    # The policy and the config have to keep agreeing, in both directions.
    privacy = (root / "PRIVACY.md").read_text(encoding="utf-8").lower()
    assert "no analytics" in privacy
    assert cfg["client"]["showErrorDetails"] is False
    assert cfg["server"]["enableXsrfProtection"] is True
    assert cfg["server"]["enableCORS"] is True


def test_no_third_party_assets_are_loaded():
    """PRIVACY.md promises no CDN scripts, no external stylesheets, no pixel.
    Anything the browser FETCHES from another host reveals the visitor IP and
    user agent to that host, which would make the promise false. Plain <a>
    links are fine - they only leak if the visitor clicks."""
    import re as _re

    root = pathlib.Path(__file__).resolve().parent.parent
    files = ["main.py", "static/app.css", "prototypes/war_room.html"]
    fetching = _re.compile(
        r"(?:src|href)\s*=\s*[\"']https?://|@import\s+[\"']?https?://"
        r"|url\(\s*[\"']?https?://",
        _re.I)
    offenders = []
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8")
        for n, ln in enumerate(text.splitlines(), 1):
            for m in fetching.finditer(ln):
                # A stylesheet/script/image fetch is a leak; an anchor is not.
                if m.group(0).lower().startswith("href") and "<a " in ln.lower():
                    continue
                offenders.append(rel + ":" + str(n))
    assert not offenders, "third-party fetch on page load: " + str(offenders)


def test_no_credentials_in_tracked_source():
    """There is no API key in this app - yfinance is keyless - so any
    credential-shaped literal in tracked source is either a real leak or dead
    weight that will be mistaken for one."""
    import re as _re

    root = pathlib.Path(__file__).resolve().parent.parent
    cred = _re.compile(
        r"AKIA[0-9A-Z]{16}"
        r"|sk-[A-Za-z0-9]{20,}"
        r"|ghp_[A-Za-z0-9]{36}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r"|(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][^\"']{12,}[\"']",
        _re.I)
    offenders = []
    for path in list(root.glob("*.py")) + list((root / "src").glob("*.py")):
        for n, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if cred.search(ln):
                offenders.append(path.name + ":" + str(n))
    assert not offenders, "credential-shaped literal in source: " + str(offenders)


def test_no_exception_text_is_rendered_to_visitors():
    """str(exc) carries the absolute server path - FileNotFoundError and
    PermissionError on a cache file both embed it - so rendering it defeats
    showErrorDetails=false. Visitors get the exception TYPE plus their session
    ref via safe_err(); the full text goes to the operator log.

    The 2026-10-15 security audit found two st.info calls that rendered
    {_bk_exc} and {_wr_exc}: the old scan read only the line holding `st.info(`
    and only matched `{exc`/`{_exc`, so a message wrapped over several lines,
    or an exception named anything else, walked past it. This reads every whole
    st.* call and every name an `except ... as` binds."""
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    assert "def safe_err(" in src, "the safe renderer was removed"

    tree = ast.parse(src)
    caught = {h.name for h in ast.walk(tree)
              if isinstance(h, ast.ExceptHandler) and h.name}
    offenders = []
    for call in ast.walk(tree):
        f = getattr(call, "func", None)
        if not (isinstance(call, ast.Call) and isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name) and f.value.id == "st"):
            continue
        for field in ast.walk(call):
            if not isinstance(field, ast.FormattedValue):
                continue
            v = field.value
            if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "safe_err":
                continue
            names = {n.id for n in ast.walk(v) if isinstance(n, ast.Name)}
            if names & caught or any("exc" in n for n in names):
                offenders.append(f"main.py:{field.lineno}: {{{ast.unparse(v)}}}")
    assert not offenders, "raw exception text reaches the page: " + str(offenders)


def test_a_stranger_cannot_park_files_in_server_memory():
    """2026-10-15 security audit: Streamlit serves PUT /_stcore/upload_file/...
    to any client whether or not the page has a file_uploader, and holds each
    upload in process memory up to server.maxUploadSize (200 MB by default)
    with no total cap. This app takes no files, so the limit drops to the
    1 MB floor - a scripted client can no longer park gigabytes in the single
    process every visitor shares."""
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg.get("server", {}).get("maxUploadSize", 200) <= 1, \
        "server.maxUploadSize is back to letting strangers upload large files"
    src = (root / "main.py").read_text(encoding="utf-8")
    assert "file_uploader" not in src, \
        "a file_uploader needs an upload budget sized for it, not the 1 MB floor"


def test_the_dependency_audit_skips_no_cve():
    """2026-10-20 to 2026-10-22: Pillow 12.3 collected 35 CVEs fixed only in
    13.0.0 while every Streamlit up to 1.66.0 capped Pillow below 13, so John
    approved skipping 24 of them in the Live audit's pip_audit (the other 11
    turned it red on main, run 80). That exception was written to expire: its
    test failed the day Streamlit accepted Pillow 13. Streamlit 1.66.1 does
    (pillow<14), so ME-16 bumped Streamlit, pinned pillow>=13 and deleted all
    the skips. This keeps it closed: no workflow or tool skips a CVE, the
    requirements ask for the fixed Pillow, and the Streamlit installed here
    allows it. A new skip needs John's call and this test changed with it."""
    import importlib.metadata
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    for f in [*(root / ".github" / "workflows").glob("*.yml"), *(root / "tools").glob("*.py")]:
        code = f.read_text(encoding="utf-8")
        assert "ignore-vuln" not in code and "ignore_vuln" not in code, \
            f"{f.name} skips a CVE in the dependency audit"
    audits = [f.name for f in (root / ".github" / "workflows").glob("*.yml")
              if re.search(r"pip_audit -r requirements\.txt\s*$", f.read_text(encoding="utf-8"), re.M)]
    assert audits == ["live-audit.yml"], f"the dependency audit moved or went: {audits}"
    req = (root / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^pillow>=13\b", req, re.M | re.I), "requirements.txt no longer asks for Pillow 13"
    caps = [r.replace(" ", "") for r in importlib.metadata.requires("streamlit") or []
            if r.lower().startswith("pillow")]
    assert caps and not any("<13" in r for r in caps), \
        f"the installed Streamlit caps Pillow below 13 again ({caps})"


def test_no_visitor_control_clears_the_shared_cache():
    """st.cache_data.clear() empties every cached result for EVERY visitor in
    this single process. A public "Refresh" button used to call it (and delete
    the disk cache the spend cap falls back on), so any anonymous viewer could
    force a cold recompute and a fresh Yahoo pull for everyone, in a loop."""
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    hits = [ln.strip() for ln in code
            if "cache_data.clear(" in ln or "clear_cache(" in ln]
    assert not hits, f"a visitor path clears the shared cache: {hits}"


def test_a_crafted_clear_cache_message_clears_nothing():
    """toolbarMode="viewer" hid the "Clear cache" menu, but the websocket
    message behind it still works: Streamlit's AppSession answers a
    `clear_cache` BackMsg from ANY client by emptying st.cache_data and
    st.cache_resource for every visitor. 2026-10-15 security audit: a scripted
    client sending it in a loop pins the single process recomputing cold
    pages. main.py replaces the handler with a no-op once per process; this
    runs that exact patch (lifted out of main.py) against a stub session."""
    import ast
    from streamlit.runtime import caching
    from streamlit.runtime.app_session import AppSession

    # A Streamlit upgrade that renames the handler would turn the patch into a
    # silent no-op on a dead attribute (requirements.txt pins ~=1.66.1).
    assert hasattr(AppSession, "_handle_clear_cache_request"), \
        "Streamlit renamed the clear_cache handler - re-point the patch in main.py"

    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    patch = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == "_handle_clear_cache_request"
                     and getattr(t.value, "id", "") == "AppSession" for t in n.targets)]
    assert len(patch) == 1, "main.py no longer neutralises the clear_cache message"
    impl = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == getattr(patch[0].value, "id", None)]

    # main.py patches the real AppSession class for the whole process, so after
    # any page boot in this process the class already carries the no-op and
    # the control below failed (ME-10: order-dependent). The control runs on a
    # pristine copy of Streamlit's own module, loaded under a private name, so
    # it holds whatever ran before it - and the real class is never touched.
    import importlib.util
    import streamlit.runtime.app_session as _app_session_mod
    spec = importlib.util.spec_from_file_location("_pristine_app_session",
                                                  _app_session_mod.__file__)
    pristine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pristine)

    class Session(pristine.AppSession):
        """A subclass of a fresh copy, so no real class is patched by the test."""

        def __del__(self):  # the stub never started, so there is nothing to shut down
            pass

    calls = []

    class _State:
        def clear(self):
            calls.append("session_state")

    def run(cls):
        calls.clear()
        stub = object.__new__(cls)
        stub._session_state = _State()
        stub._handle_clear_cache_request()
        return list(calls)

    real = (caching.cache_data.clear, caching.cache_resource.clear)
    caching.cache_data.clear = lambda *a, **k: calls.append("cache_data")
    caching.cache_resource.clear = lambda *a, **k: calls.append("cache_resource")
    try:
        # Control: unpatched, the stub really reaches the shared caches.
        assert "cache_data" in run(Session), "the stub no longer reaches the handler"
        ns = {"AppSession": Session}
        exec(compile(ast.Module(body=impl + patch, type_ignores=[]),  # nosec B102
                     "main.py", "exec"), ns)
        assert run(Session) == [], "a crafted clear_cache message still clears caches"
    finally:
        caching.cache_data.clear, caching.cache_resource.clear = real


def test_every_visitor_keyed_cache_has_an_entry_cap():
    """st.cache_data keeps one entry per distinct argument tuple, for the whole
    process, until its ttl. Arguments come from visitors (typed tickers,
    sliders, the book size), so without max_entries a script cycling inputs
    grows the cache without bound inside the ttl - memory the one process
    every visitor shares. 2026-10-15 security audit: eleven of seventeen
    cached functions had no cap. A function with no arguments holds one entry and
    needs none."""
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        a = fn.args
        if not (a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg):
            continue
        for dec in fn.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(target, ast.Attribute)
                    and target.attr in ("cache_data", "cache_resource")):
                continue
            kws = {k.arg for k in dec.keywords} if isinstance(dec, ast.Call) else set()
            if "max_entries" not in kws:
                offenders.append(f"main.py:{fn.lineno} {fn.name}")
    assert not offenders, f"visitor-keyed caches without max_entries: {offenders}"


def test_monte_carlo_is_deterministic_so_caching_is_exact():
    """main.py caches both engines with st.cache_data. That is only a pure
    speed-up if a fresh run returns exactly what the cache holds - i.e. both
    engines are seeded. An unseeded engine would make the cached headline
    differ from a recomputed one."""
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.standard_t(4, (400, 4)) * 0.01, columns=list("ABCD"))
    w = np.full(4, 0.25)
    for fn in (monte_carlo, jump_diffusion_mc):
        a = fn(rets, w, n_simulations=2_000, horizon_days=60)
        b = fn(rets, w, n_simulations=2_000, horizon_days=60)
        assert np.array_equal(a["final_values"], b["final_values"]), fn.__name__
        assert a["cvar"] == b["cvar"], fn.__name__


def test_assets_are_loaded_relative_to_main_not_cwd():
    """open("assets/...") resolves against the working directory, so
    `streamlit run` from any other folder crashed on the hero's first line."""
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    bad = re.findall(r"open\(\s*[\"'](?:assets|static|prototypes)/", src)
    assert not bad, f"cwd-relative asset path(s): {bad}"


def test_flat_price_asset_is_detected_before_the_math():
    """A constant price (money-market fund at $1.00) made risk parity divide
    by zero under the Sample/EWMA estimators: headline "nan%", then a crash.
    main.py drops what flat_columns() names; the rest must solve cleanly."""
    from src.ingestion import flat_columns
    from src.covariance import sample_covariance

    rng = np.random.default_rng(11)
    rets = pd.DataFrame({"A": rng.normal(0, 0.015, 400),
                         "B": rng.normal(0, 0.011, 400),
                         "CASH": np.zeros(400)})
    assert flat_columns(rets) == ["CASH"]
    assert flat_columns(rets[["A", "B"]]) == []
    with np.errstate(all="ignore"):
        w_raw = risk_parity_weights(sample_covariance(rets))
    assert np.isnan(w_raw).all(), "the failure this guards against"
    w = risk_parity_weights(sample_covariance(rets.drop(columns=flat_columns(rets))))
    assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-9


def test_cached_hot_paths_are_deterministic():
    """main.py caches the signal IC, the ES bootstrap and the map payload per
    input. That only changes speed, never a number, if each returns exactly the
    same result twice for the same input."""
    from src.signals import daily_ic, momentum_signal, forward_returns
    from src.pairing import es_confidence_interval
    import json

    from src.topology import build_map_payload

    rng = np.random.default_rng(12)
    # Ends today: online, the map pulls live SPY for its market line and needs
    # these dates to overlap the last two years (fixed 2023 dates gave 51 days).
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    rets = pd.DataFrame(rng.standard_t(4, (520, 4)) * 0.01, index=idx, columns=list("ABCD"))
    px = 100 * (1 + rets).cumprod()
    a = daily_ic(momentum_signal(px), forward_returns(px, horizon=5))
    b = daily_ic(momentum_signal(px), forward_returns(px, horizon=5))
    pd.testing.assert_series_equal(a, b)
    assert es_confidence_interval(rets["A"]) == es_confidence_interval(rets["A"])
    w = np.full(4, 0.25)
    m1 = build_map_payload(rets, w, list("ABCD"))
    m2 = build_map_payload(rets, w, list("ABCD"))
    assert json.dumps(m1, sort_keys=True, default=str) == json.dumps(m2, sort_keys=True, default=str)


def test_freshness_ticker_does_not_tick_every_second():
    """At 1 Hz the freshness fragment held the single process at 93% of a core
    with 15 idle tabs open (stress test 2026-10-15). Keep it at >= 30s."""
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    ticks = re.findall(r'run_every="(\d+)s"', src)
    assert ticks and all(int(t) >= 30 for t in ticks), ticks


def test_gungnir_plaque_matches_its_sketch_and_its_math():
    """The plaque states two-asset equal risk contribution as w1*s1 = w2*s2
    and says correlation cancels: a name 3x as volatile gets 1/3 the capital.
    Check that claim against the solver at several correlations, and that the
    plaque names the sketch's blocks (it said "circle" after the circles were
    redrawn as equal-area blocks)."""
    s1, s2 = 0.30, 0.10
    for rho in (-0.6, 0.0, 0.3, 0.9):
        cov = pd.DataFrame([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])
        w = risk_parity_weights(cov)
        assert abs(w[0] * s1 - w[1] * s2) < 1e-9, (rho, w)
        assert abs(w[0] - 0.25) < 1e-9 and abs(w[1] - 0.75) < 1e-9, (rho, w)
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert 'class="c">block 01<' in src and 'class="c">block 02<' in src
    assert 'class="c">circle' not in src and "Circle A" not in src


def test_public_files_give_crawlers_and_link_previews_a_real_head(tmp_path=None):
    """LinkedIn/Slack previews and search crawlers do not run JavaScript, so
    tools/public_files.py writes the head they read into Streamlit's static
    root at start: title, description, canonical, Open Graph image, robots,
    sitemap, favicon, and a branded 404 for paths a single-page app would
    otherwise serve as a blank page. Idempotent, and it must never drop
    Streamlit's own markup."""
    import tempfile

    import tools.public_files as pf

    d = pathlib.Path(tmp_path or tempfile.mkdtemp())
    (d / "index.html").write_text(
        "<!doctype html><html><head><title>Streamlit</title><script src='./static/js/a.js'>"
        "</script></head><body><noscript>You need JS.</noscript><div id='root'></div>"
        "</body></html>", encoding="utf-8")
    pf.install(d, "https://example.test")
    first = (d / "index.html").read_text(encoding="utf-8")
    pf.install(d, "https://example.test")
    assert (d / "index.html").read_text(encoding="utf-8") == first, "not idempotent"
    assert first.count(pf.MARK) == 1 and first.count("<title>") == 1
    for tag in ('name="description"', 'property="og:image" content="https://example.test/og-image.png"',
                'rel="canonical" href="https://example.test/"', "summary_large_image",
                "location.replace('/404.html')", "./static/js/a.js", "<div id='root'>",
                'id="boot-skel"', 'href="./app/static/app.css"'):
        assert tag in first, tag
    assert "Sitemap: https://example.test/sitemap.xml" in (d / "robots.txt").read_text()
    assert "<loc>https://example.test/</loc>" in (d / "sitemap.xml").read_text()
    for f in ("og-image.png", "favicon.png", "apple-touch-icon.png", "404.html"):
        assert (d / f).stat().st_size > 0, f
    assert 'name="robots" content="noindex"' in (d / "404.html").read_text()


def test_the_deployed_page_names_the_commit_it_was_built_from(tmp_path=None):
    """"Did Railway deploy it?" was answered by guesswork twice (2026-09-03,
    2026-09-09): the host posts nothing back to GitHub. Railway sets
    RAILWAY_GIT_COMMIT_SHA on a GitHub deploy; public_files stamps it into the
    raw HTML so the Live audit can compare it with the commit it expects."""
    import tempfile

    import tools.public_files as pf

    d = pathlib.Path(tmp_path or tempfile.mkdtemp())
    (d / "index.html").write_text("<html><head><title>Streamlit</title></head>"
                                  "<body><noscript></noscript></body></html>", encoding="utf-8")
    pf.install(d, "https://example.test", build="0123456789abcdef0123")
    assert '<meta name="meleona-build" content="0123456789ab">' in (d / "index.html").read_text()
    pf.install(d, "https://example.test", build="")
    page = (d / "index.html").read_text()
    assert '<meta name="meleona-build" content="unknown">' in page and page.count("meleona-build") == 1
    # site_check reads the tag, and on main it must match the commit under test
    check = (pathlib.Path(__file__).resolve().parent.parent / "tools" / "site_check.mjs").read_text()
    assert "meleona-build" in check and "EXPECT_BUILD" in check


def test_the_web_process_keeps_every_preset_warm_before_it_expires():
    """Live audit run 24: ten hours after a deploy the 6-hour cache had
    expired and the first visitor to pick Semiconductors waited 101.6s on
    Railway (12.1s on a GitHub runner). The cache is the web container's own
    disk, so a separate cron service would warm a disk nobody reads: the web
    process itself re-warms every preset, each pass refetching anything older
    than (max age - interval), so nothing a visitor picks has expired."""
    from src.ingestion import CACHE_MAX_AGE_HOURS
    from tools.warm_cache import pass_max_age

    assert pass_max_age(5) == CACHE_MAX_AGE_HOURS - 5 and pass_max_age(5) + 5 <= CACHE_MAX_AGE_HOURS
    assert pass_max_age(CACHE_MAX_AGE_HOURS + 1) == 0      # never negative
    root = pathlib.Path(__file__).resolve().parent.parent
    web = (root / "Procfile").read_text().strip()
    loop, serve = web.find("tools.warm_cache --every"), web.find("streamlit run")
    assert 0 < loop < serve, web                              # started before the app...
    assert "&)" in web[loop:serve], web                       # ...in the background
    every = float(web[loop:].split("--every", 1)[1].split()[0].rstrip(")&"))
    assert 0 < every < CACHE_MAX_AGE_HOURS, every


def test_the_warm_pass_covers_every_fetch_a_preset_page_makes():
    """John, 2026-10-20: "fix the slow first basket switch". Right after a
    deploy the deployed walk took 93.1s to switch to Semiconductors (101.6s in
    run 24). The headline came in seconds; the rest of the page then paid for
    cold pulls the warmer never made: it warmed each preset's 2y prices only,
    while the page also asks for full histories (Book and Grit tab grit), ten
    years (tail fit), page-wide pulls (conviction, factors, anchors, ^IRX) and
    the security master. Railway's disk is wiped on every deploy and the cache
    expires every 6h. A warm pass must now cover every fetch the page makes
    for a preset - recorded from the real main.py, so the two cannot drift.
    The crisis pair too (John, 2026-10-20: "fix the crisis pair slow download
    next"): the page replays its default flyer and anchor through every named
    crisis, and the warm pass now picks that same pair by the same rule."""
    import sys
    import src.security_master as sm_mod
    import tools.warm_cache as wc
    from src.ingestion import PRESETS

    def recorder(store):
        def fetch(tickers=None, period="2y", *a, **kw):
            align = kw.get("align", a[1] if len(a) > 1 else True)
            store.add((tuple(sorted(tickers or [])), period, bool(align)))
            return _offline_market_prices(tickers, period)
        return fetch

    page, page_sm = set(), set()

    def page_security_master(tickers, *a, **k):
        page_sm.update(tickers)
        return pd.DataFrame({"isin": "unavailable", "dividends_paid": 0, "total_dividends": 0.0,
                             "splits": 0, "last_split_date": None, "last_split_ratio": None},
                            index=list(tickers))

    real_sm = sm_mod.security_master
    sm_mod.security_master = page_security_master
    try:
        at, restore = _offline_app(market=recorder(page))
        try:
            pick = [w.label for w in at.selectbox].index("Preset basket")
            semis = next(n for n in PRESETS if n.startswith("Semiconductors"))
            at.selectbox[pick].set_value(semis).run()
            assert not at.exception, [e.value for e in at.exception]
        finally:
            restore()
    finally:
        sm_mod.security_master = real_sm

    warm, warm_sm = set(), set()
    saved = []
    mods = [m for n, m in sys.modules.items() if n.startswith("src.") or n == "tools.warm_cache"]
    fakes = {"fetch_prices": recorder(warm),
             "fetch_dollar_volume": lambda tickers=None, period="2y", **k: (
                 recorder(warm)(tickers, period)),
             "fetch_risk_free_rate": lambda *a, **k: 0.045,
             "security_master": lambda tickers, *a, **k: warm_sm.update(tickers)}
    for mod in mods:
        for attr, fake in fakes.items():
            if hasattr(mod, attr):
                saved.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, fake)
    sleep, wc.PAUSE_SECONDS = getattr(wc, "PAUSE_SECONDS", None), 0
    try:
        wc.run_pass()
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)
        if sleep is not None:
            wc.PAUSE_SECONDS = sleep

    pairs = {k for k in page if k[1] == "max" and len(k[0]) == 2}
    assert len(pairs) >= 2, f"the page replayed no crisis pair for each basket: {pairs}"
    missing = sorted(page - warm)
    assert not missing, f"fetches the page makes that no warm pass covers: {missing}"
    assert page_sm <= warm_sm, f"security master not warmed for: {sorted(page_sm - warm_sm)}"


def test_a_cache_file_is_never_seen_half_written():
    """The warmer rewrites cache files while visitors read them: a write goes
    to a temporary name and is renamed into place, so a reader sees the old
    file or the new one, never a torn parquet."""
    import inspect
    import tempfile

    from src import ingestion

    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2))
    d = pathlib.Path(tempfile.mkdtemp())
    ingestion._write_parquet(df, str(d / "x.parquet"))
    assert [p.name for p in d.iterdir()] == ["x.parquet"]
    pd.testing.assert_frame_equal(pd.read_parquet(d / "x.parquet"), df, check_freq=False)
    src = inspect.getsource(ingestion)
    assert ".to_parquet(cache_path)" not in src, "a cache write bypasses _write_parquet"


def test_src_is_pure_computation_and_never_imports_the_ui():
    """The layering CLAUDE.md promises: src/ is a numpy/pandas library the UI
    composes, so it is testable without a browser and auditable on its own.
    One `import streamlit` (or a reach back into main/tools/tests) inside src/
    would quietly couple the math to the page - this fails the build instead."""
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    banned = {"streamlit", "main", "tools", "tests"}
    offenders = []
    for f in sorted((root / "src").glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            offenders += [f"{f.name}: {m}" for m in mods if m.split(".")[0] in banned]
    assert not offenders, f"src/ must not import the UI layer: {offenders}"


def test_streamlit_header_bar_is_hidden():
    """Streamlit's fixed 60px header (theme-coloured, holding only the viewer
    menu) covered the hero crest on the live site. It is hidden in app.css; a
    Streamlit upgrade that renames the testid would bring the bar back."""
    css = (pathlib.Path(__file__).resolve().parent.parent / "static" / "app.css"
           ).read_text(encoding="utf-8")
    assert '[data-testid="stHeader"] { display: none !important; }' in css


def test_no_raw_exception_text_reaches_the_page():
    """str(exc) can carry an absolute server path, which is why safe_err()
    exists and showErrorDetails is off. The Book and Risk Topology still
    printed `{exc}` into st.info - twice the same slip, so it is a rule now.
    The one sanctioned exception is NoDataError, whose messages are written
    for visitors (src/ingestion.py)."""
    import ast
    tree = ast.parse((pathlib.Path(__file__).resolve().parent.parent / "main.py")
                     .read_text(encoding="utf-8"))
    hits = []
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler) or not h.name:
            continue
        if isinstance(h.type, ast.Name) and h.type.id == "NoDataError":
            continue
        for node in ast.walk(h):
            fv = isinstance(node, ast.FormattedValue) and isinstance(node.value, ast.Name)
            sv = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in ("str", "repr") and node.args
                  and isinstance(node.args[0], ast.Name))
            name = node.value.id if fv else node.args[0].id if sv else None
            if name == h.name:
                hits.append(f"line {node.lineno}: {name}")
    assert not hits, f"exception text interpolated onto the page: {hits}"


# --- council fixes 2026-10-17: replay ---
def test_replay_excludes_an_asset_priced_at_or_below_zero_in_the_window():
    """WTI (CL=F) settled -37.63 on 2020-04-20. pct_change() divides by that
    negative base, so the +47.64 $/bbl rebound to 10.01 replayed as -126.6%
    and the Futures x COVID verdict read CVaR 99.99%. Simple returns are
    undefined across a sign change: the name is excluded (and so disclosed),
    the rest keep the window."""
    import src.scenarios as sc

    idx = pd.bdate_range("2020-03-02", periods=30)
    px = pd.DataFrame({"CL": np.linspace(30.0, 20.0, 30),
                       "ES": np.linspace(3000.0, 2800.0, 30)}, index=idx)
    px.loc[idx[15], "CL"], px.loc[idx[16], "CL"] = -37.63, 10.01
    start, end = idx[5], idx[25]
    orig = sc.fetch_prices
    try:
        sc.fetch_prices = lambda *a, **k: px
        r = sc.replay_returns(["CL", "ES"], str(start.date()), str(end.date()))
        assert list(r.columns) == ["ES"], (
            f"a negative price was replayed as returns: {list(r.columns)}, "
            f"CL on {idx[16].date()} = {r['CL'].get(idx[16]) if 'CL' in r else None}")
        assert r.index[0] == start and r.index[-1] == end, (r.index[0], r.index[-1])

        # A non-positive close just BEFORE the window is the lead-in base for
        # the opening bar: it must not be used either.
        lead_neg = px.copy()
        lead_neg.loc[idx[4], "ES"] = -1.0
        lead_neg["CL"] = np.linspace(30.0, 20.0, 30)
        sc.fetch_prices = lambda *a, **k: lead_neg
        r2 = sc.replay_returns(["CL", "ES"], str(start.date()), str(end.date()))
        assert (r2.loc[start:] > -1.0).all().all() and start not in r2.index, (
            "a non-positive lead-in close was used as the opening bar's base")
    finally:
        sc.fetch_prices = orig


def test_replay_excludes_a_name_that_lists_partway_through_the_window():
    """IVV listed 2000-05-15 and IWM 2000-05-22, inside the Dot-com window
    (2000-03-01..2001-04-30). The row-wise dropna cut EVERY name to the late
    listing, so 'Index core' lost the Mar-May 2000 leg while the caption
    claimed the full window and named nothing as excluded. A late lister is
    now excluded like an absent one, and the survivors keep the window."""
    import src.scenarios as sc

    idx = pd.bdate_range("2000-02-01", periods=80)
    px = pd.DataFrame({"SPY": np.linspace(140.0, 120.0, 80),
                       "QQQ": np.linspace(200.0, 150.0, 80),
                       "IWM": np.linspace(50.0, 45.0, 80)}, index=idx)
    start, end = idx[20], idx[70]
    px.loc[:idx[45], "IWM"] = np.nan           # lists 25 sessions into the window
    px.loc[:idx[21], "QQQ"] = np.nan           # first print 2 sessions late: a holiday, kept
    orig = sc.fetch_prices
    try:
        sc.fetch_prices = lambda *a, **k: px
        r = sc.replay_returns(["SPY", "QQQ", "IWM"], str(start.date()), str(end.date()))
        assert "IWM" not in r.columns and set(r.columns) == {"SPY", "QQQ"}, (
            f"late lister kept and the window cut to {r.index[0].date()}: {list(r.columns)}")
        assert r.index[0] <= idx[23] and r.index[-1] == end, (r.index[0], r.index[-1])
    finally:
        sc.fetch_prices = orig


def test_replay_caption_states_the_dates_actually_replayed():
    """The caption printed the nominal window from HISTORICAL_REGIMES, so a
    replay cut short still claimed the whole crisis. It must be built from the
    replayed returns' own first and last dates."""
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py"
           ).read_text(encoding="utf-8")
    # The verb follows the engine (ME-07): "Replaying" or "Calibrating to".
    lines = [ln for ln in src.splitlines() if "actual market returns from" in ln]
    assert len(lines) == 1, lines
    assert "shocked_returns.index[0]" in lines[0] and "shocked_returns.index[-1]" in lines[0], (
        f"caption does not print the replayed dates: {lines[0].strip()}")


# --- council fixes 2026-10-17: feed-liquidity ---
def test_weekend_crypto_rows_do_not_pad_stock_columns_with_stale_prices():
    """BTC-USD trades on weekends; the forward-fill ran before the align, so
    every Saturday and Sunday survived with the stock's Friday price repeated:
    208 zero-return SPY rows in two years, SPY vol 0.154 against a true 0.182,
    and 252 rows spanning 0.69 calendar years under the '1-Year' labels."""
    from src.ingestion import _clean_price_frame

    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(3)
    btc = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))), days)
    biz = days[days.dayofweek < 5]
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.011, len(biz)))), biz)
    raw = pd.concat({"BTC-USD": btc, "SPY": spy}, axis=1)   # yf.download's outer join
    for align in (True, False):
        px = _clean_price_frame(raw, align)
        weekend = int((px.index.dayofweek >= 5).sum())
        assert weekend == 0, f"align={align}: {weekend} weekend rows kept"
        rets = px.pct_change().dropna()
        assert int((rets["SPY"] == 0).sum()) == 0, f"align={align}: stale SPY rows"
        # Friday close -> Monday close: nothing is lost from crypto's path.
        assert abs(px["BTC-USD"].iloc[-1] / px["BTC-USD"].iloc[0]
                   - btc[px.index[-1]] / btc[px.index[0]]) < 1e-12


def test_futures_contract_volume_is_not_reported_as_dollar_volume():
    """Yahoo's =F Volume counts contracts and Close is per unit, so close x
    volume is not dollars (no contract multiplier): SI=F read 2,240 days to
    exit a $5B book that clears in half a day. The page already says futures
    are 'excluded, not estimated'; the feed now makes that true, the same
    way an FX pair with no volume is. Offline: the downloader is replaced."""
    import src.ingestion as ing

    idx = pd.bdate_range("2025-01-02", periods=30)
    cols = ["SI=F", "SPY"]
    frame = pd.concat({"Close": pd.DataFrame(30.0, idx, cols),
                       "Volume": pd.DataFrame(6e4, idx, cols)}, axis=1)
    real = (ing.yf.download, ing.check_budget, ing.guarded_session)
    try:
        ing.yf.download = lambda *a, **k: frame
        ing.check_budget = lambda *a, **k: None
        ing.guarded_session = lambda: None
        adv = ing.average_dollar_volume(cols, use_cache=False).reindex(cols).fillna(0.0)
    finally:
        ing.yf.download, ing.check_budget, ing.guarded_session = real
    assert adv["SPY"] == 30.0 * 6e4, adv.to_dict()
    assert adv["SI=F"] == 0.0, f"contracts x price shown as dollars: {adv['SI=F']}"
    prof = liquidity_profile(days_to_liquidate([0.5, 0.5], adv, book_value=5e9))
    assert prof["no_volume"] == ["SI=F"], prof


def test_liquidity_adjustment_never_turns_a_tail_gain_into_a_bigger_gain():
    """CVaR <= 0 means the worst 5% of years still gain: there is no tail loss
    for an unwind to widen. Scaling it anyway reported -5.3% against a -5.0%
    headline - being stuck holding read as a better outcome - and the verdict
    said the tail 'widens to' that gain."""
    import ast

    for days in (1.0, 60.0, 500.0, np.inf):
        lv = liquidity_adjusted_cvar(-0.20, days)
        assert lv["lvar"] == -0.20, f"{days} exit days: lvar {lv['lvar']}"
    assert liquidity_adjusted_cvar(0.20, 60.0)["lvar"] > 0.20   # losses still widen

    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    guarded = {"that tail widens to ": False, " for the unwind": False}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test).replace(" ", "")
        body = "".join(ast.unparse(b) for b in node.body)
        orelse = "".join(ast.unparse(b) for b in node.orelse)
        for phrase in guarded:
            if ((phrase in body and "cvar']>0" in test)
                    or (phrase in orelse and "cvar']<=0" in test)):
                guarded[phrase] = True
    assert all(guarded.values()), f"widening shown for a negative CVaR: {guarded}"


# --- council fixes 2026-10-17: captions ---

def _main_source_and_tree():
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    return src, ast.parse(src)


def _lift_assign(tree, name):
    """The one module- or function-level assignment to `name` in main.py, ready
    to exec: main.py is a Streamlit script, so its wording is tested by lifting
    the statement that builds it rather than importing the page."""
    import ast
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return hits[0] if len(hits) == 1 else None


def _exec_nodes(nodes, ns):
    import ast
    exec(compile(ast.Module(body=list(nodes), type_ignores=[]),  # nosec B102
                 "main.py", "exec"), ns)
    return ns


def test_tail_fit_drops_a_young_listing_by_name_instead_of_shortening_the_whole_fit():
    """The 10y tail fit used an aligned pull, so one 2021 listing (RIVN in
    High-volatility movers) cut the whole fit to 5.1 years while the caption
    said "~10 years" and the "lack the full history" note could never fire."""
    import ast
    from src.ingestion import _clean_price_frame

    src, tree = _main_source_and_tree()
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_tail_fit"]
    assert len(fn) == 1, "load_tail_fit is gone from main.py"
    fn[0].decorator_list = []

    idx = pd.bdate_range("2016-10-14", "2026-10-14")
    rng = np.random.default_rng(5)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 3)), axis=0)),
                      index=idx, columns=["OLD", "MID", "IPO"])
    px.loc[px.index < "2021-11-10", "IPO"] = np.nan

    def fetch_prices(tickers, period="2y", align=True, **_):
        return _clean_price_frame(px[list(tickers)], align)

    ns = {"pd": pd, "np": np, "fetch_prices": fetch_prices,
          "load_universe": lambda t, period="2y": fetch_prices(list(t), period),
          "mcneil_frey_tail": lambda pr: {"fitted": True, "n": len(pr)}}
    _exec_nodes(fn, ns)
    out = ns["load_tail_fit"](("OLD", "MID", "IPO"), (0.4, 0.3, 0.3), False)
    _, days, dropped = out[:3]
    assert dropped == ["IPO"], (f"a 2021 listing shrank the fit to {days} days "
                                f"({days / 252:.1f}y) and was not named: dropped={dropped}")
    assert days > 2500, days
    assert len(out) == 4 and abs(out[3] - 10.0) < 0.05, f"span not reported from the index: {out[1:]}"
    assert "~10 years" not in src, "the tail-fit caption still hard-codes ~10 years"
    assert "{_tf_years:.1f} years" in src, "the tail-fit caption does not print the measured span"


def test_fan_chart_caption_claims_block_resampling_only_when_it_happened():
    """The cone caption said "resampled in N-day blocks so crash weeks stay
    intact" for jump-diffusion (parametric, nothing resampled) and for replay
    windows under 30 days (i.i.d. fallback, block_days=1)."""
    _, tree = _main_source_and_tree()
    node = _lift_assign(tree, "_se_txt")
    assert node is not None, "the fan-chart sampling-error caption moved"
    rng = np.random.default_rng(0)
    w = np.array([0.5, 0.5])
    long_hist = pd.DataFrame(rng.standard_t(4, (500, 2)) * 0.01, columns=["A", "B"])
    short_hist = long_hist.iloc[:22]

    def caption(mc):
        return _exec_nodes([node], {"mc": mc, "np": np})["_se_txt"]

    jd = caption(jump_diffusion_mc(long_hist, w, n_simulations=400))
    assert "resampled" not in jd and "crash weeks" not in jd, f"jump-diffusion: {jd}"
    assert "parametric Merton paths" in jd, jd
    iid = caption(monte_carlo(short_hist, w, n_simulations=400))
    assert "crash weeks stay intact" not in iid, f"22-day replay (i.i.d.): {iid}"
    assert "resampled day by day (window too short for blocks)" in iid, iid
    blk = monte_carlo(_clustered_returns(n=800), np.array([1.0]), n_simulations=400)
    assert blk["block_days"] > 1
    # A geometric block of mean N keeps nothing intact by construction, so the
    # caption says what the bootstrap does and claims nothing more (ME-03,
    # 2026-10-22: the default page printed "2-day blocks so crash weeks stay
    # intact").
    assert f"resampled in blocks averaging {blk['block_days']} days" in caption(blk), caption(blk)
    assert "crash weeks" not in caption(blk), caption(blk)


def test_market_drawdown_shock_compounds_to_the_labelled_drawdown():
    """A -50 'Market drawdown shock' added -0.5/252 to every day, which
    compounds to -39% over the simulated year, not the -50% the label and the
    audit line state."""
    import ast
    _, tree = _main_source_and_tree()
    ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "drawdown_shock != 0"]
    assert len(ifs) == 1, "the drawdown-shock step moved"
    for shock in (-50, -30, -5):
        from src.covariance import TRADING_DAYS
        ns = _exec_nodes(ifs[0].body, {"np": np, "drawdown_shock": shock,
                                       "TRADING_DAYS": TRADING_DAYS,
                                       "shocked_returns": pd.DataFrame(np.zeros((252, 2)))})
        grown = (1 + ns["shocked_returns"]).prod()
        assert np.allclose(grown, 1 + shock / 100), \
            f"shock {shock}% compounds to {float(grown.iloc[0]) - 1:+.1%} over 252 days"


def test_defensive_pair_says_who_really_holds_the_money():
    """Inverse-vol weights give the anchor under half when the flyer is the
    steadier leg (Index core SPY/GLD: anchor 47%), yet the text said the anchor
    holds most of the MONEY next to an SVG reading 47% CAPITAL."""
    src, tree = _main_source_and_tree()
    node = _lift_assign(tree, "_bv_money")
    assert node is not None, ("the pair text states 'the anchor holds most of the "
                              "MONEY' as a fixed string, whatever the weights")

    def say(w_b, vol_a, vol_b):
        pw = {"w_a": 1 - w_b, "w_b": w_b, "vol_a": vol_a, "vol_b": vol_b}
        return _exec_nodes([node], {"pw": pw, "flyer": "SPY", "bv_anchor": "GLD"})["_bv_money"]

    under = say(0.473, 0.132, 0.146)
    assert "anchor holds most" not in under, f"anchor at 47%: {under}"
    assert "SPY holds more of the" in under and "steadier leg" in under, under
    over = say(0.75, 0.30, 0.10)
    assert "the anchor holds most of the MONEY" in over, over
    assert "steady anchor holds most of the capital" not in src, \
        "the read-me still says the anchor always holds most of the capital"


def test_noise_ceiling_uses_the_ewma_effective_sample_size():
    """Under EWMA (lambda 0.94) the matrix carries ~32 effective days, but the
    Marcenko-Pastur line used T = len(returns) = 503: the ceiling sat ~1.95x
    too low and painted noise factors as signal."""
    import ast
    from src.covariance import RISKMETRICS_LAMBDA

    _, tree = _main_source_and_tree()
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == "eigen_factor_panel"][0]
    call = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == "marcenko_pastur_bounds"][0]
    t_arg = call.args[1]
    pre = []
    if isinstance(t_arg, ast.Name):
        pre = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == t_arg.id for t in n.targets)]
    returns = pd.DataFrame(np.zeros((503, 3)))
    expect = {"EWMA": (1 + RISKMETRICS_LAMBDA) / (1 - RISKMETRICS_LAMBDA),
              "Sample": 503, "Ledoit-Wolf": 503}
    for method, want in expect.items():
        ns = _exec_nodes(pre, {"cov_method": method, "returns": returns,
                               "RISKMETRICS_LAMBDA": RISKMETRICS_LAMBDA})
        got = eval(compile(ast.Expression(t_arg), "main.py", "eval"), ns)  # nosec B307
        assert abs(got - want) < 1e-9, f"{method}: noise ceiling drawn at T={got}, expected {want:.1f}"
    printed = [n for n in ast.walk(fn) if isinstance(n, ast.JoinedStr)
               and "pure noise" in ast.unparse(n)]
    assert printed and any(isinstance(v, ast.FormattedValue)
                           and ast.dump(v.value) == ast.dump(t_arg)
                           for v in printed[0].values), "the caption prints a different T"


def test_correlation_watch_says_not_enough_history_instead_of_stable_at_nan():
    """With 59 return rows and a 63-day window the rolling correlation is
    empty, latest is NaN, and the page said 'Stable: A x B at +nan'."""
    import ast
    src, tree = _main_source_and_tree()

    def seg(n):
        return ast.get_source_segment(src, n) or ""

    cands = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and "Stable:" in seg(n) and "Concentration reading" in seg(n)
             and "pick_a == pick_b" not in ast.unparse(n.test)]
    elifs = {id(c.orelse[0]) for c in cands
             if len(c.orelse) == 1 and isinstance(c.orelse[0], ast.If)}
    head = [c for c in cands if id(c) not in elifs]
    assert len(head) == 1, "the Correlation Watch verdict moved"

    said = []

    class _St:
        def __getattr__(self, kind):
            return lambda msg, *a, **k: said.append((kind, msg))

    _exec_nodes(head, {"st": _St(), "np": np, "roll": pd.Series(dtype=float),
                       "latest": float("nan"), "win": 63, "thresh": 0.75,
                       "pick_a": "A", "pick_b": "B"})
    assert not any("Stable" in m or "Concentration" in m for _, m in said), said
    assert any(k == "info" and "not enough history for a 63-day window" in m.lower()
               for k, m in said), said
    said.clear()
    _exec_nodes(head, {"st": _St(), "np": np, "roll": pd.Series([0.9]),
                       "latest": 0.9, "win": 21, "thresh": 0.75,
                       "pick_a": "A", "pick_b": "B"})
    assert said and said[0][0] == "warning" and "Concentration reading" in said[0][1], said


# --- council fixes 2026-10-17: book-sharpe-anchor ---

def _offline_market_prices(tickers=None, period="2y", **_kw):
    """Ten years of a deterministic market for booting main.py without Yahoo:
    one common factor with rare crash days (so named crises leave a drawdown
    for grit to replay) and a per-ticker beta and idio vol seeded by name."""
    import hashlib
    from src.ingestion import DEFAULT_UNIVERSE
    idx = pd.bdate_range(end="2026-10-14", periods=252 * 10)
    rng = np.random.default_rng(7)
    mkt = rng.normal(0.0004, 0.011, len(idx))
    mkt[rng.random(len(idx)) < 0.004] -= 0.05
    cols = {}
    for t in list(tickers or DEFAULT_UNIVERSE):
        r = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        beta, idio = r.uniform(0.3, 1.6), r.uniform(0.10, 0.40)
        ret = beta * mkt + r.normal(0.0002, idio / np.sqrt(252), len(idx))
        cols[t] = 100 * np.exp(np.cumsum(np.log1p(np.clip(ret, -0.6, 0.6))))
    years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10}.get(period, 10)
    return pd.DataFrame(cols, index=idx).iloc[-252 * years:]


def _offline_app(setup=None, market=None, rf=0.045, adv=5e9, dollar_volume=None):
    """Run the real main.py under AppTest on _offline_market_prices (or
    `market`); returns (AppTest, restore). Every src module is imported BEFORE
    patching, so a module-level `from src.ingestion import fetch_prices` is
    patched and then restored too, and the Streamlit cache is cleared before
    and after, so no live frame leaks in and no synthetic frame outlives the
    test."""
    import importlib
    import pkgutil
    import sys
    import src
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    for m in pkgutil.iter_modules(src.__path__):
        importlib.import_module(f"src.{m.name}")
    market = market or _offline_market_prices
    fakes = {
        "fetch_prices": market,
        "fetch_dollar_volume": dollar_volume or (lambda tickers=None, period="2y", **k: pd.DataFrame(
            5e9, index=market(tickers, period).index,
            columns=list(tickers))),
        "average_dollar_volume": lambda tickers=None, *a, **k: pd.Series(
            adv, index=list(tickers or [])),
        "fetch_risk_free_rate": lambda *a, **k: rf,
        "provenance": lambda *a, **k: None,
    }
    saved = []
    for name, mod in list(sys.modules.items()):
        if name.startswith("src."):
            for attr, fake in fakes.items():
                if hasattr(mod, attr):
                    saved.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, fake)

    def restore():
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)
        st.cache_data.clear()

    # Before, not only after: test_full_app_boots leaves the live prices and
    # ^IRX cached, and their ttls (60 s, 1 h) ran out mid-test (2026-10-20).
    st.cache_data.clear()
    try:
        app = pathlib.Path(__file__).resolve().parent.parent / "main.py"
        at = AppTest.from_file(str(app), default_timeout=300)
        at.run()
        assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    except BaseException:
        restore()
        raise
    return at, restore


def _book_grit(at) -> dict:
    """{ticker: grit} as The Book's sentences print it ("grit 73 of 100")."""
    import re
    out = {}
    for m in at.markdown:
        hit = re.search(r'line-height:1\.55;">([A-Z0-9.^=-]+) .*?grit (\d+) of 100',
                        m.value)
        if hit:
            out[hit.group(1)] = int(hit.group(2))
    return out


def test_the_books_grit_is_the_grit_zones_full_history_grit():
    """Council 2026-10-17: The Book scored grit on the 2-year window, where no
    named crisis falls, so the resilience third was 0 for every name and the
    same page printed NVDA 6.9 in the Book against 73.2 in the Grit Zone tab.
    Grit is LONG-horizon (src/narrative.py ruling): the Book must print the
    tab's numbers for the same universe."""
    at, restore = _offline_app()
    try:
        loaded = list([w for w in at.selectbox if w.label.startswith("Anchor")][0].options)
        # Defensive anchors widen the Book's peer group (stated on screen as
        # "across N names"); with them off the two universes are identical.
        at.toggle(key="book_fallback").set_value(False).run()
        tab = grit_scores(loaded)["scores"]["grit_score"]
        book = _book_grit(at)
        assert set(book) and set(book) <= set(loaded), f"no Book grit parsed: {book}"
        wrong = {t: (g, round(float(tab[t]))) for t, g in book.items()
                 if g != round(float(tab[t]))}
        assert not wrong, f"Book grit vs Grit Zone grit (book, tab): {wrong}"
        # With the defensive shelf on, the same: the shelf is offered as an
        # anchor, never added to the visitor's universe (ME-08, 2026-10-22:
        # it printed "17 of 17 names" and V 67 against 61 in the tab).
        at.toggle(key="book_fallback").set_value(True).run()
        book = _book_grit(at)
        assert set(book) <= set(loaded), f"grit printed for names outside the basket: {set(book) - set(loaded)}"
        wrong = {t: (g, round(float(tab[t]))) for t, g in book.items()
                 if g != round(float(tab[t]))}
        assert not wrong, f"Book grit vs Grit Zone grit, shelf on (book, tab): {wrong}"
        head = [m.value for m in at.markdown if " book - " in m.value and "names hold" in m.value]
        assert head and f"of your {len(loaded)} names hold" in head[0], head
    finally:
        restore()


def test_the_book_lists_an_outside_etf_only_when_it_is_the_anchor():
    """ME-03 (2026-10-22): the default page listed TLT, GLD, XLP and USMV as
    "LEFT OUT" - XLP "simply did not rank in the top 3" though a fallback can
    never be a flyer - and they held 21.5% of the book. An outside ETF appears
    only as the anchor it was chosen to be, and says it is from outside."""
    import re
    from src.pairing import DEFENSIVE_ANCHOR_TICKERS
    at, restore = _offline_app()
    try:
        loaded = list([w for w in at.selectbox if w.label.startswith("Anchor")][0].options)
        cards = {}
        for m in at.markdown:
            hit = re.match(r"\s*(LEADS|ANCHOR|NO ROLE|LEFT OUT)\s+((?:\d+\.\d%|<0\.1%) of the book\s+)?"
                           r"([A-Z0-9.^=-]+) (.*)",
                           re.sub(r"<[^>]+>", " ", m.value)) if "border-left:3px solid" in m.value else None
            if hit:
                # a NO ROLE card, and only that one, says what it holds (ME-12 review)
                assert bool(hit.group(2)) == (hit.group(1) == "NO ROLE"), hit.group(0)[:80]
                cards[hit.group(3)] = (hit.group(1), hit.group(4))
        assert cards, "no Book cards parsed"
        shelf = [t for t in DEFENSIVE_ANCHOR_TICKERS if t not in loaded]
        for t, (role, text) in cards.items():
            if t in shelf:
                assert role == "ANCHOR" and "outside the chosen universe" in text, (t, role, text)
        assert set(cards) - set(shelf) == set(loaded), sorted(cards)
    finally:
        restore()


def test_sharpe_does_not_move_with_the_vol_target_slider():
    """Council 2026-10-17: Sharpe was taken on returns @ vol-targeted weights,
    i.e. (L*mu - rf)/(L*sigma) - idle cash earning 0 and borrowing free - so
    the same strategy read 1.615 unlevered and 0.975 at L=0.28. With cash and
    financing at rf, Sharpe is leverage-invariant (Sharpe 1994)."""
    at, restore = _offline_app()
    try:
        def sharpe():
            return [m.value for m in at.metric if m.label == "Sharpe ratio"][0]
        plain = sharpe()
        [c for c in at.checkbox if c.label == "Target volatility"][0].check().run()
        for tv in (5, 10, 30):
            [s for s in at.slider if s.label == "Target annual vol (%)"][0].set_value(tv).run()
            assert sharpe() == plain, (
                f"Sharpe {sharpe()} at a {tv}% vol target vs {plain} unlevered")
        cap = " ".join(c.value for c in at.caption if c.value.startswith("Sharpe ="))
        assert "unlevered" in cap, f"Sharpe caption does not say what it is on: {cap}"
    finally:
        restore()


def test_an_offline_boot_never_reads_what_a_live_boot_left_in_the_cache():
    """Live audit 2026-10-20 (run 39347809039): the Sharpe slider test read
    0.96 unlevered, then 0.64 at a 5% vol target. Tests run in name order, so
    it was the first offline boot after test_full_app_boots, which runs main.py
    on live Yahoo and leaves st.cache_data full. Its first run read the live
    prices (load_universe, cached 60 s) and the live ^IRX (cached 1 h); the
    60 s ran out between two reruns, and the next read the synthetic prices
    with the live 4.075% rate - exactly 0.64. Passed whenever the gap fell
    outside that minute. An offline boot must start from an empty cache."""
    import streamlit as st
    # Stand-in for the live boot: another market (ending a week earlier) and
    # another T-bill rate, and a cache it leaves full on the way out.
    seed, undo = _offline_app(
        market=lambda tickers=None, period="2y", **k:
            _offline_market_prices(tickers, period).iloc[:-5],
        rf=0.02)
    real_clear = st.cache_data.clear
    st.cache_data.clear = lambda *a, **k: None   # test_full_app_boots never clears
    try:
        undo()
    finally:
        st.cache_data.clear = real_clear
    at, restore = _offline_app()
    try:
        data = [c.value for c in at.caption if c.value.startswith("Data: ")]
        rate = " ".join(c.value for c in at.caption if c.value.startswith("Sharpe ="))
        assert data and "through 2026-10-14" in data[0], f"prices from the earlier boot: {data}"
        assert "Risk-free rate: 4.50%" in rate, f"T-bill rate from the earlier boot: {rate}"
    finally:
        restore()


def test_changing_weighting_keeps_the_visitors_balance_anchor():
    """Council 2026-10-17: the Balance-tab anchor had no key and its index
    followed argmax(weights), so switching Weighting re-created the widget and
    silently swapped the visitor's anchor (GOOGL -> NVDA) and every number
    under it. Weighting does not enter the Balance maths at all."""
    at, restore = _offline_app()
    try:
        def anchor():
            return [w for w in at.selectbox if w.label.startswith("Anchor")][0]
        default = anchor().value
        pick = [o for o in anchor().options if o != default][-1]
        anchor().set_value(pick).run()
        assert anchor().value == pick
        [r for r in at.radio if r.label == "Weighting"][0].set_value("Risk parity").run()
        assert anchor().value == pick, (
            f"anchor became {anchor().value} after a Weighting change; visitor chose {pick}")
        assert not at.exception, [e.value for e in at.exception]
    finally:
        restore()


def test_metric_labels_and_values_wrap_instead_of_ending_in_an_ellipsis():
    """Streamlit cuts a metric's label and value to one line with an ellipsis.
    In the narrow metric columns of a phone or a portrait iPad that left
    "IR at effective breadth (159 bets/yr)", "Historical VaR (95%)" and
    "10 / 13 exp." unreadable (tools/device_check.mjs, 2026-10-17: every phone
    and portrait iPad). They must wrap instead."""
    import re
    css = (pathlib.Path(__file__).resolve().parent.parent / "static" / "app.css"
           ).read_text(encoding="utf-8")
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", css)
    for part in ("stMetricLabel", "stMetricValue"):
        body = " ".join(b for sel, b in rules if part in sel)
        assert re.search(r"white-space:\s*normal", body), f"{part} still kept to one line"
        assert re.search(r"text-overflow:\s*clip", body), f"{part} still ends in an ellipsis"


def _tab_strip_slot(at) -> int:
    """Top-level position of the first st.tabs block (the Risk Topology strip)."""
    return next(i for i, c in at.main.children.items()
                if getattr(c, "type", None) == "tab_container")


def test_the_tab_strip_keeps_its_slot_whatever_the_basket():
    """John, 2026-10-19: switching to MAG-7 showed the Risk Topology map twice.
    Streamlit keys every element by its position; The Book wrote one top-level
    card per name (17 on the default basket, 13 on MAG-7), so the tab strip
    moved from slot 44 to 40 and the old strip, map and all, stayed on screen
    faded until the run ended (~12 s offline, up to ~35 s live). The outside-
    anchor toggle moved it too and reloaded the map for nothing. Nothing above
    the tabs may change the number of top-level elements."""
    at, restore = _offline_app()
    try:
        slots = {"default": _tab_strip_slot(at)}
        at.selectbox[[w.label for w in at.selectbox].index("Preset basket")].set_value(
            "Mega-cap tech (Mag 7 + QQQ)").run()
        assert not at.exception, [e.value for e in at.exception]
        slots["MAG-7"] = _tab_strip_slot(at)
        at.toggle(key="book_fallback").set_value(False).run()
        slots["no outside anchor"] = _tab_strip_slot(at)
        assert len(set(slots.values())) == 1, f"tab strip moved: {slots}"
    finally:
        restore()


def _verdict_html(at) -> str:
    return next(m.value for m in at.markdown if 'class="verdict-box"' in m.value)


def test_the_headline_accents_follow_real_state():
    """ME-11 (2026-10-22), John: "make the website feel alive ... add visual
    details and accent marks that are meaningful". Each accent is a state the
    page already computed, drawn where the reader looks: a 0-100% hairline
    under the CVaR with a tick at the value and a faint tick at this session's
    last different headline, "+N points vs your last setting" in words; a
    "Stressed" mark while a shock or crisis replay is applied; a "Short book"
    mark on the bearish side. All inside the verdict's own markdown, so the
    tab strip keeps its slot."""
    import re
    from src.scenarios import HISTORICAL_REGIMES
    at, restore = _offline_app()
    try:
        slot = _tab_strip_slot(at)
        box = _verdict_html(at)
        shown = lambda b: float(re.search(r'class="verdict-number">(-?[\d.]+)%', b).group(1))
        v0 = shown(box)
        tick = re.search(r'class="cvar-tick now" style="left:([\d.]+)%', box)
        assert tick and abs(float(tick.group(1)) - min(max(v0, 0), 100)) < 0.11, box
        assert "vs your last setting" not in box and "cvar-tick prev" not in box, \
            "a first headline has nothing to compare against"
        assert "state-mark" not in box, "no stress, no short: no mark"

        at.radio[[w.label for w in at.radio].index("Weighting")].set_value("Risk parity").run()
        assert not at.exception, [e.value for e in at.exception]
        box = _verdict_html(at)
        v1 = shown(box)
        assert v1 != v0, "fixture: risk parity should move the headline"
        said = re.search(r"([+\u2212-][\d.]+) points vs your last setting", box)
        assert said, box
        delta = float(said.group(1).replace("\u2212", "-"))
        assert abs(delta - (v1 - v0)) <= 0.11, (delta, v0, v1)
        prev = re.search(r'class="cvar-tick prev" style="left:([\d.]+)%', box)
        assert prev and abs(float(prev.group(1)) - min(max(v0, 0), 100)) < 0.11, box

        at.selectbox[[w.label for w in at.selectbox].index("Scenario")].set_value(
            list(HISTORICAL_REGIMES)[-1]).run()
        assert not at.exception, [e.value for e in at.exception]
        assert ">Stressed<" in _verdict_html(at), "a crisis replay must mark the headline"
        assert _tab_strip_slot(at) == slot

        at.radio(key="bv_direction").set_value("Bearish - short the book").run()
        assert not at.exception, [e.value for e in at.exception]
        assert ">Short book<" in _verdict_html(at), "the bearish side must mark the headline"
        assert _tab_strip_slot(at) == slot, "an accent moved the tab strip"
    finally:
        restore()


def _main_function(name):
    """One top-level function of main.py, compiled on its own (main.py is a
    script: importing it would run the app)."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), scope)
    return scope[name]


def test_the_headline_move_is_read_from_the_numbers_as_displayed():
    """Council B (2026-10-22): 22.540% then 22.480% both display "22.5%", and
    the page said "-0.1 points vs your last setting" between two equal
    numbers. The move is the difference of the displayed headlines; equal
    displays say nothing. And a CVaR past 100% no longer clamps silently at
    "the whole book": the scale's end says the loss is beyond it."""
    scale = _main_function("cvar_scale_html")
    same = scale(0.22540, 0.22480)
    assert f"{0.22540:.1%}" == f"{0.22480:.1%}"
    assert "vs your last setting" not in same, same
    moved = scale(0.2266, 0.2254)                 # 22.7% after 22.5%
    assert "+0.2 points vs your last setting" in moved, moved
    down = scale(0.2049, 0.2254)                  # 20.5% after 22.5%
    assert "\u22122.0 points vs your last setting" in down, down
    assert "vs your last setting" not in scale(0.2254, None), "a first headline has nothing to compare"
    over = scale(1.374, 0.9)
    assert "beyond the whole book" in over and "cvar-tick now beyond" in over, over
    assert 'left:100.0%' in over, "a loss past the whole book sits at the scale's end"
    assert "beyond" not in scale(0.999, None), "100% or less is on the scale"


def test_the_freshness_dot_and_the_reading_line_are_honest_and_still():
    """ME-11: the freshness stamp gets a dot that pulses once per minute tick
    only while the data is live (end-of-day, at most a day old) and sits grey
    when stale; a 1px reading-progress line rides the page's own scroll
    (CSS scroll-timeline, no JS listener). Both are decorative beside words
    that already say the state, and both are still under reduced motion."""
    import re as _re
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    tick = src[src.index("def _freshness_ticker"):]
    tick = tick[:tick.index("\n\n\n")]
    assert "live" in tick.split(")")[0], "the ticker must be told whether the data is live"
    assert "fresh-dot is-live" in tick and "fresh-dot is-stale" in tick, tick
    assert _re.search(r"_freshness_ticker\([^)]*staleness_days", src) or "live=fresh" in src.replace(" ", ""), \
        "the dot's state must come from the measured staleness"
    live = _re.search(r"\.fresh-dot\.is-live\s*\{([^}]*)\}", css)
    assert live and _re.search(r"animation:[^;]*\b1\b", live.group(1)), "a live dot pulses once per tick"
    assert "@keyframes" in css
    assert _re.search(r'\[data-testid="stMain"\]\s*\{[^}]*scroll-timeline:\s*--page', css), \
        "the reading line must ride the real scroller's own timeline"
    line = _re.search(r'\[data-testid="stMain"\]::before\s*\{([^}]*)\}', css)
    assert line and "animation-timeline: --page" in line.group(1), "no JS: a scroll-driven animation"
    assert "scaleX(0)" in line.group(1), "with animations off (reduced motion) the line stays still"
    sup = css.index("@supports (animation-timeline: scroll())")
    assert sup < css.index('[data-testid="stMain"]::before {') < css.index("}\n}", sup) + 3, \
        "the reading line must sit inside an @supports guard (no full-width bar where timelines are missing)"
    assert _re.search(r"\.fresh-dot\.is-stale\s*\{[^}]*border:[^}]*#7E776B", css), \
        "stale is a hollow ring, not only a missing pulse"
    assert _contrast("#7E776B", "#EDE9E3") >= 3 and _contrast("#9A7B4F", "#EDE9E3") >= 3
    rm = _re.search(r"@media \(prefers-reduced-motion: reduce\) \{\s*\*, \*::before, \*::after \{ animation: none", css)
    assert rm, "the global reduced-motion rule must still stop every animation"


def test_the_glide_is_bound_before_any_stop_and_rebinds():
    """John, 2026-10-19: "sometimes the button does not transition". The
    glide listener came from a zero-height iframe at the END of main.py,
    behind a bind-once guard. A run that ended in st.stop() never sent the
    iframe, Streamlit removed it, its listener died, and the guard kept the
    next iframe from binding: every button jumped until a reload (15/15
    measured). On every load the buttons were live but unwired for the
    2.7-4.6 s before the script end. And a clock-driven step jumped 66-81% of
    the way in one frame when the page was busy."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    def line_of(pred):
        return next(n.lineno for n in top if pred(n.value))
    glide = line_of(lambda c: ast.unparse(c.func) == "_script_frame" and c.args
                    and isinstance(c.args[0], ast.Constant) and "__meleonaGlide" in c.args[0].value)
    css = line_of(lambda c: ast.unparse(c.func) == "load_css")
    hero = line_of(lambda c: ast.unparse(c.func) == "st.markdown" and "hero_html" in ast.unparse(c))
    stops = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "st.stop" and n.lineno > css]
    assert css < glide < hero and glide < min(stops), (
        f"glide at line {glide}: must follow load_css ({css}) and precede the hero "
        f"({hero}) and every st.stop() after it (first {min(stops)})")
    script = next(c.value.args[0].value for c in top if c.value.args
                  and isinstance(c.value.args[0], ast.Constant)
                  and "__meleonaGlide" in c.value.args[0].value)
    assert "__meleonaGlide) return" not in script, "a bind-once guard blocks rebinding again"
    assert "removeEventListener" in script, "a new glide frame must replace the old listeners"
    assert "(now - t0) / dur" not in script, "a busy frame can jump most of the flight again"


def test_the_glide_respects_reduced_motion():
    """ACCESSIBILITY.md promises every animation honours prefers-reduced-motion,
    but the buttons' 1.5 s glide is driven by JavaScript, which no CSS media
    query can stop, and the smooth-scroll fallback lived outside every
    reduced-motion block. With reduced motion a button now lands at once -
    same place, no flight, and still no hash jump."""
    import ast
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    script = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_script_frame" and n.args
                  and isinstance(n.args[0], ast.Constant) and "__meleonaGlide" in n.args[0].value)
    # go() is what every press ends in - at once, or once a section still
    # being written has mounted (onLink / waitFor) - so the question lives there.
    on_link = script[script.index("function go("):script.index("P.addEventListener")]
    assert "go(a, el)" in on_link[on_link.index("function onLink"):], "every press must end in go()"
    assert "prefers-reduced-motion: reduce" in on_link, \
        "the glide must ask for the visitor's motion setting on every click"
    still = on_link[on_link.index("prefers-reduced-motion"):on_link.index("glide(scroller, el)")]
    assert "preventDefault" in on_link and "scrollTo" in still and "return" in still, \
        "with reduced motion the button must land at once, before any glide starts"
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    blocks = re.findall(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", css, re.S)
    assert any(re.search(r"scroll-behavior:\s*auto\s*!important", b) for b in blocks), \
        "the CSS smooth-scroll fallback must stand down under reduced motion too"


def test_every_button_wears_one_focus_ring_that_clears_3_to_1_on_its_ground():
    """ME-04 (2026-10-22): the CTAs wore a square 2px bronze outline, while
    the skip link and the footer boxes fell back to Streamlit's rounded
    box-shadow ring. The bronze outline measured 2.82:1 on the dark band
    ("See the crisis record"), under the 3:1 a focus indicator needs (WCAG 2.2
    1.4.11). One rule now draws every in-page link's ring, in a tone chosen by
    the ground: the dark text bronze on beige, the light one on the charcoal
    bands. tools/cta_style_check.mjs measures the same thing from pixels."""
    import re as _re
    root = pathlib.Path(__file__).resolve().parent.parent
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    rule = next((m for m in _re.finditer(r"([^{}]*:focus-visible[^{}]*)\{([^}]*)\}", css)
                 if "--ring" in m.group(2)), None)
    assert rule, "no shared focus rule drawing var(--ring)"
    sel, body = rule.group(1), rule.group(2)
    for who in ("a.cta-btn", "a.cta-link", ".f-box"):
        assert who in sel, f"{who} does not share the focus ring"
    assert _re.search(r"outline:\s*2px solid var\(--ring\)", body), body
    assert _re.search(r"box-shadow:\s*none", body), "Streamlit's box-shadow ring must stand down"
    rings = dict((s.strip(), c) for s, c in
                 _re.findall(r"([^{}]+)\{[^}]*--ring:\s*(#[0-9A-Fa-f]{6})", css))
    light = next(c for s, c in rings.items() if ":root" in s)
    dark = next(c for s, c in rings.items() if ".meleona-footer" in s and ".showcase-row" in s)
    for ground in ("#EDE9E3", "#E9E4DB", "#D4CDBF"):
        assert _contrast(light, ground) >= 3, f"ring {light} on {ground}: {_contrast(light, ground):.2f}:1"
    assert _contrast(dark, "#3F3B35") >= 3, f"ring {dark} on the band: {_contrast(dark, '#3F3B35'):.2f}:1"
    old = _re.search(r"a\.cta-btn:focus-visible,\s*\.stButton>button:focus-visible,\s*"
                     r"\[data-baseweb=\"tab\"\]:focus-visible\s*\{([^}]*)\}", css)
    assert not old or "#9A7B4F" not in old.group(1), "the 2.82:1 bronze outline is still drawn on CTAs"


def test_the_outlier_links_and_marks_now_match_their_siblings():
    """ME-04 (2026-10-22), each one a measured outlier: the skip link was
    styled inline (no hover, an arrow that never drew, no launch), footer 01
    and 02 landed on the same spot on a desktop (the two halves of one row),
    the skip link promised "the risk map" while landing on whichever tab was
    open, the Data Quality tab was the only bright colour on the page (colour
    emoji), and Streamlit's sans-serif link icon sat on the serif headings."""
    import re as _re
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    skip = _re.search(r"<a href=\"#analysis\"[^>]*>", src).group(0)
    assert 'class="cta-link"' in skip and "style=" not in skip, skip
    assert "risk map" not in src[src.index('href="#analysis"'):][:400].lower(), \
        "the skip link lands on the analysis tabs, whichever is open - it must not promise the map"
    for state in (".cta-link:hover", ".cta-link:hover .cta-arrow .shaft", ".cta-link.is-launching .cta-arrow"):
        assert state in css, f"the skip link has no {state} rule"
    assert "a.querySelector('.cta-arrow')" in src, "every link with an arrow launches it"
    footer = src[src.index('<div class="meleona-footer">'):]
    targets = _re.findall(r'class="f-box" href="#([\w-]+)"', footer)
    assert len(set(targets)) == 3, targets
    assert not {"grit-showcase", "conviction"} <= set(targets), \
        "#grit-showcase and #conviction share one row: two footer boxes landed on one spot"
    for emoji in ("✅", "⚠️", "❌"):
        assert emoji not in src, f"colour emoji {emoji} on the page"
    # Council B: each mark says its meaning to a screen reader, and none sits
    # in a 60%-opacity caption
    for word in ("Pass", "Warning", "Fail"):
        assert f'role="img" aria-label="{word}"' in src, f"the {word} mark has no spoken name"
    assert not _re.search(r"st\.caption\([^)]*dq-mark", src, _re.S), "a mark is still inside an st.caption"
    # footer 02 and "See the crisis record" land even when the record fails to load
    assert src.index('<div id="crisis-record"') < src.index("_conv = load_conviction_data()"), \
        "#crisis-record must be written before (and whatever happens to) the crisis data"
    assert _re.search(r'\[data-testid="stHeaderActionElements"\]\s*\{[^}]*visibility:\s*hidden', css), \
        "Streamlit's sans-serif heading link icon still shows on the serif headings"
    assert _re.search(r"\.conv-core\s*>\s*\.cta-btn\s*\{[^}]*align-self:\s*flex-start", css), \
        "the crisis-record button still stretches across its column"
    # one effective rule set: every .cta-btn rule that only ever lost to a
    # later !important layer is gone
    assert len(_re.findall(r"(?m)^\.cta-btn \{", css)) == 1, "dead .cta-btn layers remain"
    assert len(_re.findall(r"(?m)^\.cta-btn:hover \{", css)) == 1, "dead .cta-btn:hover layers remain"
    assert ".cta-btn::before" not in css.replace("#boot-skel .cta-btn::before", ""), \
        "the gloss sheen layer is dead (display:none !important) and should go"


def _top_markdown(at) -> list[tuple[int, str]]:
    """(top-level index, body) of every top-level markdown element, in order."""
    return [(i, c.value) for i, c in at.main.children.items()
            if getattr(c, "type", None) == "markdown"]


def test_the_skeleton_paints_first_and_the_hero_lifts_it():
    """John, 2026-10-19: "When opening the website, I was not met with a
    skeleton loader". The skeleton was sent by main.py AFTER ~5 MB of base64
    photos on the same websocket, so it arrived with the hero it was meant to
    stand in for (phone at 9 Mbps: 8 s of blank page first); it lifted on a
    fixed 2.1 s timer; reduced motion hid it. And nothing main.py sends can
    arrive before Streamlit's JavaScript has started - 2.5 s (laptop) to 4.4 s
    (phone) of plain screen. Now index.html paints the hero's own skeleton
    from the first frame, the photos are files, the hero lifts the skeleton
    when it arrives, and a reduced-motion visitor sees it held still."""
    import re
    import tempfile
    import tools.public_files as pf
    from hero import boot_skeleton_html
    root = pathlib.Path(__file__).resolve().parent.parent
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "index.html").write_text(
            "<!doctype html><html><head><title>Streamlit</title>"
            '<link href="./static/css/index.css" rel="stylesheet"></head>'
            "<body><noscript>You need JS.</noscript><div id='root'></div></body></html>",
            encoding="utf-8")
        pf.install(d, "https://example.test")
        page = (d / "index.html").read_text(encoding="utf-8")
    head, body = page.split("<body>", 1)
    assert '<link rel="stylesheet" href="./app/static/app.css">' in head, \
        "the skeleton's stylesheet is not in place for the first paint"
    assert body.index(boot_skeleton_html()) < body.index("<div id='root'>"), \
        "index.html does not paint the hero's skeleton before the app"
    at, restore = _offline_app()
    try:
        first = _top_markdown(at)
        assert not any('id="boot-skel"' in v for _, v in first), \
            "a second skeleton is sent from the app"
        hero = next(i for i, v in first if 'id="hero"' in v)
        assert not any("data:image/jpeg;base64" in v for _, v in first), \
            "a photo is inlined into the websocket again"
        assert any("url(app/static/facade.jpg)" in v for _, v in first)
        assert any("url(app/static/band.jpg)" in v for _, v in first)
        assert re.search(r"#boot-skel\s*\{[^}]*opacity:\s*0", dict(first)[hero]), \
            "the hero does not lift the skeleton when it arrives"
        # The full-bleed hero swallows the 1rem flex gap of each block above it
        # plus one 16px overshoot the skeleton copies (its own -16px).
        n = int(re.search(r"calc\(-2\.4rem - (\d+)px\)", css).group(1))
        assert n == 16 * (hero + 1), f"hero margin {n}px for {hero} blocks above it"
        at.run()
        assert next(i for i, v in _top_markdown(at) if 'id="hero"' in v) == hero, \
            "the hero moved on a rerun"
    finally:
        restore()
    assert not re.search(r"#boot-skel\s*\{[^}]*display:\s*none", css), \
        "the skeleton is hidden somewhere (reduced motion?)"
    delay = re.search(r"#boot-skel\s*\{[^}]*animation:\s*boot-off[^;]*?(\d+)s forwards", css)
    assert delay and int(delay.group(1)) >= 20, "the skeleton lifts on a timer again"
    import tomllib
    cfg = tomllib.loads((root / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["server"].get("enableStaticServing") is True
    served = sorted(p.name for p in (root / "static").iterdir())
    assert served == ["app.css", "band.jpg", "facade.jpg"], \
        f"static/ is served to the public; keep it to public files: {served}"


def test_a_running_script_is_visible_to_the_visitor():
    """John, 2026-10-19: "the loading wheel has no animation or no clear
    noticeable visual that it is loading". Streamlit's running indicator lives
    in the header, which the page hides (it covered the crest); the spinners
    are 16 px rings, mostly off screen or inside closed drawers, and frozen
    under reduced motion. While a run is in flight the page must show its own
    sign: fixed, never in the way of a click, and still there - just still -
    when motion is reduced."""
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    on = r'body:has\(\[data-testid="stStatusWidgetRunningIcon"\]\)'
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", css)
    rail = " ".join(b for sel, b in rules if re.search(on + r"::before", sel))
    tag = " ".join(b for sel, b in rules if re.search(on + r"::after", sel))
    for name, body in (("rail", rail), ("tag", tag)):
        assert body, f"no running {name} in app.css"
        assert re.search(r"position:\s*fixed", body), f"running {name} is not fixed"
        assert re.search(r"pointer-events:\s*none", body), f"running {name} can block clicks"
    assert re.search(r'content:\s*"Updating', tag), "the running tag says nothing"
    assert re.search(r"animation:\s*run-rail", rail), "the rail does not move"
    reduce = " ".join(re.findall(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\}\s*\}", css, re.S))
    assert not re.search(on + r"::after[^{]*\{[^}]*(display:\s*none|visibility:\s*hidden)", reduce), (
        "the running tag is hidden when motion is reduced")
    # The page keys off Streamlit's own marker; an upgrade that renames it must
    # fail here, not silently drop the sign (as test_streamlit_header_bar_is_hidden).
    import streamlit
    js = pathlib.Path(streamlit.__file__).parent / "static" / "static" / "js"
    assert any("stStatusWidgetRunningIcon" in p.read_text(encoding="utf-8", errors="ignore")
               for p in js.glob("index.*.js")), "Streamlit no longer renders stStatusWidgetRunningIcon"


def test_every_cached_yahoo_pull_says_what_it_is_doing():
    """John, 2026-10-19: changing basket, the Grit Zone "looked stalled, as if
    the project crashed". Grit maths takes 0.2 s; the wait is a new basket's
    first full-history pulls from Yahoo, and four of them showed nothing (one
    showed Streamlit's developer text, "Running load_risk_free_rate()"). Every
    cached function that reaches the feed must say, in words, what it is
    doing while it waits."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    feed = {"fetch_prices", "fetch_dollar_volume", "average_dollar_volume",
            "fetch_risk_free_rate", "grit_scores", "crisis_cushion", "replay_returns",
            "anchor_frame", "build_map_payload", "load_conviction", "security_master"}
    silent = []
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        deco = [d for d in fn.decorator_list if isinstance(d, ast.Call)
                and ast.unparse(d.func) == "st.cache_data"]
        calls = {ast.unparse(c.func).split(".")[-1] for c in ast.walk(fn)
                 if isinstance(c, ast.Call)}
        if not deco or not (calls & feed):
            continue
        spin = {k.arg: k.value for k in deco[0].keywords}.get("show_spinner")
        words = (isinstance(spin, ast.Constant) and isinstance(spin.value, str)
                 and spin.value.strip()) or isinstance(spin, ast.JoinedStr)
        if not words:
            silent.append(fn.name)
    assert not silent, f"cached Yahoo pulls with no words on screen: {silent}"


def test_the_glide_follows_a_moving_target_and_yields_to_the_reader():
    """John, 2026-10-19: the "Explore what we do" glide is "very laggy and
    buggy". Measured on the real page:
      - it aimed at the showcase row once, while the row was still rising 24px
        on its scroll-driven reveal, so every press overshot 8-11px, held,
        then crawled back in a second correction glide;
      - a wheel during a flight was pulled back each frame (8 reversals), and
        a second press flew two flights at once (a 59px jump, a wrong landing,
        smooth scrolling left switched off);
      - every frame's scroll re-sent the Risk Topology map's five canvases,
        4300px off screen - 42% of the main thread at 4x CPU, 20 fps.
    The glide now re-reads its target each frame (capped per frame), flies one
    flight at a time, yields to the wheel / touch / scroll keys, and skips the
    off-screen map's rendering only while it flies."""
    import ast
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    script = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_script_frame" and n.args
                  and isinstance(n.args[0], ast.Constant) and "__meleonaGlide" in n.args[0].value)
    frame = script[script.index("(function f(now)"):script.index("function onLink")]
    assert "targetY(scroller, el)" in frame and "Math.min(2," in frame, \
        "the glide must follow the target each frame, a bounded step at a time"
    assert "__meleonaFlight" in frame, "each frame must check it is still the current flight"
    for ev in ("wheel", "touchstart", "keydown"):
        assert f"'{ev}'" in script, f"the glide must yield to the reader's {ev}"
    rebind = script[:script.index("function onTab")]
    assert "onYield" in rebind, "a new glide frame must remove the old yield listeners"
    assert "scrollBehavior = ''" in script, "landing must hand scrolling back to the stylesheet"
    assert "contentVisibility" in script and ".st-key-risk-map" in script, \
        "the off-screen map must be skipped while the glide flies"
    assert 'st.container(key="risk-map")' in src, "the map needs its own keyed container"


def test_a_press_waits_for_its_section_as_long_as_the_page_is_still_being_written():
    """ME-05, third review (Live audit run 39737935470, real Yahoo, cold start):
    "Skip to the analysis" was DEAD. #analysis is written only after the whole
    engine has run, and the press stopped waiting after a fixed 10 s; offline
    with 4 s pulls the section came 12.3-15.7 s after the press and 0 of 2
    presses landed (tools/glide_load_check.mjs early, which now waits for the
    section however long it takes). The wait now lasts while Streamlit's
    running sign is up, ends once the run is over without the section, and a
    scroll the reader makes ends it too."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    script = next(n.args[0].value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_script_frame" and n.args
                  and isinstance(n.args[0], ast.Constant) and "__meleonaGlide" in n.args[0].value)
    wait = script[script.index("function waitFor"):script.index("function onLink")]
    assert "pageRunning()" in wait and '[data-testid="stStatusWidgetRunningIcon"]' in script, \
        "the wait must last while the page is still being written"
    assert "now - t0 < 10000 || idleSince === null" in wait, \
        "10 s is a floor, not a cap: the wait goes on while the run goes on"
    assert "now - idleSince < 2000" in wait and "P.__meleonaStop()" in wait, \
        "once the run has ended without the section, the wait ends (no leak)"
    assert "requestAnimationFrame(poll)" not in wait, "a long wait polls gently, not every frame"
    assert "P.__meleonaScroller = scroller" in wait and "P.__meleonaSetY = scroller.scrollTop" in wait, \
        "a scroll the reader makes while the press waits must end the wait"


def test_only_a_decaying_flick_is_forgiven_and_every_exit_restores_the_page():
    """Council B's review of ME-05 (2026-10-22), each probe reproduced:
      - steady: a reader wheeling down at a steady 100px a notch pressed footer
        01 and was dragged UP 2.7k px - "no bigger" let a constant wheel pass
        as a trackpad's decaying tail (glide_load_check steady: 0/16 before);
      - leak: a newer press bumped the flight number without landing the old
        flight, leaving the scroller on 'auto' and the map's rendering skipped;
      - a scrollbar drag (no wheel, touch or key) was pulled back by the
        settle phase on a page still growing.
    tools/glide_load_check.mjs steady/momentum and the Live audit step 8c run
    the behaviour; this pins the rules in the script."""
    import ast
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    script = next(n.args[0].value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_script_frame" and n.args
                  and isinstance(n.args[0], ast.Constant) and "__meleonaGlide" in n.args[0].value)
    assert "function onScroll" in script, "a scrollbar drag (scroll events only) never reaches the yield"
    yield_fn = script[script.index("function onYield"):script.index("function onScroll")]
    # Council B's second review: real tails repeat integer deltas (2, 2, 1, 1),
    # so "strictly smaller" cancelled them all; a steady wheel never shrinks.
    assert "Math.abs(e.deltaY) < Math.abs(wheelDy)) shrunk = true" in yield_fn, \
        "the stream must be seen to shrink before any of it is forgiven"
    assert "noBigger && shrunk" in yield_fn, "a tail is forgiven only once it has shrunk"
    assert "* 1.25" not in yield_fn, "a harder wheel must yield"
    link = script[script.index("function onLink"):]
    assert "shrunk = false" in link, "every press starts a tail that has not shrunk yet"
    assert "Math.sign(e.deltaY) === P.__meleonaDir" in yield_fn, "a wheel against the flight must yield"
    assert "P.__meleonaDir = dir" in script, "the flight must publish its direction"
    for fn in ("function glide", "function waitFor"):
        body = script[script.index(fn):]
        assert body.index("supersede()") < body.index("__meleonaFlight = "), \
            f"{fn} must land the running flight before it takes over"
    assert "supersede();" in script[script.index("prefers-reduced-motion"):script.index("glide(scroller, el);")]
    land = script[script.index("function land"):script.index("function supersede")]
    for undo in ("scrollBehavior = ''", "overflowAnchor = ''", "quietMap(false)", "__meleonaStop = null"):
        assert undo in land, f"landing must undo {undo}"
    assert "P.addEventListener('scroll', onScroll" in script, "a scrollbar drag must reach the yield"
    assert "prev.onScroll" in script[:script.index("function onTab")], "a new frame removes the old scroll listener"
    assert "__meleonaSetY = scroller.scrollTop" in script, "the flight records where it put the page"
    wf = (root / ".github" / "workflows" / "live-audit.yml").read_text(encoding="utf-8")
    assert "glide_load_check.mjs http://127.0.0.1:8501 steady" in wf and "early 1" in wf, \
        "the Live audit must run the glide check, steady and cold mid-load presses included"


def test_a_button_pressed_while_the_page_is_still_loading_still_glides_all_the_way():
    """John, 2026-10-22: "sometimes it does not do the transition". Measured on
    the offline harness at cfceb42 (ME-04), every button glided on a settled
    page (480/480) but not while the page was still loading:
      - "See the crisis record" and "Skip to the risk map" were dead 8/8: their
        target is written only after the data arrives, the click found no
        element and fell through to the browser's hash jump, which scrolled
        nowhere and never followed the section once it appeared;
      - "To the engine" leapt 138-230px in one frame 6/8: the eased position
        ran past the bottom of the still-growing page, scrollTo clamped it,
        and the next mount released it all at once;
      - "Explore what we do" stopped 223px short 8/8 at the bottom of a short
        page, and a trackpad's momentum still arriving after the click
        cancelled every flight 30/30 (only 300 ms of the wheel was forgiven).
    A press with no target yet waits for it (bounded) and then flies; each
    frame re-aims against the page's current bottom and moves at most a
    peak-speed step, so a page that grows mid-flight is caught up smoothly;
    the flight keeps settling until it is on the target; and a wheel only
    yields when it is not the decaying tail of the gesture running at the click."""
    import ast
    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    script = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_script_frame" and n.args
                  and isinstance(n.args[0], ast.Constant) and "__meleonaGlide" in n.args[0].value)
    link = script[script.index("function onLink"):]
    assert "if (!el) return;" not in link, \
        "a press before its section exists must not fall through to the browser's hash jump"
    assert "e.preventDefault()" in link and link.index("e.preventDefault()") < link.index("getElementById"), \
        "the press is kept (default prevented) before looking for its target"
    assert "waitFor(" in link and "10000" in script, "a missing target is waited for, and only for a bounded time"
    frame = script[script.index("(function f(now)"):script.index("function onLink")]
    assert "scrollHeight - scroller.clientHeight" in frame, \
        "each frame must re-aim against the page's current bottom (a growing page clamps scrollTo)"
    assert "vmax" in frame, "each frame's step is capped at the flight's peak speed - no catch-up leap"
    assert "glide(scroller, el, 320, true)" not in script, \
        "one 320 ms correction is not enough on a growing page: the flight settles until it is on target"
    assert "deltaY" in script and "< 300) return" not in script, \
        "a wheel yields unless it is the decaying tail of the gesture running at the click"


def test_every_short_form_on_the_map_is_in_its_glossary():
    """John, 2026-10-19: "pressing the 'explain the map' button does not work
    or direct me anywhere, its a dead button" and "I don't know what OU is ...
    add a legend in the topology map that explains all the suffixes and
    shortened phrases." The button only switched on a hover mode over the plot
    and tinted itself, so nothing appeared on a click. It now opens a glossary
    right under itself that defines every short form the map prints, and the
    tiles and chips speak plain words (no "OU" subtitle; "beta"/"vol", not the
    Greek letters the chips' uppercase turned into "B"/"S"-like capitals)."""
    import re
    html = (pathlib.Path(__file__).resolve().parent.parent / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    m = re.search(r'<section id="glossary"[^>]*>(.*?)</section>', html, re.S)
    assert m, "the map has no glossary"
    gloss = re.sub(r"<[^>]+>", " ", m.group(1)).lower()
    terms = ["expected shortfall", "var 95", "t+30", "ornstein-uhlenbeck", "ar(1)", "long-run vol",
             "calm", "baseline", "stressed", "beta", "vol", "daily closes", "days fitted",
             "perimeter", "breach", "rings", "shock correlation", "leverage", "shocks widened",
             "drawdown cushion", "flyer", "anchor"]
    missing = [t for t in terms if t not in gloss]
    assert not missing, f"short forms the glossary does not define: {missing}"
    btn = re.search(r'<button[^>]*id="btnExplain"[^>]*>', html).group(0)
    assert 'aria-controls="glossary"' in btn and "aria-expanded" in btn, \
        "the button must say it opens the glossary"
    assert "OU mean-reversion level" not in html, "the tile still explains vol with 'OU'"
    assert "S.cal.toUpperCase()" not in html, "the engine chip still prints BASE/STRESS codes"
    assert "'Base · β '" not in html, "the chip still prints Greek letters under uppercase"


def test_the_pairing_never_shows_pp_or_percent_for_a_points_gap():
    """John, 2026-10-19: "find a better way to show status than 'pp', I have
    no idea what that is." The cushion is the pair's worst fall minus the
    flyer's worst fall alone (pairing.py: dd_pair - dd_solo), a gap in
    percentage points. The map printed "+16.8pp", and the Balance sentence and
    crisis table printed the same gap as "+17%", which reads as a 17% cut.
    Both now say it in words: "16.8 points shallower" / "deeper"."""
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    html = (root / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    assert not re.findall(r"[^A-Za-z]pp[^A-Za-z]", html), "the map still prints a 'pp' unit"
    assert "shallower" in html and "deeper" in html, \
        "the map must say which way the cushion points, in words"
    src = (root / "main.py").read_text(encoding="utf-8")
    assert 'bt["cushion"]:+.0%' not in src, "the Balance sentence prints a points gap as %"
    assert '"cushion": "{:+.1%}"' not in src, "the crisis table prints a points gap as %"
    assert "percentage points" in src, "the Balance sentence must name its unit"


def test_every_spinner_is_a_few_calm_words():
    """John, 2026-10-19: "Remove the 'yahoo is the slow part', I cringe
    looking at that." A loading line says what the page is doing, in a few
    words - it does not apologise, excuse the wait or blame the data feed."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    texts = []
    for d in ast.walk(ast.parse(src)):
        if isinstance(d, ast.Call) and ast.unparse(d.func) in ("st.cache_data", "st.spinner"):
            spin = ({k.arg: k.value for k in d.keywords}.get("show_spinner")
                    if ast.unparse(d.func) == "st.cache_data" else (d.args or [None])[0])
            if isinstance(spin, ast.Constant) and isinstance(spin.value, str):
                texts.append(spin.value)
            elif isinstance(spin, ast.JoinedStr):
                texts.append(ast.unparse(spin))
    assert texts, "no spinner text found - the scan is broken"
    bad = [t for t in texts if len(t) > 60
           or any(w in t.lower() for w in ("slow", "yahoo", "sorry", "please wait", "may take"))]
    assert not bad, f"spinner lines that excuse the wait or run long: {bad}"


def test_section_titles_wrap_instead_of_running_out_of_their_box():
    """Live audit run 30 on 77dcccd (device_check, iPhone SE): "VALUE AT RISK -
    METHODS & BACKTEST" needed 304px of a 301px box and was cut off. The
    .panel-head title was nowrap; measured at 375px, 5 of 25 titles were wider
    than their box ("Distribution of simulated 1-year outcomes": 374px of 263).
    The lintel is flex-wrap, so a title that fits renders the same either way;
    one that does not must wrap, never run out of its box."""
    import re
    css = (pathlib.Path(__file__).resolve().parent.parent / "static" / "app.css"
           ).read_text(encoding="utf-8")
    rules = re.findall(r"([^{}]+)\{([^}]*)\}", css)
    body = " ".join(b for sel, b in rules if re.search(r"\.panel-head\s+\.t\b", sel))
    assert body, "the .panel-head title rule moved"
    assert not re.search(r"white-space:\s*nowrap", body), "section titles still kept to one line"


def test_every_browser_pick_checks_the_option_took():
    """Live audit run 30, both attempts: recruiter_walk's pick() clicked an
    option, pressed Escape and returned, but the choice had not landed - the
    box snapped back to the old value on Escape, the page never reran, and
    the step waited 90s for a change that could not come (the COVID pick on
    attempt 1; the Semiconductors pick on attempt 2, on the PR's app and on
    the deployed main alike). A pick must read the box back and retry."""
    import re
    root = pathlib.Path(__file__).resolve().parent.parent / "tools"
    picks = 0
    for js in sorted(root.glob("*.mjs")):
        src = js.read_text(encoding="utf-8")
        m = re.search(r"async function pick\(label, option\) \{\n(.*?)\n\}\n", src, re.S)
        if not m:
            continue
        picks += 1
        assert re.search(r"inputValue\(\)[^\n]*!==\s*option", m.group(1)), (
            f"{js.name}: pick() returns without checking the box shows the option")
    assert picks >= 2, f"expected the walk's and the deep scan's pick(), found {picks}"


# --- ME-07 (2026-10-22): every sentence follows its numbers ---

def _page_text(at) -> list[str]:
    """Every sentence a visitor can read on one rendered run, tags stripped."""
    import re
    out = [m.value for m in at.markdown] + [c.value for c in at.caption]
    out += [w.value for w in at.warning] + [i.value for i in at.info]
    out += [f"{m.label} = {m.value} ({m.proto.delta})" for m in at.metric]
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)) for t in out]


_PAGES = {}


def _offline_page(name="default", **boot):
    """One offline boot per distinct setup, read once and shared by the tests
    that only read it (each boot costs ~15 s)."""
    if name not in _PAGES:
        at, restore = _offline_app(**boot)
        try:
            _PAGES[name] = {"text": _page_text(at),
                            "expanders": [e.label for e in at.expander]}
        finally:
            restore()
    return _PAGES[name]


def _crisis_pages():
    """The COVID replay under each engine, from one boot."""
    if "crisis" not in _PAGES:
        at, restore = _offline_app()
        try:
            [s for s in at.selectbox if s.label == "Scenario"][0].set_value(
                "COVID-19 crash (2020)").run()
            boot = _page_text(at)
            [r for r in at.radio if r.label == "Return model"][0].set_value(
                "Jump-diffusion (Merton)").run()
            radio = [r for r in at.radio if r.label == "Return model"][0]
            _PAGES["crisis"] = {"bootstrap": boot, "jump": _page_text(at),
                                "engine_help": radio.help}
        finally:
            restore()
    return _PAGES["crisis"]


def _verdict(text):
    hits = [t for t in text if "1-Year CVaR (95% confidence)" in t]
    assert len(hits) == 1, hits
    return hits[0]


def test_every_history_length_on_the_page_is_read_from_the_loaded_returns():
    """ME-03 (2026-10-22): one name listed 120 trading days ago cut the
    aligned basket to 120 rows - "Data: 120 trading days" - while the
    Methodology line said "resampled from 2 years of daily historical
    returns" and the pair panel "In the loaded 2-year history". Five sentences
    hard-coded the window; each must be read from the loaded returns."""
    import re

    def young(tickers=None, period="2y", align=True, **_):
        df = _offline_market_prices(tickers, period)
        if "V" in df.columns:
            df.loc[df.index[:-120], "V"] = np.nan
        return df.dropna() if align else df

    text = _offline_page("young", market=young)["text"]
    page = " ".join(text)
    rows = int(re.search(r"Data: (\d+) trading days of prices", page).group(1))
    assert rows == 120, f"fixture no longer shortens the basket: {rows} rows"
    for stale in ("2-year", "2 years", "two-year", "two years"):
        assert stale not in page, f"the page still says {stale!r} over {rows} rows"
    method = [t for t in text if t.startswith("Methodology:")]
    assert method and f"{rows - 1} daily returns" in method[0], method


def test_the_crisis_verdict_names_the_engine_that_made_it():
    """ME-03 (2026-10-22): under jump-diffusion the verdict still said
    "Replaying the actual returns of COVID-19 crash (2020) (52 trading days)",
    yet nothing is replayed - a Merton model is calibrated to the window. And
    under the bootstrap the 52 days are resampled into a full year, which the
    sentence did not say."""
    pages = _crisis_pages()
    boot, jump = _verdict(pages["bootstrap"]), _verdict(pages["jump"])
    assert "Replaying the actual returns of COVID-19 crash (2020)" in boot, boot
    assert "resampled into a year" in boot, boot
    assert "Replaying" not in jump and "replay" not in jump.lower(), jump
    assert "calibrated to the actual returns of COVID-19 crash (2020)" in jump, jump
    cap = [t for t in pages["jump"] if "actual market returns from" in t]
    assert cap and not cap[0].startswith("Replaying"), cap
    assert any(t.startswith("Replaying actual market returns from")
               for t in pages["bootstrap"])


def test_the_jump_diffusion_caption_states_its_method_not_a_result():
    """ME-03 (2026-10-22): "a fatter, more honest crash" was printed whatever
    the two engines' CVaRs were - a result stated as a fixed sentence. The
    caption and the engine help describe what the model does, not how its
    answer compares."""
    pages = _crisis_pages()
    jd = [t for t in pages["jump"] if t.startswith("Merton jump-diffusion:")]
    assert jd, "jump-diffusion caption missing"
    for said in (jd[0], pages["engine_help"]):
        for claim in ("fatter", "more honest"):
            assert claim not in said, f"{claim!r} stated as a fact: {said}"


def test_a_difference_of_two_percentages_says_points_and_a_ratio_says_relative():
    """ME-03 (2026-10-22): the defensive shift read "Annualized vol 17.8%
    (-2.9% vs current)" for a fall from 20.7% - 2.9 points, a 14% relative
    cut - and the unwind delta "+0% for the unwind" beside a CVaR in percent
    was a relative change. A difference of two percentages says points; a
    ratio says relative."""
    import re
    text = _offline_page()["text"]
    shift = [t for t in text if t.startswith(("Daily CVaR (95%) =",
                                              "Annualized vol ="))]
    assert len(shift) == 2, f"defensive simulation not on the page: {shift}"
    for t in shift:
        assert "points vs current" in t, t
    unwind = [t for t in text if t.startswith("Liquidity-adjusted CVaR =")]
    assert unwind and "relative" in unwind[0], unwind
    # The Book's tail gap is a difference of two ES figures: points too
    # (Council A review, 2026-10-22: "2.4% shallower at ES 97.5").
    gaps = [t for t in text if "shallower at ES" in t or "DEEPER at ES" in t]
    assert gaps, "no Book tail-gap sentence on the page"
    for t in gaps:
        assert re.search(r"\d points (shallower|DEEPER) at ES", t), t


def test_the_headline_unwind_names_its_book_size_and_participation():
    """ME-03 (2026-10-22): the verdict widened the tail for "the ~N trading
    days it takes to fully unwind at 20% of daily volume" without saying the
    days were for a $1,000,000 book - the one input they turn on - while The
    Book and the Liquidity tab each let the visitor pick another size."""
    from src.narrative import DEFAULT_PARTICIPATION
    v = _verdict(_offline_page("thin volume", adv=5e4)["text"])
    assert "tail widens to" in v, f"thin volume did not widen the tail: {v}"
    assert "$1,000,000 book" in v, v
    assert f"{DEFAULT_PARTICIPATION:.0%} of each name's average daily dollar volume" in v, v


def test_a_median_that_can_be_negative_is_called_a_return_not_a_gain():
    """ME-03 (2026-10-22): the crisis slabs printed "median gain three years
    after the scariest day" over -21% and "even a buyer at the pre-crash peak
    was ahead three years later (median -39%)"; the tab's metric read "Median
    1y gain from the trough = -19%". Words that assume a sign cannot sit on a
    number that can take either."""
    import re
    page = " ".join(_offline_page()["text"])
    assert re.search(r"median return -\d+%", page), \
        "fixture no longer shows a negative median"
    for word in ("median gain", "Median 1y gain", "even a buyer"):
        assert word not in page, f"{word!r} on the page"


def test_the_breadth_label_and_its_caption_quote_the_same_numbers():
    """ME-03 (2026-10-22): "IR at raw breadth (605 bets/yr)" over a caption
    "12 names x 50 rebalances a year" - 252/5 = 50.4 was rounded only in the
    caption, so the two printed products disagreed."""
    import re
    page = " ".join(_offline_page()["text"])
    label = int(re.search(r"IR at raw breadth \((\d+) bets/yr\)", page).group(1))
    names, reb = re.search(r"Raw breadth counts (\d+) names × ([\d.]+) rebalances",
                           page).groups()
    assert round(int(names) * float(reb)) == label, (names, reb, label)


def test_the_page_points_only_to_controls_that_exist():
    """ME-03 (2026-10-22): the factor caption said "Flip the covariance
    estimator to EWMA in Engine controls"; no control on the page has that
    name - the estimator lives in the 02 drawer."""
    page = _offline_page()
    text = " ".join(page["text"])
    assert "Engine controls" not in text
    alloc = [e.split(" - ")[0] for e in page["expanders"] if e.startswith("02 ")]
    assert alloc and f"to EWMA in {alloc[0]}" in text, alloc


# --- ME-09 (2026-10-22): data-quality holes ---

def test_a_day_with_no_volume_reported_is_skipped_not_read_as_zero():
    """ME-03 (2026-10-22): the Book zero-filled missing dollar-volume days
    (`fillna(0.0)`), so a name with a few gaps in its last 63 days had its
    5th-percentile volume dragged to 0 and was printed as "reports no volume
    on this feed" - a feed that did report volume on every other day."""
    def gappy(tickers=None, period="2y", **_):
        idx = _offline_market_prices(tickers, period).index
        dv = pd.DataFrame(5e9, index=idx, columns=list(tickers))
        if "V" in dv.columns:
            dv.iloc[-63::6, dv.columns.get_loc("V")] = np.nan   # 11 of 63 days
        return dv
    at, restore = _offline_app(dollar_volume=gappy)
    try:
        cards = [m.value for m in at.markdown if "border-left:3px solid" in m.value]
        v = [c for c in cards if ">V " in c]
        assert v, "no Book card for V"
        assert "reports no volume" not in v[0], v[0]
    finally:
        restore()


def test_a_crisis_with_none_of_the_basket_trading_is_left_out_of_the_race():
    """ME-03 (2026-10-22): when no basket member traded at a crisis's start,
    recovery_race set basket_fell=False, which the page reads as "never fell"
    and race_days() as 0 - a basket WIN in a race it never ran. It must be
    left out of the count and named as left out."""
    import src.conviction as conv
    from src.scenarios import HISTORICAL_REGIMES
    idx = pd.bdate_range("1999-01-04", "2026-10-14")
    rng = np.random.default_rng(4)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (len(idx), 3)), axis=0)),
                      index=idx, columns=["SPY", "NEW1", "NEW2"])
    px.loc[px.index < "2012-01-03", ["NEW1", "NEW2"]] = np.nan
    real = conv.fetch_prices
    conv.fetch_prices = lambda tickers, **k: px[list(tickers)]
    try:
        race = conv.recovery_race(basket=["NEW1", "NEW2"], benchmark="SPY")
    finally:
        conv.fetch_prices = real
    early = [n for n, (s, _) in HISTORICAL_REGIMES.items() if pd.Timestamp(s) < pd.Timestamp("2012-01-03")]
    assert early, "fixture no longer has a crisis before the basket listed"
    rec = conv.race_record(race)
    assert set(early) <= set(rec["left_out"]), rec["left_out"]
    assert not set(early) & set(rec["decided"]["crisis"]), list(rec["decided"]["crisis"])
    assert rec["n"] == len(rec["decided"]) and 0 <= rec["wins"] <= rec["n"]


def test_the_balance_tab_ranks_and_sizes_a_pair_on_one_risk_matrix():
    """ME-03 (2026-10-22): Balance ranked partners on the sample correlation
    but sized the blend on the selected estimator. Under EWMA the two differ
    by up to 0.47 in correlation and 16 points of annual vol on the offline
    market, so the page could name a "least-correlated" partner and quote a
    correlation the blend beside it was not computed from."""
    from src.covariance import estimate_covariance
    from src.comovement import correlation_from_cov
    from src.hedge import rank_hedges
    from src.ingestion import get_returns
    at, restore = _offline_app()
    try:
        [s for s in at.selectbox if s.label == "Covariance estimator"][0].set_value("EWMA").run()
        loaded = list([w for w in at.selectbox if w.label.startswith("Anchor")][0].options)
        anchor = [w for w in at.selectbox if w.label.startswith("Anchor")][0].value
        cap = [c.value for c in at.caption if "Long-only minimum-variance weights" in c.value]
        # The defensive pair beside it is a real-history backtest on plain
        # sample volatility, so it says "realized" rather than borrow the label.
        pair = [m.value for m in at.markdown if "annual vol to" in m.value]
        assert pair and "realized annual vol to" in pair[0], pair[:1]
    finally:
        restore()
    rets = get_returns(_offline_market_prices(loaded, "2y"))
    cov, _ = estimate_covariance(rets, "EWMA")
    ranked = rank_hedges(correlation_from_cov(cov), anchor)
    best, corr = ranked.index[0], float(ranked.iloc[0])
    assert cap, "Balance caption missing"
    assert f"**{best}**" in cap[0] and f"{corr:+.2f}" in cap[0], (best, f"{corr:+.2f}", cap[0])
    assert "EWMA" in cap[0], f"the caption does not name its risk matrix: {cap[0]}"


# --- ME-10 (2026-10-22): one way to do each thing ---

def test_trading_days_per_year_is_defined_once():
    """ME-03 (2026-10-22): TRADING_DAYS = 252 was defined in three modules
    and ~40 bare 252s stood beside them - one number, many places to drift.
    It lives in src.covariance; every other module imports it."""
    import ast
    root = pathlib.Path(__file__).resolve().parent.parent
    defs, bare = [], []
    for f in sorted((root / "src").glob("*.py")) + [root / "main.py"]:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        the_one = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "TRADING_DAYS" for t in n.targets):
                defs.append(f.name)
                the_one.add(id(n.value))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Constant) and id(n) not in the_one and type(n.value) in (int, float)
                    and n.value in (252, 756)):
                bare.append(f"{f.name}:{n.lineno}")
    assert defs == ["covariance.py"], f"TRADING_DAYS defined in {defs}"
    assert not bare, f"bare trading-day counts: {bare}"


def test_there_is_one_expected_shortfall_and_it_skips_missing_days():
    """ME-03 (2026-10-22): risk.cvar and pairing.expected_shortfall were two
    copies of one estimator, and only one dropped missing days - the other
    returned nan on a series with a gap."""
    from src.pairing import expected_shortfall
    rng = np.random.default_rng(9)
    r = pd.Series(rng.standard_t(4, 600) * 0.01)
    gappy = r.copy()
    gappy.iloc[::50] = np.nan
    assert np.isfinite(cvar(gappy)), "cvar read a missing day as nan"
    assert cvar(gappy) == cvar(gappy.dropna())
    for c in (0.95, 0.975, 0.99):
        assert expected_shortfall(gappy, c) == cvar(gappy, c)


def test_the_jump_calibration_uses_the_unbiased_sample_variance():
    """ME-10 (2026-10-22): calibrate_jump_diffusion is a thresholded method of
    moments - each part's mean and spread read off its own days - not a
    Gaussian MLE, so the unbiased (n-1) sample variance is the estimator. Its
    ddof=0 understated the variance by (n-1)/n: ~7% on a 15-day crisis
    window, which moved the headline CVaR +1.20 points there (+0.28 on 52
    days, +0.06 on two years; Monte Carlo s.e. ~0.25)."""
    from src.risk import calibrate_jump_diffusion
    lr = np.log1p(np.linspace(-0.01, 0.01, 15))        # 15 calm days, no jumps
    p = calibrate_jump_diffusion(np.expm1(lr))
    assert p["n_jumps"] == 0
    assert abs(p["sigma_d"] ** 2 - lr.var(ddof=1)) < 1e-15, (p["sigma_d"] ** 2, lr.var(ddof=1))



def test_a_fall_is_written_as_its_size_never_as_a_double_negative():
    """Council A review (2026-10-22): the defensive pair read "holding META
    alone fell -36% at its worst" - fell, minus: a reader can take it as a
    rise. A fall is stated by its size beside the verb."""
    import re
    page = " ".join(_offline_page()["text"])
    assert re.search(r"alone fell \d+% at its worst", page), "pair sentence missing"
    assert re.search(r"the fall was \d+%", page), "pair sentence missing"
    for bad in (r"fell -\d", r"fall was -\d"):
        assert not re.search(bad, page), bad


def test_the_loaded_window_is_counted_one_way_everywhere():
    """Council A review (2026-10-22): the same window read "Data: 504 trading
    days" in one caption and "503 trading days" in the next - 504 prices give
    503 daily returns, but one noun with two counts reads as a contradiction.
    Prices are counted in trading days, returns in daily returns, and the
    window's two counts are never swapped."""
    import re
    page = " ".join(_offline_page()["text"])
    rows = int(re.search(r"Data: (\d+) trading days of prices \((\d+) daily returns\)", page).group(1))
    assert f"{rows - 1} trading days" not in page, f"{rows - 1} returns called trading days"
    assert f"{rows} daily returns" not in page, f"{rows} prices called daily returns"
    assert f"{rows - 1} daily returns" in page



def test_the_pair_audit_download_retries_a_name_the_feed_dropped_and_fails_one_it_never_serves():
    """Pair audit runs 28 and 31 (2026-10-22) each failed TICK-01 on one name,
    BAC then BA, with every other check green. Run 31's download log names the
    cause: "['BA']: OperationalError('database is locked')" - yfinance's own
    timezone cache, locked under its download threads, dropped the name for
    one request. The download goes through the app's chokepoint, which now
    asks again for a dropped name (ME-14): dropped once, BA passes; never
    served, it still fails TICK-01. Nothing is filled in."""
    import tempfile
    import tests.pair_audit as pa
    victim = "BA"
    assert victim in pa.UNIVERSE + pa.EXTRA
    results = {}
    try:
        for name, drops in (("once", 1), ("always", 99)):
            with _FakeYahoo(flaky={victim: drops}):
                out = pathlib.Path(tempfile.mkdtemp()) / "prices.parquet"
                assert pa.download(str(out)) == 0
            pa._MASTER = pd.read_parquet(out)
            rows = pa.audit_ticker(victim, pa._MASTER.index[-1])
            results[name] = [r[2] for r in rows if r[0] == "TICK-01"]
            s = pa._MASTER[victim].dropna() if victim in pa._MASTER else pd.Series(dtype=float)
            want = len(_fake_history(victim).index)
            assert s.empty or len(s) == want, f"{name}: {len(s)} rows of {victim}, not {want}"
    finally:
        pa._MASTER = None
    assert results["once"] == ["PASS"], f"a name dropped once must be fetched again: {results}"
    assert results["always"] == ["FAIL"], f"a name never served must still fail TICK-01: {results}"


def test_the_download_asks_again_only_for_a_dropped_name_and_only_with_budget():
    """ME-14: the lock that dropped BA from the audit can drop a name from a
    visitor's basket in the same chokepoint. A name that comes back empty is
    asked for again, alone, at most twice: dropped once it is served and
    cached; never served it stays missing, uncached and left out exactly as
    before; with nothing dropped there is no extra request at all; and with
    no budget left the retry does not spend what is not there."""
    import src.ingestion as ing
    with _FakeYahoo() as fy:                              # the normal path
        ing._download_close_volume(["AAA", "BBB"], "2y")
        assert fy.downloads == [["AAA", "BBB"]], f"extra requests: {fy.downloads}"
    with _FakeYahoo(flaky={"BBB": 1}) as fy:              # dropped once
        close, _ = ing._download_close_volume(["AAA", "BBB"], "2y")
        assert fy.downloads == [["AAA", "BBB"], ["BBB"]], fy.downloads
        assert close["BBB"].notna().sum() > 100, "the second request was not used"
        assert pathlib.Path(ing._raw_path("BBB", "2y")).is_file(), "served but not cached"
    with _FakeYahoo(flaky={"BBB": 99}) as fy:             # never served
        close, _ = ing._download_close_volume(["AAA", "BBB"], "2y")
        assert fy.downloads == [["AAA", "BBB"], ["BBB"], ["BBB"]], fy.downloads
        assert close["BBB"].isna().all(), "a name never served was filled in"
        assert not pathlib.Path(ing._raw_path("BBB", "2y")).exists(), "a failure was cached"
        px = ing.fetch_prices(["AAA", "BBB"], use_cache=False)
        assert "BBB" not in px.columns, "left out, as a name the feed has no data for"
    with _FakeYahoo(flaky={"BBB": 1}) as fy:              # no budget for a retry
        spent = {"n": 0}

        def budget(*a, **k):
            spent["n"] += 1
            if spent["n"] > 1:
                raise ing.BudgetExhausted("spent")
        ing.check_budget = budget
        close, _ = ing._download_close_volume(["AAA", "BBB"], "2y")
        assert fy.downloads == [["AAA", "BBB"]], f"a retry spent a spent budget: {fy.downloads}"
        assert close["BBB"].isna().all()


def test_the_app_calls_nothing_streamlit_has_deprecated():
    """Pillar 4, zero warnings: components.html printed "Please replace
    st.components.v1.html with st.iframe" on every run of every session. The
    glide script now ships through st.iframe in a container app.css folds to
    zero height, so the layout above the hero is what components.html drew."""
    import ast
    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    names = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.alias))}
    assert not any("components" in n for n in names), "main.py still reaches for st.components"
    css = (root / "static" / "app.css").read_text(encoding="utf-8")
    assert ".st-key-glide" in css and "height: 0 !important" in css, \
        "the glide frame must fold to zero height, or the hero moves down"


def test_a_cold_start_with_the_feed_down_shows_guidance_not_a_crash():
    """Phase 3 cold start: a first visitor on an empty cache with Yahoo
    unreachable. The page must not raise, must say what to do in the visitor's
    words, and must show no number at all - no verdict, no map, no placeholder."""
    from src.ingestion import NO_DATA_MSG, NoDataError

    def feed_down(*a, **k):
        raise NoDataError(NO_DATA_MSG)

    at, restore = _offline_app(market=feed_down, dollar_volume=feed_down)
    try:
        assert [e.value for e in at.error] == [NO_DATA_MSG], [e.value for e in at.error]
        page = " ".join(m.value for m in at.markdown)
        assert '<div class="verdict-box">' not in page, "a verdict rendered with no data behind it"
        frames = [f.proto.srcdoc for f in at.get("iframe")]
        assert all("__meleonaGlide" in f for f in frames), "the map rendered with no data behind it"
        assert any("Crisis record unavailable" in c.value for c in at.caption)
    finally:
        restore()


def test_the_map_draws_no_node_or_link_the_data_did_not_produce():
    """Phase 3, zero fake nodes. Every dot on the Risk Topology map is a name
    the visitor holds (labelled Universe) or a disclosed defensive ETF from
    DEFENSIVE_ANCHOR_TICKERS (labelled Anchor pool) - nothing else, and each
    held name is drawn. Every link joins two drawn dots, starts at the book's
    most volatile holding, and every number on both is finite. Long and short."""
    import math
    import src.topology as tp
    from src.pairing import DEFENSIVE_ANCHOR_TICKERS

    held = ["AAPL", "MSFT", "NVDA", "XOM"]
    px = _offline_market_prices(held + DEFENSIVE_ANCHOR_TICKERS + ["SPY"], "2y")
    rets = px[held].pct_change().dropna()
    w = pd.Series(0.25, index=held)
    orig = tp.fetch_prices
    try:
        tp.fetch_prices = lambda tickers, *a, **k: px[list(tickers)]
        for short in (False, True):
            pl = tp.build_map_payload(rets, w, held, short)
            drawn = {a["t"]: a["f"] for a in pl["assets"]}
            assert {t for t, f in drawn.items() if f == "Universe"} == set(held), drawn
            assert {t for t, f in drawn.items() if f == "Anchor pool"} <= set(DEFENSIVE_ANCHOR_TICKERS), drawn
            assert all(math.isfinite(a["b"]) and math.isfinite(a["v"]) for a in pl["assets"])
            flyer = str((rets.std()).idxmax())
            assert pl["pairs"], "the data supports linkages here; none were drawn"
            for p in pl["pairs"]:
                assert p["fl"] == flyer and p["fl"] in drawn and p["an"] in drawn, p
                assert all(math.isfinite(v) for v in [p["rho"], p["wFlyer"], *p["real"].values()]), p
            assert pl["live"] and "Yahoo daily closes" in pl["provenance"]
    finally:
        tp.fetch_prices = orig

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
