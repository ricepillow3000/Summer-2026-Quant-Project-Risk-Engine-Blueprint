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

import numpy as np
import pandas as pd

from src.analytics import covariance_matrix, correlation_matrix
from src.risk import (
    var, cvar, monte_carlo, jump_diffusion_mc, calibrate_jump_diffusion, sharpe_ratio,
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
        at = AppTest.from_file("main.py", default_timeout=120)
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
