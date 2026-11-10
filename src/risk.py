"""
risk engine: VaR, CVaR, tail fits, backtests and two monte carlos.

how the math works:
- VaR: "how much do i lose on a bad day?" (the 95th-percentile loss)
- CVaR: "ok, and when it IS that bad, how bad on average?" - it actually
  looks inside the tail, VaR only says where the tail starts.
- monte carlo: returns aren't normal, so we resample real history instead.
  that keeps the real fat tails - the crash days a bell curve calls impossible.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats, special
from src.covariance import RISKMETRICS_LAMBDA, TRADING_DAYS
from src.ingestion import fetch_prices, get_returns


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """daily returns of the whole book for these weights."""
    return returns @ weights


def var(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """historical VaR: the loss you only beat (1 - confidence) of the time.
    positive = loss, so 0.032 means a 3.2% loss."""
    return float(-np.percentile(port_returns, (1 - confidence) * 100))


def cvar(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / expected shortfall: the average loss on the worst (1 - confidence)
    of days. this is the one risk desks actually use - it sees how deep the
    tail goes, not just where it starts."""
    # the engine's only ES; pairing.expected_shortfall calls this at 97.5.
    # missing days get dropped, never read as returns
    r = pd.Series(port_returns).dropna()
    threshold = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= threshold]
    return float(-tail.mean())


def sharpe_ratio(port_returns: pd.Series, risk_free_rate: float = 0.0,
                 periods: int = TRADING_DAYS) -> float:
    """annualized sharpe: (annual return - risk free) / annual vol.
    risk_free_rate is an annual decimal (0.05). daily mean scales by `periods`,
    daily std by sqrt(periods)."""
    mu = float(port_returns.mean()) * periods
    sigma = float(port_returns.std()) * np.sqrt(periods)
    if sigma < 1e-12:            # ~zero vol: sharpe is undefined, don't blow up
        return float("nan")
    return (mu - risk_free_rate) / sigma


# scipy's genpareto.fit burns ~95% of its time in generic arg checks around its
# objective. _lean_nnlf is that objective for genpareto with loc=0: same numpy
# calls, same order, so the optimizer walks the same path to the same floats,
# 3x faster. bit-for-bit only on scipy 1.18.1 (pinned; a test checks it).
# np.log not math.log - they differ in the last bit sometimes and move the fit
_LOGXMAX = np.log(np.finfo(float).max)


def _lean_nnlf(theta, data):
    c, scale = theta[0], theta[1]
    if not np.isfinite(c) or scale <= 0:
        return np.inf
    x = (data - 0.0) / scale
    n_log_scale = len(x) * np.log(scale)
    b = -1.0 / c if c < 0 else np.inf
    with np.errstate(invalid="ignore"):
        ok = (0.0 <= x) & (x <= b)
    n_bad = len(x) - np.count_nonzero(ok)
    if n_bad > 0:
        x = x[ok]
    if c != 0:
        logpdf = -special.xlog1py(c + 1.0, c * x) / c
        logpdf = np.where(x == x, logpdf, -x)
    else:
        logpdf = -x
    finite = np.isfinite(logpdf)
    n_bad += logpdf.size - np.count_nonzero(finite)
    return -np.sum(logpdf[finite]) + n_bad * _LOGXMAX * 100 + n_log_scale


def _lean_fmin(func, x0, args=(), disp=0):
    # scipy's default optimizer, but on the lean objective. ignores func, so
    # it's only right for the floc=0 MLE gpd_tail_fit runs
    return optimize.fmin(_lean_nnlf, x0, args=args, disp=disp)


def gpd_tail_fit(port_returns: pd.Series, threshold_quantile: float = 0.95,
                 tail_quantiles: tuple = (0.99, 0.999),
                 n_boot: int = 200, seed: int = 0,
                 min_exceedances: int = 50) -> dict:
    """
    peaks-over-threshold EVT: fit a generalized Pareto (GPD) to the loss tail.
    says how heavy the tail is and whether a worst case even exists - CVaR can't.

    how the math works:
    CVaR gives the average loss past VaR, but not the tail's SHAPE (how fast it
    dies off). Pickands-Balkema-de Haan: losses over a high threshold u end up
    looking like a generalized Pareto:

        G(y) = 1 - (1 + xi*y/beta)^(-1/xi)      y = L - u,  L = loss > u

    xi is the whole story:
      xi > 0   heavy tail, no finite worst case
      xi = 0   exponential tail
      xi < 0   bounded tail, worst case at u - beta/xi

    stock losses basically always fit xi > 0, so the honest answer to "what's
    the most i can lose" is: no finite max, just -100% from limited liability.
    we say that instead of making up a number. xi < 0 on stocks is way more
    likely noise than a real bound, and gets labelled that way.

    xi also says when our own numbers break, since E[L^k] is finite only if
    xi < 1/k:
      xi >= 0.5  infinite variance (vol can't be estimated)
      xi >= 1    infinite mean (ES doesn't exist)

    tail estimates (McNeil-Frey / Embrechts et al.), n obs, N_u exceedances:

        VaR_q = u + (beta/xi) * [ ((n/N_u)(1-q))^(-xi) - 1 ]
        ES_q  = VaR_q/(1-xi) + (beta - xi*u)/(1-xi)          (needs xi < 1)

    near xi = 0 we use the limits (VaR_q -> u + beta*ln(N_u/(n(1-q))),
    ES_q -> VaR_q + beta) so we never divide by ~0.

    limits:
    - POT assumes exceedances are iid. if christoffersen_test says they
      cluster, xi here runs high and its CI too narrow - mcneil_frey_tail
      (conditional EVT) is the fix.
    - the CI is a percentile bootstrap over exceedances: estimation noise
      only, not threshold choice.
    - under `min_exceedances` points above u the theory doesn't apply, so we
      refuse to fit instead of returning a shaky number.
    """
    losses = -pd.Series(port_returns).dropna().astype(float)
    n = int(len(losses))
    out = {
        "fitted": False, "reason": None, "xi": None, "beta": None,
        "threshold": None, "n_exceedances": 0, "n": n,
        "xi_ci": None, "tail": {}, "finite_endpoint": None,
        "moments_finite": None, "threshold_quantile": threshold_quantile,
    }
    if n < 100:
        out["reason"] = f"only {n} observations, too few for a tail fit"
        return out

    u = float(np.quantile(losses, threshold_quantile))
    exceed = losses[losses > u].values - u
    n_u = int(len(exceed))
    out["threshold"], out["n_exceedances"] = u, n_u
    if n_u < min_exceedances:
        out["reason"] = (f"only {n_u} losses above the "
                         f"{threshold_quantile:.0%} threshold; "
                         f"{min_exceedances} needed for the GPD asymptotics")
        return out

    # floc=0: exceedances are already measured from the threshold
    xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0, optimizer=_lean_fmin)
    xi, beta = float(xi), float(beta)
    out["xi"], out["beta"], out["fitted"] = xi, beta, True

    rng = np.random.default_rng(seed)   # fixed seed so the CI doesn't flicker on rerun
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(exceed, size=n_u, replace=True)
        try:
            b_xi, _, _ = stats.genpareto.fit(sample, floc=0.0, optimizer=_lean_fmin)
            boot.append(float(b_xi))
        except Exception:  # noqa: BLE001 - a failed refit is dropped, not faked
            continue
    if len(boot) >= 50:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["xi_ci"] = (float(lo), float(hi))

    # a finite worst case only exists when xi < 0
    out["finite_endpoint"] = (u - beta / xi) if xi < 0 else None
    out["moments_finite"] = {
        "mean": xi < 1.0,          # ES needs the mean to exist
        "variance": xi < 0.5,
        "kurtosis": xi < 0.25,
    }

    tiny = 1e-6
    for q in tail_quantiles:
        ratio = (n / n_u) * (1.0 - q)
        if abs(xi) < tiny:                       # exponential limit
            var_q = u + beta * np.log(1.0 / ratio)
        else:
            var_q = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        if xi >= 1.0:
            es_q = None                          # no mean, no ES
        elif abs(xi) < tiny:
            es_q = var_q + beta
        else:
            es_q = var_q / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
        out["tail"][q] = {"var": float(var_q),
                          "es": None if es_q is None else float(es_q)}
    return out


def ewma_volatility(returns, lam: float = RISKMETRICS_LAMBDA,
                    seed_window: int = 60) -> tuple:
    """
    RiskMetrics EWMA vol, plus tomorrow's forecast.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    note the t-1 on r: sigma_t is a forecast made BEFORE day t, so dividing
    r_t by it has no look-ahead. seeded from the first `seed_window` returns;
    the caller throws those early points away.

    returns (sigma per day, aligned to `returns`; sigma for T+1).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n <= seed_window:
        raise ValueError(f"need more than {seed_window} returns to seed EWMA")
    var = np.empty(n, dtype=float)
    var[0] = float(np.var(r[:seed_window], ddof=1))
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    sigma_next = float(np.sqrt(lam * var[-1] + (1.0 - lam) * r[-1] ** 2))
    return np.sqrt(var), sigma_next


def mcneil_frey_tail(port_returns: pd.Series, lam: float = RISKMETRICS_LAMBDA,
                     burn_in: int = 60, **gpd_kwargs) -> dict:
    """
    conditional EVT - McNeil & Frey (2000), two steps.

    how the math works:
    plain POT assumes big losses are iid, but vol clusters, so they come in
    bunches. when christoffersen_test rejects independence, the plain xi runs
    HIGH - clustering looks like a fatter tail. so we strip the vol out first:

      1. filter: get conditional vol sigma_t, set z_t = r_t / sigma_t. the
         clustering lives in sigma, so z is way closer to iid than r.
      2. fit: POT GPD on z. this xi is the tail's real heaviness.
      3. rescale: tomorrow's risk = the residual quantile times tomorrow's vol:

             VaR_{T+1} = sigma_{T+1} * z_q
             ES_{T+1}  = sigma_{T+1} * z_ES

    xi answers "how heavy is this tail", sigma_{T+1} answers "how risky is
    right now". the plain fit mixes the two up.

    where we differ from the paper: they use AR(1)-GARCH(1,1), we filter with
    EWMA (IGARCH(1,1) with lam fixed at RiskMetrics 0.94). nothing gets fitted
    to the sample so it can't overfit, and no extra dependency. mean is taken
    as zero: one-day drift is ~100x smaller than daily vol, so an AR(1) mean
    term would add noise, not accuracy.

    returns the GPD fit on z plus `sigma_next`, a `conditional` block with the
    rescaled VaR/ES per quantile, and `unconditional_xi` - the gap between the
    two xi's IS the clustering bias.
    """
    r = pd.Series(port_returns).dropna().astype(float)
    if len(r) <= burn_in + 100:
        return {"fitted": False,
                "reason": (f"need more than {burn_in + 100} returns for the "
                           "EWMA filter plus a tail fit"),
                "sigma_next": None, "conditional": {}, "unconditional_xi": None}

    sigma, sigma_next = ewma_volatility(r.values, lam=lam, seed_window=burn_in)
    # skip the burn-in: those sigmas still remember the seed, not the recursion
    z = pd.Series(r.values[burn_in:] / sigma[burn_in:], index=r.index[burn_in:])

    fit = gpd_tail_fit(z, **gpd_kwargs)
    # everything from that fit is in z-units (r / sigma). the UI once printed a
    # 2.1 residual threshold as "2.10%", so tag the scale and add return-scale twins
    fit["scale"] = "standardised residuals (z = r / sigma)"
    fit["threshold_z"] = fit.get("threshold")
    fit["threshold_return"] = (None if fit.get("threshold") is None
                               else sigma_next * fit["threshold"])
    fit["finite_endpoint_z"] = fit.get("finite_endpoint")
    fit["finite_endpoint_return"] = (None if fit.get("finite_endpoint") is None
                                     else sigma_next * fit["finite_endpoint"])
    fit["sigma_next"] = sigma_next
    fit["lam"] = lam
    fit["n_standardised"] = int(len(z))
    fit["conditional"] = {}
    if fit["fitted"]:
        for q, vals in fit["tail"].items():
            fit["conditional"][q] = {
                "var": sigma_next * vals["var"],
                "es": None if vals["es"] is None else sigma_next * vals["es"],
            }
    # the raw fit is only here for its xi, so skip its bootstrap (half the time)
    raw = gpd_tail_fit(r, **{**gpd_kwargs, "n_boot": 0})
    fit["unconditional_xi"] = raw["xi"]
    return fit


def christoffersen_test(breach_flags, kupiec_lr: float | None = None) -> dict:
    """
    Christoffersen (1998) independence + conditional coverage tests.

    how the math works:
    Kupiec only checks the NUMBER of breaches, not WHEN. 12 breaches spread
    over a year and 12 in one week score the same - but the second is a model
    that breaks in a crash, the only time it matters.

    so treat the breach flag as a markov chain and count transitions:

        n_ij = # days where I(t-1) = i and I(t) = j

        pi01 = n01 / (n00 + n01)      P(breach | no breach yesterday)
        pi11 = n11 / (n10 + n11)      P(breach | breach yesterday)
        pi   = (n01 + n11) / N        overall breach rate

    if breaches are independent, pi01 = pi11 = pi. the likelihood ratio

        LR_ind = -2 [ log L(independent) - log L(Markov) ]   ~ chi2(1)

    rejects when breaches predict breaches (they cluster). conditional
    coverage just adds the two:

        LR_cc = LR_uc + LR_ind                               ~ chi2(2)

    notes:
    - xlogy(x, y) = x*log(y) with xlogy(0, 0) = 0, the right limit for the
      0*log(0) terms when a count is zero, so no fake -inf.
    - if no breach is ever followed by another day (zero breaches, or one on
      the last day), pi11 can't be estimated. then LR_ind is None, not 0 -
      a 0 would read as "passed", a made-up verdict.
    """
    I = np.asarray(pd.Series(breach_flags).values, dtype=int)
    prev, cur = I[:-1], I[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    total = n00 + n01 + n10 + n11

    out = {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "transitions": total,
        "lr_ind": None, "p_ind": None, "passed_ind": None,
        "lr_cc": None, "p_cc": None, "passed_cc": None,
        "pi01": None, "pi11": None,
    }
    # need both conditional probs or there's no honest statistic
    if total == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return out

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / total

    log_l_markov = (
        special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01)
        + special.xlogy(n10, 1 - pi11) + special.xlogy(n11, pi11)
    )
    log_l_indep = (
        special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
    )
    lr_ind = float(-2.0 * (log_l_indep - log_l_markov))
    # a real negative LR is impossible (markov nests independence), so tiny
    # negatives are float noise
    if -1e-9 < lr_ind < 0:
        lr_ind = 0.0

    out["pi01"], out["pi11"] = pi01, pi11
    out["lr_ind"] = round(lr_ind, 2)
    out["p_ind"] = float(stats.chi2.sf(lr_ind, df=1))
    out["passed_ind"] = bool(lr_ind <= 3.841)          # chi2(1) at 95%

    if kupiec_lr is not None and np.isfinite(kupiec_lr):
        lr_cc = float(kupiec_lr) + lr_ind
        out["lr_cc"] = round(lr_cc, 2)
        out["p_cc"] = float(stats.chi2.sf(lr_cc, df=2))
        out["passed_cc"] = bool(lr_cc <= 5.991)        # chi2(2) at 95%
    return out


def var_backtest(port_returns: pd.Series, confidence: float = 0.95,
                 window: int = 250) -> dict:
    """
    walk-forward VaR backtest (Kupiec POF test).

    a 95% VaR should get breached on ~5% of days. too many = it hides risk,
    too few = it's overly scared. Kupiec's proportion-of-failures test makes
    "is the breach rate ok?" a real test (chi-square, 1 dof, 3.841 at 95%).

    why walk-forward: an old version took the 5th percentile of the WHOLE
    series and counted how many of those same returns fell below it - always
    ~5% by construction, so it always passed. here day t's VaR comes ONLY
    from the `window` days before t (the `.shift(1)` stops the rolling
    quantile from seeing the day it's judging), so the test can actually fail.
    """
    q = 1.0 - confidence
    n_obs = len(port_returns)
    # short sample: shrink the window (half the data, at least 60 days) so
    # there's still an out-of-sample day to test
    win = window if n_obs > window else max(60, n_obs // 2)

    var_line = port_returns.rolling(win).quantile(q).shift(1)
    mask = var_line.notna()
    tested, var_line = port_returns[mask], var_line[mask]
    breach_flags = tested < var_line

    breaches = int(breach_flags.sum())
    n = int(len(tested))
    expected_rate = q
    if n == 0:
        # no out-of-sample day at all, so say so instead of faking a verdict
        return {
            "breaches": 0, "n": 0, "expected_breaches": 0.0,
            "observed_rate": float("nan"), "expected_rate": expected_rate,
            "kupiec_lr": None, "passed": None, "testable": False,
            "window": win, "breach_flags": breach_flags,
        }
    observed_rate = breaches / n

    # Kupiec LR. x = 0 or x = n is fine: xlogy gives 0*log(0) = 0, so zero
    # breaches gives LR = -2n*ln(1-p) - usually a REJECT. (an old version
    # called that nan and then a pass.)
    p = expected_rate
    x = breaches
    lr = -2.0 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
        - special.xlogy(n - x, 1 - x / n) - special.xlogy(x, x / n)
    )
    crit = 3.841  # chi-square(1) at 95%
    # a stat we can't compute never counts as a pass
    passed = None if np.isnan(lr) else bool(lr <= crit)

    result = {
        "breaches": breaches,
        "n": n,
        "expected_breaches": round(expected_rate * n, 1),
        "observed_rate": observed_rate,
        "expected_rate": expected_rate,
        "kupiec_lr": None if np.isnan(lr) else round(float(lr), 2),
        "passed": passed,
        "testable": True,
        "window": win,
        "breach_flags": breach_flags,   # christoffersen needs these
    }
    # Kupiec counts breaches, Christoffersen checks if they cluster. failing
    # that one is worse: the model holds when calm and breaks when it matters
    result["christoffersen"] = christoffersen_test(
        breach_flags, None if np.isnan(lr) else float(lr))
    return result


def mean_block_length(port_returns: np.ndarray, cap: int = 40) -> int:
    """
    how long the bootstrap's blocks should be, measured from the data.

    vol clustering lives in |r|, not r, so a block has to be long enough to
    hold a cluster. we use the integrated autocorrelation time of |r|:
    1 + 2 * (sum of its positive autocorrelations), i.e. "how many days until
    the series forgets itself". we stop at the first non-positive lag (Geyer's
    initial-positive-sequence rule) so long-lag noise can't pump it up.

    clamped to [2, cap]: 1 is just the iid bootstrap we're replacing, and huge
    blocks on two years of data would replay the same stretch over and over.
    """
    x = np.abs(np.asarray(port_returns, dtype=float))
    x = x[np.isfinite(x)]
    if x.size < 30 or x.std(ddof=1) == 0:
        return 2
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return 2
    tau = 1.0
    for lag in range(1, min(100, x.size // 2)):
        rho = float(x[lag:] @ x[:-lag]) / denom
        if rho <= 0:
            break
        tau += 2.0 * rho
    return int(np.clip(round(tau), 2, cap))


def _stationary_bootstrap(n_obs: int, n_paths: int, horizon: int,
                          mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Politis & Romano (1994) stationary bootstrap indices.

    start on a random day; each step, jump to a new random day with
    p = 1/mean_block, else move to the next day (wrapping at the end).
    block lengths come out geometric with mean `mean_block` and the result is
    stationary - fixed blocks would put their joins on the same dates every path.
    """
    p = 1.0 / mean_block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, size=n_paths)
    for t in range(1, horizon):
        jump = rng.random(n_paths) < p
        idx[:, t] = np.where(jump, rng.integers(0, n_obs, size=n_paths),
                             (idx[:, t - 1] + 1) % n_obs)
    return idx


def monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    horizon_days: int = TRADING_DAYS,
    confidence: float = 0.95,
    block: bool = True,
) -> dict:
    """
    block-bootstrap monte carlo of the book's returns.

    how the math works: daily returns barely correlate in SIGN, but their SIZE
    clusters hard - calm weeks follow calm weeks, crashes come in bursts
    (Mandelbrot 1963, and the whole ARCH literature). shuffling single days
    spreads one crash week across a hundred fake years, so the one-year tail
    (the CVaR we report) comes out too thin.

    the stationary bootstrap (Politis & Romano 1994) resamples whole runs of
    history with random geometric lengths, so a path can inherit a full bad
    month. the average block length is measured from the book's own |returns|
    (`mean_block_length`), never typed in. block=False goes back to iid.

    returns a dict of the simulated paths and their risk numbers.
    """
    port_returns = portfolio_daily_returns(returns, weights).values
    rng = np.random.default_rng(seed=42)

    # each row = one simulated year of daily returns
    if block and port_returns.size >= 30:
        mean_block = mean_block_length(port_returns)
        sampled = port_returns[_stationary_bootstrap(
            port_returns.size, n_simulations, horizon_days, mean_block, rng)]
    else:
        mean_block = 1
        sampled = rng.choice(port_returns, size=(n_simulations, horizon_days),
                             replace=True)

    # compound into value paths starting at $1; last column = final value
    value_paths = np.cumprod(1 + sampled, axis=1)
    return _simulation_summary(
        value_paths, horizon_days, n_simulations, confidence,
        engine="stationary block bootstrap" if mean_block > 1 else "bootstrap",
        block_days=mean_block)


def _simulation_summary(value_paths: np.ndarray, horizon_days: int,
                        n_simulations: int, confidence: float, **engine) -> dict:
    """the result dict both monte carlos return, built in one place so the two
    engines can't drift apart (main.py swaps them freely). engine-specific keys
    come in through **engine."""
    final_values = value_paths[:, -1]
    total_returns = final_values - 1
    threshold = np.percentile(total_returns, (1 - confidence) * 100)
    tail = total_returns[total_returns <= threshold]
    return {
        "final_values": final_values,
        "total_returns": total_returns,
        "path_bands": _path_bands(value_paths, horizon_days),
        "path_density": path_density(value_paths),
        "median_return": float(np.median(total_returns)),
        "mean_return": float(np.mean(total_returns)),
        "var": float(-threshold),
        "cvar": float(-tail.mean()),
        "cvar_se": _mc_standard_error(tail),
        "worst_case": float(total_returns.min()),
        "best_case": float(total_returns.max()),
        "prob_loss": float((total_returns < 0).mean()),
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "confidence": confidence,
        **engine,
    }


def _mc_standard_error(tail: np.ndarray) -> float:
    """simulation noise on the CVaR: it's the mean of the tail sample, so its
    standard error is std/sqrt(n). makes the headline "19.3% ± se", not an
    exact truth. covers simulation noise only - model error (is the future
    like history?) can't be fixed with more paths."""
    if tail.size < 2:
        return float("nan")
    return float(tail.std(ddof=1) / np.sqrt(tail.size))


def _path_bands(value_paths: np.ndarray, horizon_days: int) -> dict:
    """fan-chart bands: the p5/p25/p50/p75/p95 paths, as returns (value - 1)
    so the y-axis reads straight as profit/loss."""
    pct = np.percentile(value_paths, [5, 25, 50, 75, 95], axis=0) - 1.0
    return {
        "days": np.arange(1, horizon_days + 1),
        "p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4],
    }


def path_density(value_paths: np.ndarray, n_day_steps: int = 40,
                 n_return_bins: int = 60) -> dict:
    """
    shrink the (n_simulations, horizon_days) value-path matrix (same one
    _path_bands uses) into a (day, return-bin) density surface - the fan
    chart's cone as a probability surface.

    n_day_steps evenly spaced days (all of them is noisy and slow), n_return_bins
    histogram bins per day. returns `days`, `returns` (bin centers) and `density`,
    where each row integrates to 1 so days compare whatever the path count.
    """
    n_sims, horizon_days = value_paths.shape
    day_idx = np.unique(np.linspace(0, horizon_days - 1, n_day_steps, dtype=int))
    edges = np.linspace(-0.80, 1.50, n_return_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    density = np.empty((len(day_idx), n_return_bins))
    for row, di in enumerate(day_idx):
        returns_at_day = value_paths[:, di] - 1.0
        counts, _ = np.histogram(returns_at_day, bins=edges)
        density[row] = counts / (n_sims * np.diff(edges))  # counts -> density

    return {
        "days": day_idx + 1,      # 1-indexed, same as path_bands
        "returns": centers,
        "density": density,
    }


def calibrate_jump_diffusion(port_returns, k: float = 3.0) -> dict:
    """
    split daily returns into a normal DIFFUSION part and a JUMP part, then
    estimate Merton (1976) jump-diffusion params from the data.

    how it works (plain k-sigma thresholding, nothing hidden):
      1. use log-returns so diffusion and jumps just add.
      2. any day more than k std devs from the mean is a JUMP.
      3. diffusion mu/sigma come from the calm days.
      4. jump rate lambda = jump days / all days; jump size mean/std come from
         the extra move on jump days.

    every param comes from the real series. mean-consistent by construction:
    mu_d + lambda*mu_j equals the sample mean exactly. (full Merton uses
    MLE/EM; thresholding is the version a reviewer can redo by hand.)

    returns daily-scale params plus the jump count.
    """
    r = np.asarray(port_returns, dtype=float)
    lr = np.log1p(r)                       # log-returns: diffusion + jumps add
    # ddof=1: method of moments on each part's own days, not a gaussian MLE.
    # ddof=0 shrank the variance ~7% on a 15-day crisis window
    m, s = float(lr.mean()), float(lr.std(ddof=1))
    if s == 0:                             # flat series, nothing to split
        return {"mu_d": m, "sigma_d": 0.0, "lambda_daily": 0.0,
                "mu_j": 0.0, "sigma_j": 0.0, "k": k, "n_jumps": 0, "n_days": len(lr)}

    is_jump = np.abs(lr - m) > k * s
    calm, jumps = lr[~is_jump], lr[is_jump]

    mu_d = float(calm.mean()) if calm.size else m
    sigma_d = float(calm.std(ddof=1)) if calm.size > 1 else s
    lambda_daily = float(is_jump.mean())
    if jumps.size:
        mu_j = float(jumps.mean() - mu_d)          # jump = the move beyond drift
        sigma_j = float(jumps.std(ddof=1)) if jumps.size > 1 else 0.0
    else:
        mu_j = sigma_j = 0.0

    return {
        "mu_d": mu_d, "sigma_d": sigma_d, "lambda_daily": lambda_daily,
        "mu_j": mu_j, "sigma_j": sigma_j,
        "k": k, "n_jumps": int(is_jump.sum()), "n_days": int(lr.size),
    }


def jump_diffusion_mc(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    horizon_days: int = TRADING_DAYS,
    confidence: float = 0.95,
    k: float = 3.0,
) -> dict:
    """
    Merton jump-diffusion monte carlo - same inputs and output as monte_carlo(),
    so the dashboard can swap them.

    why it's different: a bootstrap can only replay tail days it's seen. a
    jump process MAKES new ones - two jumps in one week, or a crash deeper
    than any day in the sample - so VaR/CVaR show what the process can do,
    not just what happened in the last two years.

    each simulated daily log-return:
        r_t = mu_d + sigma_d * Z                    (diffusion)
            + N_t*mu_j + sigma_j*sqrt(N_t)*Z'       (jumps, N_t ~ Poisson(lambda))
    since a sum of N_t iid Normal(mu_j, sigma_j^2) is Normal(N_t*mu_j,
    N_t*sigma_j^2), the whole jump term vectorizes.
    """
    port_returns = portfolio_daily_returns(returns, weights).values
    params = calibrate_jump_diffusion(port_returns, k=k)
    rng = np.random.default_rng(seed=42)

    shape = (n_simulations, horizon_days)
    diffusion = params["mu_d"] + params["sigma_d"] * rng.standard_normal(shape)
    n_jumps = rng.poisson(params["lambda_daily"], size=shape)
    jump = (n_jumps * params["mu_j"]
            + params["sigma_j"] * np.sqrt(n_jumps) * rng.standard_normal(shape))

    cum_log = np.cumsum(diffusion + jump, axis=1)  # compound in log space
    value_paths = np.exp(cum_log)
    return _simulation_summary(
        value_paths, horizon_days, n_simulations, confidence,
        engine="jump-diffusion", jump_params=params)


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)

    n = len(returns.columns)
    equal_weights = np.ones(n) / n
    port_returns = portfolio_daily_returns(returns, equal_weights)

    h_var = var(port_returns)
    h_cvar = cvar(port_returns)
    print("--- Historical Risk (Equal-Weight Portfolio) ---")
    print(f"  Daily VaR  (95%): {h_var:.2%}  - on a bad day, expect to lose at least this")
    print(f"  Daily CVaR (95%): {h_cvar:.2%}  - when it's bad, this is the average loss")

    print("\n--- Running 10,000 Monte Carlo Simulations (1-Year Horizon) ---")
    mc = monte_carlo(returns, equal_weights)
    print(f"  Median 1-year return : {mc['median_return']:+.1%}")
    print(f"  Mean 1-year return   : {mc['mean_return']:+.1%}")
    print(f"  1-Year VaR  (95%)    : {mc['var']:.1%}  loss in worst 5% of years")
    print(f"  1-Year CVaR (95%)    : {mc['cvar']:.1%}  avg loss in worst 5% of years")
    print(f"  Worst simulated year : {mc['worst_case']:+.1%}")
    print(f"  Best simulated year  : {mc['best_case']:+.1%}")
    print(f"  Probability of loss  : {mc['prob_loss']:.1%}")

    print("\n--- Merton Jump-Diffusion Monte Carlo (same portfolio) ---")
    jd = jump_diffusion_mc(returns, equal_weights)
    p = jd["jump_params"]
    print(f"  Calibration: {p['n_jumps']} jump days in {p['n_days']} "
          f"(> {p['k']}sigma) -> {p['lambda_daily'] * TRADING_DAYS:.1f} jumps/yr expected")
    print(f"  Diffusion vol (annual): {p['sigma_d'] * np.sqrt(TRADING_DAYS):.1%}")
    print(f"  1-Year VaR  (95%)    : {jd['var']:.1%}")
    print(f"  1-Year CVaR (95%)    : {jd['cvar']:.1%}")
    print(f"  Worst simulated year : {jd['worst_case']:+.1%}")
    print("\n  Tail comparison (CVaR): "
          f"bootstrap {mc['cvar']:.1%}  vs  jump-diffusion {jd['cvar']:.1%}")
    # mu_d + lambda*mu_j should match the sample mean
    emp = float(np.log1p(port_returns.values).mean())
    recon = p["mu_d"] + p["lambda_daily"] * p["mu_j"]
    print(f"  Mean-consistency: empirical {emp:.2e} vs mu_d+lambda*mu_j {recon:.2e}")
