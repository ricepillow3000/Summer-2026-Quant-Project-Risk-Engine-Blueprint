"""
Risk Engine: CVaR and Monte Carlo simulation.

Quant Deep Dive:
- VaR (Value at Risk) answers: "What's the most I lose on a bad day (95th pctile)?"
- CVaR (Conditional VaR) answers: "When things ARE that bad, how bad on average?"
  CVaR is strictly better - VaR ignores what happens in the tail, CVaR measures it.
- Monte Carlo: instead of assuming returns are normally distributed (they aren't),
  we bootstrap from real historical returns. This captures actual fat tails -
  the real crash days that a normal distribution would say are "impossible."
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from src.covariance import RISKMETRICS_LAMBDA
from src.ingestion import fetch_prices, get_returns


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute daily portfolio returns for a given weight vector."""
    return returns @ weights


def var(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.
    The loss threshold you exceed only (1 - confidence)% of the time.

    Returns a positive number representing the loss (e.g. 0.032 = 3.2% loss).
    """
    return float(-np.percentile(port_returns, (1 - confidence) * 100))


def cvar(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).
    Average loss on the worst (1 - confidence)% of days.

    This is the number risk desks actually use - it captures tail severity,
    not just where the tail begins.
    """
    threshold = np.percentile(port_returns, (1 - confidence) * 100)
    tail = port_returns[port_returns <= threshold]
    return float(-tail.mean())


def sharpe_ratio(port_returns: pd.Series, risk_free_rate: float = 0.0,
                 periods: int = 252) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of volatility.

    Sharpe = (annualized return - risk_free_rate) / annualized volatility

    `risk_free_rate` is an annual decimal (e.g. 0.05). Daily returns are
    annualized by 252 (return) and sqrt(252) (volatility). The single most
    common one-line summary of risk-adjusted performance on a desk.
    """
    mu = float(port_returns.mean()) * periods
    sigma = float(port_returns.std()) * np.sqrt(periods)
    if sigma < 1e-12:            # (near-)zero vol: Sharpe undefined, don't explode
        return float("nan")
    return (mu - risk_free_rate) / sigma


def parametric_var(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Variance-covariance (parametric) VaR: assumes returns are normal and reads
    the loss off the fitted distribution. Faster than historical and smooth, but
    understates tail risk when returns are fat-tailed - which is exactly why we
    backtest it below and keep CVaR as the headline.
    """
    mu, sigma = port_returns.mean(), port_returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def gpd_tail_fit(port_returns: pd.Series, threshold_quantile: float = 0.95,
                 tail_quantiles: tuple = (0.99, 0.999),
                 n_boot: int = 200, seed: int = 0,
                 min_exceedances: int = 50) -> dict:
    """
    Extreme Value Theory: peaks-over-threshold fit of a Generalized Pareto
    to the loss tail. Answers "how heavy is the tail, and is there a worst
    case at all" - which CVaR cannot.

    Quant Deep Dive:
    CVaR/ES already reports the MEAN loss beyond VaR. What neither VaR nor ES
    tells you is the tail's SHAPE - how fast it decays, and therefore how much
    worse things can get past the data you happen to own. The
    Pickands-Balkema-de Haan theorem says that for a wide class of
    distributions, exceedances over a high threshold u converge to a
    Generalized Pareto:

        G(y) = 1 - (1 + xi*y/beta)^(-1/xi)      y = L - u,  L = loss > u

    The shape parameter xi is the whole story:
      xi > 0   heavy tail, polynomial decay, NO finite worst case
      xi = 0   exponential tail
      xi < 0   bounded tail with a finite right endpoint at u - beta/xi

    Equity daily losses fit xi > 0 essentially always, so THE HONEST ANSWER TO
    "what is the absolute maximum I can lose" IS THAT NO FINITE MAXIMUM EXISTS
    - only -100% from limited liability. This function reports that rather
    than inventing a number. A fitted xi < 0 on equity data is far more likely
    sample noise than a real bound, and is labelled as such.

    xi also says when the engine's OWN numbers stop being meaningful, since
    E[L^k] is finite only if xi < 1/k:
      xi >= 0.5  infinite variance   (volatility is not estimable)
      xi >= 1    infinite mean       (ES does not exist at all)

    Tail estimates (McNeil-Frey / Embrechts et al.), with n total
    observations and N_u exceedances:

        VaR_q = u + (beta/xi) * [ ((n/N_u)(1-q))^(-xi) - 1 ]
        ES_q  = VaR_q/(1-xi) + (beta - xi*u)/(1-xi)          (needs xi < 1)

    Both are used in their xi -> 0 limiting forms near zero
    (VaR_q -> u + beta*ln(N_u/(n(1-q))), ES_q -> VaR_q + beta) rather than
    dividing by a near-zero xi.

    Honest limits, stated because they change how the number should be read:
    - POT assumes exceedances are i.i.d. The Christoffersen test in this same
      module exists precisely to detect that they cluster; when it rejects
      independence, xi here is biased upward and its interval is too narrow.
      The fix is conditional EVT (McNeil-Frey: fit the GPD to EWMA-standardised
      residuals rather than raw returns) - not implemented here, flagged.
    - The confidence interval is a percentile bootstrap over exceedances. It
      captures estimation noise, NOT threshold-choice risk.
    - With fewer than `min_exceedances` points above u the asymptotics do not
      apply. This refuses to fit rather than return a shaky number.
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
        out["reason"] = f"only {n} observations - too few for a tail fit"
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

    # floc=0 because exceedances are already measured FROM the threshold.
    xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)
    xi, beta = float(xi), float(beta)
    out["xi"], out["beta"], out["fitted"] = xi, beta, True

    rng = np.random.default_rng(seed)   # fixed: a CI must not flicker on rerun
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(exceed, size=n_u, replace=True)
        try:
            b_xi, _, _ = stats.genpareto.fit(sample, floc=0.0)
            boot.append(float(b_xi))
        except Exception:  # noqa: BLE001 - a failed refit is dropped, not faked
            continue
    if len(boot) >= 50:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["xi_ci"] = (float(lo), float(hi))

    # Finite worst case exists only for xi < 0.
    out["finite_endpoint"] = (u - beta / xi) if xi < 0 else None
    out["moments_finite"] = {
        "mean": xi < 1.0,          # ES exists only if the mean does
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
            es_q = None                          # mean does not exist
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
    RiskMetrics EWMA conditional volatility, plus the one-step-ahead forecast.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    Note the t-1 on the return: sigma_t is a FORECAST made before day t is
    observed, so standardising r_t by it involves no look-ahead. The variance
    is seeded from the first `seed_window` returns; those early points are
    burnt in by the caller rather than trusted.

    Returns (sigma array aligned to `returns`, one-step-ahead sigma for T+1).
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
    Conditional EVT - McNeil & Frey (2000), two-stage.

    Quant Deep Dive:
    Plain peaks-over-threshold assumes exceedances are i.i.d. Financial returns
    are not: volatility clusters, so big losses arrive in bunches. That is not
    a hypothetical objection here - the Christoffersen test in this module
    exists to measure exactly that, and when it rejects independence the
    unconditional xi is biased UPWARD, because clustering masquerades as a
    fatter tail. The engine would then be citing its own violated assumption
    as evidence of danger.

    McNeil-Frey removes the volatility first, then does EVT on what is left:

      1. Filter: estimate conditional volatility sigma_t, standardise
         z_t = r_t / sigma_t. Clustering lives in sigma, so z is far closer
         to i.i.d. than r.
      2. Fit: peaks-over-threshold GPD on the standardised residuals z.
         The xi from this stage is the asset's TRUE tail heaviness, with the
         volatility dynamics stripped out.
      3. Rescale: a risk number for tomorrow is the residual quantile blown
         back up by tomorrow's volatility forecast:

             VaR_{T+1} = sigma_{T+1} * z_q
             ES_{T+1}  = sigma_{T+1} * z_ES

    This is why conditional EVT is the honest version: xi answers "how heavy
    is this tail really", while sigma_{T+1} answers "how dangerous is right
    now". The unconditional fit confounds the two.

    Deviation from the paper, stated rather than glossed: McNeil-Frey specify
    AR(1)-GARCH(1,1). This filters with EWMA, which is IGARCH(1,1) with lam
    fixed at the RiskMetrics 0.94 already used elsewhere in this engine. The
    trade is deliberate - NO volatility parameters are fitted to the sample,
    so the filter cannot overfit it, and no extra dependency is required. The
    conditional mean is taken as zero: at a one-day horizon the equity drift
    is ~2 orders of magnitude below the daily volatility, so the paper's AR(1)
    mean term would add estimation noise rather than accuracy.

    Returns the GPD fit on standardised residuals, plus `sigma_next` and a
    `conditional` block holding the rescaled VaR/ES per quantile, and
    `unconditional_xi` so the two fits can be compared directly - the gap
    between them IS the clustering bias.
    """
    r = pd.Series(port_returns).dropna().astype(float)
    if len(r) <= burn_in + 100:
        return {"fitted": False,
                "reason": (f"need more than {burn_in + 100} returns for the "
                           "EWMA filter plus a tail fit"),
                "sigma_next": None, "conditional": {}, "unconditional_xi": None}

    sigma, sigma_next = ewma_volatility(r.values, lam=lam, seed_window=burn_in)
    # Drop the burn-in: those sigmas still carry the seed rather than the
    # recursion, so their residuals are not comparable to the rest.
    z = pd.Series(r.values[burn_in:] / sigma[burn_in:], index=r.index[burn_in:])

    fit = gpd_tail_fit(z, **gpd_kwargs)
    # EVERYTHING gpd_tail_fit returns here is in STANDARDISED-RESIDUAL units,
    # because it was fitted on z = r / sigma. Only var/es were being rescaled
    # below, so `threshold`, `beta` and `finite_endpoint` stayed in z-units
    # while the UI printed them as percentage losses - a 2.1 residual read as
    # "2.10%". Flag the scale explicitly and publish return-scale companions,
    # so a caller can never mistake one for the other.
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
    # For the side-by-side that makes the clustering bias visible.
    raw = gpd_tail_fit(r, **gpd_kwargs)
    fit["unconditional_xi"] = raw["xi"]
    fit["unconditional_xi_ci"] = raw["xi_ci"]
    return fit


def christoffersen_test(breach_flags, kupiec_lr: float | None = None) -> dict:
    """
    Christoffersen (1998) independence + conditional coverage tests.

    Quant Deep Dive:
    Kupiec asks only "were there about the right NUMBER of breaches?" It is
    blind to WHEN they happened. Twelve breaches spread evenly across a year
    and twelve breaches all inside one week score identically - yet the second
    is a model that works in calm markets and collapses in stress, which is
    the only time a risk number matters.

    Christoffersen tests the missing dimension. Treat the breach indicator as
    a Markov chain and count transitions:

        n_ij = # of days where I(t-1) = i and I(t) = j

        pi01 = n01 / (n00 + n01)      P(breach | no breach yesterday)
        pi11 = n11 / (n10 + n11)      P(breach | breach yesterday)
        pi   = (n01 + n11) / N        unconditional breach rate

    Under independence pi01 = pi11 = pi: yesterday tells you nothing. The
    likelihood ratio

        LR_ind = -2 [ log L(independent) - log L(Markov) ]   ~ chi2(1)

    rejects when breaches predict breaches - i.e. they cluster.

    Conditional coverage combines both dimensions, and the statistics are
    additive by construction:

        LR_cc = LR_uc + LR_ind                               ~ chi2(2)

    Implementation notes that matter for correctness:
    - `xlogy(x, y)` is x*log(y) with xlogy(0, 0) = 0, which is the correct
      limit of the 0*log(0) terms and avoids a spurious -inf when a transition
      count is zero. Every case where the log argument is 0 has a matching
      zero count, so no term is ever genuinely infinite.
    - If no breach is ever followed by another observation (n10 + n11 = 0,
      which happens with zero breaches or a single breach on the last day),
      pi11 is not estimable. Then LR_ind does not exist and this returns None
      for it rather than substituting 0 - a 0 would read as "passed
      independence", which would be a fabricated verdict.
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
    # Both conditional probabilities must be estimable, or the Markov
    # alternative is not identified and no honest statistic exists.
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
    # Clamp microscopic negatives from floating-point error; a genuine
    # negative LR is impossible since the Markov model nests independence.
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
    WALK-FORWARD backtest of historical VaR (Kupiec POF test).

    A VaR model is only trustworthy if losses breach it about as often as it
    claims - a 95% VaR should be exceeded ~5% of days. Too many breaches = the
    model understates risk; too few = it's needlessly conservative. The Kupiec
    proportion-of-failures test turns "is the breach rate acceptable?" into a
    formal hypothesis test (chi-square, 1 dof, 95% critical value 3.841).

    Why walk-forward, and why this used to be broken:
    the previous version took `np.percentile(port_returns, 5)` over the WHOLE
    series and counted how many of those same returns fell below it. That is
    not a test - a sample's own 5th percentile has exactly 5% of the sample
    below it by construction, so breaches were always ~5%, LR was always ~0,
    and `passed` was always True no matter how broken the model was. Calm
    normal returns and a series of catastrophic clustered crashes both scored
    identically.

    Here the VaR for day t is estimated ONLY from the `window` days strictly
    before t, then tested against day t's actual return. `.shift(1)` is what
    enforces that: without it the rolling quantile would include the very
    return it is judging. Now the test can genuinely fail, which is the point.
    """
    q = 1.0 - confidence
    n_obs = len(port_returns)
    # Need at least one out-of-sample day. On a short sample shrink the
    # estimation window rather than refusing to test - half the sample,
    # floored at 60 days, is still an honest walk-forward.
    win = window if n_obs > window else max(60, n_obs // 2)

    var_line = port_returns.rolling(win).quantile(q).shift(1)
    mask = var_line.notna()
    tested, var_line = port_returns[mask], var_line[mask]
    breach_flags = tested < var_line

    breaches = int(breach_flags.sum())
    n = int(len(tested))
    expected_rate = q
    if n == 0:
        # Fewer than ~60 usable days: no out-of-sample day exists. Report that
        # honestly instead of inventing a verdict.
        return {
            "breaches": 0, "n": 0, "expected_breaches": 0.0,
            "observed_rate": float("nan"), "expected_rate": expected_rate,
            "kupiec_lr": None, "passed": None, "testable": False,
            "window": win, "breach_flags": breach_flags,
        }
    observed_rate = breaches / n

    # Kupiec likelihood-ratio statistic for proportion of failures.
    p = expected_rate
    x = breaches
    # x == 0 and x == n are NOT undefined. The unrestricted likelihood is
    # (1-x/n)^(n-x) * (x/n)^x, which at x=0 equals 1, leaving LR = -2n*ln(1-p)
    # - normally a decisive REJECTION, not a pass. The previous version sent
    # both boundaries into the nan branch, and `np.isnan(lr) or ...` then
    # reported passed=True: a zero-breach model scored a fabricated PASS on
    # the one panel whose whole purpose is model validation. xlogy supplies
    # the correct 0*log(0) = 0 limits, exactly as christoffersen_test already
    # does in this module. Only n == 0 is genuinely undefined, and that is
    # handled by the `testable` branch above.
    if n > 0:
        lr = -2.0 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - special.xlogy(n - x, 1 - x / n) - special.xlogy(x, x / n)
        )
    else:
        lr = float("nan")
    crit = 3.841  # chi-square(1) at 95%
    # An incomputable statistic must never read as a pass.
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
        # Kept for the Christoffersen independence test, which asks whether
        # these breaches CLUSTER rather than just how many there are.
        "breach_flags": breach_flags,
    }
    # Second dimension: Kupiec counts breaches, Christoffersen asks whether
    # they arrive independently. A model can pass one and fail the other, and
    # failing independence is the more dangerous failure - it means the model
    # holds in calm markets and breaks exactly when it is needed.
    result["christoffersen"] = christoffersen_test(
        breach_flags, None if np.isnan(lr) else float(lr))
    return result


def monte_carlo(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    horizon_days: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap Monte Carlo simulation of portfolio returns.

    For each simulation: randomly sample `horizon_days` daily returns
    (with replacement) from history and compound them into a final value.
    No normality assumption - we use the real return distribution.

    Args:
        returns: historical daily returns DataFrame.
        weights: portfolio weight vector.
        n_simulations: number of simulated futures to run.
        horizon_days: trading days to simulate (252 = 1 year).
        confidence: CVaR confidence level.

    Returns:
        dict with simulation results and risk metrics.
    """
    port_returns = portfolio_daily_returns(returns, weights).values
    rng = np.random.default_rng(seed=42)

    # Each row = one simulated year of daily returns
    sampled = rng.choice(port_returns, size=(n_simulations, horizon_days), replace=True)

    # Compound into cumulative value paths (start = $1); last column = final value
    value_paths = np.cumprod(1 + sampled, axis=1)
    final_values = value_paths[:, -1]
    total_returns = final_values - 1

    # Risk metrics on the simulated distribution
    sim_var = float(-np.percentile(total_returns, (1 - confidence) * 100))
    threshold = np.percentile(total_returns, (1 - confidence) * 100)
    tail = total_returns[total_returns <= threshold]
    sim_cvar = float(-tail.mean())

    return {
        "final_values": final_values,
        "total_returns": total_returns,
        "path_bands": _path_bands(value_paths, horizon_days),
        "path_density": path_density(value_paths),
        "median_return": float(np.median(total_returns)),
        "mean_return": float(np.mean(total_returns)),
        "var": sim_var,
        "cvar": sim_cvar,
        "cvar_se": _mc_standard_error(tail),
        "worst_case": float(total_returns.min()),
        "best_case": float(total_returns.max()),
        "prob_loss": float((total_returns < 0).mean()),
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "confidence": confidence,
        "engine": "bootstrap",
    }


def _mc_standard_error(tail: np.ndarray) -> float:
    """
    Monte Carlo sampling error of the CVaR estimate.

    CVaR is the MEAN of the simulated tail sample, so its standard error is
    the classic std/sqrt(n) of that sample - shrinking as 1/sqrt(N) with more
    simulations. Reporting it makes the headline honest: a simulated 19.3%
    CVaR at 10,000 paths is "19.3% ± se", not an exact truth. (This prices
    SIMULATION noise only - model error, e.g. whether history resembles the
    future, is disclosed separately and cannot be reduced by more paths.)
    """
    if tail.size < 2:
        return float("nan")
    return float(tail.std(ddof=1) / np.sqrt(tail.size))


def _path_bands(value_paths: np.ndarray, horizon_days: int) -> dict:
    """
    Percentile bands of the simulated cumulative-value paths, for a fan chart.

    Returns each band as a RETURN series (value - 1) over the horizon, so the
    y-axis reads directly in profit/loss terms. p5..p95 form the outcome cone;
    p50 is the median path.
    """
    pct = np.percentile(value_paths, [5, 25, 50, 75, 95], axis=0) - 1.0
    return {
        "days": np.arange(1, horizon_days + 1),
        "p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4],
    }


def path_density(value_paths: np.ndarray, n_day_steps: int = 40,
                 n_return_bins: int = 60) -> dict:
    """
    Downsample the full simulated-path matrix into a (day, return-bin) density
    surface, for a 3D view of how the outcome distribution evolves over the
    horizon - the fan chart's cone, but as a probability surface instead of
    percentile lines.

    Args:
        value_paths: (n_simulations, horizon_days) cumulative-value matrix,
            the same array _path_bands() is built from (start = $1).
        n_day_steps: number of horizon days to sample (evenly spaced) -
            plotting all 252 days makes the surface noisy and slow to rotate.
        n_return_bins: number of return histogram bins per day.

    Returns:
        dict with `days` (n_day_steps,), `returns` (n_return_bins, bin
        centers as decimals), and `density` (n_day_steps, n_return_bins)
        where each row integrates to 1 (a proper density, not a raw count -
        comparable across days regardless of simulation count).
    """
    n_sims, horizon_days = value_paths.shape
    day_idx = np.unique(np.linspace(0, horizon_days - 1, n_day_steps, dtype=int))
    edges = np.linspace(-0.80, 1.50, n_return_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    density = np.empty((len(day_idx), n_return_bins))
    for row, di in enumerate(day_idx):
        returns_at_day = value_paths[:, di] - 1.0
        counts, _ = np.histogram(returns_at_day, bins=edges)
        density[row] = counts / (n_sims * np.diff(edges))  # normalize to a density

    return {
        "days": day_idx + 1,      # 1-indexed trading days, matching path_bands
        "returns": centers,
        "density": density,
    }


def calibrate_jump_diffusion(port_returns, k: float = 3.0) -> dict:
    """
    Split a daily return series into a Gaussian DIFFUSION part and a discrete
    JUMP part, then estimate Merton (1976) jump-diffusion parameters from data.

    Method - transparent k-sigma thresholding, not a black box:
      1. Work in log-returns, so diffusion and jumps add cleanly.
      2. Flag any day more than k standard deviations from the mean as a JUMP.
      3. Diffusion mu/sigma come from the CALM (non-jump) days.
      4. Jump intensity lambda = jump-days / total-days; jump-size mean and std
         come from the excess move on JUMP days.

    Every parameter is estimated from the real series - nothing is assumed. The
    split is mean-consistent by construction: mu_d + lambda*mu_j equals the
    empirical mean exactly. (Full Merton calibration uses MLE/EM; thresholding
    is the honest, reproducible version a reviewer can re-derive by hand.)

    Returns daily-scale parameters plus the jump count for display.
    """
    r = np.asarray(port_returns, dtype=float)
    lr = np.log1p(r)                       # log-returns: diffusion + jumps add
    m, s = float(lr.mean()), float(lr.std())
    if s == 0:                             # degenerate constant series
        return {"mu_d": m, "sigma_d": 0.0, "lambda_daily": 0.0,
                "mu_j": 0.0, "sigma_j": 0.0, "k": k, "n_jumps": 0, "n_days": len(lr)}

    is_jump = np.abs(lr - m) > k * s
    calm, jumps = lr[~is_jump], lr[is_jump]

    mu_d = float(calm.mean()) if calm.size else m
    sigma_d = float(calm.std()) if calm.size > 1 else s
    lambda_daily = float(is_jump.mean())
    if jumps.size:
        mu_j = float(jumps.mean() - mu_d)          # jump = move in EXCESS of drift
        sigma_j = float(jumps.std()) if jumps.size > 1 else 0.0
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
    horizon_days: int = 252,
    confidence: float = 0.95,
    k: float = 3.0,
) -> dict:
    """
    Merton jump-diffusion Monte Carlo - same signature and output dict as
    monte_carlo(), so it drops into the dashboard as an interchangeable engine.

    Why it differs from the bootstrap: resampling can only ever replay tail days
    it has already seen. A jump-diffusion process GENERATES new extreme paths -
    two jumps landing in the same week, or a crash deeper than any single day in
    the sample - so VaR/CVaR reflect what the process can produce, not just what
    happened to occur in the last two years.

    Each simulated daily log-return:
        r_t = mu_d + sigma_d * Z          (diffusion)
            + N_t*mu_j + sigma_j*sqrt(N_t)*Z'    (jumps, N_t ~ Poisson(lambda))
    using that a sum of N_t iid Normal(mu_j, sigma_j^2) is Normal(N_t*mu_j,
    N_t*sigma_j^2) - which lets us vectorize the whole jump term.
    """
    port_returns = portfolio_daily_returns(returns, weights).values
    params = calibrate_jump_diffusion(port_returns, k=k)
    rng = np.random.default_rng(seed=42)

    shape = (n_simulations, horizon_days)
    diffusion = params["mu_d"] + params["sigma_d"] * rng.standard_normal(shape)
    n_jumps = rng.poisson(params["lambda_daily"], size=shape)
    jump = (n_jumps * params["mu_j"]
            + params["sigma_j"] * np.sqrt(n_jumps) * rng.standard_normal(shape))

    cum_log = np.cumsum(diffusion + jump, axis=1)  # compound in log-space, per day
    value_paths = np.exp(cum_log)
    final_values = value_paths[:, -1]
    total_returns = final_values - 1

    sim_var = float(-np.percentile(total_returns, (1 - confidence) * 100))
    threshold = np.percentile(total_returns, (1 - confidence) * 100)
    tail = total_returns[total_returns <= threshold]
    sim_cvar = float(-tail.mean())

    return {
        "final_values": final_values,
        "total_returns": total_returns,
        "path_bands": _path_bands(value_paths, horizon_days),
        "path_density": path_density(value_paths),
        "median_return": float(np.median(total_returns)),
        "mean_return": float(np.mean(total_returns)),
        "var": sim_var,
        "cvar": sim_cvar,
        "cvar_se": _mc_standard_error(tail),
        "worst_case": float(total_returns.min()),
        "best_case": float(total_returns.max()),
        "prob_loss": float((total_returns < 0).mean()),
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "confidence": confidence,
        "engine": "jump-diffusion",
        "jump_params": params,
    }


if __name__ == "__main__":
    prices = fetch_prices()
    returns = get_returns(prices)

    n = len(returns.columns)
    equal_weights = np.ones(n) / n
    port_returns = portfolio_daily_returns(returns, equal_weights)

    # Historical risk metrics
    h_var = var(port_returns)
    h_cvar = cvar(port_returns)
    print("--- Historical Risk (Equal-Weight Portfolio) ---")
    print(f"  Daily VaR  (95%): {h_var:.2%}  - on a bad day, expect to lose at least this")
    print(f"  Daily CVaR (95%): {h_cvar:.2%}  - when it's bad, this is the average loss")

    # Monte Carlo
    print("\n--- Running 10,000 Monte Carlo Simulations (1-Year Horizon) ---")
    mc = monte_carlo(returns, equal_weights)
    print(f"  Median 1-year return : {mc['median_return']:+.1%}")
    print(f"  Mean 1-year return   : {mc['mean_return']:+.1%}")
    print(f"  1-Year VaR  (95%)    : {mc['var']:.1%}  loss in worst 5% of years")
    print(f"  1-Year CVaR (95%)    : {mc['cvar']:.1%}  avg loss in worst 5% of years")
    print(f"  Worst simulated year : {mc['worst_case']:+.1%}")
    print(f"  Best simulated year  : {mc['best_case']:+.1%}")
    print(f"  Probability of loss  : {mc['prob_loss']:.1%}")

    # Jump-diffusion engine - fat-tailed alternative to the bootstrap
    print("\n--- Merton Jump-Diffusion Monte Carlo (same portfolio) ---")
    jd = jump_diffusion_mc(returns, equal_weights)
    p = jd["jump_params"]
    print(f"  Calibration: {p['n_jumps']} jump days in {p['n_days']} "
          f"(> {p['k']}sigma) -> {p['lambda_daily'] * 252:.1f} jumps/yr expected")
    print(f"  Diffusion vol (annual): {p['sigma_d'] * np.sqrt(252):.1%}")
    print(f"  1-Year VaR  (95%)    : {jd['var']:.1%}")
    print(f"  1-Year CVaR (95%)    : {jd['cvar']:.1%}")
    print(f"  Worst simulated year : {jd['worst_case']:+.1%}")
    print("\n  Tail comparison (CVaR): "
          f"bootstrap {mc['cvar']:.1%}  vs  jump-diffusion {jd['cvar']:.1%}")
    # Mean-consistency check: mu_d + lambda*mu_j should match the empirical mean.
    emp = float(np.log1p(port_returns.values).mean())
    recon = p["mu_d"] + p["lambda_daily"] * p["mu_j"]
    print(f"  Mean-consistency: empirical {emp:.2e} vs mu_d+lambda*mu_j {recon:.2e}")