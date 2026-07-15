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
from scipy import stats
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


def var_backtest(port_returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Backtest historical VaR against its own history (Kupiec POF test).

    A VaR model is only trustworthy if losses breach it about as often as it
    claims - a 95% VaR should be exceeded ~5% of days. Too many breaches = the
    model understates risk; too few = it's needlessly conservative. The Kupiec
    proportion-of-failures test turns "is the breach rate acceptable?" into a
    formal hypothesis test (chi-square, 1 dof, 95% critical value 3.841).
    """
    threshold = np.percentile(port_returns, (1 - confidence) * 100)
    breaches = int((port_returns < threshold).sum())
    n = len(port_returns)
    expected_rate = 1 - confidence
    observed_rate = breaches / n

    # Kupiec likelihood-ratio statistic for proportion of failures.
    p = expected_rate
    x = breaches
    if 0 < x < n:
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - x / n) - x * np.log(x / n)
        )
    else:
        lr = float("nan")
    crit = 3.841  # chi-square(1) at 95%
    passed = bool(np.isnan(lr) or lr <= crit)

    return {
        "breaches": breaches,
        "n": n,
        "expected_breaches": round(expected_rate * n, 1),
        "observed_rate": observed_rate,
        "expected_rate": expected_rate,
        "kupiec_lr": None if np.isnan(lr) else round(float(lr), 2),
        "passed": passed,
    }


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