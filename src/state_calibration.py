"""
State-dynamics calibration for the Risk Topology map.

The map simulates the portfolio's (beta, volatility) state over a short
horizon with two mean-reverting processes:

    d(beta)   = theta_b * (mu_b - beta) dt + sigma_b dW1        (OU on level)
    d(ln vol) = theta_v * (ln mu_v - ln vol) dt + eta_v dW2     (OU on log)

with corr(dW1, dW2) = rho (betas drift up when vol spikes) and a leverage
effect lev = corr(return shock, vol shock) < 0 (losses arrive with vol).

Quant deep dive - every parameter is ESTIMATED from observed history, never
typed in. An OU process observed at interval dt is exactly an AR(1):

    x_{t+1} = c + phi * x_t + eps,   eps ~ N(0, s^2)
    phi = exp(-theta dt)      =>  theta = -ln(phi) / dt
    mu  = c / (1 - phi)
    s^2 = sigma^2 (1 - phi^2) / (2 theta)  =>  sigma = s * sqrt(2 theta / (1 - phi^2))

(the standard Vasicek/OU discretization, e.g. Glasserman, "Monte Carlo
Methods in Financial Engineering", ch. 3). We build rolling realized beta
and realized vol series from actual daily returns, fit the AR(1) by OLS,
and map back to continuous-time parameters.

Honest limits, disclosed wherever these numbers surface:
- Rolling windows overlap, which smooths the series and biases phi upward
  (mean reversion looks slower than it is). This is a stylized state model
  for a probability TERRAIN, not a forecasting model.
- That smoothing also drew the terrain too NARROW, which is now measured and
  corrected: `dispersion_correction` walks the book's own history forward and
  widens both shock sizes by the factor that makes the rings hold their
  labelled mass out of sample (see the block above `_clamp`). The factor is
  reported alongside the parameters and never shrinks the terrain.
- The "calm" and "stressed" calibrations are the base estimate with
  disclosed policy multipliers on the shock sizes, not separate estimates.
"""

import numpy as np
import pandas as pd
from scipy import stats as sstats

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS

# Sanity clamps: keep a degenerate fit (short history, flat series) from
# producing an absurd terrain. Values landing ON a clamp are flagged.
CLAMPS = {
    "theta": (0.5, 60.0),        # mean-reversion half-life between ~3 days and ~1.4y
    "sig_b": (0.05, 3.0),        # beta diffusion, per sqrt(year)
    "eta_v": (0.2, 4.0),         # vol-of-vol, per sqrt(year)
    "rho": (-0.95, 0.95),
    "mu_v": (0.05, 0.80),        # long-run vol between 5% and 80%
    "mu_b": (-0.5, 2.0),         # long-run beta the state reverts to
}

# Policy multipliers for the alternative calibrations (disclosed, not data).
STRESS_SHOCK_MULT = 1.4
CALM_SHOCK_MULT = 0.7
STRESS_RHO_ADD = 0.10

MIN_OBS = 120                    # fits on fewer state observations are refused


def rolling_state_series(port_returns: pd.Series, market_returns: pd.Series,
                         beta_window: int = 63, vol_window: int = 21) -> pd.DataFrame:
    """
    Observed (beta, vol) state history from actual daily returns.

    beta_t = rolling Cov(r_p, r_m) / Var(r_m) over `beta_window` days;
    vol_t  = rolling std of r_p over `vol_window` days, annualized.
    """
    joined = pd.concat([port_returns.rename("p"), market_returns.rename("m")],
                       axis=1).dropna()
    if len(joined) < max(beta_window, vol_window) + MIN_OBS // 2:
        raise ValueError(f"need more overlapping history, got {len(joined)} days")
    cov_pm = joined["p"].rolling(beta_window).cov(joined["m"])
    var_m = joined["m"].rolling(beta_window).var()
    beta = cov_pm / var_m
    vol = joined["p"].rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    out = pd.DataFrame({"beta": beta, "vol": vol}).dropna()
    out = out[np.isfinite(out).all(axis=1)]
    return out


def fit_ou(series: pd.Series, dt: float = DT) -> dict:
    """
    Exact AR(1) -> OU mapping by OLS. Returns theta (mean-reversion speed,
    per year), mu (long-run level), sigma (diffusion, per sqrt(year)),
    phi (daily AR coefficient) and the residual series (for cross-corrs).
    """
    x = series.dropna().to_numpy(dtype=float)
    if x.size < MIN_OBS:
        raise ValueError(f"need >= {MIN_OBS} state observations, got {x.size}")
    x0, x1 = x[:-1], x[1:]
    # OLS slope/intercept of x_{t+1} on x_t
    vx = np.var(x0)
    if vx <= 0:
        raise ValueError("state series is constant; nothing to fit")
    phi = float(np.cov(x0, x1, bias=True)[0, 1] / vx)
    phi = float(np.clip(phi, 0.20, 0.995))     # stationary, mean-reverting
    c = float(np.mean(x1) - phi * np.mean(x0))
    resid = x1 - (c + phi * x0)
    s = float(np.std(resid, ddof=1))
    theta = -np.log(phi) / dt
    mu = c / (1.0 - phi)
    sigma = s * np.sqrt(2.0 * theta / (1.0 - phi * phi))
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma),
            "s": s, "dt": dt,
            "phi": phi, "resid": pd.Series(resid, index=series.dropna().index[1:])}


# --- dispersion correction -------------------------------------------------
# Rolling windows overlap, so the observed state series is smoothed: the AR(1)
# slope comes back too close to 1 and the fitted one-step shock too small. The
# terrain the map draws is then NARROWER than what actually happens - measured
# 2026-08-26 on 11 preset universes, where the nominal 68.3% ring covered a
# median of ~53% of realized 30-day-ahead states.
#
# The fix is measured rather than assumed. Walk the book's own history forward:
# refit on observations strictly before each date, form the closed-form
# horizon-ahead distribution, and score where the state actually landed by
# Mahalanobis distance. Under a correctly-sized model d^2 ~ chi2(2). Scaling
# both shock sizes by k scales every d by exactly 1/k, so the k that restores
# coverage is a ratio of quantiles - no search, no fitted parameter:
#
#     k = quantile_p(d_observed) / sqrt(chi2.ppf(p, 2))
#
# TWO effects share the blame, and k prices both:
# - the overlapping-window smoothing above; and
# - plug-in estimation error. The horizon-ahead distribution is built from
#   POINT estimates of theta, mu and sigma, so it ignores their uncertainty.
#   Measured on synthetic state paths drawn from exactly this model - no
#   smoothing anywhere - the nominal 68.3% ring still held only ~47% of
#   realized states, and k came back ~1.2-1.5 (see
#   test_dispersion_correction_prices_plug_in_estimation_error).
#
# Honest limits, disclosed wherever the corrected numbers surface:
# - k absorbs mean-prediction error too (a mis-estimated reversion speed shows
#   up as distance, not as bias), so it is a COVERAGE correction, not a claim
#   that the shocks were literally k times too small.
# - The dates overlap heavily, so k is noisy; it is clamped, and a correction
#   below 1 is never applied - this widens a too-narrow terrain, it does not
#   shrink a conservative one into a tighter story.
COVERAGE_P = 0.683               # ring the correction is calibrated to hold
COVERAGE_HORIZON = 30            # trading days ahead, matches the map's horizon
COVERAGE_STEP = 5               # walk-forward stride
MIN_COVERAGE_DATES = 10          # below this, no correction is claimed
K_CLAMP = (1.0, 3.0)


def _end_moments(p: dict, mu_b: float, mu_v: float, b0: float, v0: float,
                 days: int) -> tuple:
    """
    Mean vector and covariance of (beta, ln vol) `days` ahead under the exact
    discrete OU recursion the simulator uses (Glasserman ch. 3):
        mean = mu + (x0 - mu) phi^n,  var = s^2 (1 - phi^(2n)) / (1 - phi^2),
        cov  = rho s_b s_v (1 - (phi_b phi_v)^n) / (1 - phi_b phi_v).
    """
    phb, phv = np.exp(-p["thB"] * DT), np.exp(-p["thV"] * DT)
    sb = p["sigB"] * np.sqrt((1 - phb ** 2) / (2 * p["thB"]))
    sv = p["etaV"] * np.sqrt((1 - phv ** 2) / (2 * p["thV"]))
    mean = np.array([mu_b + (b0 - mu_b) * phb ** days,
                     np.log(mu_v) + (np.log(v0) - np.log(mu_v)) * phv ** days])
    vb = sb ** 2 * (1 - phb ** (2 * days)) / (1 - phb ** 2)
    vv = sv ** 2 * (1 - phv ** (2 * days)) / (1 - phv ** 2)
    cbv = p["rho"] * sb * sv * (1 - (phb * phv) ** days) / (1 - phb * phv)
    return mean, np.array([[vb, cbv], [cbv, vv]])


def _fit_state(state: pd.DataFrame) -> dict | None:
    """OU parameters for both state coordinates plus their shock correlation."""
    try:
        fb = fit_ou(state["beta"])
        fv = fit_ou(np.log(state["vol"]))
    except ValueError:
        return None
    resid = pd.concat([fb["resid"].rename("b"), fv["resid"].rename("v")],
                      axis=1).dropna()
    rho = float(resid["b"].corr(resid["v"]))
    if not np.isfinite(rho):
        rho = 0.0
    return {"thB": fb["theta"], "sigB": fb["sigma"], "thV": fv["theta"],
            "etaV": fv["sigma"], "rho": float(np.clip(rho, -0.95, 0.95)),
            "muB": fb["mu"], "muV": float(np.exp(fv["mu"]))}


def state_distances(state: pd.DataFrame, horizon: int = COVERAGE_HORIZON,
                    step: int = COVERAGE_STEP) -> np.ndarray:
    """
    Walk-forward Mahalanobis distances between the predicted state distribution
    and what the state actually did `horizon` days later. Each fit uses only
    observations strictly before the date it is judged on.
    """
    b, lv = state["beta"], np.log(state["vol"])
    out = []
    for t in range(MIN_OBS, len(state) - horizon, step):
        p = _fit_state(state.iloc[:t])
        if p is None:
            continue
        mean, cov = _end_moments(p, p["muB"], p["muV"], float(b.iloc[t - 1]),
                                 float(state["vol"].iloc[t - 1]), horizon)
        if np.linalg.det(cov) <= 0:
            continue
        d = np.array([float(b.iloc[t - 1 + horizon]), float(lv.iloc[t - 1 + horizon])]) - mean
        out.append(float(np.sqrt(d @ np.linalg.inv(cov) @ d)))
    return np.array(out)


def dispersion_correction(state: pd.DataFrame, horizon: int = COVERAGE_HORIZON,
                          step: int = COVERAGE_STEP) -> dict:
    """
    The measured factor the shock sizes are widened by, plus the coverage it
    was measured from. `k = 1.0` (no correction) whenever there are too few
    out-of-sample dates to measure one, or the terrain is already wide enough.
    """
    d = state_distances(state, horizon, step)
    ring = float(np.sqrt(sstats.chi2.ppf(COVERAGE_P, 2)))
    if len(d) < MIN_COVERAGE_DATES:
        return {"k": 1.0, "n_dates": int(len(d)), "coverage_raw": None,
                "coverage_corrected": None, "measured": False}
    raw = float((d <= ring).mean())
    k_raw = float(np.quantile(d, COVERAGE_P) / ring)
    k = float(np.clip(k_raw, *K_CLAMP))
    return {"k": k, "k_uncapped": k_raw, "n_dates": int(len(d)),
            "coverage_raw": raw, "coverage_corrected": float((d / k <= ring).mean()),
            "measured": True}


def _clamp(value: float, key: str, flags: list) -> float:
    lo, hi = CLAMPS[key]
    clipped = float(np.clip(value, lo, hi))
    if clipped != value:
        flags.append(f"{key} clamped {value:.3g} -> {clipped:.3g}")
    return clipped


def calibrate_state_dynamics(port_returns: pd.Series,
                             market_returns: pd.Series) -> dict:
    """
    Full calibration: rolling state series -> two OU fits -> shock
    correlations -> {calm, base, stress} parameter sets for the map.

    Every number in the result traces to `port_returns` / `market_returns`;
    the only non-estimated inputs are the disclosed clamp and multiplier
    policies above.
    """
    state = rolling_state_series(port_returns, market_returns)
    flags: list = []

    fb = fit_ou(state["beta"])
    fv = fit_ou(np.log(state["vol"]))

    th_b = _clamp(fb["theta"], "theta", flags)
    th_v = _clamp(fv["theta"], "theta", flags)
    # If theta landed on a clamp, the (theta, sigma) pair from fit_ou no
    # longer reproduces the fitted one-step residual variance - the exact
    # transition the simulator uses would then imply a different daily shock
    # than the one measured. Recompute sigma from the fitted residual s under
    # the CLAMPED theta so the innovation variance survives:
    # s'^2 = sigma^2 (1 - e^{-2 theta dt}) / (2 theta)  ==  s^2.
    def _sigma_for(theta_c: float, fit: dict) -> float:
        return float(fit["s"] * np.sqrt(
            2.0 * theta_c / (1.0 - np.exp(-2.0 * theta_c * fit["dt"]))))
    sig_b = _clamp(_sigma_for(th_b, fb) if th_b != fb["theta"] else fb["sigma"],
                   "sig_b", flags)
    eta_v = _clamp(_sigma_for(th_v, fv) if th_v != fv["theta"] else fv["sigma"],
                   "eta_v", flags)
    mu_v = _clamp(float(np.exp(fv["mu"])), "mu_v", flags)
    mu_b = _clamp(fb["mu"], "mu_b", flags)

    # shock correlations, measured on the AR(1) residuals
    resid = pd.concat([fb["resid"].rename("b"), fv["resid"].rename("v")],
                      axis=1).dropna()
    rho = _clamp(float(resid["b"].corr(resid["v"])), "rho", flags)
    # leverage effect: portfolio return vs same-day vol innovation
    port_aligned = port_returns.reindex(resid.index)
    lev = _clamp(float(port_aligned.corr(resid["v"])), "rho", flags)

    # Widen the shocks by the factor measured against realized history, so the
    # drawn terrain holds the mass its rings are labelled with out of sample.
    disp = dispersion_correction(state)
    if disp["k"] > 1.0:
        sig_b = _clamp(sig_b * disp["k"], "sig_b", flags)
        eta_v = _clamp(eta_v * disp["k"], "eta_v", flags)

    base = {"thB": th_b, "sigB": sig_b, "thV": th_v, "etaV": eta_v,
            "rho": rho, "lev": lev}
    calm = dict(base, sigB=sig_b * CALM_SHOCK_MULT, etaV=eta_v * CALM_SHOCK_MULT)
    stress = dict(base, sigB=sig_b * STRESS_SHOCK_MULT,
                  etaV=eta_v * STRESS_SHOCK_MULT,
                  rho=float(np.clip(rho + STRESS_RHO_ADD, -0.95, 0.95)))

    return {
        "cal": {"calm": calm, "base": base, "stress": stress},
        "muV": mu_v,
        "muB": mu_b,
        "n_obs": int(len(state)),
        "beta_now": float(state["beta"].iloc[-1]),
        "vol_now": float(state["vol"].iloc[-1]),
        "clamp_flags": flags,
        "dispersion": disp,
    }


if __name__ == "__main__":
    # Smoke test: generate a synthetic OU path with known parameters and
    # confirm the fitter recovers them to sane accuracy.
    rng = np.random.default_rng(7)
    theta_true, mu_true, sigma_true = 8.0, 1.0, 0.9
    n = 2000
    x = np.empty(n)
    x[0] = mu_true
    phi = np.exp(-theta_true * DT)
    s = sigma_true * np.sqrt((1 - phi * phi) / (2 * theta_true))
    for i in range(1, n):
        x[i] = mu_true + phi * (x[i - 1] - mu_true) + s * rng.standard_normal()
    fit = fit_ou(pd.Series(x, index=pd.bdate_range("2018-01-01", periods=n)))
    print(f"true theta {theta_true} mu {mu_true} sigma {sigma_true}")
    print(f"fit  theta {fit['theta']:.2f} mu {fit['mu']:.3f} sigma {fit['sigma']:.3f}")
    assert abs(fit["mu"] - mu_true) < 0.12   # ~3 standard errors on this sample
    assert abs(fit["sigma"] - sigma_true) / sigma_true < 0.25
    assert abs(fit["theta"] - theta_true) / theta_true < 0.5
    print("OU recovery smoke test passed")
