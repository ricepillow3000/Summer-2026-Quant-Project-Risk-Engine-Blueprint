"""
fits the (beta, vol) state the Risk Topology map moves around.

the map runs two mean-reverting processes over a short horizon:

    d(beta)   = theta_b * (mu_b - beta) dt + sigma_b dW1        (OU on level)
    d(ln vol) = theta_v * (ln mu_v - ln vol) dt + eta_v dW2     (OU on log)

with corr(dW1, dW2) = rho (betas climb when vol spikes) and a leverage
effect lev = corr(return shock, vol shock) < 0 (losses show up with vol).

quant deep dive: nothing here is typed in, it's all fit from history.
an OU process sampled every dt is exactly an AR(1):

    x_{t+1} = c + phi * x_t + eps,   eps ~ N(0, s^2)
    phi = exp(-theta dt)      =>  theta = -ln(phi) / dt
    mu  = c / (1 - phi)
    s^2 = sigma^2 (1 - phi^2) / (2 theta)  =>  sigma = s * sqrt(2 theta / (1 - phi^2))

(standard Vasicek/OU discretization, Glasserman, "Monte Carlo Methods in
Financial Engineering", ch. 3). we build rolling beta and rolling vol from
real daily returns, fit the AR(1) by OLS, then map back to theta/mu/sigma.

honest limits:
- rolling windows overlap, so the series is smooth and phi comes out too high
  (reversion looks slower than it is). it's a terrain, not a forecast.
- that smoothing also made the terrain too narrow, so we measure it and fix
  it: `dispersion_correction` widens both shocks by k (see below). k is
  reported and never shrinks anything.
- "calm" and "stressed" are the base fit times disclosed policy multipliers,
  not separate fits.
"""

import numpy as np
import pandas as pd
from scipy import stats as sstats

from src.covariance import TRADING_DAYS

DT = 1.0 / TRADING_DAYS

# sanity bands so a bad fit (short or flat history) can't draw a crazy terrain.
# anything that lands on a band gets flagged.
CLAMPS = {
    "theta": (0.5, 60.0),        # half-life between ~3 days and ~1.4y
    "sig_b": (0.05, 3.0),        # beta diffusion, per sqrt(year)
    "eta_v": (0.2, 4.0),         # vol-of-vol, per sqrt(year)
    "rho": (-0.95, 0.95),
    "mu_v": (0.05, 0.80),        # long-run vol between 5% and 80%
    "mu_b": (-0.5, 2.0),         # long-run beta the state reverts to
}

# policy multipliers for calm/stressed (disclosed, not data)
STRESS_SHOCK_MULT = 1.4
CALM_SHOCK_MULT = 0.7
STRESS_RHO_ADD = 0.10

MIN_OBS = 120                    # fewer state points than this and we refuse to fit


def rolling_state_series(port_returns: pd.Series, market_returns: pd.Series,
                         beta_window: int = 63, vol_window: int = 21) -> pd.DataFrame:
    """
    (beta, vol) history from real daily returns.

    beta_t = rolling Cov(r_p, r_m) / Var(r_m) over `beta_window` days;
    vol_t  = rolling std of r_p over `vol_window` days, annualized.
    """
    joined = pd.concat([port_returns.rename("p"), market_returns.rename("m")],
                       axis=1).dropna()
    if len(joined) < max(beta_window, vol_window) + MIN_OBS // 2:
        raise ValueError(f"need more overlapping history, got {len(joined)} days")
    beta = joined["p"].rolling(beta_window).cov(joined["m"]) / joined["m"].rolling(beta_window).var()
    vol = joined["p"].rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    out = pd.DataFrame({"beta": beta, "vol": vol}).dropna()
    return out[np.isfinite(out).all(axis=1)]


def fit_ou(series: pd.Series, dt: float = DT) -> dict:
    """
    AR(1) by OLS, mapped exactly to OU. gives theta (reversion speed, per
    year), mu (long-run level), sigma (per sqrt(year)), phi (daily AR slope)
    and the residuals (for the cross-correlations).
    """
    clean = series.dropna()
    x = clean.to_numpy(dtype=float)
    if x.size < MIN_OBS:
        raise ValueError(f"need >= {MIN_OBS} state observations, got {x.size}")
    x0, x1 = x[:-1], x[1:]
    vx = np.var(x0)
    if vx <= 0:
        raise ValueError("state series is constant; nothing to fit")
    phi_raw = float(np.cov(x0, x1, bias=True)[0, 1] / vx)
    # keep it stationary + mean-reverting. smooth rolling series hit the top
    # bound on live data a lot; the caller flags it
    phi = float(np.clip(phi_raw, 0.20, 0.995))
    c = float(np.mean(x1) - phi * np.mean(x0))
    resid = x1 - (c + phi * x0)
    s = float(np.std(resid, ddof=1))
    theta = -np.log(phi) / dt
    mu = c / (1.0 - phi)
    sigma = s * np.sqrt(2.0 * theta / (1.0 - phi * phi))
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma),
            "s": s, "dt": dt, "phi": phi, "phi_raw": phi_raw,
            "resid": pd.Series(resid, index=clean.index[1:])}


# --- dispersion correction -------------------------------------------------
# the terrain came out too narrow: on 11 presets the 68.3% ring only held ~53%
# of where the state really ended up 30 days later. two reasons:
# - overlapping windows smooth the series (phi too close to 1, shocks too small)
# - plug-in error: the forecast uses point estimates of theta/mu/sigma and
#   ignores how unsure they are. even on fake paths drawn from this exact
#   model the ring held only ~47% (test_dispersion_correction_prices_plug_in_estimation_error)
#
# so we measure the fix instead of guessing it. walk the book's history
# forward: fit on data strictly before each date, build the closed-form
# 30-day-ahead distribution, score where the state landed by Mahalanobis
# distance. if the model is sized right, d^2 ~ chi2(2). scaling both shocks by
# k divides every d by k, so the k that fixes coverage is just a quantile ratio:
#
#     k = quantile_p(d_observed) / sqrt(chi2.ppf(p, 2))
#
# limits: k also soaks up mean-forecast error, so read it as a coverage fix,
# not "shocks were k times too small". dates overlap so k is noisy - it's
# clamped, and k < 1 is never applied (we widen, we never narrow).
COVERAGE_P = 0.683               # ring the fix is tuned to hold
COVERAGE_HORIZON = 30            # trading days ahead, same as the map
COVERAGE_STEP = 5               # walk-forward stride
MIN_COVERAGE_DATES = 10          # fewer than this, no correction claimed
K_CLAMP = (1.0, 3.0)


def _end_moments(p: dict, mu_b: float, mu_v: float, b0: float, v0: float,
                 days: int) -> tuple:
    """
    mean + covariance of (beta, ln vol) `days` ahead, using the same exact
    discrete OU step the simulator uses (Glasserman ch. 3):
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
    """OU fits for beta and ln vol, plus their shock correlation."""
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
    walk-forward Mahalanobis distance between the forecast and where the state
    really was `horizon` days later. each fit only sees data before its date.
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
    the k the shocks get widened by, plus the coverage it came from. k = 1.0
    (no fix) when there are too few dates to measure, or the terrain is
    already wide enough.
    """
    d = state_distances(state, horizon, step)
    ring = float(np.sqrt(sstats.chi2.ppf(COVERAGE_P, 2)))
    if len(d) < MIN_COVERAGE_DATES:
        return {"k": 1.0, "n_dates": int(len(d)), "coverage_raw": None,
                "measured": False}
    raw = float((d <= ring).mean())
    k_raw = float(np.quantile(d, COVERAGE_P) / ring)
    k = float(np.clip(k_raw, *K_CLAMP))
    # no "corrected coverage" on purpose: k is built so it'd always read 68.3%,
    # good model or bad. MAP-12 in tests/batch_audit.py checks k on unseen dates.
    return {"k": k, "k_uncapped": k_raw, "n_dates": int(len(d)),
            "coverage_raw": raw, "measured": True}


def _clamp(value: float, key: str, flags: list, band: tuple | None = None) -> float:
    lo, hi = band if band is not None else CLAMPS[key]
    clipped = float(np.clip(value, lo, hi))
    if clipped != value:
        flags.append(f"{key} clamped {value:.3g} -> {clipped:.3g}")
    return clipped


def mu_b_band(beta: pd.Series) -> tuple:
    """
    the long-run beta band, flipped for a net SHORT book.

    CLAMPS["mu_b"] = (-0.5, 2.0) is a long-book band. a short book's betas are
    the mirror image (the caller negates its returns), so the long band would
    drag a -1.24 short toward -0.5 - a fake de-risking drift that makes the
    short look tamer than it is. flipping keeps the same rule ("no absurd
    long-run beta") pointed the way the book leans. long books are untouched.
    """
    lo, hi = CLAMPS["mu_b"]
    return (-hi, -lo) if float(beta.mean()) < 0 else (lo, hi)


def calibrate_state_dynamics(port_returns: pd.Series,
                             market_returns: pd.Series) -> dict:
    """
    full fit: rolling state -> two OU fits -> shock correlations ->
    {calm, base, stress} parameter sets for the map.

    every number comes from `port_returns` / `market_returns`; the only things
    not estimated are the clamp bands and multipliers above.
    """
    state = rolling_state_series(port_returns, market_returns)
    flags: list = []

    fb = fit_ou(state["beta"])
    fv = fit_ou(np.log(state["vol"]))

    # every clamp names what it bound - the footnote lists these as policy
    for key, fit in (("phi_b", fb), ("phi_v", fv)):
        if fit["phi"] != fit["phi_raw"]:
            flags.append(f"{key} clamped {fit['phi_raw']:.6g} -> {fit['phi']:.6g}")
    th_b = _clamp(fb["theta"], "theta_b", flags, band=CLAMPS["theta"])
    th_v = _clamp(fv["theta"], "theta_v", flags, band=CLAMPS["theta"])

    # if theta got clamped, the old sigma no longer gives the measured daily
    # shock. redo sigma from the fitted s under the clamped theta so the
    # one-step variance stays the same: s^2 = sigma^2 (1 - e^{-2 theta dt}) / (2 theta)
    def _sigma_for(theta_c: float, fit: dict) -> float:
        return float(fit["s"] * np.sqrt(
            2.0 * theta_c / (1.0 - np.exp(-2.0 * theta_c * fit["dt"]))))
    sig_b = _clamp(_sigma_for(th_b, fb) if th_b != fb["theta"] else fb["sigma"],
                   "sig_b", flags)
    eta_v = _clamp(_sigma_for(th_v, fv) if th_v != fv["theta"] else fv["sigma"],
                   "eta_v", flags)
    mu_v = _clamp(float(np.exp(fv["mu"])), "mu_v", flags)
    mu_b = _clamp(fb["mu"], "mu_b", flags, band=mu_b_band(state["beta"]))

    # shock correlation, from the AR(1) residuals
    resid = pd.concat([fb["resid"].rename("b"), fv["resid"].rename("v")],
                      axis=1).dropna()
    rho = _clamp(float(resid["b"].corr(resid["v"])), "rho", flags)
    # leverage effect: book return vs same-day vol shock
    lev = _clamp(float(port_returns.reindex(resid.index).corr(resid["v"])), "lev", flags,
                 band=CLAMPS["rho"])

    # widen the shocks by the measured k so the rings hold what they say
    disp = dispersion_correction(state)
    if disp.get("k_uncapped", 1.0) > K_CLAMP[1]:
        flags.append(f"k clamped {disp['k_uncapped']:.3g} -> {disp['k']:.3g}")
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
    # smoke test: fake an OU path with known params, check the fit gets them back
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
