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
- The "calm" and "stressed" calibrations are the base estimate with
  disclosed policy multipliers on the shock sizes, not separate estimates.
"""

import numpy as np
import pandas as pd

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
            "phi": phi, "resid": pd.Series(resid, index=series.dropna().index[1:])}


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
    sig_b = _clamp(fb["sigma"], "sig_b", flags)
    eta_v = _clamp(fv["sigma"], "eta_v", flags)
    mu_v = _clamp(float(np.exp(fv["mu"])), "mu_v", flags)

    # shock correlations, measured on the AR(1) residuals
    resid = pd.concat([fb["resid"].rename("b"), fv["resid"].rename("v")],
                      axis=1).dropna()
    rho = _clamp(float(resid["b"].corr(resid["v"])), "rho", flags)
    # leverage effect: portfolio return vs same-day vol innovation
    port_aligned = port_returns.reindex(resid.index)
    lev = _clamp(float(port_aligned.corr(resid["v"])), "rho", flags)

    base = {"thB": th_b, "sigB": sig_b, "thV": th_v, "etaV": eta_v,
            "rho": rho, "lev": lev}
    calm = dict(base, sigB=sig_b * CALM_SHOCK_MULT, etaV=eta_v * CALM_SHOCK_MULT)
    stress = dict(base, sigB=sig_b * STRESS_SHOCK_MULT,
                  etaV=eta_v * STRESS_SHOCK_MULT,
                  rho=float(np.clip(rho + STRESS_RHO_ADD, -0.95, 0.95)))

    return {
        "cal": {"calm": calm, "base": base, "stress": stress},
        "muV": mu_v,
        "n_obs": int(len(state)),
        "beta_now": float(state["beta"].iloc[-1]),
        "vol_now": float(state["vol"].iloc[-1]),
        "clamp_flags": flags,
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
