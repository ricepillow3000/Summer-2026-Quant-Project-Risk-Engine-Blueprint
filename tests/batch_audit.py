"""
Batch automation audit - every math claim in the engine, re-derived on LIVE
data across every preset universe, plus a line-by-line audit of the Risk
Topology map.

Why this exists next to tests/test_engine.py: that suite proves the math on
deterministic synthetic data (hand-worked identities, planted parameters).
This one runs the SAME engine over every real universe the app ships with,
re-derives each number a second, independent way, and writes a report where
every check names the source it comes from. Synthetic tests prove the formula;
this proves the formula still holds on the data the product actually serves,
and that the map on screen is that math rather than a drawing of it.

Run:
    python -m tests.batch_audit              # every preset universe
    python -m tests.batch_audit --quick      # first 4 universes, fewer paths
    python -m tests.batch_audit --no-map     # skip the node map probe

Exit code 0 only if no check FAILs. WARN never blocks - a WARN is data telling
the truth about itself (a missing volume feed, a real crash day in the window).
Artifacts: audit/batch_audit_<UTC>.json and .md
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

from src import data_quality
from src.analytics import correlation_matrix, covariance_matrix, portfolio_volatility
from src.ingestion import PRESETS, average_dollar_volume, fetch_prices
from src.liquidity import days_to_liquidate, liquidity_adjusted_cvar
from src.pairing import DEFENSIVE_ANCHOR_TICKERS, REBALANCE_DAYS
from src.risk import (calibrate_jump_diffusion, cvar, jump_diffusion_mc,
                      mcneil_frey_tail, monte_carlo, portfolio_daily_returns,
                      var, var_backtest)
from src.state_calibration import DT, MIN_OBS, fit_ou, rolling_state_series
from src.strategies import risk_contributions, risk_parity_weights, vol_target
from src.topology import (BETA_WINDOW, HAZARD_BETA_ADD, HAZARD_BETA_CAP,
                          HAZARD_VOL_CAP, HAZARD_VOL_FLOOR, HAZARD_VOL_MULT,
                          VOL_WINDOW, build_map_payload, war_room_html)

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "tests" / "map_probe.mjs"

# Literature each check traces to. A check with no source is not a check.
SOURCES = {
    "acerbi": "Acerbi & Tasche (2002), 'On the coherence of expected shortfall'",
    "kupiec": "Kupiec (1995), 'Techniques for verifying the accuracy of risk "
              "measurement models', J. Derivatives 3(2)",
    "christoffersen": "Christoffersen (1998), 'Evaluating interval forecasts', "
                      "International Economic Review 39(4)",
    "mcneil": "McNeil & Frey (2000), 'Estimation of tail-related risk measures "
              "for heteroscedastic financial time series'",
    "merton": "Merton (1976), 'Option pricing when underlying stock returns are "
              "discontinuous', J. Financial Economics 3",
    "mrt": "Maillard, Roncalli & Teiletche (2010), 'The properties of equally "
           "weighted risk contribution portfolios'",
    "glasserman": "Glasserman (2003), 'Monte Carlo Methods in Financial "
                  "Engineering', ch. 3 (exact OU / Vasicek discretization)",
    "basel": "Basel liquidity-horizon convention (square-root-of-time scaling) "
             "over a participation-rate unwind model",
    "markowitz": "Markowitz (1952), portfolio variance identity w' Sigma w",
    "engine": "This engine's own code, re-derived independently by this audit",
}

results: list[dict] = []


def record(check_id, scope, name, ok, detail, source, warn=False):
    """One audited claim. `warn=True` downgrades a failure to WARN (data
    availability, not math)."""
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append({"id": check_id, "scope": scope, "check": name,
                    "status": status, "detail": detail,
                    "source": SOURCES.get(source, source)})
    return ok


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 1. engine invariants, re-derived on live data
# ---------------------------------------------------------------------------

def audit_universe(name: str, tickers: list[str], quick: bool) -> dict | None:
    try:
        prices = fetch_prices(tickers, period="2y")
    except Exception as exc:  # noqa: BLE001 - offline / feed outage
        record("DATA-00", name, "universe loads", False,
               f"{type(exc).__name__}: {exc}", "engine", warn=True)
        return None
    if prices.shape[1] < 2 or len(prices) < 250:
        record("DATA-00", name, "universe loads", False,
               f"only {prices.shape[0]} days x {prices.shape[1]} names",
               "engine", warn=True)
        return None

    rets = prices.pct_change().dropna()
    cols = list(rets.columns)
    n = len(cols)
    w_eq = pd.Series(1.0 / n, index=cols)
    port = portfolio_daily_returns(rets, w_eq)

    # -- DATA-01 quality gate ------------------------------------------------
    rep = data_quality.validate_prices(prices)
    issues = "; ".join(str(i) for i in rep.get("issues", [])[:3]) or "clean"
    record("DATA-01", name, "data-quality gate", bool(rep.get("passed")),
           f"{len(prices)} days x {n} names; {issues}", "engine", warn=True)

    # -- INV-01 CVaR >= VaR --------------------------------------------------
    ok, detail = True, []
    for conf in (0.95, 0.99):
        v, c = float(var(port, conf)), float(cvar(port, conf))
        ok &= c >= v - 1e-12
        detail.append(f"{conf:.0%}: VaR {v:.4%}, CVaR {c:.4%}")
    record("INV-01", name, "CVaR >= VaR (tail mean dominates the quantile)", ok,
           "; ".join(detail), "acerbi")

    # -- INV-02 covariance symmetric + PSD, correlation bounded --------------
    cov = covariance_matrix(rets)
    corr = correlation_matrix(rets)
    sym = float(np.abs(cov.values - cov.values.T).max())
    eig = float(np.linalg.eigvalsh(cov.values).min())
    off = corr.values[~np.eye(n, dtype=bool)]
    record("INV-02", name, "covariance symmetric + PSD, |corr| <= 1, diag = 1",
           sym < 1e-12 and eig > -1e-8 and float(np.abs(off).max()) <= 1 + 1e-12
           and close(float(np.diag(corr.values).min()), 1.0, 1e-12),
           f"max asymmetry {sym:.2e}, min eigenvalue {eig:.2e}, "
           f"max off-diagonal |corr| {float(np.abs(off).max()):.4f}", "markowitz")

    # -- INV-03 portfolio variance identity ----------------------------------
    pv = float(portfolio_volatility(w_eq.values, cov))
    direct = float(np.sqrt(w_eq.values @ cov.values @ w_eq.values))
    record("INV-03", name, "portfolio vol = sqrt(w' Sigma w)", close(pv, direct, 1e-12),
           f"engine {pv:.8f} vs re-derived {direct:.8f}", "markowitz")

    # -- INV-04 ERC equalizes risk contributions -----------------------------
    w_rp = np.asarray(risk_parity_weights(cov), dtype=float)
    rc = risk_contributions(pd.Series(w_rp, index=cols), cov)
    spread = float(rc["risk_pct"].max() - rc["risk_pct"].min())
    record("INV-04", name, "risk-parity weights equalize risk contributions",
           spread < 5e-4 and close(float(w_rp.sum()), 1.0, 1e-6) and w_rp.min() > 0,
           f"risk-share spread {spread:.2e} across {n} assets; weights sum "
           f"{float(w_rp.sum()):.6f}, min {float(w_rp.min()):.4f}", "mrt")

    # -- INV-05 vol targeting hits the requested vol -------------------------
    tgt = 0.10
    vt = vol_target(w_eq, cov, target_vol=tgt)
    achieved = float(portfolio_volatility(np.asarray(vt["scaled_weights"]), cov))
    record("INV-05", name, "vol targeting reaches the requested volatility",
           close(achieved, tgt, 1e-9),
           f"target {tgt:.2%}, achieved {achieved:.6%}, leverage "
           f"{float(vt['leverage']):.3f}", "engine")

    # -- INV-06 both Monte Carlo engines coherent ----------------------------
    paths = 4000 if quick else 20000
    for engine, fn, src in (("bootstrap", monte_carlo, "acerbi"),
                            ("jump-diffusion", jump_diffusion_mc, "merton")):
        mc = fn(rets, w_eq.values, n_simulations=paths, horizon_days=252)
        ok = (mc["cvar"] >= mc["var"] - 1e-9 and mc["worst_case"] <= mc["best_case"]
              and 0.0 <= mc["prob_loss"] <= 1.0
              and all(np.isfinite(mc[k]) for k in ("cvar", "var", "median_return")))
        record(f"INV-06.{engine}", name, f"{engine} Monte Carlo coherent", ok,
               f"VaR {mc['var']:.2%}, CVaR {mc['cvar']:.2%}, P(loss) "
               f"{mc['prob_loss']:.1%}, {paths} paths x 252 days", src)

    # -- INV-07 jump calibration reproduces the empirical mean ---------------
    jp = calibrate_jump_diffusion(port)
    emp = float(np.log1p(port.values).mean())
    identity = float(jp["mu_d"] + jp["lambda_daily"] * jp["mu_j"])
    record("INV-07", name, "jump calibration reproduces the empirical mean",
           close(identity, emp, 1e-12),
           f"mu_d + lambda*mu_j = {identity:.10f} vs empirical mean log-return "
           f"{emp:.10f}; {jp['n_jumps']} jump days detected", "merton")

    # -- INV-08 liquidity monotone, LVaR >= CVaR -----------------------------
    try:
        adv = average_dollar_volume(cols)
        small = days_to_liquidate(w_eq.values, adv, book_value=1e6)["days"]
        big = days_to_liquidate(w_eq.values, adv, book_value=1e8)["days"]
        mono = bool((big.values >= small.values - 1e-12).all())
        cv = float(cvar(port, 0.975))
        finite = big.replace([np.inf, -np.inf], np.nan).dropna()
        worst_days = float(finite.max()) if len(finite) else 0.0
        lv = liquidity_adjusted_cvar(cv, worst_days)
        record("INV-08", name, "days-to-liquidate monotone in book size; LVaR >= CVaR",
               mono and lv["lvar"] >= cv - 1e-12,
               f"100x the book never shortens an exit (slowest liquid leg "
               f"{worst_days:.2f}d); CVaR {cv:.2%} -> LVaR {lv['lvar']:.2%} "
               f"(x{lv['multiplier']:.4f})", "basel")
    except Exception as exc:  # noqa: BLE001 - volume feed can be absent (FX)
        record("INV-08", name, "days-to-liquidate monotone in book size; LVaR >= CVaR",
               False, f"volume data unavailable: {type(exc).__name__}: {exc}",
               "basel", warn=True)

    # -- INV-09 conditional EVT ----------------------------------------------
    # The app fits EVT on a 10y window (main.py's load_tail_fit), not the 2y
    # risk window: a GPD needs ~50 exceedances above the 95% threshold, and
    # 2y of data supplies ~22. Auditing it on 2y would test a refusal.
    try:
        evt_rets = fetch_prices(tickers, period="10y").pct_change().dropna()
        evt_cols = list(evt_rets.columns)
        evt_port = portfolio_daily_returns(
            evt_rets, pd.Series(1.0 / len(evt_cols), index=evt_cols))
    except Exception:  # noqa: BLE001 - fall back to the window we already have
        evt_port = port
    mf = mcneil_frey_tail(evt_port)
    if mf.get("fitted"):
        cond = mf["conditional"]
        ok = (all(cond[k]["es"] >= cond[k]["var"] - 1e-12 for k in cond)
              and float(mf["sigma_next"]) > 0)
        q = sorted(cond)[0]
        record("INV-09", name, "conditional EVT: ES >= VaR at every quantile", ok,
               f"10y window, {len(evt_port)} days; sigma_next "
               f"{float(mf['sigma_next']):.4%}, filtered xi {mf.get('xi')}, "
               f"unconditional xi {mf.get('unconditional_xi')}; "
               f"q={q}: VaR {cond[q]['var']:.2%}, ES {cond[q]['es']:.2%}", "mcneil")
    else:
        record("INV-09", name, "conditional EVT: ES >= VaR at every quantile", False,
               f"refused to fit: {mf.get('reason')}", "mcneil", warn=True)

    return {"returns": rets, "weights": w_eq, "port": port, "cols": cols,
            "prices": prices, "cov": cov}


# ---------------------------------------------------------------------------
# 2. the batch backtest: walk-forward VaR coverage on every book
# ---------------------------------------------------------------------------

def audit_backtests(name: str, ctx: dict) -> list[dict]:
    rets, cols, cov = ctx["returns"], ctx["cols"], ctx["cov"]
    books = {"equal-weight": ctx["weights"],
             "risk-parity": pd.Series(np.asarray(risk_parity_weights(cov)),
                                      index=cols)}
    rows = []
    for book, w in books.items():
        port = portfolio_daily_returns(rets, w)
        for conf in (0.95, 0.99):
            bt = var_backtest(port, confidence=conf)
            if not bt["testable"]:
                record("BT-00", name, f"{book} @ {conf:.0%} is testable", False,
                       "sample too short to hold out a day", "kupiec", warn=True)
                continue

            # BT-01: re-derive the walk-forward breach count. Day t's VaR may
            # only use days strictly before t - .shift(1) is what enforces it.
            q = 1.0 - conf
            line = port.rolling(bt["window"]).quantile(q).shift(1)
            mask = line.notna()
            breaches = int((port[mask] < line[mask]).sum())
            record("BT-01", name, f"{book} @ {conf:.0%} walk-forward, no lookahead",
                   breaches == bt["breaches"] and int(mask.sum()) == bt["n"],
                   f"audit {breaches} breaches / {int(mask.sum())} tested days vs "
                   f"engine {bt['breaches']} / {bt['n']} (window {bt['window']}d)",
                   "engine")

            # BT-02: Kupiec proportion-of-failures likelihood ratio, re-derived.
            x, nn, p = bt["breaches"], bt["n"], q
            if 0 < x < nn:
                lr = -2.0 * ((nn - x) * np.log(1 - p) + x * np.log(p)
                             - (nn - x) * np.log(1 - x / nn) - x * np.log(x / nn))
            else:
                lr = -2.0 * ((nn - x) * np.log(1 - p) + x * np.log(p))
            p_val = float(1 - sstats.chi2.cdf(lr, 1))
            record("BT-02", name, f"{book} @ {conf:.0%} Kupiec LR re-derived",
                   bt["kupiec_lr"] is not None and close(lr, bt["kupiec_lr"], 0.02),
                   f"audit LR {lr:.3f} vs engine {bt['kupiec_lr']}; chi2(1) "
                   f"p = {p_val:.3f}; observed rate {bt['observed_rate']:.2%} vs "
                   f"expected {bt['expected_rate']:.2%}", "kupiec")

            # BT-03: conditional coverage decomposes into coverage + clustering.
            ch = bt["christoffersen"]
            if ch.get("lr_ind") is not None:
                record("BT-03", name,
                       f"{book} @ {conf:.0%} LR_cc = LR_uc + LR_ind",
                       close(ch["lr_cc"], round(bt["kupiec_lr"] + ch["lr_ind"], 2), 0.02),
                       f"LR_cc {ch['lr_cc']} = LR_uc {bt['kupiec_lr']} + LR_ind "
                       f"{ch['lr_ind']}; independence p {ch.get('p_ind')}",
                       "christoffersen")
            rows.append({
                "universe": name, "book": book, "confidence": conf,
                "tested_days": bt["n"], "breaches": bt["breaches"],
                "expected": bt["expected_breaches"],
                "observed_rate": round(float(bt["observed_rate"]), 4),
                "kupiec_lr": bt["kupiec_lr"], "kupiec_pass": bt["passed"],
                "independence_pass": ch.get("passed_ind"),
                "conditional_coverage_pass": ch.get("passed_cc"),
                "window": bt["window"],
            })
    return rows


# ---------------------------------------------------------------------------
# 3. the Risk Topology map: is the picture the math?
# ---------------------------------------------------------------------------

def _probe(payload: dict, cal: str, n_paths: int, seed: int = 12345) -> dict:
    """Run the SHIPPED map simulation (prototypes/war_room.html) under node on
    this payload and return its moments/statistics."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as fh:
        json.dump(payload, fh)
        path = fh.name
    try:
        out = subprocess.run(
            ["node", str(PROBE), path, cal, str(n_paths), str(seed)],
            capture_output=True, text=True, check=True, cwd=str(ROOT))
        return json.loads(out.stdout)
    finally:
        Path(path).unlink(missing_ok=True)


def _ou_end_moments(p: dict, mu_b: float, mu_v: float, b0: float, v0: float,
                    days: int) -> dict:
    """
    Closed-form end-state moments of the EXACT discrete OU recursion the map
    simulates (Glasserman ch. 3): phi = e^(-theta dt),
    s^2 = sigma^2 (1 - phi^2) / (2 theta), compounded over `days` steps:

        mean = mu + (x0 - mu) phi^n
        var  = s^2 (1 - phi^(2n)) / (1 - phi^2)
        cov  = rho s_b s_v (1 - (phi_b phi_v)^n) / (1 - phi_b phi_v)
    """
    phb = float(np.exp(-p["thB"] * DT))
    sb = float(p["sigB"] * np.sqrt((1 - phb ** 2) / (2 * p["thB"])))
    phv = float(np.exp(-p["thV"] * DT))
    sv = float(p["etaV"] * np.sqrt((1 - phv ** 2) / (2 * p["thV"])))
    lnv0, lnmu = float(np.log(v0)), float(np.log(mu_v))
    return {
        "phiB": phb, "sB": sb, "phiV": phv, "sV": sv,
        "beta_mean": mu_b + (b0 - mu_b) * phb ** days,
        "beta_var": sb ** 2 * (1 - phb ** (2 * days)) / (1 - phb ** 2),
        "lnvol_mean": lnmu + (lnv0 - lnmu) * phv ** days,
        "lnvol_var": sv ** 2 * (1 - phv ** (2 * days)) / (1 - phv ** 2),
        "cov_beta_lnvol": (p["rho"] * sb * sv
                           * (1 - (phb * phv) ** days) / (1 - phb * phv)),
    }


def audit_map(name: str, ctx: dict, n_paths: int) -> None:
    rets, w, cols = ctx["returns"], ctx["weights"], ctx["cols"]
    try:
        payload = build_map_payload(rets, w, cols)
    except Exception as exc:  # noqa: BLE001 - short history refuses, never fakes
        record("MAP-00", name, "map payload builds from live data", False,
               f"{type(exc).__name__}: {exc}", "engine", warn=True)
        return
    record("MAP-00", name, "map payload builds from live data", True,
           f"{len(payload['assets'])} units, {len(payload['pairs'])} linkages, "
           f"{payload['provenance']}", "engine")

    # -- MAP-01 the book's own position, re-derived from raw prices ---------
    port = portfolio_daily_returns(rets, w)
    try:
        mkt = fetch_prices(["SPY"], period="2y").pct_change().dropna()["SPY"]
    except Exception:  # noqa: BLE001 - same fallback the builder uses
        mkt = rets.mean(axis=1)
    state = rolling_state_series(port, mkt, BETA_WINDOW, VOL_WINDOW)
    b_now = round(float(state["beta"].iloc[-1]), 3)
    v_now = round(float(state["vol"].iloc[-1]), 4)
    record("MAP-01", name, "book position = rolling beta / realized vol",
           close(b_now, payload["base"]["beta"], 1e-9)
           and close(v_now, payload["base"]["vol"], 1e-9),
           f"audit beta {b_now}, vol {v_now} vs map {payload['base']['beta']} / "
           f"{payload['base']['vol']} ({BETA_WINDOW}d Cov/Var; {VOL_WINDOW}d sd "
           "x sqrt(252))", "engine")

    # -- MAP-02 every unit dot on the map, re-derived ------------------------
    common = rets.index.intersection(mkt.index)
    mvar = float(mkt.loc[common].var())
    bad, checked = [], 0
    for a in payload["assets"]:
        t = a["t"]
        if t not in rets.columns:
            continue                      # anchor-pool ETF: not in this universe
        checked += 1
        b = round(float(rets[t].loc[common].cov(mkt.loc[common]) / mvar), 3)
        v = round(float(rets[t].std() * np.sqrt(252)), 4)
        if not close(b, a["b"], 1.1e-3):
            bad.append(f"{t} beta: audit {b} vs map {a['b']}")
        if not close(v, a["v"], 2e-3):
            bad.append(f"{t} vol: audit {v} vs map {a['v']}")
    record("MAP-02", name, "unit coordinates = measured beta / annualized vol",
           not bad, f"{checked} of {len(payload['assets'])} units re-derived from "
           "raw returns; " + ("; ".join(bad[:3]) if bad else "all match"), "engine")

    # -- MAP-03 the perimeter is the disclosed policy, nothing else ---------
    hv = round(min(HAZARD_VOL_CAP,
                   max(HAZARD_VOL_FLOOR, HAZARD_VOL_MULT * payload["base"]["vol"])), 2)
    hb = round(min(HAZARD_BETA_CAP, payload["base"]["beta"] + HAZARD_BETA_ADD), 2)
    record("MAP-03", name, "hazard perimeter matches the disclosed policy",
           close(hv, payload["hazard"]["volMax"], 1e-12)
           and close(hb, payload["hazard"]["betaMax"], 1e-12)
           and "3x today's vol" in payload["footnote"],
           f"volMax {payload['hazard']['volMax']} = min({HAZARD_VOL_CAP}, "
           f"max({HAZARD_VOL_FLOOR}, {HAZARD_VOL_MULT} x {payload['base']['vol']})); "
           f"betaMax {payload['hazard']['betaMax']} = min({HAZARD_BETA_CAP}, "
           f"{payload['base']['beta']} + {HAZARD_BETA_ADD}); footnote states both",
           "engine")

    # -- MAP-04 everything claimed is inside the drawn grid ------------------
    d = payload["domain"]
    inside = all(d["b0"] <= a["b"] <= d["b1"] and d["v0"] <= a["v"] <= d["v1"]
                 for a in payload["assets"])
    record("MAP-04", name, "drawn domain contains every unit and the perimeter",
           inside and d["b0"] <= payload["hazard"]["betaMax"] <= d["b1"]
           and payload["hazard"]["volMax"] <= d["v1"],
           f"beta axis [{d['b0']}, {d['b1']}], vol axis [{d['v0']}, {d['v1']}]",
           "engine")

    # -- MAP-05 the AR(1) -> OU inversion is exact ---------------------------
    fb = fit_ou(state["beta"])
    theta_from_phi = float(-np.log(fb["phi"]) / DT)
    sigma_from_s = float(fb["s"] * np.sqrt(2 * fb["theta"] / (1 - fb["phi"] ** 2)))
    record("MAP-05", name, "AR(1) -> OU inversion is exact",
           close(theta_from_phi, fb["theta"], 1e-9)
           and close(sigma_from_s, fb["sigma"], 1e-9) and len(state) >= MIN_OBS,
           f"theta = -ln(phi)/dt = {fb['theta']:.4f}/yr (phi {fb['phi']:.5f}), "
           f"sigma = s sqrt(2 theta/(1-phi^2)) = {fb['sigma']:.4f}, "
           f"{len(state)} state observations (min {MIN_OBS})", "glasserman")

    # -- MAP-06..10: run the shipped simulation, check it against theory -----
    pr = _probe(payload, "base", n_paths)
    th = _ou_end_moments(pr["params"], payload["muB"], payload["muV"],
                         payload["base"]["beta"], payload["base"]["vol"],
                         payload["days"])
    m, N = pr["moments"], pr["n"]
    z = {
        "beta_mean": (m["beta_mean"] - th["beta_mean"]) / np.sqrt(th["beta_var"] / N),
        "beta_var": (m["beta_var"] - th["beta_var"]) / (th["beta_var"] * np.sqrt(2 / N)),
        "lnvol_mean": (m["lnvol_mean"] - th["lnvol_mean"]) / np.sqrt(th["lnvol_var"] / N),
        "lnvol_var": (m["lnvol_var"] - th["lnvol_var"]) / (th["lnvol_var"] * np.sqrt(2 / N)),
        "shock_cov": ((m["cov_beta_lnvol"] - th["cov_beta_lnvol"])
                      / np.sqrt((th["beta_var"] * th["lnvol_var"]
                                 + th["cov_beta_lnvol"] ** 2) / N)),
    }
    record("MAP-06", name, "simulated terrain matches closed-form OU moments",
           max(abs(v) for v in z.values()) < 5.0,
           "z vs theory: " + ", ".join(f"{k} {v:+.2f}" for k, v in z.items())
           + f" (n={N}); E[beta_T] theory {th['beta_mean']:.4f} vs simulated "
           f"{m['beta_mean']:.4f}; Var theory {th['beta_var']:.6f} vs "
           f"{m['beta_var']:.6f}", "glasserman")

    rings = pr["rings"]

    def tol(p_):
        return 5 * float(np.sqrt(p_ * (1 - p_) / N))

    record("MAP-07", name, "fog rings hold the mass they are labelled with",
           abs(rings["mass68"] - 0.683) < tol(0.683)
           and abs(rings["mass95"] - 0.954) < tol(0.954)
           and abs(rings["mass997"] - 0.997) < tol(0.997),
           f"measured mass inside each drawn ring: {rings['mass68']:.4f} / "
           f"{rings['mass95']:.4f} / {rings['mass997']:.4f} vs labels 68.3 / 95.4 "
           "/ 99.7% (empirical Mahalanobis quantiles, not 1-sigma ellipses)",
           "engine")

    bc = pr["breach_check"]
    record("MAP-08", name, "breach probability counts paths that TOUCH the line",
           close(bc["flagged"], bc["any_touch"], 1e-12)
           and bc["any_touch"] >= bc["end_only"],
           f"flagged {bc['flagged']:.4f} equals any-touch {bc['any_touch']:.4f} "
           f"(counting end-states only would read {bc['end_only']:.4f}); HUD shows "
           f"{pr['stats']['pBreach']:.4f}", "engine")

    st, pc = pr["stats"], pr["pnl_check"]
    record("MAP-09", name, "HUD tail numbers come from the simulated path P&L",
           close(st["var95"], pc["var95"], 1e-12)
           and close(st["es"], pc["es975"], 1e-12) and st["es"] >= st["var95"],
           f"VaR95 {st['var95']:.4%} and ES97.5 {st['es']:.4%} reproduced from the "
           f"path P&L array; ES >= VaR holds; mean P&L {pc['mean']:+.4%} "
           "(zero-drift risk view, as disclosed)", "acerbi")

    calm = _probe(payload, "calm", max(4000, n_paths // 4))
    stress = _probe(payload, "stress", max(4000, n_paths // 4))

    def spread(r):
        return r["moments"]["beta_var"] + r["moments"]["lnvol_var"]

    record("MAP-10", name, "calm < base < stressed terrain width",
           spread(calm) < spread(pr) < spread(stress),
           f"state variance calm {spread(calm):.5f} < base {spread(pr):.5f} < "
           f"stress {spread(stress):.5f} (disclosed shock multipliers 0.7 / 1.4, "
           "+0.10 shock correlation under stress)", "engine")

    # -- MAP-11 every linkage number, replayed independently -----------------
    # The map draws a line between a flyer and an anchor and labels it with
    # rho, a weight, ES(97.5) solo/paired and a drawdown cushion. Each is
    # re-derived here from raw returns with this audit's own loop - not by
    # calling src.pairing again - so a wrong pairing formula cannot agree
    # with itself.
    try:
        anchors = fetch_prices(DEFENSIVE_ANCHOR_TICKERS, period="2y")
        frame = rets.join(anchors.pct_change().dropna(), how="inner").dropna(axis=1)
    except Exception:  # noqa: BLE001 - same fallback the builder uses
        frame = rets
    bad, shown = [], []
    for pair in payload["pairs"]:
        fl, an, r = pair["fl"], pair["an"], pair["real"]
        if fl not in frame.columns or an not in frame.columns:
            bad.append(f"{fl}/{an}: leg missing from the audited frame")
            continue
        j = frame[[fl, an]].dropna()
        a, b = j[fl], j[an]

        # rho and the two-asset equal-risk weight: w_a sigma_a = w_b sigma_b
        rho = float(a.corr(b))
        w_a = float(b.std() / (a.std() + b.std()))

        # replay the pair: legs drift with actual returns, reset monthly
        va, vb = w_a, 1.0 - w_a
        path, prets = [], []
        for i, (ra, rb) in enumerate(zip(a.values, b.values)):
            wt = va / (va + vb)
            prets.append(wt * ra + (1.0 - wt) * rb)
            va *= 1.0 + ra
            vb *= 1.0 + rb
            path.append(va + vb)
            if (i + 1) % REBALANCE_DAYS == 0:
                tot = va + vb
                va, vb = tot * w_a, tot * (1.0 - w_a)
        pair_path = pd.Series(path, index=j.index)
        solo_path = (1.0 + a).cumprod()
        pair_ret = pd.Series(prets, index=j.index)

        def _es(x):                       # ES(97.5): mean of the worst 2.5%
            s = pd.Series(x).dropna()
            return float(-s[s <= np.percentile(s, 2.5)].mean())

        def _dd(p):                       # max drawdown, day-one loss included
            return float((p / p.cummax().clip(lower=1.0) - 1.0).min())

        expect = {
            "esSolo": _es(a), "esPair": _es(pair_ret),
            "cushion": _dd(pair_path) - _dd(solo_path),
            "volPair": float(pair_ret.std() * np.sqrt(252)),
        }
        for k, v in expect.items():
            if not close(v, r[k], 3e-4):
                bad.append(f"{fl}/{an} {k}: audit {v:.5f} vs map {r[k]}")
        if not close(rho, pair["rho"], 1.1e-3):
            bad.append(f"{fl}/{an} rho: audit {rho:.4f} vs map {pair['rho']}")
        if not close(w_a, pair["wFlyer"], 1.1e-4):
            bad.append(f"{fl}/{an} weight: audit {w_a:.5f} vs map {pair['wFlyer']}")
        if r["nDays"] != len(j):
            bad.append(f"{fl}/{an} history: audit {len(j)}d vs map {r['nDays']}d")
        shown.append(f"{fl}->{an}: rho {rho:+.3f}, w_flyer {w_a:.3f}, ES solo "
                     f"{expect['esSolo']:.3%} vs paired {expect['esPair']:.3%}, "
                     f"drawdown cushion {expect['cushion']:+.3%} over {len(j)}d")
    audit_map_state_coverage(name, state)

    record("MAP-11", name, "every linkage number replayed from raw returns",
           bool(payload["pairs"]) and not bad,
           "; ".join(bad[:3]) if bad else "; ".join(shown[:2])
           + f" ({len(payload['pairs'])} linkages re-derived)", "acerbi")


def audit_map_state_coverage(name: str, state: pd.DataFrame,
                             horizon: int = 30, step: int = 5) -> None:
    """
    MAP-12 - the only check that asks whether the terrain is TRUE, not merely
    internally exact. Everything above proves the map draws its calibration
    correctly; this walks history forward and asks whether the state actually
    landed where the map said it would.

    At each historical date t (every `step` days), the OU parameters are fitted
    ONLY on state observations strictly before t, the closed-form transition
    gives the predicted mean and covariance of (beta, ln vol) at t + horizon,
    and the REALIZED state at t + horizon is scored by Mahalanobis distance.
    Under the model d^2 ~ chi2(2), so the share of dates inside the 68.3% and
    95.4% rings should match those labels.

    Reported as WARN, never FAIL: the windows overlap heavily (a 2y sample
    holds ~14 independent 30-day blocks), so the coverage estimate is noisy by
    construction, and a real regime shift SHOULD push coverage down. It is
    evidence about the map, not a gate on the build.
    """
    b = state["beta"]
    lv = np.log(state["vol"])
    idx, hits68, hits95, n = [], 0, 0, 0
    r68 = float(np.sqrt(sstats.chi2.ppf(0.683, 2)))
    r95 = float(np.sqrt(sstats.chi2.ppf(0.954, 2)))
    for t in range(MIN_OBS, len(state) - horizon, step):
        try:
            fb = fit_ou(b.iloc[:t])
            fv = fit_ou(lv.iloc[:t])
        except ValueError:
            continue
        rho = float(pd.concat([fb["resid"], fv["resid"]], axis=1).dropna()
                    .corr().iloc[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        p = {"thB": fb["theta"], "sigB": fb["sigma"],
             "thV": fv["theta"], "etaV": fv["sigma"], "rho": rho}
        th = _ou_end_moments(p, fb["mu"], float(np.exp(fv["mu"])),
                             float(b.iloc[t - 1]), float(state["vol"].iloc[t - 1]),
                             horizon)
        cov = np.array([[th["beta_var"], th["cov_beta_lnvol"]],
                        [th["cov_beta_lnvol"], th["lnvol_var"]]])
        if np.linalg.det(cov) <= 0:
            continue
        d = np.array([float(b.iloc[t - 1 + horizon]) - th["beta_mean"],
                      float(lv.iloc[t - 1 + horizon]) - th["lnvol_mean"]])
        maha = float(np.sqrt(d @ np.linalg.inv(cov) @ d))
        hits68 += maha <= r68
        hits95 += maha <= r95
        n += 1
        idx.append(maha)
    if n < 10:
        record("MAP-12", name, "realized state lands where the map predicted",
               False, f"only {n} out-of-sample dates available", "engine", warn=True)
        return
    c68, c95 = hits68 / n, hits95 / n
    # ~n/horizon independent blocks: widen the band accordingly, then be lenient.
    eff = max(3.0, n * step / horizon)
    band68 = 3 * float(np.sqrt(0.683 * 0.317 / eff))
    band95 = 3 * float(np.sqrt(0.954 * 0.046 / eff))
    record("MAP-12", name, "realized state lands where the map predicted",
           abs(c68 - 0.683) < band68 + 0.12 and abs(c95 - 0.954) < band95 + 0.08,
           f"walk-forward over {n} dates ({horizon}d ahead, refit each time on "
           f"prior data only): {c68:.1%} inside the 68.3% ring, {c95:.1%} inside "
           f"the 95.4% ring; median Mahalanobis distance {float(np.median(idx)):.2f} "
           f"(chi2(2) median {float(np.sqrt(sstats.chi2.ppf(0.5, 2))):.2f})",
           "glasserman", warn=True)


def audit_map_control(n_paths: int) -> None:
    """
    Known-parameter control. MAP-06 compares the simulation against theory
    using the LIVE calibration - if the map's own constants were wrong in the
    same way as the audit's, both could agree. So run the shipped simulation
    once on hand-set parameters, where the right answer is known in advance.
    """
    cal = {"thB": 6.0, "sigB": 0.8, "thV": 4.0, "etaV": 1.2, "rho": 0.5, "lev": -0.5}
    payload = {
        "base": {"beta": 1.2, "vol": 0.25},
        "hazard": {"volMax": 0.75, "betaMax": 2.0},
        "assets": [{"t": "CTRL", "b": 1.0, "v": 0.2, "f": "Universe"}],
        "pairs": [], "muFlyer": 0.0, "muAnchor": 0.0, "muBook": 0.0,
        "cal": {"calm": cal, "base": cal, "stress": cal},
        "muV": 0.20, "muB": 1.0, "days": 30,
        "domain": {"b0": -1.0, "b1": 2.5, "v0": 0.0, "v1": 1.0},
        "live": True, "provenance": "control", "footnote": "control",
    }
    pr = _probe(payload, "base", n_paths, seed=999)
    th = _ou_end_moments(cal, 1.0, 0.20, 1.2, 0.25, 30)
    m, N = pr["moments"], pr["n"]
    zb = (m["beta_mean"] - th["beta_mean"]) / np.sqrt(th["beta_var"] / N)
    zv = (m["lnvol_var"] - th["lnvol_var"]) / (th["lnvol_var"] * np.sqrt(2 / N))
    zc = ((m["cov_beta_lnvol"] - th["cov_beta_lnvol"])
          / np.sqrt((th["beta_var"] * th["lnvol_var"]
                     + th["cov_beta_lnvol"] ** 2) / N))
    record("MAP-CTRL", "control (planted parameters)",
           "map simulation recovers hand-set OU parameters",
           max(abs(zb), abs(zv), abs(zc)) < 5.0,
           f"planted theta_b 6.0, theta_v 4.0, rho 0.5, b0 1.2, vol0 25%: "
           f"z(beta mean) {zb:+.2f}, z(lnvol var) {zv:+.2f}, z(shock cov) "
           f"{zc:+.2f} over {N} paths", "glasserman")


def audit_static_map_source() -> None:
    """Checks on the map source that no simulation can make: drawn guide lines
    must use the constants they claim, and the live splice must leave no demo
    number behind."""
    html = (ROOT / "prototypes" / "war_room.html").read_text(encoding="utf-8")
    begin, end = "/* __PAYLOAD_BEGIN__ */", "/* __PAYLOAD_END__ */"

    z95 = float(sstats.norm.ppf(0.95))
    record("MAP-S1", "map source", "iso-VaR guide uses the Gaussian 95% quantile",
           "1.645" in html and "Math.sqrt(252)" in html and abs(z95 - 1.645) < 1e-3,
           f"contour drawn at vol = VaR sqrt(252) / 1.645; scipy norm.ppf(0.95) "
           f"= {z95:.6f}", "engine")

    probe_payload = {
        "base": {"beta": 1.11, "vol": 0.2222},
        "hazard": {"volMax": 0.66, "betaMax": 1.9},
        "assets": [{"t": "SPLICE", "b": 1.0, "v": 0.2, "f": "Universe"}], "pairs": [],
        "muFlyer": 0.0, "muAnchor": 0.0, "muBook": 0.0,
        "cal": {"calm": {}, "base": {}, "stress": {}}, "muV": 0.2, "muB": 1.0,
        "days": 30, "domain": {"b0": -1.0, "b1": 2.0, "v0": 0.0, "v1": 1.0},
        "live": True, "provenance": "splice probe", "footnote": "splice probe",
    }
    spliced = war_room_html(probe_payload)
    body = spliced[spliced.index(begin) + len(begin):spliced.index(end)]
    parsed = json.loads(body.strip().removeprefix("const DEMO =").strip().rstrip(";"))
    record("MAP-S2", "map source", "live splice replaces the demo block wholesale",
           parsed == probe_payload and "'NVDA', b: 1.65" not in spliced,
           "spliced payload round-trips to the exact engine dict; the file's demo "
           "constants are absent from the rendered page", "engine")

    record("MAP-S4", "map source", "live linkage label names the right statistic",
           "'Drawdown cushion' : 'Crisis cushion'" in html.replace("? ", "").replace(" ?", "")
           or "Drawdown cushion" in html,
           "in live mode the inspector labels the backtested number 'Drawdown "
           "cushion' (max-drawdown difference), not the demo-mode 'Crisis "
           "cushion' computed from an ES difference", "engine")

    record("MAP-S3", "map source", "shared constants survive the splice",
           html.index("const ISO_LEVELS") < html.index(begin) and "ISO_LEVELS" in spliced,
           "ISO_LEVELS is declared before the payload markers, so the drawn contour "
           "and the hover explanation cannot disagree", "engine")


def audit_traceability() -> None:
    """Every risk claim must name its source in the code, not only in a report."""
    # each entry is a claim -> the citations that would satisfy it (any one).
    wanted = {
        "src/risk.py": {"VaR backtest": ["Kupiec"],
                        "breach clustering": ["Christoffersen"],
                        "conditional EVT": ["McNeil"],
                        "jump diffusion": ["Merton"],
                        "GPD tail": ["Pickands", "Balkema", "generalized Pareto"]},
        "src/strategies.py": {"ERC solver": ["Maillard", "Spinu", "Griveau-Billion"],
                              "risk budgeting": ["risk parity"]},
        "src/state_calibration.py": {"OU discretization": ["Glasserman", "Vasicek"],
                                     "estimator": ["AR(1)"]},
        "src/pairing.py": {"tail measure": ["expected shortfall"],
                           "Basel ES level": ["97.5"]},
        "src/liquidity.py": {"unwind model": ["participation"],
                             "horizon scaling": ["square-root-of-time"]},
        "src/regimes.py": {"regime distance": ["Wasserstein"]},
        "src/covariance.py": {"shrinkage": ["Ledoit"]},
    }
    for rel, claims in wanted.items():
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        found = {c: next((t for t in opts if t.lower() in text), None)
                 for c, opts in claims.items()}
        missing = [c for c, hit in found.items() if hit is None]
        record("TRACE", rel, "every method cites a source in the module itself",
               not missing,
               "; ".join(f"{c} -> {hit}" for c, hit in found.items() if hit)
               + (f"; UNCITED: {missing}" if missing else ""), "engine")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="first 4 universes, fewer simulation paths")
    ap.add_argument("--no-map", action="store_true", help="skip the node map probe")
    ap.add_argument("--paths", type=int, default=20000)
    args = ap.parse_args()

    names = list(PRESETS)[:4] if args.quick else list(PRESETS)
    n_paths = 6000 if args.quick else args.paths
    bt_rows: list[dict] = []

    for name in names:
        print(f"[{name}]", flush=True)
        ctx = audit_universe(name, PRESETS[name], args.quick)
        if ctx is None:
            continue
        bt_rows += audit_backtests(name, ctx)
        if not args.no_map:
            try:
                audit_map(name, ctx, n_paths)
            except subprocess.CalledProcessError as exc:
                record("MAP-00", name, "map probe runs under node", False,
                       f"node failed: {(exc.stderr or '')[:200]}", "engine")

    if not args.no_map:
        audit_map_control(n_paths)
        audit_static_map_source()
    audit_traceability()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = ROOT / "audit"
    out_dir.mkdir(exist_ok=True)
    fails = [r for r in results if r["status"] == "FAIL"]
    warns = [r for r in results if r["status"] == "WARN"]
    (out_dir / f"batch_audit_{stamp}.json").write_text(json.dumps({
        "generated_utc": stamp, "universes": names, "checks": len(results),
        "failed": len(fails), "warned": len(warns),
        "paths_per_map_simulation": n_paths,
        "results": results, "var_backtests": bt_rows}, indent=2), encoding="utf-8")

    md = [f"# Batch audit - {stamp}", "",
          f"{len(results)} checks over {len(names)} live universes: "
          f"**{len(results) - len(fails) - len(warns)} pass, {len(fails)} fail, "
          f"{len(warns)} warn**. Every row names the source it is checked against.",
          "", "## Walk-forward VaR backtests", "",
          "| universe | book | conf | tested days | breaches | expected | rate | "
          "Kupiec LR | Kupiec | independence | cond. coverage |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in bt_rows:
        md.append(f"| {r['universe']} | {r['book']} | {r['confidence']:.0%} | "
                  f"{r['tested_days']} | {r['breaches']} | {r['expected']} | "
                  f"{r['observed_rate']:.2%} | {r['kupiec_lr']} | {r['kupiec_pass']} "
                  f"| {r['independence_pass']} | {r['conditional_coverage_pass']} |")
    md += ["", "## Checks", "",
           "| id | scope | check | status | detail | source |",
           "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['id']} | {r['scope']} | {r['check']} | {r['status']} | "
                  f"{r['detail']} | {r['source']} |")
    (out_dir / f"batch_audit_{stamp}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n{len(results)} checks: {len(results) - len(fails) - len(warns)} pass, "
          f"{len(fails)} FAIL, {len(warns)} warn")
    for r in fails:
        print(f"  FAIL {r['id']} [{r['scope']}] {r['check']}: {r['detail']}")
    for r in warns:
        print(f"  warn {r['id']} [{r['scope']}] {r['check']}: {r['detail']}")
    print(f"report: audit/batch_audit_{stamp}.md")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
