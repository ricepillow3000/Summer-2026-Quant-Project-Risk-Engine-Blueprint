"""
Every option on the control deck, on LIVE data, mixed and matched.

pair_audit.py proves every symbol and pair a visitor can pick, on one setting.
John, 2026-10-30: every asset one by one with every allocation, every target
volatility from 5% to 30%, every covariance estimator, both return models,
every crisis, both sides, book size and participation - "mix and match until
no pairs are left untested", nothing hallucinated, the cone reflecting what
was chosen, the map's every dot and line real. Two halves:

ENGINE (`engine`) - every preset basket, and every suggested symbol paired
with SPY, through the app's real functions, each answer re-derived by this
file's own numpy from the raw Yahoo closes:
  OPT-00  the returns are the closes' day-over-day change
  OPT-01  each covariance estimator (Sample, Ledoit-Wolf 2004, RiskMetrics EWMA)
  OPT-02  equal weight is 1/n; risk parity gives every name the same risk
  OPT-03  every target vol 5..30%: leverage = target / book vol, hit exactly
  OPT-04  both Monte Carlo engines: VaR/CVaR are the paths' own tail, the cone
          is the paths' own percentiles, and the mean simulated year is the
          history's mean year (Politis-Romano marginal; Merton mean identity)
  OPT-05  jump-diffusion parameters re-fitted from the book's own returns
  OPT-06  a setting where one day takes more than the whole book is not
          simulated: the page refuses it in words (counted as INFO here)
  OPT-07  every crisis replay is the raw closes' change inside the window
  OPT-08  days to liquidate for every book size x participation
  MAP-*   the Risk Topology map for every allocation (pair_audit.map_checks):
          the book's dot, every asset's dot, the hazard line, every defensive
          link replayed from raw returns
  TERRAIN-* the map's own simulation - the web of 30-day futures - run under
          node for every estimator and weighting, held to closed-form OU
          moments, ring masses and breach counting (batch_audit.audit_map)

PAGE (`page`) - the real main.py under Streamlit's AppTest, on the same
served data. Every widget value is a dimension; `pairwise` builds a covering
array so EVERY pair of values of EVERY two widgets appears in some run
(all-pairs testing: most interaction bugs need only two settings to meet).
Each run checks the page against the engine for the settings it was given:
  PAGE-00 no exception, no error box
  PAGE-01 the headline CVaR is the simulation of exactly these settings
  PAGE-02 the cone (cumulative return % by trading day) is that simulation's
          percentiles, day by day - not a picture
  PAGE-03 the Liquidity tab's exit days are the closed form for its two inputs
  PAGE-04 The Book quotes the chosen book size, participation and names
  PAGE-05 Correlation Watch draws the chosen pair's own rolling correlation
  MAP-*   the map in the page's iframe, re-derived dot by dot and line by line

    python -m tests.options_audit download audit/options [--offline]
    python -m tests.options_audit engine audit/options --shard 3/16
    python -m tests.options_audit page audit/options --shard 3/16
    python -m tests.options_audit summary audit

Yahoo is asked once (`download`, every period the page reads); the shards are
served from those files through the app's own chokepoint, as in pair_audit.
Exit code is non-zero on any FAIL.
"""

import argparse
import base64
import json
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import src.ingestion as ing
from hero import N_PATHS
from src.conviction import AI_CAPEX_BASKET, BENCHMARK
from src.covariance import RISKMETRICS_LAMBDA, TRADING_DAYS, estimate_covariance
from src.factors import _PROXIES as FACTOR_PROXIES
from src.liquidity import days_to_liquidate, liquidity_adjusted_cvar, liquidity_profile
from src.pairing import DEFENSIVE_ANCHOR_TICKERS
from src.risk import jump_diffusion_mc, monte_carlo
from src.scenarios import HISTORICAL_REGIMES, MAX_LEAD_GAP_DAYS, replay_returns
from src.strategies import risk_parity_weights, vol_target
from tests import pair_audit

PERIODS = ("2y", "10y", "max")           # every period the page reads (main.py, src/)
SYMBOLS = sorted(ing.SUGGESTIONS)
NEEDED = sorted(set(SYMBOLS) | {"SPY", "QQQ", BENCHMARK, *DEFENSIVE_ANCHOR_TICKERS,
                                *FACTOR_PROXIES, *AI_CAPEX_BASKET})
ESTIMATORS = ["Ledoit-Wolf", "Sample", "EWMA"]
WEIGHTINGS = ["Equal weight", "Risk parity"]
TARGETS = list(range(5, 31))             # the slider: 5..30 in steps of 1
ENGINES = ["Bootstrap (empirical)", "Jump-diffusion (Merton)"]
CRISES = list(HISTORICAL_REGIMES)
DRAWDOWNS = list(range(-50, 1, 5))       # "Market drawdown shock" slider
VOL_SHOCKS = list(range(0, 301, 10))     # "Volatility shock" slider
BOOKS = [10_000, 100_000, 1_000_000, 25_000_000, 500_000_000, 5_000_000_000]
PARTS = list(range(5, 51, 5))            # both participation sliders
N_MIXES = 12                             # typed-in baskets the page is also run on

_M: dict = {}                            # period -> (close, volume), the served download


# ---- data: one download, then served ----------------------------------------

def download(folder: str, offline: bool = False) -> int:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for period in PERIODS:
        if offline:                      # a synthetic market, to run this file without Yahoo
            from tests.test_engine import _offline_market_prices
            close = _offline_market_prices(NEEDED, "10y" if period != "2y" else "2y")
            vol = pd.DataFrame(2e6, index=close.index, columns=close.columns)
        else:
            parts = [ing._download_close_volume(NEEDED[i:i + 50], period, use_cache=False)
                     for i in range(0, len(NEEDED), 50)]
            close = pd.concat([p[0] for p in parts], axis=1).sort_index()
            vol = pd.concat([p[1] for p in parts], axis=1).sort_index()
        close = close.loc[:, ~close.columns.duplicated()]
        vol = vol.loc[:, ~vol.columns.duplicated()]
        close.to_parquet(out / f"close_{period}.parquet")
        vol.to_parquet(out / f"volume_{period}.parquet")
        print(f"{period}: {close.shape[1]} symbols x {len(close)} days", flush=True)
    rf = 0.045 if offline else ing.fetch_risk_free_rate(use_cache=False)
    (out / "riskfree.json").write_text(json.dumps({"rate": rf}))
    return 0


def _serve(tickers, period, *_, **__):
    """Stand-in for the Yahoo round trip: the columns yfinance handed the
    download job for exactly these symbols and this period."""
    close, vol = _M[period if period in _M else "max"]
    c = close.reindex(columns=tickers).dropna(how="all")
    if c.dropna(how="all", axis=1).empty:
        raise ing.NoDataError(ing.NO_DATA_MSG)
    return c, vol.reindex(index=c.index, columns=tickers)


def _init(folder: str) -> None:
    for p in PERIODS:
        _M[p] = (pd.read_parquet(Path(folder) / f"close_{p}.parquet"),
                 pd.read_parquet(Path(folder) / f"volume_{p}.parquet"))
    rf = json.loads((Path(folder) / "riskfree.json").read_text())["rate"]
    ing._download_close_volume = _serve
    ing.fetch_risk_free_rate = lambda *a, **k: rf
    ing.DATA_DIR = tempfile.mkdtemp(prefix="options_audit_")


# ---- this file's own maths --------------------------------------------------

def own_cov(r: np.ndarray, method: str) -> np.ndarray:
    """Each estimator from its paper, not from src.covariance."""
    t, p = r.shape
    if method == "Sample":
        return np.cov(r, rowvar=False, ddof=1) * TRADING_DAYS
    if method == "EWMA":                 # RiskMetrics (1996) recursion, from zero
        s = np.zeros((p, p))
        for row in r:
            s = RISKMETRICS_LAMBDA * s + (1 - RISKMETRICS_LAMBDA) * np.outer(row, row)
        return s / (1 - RISKMETRICS_LAMBDA ** t) * TRADING_DAYS
    x = r - r.mean(axis=0)               # Ledoit & Wolf (2004), scaled-identity target
    s = x.T @ x / t
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2)
    b2 = min((np.sum(np.sum(x ** 2, axis=1) ** 2) - t * np.sum(s ** 2)) / t ** 2, d2)
    k = b2 / d2 if d2 > 0 else 0.0
    return (k * mu * np.eye(p) + (1 - k) * s) * TRADING_DAYS


def shocked(rets: pd.DataFrame, drawdown: int, vol: int) -> pd.DataFrame:
    """The custom shock as its labels define it: the year compounds to the
    drawdown on the slider, and the spread of daily returns grows by the
    volatility shock."""
    out = rets.copy()
    if drawdown:
        out = (1 + out) * (1 + drawdown / 100) ** (1 / TRADING_DAYS) - 1
    if vol:
        m = out.mean()
        out = m + (out - m) * (1 + vol / 100)
    return out


def own_replay(full: pd.DataFrame, cols: list, start: str, end: str) -> pd.DataFrame:
    """A crisis replay from the raw closes: names trading at the window's start
    (within the lead-gap allowance) and positive throughout, aligned, measured
    against the previous session's close."""
    win = full.loc[start:end, cols]
    keep = [c for c in cols if win[c].notna().any() and (win[c].dropna() > 0).all()]
    if keep:
        firsts = {c: win[c].first_valid_index() for c in keep}
        lead_day = min(firsts.values())
        keep = [c for c in keep if (firsts[c] - lead_day).days <= MAX_LEAD_GAP_DAYS]
    win = win[keep].dropna()
    if win.empty:
        return win
    before = full.loc[full.index < win.index[0], keep].dropna()
    rows = [win]
    if len(before) and (win.index[0] - before.index[-1]).days <= MAX_LEAD_GAP_DAYS \
            and (before.iloc[-1] > 0).all():
        rows = [before.iloc[[-1]], win]
    px = pd.concat(rows).to_numpy()
    return pd.DataFrame(px[1:] / px[:-1] - 1, index=pd.concat(rows).index[1:], columns=keep)


def own_jump_params(port: np.ndarray, k: float = 3.0) -> dict:
    lr = np.log(1 + port)
    m, s = lr.mean(), lr.std(ddof=1)
    jump = np.abs(lr - m) > k * s
    calm = lr[~jump]
    mu_d = calm.mean()
    return {"mu_d": mu_d, "sigma_d": calm.std(ddof=1), "lambda_daily": jump.mean(),
            "mu_j": (lr[jump].mean() - mu_d) if jump.any() else 0.0}


def own_adv(cols: list) -> pd.Series:
    """Mean daily dollar volume over the basket's last 63 shared trading days
    (weekdays every name priced, holiday gaps of up to 3 days carried), straight
    from the download. Futures volume counts contracts, so it is no dollar
    figure: excluded (0), as the page discloses."""
    close, vol = _M["2y"]
    c = close.reindex(columns=cols).dropna(how="all").dropna(axis=1, how="all")
    days = c[c.index.dayofweek < 5].ffill(limit=3).dropna().index
    dv = (c * vol.reindex(index=c.index, columns=c.columns)).reindex(days)
    dv[[x for x in dv.columns if x.endswith("=F")]] = 0.0
    return dv.tail(63).mean().reindex(cols).fillna(0.0)


def mc_checks(mc: dict, sim: pd.DataFrame, w: np.ndarray, jd: bool) -> list:
    """OPT-04..06 for one simulation of book `sim @ w` (already negated for a short)."""
    out = []
    port = sim.to_numpy() @ np.asarray(w, dtype=float)
    tr = np.asarray(mc["total_returns"])
    thr = np.percentile(tr, 5)
    tail = tr[tr <= thr]
    b = mc["path_bands"]
    p = np.vstack([b[k] for k in ("p5", "p25", "p50", "p75", "p95")])
    ok = (len(tr) == N_PATHS and mc["horizon_days"] == TRADING_DAYS and np.isfinite(tr).all()
          and abs(mc["var"] + thr) < 1e-12 and abs(mc["cvar"] + tail.mean()) < 1e-12
          and mc["cvar"] >= mc["var"] - 1e-12
          and list(b["days"]) == list(range(1, TRADING_DAYS + 1))
          and (np.diff(p, axis=0) >= -1e-12).all()
          and abs(b["p50"][-1] - np.median(tr)) < 1e-12 and abs(b["p5"][-1] - thr) < 1e-12)
    out.append(("OPT-04", ok, f"paths {len(tr)}, VaR {mc['var']:.6f} vs {-thr:.6f}, "
                f"CVaR {mc['cvar']:.6f} vs {-tail.mean():.6f}, cone ordered "
                f"{(np.diff(p, axis=0) >= -1e-12).all()}, cone end {b['p50'][-1]:.6f} vs "
                f"median {np.median(tr):.6f}"))
    if (1 + port > 0).all():
        lf = np.log1p(tr)
        want = TRADING_DAYS * np.log1p(port).mean()
        se = lf.std(ddof=1) / np.sqrt(len(lf))
        out.append(("OPT-04", abs(lf.mean() - want) <= 5 * se + 1e-12,
                    f"mean simulated log-year {lf.mean():.5f} vs history's {want:.5f} "
                    f"(5 standard errors = {5 * se:.5f})"))
        if jd:
            own = own_jump_params(port)
            got = mc["jump_params"]
            out.append(("OPT-05", all(abs(got[k] - v) <= 1e-12 * max(1, abs(v))
                                      for k, v in own.items()),
                        f"engine {({k: got[k] for k in own})} vs own {own}"))
    return out


# ---- ENGINE: every preset and every symbol, every option ---------------------

def engine_jobs() -> list:
    jobs = [(f"preset:{k}", list(v), True) for k, v in ing.PRESETS.items()]
    jobs += [(f"{s}+SPY", sorted({s, "SPY" if s != "SPY" else "QQQ"}), False) for s in SYMBOLS]
    return jobs


def audit_book(job) -> list:
    name, tickers, full = job
    out = []

    def rec(cid, ok, detail="", status=None):
        out.append((cid, name, status or ("PASS" if ok else "FAIL"), "" if ok else detail))

    close2 = _M["2y"][0]
    if any(t not in close2.columns or close2[t].dropna().empty for t in tickers):
        if not full:                     # a delisted name fails TICK-01 in pair_audit once
            rec("OPT-00", True, status="SKIP")
            return out
    try:
        px = ing.fetch_prices(tickers, period="2y")
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        rec("OPT-00", not full, f"fetch_prices raised {type(exc).__name__}: {exc}",
            None if full else "INFO")
        return out
    rets = ing.get_returns(px)
    flat = ing.flat_columns(rets)
    px, rets = px.drop(columns=flat), rets.drop(columns=flat)
    if rets.shape[1] < 2 or len(px) < ing.MIN_ROWS:
        rec("OPT-00", True, status="INFO")   # the page refuses it, in words
        return out
    cols, n = list(rets.columns), rets.shape[1]
    a = px.to_numpy()
    r = a[1:] / a[:-1] - 1
    rec("OPT-00", np.allclose(rets.to_numpy(), r, rtol=0, atol=1e-15),
        "returns differ from the closes' change")

    covs, books = {}, {}
    for m in ESTIMATORS:
        cov, _ = estimate_covariance(rets, m)
        c, own = cov.loc[cols, cols].to_numpy(), own_cov(r, m)
        rec("OPT-01", np.allclose(c, own, rtol=1e-8, atol=1e-13) and np.allclose(c, c.T)
            and np.linalg.eigvalsh(c).min() > -1e-12,
            f"{m}: worst gap {np.abs(c - own).max():.2e}, min eigenvalue {np.linalg.eigvalsh(c).min():.2e}")
        covs[m] = cov
        for wn in WEIGHTINGS:
            w = risk_parity_weights(cov) if wn == "Risk parity" else np.ones(n) / n
            if wn == "Risk parity":
                rc = w * (c @ w)
                ok = (w > 0).all() and abs(w.sum() - 1) < 1e-9 \
                    and (rc.max() - rc.min()) / rc.mean() < 1e-6
                detail = f"{m} risk parity: risk shares {np.round(rc / rc.sum(), 6)}"
            else:
                ok, detail = bool(np.allclose(w, 1 / n)), f"equal weight {w}"
            rec("OPT-02", ok, detail)
            books[(m, wn)] = w
            sig = float(np.sqrt(w @ c @ w))
            for t in TARGETS:
                vt = vol_target(w, cov, t / 100)
                lev = t / 100 / sig
                sw = vt["scaled_weights"]
                rec("OPT-03", abs(vt["leverage"] - lev) <= 1e-12 * max(1, lev)
                    and np.allclose(sw, w * lev, rtol=1e-12)
                    and abs(np.sqrt(sw @ c @ sw) - t / 100) < 1e-12,
                    f"{m} {wn} target {t}%: leverage {vt['leverage']:.6f} vs {lev:.6f}, "
                    f"scaled vol {np.sqrt(sw @ c @ sw):.6f}")

    # The simulations: (label, sim returns, weights, engine) for each setting
    rp = books[("Ledoit-Wolf", "Risk parity")]
    ew = books[("Ledoit-Wolf", "Equal weight")]
    full_px = ing.fetch_prices(cols, period="max", align=False)
    runs = []
    for short in (False, True):
        sgn = -1 if short else 1
        side = "short" if short else "long"
        for jd in (False, True):
            eng = "jump-diffusion" if jd else "bootstrap"
            for wn, w in (("equal weight", ew), ("risk parity", rp)):
                runs.append((f"{side} {eng} {wn} base", sgn * rets, w, jd))
                for crisis in CRISES:
                    s, e = HISTORICAL_REGIMES[crisis]
                    rr = replay_returns(cols, s, e)
                    mine = own_replay(full_px, cols, s, e)
                    if not short and not jd and wn == "equal weight":
                        rec("OPT-07", list(rr.columns) == list(mine.columns)
                            and rr.index.equals(mine.index)
                            and np.allclose(rr.to_numpy(), mine.to_numpy(), rtol=0, atol=1e-12),
                            f"{crisis}: engine {list(rr.columns)} x {len(rr)} days vs raw "
                            f"{list(mine.columns)} x {len(mine)}")
                    if rr.shape[1] < 2:
                        continue         # the page says too few names traded, and stops
                    idx = [cols.index(x) for x in rr.columns]
                    sw = w[idx] / w[idx].sum() * w.sum()     # the page's exposure rule
                    runs.append((f"{side} {eng} {wn} {crisis}", sgn * rr, sw, jd))
            for t in TARGETS:
                lev = t / 100 / float(np.sqrt(rp @ covs["Ledoit-Wolf"].to_numpy() @ rp))
                runs.append((f"{side} {eng} risk parity vol-target {t}%", sgn * rets, rp * lev, jd))
            for m in ("Sample", "EWMA"):
                runs.append((f"{side} {eng} risk parity {m}", sgn * rets, books[(m, "Risk parity")], jd))
            if full:
                for dd, vs in ([(d, 0) for d in DRAWDOWNS if d] + [(0, v) for v in VOL_SHOCKS if v]
                               + [(-50, 300), (-25, 150), (-5, 10)]):
                    runs.append((f"{side} {eng} equal weight shock {dd:+d}%/{vs:+d}%",
                                 sgn * shocked(rets, dd, vs), ew, jd))
                lev30 = 0.30 / float(np.sqrt(rp @ covs["EWMA"].to_numpy() @ rp))
                for dd, vs in ((-50, 300), (0, 300), (-5, 260)):
                    runs.append((f"{side} {eng} EWMA-scaled vol-target 30% shock {dd:+d}%/{vs:+d}%",
                                 sgn * shocked(rets, dd, vs), rp * lev30, jd))
    for label, sim, w, jd in runs:
        worst = float((sim.to_numpy() @ np.asarray(w, dtype=float)).min())
        if worst <= -1:                  # main.py refuses it in words (a day past 100%)
            rec("OPT-06", True, status="INFO")
            continue
        fn = jump_diffusion_mc if jd else monte_carlo
        try:
            mc = fn(sim, np.asarray(w, dtype=float), n_simulations=N_PATHS,
                    horizon_days=TRADING_DAYS)
        except Exception as exc:  # noqa: BLE001
            rec("OPT-04", False, f"{label}: {type(exc).__name__}: {exc}")
            continue
        for cid, ok, detail in mc_checks(mc, sim, w, jd):
            rec(cid, ok, f"{label}: {detail}")

    # OPT-08 liquidity: every book size x participation, three books
    adv = ing.average_dollar_volume(cols).reindex(cols).fillna(0.0)
    mine = own_adv(cols).reindex(cols).fillna(0.0)
    rec("OPT-08", np.allclose(adv.to_numpy(), mine.to_numpy(), rtol=1e-12, atol=0),
        f"ADV engine {adv.round(0).to_dict()} vs download {mine.round(0).to_dict()}")
    lev10 = 0.10 / float(np.sqrt(rp @ covs["Ledoit-Wolf"].to_numpy() @ rp))
    for wn, w in (("equal weight", ew), ("risk parity", rp), ("risk parity vol-target 10%", rp * lev10)):
        for book in BOOKS:
            for part in PARTS:
                dtl = days_to_liquidate(w, adv, book_value=book, participation_rate=part / 100)
                cap = part / 100 * mine.to_numpy()
                with np.errstate(divide="ignore"):
                    days = np.where(cap > 0, np.abs(w) * book / cap, np.inf)
                prof = liquidity_profile(dtl)
                fin = days[np.isfinite(days)]
                full_exit = fin.max() if fin.size else np.inf
                ok = np.allclose(dtl["days"].to_numpy(), days, rtol=1e-12, equal_nan=False) \
                    and (prof["full_exit_days"] == full_exit or
                         abs(prof["full_exit_days"] - full_exit) <= 1e-12 * full_exit)
                if np.isfinite(full_exit):
                    lv = liquidity_adjusted_cvar(0.2, full_exit)
                    ok = ok and abs(lv["multiplier"] - np.sqrt(1 + full_exit / TRADING_DAYS)) < 1e-12
                rec("OPT-08", ok, f"{wn} ${book:,} at {part}%: days {dtl['days'].round(4).tolist()} "
                    f"vs {np.round(days, 4).tolist()}")

    # MAP-* the map for every allocation the deck can make
    spy = _M["2y"][0]["SPY"].dropna()
    mkt = pd.Series(spy.to_numpy()[1:] / spy.to_numpy()[:-1] - 1, index=spy.index[1:])
    vts = [None, 5, 10, 20, 30] if full else [None, 10]
    for (m, wn), w in books.items():
        if not full and (m, wn) not in (("Ledoit-Wolf", "Equal weight"), ("Ledoit-Wolf", "Risk parity")):
            continue
        sig = float(np.sqrt(w @ covs[m].to_numpy() @ w))
        for t in vts:
            ww = w if t is None else w * (t / 100 / sig)
            for short in (False, True):
                tag = f"{m} {wn}{'' if t is None else f' vol-target {t}%'} {'short' if short else 'long'}"
                for cid, ok, detail, status in pair_audit.map_checks(px, ww, short, mkt):
                    rec(cid, ok, f"{tag}: {detail}", status)
            # TERRAIN-*: the map's own simulation (the web of 30-day futures),
            # run under node and held to closed-form OU theory (batch_audit)
            if full and t in (None, 10, 30) and shutil.which("node"):
                from tests import batch_audit
                batch_audit.results.clear()
                batch_audit.audit_map(f"{name} {m} {wn} {t}", {"returns": rets, "weights": ww,
                                                               "cols": cols}, 6000)
                for x in batch_audit.results:
                    rec(f"TERRAIN-{x['id']}", x["status"] == "PASS", f"{m} {wn} target {t}: {x['detail']}",
                        None if x["status"] in ("PASS", "FAIL") else x["status"])
    return out


def run_engine(folder: str, shard: str) -> int:
    i, n = (int(x) for x in shard.split("/"))
    jobs = [j for k, j in enumerate(engine_jobs()) if k % n == i]
    print(f"engine shard {i}/{n}: {len(jobs)} books", flush=True)
    rows = []
    with mp.Pool(mp.cpu_count(), initializer=_init, initargs=(folder,)) as pool:
        for k, res in enumerate(pool.imap_unordered(audit_book, jobs), 1):
            rows += res
            print(f"  {k}/{len(jobs)} {res[0][1] if res else ''}", flush=True)
    return _report(rows, {"books": len(jobs)}, Path(folder) / f"options_engine_{i}.json")


# ---- PAGE: the real page, every pair of widget values -------------------------

def pairwise(dims: dict) -> list:
    """A covering array: every value of every widget meets every value of every
    other widget in at least one row. Greedy (AETG, Cohen et al. 1997): each
    row starts from the first pair not yet covered, then each remaining widget
    takes the value that covers the most new pairs. Deterministic."""
    names = list(dims)
    size = [len(dims[k]) for k in names]
    todo = {(i, a, j, b) for i, j in combinations(range(len(names)), 2)
            for a in range(size[i]) for b in range(size[j])}
    used = [[0] * s for s in size]
    rows = []
    while todo:
        i, a, j, b = min(todo)
        row = {i: a, j: b}
        for k in range(len(names)):
            if k in row:
                continue

            def gain(v, k=k):
                return sum((min(k, m), v if k < m else row[m], max(k, m), row[m] if k < m else v)
                           in todo for m in row)
            row[k] = max(range(size[k]), key=lambda v: (gain(v), -used[k][v]))
        for k, v in row.items():
            used[k][v] += 1
        todo -= {(p, row[p], q, row[q]) for p, q in combinations(sorted(row), 2)}
        rows.append({names[k]: dims[names[k]][row[k]] for k in range(len(names))})
    return rows


def mixes() -> dict:
    """Typed-in baskets: seeded draws from the suggestion list, two to eight
    names, so the page also meets universes no preset offers."""
    rng = np.random.default_rng(20261030)
    return {f"mix {k + 1}": sorted(rng.choice(SYMBOLS, size=int(rng.integers(2, 9)),
                                              replace=False).tolist())
            for k in range(N_MIXES)}


def page_rows() -> list:
    universes = list(ing.PRESETS) + list(mixes())
    common = {
        "universe": universes, "side": [False, True], "estimator": ESTIMATORS,
        "weighting": WEIGHTINGS, "target": [None] + TARGETS, "engine": ENGINES,
        "book": BOOKS, "book_part": PARTS, "grit_floor": [False, True],
        "outside_anchor": [True, False], "liq_book": BOOKS, "liq_part": PARTS,
        "watch_a": [0, 1, -1], "watch_b": [1, 0, -1], "window": [10, 21, 42, 63],
        "threshold": [0.5, 0.75, 0.95], "anchor": [0, -1], "flyer": [0, -1],
    }
    custom = pairwise({**common, "drawdown": DRAWDOWNS, "vol": VOL_SHOCKS})
    for r in custom:
        r["scenario"] = "Custom shock (sliders)"
    replay = pairwise({**common, "scenario": CRISES})
    for r in replay:
        r["drawdown"] = r["vol"] = 0
    rows = custom + replay
    order = {u: k for k, u in enumerate(universes)}
    return sorted(rows, key=lambda r: order[r["universe"]])


def _widget(seq, label):
    hits = [w for w in seq if w.label == label]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} widgets labelled {label!r}")
    return hits[0]


def _pick(widget, idx):
    opts = list(widget.options)
    widget.set_value(opts[max(-len(opts), min(idx, len(opts) - 1))])


def apply_row(at, row, mix_names):
    """Set every widget the way a visitor would, universe first (it re-draws
    every universe-dependent list), then the rest. Stops setting as soon as
    the page stops or raises: check_page reads what it said."""
    def go():
        at.run()
        return bool(at.exception)

    u = row["universe"]
    preset = u if u in ing.PRESETS else list(ing.PRESETS)[0]
    if _widget(at.selectbox, "Preset basket").value != preset:
        _widget(at.selectbox, "Preset basket").set_value(preset)
        if go():
            return
    if u in mix_names:
        box = at.multiselect(key=f"tickers__{preset}")
        if list(box.value) != mix_names[u]:
            box.set_value(mix_names[u])
            if go():
                return
    _widget(at.radio, "Which side of the trade are you on?").set_value(
        "Bearish - short the book" if row["side"] else "Bullish - long the book")
    _widget(at.selectbox, "Covariance estimator").set_value(row["estimator"])
    _widget(at.radio, "Weighting").set_value(row["weighting"])
    vt = _widget(at.checkbox, "Target volatility")
    if row["target"] is None:
        vt.uncheck()
    else:
        vt.check()
        if go():                         # the slider is disabled until the box is ticked
            return
        _widget(at.slider, "Target annual vol (%)").set_value(row["target"])
    _widget(at.radio, "Return model").set_value(row["engine"])
    _widget(at.selectbox, "Scenario").set_value(row["scenario"])
    if go():
        return
    if row["scenario"].startswith("Custom"):
        _widget(at.slider, "Market drawdown shock").set_value(row["drawdown"])
        _widget(at.slider, "Volatility shock").set_value(row["vol"])
        if go():
            return
    if "book_size" not in {w.key for w in at.number_input}:
        return                           # the page stopped above The Book, in words
    at.number_input(key="book_size").set_value(row["book"])
    at.slider(key="book_part").set_value(row["book_part"])
    at.toggle(key="book_floor").set_value(row["grit_floor"])
    at.toggle(key="book_fallback").set_value(row["outside_anchor"])
    if go():
        return
    for w in at.number_input:
        if w.label == "Portfolio size ($)":
            w.set_value(row["liq_book"])
    for w in at.slider:
        if w.label == "Max daily participation (% of ADV)":
            w.set_value(row["liq_part"])
    keys = {w.key for w in at.selectbox} | {w.key for w in at.slider}
    for key, idx in (("watch_a", row["watch_a"]), ("watch_b", row["watch_b"])):
        if key in keys:
            _pick(at.selectbox(key=key), idx)
    if "watch_w" in keys:
        at.slider(key="watch_w").set_value(row["window"])
        at.slider(key="watch_t").set_value(row["threshold"])
    for label, idx in (("Anchor (the position you want to balance)", row["anchor"]),
                       ("High-flyer (Circle 1 - the volatile conviction position)", row["flyer"])):
        hits = [w for w in at.selectbox if w.label == label]
        if hits:
            _pick(hits[0], idx)
    at.run()


def _arr(v) -> np.ndarray:
    """A plotly trace's numbers: plotly ships numpy arrays as base64 typed
    arrays ({"dtype", "bdata"}) and plain lists otherwise."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v, dtype=float)


def _fmt_days(d: float) -> str:          # the Liquidity tab's own rounding (main.py)
    if not np.isfinite(d):
        return "∞"
    return "<0.1d" if d < 0.1 else f"{d:.1f}d" if d < 10 else f"{d:.0f}d"


def expected(row, loaded):
    """The settings' answer from the engine, not from main.py: the book the
    deck describes, its stress, its simulation."""
    px = ing.fetch_prices(loaded, period="2y")
    rets = ing.get_returns(px)
    rets = rets.drop(columns=ing.flat_columns(rets))
    cols = list(rets.columns)
    cov, _ = estimate_covariance(rets, row["estimator"])
    w = risk_parity_weights(cov) if row["weighting"] == "Risk parity" else np.ones(len(cols)) / len(cols)
    if row["target"] is not None:
        w = vol_target(w, cov, row["target"] / 100)["scaled_weights"]
    if row["scenario"].startswith("Custom"):
        sim, sw = shocked(rets, row["drawdown"], row["vol"]), w
    else:
        s, e = HISTORICAL_REGIMES[row["scenario"]]
        sim = replay_returns(cols, s, e)
        if sim.shape[1] < 2:
            return {"stops": True, "why": "traded during", "px": px, "rets": rets, "w": w}
        idx = [cols.index(x) for x in sim.columns]
        sw = w[idx] / w[idx].sum() * w.sum()
    if row["side"]:
        sim = -sim
    if float((sim.to_numpy() @ np.asarray(sw, dtype=float)).min()) <= -1:
        return {"stops": True, "why": "loses", "px": px, "rets": rets, "w": w}
    fn = jump_diffusion_mc if row["engine"].startswith("Jump") else monte_carlo
    mc = fn(sim, np.asarray(sw, dtype=float), n_simulations=N_PATHS, horizon_days=TRADING_DAYS)
    return {"stops": False, "px": px, "rets": rets, "w": w, "mc": mc, "sim": sim, "sw": sw}


def check_page(at, row, mix_names) -> list:
    out = []

    def rec(cid, ok, detail="", status=None):
        out.append((cid, ok, "" if ok else detail, status))

    errors = [e.value for e in at.error]
    rec("PAGE-00", not at.exception and not errors,
        f"exception {[e.message for e in at.exception][:1]} at "
        f"{' | '.join(at.exception[0].stack_trace[-6:]) if at.exception else ''}; error {errors[:1]}")
    if at.exception:
        return out
    trail = {}
    for d in at.dataframe:
        v = d.value
        if list(v.columns) == ["step", "detail"]:
            trail = dict(zip(v["step"], v["detail"], strict=True))
    hit = re.match(r"\d+ tickers loaded from .*?: ([^(]+?)(?: \(missing|$)", trail.get("Data fetch", ""))
    loaded = [t.strip() for t in hit.group(1).split(",")] if hit else None
    if not loaded:                       # a page that stopped draws no audit trail
        chosen = sorted(mix_names.get(row["universe"]) or ing.PRESETS[row["universe"]])
        px = ing.fetch_prices(chosen, period="2y")
        loaded = [c for c in px.columns if c not in ing.flat_columns(ing.get_returns(px))]
    exp = expected(row, loaded)
    warns = " ".join(w.value for w in at.warning)
    if not hit and not exp["stops"]:
        rec("PAGE-00", False, f"no audit trail on a page that should not stop {trail}")
        return out
    if exp["stops"]:
        rec("PAGE-01", exp["why"] in warns and not any('class="verdict-box"' in m.value for m in at.markdown),
            f"engine: the page should stop ({exp['why']!r}) for {row['scenario']}; "
            f"page warnings: {warns[:200]}")
        return out
    mc = exp["mc"]
    verdict = next((m.value for m in at.markdown if 'class="verdict-box"' in m.value), "")
    shown = re.search(r'verdict-number">([^<]+)<', verdict)
    rec("PAGE-01", bool(shown) and shown.group(1).strip() == f"{mc['cvar']:.1%}",
        f"page {shown.group(1) if shown else None} vs engine {mc['cvar']:.1%} "
        f"for {_row_label(row)}")

    fan = None
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        title = spec.get("layout", {}).get("xaxis", {}).get("title", {})
        if (title.get("text") if isinstance(title, dict) else title) == "Trading days":
            fan = spec
    if fan is None:
        rec("PAGE-02", False, "no cumulative-return cone on the page")
    else:
        med = next(t for t in fan["data"] if t.get("name") == "median")
        b = mc["path_bands"]
        y = {k: _arr(fan["data"][i]["y"]) for k, i in (("p95", 0), ("p5", 1), ("p75", 2), ("p25", 3))}
        y["p50"] = _arr(med["y"])
        rec("PAGE-02", _arr(med["x"]).tolist() == list(b["days"])
            and all(np.allclose(y[k], b[k] * 100, rtol=1e-12, atol=1e-12) for k in y),
            f"cone median ends {y['p50'][-1]:.4f}% vs engine {b['p50'][-1] * 100:.4f}%")

    cols, w = list(exp["rets"].columns), exp["w"]
    chosen = sorted(mix_names.get(row["universe"]) or ing.PRESETS[row["universe"]])
    adv = own_adv(chosen).reindex(cols).fillna(0.0).to_numpy()   # the page reads the whole basket
    cap = row["liq_part"] / 100 * adv
    with np.errstate(divide="ignore"):
        days = np.where(cap > 0, np.abs(w) * row["liq_book"] / cap, np.inf)
    fin = days[np.isfinite(days)]
    metric = {m.label: m.value for m in at.metric}
    bars = None
    for ch in at.get("plotly_chart"):
        spec = json.loads(ch.proto.spec)
        title = spec.get("layout", {}).get("xaxis", {}).get("title", {})
        if (title.get("text") if isinstance(title, dict) else title) == "days to liquidate":
            bars = dict(zip(spec["data"][0]["y"], _arr(spec["data"][0]["x"]), strict=True))
    if "Full-exit horizon" not in metric:
        rec("PAGE-03", False, "the Liquidity tab drew no exit horizon")
    else:
        want = {c: d for c, d in zip(cols, days, strict=True) if np.isfinite(d)}
        rec("PAGE-03", metric["Full-exit horizon"] == _fmt_days(fin.max() if fin.size else np.inf)
            and (bars or {}).keys() == want.keys()
            and all(abs(bars[c] - want[c]) <= 1e-9 * max(1.0, want[c]) for c in want),
            f"page {metric['Full-exit horizon']}, bars {bars} vs "
            f"{_fmt_days(fin.max() if fin.size else np.inf)}, {want} "
            f"(${row['liq_book']:,} at {row['liq_part']}%)")

    book = next((m.value for m in at.markdown if " book - " in m.value and "quiet-tape" in m.value), "")
    bh = re.search(r"\$([\d,]+) book - (\d+) of your (\d+) names .* at (\d+)% of quiet-tape", book)
    rec("PAGE-04", bool(bh) and int(bh.group(1).replace(",", "")) == row["book"]
        and int(bh.group(4)) == row["book_part"] and int(bh.group(3)) == len(cols),
        f"book headline {re.sub('<[^>]+>', '', book)[:160]!r} for ${row['book']:,} at {row['book_part']}%")

    pa, pb = at.selectbox(key="watch_a").value, at.selectbox(key="watch_b").value
    if pa == pb:
        rec("PAGE-05", "correlated with itself" in warns, f"{pa} against itself: no notice")
    else:
        rr = exp["rets"]
        wfig = next((json.loads(c.proto.spec) for c in at.get("plotly_chart")
                     if json.loads(c.proto.spec).get("data", [{}])[-1].get("name") == f"{pa} × {pb}"), None)
        x, y = rr[pa].to_numpy(), rr[pb].to_numpy()
        win = row["window"]
        own = [np.corrcoef(x[k - win:k], y[k - win:k])[0, 1] for k in range(win, len(x) + 1)]
        got = _arr(wfig["data"][-1]["y"]) if wfig is not None else np.array([])
        ok = len(got) == len(own) and np.allclose(got, own, rtol=1e-9, atol=1e-12)
        rec("PAGE-05", ok and metric.get("Full-period correlation") == f"{np.corrcoef(x, y)[0, 1]:+.2f}",
            f"{pa} x {pb} {win}-day: page {len(got)} points "
            f"vs own {len(own)}; full-period {metric.get('Full-period correlation')} vs "
            f"{np.corrcoef(x, y)[0, 1]:+.2f}")

    frames = [f.proto.srcdoc for f in at.get("iframe") if "const DEMO = " in f.proto.srcdoc]
    if frames:
        blob = frames[0].split("const DEMO = ", 1)[1].split(";\n", 1)[0]
        pl = json.loads(blob)
        spy = _M["2y"][0]["SPY"].dropna()
        mkt = pd.Series(spy.to_numpy()[1:] / spy.to_numpy()[:-1] - 1, index=spy.index[1:])
        px = exp["px"][cols]
        for cid, ok, detail, status in pair_audit.map_checks(px, w, row["side"], mkt, pl):
            rec(cid, ok, detail, status)
    else:
        rec("MAP-00", False, "no map on the page")
    return out


def _row_label(row) -> str:
    return " / ".join(f"{k}={v}" for k, v in row.items())


def run_page(folder: str, shard: str) -> int:
    from streamlit import config
    from streamlit.testing.v1 import AppTest
    config.set_option("client.showErrorDetails", "full")   # tracebacks in the report
    i, n = (int(x) for x in shard.split("/"))
    _init(folder)
    for m in list(sys.modules.values()):          # names imported before the patch
        if getattr(m, "__name__", "").startswith("src.") and hasattr(m, "fetch_risk_free_rate"):
            m.fetch_risk_free_rate = ing.fetch_risk_free_rate
    rows = page_rows()
    universes = list(dict.fromkeys(r["universe"] for r in rows))
    mine = [r for r in rows if universes.index(r["universe"]) % n == i]
    names = mixes()
    print(f"page shard {i}/{n}: {len(mine)} of {len(rows)} runs", flush=True)
    app = Path(__file__).resolve().parent.parent / "main.py"
    at = AppTest.from_file(str(app), default_timeout=600)
    at.run()
    out = []
    for k, row in enumerate(mine, 1):
        scope = f"{row['universe']} #{k}"
        try:
            apply_row(at, row, names)
            res = check_page(at, row, names)
        except Exception as exc:  # noqa: BLE001 - one bad run must not hide the rest
            res = [("PAGE-00", False, f"{type(exc).__name__}: {exc} for {_row_label(row)}", None)]
        out += [(cid, scope, status or ("PASS" if ok else "FAIL"),
                 "" if ok else f"{detail} [{_row_label(row)}]") for cid, ok, detail, status in res]
        if k % 10 == 0:
            print(f"  {k}/{len(mine)}", flush=True)
    return _report(out, {"runs": len(mine), "of": len(rows)}, Path(folder) / f"options_page_{i}.json")


# ---- reports -----------------------------------------------------------------

def _report(rows: list, meta: dict, out: Path) -> int:
    counts: dict = {}
    for cid, _, status, _ in rows:
        counts.setdefault(cid, {}).setdefault(status, 0)
        counts[cid][status] += 1
    notes = [dict(zip(("id", "scope", "status", "detail"), x, strict=True))
             for x in rows if x[2] == "FAIL"]
    out.write_text(json.dumps({**meta, "counts": counts, "notes": notes}, indent=1))
    return _print(meta, counts, notes)


def _print(meta: dict, counts: dict, notes: list) -> int:
    print("\n" + ", ".join(f"{k} {v}" for k, v in meta.items()))
    for cid in sorted(counts):
        print(f"  {cid}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts[cid].items())))
    for x in notes[:80]:
        print(f"  FAIL {x['id']} [{x['scope']}] {x['detail'][:1500]}")
    print(f"\n{'FAILED: ' + str(len(notes)) if notes else 'every check passes'}")
    return 1 if notes else 0


def summary(folder: str) -> int:
    rc = 0
    for kind, total in (("engine", len(engine_jobs())), ("page", len(page_rows()))):
        files = sorted(Path(folder).glob(f"options_{kind}_*.json"))
        n, counts, notes = 0, {}, []
        for f in files:
            d = json.loads(f.read_text())
            n += d.get("books", d.get("runs", 0))
            notes += d["notes"]
            for cid, by in d["counts"].items():
                for k, v in by.items():
                    counts.setdefault(cid, {}).setdefault(k, 0)
                    counts[cid][k] += v
        print(f"\n== {kind}: {len(files)} shards")
        rc |= _print({"covered": n, "of": total}, counts, notes)
        if n != total:
            print(f"FAILED: {n} of {total} {kind} jobs reported - a shard is missing")
            rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("folder")
    d.add_argument("--offline", action="store_true")
    for name in ("engine", "page"):
        p = sub.add_parser(name)
        p.add_argument("folder")
        p.add_argument("--shard", default="0/1")
    sub.add_parser("summary").add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "download":
        return download(a.folder, a.offline)
    if a.cmd == "engine":
        return run_engine(a.folder, a.shard)
    if a.cmd == "page":
        return run_page(a.folder, a.shard)
    return summary(a.folder)


if __name__ == "__main__":
    sys.exit(main())
