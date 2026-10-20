"""
Every ticker and every pair a visitor can pick, audited on LIVE data.

batch_audit.py proves each preset. The universe box offers every symbol in
SUGGESTIONS, so a visitor can build any pair of them; on 2026-10-17 John asked
for every one of those choices to be proven, not sampled. Asking Yahoo for
51,681 universes would hammer the feed, so `download` fetches every symbol
ONCE through the app's own chokepoint (ingestion._download_close_volume, behind
netguard), and `run` serves each pair's request from that download. Every pair
still takes the app's real path - fetch_prices (cleaning, alignment, cache) ->
returns -> estimators, weights and build_map_payload, long and short - and
every number is then re-derived from the raw closes by this file's own numpy.

    python -m tests.pair_audit download audit/pair_prices.parquet
    python -m tests.pair_audit run audit/pair_prices.parquet --shard 3/16
    python -m tests.pair_audit summary audit

Shard 0 also audits each symbol on its own (TICK-01). Exit code is non-zero on
any FAIL. The Pair audit workflow runs the shards in parallel.
"""

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import src.ingestion as ing
from src.covariance import estimate_covariance
from src.data_quality import validate_prices
from src.pairing import DEFENSIVE_ANCHOR_TICKERS
from src.risk import cvar, var
from src.strategies import risk_parity_weights
from src.topology import (BETA_WINDOW, HAZARD_BETA_ADD, HAZARD_BETA_CAP,
                          HAZARD_VOL_CAP, HAZARD_VOL_FLOOR, HAZARD_VOL_MULT,
                          VOL_WINDOW, build_map_payload)
from tests.batch_audit import replay_linkages

UNIVERSE = sorted(ing.SUGGESTIONS)
EXTRA = sorted(set(["SPY", *DEFENSIVE_ANCHOR_TICKERS]) - set(UNIVERSE))
CHUNK = 50                   # symbols per Yahoo request
ESTIMATORS = ["Ledoit-Wolf", "Sample", "EWMA"]
# The map's state model needs a 63-day beta window plus MIN_OBS/2 rows
# (state_calibration.rolling_state_series); a younger listing is refused there.
MAP_MIN_DAYS = BETA_WINDOW + 60
SHORT_HISTORY = "need more overlapping history"

_MASTER: pd.DataFrame | None = None


def download(out: str) -> int:
    closes = []
    for i in range(0, len(UNIVERSE + EXTRA), CHUNK):
        chunk = (UNIVERSE + EXTRA)[i:i + CHUNK]
        close, _ = ing._download_close_volume(chunk, "2y", use_cache=False)
        closes.append(close)
        print(f"downloaded {i + len(chunk)}/{len(UNIVERSE + EXTRA)}", flush=True)
    master = pd.concat(closes, axis=1).sort_index()
    master = master.loc[:, ~master.columns.duplicated()]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out)
    print(f"{master.shape[1]} symbols x {len(master)} days -> {out}")
    return 0


def _serve(tickers, period, *_, **__):
    """Stand-in for the Yahoo round trip: the same columns yfinance would hand
    back for exactly these symbols, on exactly the days any of them traded."""
    close = _MASTER.reindex(columns=tickers).dropna(how="all")
    if close.dropna(how="all", axis=1).empty:
        raise ing.NoDataError(ing.NO_DATA_MSG)
    return close, pd.DataFrame(np.nan, index=close.index, columns=close.columns)


def _init(path: str) -> None:
    global _MASTER
    _MASTER = pd.read_parquet(path)
    ing._download_close_volume = _serve
    ing.DATA_DIR = tempfile.mkdtemp(prefix="pair_audit_")   # one cache per worker


def _rets(px: np.ndarray) -> np.ndarray:
    return px[1:] / px[:-1] - 1.0


def _finite(v) -> bool:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return True
    if isinstance(v, (int, float)):
        return bool(np.isfinite(v))
    if isinstance(v, dict):
        return all(_finite(x) for x in v.values())
    if isinstance(v, list):
        return all(_finite(x) for x in v)
    return True


def audit_ticker(t: str, last_day: pd.Timestamp) -> list:
    s = _MASTER[t].dropna() if t in _MASTER.columns else pd.Series(dtype=float)
    if len(s) < ing.MIN_ROWS or (s <= 0).any():
        return [("TICK-01", t, "FAIL", f"Yahoo returns {len(s)} usable days for {t}")]
    out = []
    age = (last_day - s.index[-1]).days
    if age > 7:
        out.append(("TICK-01", t, "FAIL", f"{t} last priced {s.index[-1].date()}, {age}d stale"))
    rep = validate_prices(s.to_frame())
    if not rep["passed"]:
        bad = [c["detail"] for c in rep["checks"] if c["status"] == "FAIL"]
        out.append(("TICK-01", t, "FAIL", f"{t} data-quality gate: {bad[:2]}"))
    if len(s) < MAP_MIN_DAYS + 120:
        out.append(("TICK-02", t, "WARN", f"{t} has {len(s)} days: the map may refuse it"))
    return out or [("TICK-01", t, "PASS", "")]


def audit_pair(pair: tuple) -> list:
    a, b = pair
    scope = f"{a}+{b}"
    out = []

    def rec(cid, ok, detail="", status=None):
        out.append((cid, scope, status or ("PASS" if ok else "FAIL"), "" if ok else detail))

    # A symbol Yahoo has no data for fails TICK-01 once; its 300-odd pairs are
    # counted here as SKIP rather than repeating that failure (a crash on the
    # first live run, 2026-10-17: BK and K had been delisted).
    if any(t not in _MASTER.columns or _MASTER[t].dropna().empty for t in (a, b)):
        rec("PAIR-01", True, status="SKIP")
        return out

    # -- PAIR-01 the app's own fetch returns the raw closes -------------------
    try:
        px = ing.fetch_prices([a, b], period="2y")
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        rec("PAIR-01", False, f"fetch_prices raised {type(exc).__name__}: {exc}")
        return out
    raw = _MASTER[[a, b]].dropna(how="all")
    raw = raw[raw.index.dayofweek < 5]      # a 252-day year: crypto weekends fold into Monday
    both = raw.dropna()
    kept = px.index
    missing = both.index.difference(kept)
    eq = all(np.allclose(px[c].to_numpy(), raw[c].ffill(limit=3).loc[kept].to_numpy(),
                         rtol=0, atol=1e-12) for c in (a, b))
    rec("PAIR-01", list(px.columns) == sorted([a, b]) and eq and not len(missing),
        f"columns {list(px.columns)}; values match raw {eq}; {len(missing)} shared days dropped")
    cols = list(px.columns)
    rets = ing.get_returns(px)
    r = np.column_stack([_rets(px[c].to_numpy()) for c in cols])

    # -- PAIR-02 every estimator, and the weights each gives ------------------
    # Two-asset ERC is inverse volatility whatever the correlation (Maillard,
    # Roncalli & Teiletche 2010), so the solver has a closed form to meet.
    for m in ESTIMATORS:
        try:
            cov, _ = estimate_covariance(rets, m)
            c = cov.loc[cols, cols].to_numpy()
            w = np.asarray(risk_parity_weights(cov.loc[cols, cols]))
            sa, sb = np.sqrt(c[0, 0]), np.sqrt(c[1, 1])
            ok = (np.allclose(c, c.T) and np.linalg.eigvalsh(c).min() > -1e-12
                  and abs(w.sum() - 1) < 1e-9 and abs(w[0] - sb / (sa + sb)) < 1e-6)
            if m == "Sample":
                ok = ok and np.allclose(c, np.cov(r, rowvar=False, ddof=1) * 252,
                                        rtol=1e-9, atol=1e-14)
            rec("PAIR-02", ok, f"{m}: weights {np.round(w, 6)} vs inverse vol "
                f"{sb / (sa + sb):.6f}; min eigenvalue {np.linalg.eigvalsh(c).min():.2e}")
        except Exception as exc:  # noqa: BLE001
            rec("PAIR-02", False, f"{m} raised {type(exc).__name__}: {exc}")

    spy = _MASTER["SPY"].dropna()
    mkt = pd.Series(_rets(spy.to_numpy()), index=spy.index[1:])
    w_eq = np.array([0.5, 0.5])
    for short in (False, True):
        side = "short" if short else "long"
        port = pd.Series(r @ w_eq * (-1 if short else 1), index=rets.index)

        # -- PAIR-03 historical VaR / CVaR of the book ------------------------
        q = float(-np.percentile(port.to_numpy(), 5))
        tail = float(-port[port <= -q].mean())
        rec("PAIR-03", abs(var(port) - q) < 1e-12 and abs(cvar(port) - tail) < 1e-12
            and cvar(port) >= var(port),
            f"{side}: VaR95 {var(port):.5f} vs {q:.5f}; CVaR95 {cvar(port):.5f} vs {tail:.5f}")

        try:
            pl = build_map_payload(rets, w_eq, cols, short)
        except ValueError as exc:
            if SHORT_HISTORY in str(exc):    # refused, never faked
                rec("MAP-00", True, status="INFO")
                continue
            rec("MAP-00", False, f"{side}: {type(exc).__name__}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            rec("MAP-00", False, f"{side}: {type(exc).__name__}: {exc}")
            continue
        rec("MAP-00", _finite(pl), f"{side}: non-finite number in the payload")

        # -- MAP-01 the book's dot: 63d beta and 21d vol at the last date -----
        j = pd.concat([port.rename("p"), mkt.rename("m")], axis=1).dropna()
        p63, m63 = j["p"].to_numpy()[-BETA_WINDOW:], j["m"].to_numpy()[-BETA_WINDOW:]
        b_now = np.cov(p63, m63, ddof=1)[0, 1] / np.var(m63, ddof=1)
        v_now = np.std(j["p"].to_numpy()[-VOL_WINDOW:], ddof=1) * np.sqrt(252)
        rec("MAP-01", abs(b_now - pl["base"]["beta"]) <= 5e-4 + 1e-9
            and abs(v_now - pl["base"]["vol"]) <= 5e-5 + 1e-9,
            f"{side}: book beta {b_now:.4f} / vol {v_now:.5f} vs map "
            f"{pl['base']['beta']} / {pl['base']['vol']}")

        # -- MAP-02 every dot: full-sample beta to SPY and annualized vol -----
        anc = ing.fetch_prices([t for t in DEFENSIVE_ANCHOR_TICKERS if t not in cols],
                               period="2y")      # a held anchor is a holding
        frame = rets.join(anc.pct_change().dropna(), how="inner").dropna(axis=1)
        common = frame.index.intersection(mkt.index)
        mv = np.var(mkt.loc[common].to_numpy(), ddof=1)
        want, bad = set(frame.columns), []
        for u in pl["assets"]:
            t = u["t"]
            if t not in frame.columns:
                bad.append(f"{t} drawn but not in the data")
                continue
            x = frame[t]
            bt = np.cov(x.loc[common].to_numpy(), mkt.loc[common].to_numpy(), ddof=1)[0, 1] / mv
            if short and t in cols:
                bt = -bt                     # a short of a name carries the opposite beta
            vt = np.std(x.to_numpy(), ddof=1) * np.sqrt(252)
            if abs(bt - u["b"]) > 5e-4 + 1e-9 or abs(vt - u["v"]) > 5e-5 + 1e-9:
                bad.append(f"{t}: audit {bt:.4f}/{vt:.5f} vs map {u['b']}/{u['v']}")
            if u["f"] != ("Universe" if t in cols else "Anchor pool"):
                bad.append(f"{t} labelled {u['f']}")
        drawn = {u["t"] for u in pl["assets"]}
        rec("MAP-02", not bad and drawn == want,
            f"{side}: {bad[:3]}; not drawn {sorted(want - drawn)}")

        # -- MAP-03 hazard perimeter = the disclosed policy -------------------
        b0, v0 = pl["base"]["beta"], pl["base"]["vol"]
        hv = round(min(HAZARD_VOL_CAP, max(HAZARD_VOL_FLOOR, HAZARD_VOL_MULT * v0)), 2)
        hb = (round(-min(HAZARD_BETA_CAP, -b0 + HAZARD_BETA_ADD), 2) if short
              else round(min(HAZARD_BETA_CAP, b0 + HAZARD_BETA_ADD), 2))
        rec("MAP-03", pl["hazard"] == {"volMax": hv, "betaMax": hb}
            and pl["betaSign"] == (-1 if short else 1),
            f"{side}: hazard {pl['hazard']} vs policy vol {hv} beta {hb}")

        # -- MAP-04 everything drawn sits inside the plotted axes -------------
        d = pl["domain"]
        pts = [(u["b"], u["v"]) for u in pl["assets"]] + [(b0, v0), (hb, hv)]
        rec("MAP-04", all(d["b0"] <= x <= d["b1"] and d["v0"] <= y <= d["v1"] for x, y in pts),
            f"{side}: domain {d} misses a point")

        # -- MAP-05 every linkage line, replayed from raw returns -------------
        flyer = str(rets.std().idxmax())
        bt_frame = frame.copy()
        if short:
            bt_frame[flyer] = -bt_frame[flyer]
        ans = [p_["an"] for p_ in pl["pairs"]]
        n_expect = min(4, frame.shape[1] - 1) if flyer in frame.columns else 0
        bad, _ = replay_linkages(bt_frame, pl["pairs"])
        rec("MAP-05", not bad and len(ans) == n_expect and len(set(ans)) == len(ans)
            and all(p_["fl"] == flyer and p_["an"] != flyer for p_ in pl["pairs"]),
            f"{side}: flyer {flyer}, anchors {ans} (expected {n_expect}); {bad[:2]}")
    return out


def run(path: str, shard: str) -> int:
    i, n = (int(x) for x in shard.split("/"))
    _init(path)
    rows = []
    if i == 0:
        last = _MASTER.dropna(how="all").index[-1]
        for t in UNIVERSE:
            rows += audit_ticker(t, last)
    pairs = [p for k, p in enumerate(combinations(UNIVERSE, 2)) if k % n == i]
    print(f"shard {i}/{n}: {len(pairs)} pairs", flush=True)
    with mp.Pool(mp.cpu_count(), initializer=_init, initargs=(path,)) as pool:
        for k, res in enumerate(pool.imap_unordered(audit_pair, pairs, chunksize=8), 1):
            rows += res
            if k % 500 == 0:
                print(f"  {k}/{len(pairs)}", flush=True)
    return _report(rows, len(pairs), Path(path).parent / f"pair_audit_{i}.json")


def _report(rows: list, n_pairs: int, out: Path) -> int:
    counts: dict = {}
    for cid, _, status, _ in rows:
        counts.setdefault(cid, {}).setdefault(status, 0)
        counts[cid][status] += 1
    notes = [dict(zip(("id", "scope", "status", "detail"), x, strict=True))
             for x in rows if x[2] in ("FAIL", "WARN")]
    out.write_text(json.dumps({"pairs": n_pairs, "counts": counts, "notes": notes}, indent=1))
    return _print(n_pairs, counts, notes)


def _print(n_pairs: int, counts: dict, notes: list) -> int:
    print(f"\n{n_pairs} pairs, long and short")
    for cid in sorted(counts):
        print(f"  {cid}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts[cid].items())))
    fails = [x for x in notes if x["status"] == "FAIL"]
    for x in fails[:60]:
        print(f"  FAIL {x['id']} [{x['scope']}] {x['detail']}")
    for x in [x for x in notes if x["status"] == "WARN"][:60]:
        print(f"  warn {x['id']} [{x['scope']}] {x['detail']}")
    print(f"\n{'FAILED: ' + str(len(fails)) if fails else 'every pair passes'}")
    return 1 if fails else 0


def summary(folder: str) -> int:
    n, counts, notes = 0, {}, []
    files = sorted(Path(folder).glob("pair_audit_*.json"))
    for f in files:
        d = json.loads(f.read_text())
        n += d["pairs"]
        notes += d["notes"]
        for cid, by in d["counts"].items():
            for k, v in by.items():
                counts.setdefault(cid, {}).setdefault(k, 0)
                counts[cid][k] += v
    total = len(list(combinations(UNIVERSE, 2)))
    print(f"{len(files)} shards")
    rc = _print(n, counts, notes)
    if n != total:
        print(f"FAILED: {n} of {total} pairs audited - a shard is missing")
        return 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").add_argument("out")
    r = sub.add_parser("run")
    r.add_argument("prices")
    r.add_argument("--shard", default="0/1")
    sub.add_parser("summary").add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "download":
        return download(a.out)
    if a.cmd == "run":
        return run(a.prices, a.shard)
    return summary(a.folder)


if __name__ == "__main__":
    sys.exit(main())
