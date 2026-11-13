"""
break it on purpose: plant known bugs, one at a time, and make sure a test goes red.

    python -m tools.mutation_check            # all of them, 4 at a time
    python -m tools.mutation_check M19 M34    # just these

green checks only mean something if they'd go red when the code is wrong. each
bug below is copied into a temp copy of the repo (never this checkout), the
engine and narrative tests run against it, and the first red test is named. a
bug no test catches is a hole: write the test that kills it (John, 2026-11-10;
the first run caught 21 of 36, the tests added after it catch all 36; 37-40
guard end-of-day, 41-45 the 2026-11-12 security review, 46-49 the
2026-11-13 audit).
takes ~40 minutes on 4 cores: run it after touching src/, serve.py or safe_err.
exit code 1 if any planted bug survives.
"""

import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# id, file, exact text, the bug, what it would cost a visitor
BUGS = [
    ("M01", "src/ingestion.py", r'[A-Z0-9][A-Z0-9.\-=]{0,14}$")', r'[A-Z0-9][A-Z0-9.\-=<>/ ]{0,14}$")', "ticker funnel lets html through"),
    ("M02", "src/ingestion.py", "    return valid[:MAX_UNIVERSE]", "    return valid", "no cap on basket size"),
    ("M03", "src/ingestion.py", "    return age <= datetime.timedelta(hours=max_age_hours)", "    return True", "cache never goes stale"),
    ("M04", "src/ingestion.py", '    with open(tmp, "w", encoding="utf-8") as fh:\n        json.dump(record, fh, indent=2)\n    os.replace(tmp, path)',
     '    with open(path, "w", encoding="utf-8") as fh:\n        json.dump(record, fh, indent=2)', "json record written in place"),
    ("M05", "src/ingestion.py", "    except (OSError, ValueError):   # no record yet", "    except (OSError,):   # no record yet", "torn provenance crashes the page"),
    ("M06", "src/ingestion.py", "prices.index.dayofweek < 5]", "prices.index.dayofweek < 7]", "weekends kept"),
    ("M07", "src/ingestion.py", "    dv[fut] = dv[fut] * 0.0", "    dv[fut] = dv[fut] * 1.0", "futures contracts read as dollars"),
    ("M08", "src/ingestion.py", ".iloc[-1]) / 100.0", ".iloc[-1]) / 10.0", "risk-free rate 10x"),
    ("M09", "src/ingestion.py", "    entries.sort()  ", "    entries.sort(reverse=True)  ", "cache prune drops the wrong files"),
    ("M10", "src/netguard.py", '    return parsed.scheme == "https" and ', "    return ", "plain http allowed out"),
    ("M11", "src/netguard.py", "*args, allow_redirects=False, **kwargs)", "*args, allow_redirects=follow, **kwargs)", "curl follows redirects before they're checked"),
    ("M12", "src/netguard.py", '        if _SPENT["count"] >= MAX_REQUESTS_PER_DAY:', '        if _SPENT["count"] > MAX_REQUESTS_PER_DAY:', "one request past the cap"),
    ("M13", "src/netguard.py", '    if status["remaining"] < needed:', '    if status["remaining"] < 0:', "spent cap not caught up front"),
    ("M14", "src/topology.py", '.replace("<", "\\\\u003c")', '.replace("<", "<")', "map json can close its script"),
    ("M15", "src/topology.py", "json.dumps(payload, allow_nan=False)", "json.dumps(payload, allow_nan=False, ensure_ascii=False)", "line separator breaks the map js"),
    ("M16", "src/risk.py", "    tail = r[r <= threshold]", "    tail = r[r < threshold]", "cvar skips the var day"),
    ("M17", "src/risk.py", "    return float(-tail.mean())", "    return float(tail.mean())", "cvar sign flipped"),
    ("M18", "src/risk.py", "port_returns, (1 - confidence) * 100))", "port_returns, confidence * 100))", "var reads the gains"),
    ("M19", "src/risk.py", ".rolling(win).quantile(q).shift(1)", ".rolling(win).quantile(q)", "backtest peeks at the day it judges"),
    ("M20", "src/risk.py", "    crit = 3.841", "    crit = 6.635", "kupiec at 99% not 95%"),
    ("M21", "src/risk.py", "bool(lr_ind <= 3.841)", "bool(lr_ind <= 38.41)", "christoffersen never fails"),
    ("M22", "src/risk.py", "np.clip(round(tau), 2, cap)", "np.clip(round(tau), 1, cap)", "block bootstrap collapses to iid"),
    ("M23", "src/strategies.py", "leverage = target_vol / realized if", "leverage = realized / target_vol if", "vol target inverted"),
    ("M24", "src/strategies.py", "    b = np.ones(n) / n", "    b = np.arange(1, n + 1) / n", "risk parity budgets unequal"),
    ("M25", "src/state_calibration.py", "K_CLAMP = (1.0, 3.0)", "K_CLAMP = (0.5, 3.0)", "map rings can narrow"),
    ("M26", "src/liquidity.py", "    capacity = participation_rate * adv_v", "    capacity = adv_v", "participation cap ignored"),
    ("M27", "src/liquidity.py", 'full_exit_days = float(finite["days"].max())', 'full_exit_days = float(finite["days"].mean())', "exit time = average leg"),
    ("M28", "src/grit.py", "    out = pd.Series(0.0, index=raw.index)", "    out = pd.Series(0.5, index=raw.index)", "unknown grit gets credit"),
    ("M29", "src/grit.py", "    x = raw if higher_is_better else -raw", "    x = raw", "grit ranks backwards"),
    ("M30", "src/grit.py", "MIN_HISTORY_DAYS = 400", "MIN_HISTORY_DAYS = 40", "grit scores thin history"),
    ("M31", "src/data_quality.py", "prices.le(0).sum().sum()", "prices.lt(0).sum().sum()", "a zero price passes"),
    ("M32", "src/data_quality.py", "EXTREME_MOVE_THRESHOLD = 0.50", "EXTREME_MOVE_THRESHOLD = 5.0", "extreme moves never flagged"),
    ("M33", "src/data_quality.py", '"PASS" if staleness_days <= 1 else "WARN"', '"PASS" if staleness_days <= 100 else "WARN"', "stale feed reads fresh"),
    ("M34", "serve.py", '    "x-content-type-options": "nosniff",', '    "x-content-type-options": "",', "nosniff dropped"),
    ("M35", "serve.py", 'HIDDEN = ("/_stcore/metrics", "/_stcore/upload_file")', 'HIDDEN = ("/_stcore/upload_file",)', "metrics endpoint exposed"),
    ("M36", "main.py", 'return f"{type(exc).__name__}; quote', 'return f"{exc}; quote', "server paths shown to visitors"),
    ("M37", "src/ingestion.py", "else _finished_rows(_own_rows(raw, t, alone=len(ask) == 1), t))", "else _own_rows(raw, t, alone=len(ask) == 1))", "today's live price kept as a close"),
    ("M38", "src/ingestion.py", "    if fetched < last_close_boundary():", "    if False:", "a pre-close pull hides the close"),
    ("M39", "src/ingestion.py", "CLOSE_FINAL = datetime.time(16, 30)", "CLOSE_FINAL = datetime.time(9, 30)", "bar kept from the opening bell"),
    ("M40", "src/ingestion.py", 'ALL_DAY = ("=X", "=F", "-USD")', "ALL_DAY = ()", "fx bar kept before its day ends"),
    ("M41", "serve.py", 'HIDDEN = ("/_stcore/metrics", "/_stcore/upload_file")', 'HIDDEN = ("/_stcore/metrics",)', "uploads park files in memory"),
    ("M42", "serve.py", "            code = 1009 if size > MAX_MESSAGE_BYTES else 1013", "            code = 1013", "a 16 MB message reaches the app"),
    ("M43", "serve.py", '    addr = hops[-1] if hops else', '    addr = hops[0] if hops else', "client picks its own address"),
    ("M44", "src/ingestion.py", "                   if clock - _NO_DATA.get(", "                   if True or clock - _NO_DATA.get(", "made-up names spend the budget"),
    ("M45", "serve.py", '    return str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else addr', "    return addr", "one ipv6 block = 2^64 addresses"),
    ("M46", "src/ingestion.py", "    return hashlib.sha256(text.encode()).hexdigest()", "    return hashlib.sha256(text.encode()).hexdigest()[:12]", "a stranger's basket takes a preset's cache slot"),
    ("M47", "src/ingestion.py", "             | set(DEFENSIVE_ANCHOR_TICKERS))", "             | set())", "the anchor shelf pruned out of the cache"),
    ("M48", "main.py", '        log_no_data(SESSION_REF, "load_universe")', '        log_incident(SESSION_REF, "load_universe", exc)', "a script buries real errors under typos"),
    ("M49", "src/ingestion.py", "            ask = [t for t in missing if t in known]", "            ask = missing", "a real name beside made-up ones triples their cost"),
]
SKIP = shutil.ignore_patterns(".git", "data", "logs", "audit", ".claude", "__pycache__", "node_modules")


def first_red(cmd: list[str], cwd: Path) -> str | None:
    """the first failing test's line, or None when everything passes."""
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,  # nosec B603
                          text=True, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}) as p:
        for line in p.stdout:
            if line.startswith(("FAIL ", "ERROR ")):
                p.kill()                    # one red test is enough: the bug is caught
                return line.strip()
    return None


def run(bug) -> tuple[str, str | None, str]:
    bid, rel, old, new, why = bug
    with tempfile.TemporaryDirectory(prefix=f"mut-{bid}-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=SKIP)
        f = copy / rel
        text = f.read_text(encoding="utf-8")
        if text.count(old) != 1:
            return bid, f"STALE: the text to break is not in {rel} exactly once - update BUGS", why
        f.write_text(text.replace(old, new), encoding="utf-8")
        red = (first_red([sys.executable, "-m", "tests.test_engine"], copy)
               or first_red([sys.executable, "-m", "tests.test_narrative"], copy))
        return bid, red, why


def main(argv: list[str]) -> int:
    bugs = [b for b in BUGS if not argv or b[0] in argv]
    survived = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for bid, red, why in pool.map(run, bugs):
            if red and not red.startswith("STALE"):
                print(f"caught    {bid} {why}  <- {red[:110]}", flush=True)
            else:
                survived += 1
                print(f"SURVIVED  {bid} {why}  {red or '(every test stayed green)'}", flush=True)
    print(f"\n{len(bugs) - survived}/{len(bugs)} planted bugs caught")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
