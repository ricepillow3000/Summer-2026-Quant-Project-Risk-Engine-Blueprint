"""
Post-deploy cache warm, and the executable form of this app's restore drill.

`RUNBOOK.md` claims recovery is "redeploy the last good commit and let the
cache refill". A claim nobody runs is a claim nobody can trust, so this is that
claim as a command: fetch every preset universe the app ships and report what
each one cost in wall-clock time and outbound requests.

Two uses:
- **After a deploy or a restore**, run it so the first visitor lands on a warm
  cache instead of paying for eleven cold Yahoo fetches.
- **As a drill**, run it with --drill and read the total: that number IS the
  recovery time for the only regenerable state this app has.

    python -m tools.warm_cache            # warm every preset
    python -m tools.warm_cache --drill    # force real fetches, time a cold refill
    python -m tools.warm_cache --every 5  # keep warming, every 5 hours (Procfile)

--every runs inside the web process's container on purpose: the cache is that
container's own disk, so a separate cron service would warm a disk no visitor
reads. Each pass refetches anything older than (max age - interval), so no
preset expires between passes (live audit run 24: an expired cache made the
first Semiconductors pick on Railway take 101.6s).
"""

import argparse
import time

from src.ingestion import CACHE_MAX_AGE_HOURS, PRESETS, fetch_prices
from src.netguard import BudgetExhausted, budget_status


def warm(name: str, tickers: list[str], drill: bool = False,
         max_age: float | None = None) -> dict:
    """Fetch one universe, returning what it cost."""
    started = time.monotonic()
    spent_before = budget_status()["spent"]
    try:
        # max_age_hours=0 in drill mode forces a real fetch, so the timing is a
        # cold-start number rather than a parquet read.
        if max_age is None:
            max_age = 0 if drill else CACHE_MAX_AGE_HOURS
        prices = fetch_prices(tickers, period="2y", max_age_hours=max_age)
        detail, ok = f"{prices.shape[0]}d x {prices.shape[1]} names", True
    except BudgetExhausted as exc:
        detail, ok = f"budget: {exc}", False
    except Exception as exc:  # noqa: BLE001 - a warm-up must never take a deploy down
        detail, ok = f"{type(exc).__name__}: {exc}", False
    return {"universe": name, "ok": ok, "detail": detail,
            "seconds": round(time.monotonic() - started, 2),
            "requests": budget_status()["spent"] - spent_before}


def pass_max_age(every_hours: float) -> float:
    """Oldest cache entry a repeating pass may leave alone: anything older
    would expire before the next pass comes round."""
    return max(0.0, CACHE_MAX_AGE_HOURS - every_hours)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", action="store_true",
                    help="ignore cache freshness and time a cold refill")
    ap.add_argument("--every", type=float, metavar="HOURS",
                    help="repeat forever, refreshing before anything expires")
    args = ap.parse_args()
    if args.every:
        while True:
            run_pass(max_age=pass_max_age(args.every))
            time.sleep(args.every * 3600)
    return run_pass(drill=args.drill)


def run_pass(drill: bool = False, max_age: float | None = None) -> int:
    """Warm every preset once and print what each cost."""
    rows = []
    for name, tickers in PRESETS.items():
        row = warm(name, tickers, drill, max_age)
        rows.append(row)
        flag = "ok  " if row["ok"] else "FAIL"
        print(f"{flag} {row['seconds']:6.2f}s {row['requests']:3d} req  "
              f"{name[:38]:38} {row['detail'][:60]}", flush=True)

    total = sum(r["seconds"] for r in rows)
    failed = [r for r in rows if not r["ok"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} universes warm in "
          f"{total:.1f}s; budget {budget_status()}")
    if drill:
        print(f"Cold-refill recovery time for the whole cache: {total:.1f}s - "
              "that is the RTO for the only regenerable state this app has.")
    for row in failed:
        print(f"  FAILED {row['universe']}: {row['detail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
