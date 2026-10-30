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

A pass warms EVERY pull the page makes for a preset, not only its 2y prices
(John, 2026-10-20: the first switch to Semiconductors after a deploy still
took 93.1s - the 2y pull was warm, but the full histories behind the Book and
the Grit tab, the ten years behind the tail fit and the page-wide pulls were
all cold, and Railway's disk is wiped on every deploy).
Each pull goes through the same function with the same arguments main.py
uses, so the cache keys match exactly and no number can differ from a cold
pull; `test_the_warm_pass_covers_every_fetch_a_preset_page_makes` records the
real page's fetches and fails if a pass misses one, the Balance tab's default
crisis pair included (`default_crisis_pair`: the page's own rule). Not
covered: baskets a visitor types in themselves.

Budget: src/ingestion keeps each name's download per period, so a pass
downloads each name once per period however many presets share it - about
450 name-downloads (2y, full history and 10y of ~150 names) plus the page-wide
pulls - inside the
5,000-a-day netguard cap this process keeps for itself (the cap is per
process). The same files serve a basket a visitor types: only its names the
pass never saw are downloaded. A 1s pause between pulls keeps it from
bursting at Yahoo.
"""

import argparse
import time

from src.conviction import AI_CAPEX_BASKET, BENCHMARK
from src.factors import _PROXIES as FACTOR_PROXIES
from src.grit import grit_scores
from src.ingestion import (CACHE_MAX_AGE_HOURS, PRESETS, fetch_dollar_volume,
                           fetch_prices, fetch_risk_free_rate, flat_columns,
                           get_returns)
from src.netguard import BudgetExhausted, budget_status
from src.pairing import DEFENSIVE_ANCHOR_TICKERS, anchor_rank
from src.topology import anchor_frame

PAUSE_SECONDS = 1.0     # between pulls, so a pass never bursts at Yahoo


def preset_pulls(tickers: list[str]) -> list[tuple[str, object]]:
    """Every pull the page makes for one preset basket, as (what, call(age)),
    with the arguments main.py passes - so each lands on the page's own key."""
    held = set(tickers)
    pool = [a for a in DEFENSIVE_ANCHOR_TICKERS if a not in held]   # topology.anchor_frame
    pulls = [
        ("2y prices", lambda age: fetch_prices(tickers, period="2y", max_age_hours=age)),
        # The Book and the Balance tab score grit on the universe plus the
        # defensive shelf (main.py load_grit / _bv_grit), the Grit tab on the
        # universe alone: two keys, both full history.
        ("full history + anchors", lambda age: fetch_prices(
            sorted(held | set(DEFENSIVE_ANCHOR_TICKERS)), period="max", align=False,
            max_age_hours=age)),
        ("full history", lambda age: fetch_prices(tickers, period="max", align=False,
                                                  max_age_hours=age)),
        ("10y (tail fit)", lambda age: fetch_prices(tickers, period="10y", align=False,
                                                    max_age_hours=age)),
    ]
    if pool:
        pulls.append(("anchor pool", lambda age: fetch_prices(pool, period="2y",
                                                              max_age_hours=age)))
    # The Balance tab replays its default pair through every named crisis
    # (src.scenarios.replay_returns: the pair's own full history). Left cold,
    # it was the last Yahoo wait of a first switch (John, 2026-10-20).
    pulls.append(("crisis pair", lambda age: fetch_prices(
        default_crisis_pair(tickers, age), period="max", align=False,
        max_age_hours=age)))
    return pulls


def default_crisis_pair(tickers: list[str], age: float) -> list[str]:
    """The flyer and anchor the Balance tab opens on, picked the way main.py
    picks them: the most volatile name over the 2y window, and the top of
    anchor_rank over the universe plus the defensive pool, scored with grit
    (long side - the page's default). Every input is a pull this pass has
    already warmed, so this costs CPU, not Yahoo."""
    returns = get_returns(fetch_prices(tickers, period="2y", max_age_hours=age))
    returns = returns.drop(columns=flat_columns(returns))
    flyer = (returns.std() * 252 ** 0.5).idxmax()
    try:
        frame = anchor_frame(returns)
    except Exception:  # noqa: BLE001 - the page falls back the same way
        frame = returns
    try:
        grit = grit_scores(sorted(frame.columns))["scores"]["grit_score"]
    except Exception:  # noqa: BLE001
        grit = None
    anchor = anchor_rank(frame, flyer, grit=grit, direction="long").index[0]
    return [flyer, anchor]


def page_pulls() -> list[tuple[str, object]]:
    """The pulls every page view makes whatever the basket."""
    anchors = list(DEFENSIVE_ANCHOR_TICKERS)
    return [
        ("benchmark full history", lambda age: fetch_prices([BENCHMARK], period="max",
                                                            align=False, max_age_hours=age)),
        ("recovery race", lambda age: fetch_prices(AI_CAPEX_BASKET + [BENCHMARK], period="max",
                                                   align=False, max_age_hours=age)),
        ("market 2y", lambda age: fetch_prices([BENCHMARK], period="2y", max_age_hours=age)),
        ("factor proxies", lambda age: fetch_prices(FACTOR_PROXIES, period="2y",
                                                    max_age_hours=age)),
        ("defensive shelf", lambda age: fetch_prices(anchors, period="2y", align=False,
                                                     max_age_hours=age)),
        ("defensive shelf volume", lambda age: fetch_dollar_volume(anchors, period="2y",
                                                                   max_age_hours=age)),
        ("anchors 2y", lambda age: fetch_prices(anchors, period="2y", max_age_hours=age)),
        ("T-bill rate", lambda age: fetch_risk_free_rate(max_age_hours=age)),
    ]


def warm(name: str, tickers: list[str], drill: bool = False,
         max_age: float | None = None) -> dict:
    """Warm one preset universe - every pull its page makes - and return what
    it cost."""
    return _run(name, preset_pulls(tickers), drill, max_age)


def _run(name: str, pulls: list, drill: bool = False,
         max_age: float | None = None) -> dict:
    """Run a list of pulls, returning what they cost."""
    started = time.monotonic()
    spent_before = budget_status()["spent"]
    # max_age_hours=0 in drill mode forces real fetches, so the timing is a
    # cold-start number rather than a parquet read.
    if max_age is None:
        max_age = 0 if drill else CACHE_MAX_AGE_HOURS
    done, failed = 0, []
    for what, call in pulls:
        try:
            call(max_age)
            done += 1
        except BudgetExhausted as exc:
            failed.append(f"{what}: budget: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - a warm-up must never take a deploy down
            failed.append(f"{what}: {type(exc).__name__}: {exc}")
        time.sleep(PAUSE_SECONDS)
    detail = f"{done}/{len(pulls)} pulls" + (f" - {failed[0]}" if failed else "")
    return {"universe": name, "ok": not failed, "detail": detail,
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
    for name, pulls in [("Every page view", page_pulls())] + [
            (name, preset_pulls(tickers)) for name, tickers in PRESETS.items()]:
        row = _run(name, pulls, drill, max_age)
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
