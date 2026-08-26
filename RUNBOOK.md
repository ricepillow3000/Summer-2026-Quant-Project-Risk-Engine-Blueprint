# Operator runbook

Everything an operator needs when something is wrong, in the order it is
usually needed. One person runs this; the procedures assume that.

## 0. Where state lives

| State | Durable? | Restore |
|---|---|---|
| Code and config | Yes - git, mirrored on GitHub | `git checkout <sha>` and redeploy |
| Market-data cache (`data/*.parquet`) | No - derived, regenerable | Delete it; the next page load refetches |
| Crash logs (`logs/`) | No - rotating, size-capped | Nothing to restore; they are diagnostics |
| User data | **None exists** | Nothing to restore (no accounts, no database) |

There is no database, so there is no database restore. The recovery story is
"redeploy the last good commit", and the recovery point is whatever is on
`origin/main`.

## 1. Kill switch

Set an environment variable on the host - no redeploy, no code change:

    MELEONA_MAINTENANCE=1
    MELEONA_MAINTENANCE_NOTE=Back shortly - fixing a data feed issue.   (optional)

The next page load serves the maintenance notice with the support address and
stops before any market-data call. Unset it to bring the app back.

Use it when the feed returns garbage, a bad deploy is live and a rollback is
still minutes away, or an abuse problem needs to stop now.

## 2. Hotfix and rollback (the "OTA update" of a web app)

No app-store review sits in the path, so shipping a fix is a push:

    git revert <bad-sha>        # or commit the fix
    git push origin main        # host redeploys on push

Rollback to a known-good release:

    git checkout main
    git reset --hard <good-sha>
    git push --force-with-lease origin main

Prefer `git revert` (forward-only history) unless the bad commit leaked
something that must not stay in history. Re-run `python -m tests.test_engine`
before pushing; the suite includes a full-app boot test.

## 3. Crash triage

Every browser session gets a random reference shown in the footer. A user
reports it; you grep for it:

    grep 9F3A2B7C logs/meleona.log*

That returns the session-start line and any traceback logged during it. For a
sweep of recent failures with no specific reference:

    python -c "from src.observability import recent_incidents; print(chr(10).join(recent_incidents()))"

Logs rotate at 1 MB x 3 files and hold tickers and stack traces - no personal
data, which is what lets the privacy note say what it says.

## 4. Spend and rate-limit control

The engine calls no metered API and no LLM, so there is no credit balance to
top up. The cost it *can* incur is a Yahoo rate-limit or IP ban on a shared
free-tier egress address, so the budget is denominated in requests:

- `src/netguard.MAX_REQUESTS_PER_DAY` (default 5000) - checked before each
  fetch and charged per outbound request.
- On exhaustion the app serves cached data instead of hammering the feed, and
  the provenance panel keeps showing the true age of that data.
- `budget_status()` reports spent/remaining for the running process.
- Raise or lower the cap in that module: it is the single chokepoint every
  outbound call already passes through, so a future paid feed's key belongs
  behind the same counter.

Other bounds: `ingestion.MAX_UNIVERSE` (25 symbols), `CACHE_MAX_FILES` /
`CACHE_MAX_MB` (disk budget, oldest evicted first).

## 5. Phased release

1. **Local** - `python -m streamlit run main.py`, then `python -m tests.test_engine`
   (108 checks) and `python -m tests.batch_audit` (live-data audit).
2. **Preview** - deploy the branch to a preview URL. Same build, unlisted.
3. **Beta** - share that link with a handful of people. Watch
   `logs/meleona.log` and the budget counter; keep the kill switch one variable
   away.
4. **Public** - promote to the production URL and put it on the resume.

Roll back at any stage with section 2 - there is no staged store rollout to
wait on.

## 6. Reviewer / demo notes

Nothing to log into: the URL *is* the demo, every visitor sees the same public
market data, and no test account, promo code or sandbox key exists. A
five-minute reviewer path:

1. Land on the hero, click **The Grit Zone** - resilience ranked from real
   drawdowns.
2. Click **The Engine**, keep the default basket, read the headline CVaR
   verdict and the fan chart under it.
3. Switch the stress scenario to a real crisis window (e.g. COVID) and watch
   the verdict change; excluded assets are named, not hidden.
4. Open **Risk Topology** - the Monte Carlo drawn as a map, with the state
   model and its measured widening factor disclosed in the footnote.
5. Open **Lineage & Audit** for provenance, the run's audit trail, and the
   privacy panel with the session-clear control.

## 7. Email deliverability (only once a custom domain exists)

Support runs on a Gmail address today and the app **sends no email at all**, so
SPF, DKIM and DMARC do not apply - there is no sending domain to authenticate.
They matter the moment mail is sent from a domain you own
(`support@yourdomain`). At that point, at the DNS provider:

    ; SPF - authorise only the provider you actually send through
    yourdomain.                    TXT  "v=spf1 include:_spf.google.com -all"

    ; DKIM - publish the key your mail provider generates for you
    google._domainkey.yourdomain.  TXT  "v=DKIM1; k=rsa; p=<public-key-from-provider>"

    ; DMARC - monitor first, then tighten
    _dmarc.yourdomain.             TXT  "v=DMARC1; p=none; rua=mailto:dmarc@yourdomain; adkim=s; aspf=s"

Order that works: publish SPF and DKIM, run DMARC at `p=none` for a week or two
and read the aggregate reports, then move to `p=quarantine` and finally
`p=reject`. Going straight to `reject` before the reports are clean is how
legitimate mail gets silently dropped.
