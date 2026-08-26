# Operator runbook

Everything an operator needs when something is wrong, in the order it is
usually needed. One person runs this; the procedures assume that.

## 0. Where state lives

| State | Durable? | Restore |
|---|---|---|
| Code and config | Yes - git, mirrored on GitHub | `git checkout <sha>` and redeploy |
| Market-data cache (`data/*.parquet`) | No - derived, regenerable | Delete it; the next page load refetches, or run `python -m tools.warm_cache` |
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
4. **Public** - promote to the production URL, unset `MELEONA_CHANNEL`, and
   put it on the resume.

Set `MELEONA_CHANNEL=beta` on the preview and beta deploys: it shows a beta
notice with a one-click "report this session" link whose subject already
carries the session reference. Unset on production. After any deploy or
restore, run `python -m tools.warm_cache` so the first visitor is not paying
for eleven cold fetches; `--drill` forces real fetches and prints the
cold-refill time, which is this app's recovery-time objective.

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

## 6b. Support mailbox setup (do this before the link is public)

Support runs on `meleona.support@gmail.com`, forwarded to the personal inbox.
Creating a Gmail account requires phone verification, so it is a manual step -
five minutes, once:

1. **Create it.** accounts.google.com -> create account -> "for my personal
   use" -> username `meleona.support`. If that exact username is taken, pick
   the next best (`meleona.risk.support`, `meleona.engine`) and set
   `MELEONA_SUPPORT_EMAIL` on the host to the one you got - the footer,
   maintenance page, feedback link and privacy panel all follow that variable.
2. **Forward it to yourself.** In the new mailbox: Settings -> See all
   settings -> Forwarding and POP/IMAP -> Add a forwarding address -> enter the
   personal address -> Google emails a confirmation link to the personal inbox
   -> click it -> back in the new mailbox select **Forward a copy of incoming
   mail** and *keep Gmail's copy in the Inbox* (so nothing is lost if
   forwarding is ever turned off).
3. **Label it on arrival.** In the personal inbox: Settings -> Filters ->
   Create filter -> To: `meleona.support@gmail.com` -> Apply label "Meleona
   support". Support mail then lands sorted rather than mixed in.
4. **Verify.** Send a test message to the project address from any other
   account and confirm it arrives in the personal inbox under that label.
   Do this before the public link goes out - a support address that bounces is
   worse than none.

If the address is ever retired, change `MELEONA_SUPPORT_EMAIL` on the host and
keep forwarding for a month so nothing in flight is lost.

## 6c. First deploy (Phase VI)

The app is a single web process with no database, no build step and no
secrets, so the whole deployment is: point a host at the repo, set two
environment variables, and let it run `Procfile`.

**Pinned for the host, not for this laptop:** `runtime.txt` (`python-3.13.7`)
and `.python-version` (`3.13`) - Render reads the former, Railway/nixpacks the
latter. Local development stays on whatever Python is installed; the pin
exists so a host never silently picks a version without wheels for
numpy/scipy/pyarrow/curl_cffi.

### Railway

1. railway.app -> New Project -> Deploy from GitHub repo -> pick
   `Summer-2026-Quant-Project-Risk-Engine-Blueprint`.
2. Variables -> add `MELEONA_CHANNEL=beta`. Add `MELEONA_SUPPORT_EMAIL` only if
   the mailbox username differs from `meleona.support@gmail.com`. Leave
   `MELEONA_MAINTENANCE` unset.
3. Settings -> Networking -> Generate Domain. That URL is the beta link.
4. Deploy. Railway injects `$PORT`; the `Procfile` already binds it on
   `0.0.0.0`.

### Render

1. render.com -> New -> Web Service -> connect the repo.
2. Runtime **Python 3**, Build `pip install -r requirements.txt`, Start
   `streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`.
3. Environment -> `MELEONA_CHANNEL=beta` (plus `MELEONA_SUPPORT_EMAIL` if
   needed). Health check path: `/_stcore/health`.
4. Free instances sleep when idle; the first hit after a sleep is slow and
   cold-fetches data. Warm it (below) before sharing the link with anyone.

### After every deploy

    python -m tools.warm_cache        # against the deployed host's shell, or locally to prime a shared cache

Then walk the page once: hero -> The Engine -> a crisis scenario -> Risk
Topology -> Lineage & Audit, and confirm the footer shows the support address
and a session reference. Trigger one deliberate error and confirm that
reference finds it in `logs/meleona.log` (section 3).

### Rolling back a bad deploy

Kill switch first (section 1) if visitors are seeing something broken, then
revert and push (section 2). Both hosts redeploy on push.

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
