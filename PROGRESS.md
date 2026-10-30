# Meliona - Session Handoff

Paste this into a new chat session to bring it up to speed instantly.

---

## What this project is

**Meliona** is an institutional-grade **Portfolio Optimization & Risk Engine**
with a live Streamlit dashboard, built by a 2nd-year Data Analytics student to
be recruiter-facing (the goal is a public live link on a resume). The name
merges "Durand" (Old French, "enduring/built to withstand hard times") with
"Mereoleona" (mother lioness) - it ties into the existing lion-crest logo and
"Pride · Integrity" tagline, and into the Grit Zone feature (below), which
scores assets on resilience and perseverance rather than market mood.

- **Repo:** https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint
- **Local path:** `C:\Users\john4\Claude\Projects\risk-engine`
- **Stack:** Python 3.14 · numpy, pandas, scipy · yfinance · Streamlit · Plotly
- **Ship target:** live deployment by Aug 23, 2026

## How to run it

```powershell
cd "C:\Users\john4\Claude\Projects\risk-engine"
python -m streamlit run main.py
```
Opens at http://localhost:8501. Note: `pip` is `python -m pip` on this machine.
**Only run ONE Streamlit server** - stale servers caused an ImportError once.
Kill strays: PowerShell → `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`

## Architecture

```
risk-engine/
├── main.py              # Streamlit app: universe → allocation → stress → verdict → breakdown
├── src/
│   ├── ingestion.py     # DataEngine: Yahoo fetch, per-universe freshness-aware cache, provenance, data_health
│   ├── analytics.py     # covariance, correlation, eigen-decomposition
│   ├── risk.py          # Monte Carlo, historical/parametric VaR, CVaR, Kupiec backtest
│   ├── factors.py       # named factor exposures (market/size/value/momentum via ETF proxies)
│   ├── strategies.py    # risk parity (ERC), vol targeting, risk-contribution decomposition
│   ├── scenarios.py     # historical regime replication (real crisis-window replay)
│   ├── grit.py          # Grit Zone: drawdown recovery, rolling consistency, regime resilience
│   ├── security_master.py  # ISIN + corporate actions (dividends/splits) via yfinance
│   └── data_quality.py  # automated schema/sanity validation gate on every price pull
├── assets/logo.svg      # crest: hexagon cut by an X (2 triangles, 3 rhombuses) on a ringed disc
├── .streamlit/config.toml  # beige/bronze institutional theme
├── requirements.txt · Procfile · README.md · .gitignore
```

## Status vs. roadmap

- ✅ **Phase I** - DataEngine with caching
- ✅ **Phase II** - covariance + eigen-decomposition
- ✅ **Phase III** - CVaR + Monte Carlo
- ✅ **Phase IV** - Streamlit dashboard, logo, institutional theme
- ✅ **Phase V** - configurable universe, data integrity, VaR backtest, named
  factors, provenance, risk parity, vol targeting, risk contribution,
  **true historical regime replication**, **liquidity modeling
  (days-to-liquidate via participation-rate model)**, **Merton jump-diffusion
  Monte Carlo engine (fat-tailed alternative to the bootstrap)**
- ✅ **Regression test suite** (`tests/test_engine.py`) - math-invariant tests +
  closed-form validation (Gaussian CVaR, Sharpe) + full-app boot test; run
  `python -m tests.test_engine` or `pytest`
- ✅ **Data engine speed** - one Yahoo download now fills both price + dollar-volume
  caches (cold load = 1 round-trip, not 2); Sharpe ratio vs live ^IRX risk-free rate
- ✅ **Visual upgrade** - themed Plotly charts (beige/bronze); hero Monte Carlo
  fan chart (outcome cone) under the verdict; themed histogram, risk-contribution,
  factor, and liquidity charts
- ✅ **3D outcome distribution** - rotatable Plotly surface (day x return-bin x
  density) under the fan chart, collapsed by default
- ✅ **Grit Zone** (`src/grit.py`) - a "grit score" concept as a counterpart to
  Fear & Greed: ranks each asset's OWN price history on drawdown-recovery
  speed/completeness, rolling 1-year consistency, and drawdown/recovery
  behavior across the real historical crisis windows in `scenarios.py`.
  Percentile-ranked RELATIVE to the chosen universe (no absolute "grit" scale
  claimed). 7 deterministic unit tests; verified against live data (history
  length and regimes-survived per ticker line up with real IPO dates).
- ✅ **Rebrand to Meliona** - page title, header wordmark, logo alt text, and
  docs updated from "Portfolio Risk Engine"
- ✅ **Presentation-style redesign** - the app used to open as one long stack
  of controls plus five nested expanders (a wall of information before you'd
  touched a slider). Restructured into a scroll-driven flow: hero pitch →
  dedicated "Grit Zone" showcase section (with anchor-scroll CTA buttons) →
  "the engine" (the existing interactive dashboard). The five supporting
  expanders (3D distribution, risk breakdown, Grit Zone detail, liquidity,
  provenance) are now one `st.tabs()` strip instead of stacked accordions.
  Added CSS scroll-reveal animation, smooth-scroll CTAs, hover-lift cards -
  pure CSS, no new toolchain, `streamlit run main.py` deploy story unchanged.
- ✅ **"Living" 3D particle effect** (`main.py`'s `living_surface_html()`) -
  the 3D Distribution tab is now a self-contained `st.iframe` component: raw
  plotly.js loaded from CDN, a density-weighted particle overlay
  (`scatter3d`) whose points jitter every 150ms, and a slow continuous
  camera auto-rotate that pauses on manual drag/touch and resumes after 4s
  idle. `st.components.v1.html` was already past its removal window
  (deprecated in favor of `st.iframe` in Streamlit 1.58) - built directly on
  `st.iframe` instead of the soon-to-be-removed API.
- ✅ **Security Master & Corporate Actions** (`src/security_master.py`) -
  free-tier reference data: ISIN via `yfinance`'s `Ticker.isin` (available
  for some tickers, honestly flagged `"unavailable"` for others - e.g. many
  US large caps don't expose one on the free feed), plus real dividend/split
  event history via `Ticker.dividends` / `Ticker.splits`. Verified against
  live data: AVGO's actual 2024-07-15 10:1 split, GOOGL's 2022-07-18 20:1
  split, NVDA's 2024-06-10 10:1 split, AMZN's 2022-06-06 20:1 split all came
  back correctly. SEDOL/CUSIP/merger history explicitly flagged as needing a
  paid vendor, not fabricated.
- ✅ **Data Quality validation gate** (`src/data_quality.py`) - hand-rolled
  (no new dependency) automated schema + sanity checks run on every price
  pull: DatetimeIndex/dtype/sort/duplicate checks, positivity, minimum row
  coverage, staleness, calendar-gap %, and an extreme-single-day-move flag
  (WARN, not FAIL - a real crash day should surface, not silently block).
  5 deterministic unit tests, each engineered to trip exactly one check.
- ✅ **Lineage & Audit tab** - extends the existing provenance panel with a
  session-scoped audit trail (`audit_log` built as the script runs: data
  fetch → allocation → stress scenario → Monte Carlo engine, each with real
  parameters). Explicitly labeled as session-scoped, not durable storage -
  same concept a full compliance system uses, at the scale this engine
  actually operates at.
- ✅ **Fast polling ("as live as honestly possible")** - `load_universe`'s
  Streamlit-session cache TTL dropped from 1h to 60s (re-checks the local
  disk cache far more often) while the underlying disk-cache freshness
  window stays at 6h (`ingestion.CACHE_MAX_AGE_HOURS`) so Yahoo itself isn't
  hit any harder - this is what actually protects against rate-limiting, not
  the session TTL. A `st.fragment(run_every="1s")` ticker shows "data pulled
  Xs ago," reruns only itself, not the whole page/Monte Carlo computation.
- ⬜ **Phase V polish remaining** - optional auto "executive summary";
  further UI refinement
- ✅ **Batch automation audit** (`tests/batch_audit.py`, 2026-08-26) - a second
  suite that runs the engine on LIVE data across all 11 preset universes and
  re-derives every number a second, independent way, writing
  `audit/batch_audit_<UTC>.{json,md}` where each check names its source paper.
  393 checks, 393 pass. Covers: engine invariants (CVaR>=VaR, covariance
  symmetry/PSD, portfolio-variance identity, ERC equalization, vol-target
  accuracy, both Monte Carlo engines, the Merton mean identity, liquidity
  monotonicity + LVaR, conditional EVT on the same 10y window the app uses);
  a 44-row walk-forward VaR backtest matrix (equal-weight and risk-parity
  books at 95%/99%, Kupiec LR re-derived from the likelihood ratio,
  Christoffersen independence, LR_cc = LR_uc + LR_ind); and a full audit of
  the Risk Topology map. The map audit runs the SHIPPED simulation - node
  slices the real math region out of `prototypes/war_room.html`
  (`tests/map_probe.mjs`) - and checks its end-state moments against the
  closed-form OU recursion, ring masses against their labels (68.3/95.4/99.7%),
  breach counting against the "touched the perimeter" caption, the HUD tail
  numbers against the path P&L, calm<base<stress ordering, plus a
  planted-parameter control run and a replay of every linkage statistic
  (rho, equal-risk weight, ES 97.5 solo/paired, drawdown cushion) from raw
  returns. `src/topology.py` was extracted from `main.py` so the audited
  payload IS the shipped payload.
  MAP-12 is the check that asks whether the terrain is TRUE rather than merely
  self-consistent: it walks history forward, refits the OU parameters on prior
  data only, and asks whether the realized state 30 days later landed inside
  the ring the map would have drawn. Across 11 universes (58 dates each) the
  68.3% ring covers 33-83% (median ~53%) and the 95.4% ring covers 64-98%
  (median ~81%) - the terrain is systematically NARROWER than reality, which
  is the overlapping-rolling-window bias `state_calibration.py` already
  discloses in prose, now measured. Reported as WARN, never a build gate.
  Live findings, not defects: 99% historical VaR takes zero breaches on
  several baskets (Kupiec rejects as too conservative) and 95% historical VaR
  under-covers on Futures (8.6%) and Global macro (8.0%) - independence
  passes, so it is a level problem, which is what the jump-diffusion and EVT
  engines exist for.

- ✅ **Map dispersion correction** (2026-08-26) - MAP-12 measured the Risk
  Topology terrain as systematically too narrow, so the shocks are now widened
  by a MEASURED factor rather than left disclosed-but-wrong.
  `state_calibration.dispersion_correction()` walks the book's own history
  forward - refit on prior data only, closed-form 30-day-ahead distribution,
  Mahalanobis score against what the state actually did - and returns
  `k = quantile_0.683(d) / sqrt(chi2.ppf(0.683, 2))`, the ratio that restores
  coverage exactly (widening both shocks by k scales every distance by 1/k, so
  no search and no fitted parameter). Clamped to [1.0, 3.0]: it widens a
  too-narrow terrain, never narrows a conservative one. The map footnote
  discloses k with its before/after coverage.
  Out-of-sample result (MAP-12, k re-measured at each date from outcomes
  already resolved by then): the nominal 68.3% ring went from a median 53.4%
  to 69.0% across 11 universes - Commodities 32.8% -> 70.7%, Futures
  34.5% -> 74.1%, Global macro 41.4% -> 67.2%. Baskets already wide enough
  (FX majors, Index core) get k = 1.00 and are untouched. Shipped k today
  ranges 1.00-1.92. Batch audit: 404 checks, 404 pass, 0 warn. Suite 103/103.
  Finding worth keeping: a state path drawn from EXACTLY the simulated model
  still under-covers (~47% inside the 68.3% ring), because the horizon
  distribution is built from point estimates - so the correction prices
  plug-in estimation error as well as the overlapping-window smoothing. That
  is now a regression test, not a footnote.

- ✅ **Pre-deploy security pass** (2026-08-26) - council-reviewed (four voices:
  architect/skeptic/pragmatist/critic) rather than checklist-applied. Verdict:
  of the four requested hardenings, two mapped onto real surfaces here and two
  did not, because this app has no accounts, no roles, no secrets, no database,
  no GraphQL and no webhooks. Threat model is **resource exhaustion and
  injection, not authorization**. Shipped:
  - **Egress allowlist** (`src/netguard.py`) - the SSRF control adapted to an
    app with no user-supplied URLs. Every `yf.download`/`yf.Ticker` call runs
    through a `curl_cffi` session that refuses any host outside
    `ALLOWED_HOSTS` and re-checks the final URL so a redirect chain cannot
    walk out. It subclasses curl_cffi, not `requests`: yfinance accepts a
    plain requests session but Yahoo answers it with `YFRateLimitError`, so
    the naive version would have broken the data path. The allowlist
    immediately caught a real dependency nobody had noticed - `Ticker.isin`
    queries Business Insider, not Yahoo - which is now an enumerated,
    reviewed host instead of a silent one.
  - **Complexity budget** (`MAX_UNIVERSE = 25`, `CACHE_MAX_FILES`,
    `CACHE_MAX_MB`) - the real analogue of a query depth/complexity limit for
    an app with no query language. Enforced at the ticker funnel so no caller
    can exceed it, and disclosed in the UI rather than silently truncating.
    Cache eviction is oldest-first over derived market data.
  - **Visitor is not an operator** (`toolbarMode = "viewer"`) - Streamlit's
    default toolbar hands every anonymous visitor Rerun / Clear cache /
    Settings against shared server state. That menu was the entire "admin"
    surface; it is gone. Residual stated honestly in the config: hiding the
    affordance does not remove the protocol message, so a crafted websocket
    frame could still force a rerun - bounded to DoS over public data.
  - **Script-splice escaping** (`src/topology.py`) - the map payload lands
    inside a `<script>`, and `json.dumps` does not escape `<`, so `</script>`
    in any engine string would have closed the block early. Now escaped to
    `<` at the splice, plus `html.escape` on ticker names at the one
    HTML sink that renders them.
  Deliberately NOT done: constant-time comparison (no secret is compared) and
  query depth limiting (no query language). Both are theatre without the
  surface, and read as checklist-following in an interview.
  5 new regression tests (108/108); batch audit still 404/404.

- ✅ **Launch readiness pass** (2026-08-26) - the pre-beta checklist, mapped
  onto an app with no accounts, no database, no payments, no LLM calls and no
  outbound email. Built: crash wire (`src/observability.py`: rotating local
  log, per-session reference in the footer, `log_incident()`; no third-party
  SDK, because one would contradict the privacy note it sits beside), kill
  switch (`MELIONA_MAINTENANCE=1` serves a maintenance page and stops before
  any data call), outbound spend cap (`netguard.MAX_REQUESTS_PER_DAY`, charged
  at the single chokepoint, degrading to cached data on exhaustion instead of
  hammering Yahoo), a "Your data" privacy panel with a session-clear control
  (the honest form of "delete my account" when there is no account), and a
  footer carrying support/privacy/terms plus the session reference. Docs:
  `PRIVACY.md`, `TERMS.md`, `SUPPORT.md`, `RUNBOOK.md` (kill switch, rollback,
  crash triage, budgets, phased release, reviewer walkthrough, and the
  SPF/DKIM/DMARC records for the day a custom domain exists) and
  `LAUNCH_CHECKLIST.md` (all 19 asks, each built or marked N/A with its
  reason and its trigger). N/A and stated as such: test sign-up, LLM credit
  balance, database restore, demo accounts, restore purchases, reinstall
  behaviour, sandbox keys, SPF/DKIM/DMARC today. 4 new regression tests
  (112/112) covering the crash wire, the spend cap, the kill switch ordering
  and the load-bearing claims in the policy docs.

- ✅ **Support mailbox + beta channel + restore drill** (2026-08-26) - support
  moved off the personal address to the project mailbox
  `meleona.support@gmail.com`, forwarded to the personal inbox, overridable per
  deployment with `MELIONA_SUPPORT_EMAIL`; a regression test now FAILS if the
  personal address reappears in the app or in any public document. Creating the
  mailbox and turning on forwarding is a manual, phone-verified step, so it is
  written up in `RUNBOOK.md` §6b and is a blocking item in
  `LAUNCH_CHECKLIST.md`. Also shipped from the remaining checklist items:
  `MELIONA_CHANNEL=beta` puts a beta notice plus a one-click "report this
  session" mailto (session reference already in the subject) on pre-release
  deploys only, and `tools/warm_cache.py` warms every preset after a deploy -
  `--drill` forces real fetches and prints the cold-refill time, which is the
  recovery-time objective for the only regenerable state this app has. 113/113.

- ✅ **Security review + bloat sweep** (2026-08-26) - support mailbox
  confirmed working end to end (test mail arrived in BOTH mailboxes), so the
  last blocking launch item is closed. Project security: `pip-audit` over the
  full dependency tree reports **no known vulnerabilities**; a secret sweep of
  every tracked file (key/token/private-key patterns) finds nothing; effective
  Streamlit settings verified at runtime rather than assumed - static serving
  off, XSRF and CORS on, tracebacks off, toolbar `viewer`. Added
  `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none` as a HOST env var rather than a
  config line, since `.streamlit/config.toml` is shared with local development
  where hot reload is wanted. Bloat sweep (AST scan for unused imports and
  unreferenced definitions across every module): removed four dead imports
  (`var`, `gpd_tail_fit` in main.py, `numpy` in conviction.py - its only `np.`
  was inside a docstring - `budget_status` in ingestion.py, `get_returns` in
  liquidity.py); zero unreferenced functions or classes anywhere. The one
  import that was dead because it was NOT wired, `log_incident`, is now
  connected to the three visitor-visible failure paths (data load, scenario
  replay, Risk Topology) so a degraded panel leaves a trace with the session
  reference instead of failing silently. `_score01` (grit) and `_rank01`
  (pairing) look like duplicates but differ in NaN and tie handling - left
  alone deliberately, since merging them would change published numbers.
  Repo slimmed: the ~3,900-line machine-readable JSON dump per audit run is
  now gitignored (regenerable with one command) and only the newest
  human-readable report is committed. Math re-verified after every edit:
  113/113 unit tests, and the live-data batch audit is 404 checks, 404 pass,
  0 warn, each traced to its source paper.

- ✅ **Accessibility + code-volume pass** (2026-08-26) - a council (skeptic /
  pragmatist / critic) was asked what to cut; all three landed on the same
  answer, and it was not "delete code": an AST sweep had already proved there
  is no dead code, so the target was **repetition and file size**, not lines.
  Shipped: the 778-line inline `<style>` block moved verbatim to
  `static/app.css`, loaded once by `load_css()` - **main.py 3,690 -> 2,904
  lines** with zero behaviour change. Not done, deliberately: collapsing the 36
  `unsafe_allow_html` blocks into helpers (the critic was right - under
  Streamlit rerun semantics that moves side effects, and there is no visual
  regression test to catch a silent break), and deleting
  `prototypes/war_room.html`, which the skeptic believed was retired - it is
  loaded by `src/topology.py` and rendered every session. Verified before
  acting on either claim.
  **Accessibility, and it found real defects.** Contrast was computed for every
  text colour against the ground it actually sits on: **seven failed**, worst
  2.01:1 against a 4.5:1 floor, including the tab strip, every panel eyebrow
  and the bronze links on the dark bands. All seven were lifted while leaving
  decorative bronze untouched, so the palette still reads as bronze and the
  words are legible. The Risk Topology map is canvas, invisible to a screen
  reader, so its five layers are now `aria-hidden` and `updateSummary()` writes
  the same numbers the HUD shows as sentences into an off-screen live region -
  generated from the simulation, so picture and description cannot drift.
  `ACCESSIBILITY.md` states the target (WCAG 2.2 AA), the measured table, and
  four known gaps (Streamlit's own widgets, Plotly charts, map interaction, no
  screen-reader user testing yet) - a statement claiming full conformance would
  be the overclaim it exists to prevent. Footer carries the statement plus a
  "report a barrier" route; barriers are treated as defects. Two regression
  tests: one recomputes every contrast ratio from the stylesheet, one checks the
  statement's claims against the code. 115/115.

- ✅ **Employer beta test + block bootstrap** (2026-08-26) - walked the app as
  a first-time visitor with `tools/demo_check.py`, a new adversarial harness
  that drives the real Streamlit app headlessly and reports what a stranger
  hits: crashes, non-finite numbers, contradictory headline figures, jargon
  with no explanation nearby, and holdings missing from the picker.
  **Four real defects, all now fixed:**
  1. **The Monte Carlo resampled days i.i.d.** Returns are near-uncorrelated in
     sign but cluster hard in magnitude, so shuffling single days scattered
     crash weeks across simulated years and thinned the one-year tail. Replaced
     with the **stationary block bootstrap** (Politis & Romano 1994); expected
     block length measured from the book's own |returns| autocorrelation
     (`mean_block_length`), clamped [2, 40], disclosed in the UI. On Mag-7 the
     measured block is 6 days and CVaR moves 25.1% -> 22.7%, so the old number
     was an artefact of an assumption the data does not support.
  2. **A basket could be analysed on 20 days of history.** Aligning intersects
     members onto common trading days, so EA - 17 days old after going private
     - cut a ten-name basket down with it, and the engine rendered a year of
     risk plus a `+nan` rolling correlation from it. `data_quality` had already
     returned `passed=False`; nothing read it. There is now a guard that names
     the offending ticker, excludes it, and re-runs on the rest, with a hard
     stop if too little history survives.
  3. **A visitor's own stock was not in the picker.** 15 of 15 common holdings
     (NFLX, AMD, BAC, PFE, ORCL...) were absent from every dropdown. Added nine
     sector baskets (semis, banks, healthcare, consumer, media, software,
     industrials, energy, high-volatility) and a 324-name `SUGGESTIONS`
     catalogue - the box always accepted any symbol, but you had to know the
     ticker by heart to use it.
  4. **Contrast failures in inline styles.** The stylesheet was clean but
     `main.py`'s inline `style=` attributes were not - four more colours below
     AA, including the engine's own "skip to the risk map" link at 2.50:1. The
     contrast test now scans both surfaces.
  Also added: a plain-English opener on the VaR backtest panel, and a "New
  here? Read this first" orientation strip before any control - a walkthrough
  found the dials arrive before any statement of what the page is for.
  118/118 tests (3 new proving the block bootstrap preserves clustering,
  leaves the mean alone, and reports its scheme); live audit 154/154 quick.

- ✅ **Guided descent + the workshop reopened** (2026-08-26) - John reported
  Signal Lab and Regime Atlas "do not show up". They were not missing: both sit
  inside the workshop expander, which a past declutter pass left COLLAPSED, so
  six tabs were invisible unless you thought to open a door. It now opens by
  default; the declutter it was built for is still served by the tab strip
  inside. All 12 tabs verified reachable.
  Motion, to his brief ("guiding them, a sense of calmness", "like an airline
  commercial"): the five CTA arrows are no longer `&darr;` glyphs but drawn
  SVG marks. At rest they are invisible; on hover the shaft draws downward over
  520ms and the head lands 340ms later, so the arrow assembles itself one
  stroke at a time in the reading direction. On click it completes instantly
  and is released - `arrow-descend` carries it down and fades it while the page
  glides after it. The glide itself went 1100ms -> 1500ms and its easing from
  easeInOutCubic to easeInOutQuart, so the last third is a drift rather than a
  stop. Measured live in Chrome: 0 -> 749px across 11 distinct positions in
  1.4s, slow-out profile intact.
  Verified in the browser rather than assumed, which caught two things worth
  recording: (1) this Chrome requests `prefers-reduced-motion: reduce`, so the
  arrow animation is correctly suppressed and the mark renders fully drawn -
  anyone with Windows animation effects off sees the static version by design;
  (2) an early "clicking does nothing" reading was a measurement artefact - the
  page was already parked at the destination (target top = 26px, exactly the
  glide's own offset). 118/118, walkthrough clean.

- ✅ **Phase VI** - live on Railway since 2026-09-03
  (https://summer-2026-quant-project-risk-engine-blueprint-production.up.railway.app)
- ✅ **Bug sweep + Cloudflare decision (2026-10-15)** - whole app driven
  headless through every preset, edge-case input and control. Fixed: the app
  crashed when started from any folder but the repo root (asset paths were
  cwd-relative); a public Refresh button let any visitor wipe every cache for
  everyone (`st.cache_data.clear()`) - removed; the 10,000-path Monte Carlo
  re-ran on every click - now cached (seeded, so identical); mistyped symbols
  showed a crash reference instead of "check the spelling" (`NoDataError`);
  one failed corporate-action fetch (yfinance returns None) blanked the whole
  Security Master tab - now that row reads unknown, never zero; the boot test
  failed offline instead of skipping; "quotes cached 60s" implied a pull every
  minute (Yahoo is hit every 6h) - relabelled. Cloudflare as reverse proxy:
  **no**, see RUNBOOK.md section 4 for the council's reasons and the revisit
  trigger.
- ✅ **Feature map, 2026-10-15** - every change of the day, and what now guards it:

  | change | kind | guarded by |
  |---|---|---|
  | Public "Refresh" button (wiped every visitor's cache) | removed | `test_no_visitor_control_clears_the_shared_cache` |
  | Streamlit's top header bar | removed | `test_streamlit_header_bar_is_hidden`, `site_check` |
  | WBD in the Streaming preset (dead on Yahoo) | removed | `live_provenance` preset check |
  | Cached Monte Carlo, signal IC, ES bootstrap, map payload | added | determinism tests |
  | Constant-price (cash fund) exclusion | added | `test_flat_price_asset_is_detected_before_the_math` |
  | Readable error for mistyped symbols (`NoDataError`) | added | `test_unknown_symbols_get_a_readable_error` |
  | Freshness tick 1s -> 60s | changed | `test_freshness_ticker_does_not_tick_every_second` |
  | Gungnir plaque labels circle -> block | changed | `test_gungnir_plaque_matches_its_sketch_and_its_math` |
  | Negative CVaR reads "still ends up +X%", not "loses -X%" | changed | `recruiter_walk.mjs` contradiction check |
  | Drawers no longer snap shut on the first change | fixed | `recruiter_walk.mjs` drawer check |
  | SEO head, OG image, favicon, robots, sitemap, 404, pre-boot paint | added | `test_public_files_...`, `site_check` (SEO=1) |
  | Live audit workflow, provenance, security probe, recruiter walk, `tools/check` | added | runs on every `claude/**` push |
  | Layering rule: `src/` never imports the UI | added | `test_src_is_pure_computation_and_never_imports_the_ui` |
- ✅ **Top bar removed + stress test (2026-10-15)** - Streamlit's fixed header
  (the beige bar over the crest) hidden. Stress test in four layers
  (`audit/stress_2026-10-15T020905Z.md`): planted-truth math 11/11; a
  constant-price asset (money-market fund) crashed risk parity with "nan%" -
  now excluded with a reason; three deterministic hot paths cached (warm page
  view 5.1s -> 1.8s CPU); the 1 Hz freshness tick held the process at 93% of a
  core with 15 idle tabs - now 60s, 4%. Synthetic-verified only: the live
  batch audit still needs a machine with Yahoo access.
- ✅ **Risk Topology map audit (2026-10-15)** - the batch audit's map section
  (MAP-00..12, control, source checks) run on a realistic synthetic market:
  732/732 before and after. The maths held; the defects were in what is drawn
  and said around it, each fixed test-first:
  - **Layout** (`tests/map_layout.mjs` renders the shipped page in Chromium at
    the iframe's 820px height, 390-1920px wide): the bottom chips covered the
    beta tick labels at EVERY width; at 1024px the HUD tiles slid under the
    inspector; at 1000px and below the panels covered each other (Run /
    Explain / Calibration painted but unclickable), the plot shrank to 180px at
    790px and 18px on a phone, and under 640px the canvas was stretched. Now
    the plot is measured from its own stage, the chips stay clear of the ticks,
    and below 1000px the panels stack (HUD, map, chips, controls, pairing).
    Wide screens draw as before apart from the tick margin.
  - **Short book:** the hover read risk backwards (moving toward the perimeter
    said "de-risking"); the anchor pool (TLT, GLD...) was drawn as if shorted,
    though the pair holds it long.
  - **Disclosure:** fit_ou clipped phi to 0.995 silently (it binds on live data:
    2 of 4 universes in the 2026-09-10 audit); the leverage clamp was reported
    as "rho"; k capped at 3 was called measured. Every clamp now names itself.
  - **Words:** "the flyer holds most of the risk" (two-asset risk parity = equal
    risk); "a crash in the flyer" on a short book (its loss is a squeeze); the
    perimeter hover now states the floor and caps.
  Open, by design: a universe name can be both a shorted holding and a pair's
  long anchor, so its dot and its link disagree on a short book. Synthetic-
  verified; `python -m tests.batch_audit` on live data is still owed.
- ✅ **Map audit follow-up (2026-10-15)** - an adversarial review AFTER the merge
  found what the first pass missed, each now a failing probe check first:
  HUD tiles/chips still spilled at 1181-1249 and 1451-1465px (the probe only
  tried round widths; it now covers the gaps); on the stacked layout the
  x-axis sat below the iframe's 820px fold (800-862px) and explain mode showed
  the x-axis tip with the pointer over the panels BELOW the map (the shell
  listener had no bounds check once the shell outgrew the stage); a book near
  beta 0 read a move away from zero as "exposure sheds" (now |beta| change);
  anchor-pool dots were explained as "one holding"; `.4g` printed a phi clamp
  as "1 -> 0.995"; the pair_weights docstring kept the false "flyer holds most
  of the risk". Swept every 10px, 360-1920: clean. Below 360px the plot is
  narrower than 260px - the screen, not the layout. Desktop dots unchanged.
- ✅ **The tail model fits 6x faster, same numbers (2026-10-20)** - John: "speed up
  the tail model fit next". The Tail Risk panel spent 3.8 s fitting: two GPD
  fits (residuals, then raw returns), each with a 200-refit bootstrap of xi.
  (1) The raw fit's bootstrap fed only `unconditional_xi_ci`, which nothing
  read: it no longer runs (`n_boot=0`), and the key is gone. (2) scipy's
  generic fit objective is mostly dispatch overhead; `src/risk._lean_nnlf` is
  the same numpy calls for genpareto with loc 0, passed as the optimizer's
  objective, equal bit for bit (3,000+ parameter points incl. every edge the fit can reach) on
  the pinned scipy 1.18.1. Measured on four 10-year series: 3.82 s -> 0.63 s
  per call, all 22 published numbers identical. Council caught one real bug in
  the prototype: `math.log` instead of `np.log` differs in the last bit on
  some scales (AVX512), which would have moved fits on some machines.
- ✅ **Typed-in baskets download only their new names (2026-10-20)** - John:
  "make typed-in baskets load faster too". Two investigators: offline, the
  default basket plus three typed names took 29.3s to settle (34.7s for six
  new names). The cache kept whole baskets, so the three new names
  re-downloaded all fifteen under each of the page's five keys (67
  name-downloads where 9 would do), and a cold download is almost all
  yfinance's repair pass - CPU, ~2.75s per name for full history, which
  threads cannot overlap. A council (data honesty + minimalist) chose a
  per-name cache inside `_download_close_volume`: each name's own download is
  kept per period and only missing names are downloaded, in one call.
  yf.download is itself per name, so a stitched basket equals one download of
  it exactly (`test_a_basket_stitched_from_single_names_matches_one_download`,
  mixed calendars, failed and empty names). No period is cut out of another.
  Names in a basket may now be pulled at different moments inside the 6h
  window: provenance keeps the oldest pull as the basket's age and the Lineage
  tab shows the span. Failed names are never remembered; `use_cache=False`
  never reads or writes a name file; an ISIN basket takes one download.
  yfinance pinned at 1.7.0 (the stitch mirrors its assembly). The cache cap
  went 240 -> 1500: a warm pass already evicted 37 of its own files.
  The deployed walk now times a typed basket (three names in no preset).
- ✅ **The crisis pair, warm too (2026-10-20)** - John: "fix the crisis pair
  slow download next". After the warm pass the one cold pull left on a first
  switch was the Balance tab's crisis replay: the default flyer and anchor's
  full history (`replay_returns`). The warm pass now picks that pair by the
  page's own rule - the most volatile name over 2y, the top of `anchor_rank`
  over the universe plus the defensive pool, scored with grit - from pulls it
  has already warmed (CPU, not Yahoo), and warms the pair's history; ~20 more
  requests a pass. `test_the_warm_pass_covers_every_fetch_a_preset_page_makes`
  now has no exemption: red until the pair was warmed.
- ✅ **The first basket switch, warm (2026-10-20)** - John: "fix the slow first
  basket switch". After a deploy the deployed walk took 93.1s to switch to
  Semiconductors (101.6s in run 24). Two investigators: the headline came in
  seconds, then the page paid five cold Yahoo pulls (the cache is keyed per
  basket, and the warmer warmed only 2y) plus 20 serial security-master calls
  with no disk cache. A council rejected a per-ticker store and shared keys
  (both can move numbers: mixed snapshots, slices vs direct pulls, mixed
  calendars). Instead the warm pass now makes every pull the page makes, by the
  same calls (cache keys match, no number changes), and the security master
  keeps a day on disk (an hour for a failed row). Offline, 3s per cold pull:
  switch 40.9s -> 12.8s, headline 4.1s -> 1.0s; left cold: the 2-name crisis
  pair. Tests: `test_the_warm_pass_covers_every_fetch_a_preset_page_makes`,
  `test_security_master_reads_its_disk_cache_and_never_keeps_a_failure`.
- ✅ **Every word on the map clears AA (2026-10-20)** - the 2026-10-19 council
  review measured the Risk Topology map's small labels below 4.5:1 on its dark
  glass (`--dim` 3.86:1, `--bronze` 4.38:1, `--emberhi` 3.99:1).
  `test_text_colours_meet_wcag_aa` read only `static/app.css` and `main.py`, so
  the map was never checked. It now also scans `prototypes/war_room.html`: CSS
  colours (vars resolved), the pairing diagram's SVG text, and every canvas
  `fillStyle` a `fillText`/`spacedText` draws with (rgba blended onto the
  ground). Grounds: the glass `#1F1A13`, `#282117` over the bright map, and
  `#362C1E`, the selected pairing row over the bright map, the lightest ground
  any word gets. Red first (`--bronze` 3.46:1 on that row). Words got their own
  tones; rules, bars, dots and focus rings kept theirs: `--dimtext` `#A09788`
  (3.05 -> 4.74), `--bronzetext` `#B49563` (3.46 -> 4.83), canvas ticks and
  ring captions `#A09788`, axis titles and sector names `#B49563` (2.67 ->
  4.83), VaR labels gold 55% -> 75% (3.95 -> 5.05), breach perimeter label
  `#D98463` (2.16 -> 4.84), Run button hover fill `#6A5030` (beige label 3.26
  -> 6.20). The suggested `#948B7D` cleared the plain panel (5.14:1) but not
  the selected row (4.07:1), so it was not used. Layout probe green at every
  width; ACCESSIBILITY.md carries the table.
- ✅ **Round 2 of John's laptop test (2026-10-19)** - six complaints, four
  investigators, a four-member council, each fix test-first:
  (1) the "Explore what we do" glide aimed once at a showcase row still rising
  24px on its reveal - overshoot 8-11px, then a crawl back - fought the wheel,
  flew two flights on a double press, and re-sent the off-screen map's five
  canvases every frame. Now it follows the target (2px/frame cap), flies one
  flight, yields to wheel/touch/keys and skips the map's rendering only while
  flying: 1536x864@1.25, 4x CPU, frame median 50 -> 16.7 ms, overshoot 11 -> 0;
  wheel reversals 8 -> 0-1. (2) the grit spinner says "Reading full price
  histories for the grit scores…" (no "Yahoo is the slow part"). (3) "Explain
  the map" opens a glossary under itself at every width. (4) "pp" -> "16.8
  points shallower"; the Balance sentence and crisis table say percentage
  points, not a misleading %. (5) the HUD chips: the app gives the map
  1471-1488px, just past the old 1470px breakpoint, so they ran out of the
  band - now a wrapping row at every width. (6) no "OU" on the tiles; chips in
  words (beta, vol, Baseline), every short form in the glossary. Rule tests:
  `test_the_glide_follows_a_moving_target_and_yields_to_the_reader`,
  `test_every_short_form_on_the_map_is_in_its_glossary`,
  `test_the_pairing_never_shows_pp_or_percent_for_a_points_gap`,
  `test_every_spinner_is_a_few_calm_words`; map_layout.mjs runs the app's real
  desktop widths with a Windows-width font.
- ✅ **The buttons respect reduced motion (2026-10-19)** - John's call on the
  one gap the council flagged: ACCESSIBILITY.md promised every animation honours
  `prefers-reduced-motion`, but the 1.5 s glide is JavaScript, which no media
  query can stop, and the CSS smooth-scroll fallback sat outside every
  reduced-motion block. Now the glide asks on every click and lands at once
  (same place, no hash jump), and the fallback stands down. Offline probe, hero
  CTA + footer link: reduced motion 87/82 frames moving -> 1/1; normal still
  92/76 frames (glides), both land 26 px from the top. Rule test:
  `test_the_glide_respects_reduced_motion`.
- ✅ **Loading you can see, and one map (2026-10-19)** - John, on his laptop:
  no skeleton on opening, the Risk Topology map shown twice and slow after
  switching to MAG-7, the Grit Zone "stalling as if crashed", a loading wheel
  that did not move, buttons that sometimes jumped. Five investigators proved
  one cause each (offline harness = the real `main.py` on the deterministic
  test market + the Live audit log), a council reviewed the fix plan:
  (1) The Book wrote one top-level card per name, so the tab strip moved with
  the basket (slot 44 -> 40) and the old map stayed on screen, faded, for up to
  ~35 s - The Book and the come-and-go notices now sit in one container each.
  (2) The skeleton came after ~5 MB of base64 photos on the websocket and
  lifted on a 2.1 s timer; nothing Python sends can arrive before Streamlit's
  JavaScript starts - so `tools/public_files.py` paints the hero's own skeleton
  (`hero.py`) from index.html with app.css linked in the head, the hero lifts
  it, the photos are files (`enableStaticServing`). Phone at 9 Mbps: skeleton
  from 0.6 s, hero 5.6 s (was blank 5.6-12.9 s); skeleton vs hero 0 px at six
  widths. Shimmers now run until the lift stops them (the 10-15 "finite
  iterations" rule is replaced by that). (3) Streamlit's running sign lives in
  the hidden header - app.css now draws a bronze rail and an "UPDATING…" tag
  off it (still under reduced motion). (4) Every cached Yahoo pull says what
  it is doing (the grit spinner blamed a 0.2 s step; five pulls were silent).
  (5) The glide frame sat after every `st.stop()` with a bind-once guard: a
  stopped run killed it until reload (15/15 jumps), and buttons were unwired
  for 2.4-4.6 s of each load - it now loads right after app.css, rebinds, and
  caps each frame's progress (400 ms stall: 5% step, was 40%). deep_scan's
  glide check now catches the browser's own jump (hash, flight time). Five
  rule tests, each red on 5b7d575.
- ✅ **Council review + every device (2026-10-17)** - four independent
  reviewers (map maths vs its papers, the data chain, every chart, every
  control), each claim then reproduced or refuted by a skeptic. The map's maths
  came out clean (57 checks, incl. OU moments, ring masses, breach semantics,
  every linkage, long and short). 14 defects elsewhere were confirmed and fixed,
  each with a test that failed first: a crisis replay over WTI's -37.63 settle
  turned a +$47.64 rebound into a -126.6% "loss" (non-positive prices now
  excluded and disclosed); a late listing silently cut every name's replay
  window (now excluded, caption prints the replayed dates); the Book's grit came
  from 2y (no crisis inside) and contradicted the Grit Zone tab (now full
  history); the tail fit claimed "~10 years" on 5.1 (unaligned pull, young names
  named, real span printed); Sharpe moved with the vol-target slider (now on the
  unlevered book); futures contracts x price was shown as dollars (now the FX
  no-volume path); a negative CVaR "widened" into a bigger gain; the -50% shock
  compounded to -39% (now multiplicative); crypto weekend rows understated stock
  vol 16% and made "1 year" 0.69 years (weekends dropped); plus the cone,
  pair-money, noise-line, NaN-correlation and sticky-anchor captions. Then
  `tools/device_check.mjs` (Live audit 8b): the whole page, every drawer and
  tab, on 13 phones/tablets/desktops. Found one defect class - Streamlit cut
  metric labels and values to "..." on every phone and portrait iPad - fixed by
  letting them wrap (`static/app.css`, rule test).
- ✅ **Every ticker, every pair (2026-10-17)** - `tests/pair_audit.py` + the
  Pair audit workflow: each of the 322 suggested symbols on its own, and all
  51,681 pairs long and short through the real fetch -> estimators/weights ->
  map payload path on live data, every number re-derived with independent
  numpy (two-asset ERC = inverse vol, MRT 2010). Yahoo asked once, 16 shards.
  Mutation-tested: a wrong beta window, a wrong perimeter cap and the bug below
  each turn it red. Its first run found that bug: the anchor pool was joined
  with DataFrame.join, which raises on a shared column, and the fallback
  dropped ALL anchors - so 4 of 20 presets (sector ETFs, rates & credit,
  commodities, global macro) and any pair holding TLT/GLD/XLP/XLU/USMV drew no
  anchor pool, and the Balance tab offered no defensive candidates. The batch
  audit used the same join and agreed with it. Fixed once in
  `topology.anchor_frame` (map and Balance both read it); batch audit MAP-02
  now checks the drawn pool equals the unheld anchors.
- ✅ **The map on an iPhone (2026-10-17)** - John saw the map cut off on his
  iPhone. The 10-15 note above ("below 360px ... the screen, not the layout")
  was WRONG: the probe sized the map's page, never the page AROUND it. The
  page's 56px gutters gave the map a 263-318px frame on every iPhone, a fixed
  300px pairing SVG pushed it 9-64px sideways, and the numbers panel above the
  map put the x-axis at 818-868px of the 820px frame. Fixed: the map's iframe
  spans a phone's screen 8px from each edge (`static/app.css`, phones only;
  frames 359/374/414px), the SVG scales (`.linkdiag max-width:100%`), and on
  frames <=520px the map comes first with its numbers under it (x-axis at
  ~404px). Hover/explain now pick the NEAREST dot - on a phone AAPL and MSFT
  sit 8px apart and a finger on MSFT read "AAPL". Probe: iPhone frames
  (m414/m374/m359/m320, touch, 3x) failing on main first; `site_check` phone
  pass now measures the map inside the real page. Desktop: 0px dot shift.
- ✅ **Deep scan, every inch (2026-10-15)** - `tools/deep_scan.mjs` (Live audit
  steps 10-11) walks the page as a visitor meets it, then drives every control,
  and measures each button's flight frame by frame. What it found on the live
  site, each fixed and green after:

  | change | kind | guarded by |
  |---|---|---|
  | Footer links dead (scroller probe nudged +1px only; at the page bottom that cannot move) | fixed | `deep_scan` glide: lands on its section |
  | Button arrivals faded the destination to 25% + slid it 26px (+ band "launch", Conviction frame) | removed | `deep_scan` glide: destination never fades |
  | Glide easing quart (peak 4x average, ~270px/frame on footer flights) -> sine (1.6x) | changed | `deep_scan` glide: even pace |
  | Loading-veil shimmers ran forever on the hidden veil (12 animations) | fixed | finite iterations in `app.css` |
  | The Book / Risk Topology printed raw exception text | fixed | `test_no_raw_exception_text_reaches_the_page` |
  | Balance: "the flyer holds most of the RISK" after an equal-risk split | fixed | wording |
  | Conviction: hand-typed "the one honest exception is the dot-com peak" | changed | read from the live table |
  | Hero/footer crisis count, path count typed in | changed | read from `HISTORICAL_REGIMES`, `N_PATHS` |

- ✅ **New crest, and the name spelled Meliona (2026-10-28)** - John's logo
  mandate, from his two sketches with the red-X'd outer triangles removed: a
  charcoal disc, three rings fading outward, and a regular hexagon cut by an X
  into two inward-pointing equilateral triangles and top/bottom rhombuses, plus
  a keystone rhombus where the X crosses. Nothing touches (checked in code).
  `tools/brand_icons.mjs` renders the favicon and touch icon from
  `assets/logo.svg`; the share image got the crest and the new spelling in
  place. Then every "Meleona" became "Meliona" (words, CSS classes, glide
  globals, log file, build tag, setting names). Two deliberate leftovers: the
  mailbox `meleona.support@gmail.com` (a Gmail address cannot be renamed -
  RUNBOOK §6b says how to move), and `observability.host_setting()` still
  reads `MELEONA_*` so a variable already set on Railway keeps working.
- ✅ **The crest, beige and unframed, ripples on hover (2026-10-29)** - John
  chose the beige pieces, asked for the glossy square plate to go, for the
  rings to ripple inside to outside while hovered and stop when the pointer
  leaves, and for the mark to be perfectly symmetric. The plate (border, tint,
  shadow, padding) is gone and the crest fills the same box, so nothing below
  moves (62 -> 104px on a 1440x900 laptop). The SVG is drawn about (0,0): every
  mirror pair is the same number with the sign flipped. On hover each ring
  swells 3.5% and fades, 0.3s behind the one inside it; on leave the glide
  script lets each finish its pass and stop at rest (re-entering mid-settle
  wakes all three), off under reduced motion. Measured: rings end at exactly
  scale 1 within ~1s of leaving; rendered crest centred to 0.015px alone, and
  on a 1x laptop it matches its mirror image but for 12 edge pixels.
  `test_the_crest_is_symmetric_and_its_pieces_never_touch` (mutation-tested:
  a 0.001 shift, touching rings, touching pieces, the plate returning).
- ✅ **No lion left, and a new link preview (2026-10-29)** - the last two pieces
  of the old crest were faint engravings: the balance scale behind the Grit Zone
  and the lion's tail by the fan chart. They are now the new crest's hexagon in
  outline and a quarter of its rings. The browser tab gets the new crest both
  ways (the page's icon link and the icon Streamlit sets at run time). The link
  preview (`public/og-image.png`) is rebuilt from `tools/og_image.html`, rendered
  by `tools/brand_icons.mjs`: the crest, the name, one sentence, and a ruled
  receipt of the 15 Oct 2026 live audit (727 of 728 checks; prices identical to
  Yahoo's raw feed; T-bill 4.085% vs Treasury 4.100%; SPY/IVV 0.9995), every
  figure copied from `audit/live_audit_2026-10-15T025826Z.md` and dated on the
  image. No fan chart, no stat tiles, no eyebrow caps, no italic subtitle.

- ✅ **showcase/: the engine in ~330 lines (2026-10-29)** - John wanted a short,
  blunt copy of the Python for hiring managers, with plain Gen Z comments, kept
  apart from the app (his call: the live site and the interview-depth notes in
  `src/` stay as they are). `data, risk, portfolio, stress, run` cover prices,
  VaR/CVaR, the walk-forward Kupiec backtest, both Monte Carlo engines, risk
  parity, vol target, days to exit, the 10 crash replays and the Grit score.
  `showcase/test_showcase.py` proves every function equals `src/` (to 1e-12,
  same seeds) and runs in `tools.check`, so the copy cannot drift.
- ✅ **Risk Engine purge - the overwhelm audit** (2026-10-29). Snapshot commit
  first ("Auto-Snapshot before Risk Engine Purge"). Non-UI: the glide script
  left the deprecated `components.html` (a warning on every run) for a folded
  `st.iframe`; a cold start with the feed down is tested to show guidance and
  no number; the map is tested to draw no node or link the data did not
  produce. UI, on John's rule - cut what the risk engine does not need, keep
  what carries its identity (grit, the linkages between names): the research
  workshop (Signal Lab, Regime Atlas, Security Master - and
  `src/security_master.py` with its Yahoo calls) is gone; Liquidity joined the
  main strip; Data Quality and Lineage fold into one "Where the numbers come
  from" drawer; the Slice keeps its pairing sketch and equation and loses the
  dense paragraph under them; The Book keeps its paragraph and its LEADS and
  ANCHOR cards on the page and folds the no-role cards and two finer settings
  into a drawer; the hero and headline speak plainly ("Bad-year loss",
  "Simulated years"). The Risk Topology map is untouched.
- ✅ **Options audit - every setting, mixed and matched, on live data**
  (2026-10-30). John: every asset with every allocation, target vol 5-30%,
  every estimator, both return models, every crisis, book size and
  participation, "until no pairs are left untested", nothing hallucinated.
  `tests/options_audit.py` + the **Options audit** workflow. ENGINE: all 20
  presets and every suggested symbol paired with SPY, through the app's own
  functions, each answer re-derived by the file's own numpy (the three
  covariance estimators from their papers, risk parity's equal risk shares,
  leverage = target / book vol at every target, both Monte Carlo engines' tail
  and cone from their own paths plus the history's mean year, every crisis
  replay from raw closes, days to exit for 6 book sizes x 10 participations,
  and the map - every dot, the hazard line, every defensive link, and its web
  of simulated futures held to closed-form OU theory under node). PAGE: the
  real `main.py` under AppTest on 1,906 runs where every pair of widget values
  meets (an all-pairs covering array over 20 widgets; 20 presets and 12
  typed-in baskets); each run's headline CVaR, cone, Liquidity bars, Book
  sentence, Correlation Watch line and map must equal the engine for exactly
  those settings. Planted wiring bugs (a VT slider ignored, a wrong rolling
  window, a book size ignored) fail it. The four crisis-record boxes are now
  worked out a second time from Yahoo's raw chart API in the provenance step
  (`crisis_boxes`, checked offline against `src.conviction`), and every in-page
  button must sit on both glide scans' lists. The Workshop stays removed (John,
  2026-10-30): Yahoo carries no ISIN (yfinance scrapes Business Insider,
  "experimental"; a real one needs a CUSIP licence), so that column could
  never be filled honestly.

## Non-negotiable constraints (the "why")

1. **No LLM data, ever.** Every market number comes from Yahoo Finance via
   yfinance at runtime, computed by the engine's own numpy/scipy. The
   provenance panel states this explicitly. Never hardcode/estimate a market figure.
2. **Honest labeling - no overclaiming.** It's "live end-of-day data," NOT
   "real-time." The hedge-fund basket is "13F-popular," NOT "Citadel's picks."
   Scenarios "replay actual returns," exclusions are disclosed. Overclaiming is
   the #1 thing that fails a quant interview.
3. **Lead with one number.** Design philosophy: one headline CVaR verdict +
   one sentence; all depth collapsed a click away (tabs, as of the redesign
   below). Simplicity is a feature.
4. **Defensible in an interview.** Every feature needs a "Quant Deep Dive"
   explanation. Methodology depth > visual complexity > latency.
5. **Aesthetic:** Citadel-style - beige `#EDE9E3`/`#D4CDBF`, bronze `#9A7B4F`/
   `#8A6A3C`, charcoal `#3F3B35`, serif (Georgia). Calm, neutral, not flashy.

## Workflow conventions

- Verify every change by running it (smoke-test modules with `python -m src.X`,
  check the app returns HTTP 200, independently recompute key numbers).
- Commit after each working feature with a descriptive message; push to GitHub.
- Cache files (`data/*.parquet`, `*.meta.json`) are gitignored.
- The user is new to Git/GitHub - explain steps plainly, do the git work for them.

## Data-engineering domain checklist - free-tier status

A "resume checklist" of hedge-fund-grade data-engineering capabilities was
requested. Everything with a genuinely free alternative is now built; the
two paid-vendor-dependent pieces remain explicit backlog items:

1. **Risk fundamentals (VaR/ES/vol/stress testing)** - ✅ already built:
   `src/risk.py` (historical + parametric VaR, CVaR, Kupiec backtest, bootstrap
   + Merton jump-diffusion Monte Carlo), `src/scenarios.py` (historical-regime
   stress replay), `src/grit.py` (drawdown/resilience scoring).
2. **Corporate actions & security master** - ✅ built free-tier
   (`src/security_master.py`): ISIN via `yfinance`, real dividend/split event
   history. ⬜ SEDOL/CUSIP and full merger/ticker-change history still need a
   paid reference-data vendor (Bloomberg, Refinitiv) - explicitly flagged as
   unavailable in the UI, not fabricated.
3. **Regulatory awareness (data lineage, audit trails)** - ✅ built: the
   Lineage & Audit tab combines the existing provenance record with a
   session-scoped audit trail of what this run actually did. Session-scoped
   is a deliberate honesty choice, not a durable compliance log - see Status
   above.
4. **Real-time / low-latency streaming** - ✅ built free-tier (fast session
   polling + live "Xs ago" ticker, disk-cache freshness window unchanged so
   Yahoo isn't hit harder). ⬜ True tick-level streaming still needs a paid
   vendor (Polygon.io, Alpaca, IEX Cloud) and an API key from the user.
5. **Automated data-quality validation framework** - ✅ built
   (`src/data_quality.py`): schema, positivity, coverage, staleness,
   calendar-gap, and extreme-move checks, run on every price pull and
   surfaced in its own Data Quality tab.

## Good next steps to offer

1. **Phase VI deployment** to a live URL (the resume link - the whole point).
   Streamlit Community Cloud is the fastest free path; Procfile + requirements
   are already set for Railway/Render too.
2. If a paid data vendor becomes available: true tick-level streaming
   (Polygon.io/Alpaca/IEX) is scoped and ready to wire in once an API key exists.
3. Extend liquidity: per-asset liquidity-adjusted VaR, or a book-size slider
   preset that showcases a small/mid-cap basket where days-to-liquidate bites.
