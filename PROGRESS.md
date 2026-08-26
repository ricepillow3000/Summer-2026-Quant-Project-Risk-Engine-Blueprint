# Meleona - Session Handoff

Paste this into a new chat session to bring it up to speed instantly.

---

## What this project is

**Meleona** is an institutional-grade **Portfolio Optimization & Risk Engine**
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
├── assets/logo.svg      # lion + scale + triangle crest (bronze, "Pride · Integrity")
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
- ✅ **Rebrand to Meleona** - page title, header wordmark, logo alt text, and
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

- ⬜ **Phase VI** - deploy to Railway/Render for the live recruiter link
  (Procfile + requirements.txt already set up)

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
2. If a paid data vendor becomes available: SEDOL/CUSIP/merger history
   (security master) or true tick-level streaming (Polygon.io/Alpaca/IEX) -
   both are scoped and ready to wire in once an API key exists.
3. Extend liquidity: per-asset liquidity-adjusted VaR, or a book-size slider
   preset that showcases a small/mid-cap basket where days-to-liquidate bites.
