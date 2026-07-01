# Meleona — Session Handoff

Paste this into a new chat session to bring it up to speed instantly.

---

## What this project is

**Meleona** is an institutional-grade **Portfolio Optimization & Risk Engine**
with a live Streamlit dashboard, built by a 2nd-year Data Analytics student to
be recruiter-facing (the goal is a public live link on a resume). The name
merges "Durand" (Old French, "enduring/built to withstand hard times") with
"Mereoleona" (mother lioness) — it ties into the existing lion-crest logo and
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
**Only run ONE Streamlit server** — stale servers caused an ImportError once.
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

- ✅ **Phase I** — DataEngine with caching
- ✅ **Phase II** — covariance + eigen-decomposition
- ✅ **Phase III** — CVaR + Monte Carlo
- ✅ **Phase IV** — Streamlit dashboard, logo, institutional theme
- ✅ **Phase V** — configurable universe, data integrity, VaR backtest, named
  factors, provenance, risk parity, vol targeting, risk contribution,
  **true historical regime replication**, **liquidity modeling
  (days-to-liquidate via participation-rate model)**, **Merton jump-diffusion
  Monte Carlo engine (fat-tailed alternative to the bootstrap)**
- ✅ **Regression test suite** (`tests/test_engine.py`) — math-invariant tests +
  closed-form validation (Gaussian CVaR, Sharpe) + full-app boot test; run
  `python -m tests.test_engine` or `pytest`
- ✅ **Data engine speed** — one Yahoo download now fills both price + dollar-volume
  caches (cold load = 1 round-trip, not 2); Sharpe ratio vs live ^IRX risk-free rate
- ✅ **Visual upgrade** — themed Plotly charts (beige/bronze); hero Monte Carlo
  fan chart (outcome cone) under the verdict; themed histogram, risk-contribution,
  factor, and liquidity charts
- ✅ **3D outcome distribution** — rotatable Plotly surface (day x return-bin x
  density) under the fan chart, collapsed by default
- ✅ **Grit Zone** (`src/grit.py`) — a "grit score" concept as a counterpart to
  Fear & Greed: ranks each asset's OWN price history on drawdown-recovery
  speed/completeness, rolling 1-year consistency, and drawdown/recovery
  behavior across the real historical crisis windows in `scenarios.py`.
  Percentile-ranked RELATIVE to the chosen universe (no absolute "grit" scale
  claimed). 7 deterministic unit tests; verified against live data (history
  length and regimes-survived per ticker line up with real IPO dates).
- ✅ **Rebrand to Meleona** — page title, header wordmark, logo alt text, and
  docs updated from "Portfolio Risk Engine"
- ✅ **Presentation-style redesign** — the app used to open as one long stack
  of controls plus five nested expanders (a wall of information before you'd
  touched a slider). Restructured into a scroll-driven flow: hero pitch →
  dedicated "Grit Zone" showcase section (with anchor-scroll CTA buttons) →
  "the engine" (the existing interactive dashboard). The five supporting
  expanders (3D distribution, risk breakdown, Grit Zone detail, liquidity,
  provenance) are now one `st.tabs()` strip instead of stacked accordions.
  Added CSS scroll-reveal animation, smooth-scroll CTAs, hover-lift cards —
  pure CSS, no new toolchain, `streamlit run main.py` deploy story unchanged.
- ✅ **"Living" 3D particle effect** (`main.py`'s `living_surface_html()`) —
  the 3D Distribution tab is now a self-contained `st.iframe` component: raw
  plotly.js loaded from CDN, a density-weighted particle overlay
  (`scatter3d`) whose points jitter every 150ms, and a slow continuous
  camera auto-rotate that pauses on manual drag/touch and resumes after 4s
  idle. `st.components.v1.html` was already past its removal window
  (deprecated in favor of `st.iframe` in Streamlit 1.58) — built directly on
  `st.iframe` instead of the soon-to-be-removed API.
- ✅ **Security Master & Corporate Actions** (`src/security_master.py`) —
  free-tier reference data: ISIN via `yfinance`'s `Ticker.isin` (available
  for some tickers, honestly flagged `"unavailable"` for others — e.g. many
  US large caps don't expose one on the free feed), plus real dividend/split
  event history via `Ticker.dividends` / `Ticker.splits`. Verified against
  live data: AVGO's actual 2024-07-15 10:1 split, GOOGL's 2022-07-18 20:1
  split, NVDA's 2024-06-10 10:1 split, AMZN's 2022-06-06 20:1 split all came
  back correctly. SEDOL/CUSIP/merger history explicitly flagged as needing a
  paid vendor, not fabricated.
- ✅ **Data Quality validation gate** (`src/data_quality.py`) — hand-rolled
  (no new dependency) automated schema + sanity checks run on every price
  pull: DatetimeIndex/dtype/sort/duplicate checks, positivity, minimum row
  coverage, staleness, calendar-gap %, and an extreme-single-day-move flag
  (WARN, not FAIL — a real crash day should surface, not silently block).
  5 deterministic unit tests, each engineered to trip exactly one check.
- ✅ **Lineage & Audit tab** — extends the existing provenance panel with a
  session-scoped audit trail (`audit_log` built as the script runs: data
  fetch → allocation → stress scenario → Monte Carlo engine, each with real
  parameters). Explicitly labeled as session-scoped, not durable storage —
  same concept a full compliance system uses, at the scale this engine
  actually operates at.
- ✅ **Fast polling ("as live as honestly possible")** — `load_universe`'s
  Streamlit-session cache TTL dropped from 1h to 60s (re-checks the local
  disk cache far more often) while the underlying disk-cache freshness
  window stays at 6h (`ingestion.CACHE_MAX_AGE_HOURS`) so Yahoo itself isn't
  hit any harder — this is what actually protects against rate-limiting, not
  the session TTL. A `st.fragment(run_every="1s")` ticker shows "data pulled
  Xs ago," reruns only itself, not the whole page/Monte Carlo computation.
- ⬜ **Phase V polish remaining** — optional auto "executive summary";
  further UI refinement
- ⬜ **Phase VI** — deploy to Railway/Render for the live recruiter link
  (Procfile + requirements.txt already set up)

## Non-negotiable constraints (the "why")

1. **No LLM data, ever.** Every market number comes from Yahoo Finance via
   yfinance at runtime, computed by the engine's own numpy/scipy. The
   provenance panel states this explicitly. Never hardcode/estimate a market figure.
2. **Honest labeling — no overclaiming.** It's "live end-of-day data," NOT
   "real-time." The hedge-fund basket is "13F-popular," NOT "Citadel's picks."
   Scenarios "replay actual returns," exclusions are disclosed. Overclaiming is
   the #1 thing that fails a quant interview.
3. **Lead with one number.** Design philosophy: one headline CVaR verdict +
   one sentence; all depth collapsed a click away (tabs, as of the redesign
   below). Simplicity is a feature.
4. **Defensible in an interview.** Every feature needs a "Quant Deep Dive"
   explanation. Methodology depth > visual complexity > latency.
5. **Aesthetic:** Citadel-style — beige `#EDE9E3`/`#D4CDBF`, bronze `#9A7B4F`/
   `#8A6A3C`, charcoal `#3F3B35`, serif (Georgia). Calm, neutral, not flashy.

## Workflow conventions

- Verify every change by running it (smoke-test modules with `python -m src.X`,
  check the app returns HTTP 200, independently recompute key numbers).
- Commit after each working feature with a descriptive message; push to GitHub.
- Cache files (`data/*.parquet`, `*.meta.json`) are gitignored.
- The user is new to Git/GitHub — explain steps plainly, do the git work for them.

## Data-engineering domain checklist — free-tier status

A "resume checklist" of hedge-fund-grade data-engineering capabilities was
requested. Everything with a genuinely free alternative is now built; the
two paid-vendor-dependent pieces remain explicit backlog items:

1. **Risk fundamentals (VaR/ES/vol/stress testing)** — ✅ already built:
   `src/risk.py` (historical + parametric VaR, CVaR, Kupiec backtest, bootstrap
   + Merton jump-diffusion Monte Carlo), `src/scenarios.py` (historical-regime
   stress replay), `src/grit.py` (drawdown/resilience scoring).
2. **Corporate actions & security master** — ✅ built free-tier
   (`src/security_master.py`): ISIN via `yfinance`, real dividend/split event
   history. ⬜ SEDOL/CUSIP and full merger/ticker-change history still need a
   paid reference-data vendor (Bloomberg, Refinitiv) — explicitly flagged as
   unavailable in the UI, not fabricated.
3. **Regulatory awareness (data lineage, audit trails)** — ✅ built: the
   Lineage & Audit tab combines the existing provenance record with a
   session-scoped audit trail of what this run actually did. Session-scoped
   is a deliberate honesty choice, not a durable compliance log — see Status
   above.
4. **Real-time / low-latency streaming** — ✅ built free-tier (fast session
   polling + live "Xs ago" ticker, disk-cache freshness window unchanged so
   Yahoo isn't hit harder). ⬜ True tick-level streaming still needs a paid
   vendor (Polygon.io, Alpaca, IEX Cloud) and an API key from the user.
5. **Automated data-quality validation framework** — ✅ built
   (`src/data_quality.py`): schema, positivity, coverage, staleness,
   calendar-gap, and extreme-move checks, run on every price pull and
   surfaced in its own Data Quality tab.

## Good next steps to offer

1. **Phase VI deployment** to a live URL (the resume link — the whole point).
   Streamlit Community Cloud is the fastest free path; Procfile + requirements
   are already set for Railway/Render too.
2. If a paid data vendor becomes available: SEDOL/CUSIP/merger history
   (security master) or true tick-level streaming (Polygon.io/Alpaca/IEX) —
   both are scoped and ready to wire in once an API key exists.
3. Extend liquidity: per-asset liquidity-adjusted VaR, or a book-size slider
   preset that showcases a small/mid-cap basket where days-to-liquidate bites.
