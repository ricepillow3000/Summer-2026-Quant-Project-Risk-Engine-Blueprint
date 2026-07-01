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
│   └── grit.py          # Grit Zone: drawdown recovery, rolling consistency, regime resilience
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
- ⬜ **"Living" 3D particle effect** (requested, not yet built) — make the 3D
  outcome-distribution surface feel animated/alive (drifting particles, not
  just rotatable). Needs a custom `st.components.v1.html` component (raw
  plotly.js + a small JS animation loop restyling a particle overlay trace),
  since `st.plotly_chart`'s embedding doesn't expose that. Scoped out of this
  pass — check in on approach before building (novel, harder to iterate on
  than CSS).
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

## Data-engineering domain roadmap (requested, scoped as backlog)

A broader "resume checklist" of hedge-fund-grade data-engineering capabilities
was requested alongside the redesign above. Deliberately scoped OUT of the
redesign pass — each item below is substantial on its own, and a couple
directly interact with this project's own honesty principles or need a paid
vendor decision from the user before implementation can start:

1. **Risk fundamentals (VaR/ES/vol/stress testing)** — largely already built:
   `src/risk.py` (historical + parametric VaR, CVaR, Kupiec backtest, bootstrap
   + Merton jump-diffusion Monte Carlo), `src/scenarios.py` (historical-regime
   stress replay), `src/grit.py` (drawdown/resilience scoring). If "understand
   it deeply" means more explanatory depth (more Quant Deep Dive copy, worked
   examples), that's a smaller, low-risk follow-up.
2. **Corporate actions & security master** (ISIN/SEDOL cross-referencing,
   dividend/split/merger handling) — NOT built. `yfinance` with
   `auto_adjust=True` (already used) silently folds splits/dividends into
   adjusted close, but there's no explicit corporate-actions ledger, no
   merger/ticker-change handling, and no identifier cross-reference layer.
   OpenFIGI (free) can map identifiers; structured corporate-actions calendars
   are usually a paid vendor. Needs a data-source decision before scoping.
3. **Regulatory awareness (data lineage, audit trails)** — partially built:
   `ingestion.py`'s `.meta.json` provenance record covers source/timestamp/
   coverage, but there's no formal lineage graph or audit log of what was
   queried/computed/shown when.
4. **Real-time / low-latency streaming** — NOT built, and in tension with
   constraint #2 above (honest "live end-of-day," not "real-time"). `yfinance`
   is a polling scraper, not a streaming API. Fast honest polling (e.g. every
   15-60s during market hours, clearly labeled "refreshed Xs ago") is
   achievable for free; true tick-level streaming needs a paid vendor
   (Polygon.io, Alpaca, IEX Cloud) and an API key.
5. **Automated data-quality validation framework** — partially built
   (`data_health()` staleness/gap/row-count checks, various "excluded, not
   estimated" honesty patterns throughout); no formal schema-validation or
   anomaly-detection gate on ingest/egress yet.

## Good next steps to offer

1. **Phase VI deployment** to a live URL (the resume link — the whole point).
   Streamlit Community Cloud is the fastest free path; Procfile + requirements
   are already set for Railway/Render too.
2. The "living" 3D particle effect (see Status above) — highest-novelty piece
   of the redesign ask, worth a design check-in before building.
3. Pick one item from the data-engineering roadmap above to scope properly
   (start with #2 or #5 — they don't need a paid vendor decision first).
4. Extend liquidity: per-asset liquidity-adjusted VaR, or a book-size slider
   preset that showcases a small/mid-cap basket where days-to-liquidate bites.
