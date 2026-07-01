# Portfolio Risk Engine — Session Handoff

Paste this into a new chat session to bring it up to speed instantly.

---

## What this project is

An institutional-grade **Portfolio Optimization & Risk Engine** with a live
Streamlit dashboard, built by a 2nd-year Data Analytics student to be
recruiter-facing (the goal is a public live link on a resume).

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
│   └── scenarios.py     # historical regime replication (real crisis-window replay)
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
   one sentence; all depth collapsed in expanders. Simplicity is a feature.
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

## Good next steps to offer

1. **Phase VI deployment** to a live URL (the resume link — the whole point).
   Streamlit Community Cloud is the fastest free path; Procfile + requirements
   are already set for Railway/Render too.
2. UI polish / optional auto-generated executive-summary paragraph (built from
   the engine's own numbers — no LLM data).
3. Extend liquidity: per-asset liquidity-adjusted VaR, or a book-size slider
   preset that showcases a small/mid-cap basket where days-to-liquidate bites.
