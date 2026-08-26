# Meleona - Institutional-Grade Portfolio Optimization & Risk Engine

A quantitative risk dashboard that lets anyone load a portfolio - equities, ETFs, FX, or futures - and stress-test it against thousands of simulated and historical market paths, live. Built by a 2nd-year Data Analytics student to recruiter-facing, quant-interview standards.

**Live demo:** _coming August 2026_

---

## What This Is

Most student finance projects pull data and plot a chart. This one is built to the standard a quant desk would actually defend: modular Python, statistically rigorous math, honest data provenance, and a Streamlit dashboard that leads with **one number** - a headline CVaR verdict - with all the depth a step behind it.

Every market figure is fetched from Yahoo Finance at runtime and computed by the engine's own `numpy`/`scipy` code. **No value is ever hardcoded, estimated, or model-generated** - the provenance panel makes that auditable.

---

## Features (shipped)

- **Configurable universe** - one-click preset baskets (mega-cap tech, 13F-popular names, sector ETFs, FX majors, futures) or type any Yahoo symbol.
- **Freshness-aware data engine** - per-universe parquet cache with a UTC provenance record on every pull; a data-health check flags staleness, gaps, and insufficient history.
- **Two allocation methods** - equal weight and risk parity (equal-risk-contribution), with an optional volatility-targeting overlay (leverage up/down to a target annual vol).
- **Two Monte Carlo engines** - bootstrap resampling *and* a Merton jump-diffusion process (Poisson jumps on Gaussian diffusion) for a fatter, more honest tail. Swappable live.
- **Tail risk** - historical & parametric VaR, CVaR (expected shortfall), and a Kupiec proportion-of-failures backtest that validates the VaR model rather than just reporting it.
- **Stress testing** - custom parametric shocks (drawdown + volatility) *or* replay of the actual daily returns of real crisis windows (dot-com, GFC, COVID, 2022, SVB, …), preserving real correlation breakdown.
- **Risk decomposition** - per-asset risk-contribution vs. dollar-weight, named factor exposures (market/size/value/momentum via ETF proxies), and a correlation matrix.
- **Liquidity modeling** - days-to-liquidate via a participation-rate model on average daily dollar volume; names with no volume feed are flagged, not faked.
- **Grit Zone** - ranks each asset's own price history on drawdown-recovery speed, rolling 1-year consistency, and resilience across real historical crisis windows; a relative "perseverance" score, not a mood index.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, `pandas`, freshness-aware parquet cache (`pyarrow`) |
| Math | `numpy`, `scipy` - covariance/eigen-decomposition, CVaR, Monte Carlo, jump-diffusion |
| Allocation | Risk parity (ERC), volatility targeting |
| Dashboard | `streamlit`, `plotly` |
| Deployment | Streamlit Community Cloud / Railway / Render (`Procfile` ready) |

---

## Run It

```powershell
python -m pip install -r requirements.txt
python -m streamlit run main.py
```
Opens at http://localhost:8501.

### Tests

```powershell
python -m tests.test_engine      # standalone, no extra deps
# or, if you have pytest installed:
pytest
```
Deterministic math-invariant tests (CVaR ≥ VaR, risk parity equalizes contributions, the jump-diffusion mean-consistency identity, vol targeting hits its target, liquidity monotonic in book size) plus a full-app boot test that self-skips when offline.

---

## Project Structure

```
risk-engine/
├── .streamlit/config.toml   # Institutional beige/bronze theme
├── assets/logo.svg          # Lion-crest wordmark
├── data/                    # Cached prices + provenance (gitignored)
├── src/
│   ├── ingestion.py         # DataEngine: fetch, cache, provenance, dollar volume
│   ├── analytics.py         # Covariance, correlation, eigen-decomposition
│   ├── risk.py              # VaR, CVaR, Kupiec backtest, bootstrap + jump-diffusion MC
│   ├── factors.py           # Named factor exposures (ETF-proxy regression)
│   ├── strategies.py        # Risk parity, vol targeting, risk contribution
│   ├── scenarios.py         # Historical regime replay (real crisis windows)
│   ├── liquidity.py         # Days-to-liquidate (participation-rate model)
│   └── grit.py              # Grit Zone: recovery/consistency/resilience scoring
├── tests/test_engine.py     # Regression suite
├── requirements.txt · Procfile · main.py
```

---

## Operations, support and policies

- **Support:** meleona.support@gmail.com - quote the session reference shown in
  the page footer and it maps straight to that session's logs (`SUPPORT.md`).
- **Policies:** `PRIVACY.md` (no accounts, no tracking, no third-party SDKs),
  `TERMS.md` (educational analysis, not investment advice) and
  `ACCESSIBILITY.md` (WCAG 2.2 AA target, the measured contrast table, and the
  four known gaps - stated rather than glossed).
- **Runbook:** `RUNBOOK.md` - kill switch, hotfix/rollback, crash triage,
  outbound spend caps, phased release, reviewer walkthrough.
- **Launch checklist:** `LAUNCH_CHECKLIST.md` - every pre-launch item, what was
  built, and what is deliberately not applicable to an app with no accounts,
  no database, no payments and no email.

## Security posture

The app is public, anonymous and single-process, so the threat model is
**resource exhaustion and injection, not authorization** - there are no
accounts, no roles, no secrets and no database to protect. Controls are
matched to that model rather than to a checklist:

| Control | Where | What it stops |
|---|---|---|
| Ticker funnel (`VALID_TICKER`) | `src/ingestion.py` | Free-text symbols reaching HTML sinks, fetch URLs or cache paths |
| Egress allowlist | `src/netguard.py` | Any outbound request leaving the two enumerated market-data hosts, including via redirect - the SSRF control for an app with no user-supplied URLs |
| Complexity budget (`MAX_UNIVERSE`, cache size caps) | `src/ingestion.py` | One visitor spending the whole server's compute, or filling the disk with novel cache keys |
| Script-splice escaping | `src/topology.py` | A payload string closing the map's `<script>` block early |
| `toolbarMode = "viewer"` | `.streamlit/config.toml` | Visitors reaching Streamlit's operator actions (Rerun, Clear cache, Settings) |
| XSRF + CORS on, tracebacks off | `.streamlit/config.toml` | Cross-site request forgery; stack traces leaking paths and versions |

Each one has a regression test in `tests/test_engine.py`. Two things are
deliberately NOT done, because they would be theatre here: constant-time
comparison (there is no secret to compare) and query depth limiting (there is
no query language - the compute budget above is its real analogue).

Known residual risks, stated rather than hidden: hiding the toolbar removes
the affordance, not the underlying Streamlit protocol messages, so a crafted
websocket frame could still force a rerun - bounded to denial-of-service over
public market data; and cache hit/miss latency is a weak timing signal about
which symbols other visitors requested.

## Design Principles

- **No LLM-originated data, ever.** Every number traces to Yahoo Finance at runtime; the provenance panel proves it.
- **Honest labeling.** "Live end-of-day," not "real-time." Scenarios "replay actual returns." Excluded assets are disclosed, not silently dropped.
- **Lead with one number.** One headline CVaR verdict; all depth collapsed into expanders.
- **Defensible in an interview.** Every feature carries a "Quant Deep Dive" explaining the math - methodology depth over visual complexity.

---

## Roadmap

| Status | Milestone |
|---|---|
| ✅ | Data engine, caching, provenance, data-integrity checks |
| ✅ | Covariance / correlation / eigen-decomposition |
| ✅ | CVaR + bootstrap Monte Carlo; VaR methods + Kupiec backtest |
| ✅ | Streamlit dashboard, institutional theme, crest |
| ✅ | Risk parity, vol targeting, risk contribution, named factors |
| ✅ | Historical regime replay, liquidity modeling, jump-diffusion engine |
| ⬜ | **Live deployment to a public URL** (the recruiter link) |
| ⬜ | Optional: auto executive-summary, further UI polish |

**Target ship date: August 23, 2026**
