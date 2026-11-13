# Meliona: portfolio optimization and risk engine

A quantitative risk dashboard that lets anyone load a portfolio (equities, ETFs, FX or futures) and stress-test it against thousands of simulated and historical market paths, live. Built to recruiter-facing, quant-interview standards by a Computer Information Systems student specializing in AI & Analytics.

Live: https://summer-2026-quant-project-risk-engine-blueprint-production.up.railway.app

## What this is

Modular Python, math tested against the papers it comes from, data provenance on every pull, and a Streamlit dashboard that leads with one number, a headline CVaR verdict, with all the depth a step behind it.

Every market figure is fetched from Yahoo Finance at runtime and computed by the engine's own `numpy`/`scipy` code. No value is hardcoded, estimated or model-generated, and the provenance panel makes that auditable.

## Features (shipped)

- Configurable universe: one-click preset baskets (mega-cap tech, 13F-popular names, sector ETFs, FX majors, futures), or type any Yahoo symbol.
- Freshness-aware data engine: a per-universe parquet cache with a UTC provenance record on every pull; a data-health check flags staleness, gaps and insufficient history.
- Two allocation methods: equal weight and risk parity (equal risk contribution), with an optional volatility-targeting overlay that levers up or down to a target annual vol.
- Two Monte Carlo engines: a stationary block bootstrap, which keeps crash weeks in one piece, and a Merton jump-diffusion process (Poisson jumps on Gaussian diffusion), which can simulate a day worse than any in the sample. Swappable live.
- Tail risk: the headline is a one-year CVaR (expected shortfall) of the simulated outcomes, widened for the days it takes to unwind the book.
- Stress testing: custom parametric shocks (drawdown and volatility), or a replay of the actual daily returns of real crisis windows (dot-com, GFC, COVID, 2022, SVB, …), which keeps their real correlation breakdown.
- Risk decomposition: per-asset risk contribution against dollar weight, and a Sharpe ratio against the real T-bill rate.
- Liquidity modeling: days to liquidate from a participation-rate model on average daily dollar volume, which sizes The Book and widens the headline tail; names with no volume feed are flagged, not filled in.
- The Book and the Risk Topology map: which high-flying names lead the book and which steady anchors pair with them, from the book's own history.
- Crisis Conviction: what a buyer of the S&P 500 actually earned after every named crisis, at the trough and at the worst-timed peak.
- Grit Zone: ranks each asset's own price history on drawdown-recovery speed, rolling 1-year consistency and resilience across real crisis windows. The score is relative to the chosen basket.
- Case studies: three named crashes (COVID-19, 2022, the 2008 crisis) as one row each - the fall, the trading days back to the old peak, and a bottom-day buyer a year on - read from the live crisis record.
- About, questions and contact: the builder's portrait and profile links, five plain questions (also served as FAQPage JSON-LD), and a contact form that prepares an email in the visitor's own app - there is no mail server, so nothing is sent or stored by the site.

## Tech stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, `pandas`, freshness-aware parquet cache (`pyarrow`) |
| Math | `numpy`, `scipy`: covariance and eigen-decomposition, CVaR, Monte Carlo, jump-diffusion |
| Allocation | Risk parity (ERC), volatility targeting |
| Dashboard | `streamlit`, `plotly` |
| Deployment | Railway (`Procfile`), static files served by Streamlit |

## Run it

```powershell
python -m pip install -r requirements.txt
python -m streamlit run main.py
```
Opens at http://localhost:8501.

### Run it anywhere (Docker)

The same app in a box, on any machine with [Docker](https://docs.docker.com/get-docker/), with no Python or packages to install:

```bash
docker run -p 8501:8501 ghcr.io/ricepillow3000/meliona:latest
```

Then open http://localhost:8501. The image is built, started and checked on
every push, and published from `main` for Intel/AMD and Apple Silicon machines
(`.github/workflows/container.yml`). To build it yourself:

```bash
docker build -f docker/Dockerfile -t meliona .
docker run -p 8501:8501 meliona
```

The box runs exactly what the live site runs (`Procfile`), as a non-root user,
with Docker's health check on Streamlit's `/_stcore/health`. Why Docker and not
Kubernetes: `RUNBOOK.md` §4.

### Tests

```powershell
python -m tests.test_engine      # standalone, no extra deps
# or, if you have pytest installed:
pytest
```
Deterministic math-invariant tests (CVaR ≥ VaR, risk parity equalizes contributions, the jump-diffusion mean-consistency identity, vol targeting hits its target, liquidity monotonic in book size), plus a full-app boot test that skips itself when offline.

## Project structure

```
risk-engine/
├── .streamlit/config.toml   # Beige/bronze theme, server hardening
├── assets/logo.svg          # Crest: hexagon cut by an X, on a ringed charcoal disc
├── data/                    # Cached prices + provenance (gitignored)
├── src/
│   ├── ingestion.py         # DataEngine: fetch, cache, provenance, dollar volume
│   ├── analytics.py         # Covariance, correlation, eigen-decomposition
│   ├── risk.py              # VaR, CVaR, Kupiec backtest, bootstrap + jump-diffusion MC
│   ├── strategies.py        # Risk parity, vol targeting, risk contribution
│   ├── scenarios.py         # Historical regime replay (real crisis windows)
│   ├── liquidity.py         # Days-to-liquidate (participation-rate model)
│   └── grit.py              # Grit Zone: recovery/consistency/resilience scoring
├── tests/test_engine.py     # Regression suite
├── requirements.txt · Procfile · main.py
```

## Operations, support and policies

- Support: meleona.support@gmail.com. Quote the session reference shown in
  the page footer and it maps straight to that session's logs (`SUPPORT.md`).
- Policies: `PRIVACY.md` (no accounts, no tracking, no third-party SDKs),
  `TERMS.md` (educational analysis, not investment advice) and
  `ACCESSIBILITY.md` (WCAG 2.2 AA target, the measured contrast table, and the
  four known gaps).
- Runbook: `RUNBOOK.md` covers the kill switch, hotfix and rollback, crash
  triage, outbound spend caps, phased release and the reviewer walkthrough.
- Launch checklist: `LAUNCH_CHECKLIST.md` lists every pre-launch item, what was
  built, and what does not apply to an app with no accounts, no database, no
  payments and no email.

## Security posture

The app is public, anonymous and single-process, so the threat model is
resource exhaustion and injection, not authorization: there are no accounts,
no roles, no secrets and no database to protect. The controls match that
model:

| Control | Where | What it stops |
|---|---|---|
| Ticker funnel (`VALID_TICKER`) | `src/ingestion.py` | Free-text symbols reaching HTML sinks, fetch URLs or cache paths |
| Egress allowlist | `src/netguard.py` | Any outbound request leaving the enumerated Yahoo Finance hosts, including via redirect; the SSRF control for an app with no user-supplied URLs |
| Complexity budget (`MAX_UNIVERSE`, cache size caps) | `src/ingestion.py` | One visitor spending the whole server's compute, or filling the disk with novel cache keys |
| Script-splice escaping | `src/topology.py` | A payload string closing the map's `<script>` block early |
| `toolbarMode = "viewer"` | `.streamlit/config.toml` | Visitors reaching Streamlit's operator actions (Rerun, Clear cache, Settings) |
| XSRF + CORS on, tracebacks off | `.streamlit/config.toml` | Cross-site request forgery; stack traces leaking paths and versions |

Each one has a regression test in `tests/test_engine.py`. Two things are
deliberately not done because nothing here needs them: constant-time
comparison (there is no secret to compare) and query depth limiting (there is
no query language; the compute budget above is its analogue).

Known residual risks: hiding the toolbar removes the affordance, not the
underlying Streamlit protocol messages, so a crafted websocket frame could
still force a rerun, which is bounded to denial of service over public market
data; and cache hit/miss latency is a weak timing signal about which symbols
other visitors requested.

## Design principles

- No LLM-originated data, ever. Every number traces to Yahoo Finance at runtime, and the provenance panel shows it.
- Labels claim no more than the data supports: "live end-of-day", not "real-time"; scenarios "replay actual returns"; excluded assets are disclosed, not silently dropped.
- Lead with one number: one headline CVaR verdict, with the depth folded away below it.
- Defensible in an interview: every feature carries a "Quant Deep Dive" explaining the math.

## Roadmap

| Status | Milestone |
|---|---|
| Done | Data engine, caching, provenance, data-integrity checks |
| Done | Covariance, correlation and eigen-decomposition |
| Done | CVaR and bootstrap Monte Carlo; VaR methods and the Kupiec backtest |
| Done | Streamlit dashboard, beige/bronze theme, crest |
| Done | Risk parity, vol targeting, risk contribution |
| Done | Historical regime replay, liquidity modeling, jump-diffusion engine |
| Done | Live deployment to a public URL (the recruiter link) |
| Planned | Optional: an automatic executive summary, further UI polish |

Target ship date: August 23, 2026.

## Credits

The page's blocks are flat paper and stone, after the share banner
(`tools/og_image.html`). Its only photographs are John Nguyen's own: the
About portrait and the photo beside his story (cropped and graded by
`tools/portrait.py`; not for reuse). The crest
(`assets/logo.svg`) was drawn for this project. The GitHub and Handshake marks
are Simple Icons' paths (CC0); the LinkedIn mark is drawn here.
