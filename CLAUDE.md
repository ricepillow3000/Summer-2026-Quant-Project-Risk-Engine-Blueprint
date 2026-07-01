# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An institutional-grade **Portfolio Optimization & Risk Engine** with a live Streamlit dashboard, built to recruiter-facing / hedge-fund-interview standards (2nd-year Data Analytics student portfolio project, targeting a public live link on a resume, ship target Aug 23, 2026).

## Commands

```powershell
python -m streamlit run main.py
```
Opens at http://localhost:8501. On this machine `pip` must be invoked as `python -m pip`.

**Only run ONE Streamlit server at a time** — stale servers have caused an `ImportError` before. Kill strays with:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Tests live in `tests/test_engine.py` — run `python -m tests.test_engine` (standalone, no extra deps) or `pytest` if installed. They assert math invariants (CVaR ≥ VaR, risk-parity equalization, the jump-diffusion mean-consistency identity, vol-target accuracy, liquidity monotonicity) on deterministic synthetic data, plus a full-app boot test via Streamlit's `AppTest` that self-skips when offline. Also smoke-test the affected module directly (each `src/*.py` runs as `python -m src.<module>`) and run the full app to confirm it renders without exceptions.

## Architecture

`main.py` is the entire Streamlit UI — a single linear script (no multi-page/component split) that renders top to bottom: **universe selection → data load → allocation → stress test → headline verdict → expandable risk breakdown → liquidity → provenance**. Streamlit reruns this script on every widget interaction, so ordering matters: each section's variables (`weights`, `returns`, `shocked_returns`, etc.) feed the sections below it.

`src/` holds pure computation, no Streamlit imports — every module here is a numpy/pandas library independent of the UI, which is what `main.py` composes:

- **`ingestion.py`** — `DataEngine`. Fetches from Yahoo Finance (`yfinance`) at runtime, never from a static fixture or an LLM. Per-universe disk cache (parquet, keyed by an md5 hash of sorted tickers + period) under `data/`, freshness-aware (`CACHE_MAX_AGE_HOURS = 6`) so a deployed app never serves indefinitely-stale data. A single download (`_download_close_volume`) populates BOTH the price cache and the dollar-volume cache, so the app's cold load is one network round-trip, not two. `fetch_risk_free_rate` pulls `^IRX` (13-week T-bill) for the Sharpe ratio, returning `None` (never a fabricated rate) on failure. Every fetch writes a `.meta.json` provenance record (source, UTC timestamp, symbols, coverage) — this is what powers the "Data source & provenance" panel. `PRESETS` defines the one-click starter baskets shown in the UI. `align=True` intersects all tickers onto common trading days (right for live risk calcs); `align=False` is used by `scenarios.py` for historical replay, where truncating to the youngest asset's inception would destroy older regime windows.
- **`analytics.py`** — annualized covariance/correlation, eigen-decomposition, portfolio volatility.
- **`risk.py`** — VaR (historical + parametric), CVaR, Kupiec VaR backtest, bootstrap Monte Carlo (`monte_carlo`), and Merton jump-diffusion Monte Carlo (`calibrate_jump_diffusion` + `jump_diffusion_mc`) as a fat-tailed alternative to plain bootstrapping.
- **`factors.py`** — named factor exposures (market/size/value/momentum) via ETF-proxy regression.
- **`strategies.py`** — risk-contribution decomposition, risk parity (ERC) weights, volatility targeting/leverage scaling.
- **`scenarios.py`** — `HISTORICAL_REGIMES`: replays the *actual* historical daily returns of named crisis windows (dot-com, GFC, COVID, etc.) rather than a parametric shock, preserving real cross-asset correlation breakdown. Assets that didn't trade in a window are excluded and the UI discloses this.
- **`liquidity.py`** — days-to-liquidate via a participation-rate model (`ADV = avg daily dollar volume`, capped participation rate, slowest leg determines the exit horizon), fed by `ingestion.average_dollar_volume`.

Two Monte Carlo engines are swappable in the UI (`monte_carlo` vs `jump_diffusion_mc` in `risk.py`), both consumed identically by `main.py` (same input signature: return series + weights + horizon).

## Non-negotiable project constraints

These come from the project's design intent and should guide any feature work, not just be treated as style preferences:

1. **No LLM-originated data, ever.** Every market number must come from Yahoo Finance via `yfinance` at runtime, computed by the engine's own numpy/scipy code. Never hardcode or estimate a market figure. The provenance panel exists specifically to make this auditable.
2. **Honest labeling — no overclaiming.** E.g. "live end-of-day data" not "real-time"; scenarios "replay actual returns" not "simulate"; excluded assets are disclosed, not silently dropped. Overclaiming is called out as the single biggest way this fails a quant-interview credibility check.
3. **Lead with one number.** UI philosophy: one headline CVaR verdict + one sentence up top; everything else (correlation matrix, factor exposures, VaR backtest, liquidity) collapsed into expanders. Don't flatten this back into a wall of charts.
4. **Defensible in an interview.** Each risk feature carries a "Quant Deep Dive" docstring/caption explaining the underlying math (see the module docstrings in `src/`) — methodology depth over visual complexity.
5. **Aesthetic:** Citadel-style institutional theme — beige `#EDE9E3`/`#D4CDBF`, bronze `#9A7B4F`/`#8A6A3C`, charcoal `#3F3B35`, serif (Georgia). Defined in `.streamlit/config.toml` and the inline `<style>` block at the top of `main.py`. Keep new UI additions consistent with this (no bright/flashy colors, no sans-serif headings).

## Workflow conventions

- Verify every change by running it — smoke-test the affected `src/` module directly, then run the Streamlit app end-to-end.
- Cache files (`data/*.parquet`, `data/*.meta.json`) and `streamlit.log` are gitignored; don't hand-edit or commit them.
- Commit after each working feature with a descriptive message.
