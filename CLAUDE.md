# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Second brain (read first)

John's vault at `C:\Users\john4\Claude\Vault` is the cross-session memory for this project. **At session start:** read `90-System\Memory-Bridge.md` (~1 page, current state of everything) and the Active table in `90-System\Lessons.md` (binding rules from past collaboration) - then work; don't re-explore for context those already provide. **At session end / after milestones:** update both, and commit+push the vault (it is its own private git repo, separate from this one). Vault notes summarize; this repo's `CLAUDE.md`/`PROGRESS.md` stay authoritative for repo detail. Never copy vault content into this public repo.

## What this is

**Meleona** is an institutional-grade **Portfolio Optimization & Risk Engine** with a live Streamlit dashboard, built to recruiter-facing / hedge-fund-interview standards (2nd-year Data Analytics student portfolio project, targeting a public live link on a resume, ship target Aug 23, 2026).

## Commands

```powershell
python -m streamlit run main.py
```
Opens at http://localhost:8501. On this machine `pip` must be invoked as `python -m pip`.

**Only run ONE Streamlit server at a time** - stale servers have caused an `ImportError` before. Kill strays with:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'streamlit' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Tests live in `tests/test_engine.py` - run `python -m tests.test_engine` (standalone, no extra deps) or `pytest` if installed. They assert math invariants (CVaR ≥ VaR, risk-parity equalization, the jump-diffusion mean-consistency identity, vol-target accuracy, liquidity monotonicity) on deterministic synthetic data, plus a full-app boot test via Streamlit's `AppTest` that self-skips when offline. Also smoke-test the affected module directly (each `src/*.py` runs as `python -m src.<module>`) and run the full app to confirm it renders without exceptions.

A second, live-data suite lives in `tests/batch_audit.py` - run `python -m tests.batch_audit` (add `--quick` for the first 4 universes, `--no-map` to skip the node probe). Where `test_engine.py` proves the math on synthetic data, this runs the SAME engine across every `PRESETS` universe on live Yahoo data, re-derives each number a second independent way, and writes `audit/batch_audit_<UTC>.{json,md}` where every check names the paper it is checked against. It covers the engine invariants, a walk-forward VaR backtest matrix (equal-weight and risk-parity books at 95% and 99%, Kupiec + Christoffersen), and a full audit of the Risk Topology map: the map's shipped JS simulation is executed under node (`tests/map_probe.mjs`, which slices the real math region out of `prototypes/war_room.html`) and its output is checked against closed-form OU moments, ring masses, breach semantics and a planted-parameter control. Exit code is non-zero on any FAIL; WARN never blocks. Requires `node` on PATH for the map checks.

## Architecture

`main.py` is the entire Streamlit UI - a single linear script (no multi-page/component split) structured as a scroll-driven presentation: **hero pitch → Grit Zone showcase (anchor-scroll CTAs) → "the engine"**, i.e. universe selection → data load → allocation → stress test → headline verdict → fan chart → one `st.tabs()` strip (3D Distribution, Risk Breakdown, Grit Zone, Liquidity, Security Master, Data Quality, Lineage & Audit) instead of stacked expanders. Streamlit reruns this script on every widget interaction, so ordering matters: each section's variables (`weights`, `returns`, `shocked_returns`, etc.) feed the sections below it. The 3D Distribution tab is a self-contained `st.iframe` (raw plotly.js + a particle-overlay animation loop, `living_surface_html()`) rather than `st.plotly_chart`, since that's the only way to get a continuously-animating scene.

`src/` holds pure computation, no Streamlit imports - every module here is a numpy/pandas library independent of the UI, which is what `main.py` composes:

- **`ingestion.py`** - `DataEngine`. Fetches from Yahoo Finance (`yfinance`) at runtime, never from a static fixture or an LLM. Per-universe disk cache (parquet, keyed by an md5 hash of sorted tickers + period) under `data/`, freshness-aware (`CACHE_MAX_AGE_HOURS = 6`) so a deployed app never serves indefinitely-stale data. A single download (`_download_close_volume`) populates BOTH the price cache and the dollar-volume cache, so the app's cold load is one network round-trip, not two. `fetch_risk_free_rate` pulls `^IRX` (13-week T-bill) for the Sharpe ratio, returning `None` (never a fabricated rate) on failure. Every fetch writes a `.meta.json` provenance record (source, UTC timestamp, symbols, coverage) - this is what powers the "Data source & provenance" panel. `PRESETS` defines the one-click starter baskets shown in the UI. `align=True` intersects all tickers onto common trading days (right for live risk calcs); `align=False` is used by `scenarios.py` for historical replay, where truncating to the youngest asset's inception would destroy older regime windows.
- **`analytics.py`** - annualized covariance/correlation, eigen-decomposition, portfolio volatility.
- **`risk.py`** - VaR (historical + parametric), CVaR, Kupiec VaR backtest, bootstrap Monte Carlo (`monte_carlo`), and Merton jump-diffusion Monte Carlo (`calibrate_jump_diffusion` + `jump_diffusion_mc`) as a fat-tailed alternative to plain bootstrapping.
- **`factors.py`** - named factor exposures (market/size/value/momentum) via ETF-proxy regression.
- **`strategies.py`** - risk-contribution decomposition, risk parity (ERC) weights, volatility targeting/leverage scaling.
- **`scenarios.py`** - `HISTORICAL_REGIMES`: replays the *actual* historical daily returns of named crisis windows (dot-com, GFC, COVID, etc.) rather than a parametric shock, preserving real cross-asset correlation breakdown. Assets that didn't trade in a window are excluded and the UI discloses this.
- **`liquidity.py`** - days-to-liquidate via a participation-rate model (`ADV = avg daily dollar volume`, capped participation rate, slowest leg determines the exit horizon), fed by `ingestion.average_dollar_volume`.
- **`grit.py`** - Grit Zone: scores each asset on drawdown-recovery speed/completeness, rolling 1-year consistency, and drawdown/recovery behavior across `scenarios.HISTORICAL_REGIMES`. Pulls each ticker's OWN full price history (`fetch_prices(period="max", align=False)`, same unaligned pattern as `scenarios.py`) rather than the 2y window used elsewhere. Every component is percentile-ranked RELATIVE to the chosen universe (`_score01`), not an absolute scale - a different basket changes every ticker's score. Tickers with `< MIN_HISTORY_DAYS` are excluded from ranking, not scored on thin data.
- **`security_master.py`** - free-tier reference data: ISIN via `yfinance`'s `Ticker.isin` (flagged `"unavailable"`, not guessed, when the feed doesn't have one), plus real dividend/split event history via `Ticker.dividends`/`Ticker.splits`. `auto_adjust=True` already folds these into every price series used elsewhere - this module makes the underlying events visible, it doesn't change any risk number. SEDOL/CUSIP/merger history need a paid vendor and are explicitly not attempted.
- **`state_calibration.py`** - OU/AR(1) calibration for the map, plus the **dispersion correction**: `dispersion_correction()` walks the book's own state history forward (refit on prior data only, closed-form 30-day-ahead distribution, Mahalanobis score against what actually happened) and returns the factor `k` that makes the drawn rings hold their labelled mass. `calibrate_state_dynamics` multiplies BOTH shock sizes by it, and the map footnote discloses `k` with the before/after coverage. `k` prices plug-in estimation error as well as the overlapping-window smoothing - a state path drawn from exactly this model still under-covers, because the horizon distribution is built from point estimates. It is clamped to [1.0, 3.0]: it widens a too-narrow terrain, it never narrows a conservative one.
- **`netguard.py`** - egress allowlist. Every outbound market-data call (`yf.download`, `yf.Ticker`) goes through `guarded_session()`, a `curl_cffi` session that refuses any host outside `ALLOWED_HOSTS` and re-checks the final URL so a redirect cannot walk out. It subclasses curl_cffi rather than `requests` deliberately: yfinance accepts a plain `requests.Session` but Yahoo answers it with `YFRateLimitError` (non-impersonated TLS fingerprint), so the "safer" transport would have broken the data path. Adding a feed means adding its host here, on purpose.
- **`topology.py`** - the Risk Topology map's payload builder (`build_map_payload`) and the HTML splice (`war_room_html`). It is the ONE place the map's numbers are assembled, which is what lets `tests/batch_audit.py` audit the exact payload the app ships rather than a copy of it. Hazard-perimeter constants (3x vol floored at 35% / capped at 90%, beta +0.8 capped at 1.9) live here as named constants because they are disclosed policy, not estimates.
- **`data_quality.py`** - hand-rolled (no new dependency) validation gate: `validate_prices(prices)` runs schema/positivity/coverage/staleness/calendar-gap/extreme-move checks on every price pull and returns a PASS/WARN/FAIL report. WARN surfaces an issue (e.g. a real crash day) without blocking; only FAIL flips `passed=False`.

Two Monte Carlo engines are swappable in the UI (`monte_carlo` vs `jump_diffusion_mc` in `risk.py`), both consumed identically by `main.py` (same input signature: return series + weights + horizon).

## Security posture (public, anonymous, single-process)

No accounts, no roles, no secrets, no database - so the threat model is resource exhaustion and injection, NOT authorization. Controls, each with a regression test in `tests/test_engine.py`: the `VALID_TICKER` funnel (the primary control - it is what keeps free text out of HTML sinks, fetch URLs and cache paths), the `src/netguard.py` egress allowlist, the `MAX_UNIVERSE` complexity budget plus `CACHE_MAX_FILES`/`CACHE_MAX_MB`, `<`-escaping of the JSON spliced into the map's `<script>` block, and `.streamlit/config.toml` (`toolbarMode = "viewer"` so visitors get no operator toolbar, XSRF/CORS on, `showErrorDetails` off).

Two things are deliberately absent and should stay absent unless the app gains the surface: constant-time comparison (nothing secret is compared) and query depth/complexity limiting (no query language - `MAX_UNIVERSE` is its analogue). Adding either without a matching surface is checklist theatre and reads that way in an interview. When touching this area, keep the funnel first: widening `VALID_TICKER` weakens every downstream sink at once.

## Launch operations

- **Kill switch:** `MELEONA_MAINTENANCE=1` (plus optional `MELEONA_MAINTENANCE_NOTE`) in the host environment serves a maintenance page and `st.stop()`s before any market-data call. No redeploy.
- **Crash wire:** `src/observability.py` - rotating local log under `logs/` (gitignored), one random `session_ref` per browser session shown in the page footer, `log_incident()` for handled failures. Deliberately NOT a hosted SDK: shipping visitor data to a vendor would contradict `PRIVACY.md`.
- **Spend caps:** `netguard.MAX_REQUESTS_PER_DAY` charged at the single outbound chokepoint; exhaustion degrades `fetch_prices` to cached data rather than hammering Yahoo. `ingestion.MAX_UNIVERSE` and the cache size caps bound compute and disk.
- **Support:** `SUPPORT_EMAIL` in `main.py` is the single source - footer, maintenance page and the privacy panel all read it. Users quote the session reference; `RUNBOOK.md` §3 turns it into a traceback.
- **Docs:** `PRIVACY.md`, `TERMS.md`, `SUPPORT.md`, `RUNBOOK.md` (kill switch, rollback, triage, budgets, phased release, reviewer path, SPF/DKIM/DMARC for the day a domain exists) and `LAUNCH_CHECKLIST.md` (every launch ask, built or N/A with the reason).

There is no database, no account, no payment and no email sending. When a launch-checklist item assumes one of those, the honest move is to build the analogue and say plainly why the original does not apply - see the N/A column in `LAUNCH_CHECKLIST.md`. Adding an account-deletion flow with no accounts, or SPF records with no sending domain, is theatre.

## Non-negotiable project constraints

These come from the project's design intent and should guide any feature work, not just be treated as style preferences:

1. **No LLM-originated data, ever.** Every market number must come from Yahoo Finance via `yfinance` at runtime, computed by the engine's own numpy/scipy code. Never hardcode or estimate a market figure. The provenance panel exists specifically to make this auditable.
2. **Honest labeling - no overclaiming.** E.g. "live end-of-day data" not "real-time"; scenarios "replay actual returns" not "simulate"; excluded assets are disclosed, not silently dropped. Overclaiming is called out as the single biggest way this fails a quant-interview credibility check.
3. **Lead with one number.** UI philosophy: one headline CVaR verdict + one sentence up top; everything else (correlation matrix, factor exposures, VaR backtest, liquidity) collapsed into expanders. Don't flatten this back into a wall of charts.
4. **Defensible in an interview.** Each risk feature carries a "Quant Deep Dive" docstring/caption explaining the underlying math (see the module docstrings in `src/`) - methodology depth over visual complexity.
5. **Aesthetic:** Citadel-style institutional theme - beige `#EDE9E3`/`#D4CDBF`, bronze `#9A7B4F`/`#8A6A3C`, charcoal `#3F3B35`, serif (Georgia). Defined in `.streamlit/config.toml` and the inline `<style>` block at the top of `main.py`. Keep new UI additions consistent with this (no bright/flashy colors, no sans-serif headings).

## Workflow conventions

- Verify every change by running it - smoke-test the affected `src/` module directly, then run the Streamlit app end-to-end.
- Cache files (`data/*.parquet`, `data/*.meta.json`) and `streamlit.log` are gitignored; don't hand-edit or commit them.
- Commit after each working feature with a descriptive message.
