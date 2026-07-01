"""
Portfolio Risk Engine — Streamlit entry point.

Design philosophy:
A risk desk doesn't hand a PM eight charts and say "figure it out." It leads
with one verdict and one number. Everything else is detail you open on demand.

Phase V:
The universe is now chosen by the viewer, not hard-coded. Anyone can load a
preset basket (equities, sector ETFs, FX, futures) or type their own symbols,
so the engine speaks to any audience — not just one watchlist.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.ingestion import (
    fetch_prices, get_returns, data_health, provenance, clear_cache,
    average_dollar_volume, fetch_risk_free_rate, PRESETS,
)
from src.analytics import correlation_matrix, covariance_matrix
from src.risk import (
    monte_carlo, jump_diffusion_mc, parametric_var, var_backtest, sharpe_ratio,
)
from src.factors import factor_exposures
from src.strategies import risk_contributions, risk_parity_weights, vol_target
from src.scenarios import HISTORICAL_REGIMES, replay_returns
from src.liquidity import days_to_liquidate, liquidity_profile

st.set_page_config(page_title="Portfolio Risk Engine", layout="centered")

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
st.markdown("""
<style>
html, body, [class*="css"] { font-family: Georgia, 'Times New Roman', serif; }
h1, h2, h3 { color: #3F3B35; font-weight: 400; }
/* Captions: larger + darker so the fine print is actually readable */
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] div {
    font-size: 13.5px !important; color: #524E47 !important; line-height: 1.5 !important; }
[data-testid="stMetricLabel"] p { font-size: 13px !important; color: #6A645A !important; }

/* Header crest + wordmark */
.brand-row { display: flex; align-items: center; gap: 16px; margin-bottom: 2px; }
.brand-title { font-size: 32px; color: #3F3B35; line-height: 1.1; }
.brand-tag { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
             letter-spacing: 0.16em; text-transform: uppercase; color: #8A6E45; }

/* Hero verdict card */
.verdict-box { background: #F4F1EA; border: 1px solid #BFB8A9; border-radius: 6px;
               padding: 28px 32px; margin-bottom: 24px;
               box-shadow: 0 2px 6px rgba(63,59,53,0.10); }
.verdict-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
                  letter-spacing: 0.12em; text-transform: uppercase; color: #9A8E7C; }
.verdict-number { font-size: 48px; color: #3F3B35; margin: 4px 0; }
.verdict-sentence { font-size: 17px; color: #54504A; line-height: 1.5; }

/* Control panels — distinct cream blocks with a bronze top accent */
.panel-label { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
               letter-spacing: 0.12em; text-transform: uppercase; color: #7A6E5A;
               margin-bottom: 4px; }

/* Refined slider */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #C4BDAE !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #9A7B4F !important; border: 2px solid #F4F1EA !important;
    box-shadow: 0 1px 3px rgba(63,59,53,0.25) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #3F3B35 !important; font-family: 'Helvetica Neue', sans-serif !important;
    font-size: 12px !important; }

/* Expander surface */
[data-testid="stExpander"] { border: 1px solid #C4BDAE !important;
    border-radius: 6px !important; background: #ECE7DD !important; }
</style>
""", unsafe_allow_html=True)

# ---- Header: crest + wordmark ----
with open("assets/logo.svg", "r", encoding="utf-8") as f:
    logo_svg = f.read()

st.markdown(f"""
<div class="brand-row">
  <div style="width:58px; height:58px;">{logo_svg}</div>
  <div>
    <div class="brand-title">Portfolio Risk Engine</div>
    <div class="brand-tag">Pride &middot; Integrity</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("Stress-test any portfolio against thousands of simulated market paths.")

# ---- Universe selection ----
with st.container(border=True):
    st.markdown('<div class="panel-label">Universe</div>', unsafe_allow_html=True)
    preset = st.selectbox("Preset basket", list(PRESETS.keys()), label_visibility="collapsed")

    # Keying the multiselect on the preset name makes it re-initialize with the
    # new default whenever the preset changes — while still letting users add or
    # remove individual symbols (accept_new_options allows arbitrary tickers).
    suggestions = sorted({t for lst in PRESETS.values() for t in lst})
    chosen = st.multiselect(
        "Tickers to analyze",
        options=suggestions,
        default=PRESETS[preset],
        key=f"tickers__{preset}",
        accept_new_options=True,
        help="Pick a preset above, or add any Yahoo Finance symbol "
             "(e.g. BRK-B, EURUSD=X for FX, GC=F for gold futures).",
    )

tickers = sorted({t.strip().upper() for t in chosen if t.strip()})
if len(tickers) < 2:
    st.warning("Add at least two symbols to analyze a portfolio.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def load_universe(tickers_tuple: tuple[str, ...], period: str = "2y"):
    return fetch_prices(list(tickers_tuple), period=period)


@st.cache_data(ttl=3600, show_spinner="Loading volume data…")
def load_adv(tickers_tuple: tuple[str, ...]):
    """Average daily dollar volume per ticker (recent 3-month lookback)."""
    return average_dollar_volume(list(tickers_tuple))


@st.cache_data(ttl=3600)
def load_risk_free_rate():
    """Latest 13-week T-bill yield (^IRX) as an annual decimal, or None."""
    return fetch_risk_free_rate()


try:
    prices = load_universe(tuple(tickers))
except Exception as exc:  # noqa: BLE001 — surface any fetch failure to the user
    st.error(f"Couldn't load market data: {exc}")
    st.stop()

returns = get_returns(prices)
loaded = list(prices.columns)
missing = [t for t in tickers if t not in loaded]
if len(loaded) < 2:
    st.error("Fewer than two symbols returned data. Try different tickers.")
    st.stop()
if missing:
    st.caption(f"Couldn't load: {', '.join(missing)} — skipped.")

# ---- Data-freshness indicator (honest, not a fake real-time feed) ----
health = data_health(prices)
fresh_col, refresh_col = st.columns([5, 1])
fresh = "live" if health["staleness_days"] <= 1 else f"{health['staleness_days']}d old"
fresh_col.caption(
    f"Data: {health['rows']} trading days · through {health['end']} · {fresh}"
)
if refresh_col.button("Refresh", help="Clear cache and re-pull the latest prices."):
    clear_cache(tickers)       # drop disk cache so Yahoo is hit fresh
    st.cache_data.clear()      # drop Streamlit's in-memory cache
    st.rerun()

# ---- Allocation: equal-weight vs risk parity, optional vol target ----
cov = covariance_matrix(returns)  # annualized covariance for risk math
with st.container(border=True):
    st.markdown('<div class="panel-label">Allocation</div>', unsafe_allow_html=True)
    acol1, acol2 = st.columns(2)
    method = acol1.radio(
        "Weighting", ["Equal weight", "Risk parity"], label_visibility="collapsed",
        help="Risk parity equalizes each asset's RISK contribution, so no single "
             "name dominates — the Bridgewater All-Weather idea.")
    use_vt = acol2.checkbox(
        "Target volatility", help="Scale exposure to hold a constant annual vol "
        "(AQR managed-vol style). Leverage < 1 de-risks; > 1 levers up.")
    target_vol = acol2.slider("Target annual vol (%)", 5, 30, 10, step=1,
                              disabled=not use_vt) / 100

base_weights = risk_parity_weights(cov) if method == "Risk parity" else \
    np.ones(len(loaded)) / len(loaded)

leverage = 1.0
if use_vt:
    vt = vol_target(base_weights, cov, target_vol)
    weights, leverage = vt["scaled_weights"], vt["leverage"]
else:
    weights = base_weights

port_returns = returns @ weights  # real (unshocked) portfolio series for VaR/factors

# ---- Stress test: custom parametric shock OR historical regime replay ----
alloc_label = "risk-parity" if method == "Risk parity" else "equal-weight"
lev_txt = f", levered {leverage:.2f}×" if use_vt else ""

with st.container(border=True):
    st.markdown('<div class="panel-label">Stress test</div>', unsafe_allow_html=True)
    engine = st.radio(
        "Return model", ["Bootstrap (empirical)", "Jump-diffusion (Merton)"],
        horizontal=True,
        help="Bootstrap resamples real historical days — it can only replay tails "
             "it has already seen. Jump-diffusion (Merton 1976) adds Poisson jumps "
             "on top of Gaussian diffusion, generating NEW extremes — deeper crashes "
             "and jump clusters — for a fatter, more honest tail.")
    mode = st.selectbox(
        "Scenario", ["Custom shock (sliders)"] + list(HISTORICAL_REGIMES.keys()),
        help="Custom: set your own drawdown and volatility shock. Or replay the "
             "ACTUAL daily returns of a real crisis — real correlations, real "
             "volatility, real path, not an approximation.")
    if mode == "Custom shock (sliders)":
        col1, col2 = st.columns(2)
        drawdown_shock = col1.slider(
            "Market drawdown shock", -50, 0, 0, step=5,
            help="Shifts every historical daily return down before resampling.")
        vol_shock = col2.slider(
            "Volatility shock", 0, 300, 0, step=10,
            help="Scales the spread of daily returns to simulate a higher-vol regime.")
    else:
        s_date, e_date = HISTORICAL_REGIMES[mode]
        st.caption(f"Replaying actual market returns from {s_date} to {e_date}.")

# Build the return distribution + weights the simulation will sample from.
if mode == "Custom shock (sliders)":
    shocked_returns = returns.copy()
    if drawdown_shock != 0:
        shocked_returns = shocked_returns + (drawdown_shock / 100) / 252
    if vol_shock != 0:
        m = shocked_returns.mean()
        shocked_returns = m + (shocked_returns - m) * (1 + vol_shock / 100)
    sim_weights = weights
    excluded = []
    is_shocked = drawdown_shock != 0 or vol_shock != 0
    scenario_label = None
else:
    s_date, e_date = HISTORICAL_REGIMES[mode]
    try:
        shocked_returns = replay_returns(loaded, s_date, e_date)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load history for {mode}: {exc}")
        st.stop()
    sim_assets = list(shocked_returns.columns)
    excluded = [t for t in loaded if t not in sim_assets]
    if len(sim_assets) < 2:
        st.warning(f"Too few of your assets traded during {mode}. Try another scenario.")
        st.stop()
    idx = [loaded.index(a) for a in sim_assets]
    sim_weights = weights[idx]
    sim_weights = sim_weights / sim_weights.sum() * weights.sum()  # preserve exposure
    is_shocked = True
    scenario_label = mode

use_jd = engine.startswith("Jump-diffusion")
mc_fn = jump_diffusion_mc if use_jd else monte_carlo
mc = mc_fn(shocked_returns, sim_weights, n_simulations=10_000, horizon_days=252)

# ---- Headline verdict ----
if scenario_label:
    verdict = (
        f"Replaying the actual returns of {scenario_label} "
        f"({len(shocked_returns)} trading days), a {alloc_label} portfolio{lev_txt} "
        f"loses an average of **{mc['cvar']:.1%}** in the worst 5% of simulated years."
    )
    if excluded:
        verdict += f" *(Excludes {', '.join(excluded)} — not trading in that period.)*"
else:
    verdict = (
        f"In the worst 5% of simulated years, a {alloc_label} portfolio of these "
        f"{len(loaded)} assets{lev_txt} loses an average of **{mc['cvar']:.1%}**."
    )
    if is_shocked:
        verdict += " *(under the stress scenario applied above)*"

st.markdown(f"""
<div class="verdict-box">
  <div class="verdict-label">1-Year CVaR (95% confidence)</div>
  <div class="verdict-number">{mc['cvar']:.1%}</div>
  <div class="verdict-sentence">{verdict}</div>
</div>
""", unsafe_allow_html=True)

# ---- Supporting context, only if you want it ----
with st.expander("See the full risk breakdown"):
    st.caption(f"Universe ({len(loaded)}): {', '.join(loaded)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median 1-year return", f"{mc['median_return']:+.1%}")
    c2.metric("Probability of loss", f"{mc['prob_loss']:.1%}")
    c3.metric("Worst simulated year", f"{mc['worst_case']:+.1%}")

    # --- Risk-adjusted performance (Sharpe vs a real risk-free rate) ---
    rf = load_risk_free_rate()
    ann_ret = float(port_returns.mean()) * 252
    ann_vol = float(port_returns.std()) * np.sqrt(252)
    sharpe = sharpe_ratio(port_returns, rf if rf is not None else 0.0)
    s1, s2, s3 = st.columns(3)
    s1.metric("Sharpe ratio", f"{sharpe:.2f}")
    s2.metric("Annualized return", f"{ann_ret:+.1%}")
    s3.metric("Annualized volatility", f"{ann_vol:.1%}")
    rf_txt = (f"{rf:.2%} (13-week T-bill, ^IRX)" if rf is not None
              else "unavailable — Sharpe computed against 0%")
    st.caption(
        f"Sharpe = (annualized return − risk-free) / annualized volatility, on "
        f"the real (unshocked) portfolio. Risk-free rate: {rf_txt}."
    )

    # --- Risk-contribution decomposition (where the risk actually lives) ---
    st.markdown("###### Risk contribution by asset")
    rc = risk_contributions(weights, cov)
    st.bar_chart(
        pd.DataFrame({"dollar weight": rc["weight"], "risk share": rc["risk_pct"]}),
        horizontal=True,
    )
    top = rc["risk_pct"].idxmax()
    st.caption(
        f"Share of total portfolio volatility per asset. {top} contributes the most "
        f"risk ({rc.loc[top, 'risk_pct']:.0%}). Equal dollar weight ≠ equal risk — "
        "switch Allocation to Risk parity to flatten these bars."
    )

    st.markdown("###### Correlation matrix")
    corr = correlation_matrix(shocked_returns)

    def beige_scale(val):
        # higher correlation -> deeper warm gray, no matplotlib needed
        shade = int(245 - max(0.0, min(1.0, val)) * 90)
        text = "#4A4640" if val < 0.7 else "#FFFFFF"
        return f"background-color: rgb({shade},{shade-6},{shade-14}); color: {text};"

    st.dataframe(corr.style.format("{:.2f}").map(beige_scale))

    st.markdown("###### Distribution of simulated 1-year outcomes")
    st.bar_chart(np.histogram(mc["total_returns"], bins=40)[0])
    if mc.get("engine") == "jump-diffusion":
        jp = mc["jump_params"]
        st.caption(
            f"Merton jump-diffusion: the engine flagged **{jp['n_jumps']} jump days** "
            f"in {jp['n_days']} (moves beyond {jp['k']:.0f}σ), implying "
            f"**~{jp['lambda_daily'] * 252:.1f} jumps/year** on a diffusion vol of "
            f"{jp['sigma_d'] * np.sqrt(252):.0%}. Poisson jumps let the tail run "
            "deeper than any single historical day — a fatter, more honest crash."
        )

    # --- VaR methods + backtest (validates the model, not just reports it) ---
    st.markdown("###### Value at Risk — methods & backtest")
    hist_var = float(-np.percentile(port_returns, 5))
    bt = var_backtest(port_returns)
    v1, v2, v3 = st.columns(3)
    v1.metric("Historical VaR (95%)", f"{hist_var:.2%}")
    v2.metric("Parametric VaR (95%)", f"{parametric_var(port_returns):.2%}")
    v3.metric("VaR breaches", f"{bt['breaches']} / {bt['expected_breaches']:.0f} exp.")
    verdict_word = "passes" if bt["passed"] else "fails"
    st.caption(
        f"Daily VaR backtest {verdict_word} the Kupiec test "
        f"(LR = {bt['kupiec_lr']}, 95% critical = 3.84): the model's breach rate "
        f"of {bt['observed_rate']:.1%} is statistically consistent with the 5% it claims."
    )

    # --- Named factor exposures ---
    st.markdown("###### Factor exposures")
    try:
        fx = factor_exposures(port_returns)
        st.bar_chart(pd.Series(fx["betas"], name="beta"), horizontal=True)
        st.caption(
            f"Market beta {fx['betas']['Market']:+.2f} · "
            f"R-squared {fx['r_squared']:.0%} · "
            f"annualized alpha {fx['alpha_annual']:+.1%}. "
            "Size/Value/Momentum are tilts vs. broad market (ETF-proxy factors)."
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Factor exposures unavailable: {exc}")

source_txt = f"the {scenario_label} window" if scenario_label else \
    "2 years of daily historical returns"
engine_txt = (
    "Merton jump-diffusion (Poisson jumps on Gaussian diffusion), calibrated to"
    if use_jd else "bootstrap Monte Carlo, resampled from"
)
st.caption(
    f"Methodology: 10,000-path {engine_txt} {source_txt}, over a 252-day horizon. "
    f"{alloc_label.capitalize()} allocation{lev_txt}."
)

# ---- Liquidity: how fast could you actually get out? ----
def _fmt_days(d: float) -> str:
    """Human days: infinity for no-volume names, <1 day rounded sensibly."""
    if not np.isfinite(d):
        return "∞"
    if d < 0.1:
        return "<0.1d"
    if d < 10:
        return f"{d:.1f}d"
    return f"{d:.0f}d"


with st.expander("Liquidity — how fast could you exit?"):
    lc1, lc2 = st.columns(2)
    book = lc1.number_input(
        "Portfolio size ($)", min_value=10_000, max_value=5_000_000_000,
        value=1_000_000, step=100_000,
        help="Total dollars invested. Position sizes — and so the days to unwind "
             "them — scale from this.")
    participation = lc2.slider(
        "Max daily participation (% of ADV)", 5, 50, 20, step=5,
        help="How much of a name's average daily dollar volume you'll be before "
             "your own trading moves the price. Risk desks use ~10–20%.") / 100

    try:
        adv = load_adv(tuple(tickers)).reindex(loaded).fillna(0.0)
        dtl = days_to_liquidate(weights, adv, book_value=book,
                                participation_rate=participation)
        prof = liquidity_profile(dtl)

        m1, m2, m3 = st.columns(3)
        m1.metric("Full-exit horizon", _fmt_days(prof["full_exit_days"]))
        m2.metric("Exitable in 1 day", f"{prof['pct_exitable_1day']:.0%}")
        m3.metric("Avg position horizon", _fmt_days(prof["weighted_avg_days"]))

        chart_days = dtl["days"].replace([np.inf, -np.inf], np.nan).dropna()
        if not chart_days.empty:
            st.bar_chart(chart_days.rename("days to liquidate"), horizontal=True)

        caption = (
            f"Days to unwind a **${book:,.0f}** {alloc_label} book at "
            f"{participation:.0%} of each name's average daily dollar volume "
            f"(recent 3-month lookback). "
        )
        if prof["least_liquid"] is not None:
            caption += (
                f"**{prof['least_liquid']}** is the bottleneck at "
                f"{_fmt_days(prof['full_exit_days'])} to fully exit. "
            )
        if prof["no_volume"]:
            caption += (
                f"*No volume feed for {', '.join(prof['no_volume'])} "
                "(e.g. FX/futures on Yahoo) — excluded, not estimated.*"
            )
        st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Liquidity data unavailable: {exc}")

# ---- Data source & provenance (traceability) ----
with st.expander("Data source & provenance"):
    prov = provenance(tickers)
    if prov:
        st.markdown(
            f"- **Source:** {prov['source']}\n"
            f"- **Fetched (UTC):** {prov['fetched_at_utc']}\n"
            f"- **Symbols:** {', '.join(prov['symbols'])}\n"
            f"- **Coverage:** {prov['start']} → {prov['end']} "
            f"({prov['rows']} trading days)\n"
            f"- **Library:** yfinance {prov['yfinance_version']}"
        )
        st.caption(
            "Prices are live end-of-day adjusted closes, pulled on demand from "
            "Yahoo Finance and cached for one hour. Every figure above is computed "
            "from this source by the engine's own code — no value originates from a "
            "language model. Use Refresh to re-pull and update this timestamp."
        )
    else:
        st.caption("Provenance record appears after the first live fetch.")
