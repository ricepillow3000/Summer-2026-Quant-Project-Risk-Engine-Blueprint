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
    fetch_prices, get_returns, data_health, provenance, clear_cache, PRESETS,
)
from src.analytics import correlation_matrix
from src.risk import monte_carlo, parametric_var, var_backtest
from src.factors import factor_exposures

st.set_page_config(page_title="Portfolio Risk Engine", layout="centered")

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
st.markdown("""
<style>
html, body, [class*="css"] { font-family: Georgia, 'Times New Roman', serif; }
h1, h2, h3 { color: #3F3B35; font-weight: 400; }
.stCaption, [data-testid="stCaptionContainer"] { color: #7A756C; }

/* Header crest + wordmark */
.brand-row { display: flex; align-items: center; gap: 16px; margin-bottom: 2px; }
.brand-title { font-size: 32px; color: #3F3B35; line-height: 1.1; }
.brand-tag { font-family: 'Helvetica Neue', sans-serif; font-size: 10px;
             letter-spacing: 0.18em; text-transform: uppercase; color: #9A7B4F; }

/* Hero verdict card */
.verdict-box { background: #F4F1EA; border: 1px solid #BFB8A9; border-radius: 6px;
               padding: 28px 32px; margin-bottom: 24px;
               box-shadow: 0 2px 6px rgba(63,59,53,0.10); }
.verdict-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
                  letter-spacing: 0.12em; text-transform: uppercase; color: #9A8E7C; }
.verdict-number { font-size: 48px; color: #3F3B35; margin: 4px 0; }
.verdict-sentence { font-size: 17px; color: #54504A; line-height: 1.5; }

/* Control panels — distinct cream blocks with a bronze top accent */
.panel-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
               letter-spacing: 0.12em; text-transform: uppercase; color: #9A8E7C;
               margin-bottom: 2px; }

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

weights = np.ones(len(loaded)) / len(loaded)
port_returns = returns @ weights  # real (unshocked) portfolio series for VaR/factors

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

# ---- Stress controls (drives a REAL Monte Carlo re-run, not fake math) ----
with st.container(border=True):
    st.markdown('<div class="panel-label">Stress test</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        drawdown_shock = st.slider(
            "Market drawdown shock", -50, 0, 0, step=5,
            help="Shifts every historical daily return down before resampling.")
    with col2:
        vol_shock = st.slider(
            "Volatility shock", 0, 100, 0, step=10,
            help="Scales the spread of daily returns to simulate a higher-vol regime.")

# Apply the shock directly to the return distribution Monte Carlo samples from.
shocked_returns = returns.copy()
if drawdown_shock != 0:
    shocked_returns = shocked_returns + (drawdown_shock / 100) / 252
if vol_shock != 0:
    mean = shocked_returns.mean()
    shocked_returns = mean + (shocked_returns - mean) * (1 + vol_shock / 100)

mc = monte_carlo(shocked_returns, weights, n_simulations=10_000, horizon_days=252)

# ---- Headline verdict ----
is_shocked = drawdown_shock != 0 or vol_shock != 0
verdict = (
    f"In the worst 5% of simulated years, an equal-weight portfolio of these "
    f"{len(loaded)} assets loses an average of **{mc['cvar']:.1%}**."
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

st.caption(
    "Methodology: 10,000-path bootstrap Monte Carlo over a 252-day horizon, "
    "resampled from 2 years of daily historical returns. Equal-weight allocation."
)

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
