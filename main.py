"""
Portfolio Risk Engine — Streamlit entry point.

Design philosophy:
A risk desk doesn't hand a PM eight charts and say "figure it out." It leads
with one verdict and one number. Everything else is detail you open on demand.
That's the layout here: headline first, stress controls second, deep detail
collapsed in an expander.
"""

import numpy as np
import streamlit as st

from src.ingestion import fetch_prices, get_returns, TICKERS
from src.analytics import covariance_matrix, correlation_matrix
from src.risk import portfolio_daily_returns, monte_carlo

st.set_page_config(page_title="Portfolio Risk Engine", layout="centered")

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
# This block only handles the custom verdict card, header crest, and panels.
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

/* Hero verdict card — cream on the darker taupe page so it lifts off the bg */
.verdict-box { background: #F4F1EA; border: 1px solid #BFB8A9; border-radius: 6px;
               padding: 28px 32px; margin-bottom: 24px;
               box-shadow: 0 2px 6px rgba(63,59,53,0.10); }
.verdict-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
                  letter-spacing: 0.12em; text-transform: uppercase; color: #9A8E7C; }
.verdict-number { font-size: 48px; color: #3F3B35; margin: 4px 0; }
.verdict-sentence { font-size: 17px; color: #54504A; line-height: 1.5; }

/* Stress-test panel — distinct cream block with a bronze top accent */
.stress-panel { background: #E5E0D6; border: 1px solid #C4BDAE;
                border-top: 3px solid #9A7B4F; border-radius: 6px;
                padding: 14px 22px 6px; margin-bottom: 10px; }

/* Refined slider — slimmer track, restrained bronze, no betting-app look */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #C4BDAE !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #9A7B4F !important; border: 2px solid #F4F1EA !important;
    box-shadow: 0 1px 3px rgba(63,59,53,0.25) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #3F3B35 !important; font-family: 'Helvetica Neue', sans-serif !important;
    font-size: 12px !important; }

/* Expander — give the breakdown its own readable cream surface */
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
st.caption(f"Universe: {', '.join(TICKERS)}  ·  Equal-weight allocation")

# ---- Load data once per session ----
prices = fetch_prices()
returns = get_returns(prices)
weights = np.ones(len(TICKERS)) / len(TICKERS)

# ---- Stress controls (drives a REAL Monte Carlo re-run, not fake math) ----
st.markdown('<div class="stress-panel">', unsafe_allow_html=True)
st.markdown("##### Stress test")
col1, col2 = st.columns(2)
with col1:
    drawdown_shock = st.slider(
        "Market drawdown shock", -50, 0, 0, step=5,
        help="Shifts every historical daily return down by this amount before resampling."
    )
with col2:
    vol_shock = st.slider(
        "Volatility shock", 0, 100, 0, step=10,
        help="Scales the spread of daily returns to simulate a higher-volatility regime."
    )

st.markdown('</div>', unsafe_allow_html=True)

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
    f"In the worst 5% of simulated years, this portfolio loses an average of "
    f"**{mc['cvar']:.1%}**."
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
        bg = f"rgb({shade},{shade-6},{shade-14})"
        return f"background-color: {bg}; color: {text};"

    st.dataframe(corr.style.format("{:.2f}").map(beige_scale))

    st.markdown("###### Distribution of simulated 1-year outcomes")
    st.bar_chart(
        np.histogram(mc["total_returns"], bins=40)[0],
    )

st.caption(
    "Methodology: 10,000-path bootstrap Monte Carlo over 252-day horizon, "
    "resampled from 2 years of daily historical returns."
)
