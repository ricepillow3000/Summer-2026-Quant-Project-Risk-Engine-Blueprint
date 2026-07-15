"""
Meleona - Streamlit entry point.

Design philosophy:
A risk desk doesn't hand a PM eight charts and say "figure it out." It leads
with one verdict and one number. Everything else is detail you open on demand.

Phase V:
The universe is now chosen by the viewer, not hard-coded. Anyone can load a
preset basket (equities, sector ETFs, FX, futures) or type their own symbols,
so the engine speaks to any audience - not just one watchlist.
"""

import base64
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from src.ingestion import (
    fetch_prices, get_returns, data_health, provenance, clear_cache,
    average_dollar_volume, fetch_risk_free_rate, PRESETS,
)
from src.analytics import correlation_matrix, covariance_matrix
from src.risk import (
    monte_carlo, jump_diffusion_mc, parametric_var, var_backtest, sharpe_ratio,
    var, cvar, portfolio_daily_returns,
)
from src.comovement import (
    correlation_from_cov, rolling_correlation, most_correlated_pair,
    defensive_shift, least_correlated_to_pair,
)
from src.factors import factor_exposures
from src.strategies import risk_contributions, risk_parity_weights, vol_target
from src.hedge import min_variance_pair, rank_hedges
from src.covariance import estimate_covariance
from src.eigenrisk import eigen_factors, marcenko_pastur_bounds, pc1_exposure
from src.scenarios import HISTORICAL_REGIMES, replay_returns
from src.liquidity import (days_to_liquidate, liquidity_profile,
                           liquidity_adjusted_cvar)
from src.grit import grit_scores, MIN_HISTORY_DAYS
from src.security_master import security_master
from src.data_quality import validate_prices
from src.regimes import (
    rolling_windows, wasserstein_kmeans, vol_ordered_labels,
    regime_stats, transition_matrix,
)
from src.signals import (
    momentum_signal, forward_returns, daily_ic, ic_summary,
    fundamental_law_ir, effective_breadth,
)
from src.conviction import (
    load_conviction, AI_CAPEX_BASKET, RECOVERY_HORIZON_DAYS,
)

st.set_page_config(page_title="Meleona", layout="wide")

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
st.markdown("""
<style>
/* ============================================================
   "THE TEARSHEET" - private-bank editorial design language.
   Doctrine: sharp edges (no rounded pills), hairline bronze rules,
   extreme type contrast (huge serif numerals vs tiny tracked labels),
   numbered ruled sections. Numbers are king; craft gives it soul.
   ============================================================ */
html, body, [class*="css"] { font-family: Georgia, 'Times New Roman', serif; }
h1, h2, h3 { color: #3F3B35; font-weight: 400; letter-spacing: -0.01em; }
/* Sharpen the whole app - kill Streamlit's default rounded corners */
[data-testid="stExpander"], [data-baseweb="tab"], .stButton>button,
[data-testid="stMetric"], div[data-baseweb="select"]>div { border-radius: 0 !important; }
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] div {
    font-size: 13.5px !important; color: #524E47 !important; line-height: 1.5 !important; }
[data-testid="stMetricLabel"] p { font-size: 11px !important; color: #8A8172 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: Georgia, serif !important;
    color: #3F3B35 !important; letter-spacing: -0.01em; }

/* Header crest + wordmark */
.brand-row { display: flex; align-items: center; gap: 16px; margin-bottom: 2px; }
.brand-title { font-size: 32px; color: #3F3B35; line-height: 1.1; letter-spacing: -0.015em; }
.brand-tag { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
             letter-spacing: 0.2em; text-transform: uppercase; color: #8A6E45; }

/* HERO VERDICT - the editorial centerpiece. No card: a stat framed by
   bronze hairlines, the one number that owns the page. */
.verdict-box { background: transparent; border: none;
    border-top: 2px solid #9A7B4F; border-bottom: 1px solid #C4BDAE;
    padding: 26px 4px 30px; margin: 12px 0 30px; }
.verdict-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.22em; text-transform: uppercase; color: #9A7B4F; }
.verdict-number { font-size: 96px; color: #3F3B35; margin: 8px 0 4px;
    line-height: 1; letter-spacing: -0.035em; font-weight: 400; }
.verdict-sentence { font-size: 18px; color: #54504A; line-height: 1.6; max-width: 580px; }

/* Numbered section eyebrow - editorial ledger markers (01 - UNIVERSE) */
.sec-mark { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.24em; text-transform: uppercase; color: #9A7B4F;
    border-top: 1px solid #C4BDAE; padding-top: 14px; margin: 26px 0 12px;
    display: flex; align-items: baseline; gap: 12px; }
.sec-mark b { color: #B7A98E; font-weight: 400; }
.panel-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.16em; text-transform: uppercase; color: #8A7E6A;
    margin-bottom: 4px; }

/* Slider - squared thumb, hairline track */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #C4BDAE !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #9A7B4F !important; border-radius: 0 !important;
    border: 2px solid #F4F1EA !important;
    box-shadow: 0 1px 3px rgba(63,59,53,0.25) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #3F3B35 !important; font-family: 'Helvetica Neue', sans-serif !important;
    font-size: 12px !important; }

/* Expander - flat cream, sharp, hairline */
[data-testid="stExpander"] { border: 1px solid #C4BDAE !important;
    background: #ECE7DD !important; }

/* ---- Presentation flow: hero, showcase, CTAs, scroll reveal ---- */
html { scroll-behavior: smooth; }
@keyframes meleona-rise { from { opacity: 0; transform: translateY(24px); }
                          to   { opacity: 1; transform: translateY(0); } }
.reveal { animation: meleona-rise linear both;
          animation-timeline: view(); animation-range: entry 0% cover 30%; }

/* WIDE COURT - the whole page works now, capped for taste. Layout is
   `wide`; this rules the court width and the gutter rhythm. */
.block-container { max-width: 1600px !important;
    padding-left: 56px !important; padding-right: 56px !important;
    padding-top: 2.4rem !important; }

/* The page scrolls inside Streamlit's own <section>, NOT <html> - smooth
   behavior must live on the real scroller or anchors hard-teleport. The
   glide script (end of page) drives an eased scroll; this is the fallback. */
[data-testid="stAppViewContainer"] section, section.main,
[data-testid="stMain"] { scroll-behavior: smooth; }

/* Presentation arrivals - CTA anchors land like a slide change: the
   target section rises into place under a smooth scroll. */
.hero-section, .showcase-section, .showcase-row, .engine-heading,
#engine { scroll-margin-top: 28px; }
@keyframes section-arrive { from { opacity: .25; transform: translateY(26px); }
                            to   { opacity: 1; transform: none; } }
#grit-showcase:target, #conviction:target, #engine:target {
    animation: section-arrive .85s cubic-bezier(.16,1,.3,1); }

/* HERO - the great hall, now a two-column court: the pitch on the
   left, a 2x2 deck of engine-fact tiles on the right (reference:
   dashboard stat cards), watermark crest behind. 58vh, not 88. */
.hero-section { min-height: 58vh; display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(300px, 1fr);
    gap: 56px; align-items: center; text-align: left;
    padding: 40px 8px 44px; border-bottom: 1px solid #C4BDAE;
    position: relative; overflow: hidden; }
.hero-left { display: flex; flex-direction: column; align-items: flex-start;
    gap: 16px; min-width: 0; }
@media (max-width: 1100px) {
  .hero-section { grid-template-columns: 1fr; }
  .showcase-row { grid-template-columns: 1fr !important; } }
/* The stat deck lives in Casper's wash with no plate - but the TILES stay
   fully solid (John: fading them read unprofessional). The background does
   the fading; the instruments never do. */
.hero-stats { position: relative; }
.hero-crest { width: 132px; height: 132px; padding: 20px;
    border: 1px solid #C4BDAE; background: rgba(154,123,79,0.05);
    box-shadow: 0 0 0 1px rgba(154,123,79,.12),
                0 24px 60px -34px rgba(63,59,53,.55); }
.hero-crest svg { width: 100%; height: 100%; }
.hero-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.32em; text-transform: uppercase; color: #9A7B4F; }
.hero-title { font-size: clamp(44px, 10.5vw, 96px) !important; color: #3F3B35;
    line-height: 0.98 !important; margin: 2px 0; letter-spacing: -0.035em; }
.hero-sub { font-size: 21px; color: #54504A; max-width: 640px; line-height: 1.6; }
.hero-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    position: relative; z-index: 1; }
.hstat { border: 1px solid #C4BDAE; border-top: 2px solid #9A7B4F;
    background: #F1EDE5; padding: 20px 22px 16px;
    box-shadow: 0 1px 2px rgba(63,59,53,.05), 0 8px 24px -18px rgba(63,59,53,.35);
    transition: transform .3s cubic-bezier(.16,1,.3,1), border-color .3s ease,
                box-shadow .3s ease; }
.hstat:hover { transform: translateY(-2px); border-color: #9A7B4F;
    box-shadow: 0 2px 4px rgba(63,59,53,.06), 0 16px 34px -20px rgba(63,59,53,.45); }
.hstat .n { font-size: 38px; color: #3F3B35; line-height: 1.05;
    letter-spacing: -0.02em; }
.hstat .l { font-family: 'Helvetica Neue', sans-serif; font-size: 10px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #9A7B4F;
    margin-top: 6px; }

/* Twin showcases share one row - half the scroll, none of the clutter */
.showcase-row { display: grid; grid-template-columns: 1fr 1fr; gap: 64px;
    padding: 48px 8px 20px; }
.showcase-row .showcase-section { padding: 0; }

/* CTA - sharp charcoal slab, bronze on hover, alive to the touch */
.cta-btn { display: inline-block; margin-top: 14px; padding: 15px 34px;
    background: #3F3B35; color: #F4F1EA !important; text-decoration: none !important;
    border-radius: 0; font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.16em; text-transform: uppercase;
    transition: background .25s ease, letter-spacing .25s ease,
                transform .25s cubic-bezier(.16,1,.3,1), box-shadow .25s ease; }
.cta-btn:hover { background: #9A7B4F; letter-spacing: 0.2em;
    transform: translateY(-2px);
    box-shadow: 0 16px 30px -18px rgba(63,59,53,.6); }
.cta-btn:active { transform: translateY(0) scale(.985); }

.showcase-section { padding: 56px 8px 44px; text-align: left; display: flex;
    flex-direction: column; align-items: flex-start; gap: 18px; }
.showcase-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.28em; text-transform: uppercase; color: #9A7B4F; }
.showcase-title { font-size: 52px; color: #3F3B35; margin: 0; font-weight: 400;
    letter-spacing: -0.025em; line-height: 1.05; }
.showcase-body { font-size: 16px; color: #54504A; max-width: 620px; line-height: 1.6; }

/* Pillars - no cards: ledger columns divided by bronze hairlines */
.pillar-row { display: flex; gap: 0; flex-wrap: wrap; margin-top: 16px;
    border-top: 1px solid #C4BDAE; }
.pillar-card { background: transparent; border: none;
    border-left: 1px solid #C4BDAE; padding: 26px 30px 14px; width: 220px;
    text-align: left; transition: border-color 0.25s ease; }
.pillar-card:first-child { border-left: none; padding-left: 4px; }
.pillar-card:hover { border-left-color: #9A7B4F; }
.pillar-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.16em; text-transform: uppercase; color: #9A7B4F; margin-bottom: 6px; }
.pillar-desc { font-size: 13.5px; color: #524E47; line-height: 1.5; }

/* Apple-fine hairline: bronze breathes in the centre, fades at the edges */
.section-divider { border: none; height: 1px; margin: 8px 0 30px;
    background: linear-gradient(90deg, transparent,
        rgba(154,123,79,.55) 18%, rgba(154,123,79,.55) 82%, transparent); }

/* ENGRAVINGS - crest fragments carved into the stone at low relief.
   Pure ornament: pointer-events none, behind everything. */
[data-testid="stColumn"] { position: relative; }   /* engrave anchor */
.engrave { position: absolute; pointer-events: none; z-index: 0;
    opacity: .075; }
.engrave svg { width: 100%; height: 100%; }
.engrave.scale { left: -70px; top: 34px; width: 340px; height: 340px;
    transform: rotate(-6deg); }
.engrave.line { right: -46px; bottom: -34px; width: 460px; height: 400px;
    transform: rotate(-8deg); }

/* Tables read as cut slabs too */
[data-testid="stDataFrame"] { border: 1px solid #D4CDBF;
    border-top: 2px solid #9A7B4F; }

/* Keyboard focus carries the same bronze - prestige includes a11y */
a.cta-btn:focus-visible, .stButton>button:focus-visible,
[data-baseweb="tab"]:focus-visible {
    outline: 2px solid #9A7B4F; outline-offset: 2px; }
.engine-heading { text-align: left; padding: 4px 0 22px; }

/* ============================================================
   TABS - "the gatehouse". Each label is a stone lintel: generous
   breathing room, wide tracking, a bronze rule that slides in.
   The previous rule tracked the type but gave it NO horizontal
   padding, so uppercase labels collided into one another.
   ============================================================ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid #C4BDAE;
    overflow-x: auto; scrollbar-width: none; }
[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 11.5px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #8A7E6A; white-space: nowrap;             /* never wrap a label */
    padding: 15px 26px !important;                    /* the missing air */
    background: transparent; position: relative;
    transition: color .28s ease, background .28s ease; }
[data-testid="stTabs"] [data-baseweb="tab"] * { letter-spacing: inherit; }
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #3F3B35; background: rgba(154,123,79,0.06); }
/* bronze rule grows from the centre - no jump, no flash */
[data-testid="stTabs"] [data-baseweb="tab"]::after {
    content: ''; position: absolute; left: 50%; right: 50%; bottom: -1px;
    height: 2px; background: #9A7B4F;
    transition: left .3s cubic-bezier(.16,1,.3,1), right .3s cubic-bezier(.16,1,.3,1); }
[data-testid="stTabs"] [aria-selected="true"] { color: #3F3B35 !important; }
[data-testid="stTabs"] [aria-selected="true"]::after { left: 0; right: 0; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }

/* Panel swap: content settles in rather than snapping */
@keyframes panel-settle {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); } }
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    animation: panel-settle .42s cubic-bezier(.16,1,.3,1) both;
    padding-top: 22px; }

/* ============================================================
   THE KEEP - stone-slab surfaces. Sharp corners, hairline mortar,
   one soft shadow so a panel reads as a cut block, not a web card.
   Same palette; only the structure is new.
   ============================================================ */
.slab { background: #F1EDE5; border: 1px solid #D4CDBF;
    border-top: 2px solid #9A7B4F; padding: 22px 24px 18px; height: 100%;
    box-shadow: 0 1px 2px rgba(63,59,53,.05), 0 8px 24px -18px rgba(63,59,53,.35);
    transition: transform .3s cubic-bezier(.16,1,.3,1),
                box-shadow .3s ease, border-color .3s ease; }
.slab:hover { transform: translateY(-2px); border-color: #9A7B4F;
    box-shadow: 0 2px 4px rgba(63,59,53,.06), 0 16px 34px -20px rgba(63,59,53,.45); }
.slab-label { font-family: 'Helvetica Neue', sans-serif; font-size: 10.5px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #9A7B4F;
    margin-bottom: 10px; }
.slab-num { font-size: 40px; color: #3F3B35; line-height: 1;
    letter-spacing: -0.03em; margin-bottom: 6px; }
.slab-note { font-size: 13px; color: #6B6459; line-height: 1.55; }

/* Plain-language read-out under every chart: what am I looking at? */
.read-me { border-left: 2px solid #9A7B4F; background: rgba(154,123,79,0.05);
    padding: 12px 16px; margin: 10px 0 4px; font-size: 14px; color: #524E47;
    line-height: 1.6; }
.read-me b { color: #3F3B35; font-weight: 400; }

/* Ruled panel header inside a tab - the lintel over each block */
.panel-head { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    border-top: 1px solid #C4BDAE; padding-top: 13px; margin: 34px 0 14px;
    position: relative; }
/* A bronze accent draws itself across the lintel as the section scrolls
   into view - one pass, scroll-driven. Where view-timeline is unsupported
   or motion is reduced it stays scaleX(0) (invisible) and the 1px border
   above is the base, so there is no regression. */
.panel-head::before { content: ''; position: absolute; top: -1px; left: 0;
    height: 2px; width: 100%; transform: scaleX(0); transform-origin: left;
    background: linear-gradient(90deg, #9A7B4F, #C8A86E 55%, transparent); }
@supports (animation-timeline: view()) {
  .panel-head::before { animation: lintel-draw linear both;
      animation-timeline: view(); animation-range: entry 0% cover 20%; }
  @keyframes lintel-draw { to { transform: scaleX(1); } }
}
.panel-head .t { font-family: 'Helvetica Neue', sans-serif; font-size: 11.5px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #3F3B35;
    white-space: nowrap; }
.panel-head .s { font-size: 13px; color: #8A8172; line-height: 1.5; }

/* Charts glide in with the panel instead of popping */
[data-testid="stPlotlyChart"], [data-testid="stIFrame"] {
    animation: panel-settle .5s cubic-bezier(.16,1,.3,1) both;
    animation-delay: .06s; }

/* Metric slabs pick up the same masonry - and a glossy specular sheen that
   sweeps across on hover (three-finish "trophy" role). Hover-only, one pass;
   the prefers-reduced-motion guard nulls the transition to a static tile. */
[data-testid="stMetric"] { background: #F1EDE5; border: 1px solid #D4CDBF;
    border-top: 2px solid #9A7B4F; padding: 16px 18px 12px;
    position: relative; overflow: hidden;
    transition: transform .3s cubic-bezier(.16,1,.3,1), border-color .3s ease,
                box-shadow .3s ease; }
[data-testid="stMetric"]::before { content: ''; position: absolute; top: 0;
    left: -70%; width: 45%; height: 100%; pointer-events: none;
    background: linear-gradient(100deg, transparent,
        rgba(200,168,110,.22), transparent);
    transform: skewX(-18deg);
    transition: left .6s cubic-bezier(.16,1,.3,1); }
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: #9A7B4F;
    box-shadow: 0 14px 28px -18px rgba(63,59,53,.5); }
[data-testid="stMetric"]:hover::before { left: 125%; }

/* The many fold-away expanders feel touchable: bronze border warms on hover */
[data-testid="stExpander"] { transition: border-color .25s ease,
    box-shadow .25s ease; }
[data-testid="stExpander"]:hover { border-color: #9A7B4F !important;
    box-shadow: 0 8px 20px -16px rgba(63,59,53,.4); }

/* Buttons: cut stone, bronze on press, alive to the touch */
.stButton>button { font-family: 'Helvetica Neue', sans-serif; font-size: 11.5px;
    letter-spacing: 0.16em; text-transform: uppercase; border: 1px solid #C4BDAE;
    background: #F1EDE5; color: #3F3B35;
    transition: background .25s ease, border-color .25s ease,
                letter-spacing .25s ease, transform .25s cubic-bezier(.16,1,.3,1),
                box-shadow .25s ease; }
.stButton>button:hover { background: #3F3B35; color: #F4F1EA;
    border-color: #3F3B35; letter-spacing: 0.2em; transform: translateY(-1px);
    box-shadow: 0 10px 22px -14px rgba(63,59,53,.55); }
.stButton>button:active { transform: translateY(0) scale(.985); }

/* Inputs breathe too: bronze focus, soft glow - nothing snaps */
div[data-baseweb="select"] > div, .stNumberInput input,
[data-testid="stTextInput"] input, .stMultiSelect [data-baseweb="select"] > div {
    transition: border-color .25s ease, box-shadow .25s ease; }
div[data-baseweb="select"] > div:hover { border-color: #9A7B4F !important; }
.stNumberInput input:focus, [data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 0 1px #9A7B4F !important; }

/* ============================================================
   SKELETON LOADING - the scaffold shows before the stone.
   1) Boot veil: a full-page skeleton of the hero (crest block,
      title bars, a chart slab) that shimmers, then lifts.
   2) Scroll skeletons: charts entering the viewport wear a
      shimmer plate that dissolves as they arrive.
   ============================================================ */
@keyframes sk-sheen { from { background-position: 200% 0; }
                      to   { background-position: -200% 0; } }
.sk { background: linear-gradient(90deg, #E2DCD0 25%, #EFEAE0 45%, #E2DCD0 65%);
    background-size: 200% 100%; animation: sk-sheen 1.15s linear infinite; }
#boot-skel { position: fixed; inset: 0; z-index: 999; background: #EDE9E3;
    display: flex; flex-direction: column; gap: 18px; padding: 14vh 10vw;
    pointer-events: none; animation: boot-off .55s ease 2.1s forwards; }
@keyframes boot-off { to { opacity: 0; visibility: hidden; } }
#boot-skel .sk.crest { width: 132px; height: 132px; }
#boot-skel .sk.title { height: 64px; width: 42%; }
#boot-skel .sk.line  { height: 15px; width: 58%; }
#boot-skel .sk.line.short { width: 32%; }
#boot-skel .sk.chart { height: 200px; width: 100%; margin-top: 24px; }

@supports (animation-timeline: view()) {
  [data-testid="stPlotlyChart"] { position: relative; }
  [data-testid="stPlotlyChart"]::before { content: ''; position: absolute;
      inset: 0; z-index: 2; pointer-events: none; opacity: 0;
      background: linear-gradient(90deg, #E2DCD0 25%, #EFEAE0 45%, #E2DCD0 65%);
      background-size: 200% 100%;
      animation: sk-sheen 1.15s linear infinite, sk-dissolve linear both;
      animation-timeline: auto, view();
      animation-range: normal, entry 0% cover 40%; }
  @keyframes sk-dissolve { from { opacity: 1; } 80% { opacity: 1; }
                           to { opacity: 0; visibility: hidden; } }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
    #boot-skel { display: none !important; } }
</style>
""", unsafe_allow_html=True)

# ---- GLOSS LAYER (2026-07-09) ----
# Institutional sheen over the matte editorial base: specular highlights,
# layered depth, and a few one-pass arrival glints. Same Citadel palette -
# no new hues, no rounded pills. Every effect degrades to the matte base
# under prefers-reduced-motion (guard at the end).
st.markdown("""
<style>
/* Verdict numeral - brushed-bronze-to-charcoal specular fill with a single
   sheen sweep on arrival. @supports-guarded so it stays solid charcoal where
   background-clip:text is unsupported (the .verdict-number base keeps color). */
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  .verdict-number {
    background: linear-gradient(100deg,
        #3F3B35 0%, #46413A 30%, #C8A86E 50%, #46413A 70%, #3F3B35 100%);
    background-size: 240% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 1px 0 rgba(255,255,255,.28);
    animation: verdict-sheen 2.6s cubic-bezier(.4,0,.2,1) .25s 1 both; }
}
@keyframes verdict-sheen { 0% { background-position: 160% 0; }
                           62%, 100% { background-position: 0 0; } }

/* Verdict frame - faint top glass + one light-glint pass across it on load. */
.verdict-box { position: relative; overflow: hidden;
    background: linear-gradient(180deg, rgba(255,253,248,.55), rgba(255,253,248,0) 62%); }
.verdict-box::after { content: ""; position: absolute; top: 0; left: -45%;
    width: 45%; height: 100%; pointer-events: none;
    background: linear-gradient(100deg, transparent,
        rgba(255,255,255,.5) 50%, transparent);
    transform: skewX(-18deg); animation: gloss-glint 2.4s ease-out .2s 1 both; }
@keyframes gloss-glint { from { left: -45%; } to { left: 150%; } }

/* Hero stat tiles - frosted glass with an inner top highlight; the lift on
   hover already exists, now it reads as a pane of glass catching light. */
.hstat { background: linear-gradient(157deg, #F6F2EA 0%, #ECE6DA 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.7),
        0 1px 2px rgba(63,59,53,.05), 0 8px 24px -18px rgba(63,59,53,.35); }
.hstat:hover { box-shadow: inset 0 1px 0 rgba(255,255,255,.85),
        0 2px 4px rgba(63,59,53,.06), 0 18px 38px -20px rgba(63,59,53,.5); }
/* Specular sheen sweeps the hero tiles on hover - same trophy finish as
   the metric slabs, so the brand's front door feels alive. Children sit
   above the sweep (z-index 1) so the gradient numbers stay crisp. */
.hstat { position: relative; overflow: hidden; }
.hstat > div { position: relative; z-index: 1; }
.hstat::before { content: ''; position: absolute; top: 0; left: -70%;
    width: 45%; height: 100%; pointer-events: none; z-index: 0;
    background: linear-gradient(100deg, transparent,
        rgba(200,168,110,.25), transparent);
    transform: skewX(-18deg);
    transition: left .6s cubic-bezier(.16,1,.3,1); }
.hstat:hover::before { left: 125%; }
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  .hstat .n { background: linear-gradient(120deg, #3F3B35, #6E5B41);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; }
}

/* CTA - was a matte charcoal slab; now polished bronze metal with a running
   sheen band on hover. */
.cta-btn { background: linear-gradient(180deg, #4A453D 0%, #3A362F 100%);
    position: relative; overflow: hidden;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.14),
                0 8px 20px -12px rgba(63,59,53,.55); }
.cta-btn:hover { background: linear-gradient(180deg, #B08A55 0%, #8A6A3C 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.32),
                0 16px 32px -16px rgba(63,59,53,.65); }
.cta-btn::before { content: ""; position: absolute; top: 0; left: -60%;
    width: 40%; height: 100%; transform: skewX(-20deg); pointer-events: none;
    background: linear-gradient(100deg, transparent,
        rgba(255,255,255,.35), transparent); transition: none; }
.cta-btn:hover::before { animation: gloss-glint 0.7s ease-out 1; }

/* Streamlit metric readouts - lift the flat numbers onto small glass tiles,
   consistent with the hero deck, with a hover lift for tactility. */
[data-testid="stMetric"] {
    background: linear-gradient(157deg, rgba(246,242,234,.92), rgba(233,227,215,.62));
    border: 1px solid #D3CBBA; padding: 15px 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.6),
                0 10px 26px -20px rgba(63,59,53,.5);
    transition: transform .35s cubic-bezier(.16,1,.3,1), box-shadow .35s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.75),
                0 18px 40px -22px rgba(63,59,53,.6); }

/* Charts float on soft depth so the surfaces read as glossy panes, not
   flat ink on paper. */
[data-testid="stPlotlyChart"], [data-testid="stImage"] {
    box-shadow: 0 18px 44px -32px rgba(63,59,53,.5);
    transition: box-shadow .4s ease; }
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 22px 52px -30px rgba(63,59,53,.6); }

/* Expanders + dataframes - a whisper of gradient + inner highlight so panels
   catch light instead of sitting dead flat. */
[data-testid="stExpander"] {
    background: linear-gradient(160deg, #EFEAE0, #E6E0D4) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.5),
                0 8px 22px -18px rgba(63,59,53,.4) !important; }
[data-testid="stDataFrame"] {
    box-shadow: 0 12px 30px -24px rgba(63,59,53,.45); }

/* Section hairline draws itself in - a bronze line unspooling left-to-right. */
.section-divider { background-size: 200% 100%;
    animation: divider-draw 1.1s cubic-bezier(.16,1,.3,1) both; }
@keyframes divider-draw { from { background-size: 0% 100%; opacity: 0; }
                          to   { background-size: 200% 100%; opacity: 1; } }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .verdict-box::after, .cta-btn::before { display: none !important; }
    .section-divider { background-size: 200% 100% !important; opacity: 1 !important; } }
</style>
""", unsafe_allow_html=True)

# ---- ROUND 3 (2026-07-09) ----
# Softened geometry (loosen the border-radius:0 doctrine into a restrained
# radius scale + organic/circular accents), glossy chart halos + glowing bars,
# a whisper of scroll-motion-blur on decorative layers, and UFO tile arrivals.
# Still institutional: radii are gentle (10-26px), no candy.
st.markdown("""
<style>
/* --- Radius scale: unblock the corners the base pinned to 0 --- */
[data-testid="stExpander"], [data-testid="stMetric"],
div[data-baseweb="select"]>div, .stNumberInput input,
[data-testid="stTextInput"] input, .stMultiSelect [data-baseweb="select"]>div,
[data-testid="stDataFrame"], [data-baseweb="tab"] { border-radius: 14px !important; }
.stButton>button { border-radius: 12px !important; }
[data-testid="stPlotlyChart"] { border-radius: 16px !important; }
.verdict-box { border-radius: 20px; }
.cta-btn { border-radius: 40px !important; }        /* CTA becomes a soft capsule */
.hero-crest { border-radius: 26px; }                /* squircle, not a hard square */

/* Diversify: each hero tile carries a different corner profile so the deck
   reads as a composition, not a grid of identical boxes. */
.hstat { border-radius: 16px; }
.hstat:nth-child(1) { border-radius: 22px 22px 22px 6px; }
.hstat:nth-child(2) { border-radius: 22px 6px 22px 22px; }
.hstat:nth-child(3) { border-radius: 6px 22px 22px 22px; }
.hstat:nth-child(4) { border-radius: 22px 22px 6px 22px; }

/* Organic bronze light - soft radial glows behind the hero. Not flat blobs:
   low-opacity, no hard edge, palette-only. */
.hero-section::before, .hero-section::after { content:""; position:absolute;
    border-radius:50%; pointer-events:none; z-index:0; filter: blur(48px); }
.hero-section::before { width:340px; height:340px; left:-90px; top:-40px;
    background: radial-gradient(circle, rgba(154,123,79,.16), transparent 68%); }
.hero-section::after { width:420px; height:420px; right:5%; bottom:-150px;
    background: radial-gradient(circle, rgba(154,123,79,.12), transparent 70%); }
.hero-left, .hero-stats { position: relative; z-index: 1; }

/* --- CHARTS: glossy finish - a bronze halo behind, glow on the bars --- */
[data-testid="stPlotlyChart"] {
    box-shadow: 0 0 42px -6px rgba(154,123,79,.30),
                0 18px 44px -30px rgba(63,59,53,.5);
    transition: filter .18s ease, box-shadow .4s ease; }
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 0 52px -4px rgba(154,123,79,.38),
                0 22px 52px -28px rgba(63,59,53,.6); }
[data-testid="stPlotlyChart"] g.bars .point path,
[data-testid="stPlotlyChart"] g.bars path {
    filter: drop-shadow(0 0 5px rgba(154,123,79,.55)); }

/* --- UFO ARRIVAL: intro tiles drop from above with a light settle. Trigger
   is injected only on first load (boot block), so reruns never replay it. --- */
@keyframes ufo-drop {
    0%   { opacity:0; transform: translateY(-52px) scale(.96); }
    60%  { opacity:1; transform: translateY(6px)  scale(1.005); }
    100% { opacity:1; transform: translateY(0)    scale(1); } }

/* Skeleton bones + tiles pick up the softer radii */
#boot-skel .sk { border-radius: 14px; }
#boot-skel .sk.crest { border-radius: 26px; }
#boot-skel .sk-head { display:flex; gap:24px; align-items:flex-start; }
#boot-skel .sk-titles { display:flex; flex-direction:column; gap:14px;
    flex:1; padding-top:6px; }
#boot-skel .sk-tiles { display:grid; grid-template-columns:repeat(4,1fr);
    gap:16px; margin-top:10px; }
#boot-skel .sk.tile { height:96px; border-radius:16px; }
#boot-skel .sk-load { font-family:'Helvetica Neue',sans-serif; font-size:11px;
    letter-spacing:.24em; text-transform:uppercase; color:#9A7B4F;
    margin-top:8px; animation: sk-pulse 1.3s ease-in-out infinite; }
@keyframes sk-pulse { 0%,100%{opacity:.45;} 50%{opacity:1;} }

@media (prefers-reduced-motion: reduce) {
    .hero-section::before, .hero-section::after { display:none !important; } }
</style>
""", unsafe_allow_html=True)

# ---- ROUND 4 (2026-07-09): THE THREE FINISHES + RUNES ----
# One material system, three finishes, each with a job:
#   MATTE  - the reading surface: body text, sections, pillars, expanders,
#            tables. Paper. (Rebalanced back from round 2's gloss.)
#   GLOSSY - the trophies: verdict numeral, CTA, hero tiles, chart halos.
#   GLASS  - the instruments you touch: buttons, selects + their dropdown
#            menus, read-me panels. Real translucency (backdrop-filter),
#            inner light, react on hover. (Liquid-glass principles in CSS.)
# Plus: lacquered page background, and Nordic runes engraved in the stone -
# Tiwaz (Tyr's arrow: honesty/justice), Dagaz (daybreak: clarity/problem-
# solving), Ingwaz (completion: integrity), Ansuz (wisdom).
st.markdown("""
<style>
/* --- LACQUER: light falls on the page from above; corners hold shade --- */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1100px 520px at 50% -140px, rgba(255,253,247,.95), rgba(255,253,247,0) 65%),
      radial-gradient(760px 480px at 88% 12%, rgba(154,123,79,.06), transparent 70%),
      radial-gradient(820px 520px at 6% 92%, rgba(63,59,53,.05), transparent 72%),
      #EDE9E3 !important;
    background-attachment: fixed; }

/* --- MATTE REBALANCE: expanders and tables go back to paper --- */
[data-testid="stExpander"] { background: #ECE7DD !important;
    box-shadow: none !important; }
[data-testid="stDataFrame"] { box-shadow: none; }

/* --- GLASS: the instruments --- */
@supports ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
  .stButton>button { background: rgba(246,242,234,.5);
      -webkit-backdrop-filter: blur(14px) saturate(1.2);
      backdrop-filter: blur(14px) saturate(1.2);
      border: 1px solid rgba(154,123,79,.4);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.65),
                  inset 0 -8px 14px -12px rgba(154,123,79,.35),
                  0 8px 22px -16px rgba(63,59,53,.4); }
  div[data-baseweb="select"] > div, .stNumberInput input,
  [data-testid="stTextInput"] input,
  .stMultiSelect [data-baseweb="select"] > div {
      background: rgba(246,242,234,.48) !important;
      -webkit-backdrop-filter: blur(12px) saturate(1.15);
      backdrop-filter: blur(12px) saturate(1.15);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.55); }
  /* Dropdown menus float as true glass panes over the page */
  [data-baseweb="popover"] [data-baseweb="menu"],
  [data-baseweb="popover"] ul[role="listbox"] {
      background: rgba(244,240,232,.72) !important;
      -webkit-backdrop-filter: blur(20px) saturate(1.3);
      backdrop-filter: blur(20px) saturate(1.3);
      border: 1px solid rgba(154,123,79,.35); border-radius: 14px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.6),
                  0 24px 52px -22px rgba(63,59,53,.5); overflow: hidden; }
  [data-baseweb="menu"] li[role="option"]:hover,
  ul[role="listbox"] li:hover { background: rgba(154,123,79,.16) !important; }
  /* Read-me panels: frosted glass slabs, the briefing cards you lean on */
  .read-me { background: rgba(246,242,234,.5) !important;
      -webkit-backdrop-filter: blur(16px) saturate(1.15);
      backdrop-filter: blur(16px) saturate(1.15);
      border: 1px solid rgba(154,123,79,.3) !important; border-radius: 14px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.55),
                  0 12px 30px -24px rgba(63,59,53,.4); }
}

/* (Nordic rune engravings removed 2026-07-09 at John's call - read as
   ornament noise against the institutional register. Crest-fragment
   engravings stay; they are the brand.) */

/* --- ROUND 5: precision numerals + one-pass ignite ---
   Tabular figures on every metric so columns of numbers align like a
   ledger - precision feel, serif untouched. Charts arrive with a single
   ignite bloom (their one good easing token, cubic-bezier(.25,1,.5,1)) -
   one pass, never looping, killed under reduced-motion by the global guard. */
[data-testid="stMetricValue"], .verdict-number, .hstat .n {
    font-variant-numeric: tabular-nums; }
@keyframes heat-bloom { from { opacity: 0; transform: scale(.985);
                               filter: saturate(.55) brightness(.92); }
                        to   { opacity: 1; transform: none; filter: none; } }
[data-testid="stPlotlyChart"] {
    animation: heat-bloom .9s cubic-bezier(.25,1,.5,1) both; }

/* --- ROUND 6: reference-layout lifts (Sculptor / RiverNorth), our palette ---
   Sculptor: the hero display element is a stack of period-terminated serif
   words; the brand recedes to the eyebrow. RiverNorth: section rhythm comes
   from full-width alternating color bands - here the twin showcases sit on
   one full-bleed charcoal band between beige fields. */
.hero-title .hline { display: block; }
.hero-title { font-size: clamp(40px, 10.5vw, 84px) !important; }

.showcase-row { background: #3F3B35; position: relative;
    margin-left: calc(50% - 50vw); margin-right: calc(50% - 50vw);
    padding: 72px max(7vw, calc(50vw - 744px)) 64px;
    border-top: 2px solid #9A7B4F; border-bottom: 2px solid #9A7B4F; }
/* Architectural plate behind the band - photo arrives via a small injected
   style (base64) layered UNDER a charcoal scrim; this rule holds geometry
   and the duotone, tuned WARM (dusk-bronze city, per John's references),
   not cold fog. Text sits above on its own layer. */
.showcase-row::before { content: ""; position: absolute; inset: 0;
    background-size: cover; background-position: center 30%;
    filter: grayscale(.4) sepia(.42) brightness(.6) contrast(1.03); }
/* The band's dressed edges - no more naked ribbon: outside, a 2px bronze
   rule (border above); inside, a 1px bronze hairline set 14px in (the
   classic double-rule ledger frame), plus soft gradient eases at top and
   bottom so beige melts into the dark instead of snapping. */
.showcase-row::after { content: ""; position: absolute; inset: 0;
    pointer-events: none;
    background:
      linear-gradient(rgba(176,138,85,.55), rgba(176,138,85,.55))
        left 0 top 14px / 100% 1px no-repeat,
      linear-gradient(rgba(176,138,85,.55), rgba(176,138,85,.55))
        left 0 bottom 14px / 100% 1px no-repeat,
      linear-gradient(180deg, rgba(237,233,227,.16), rgba(237,233,227,0) 120px),
      linear-gradient(0deg, rgba(30,27,23,.5), rgba(30,27,23,0) 140px); }
.showcase-row > * { position: relative; z-index: 1; }
/* CTA arrival - THE EARTHBENDER: the towers launch from their foundations
   (buried 70% below the frame, dim as underground rock), shoot up hard,
   overshoot the rest position a hair - the slam - flare bright as they
   crest into the light, then settle into the skyline. "Built from the
   ground up," literally. Delayed ~0.35s so the launch detonates just as
   the glide delivers the reader onto the band. One pass, clipped by the
   row's frame, killed by the global reduced-motion guard. */
.showcase-row { overflow: hidden; }
.showcase-row.band-arrive::before {
    animation: band-rise 1.45s cubic-bezier(.22,.9,.24,1) .35s both; }
@keyframes band-rise {
  0%   { transform: translateY(70%) scale(1.14);
         filter: grayscale(.4) sepia(.42) brightness(.3) contrast(1.05); }
  55%  { transform: translateY(-2.6%) scale(1.04);
         filter: grayscale(.4) sepia(.42) brightness(.68) contrast(1.03); }
  74%  { transform: translateY(1.1%) scale(1.01);
         filter: grayscale(.4) sepia(.42) brightness(.5) contrast(1.03); }
  100% { transform: none;
         filter: grayscale(.4) sepia(.42) brightness(.54) contrast(1.03); }
}

/* "See the hardest trade" arrival - no earthquake twice. The whole
   conviction text stack (eyebrow + title + paragraph, wrapped in
   .conv-core) gets the highlight: a DOUBLE FRAME in the band-border
   bronze - 2px outer rule, 1px inner hairline set 8px in, near-square
   corners to match the pillar boxes - filled with LATE SUN, a warm gold
   wash blooming from the upper corner the way evening light falls
   through a childhood window onto a wall. Draws in, holds so the eye
   lands, then the light leaves the room. */
.conv-core { position: relative; display: flex;
    flex-direction: column; gap: 18px; }
#conviction.ring-arrive .conv-core::after { content: ""; position: absolute;
    inset: -18px -24px; border: 2px solid rgba(176,138,85,.85);
    border-radius: 4px; pointer-events: none;
    background: radial-gradient(130% 140% at 84% 4%,
        rgba(201,162,39,.16), rgba(154,123,79,.08) 46%, transparent 74%);
    box-shadow: 0 0 26px rgba(201,162,39,.13),
                inset 0 0 34px rgba(201,162,39,.07);
    animation: ring-hold 2.8s cubic-bezier(.25,1,.5,1) .55s both; }
#conviction.ring-arrive .conv-core::before { content: ""; position: absolute;
    inset: -10px -16px; border: 1px solid rgba(176,138,85,.45);
    border-radius: 2px; pointer-events: none;
    animation: ring-hold 2.8s cubic-bezier(.25,1,.5,1) .62s both; }
@keyframes ring-hold {
  0%   { opacity: 0; transform: scale(1.05); }
  22%  { opacity: 1; transform: scale(1); }
  70%  { opacity: 1; }
  100% { opacity: 0; transform: scale(1.006); }
}
.showcase-row .showcase-title { color: #EDE9E3; }
.showcase-row .showcase-body { color: #C4BDAE; }
.showcase-row .showcase-eyebrow { color: #B08A55; }
.showcase-row .pillar-row { border-top-color: rgba(196,189,174,.28); }
.showcase-row .pillar-card { border-left-color: rgba(196,189,174,.28); }
.showcase-row .pillar-card:hover { border-left-color: #B08A55; }
.showcase-row .pillar-label { color: #B08A55; }
.showcase-row .pillar-desc { color: #A89F8F; }
.showcase-row .engrave { opacity: .12; }
/* Sculptor's outlined CTA, at home on the dark band */
.showcase-row .cta-btn { background: transparent; border: 1px solid #B08A55;
    color: #EDE9E3 !important; box-shadow: none; }
.showcase-row .cta-btn:hover { background: #9A7B4F; border-color: #9A7B4F; }

/* Full-bleed bands overflow the court by design - never let that leak into
   a horizontal scrollbar (it also seeded phantom mini-scrollers). */
[data-testid="stAppViewContainer"] section, [data-testid="stMain"] {
    overflow-x: clip; }

/* The conviction slabs breathe below the dark band instead of touching it */
.slab { margin-top: 48px; }

/* --- MCAP FOOTER (private-equity-rebrand reference): the page closes on a
   full-bleed charcoal band - a "where to next" rail of outlined link boxes,
   then a thin copyright/disclaimer bar. Our twist: the rail is numbered like
   a ledger and every claim on it stays honest. --- */
.meleona-footer { background: #3F3B35;
    margin: 84px calc(50% - 50vw) 0; padding: 56px max(7vw, calc(50vw - 744px)) 36px; }
.meleona-footer .f-rail { display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin-bottom: 44px; }
@media (max-width: 1100px) { .meleona-footer .f-rail { grid-template-columns: 1fr; } }
.meleona-footer .f-box { display: block; border: 1px solid rgba(196,189,174,.3);
    padding: 20px 22px; color: #EDE9E3 !important; text-decoration: none !important;
    font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: .18em; text-transform: uppercase;
    transition: border-color .25s ease, background .25s ease; }
.meleona-footer .f-box:hover { border-color: #B08A55;
    background: rgba(154,123,79,.12); }
.meleona-footer .f-box small { display: block; margin-top: 8px; color: #A89F8F;
    font-family: Georgia, serif; font-size: 13px; letter-spacing: .02em;
    text-transform: none; line-height: 1.5; }
.meleona-footer .f-num { color: #B08A55; margin-right: 10px; }
.meleona-footer .f-bar { border-top: 1px solid rgba(196,189,174,.25);
    padding-top: 18px; display: flex; justify-content: space-between;
    flex-wrap: wrap; gap: 8px; font-family: 'Helvetica Neue', sans-serif;
    font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
    color: #A89F8F; }
</style>
""", unsafe_allow_html=True)

# ---- Themed Plotly palette + chart helpers (institutional beige/bronze) ----
BRONZE = "#9A7B4F"
BRONZE_DK = "#8A6A3C"
CHARCOAL = "#3F3B35"
BAND_OUTER = "rgba(154,123,79,0.14)"   # light bronze - 5–95 percentile cone
BAND_INNER = "rgba(154,123,79,0.30)"   # medium bronze - 25–75 percentile cone
GRID = "rgba(63,59,53,0.12)"
AXIS_LINE = "rgba(63,59,53,0.28)"

PLOTLY_CFG = {"displayModeBar": False, "staticPlot": False}


def _style_fig(fig, height: int = 300):
    """Apply the calm serif/beige institutional theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, 'Times New Roman', serif", color=CHARCOAL, size=13),
        showlegend=False,
        bargap=0.12,
        hoverlabel=dict(bgcolor="#F4F1EA", bordercolor=BRONZE,
                        font=dict(family="Georgia, serif", color=CHARCOAL, size=13)),
        transition=dict(duration=380, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS_LINE, ticks="outside",
                     tickcolor=AXIS_LINE)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE)
    return fig


def fan_chart(bands: dict):
    """
    Monte Carlo outcome cone, rendered as a wind-tunnel envelope: median path
    + 25–75 and 5–95 percentile bands.

    Rendering note: the percentile paths are drawn as splines. Every plotted
    point is a real computed percentile from the simulation - the spline only
    interpolates *between* those points instead of connecting them with jagged
    straight segments. Hover reports the true underlying value, so nothing is
    smoothed away from the numbers themselves; only the ink between them.
    """
    d = bands["days"]

    def p(a):
        return np.asarray(a) * 100.0

    # One curve style for every edge of the cone - laminar, not stair-stepped.
    edge = dict(width=0, shape="spline", smoothing=1.0)

    fig = go.Figure()
    # Outer 5–95 cone (draw upper first, then lower with fill-to-previous)
    fig.add_trace(go.Scatter(x=d, y=p(bands["p95"]), line=edge, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p5"]), fill="tonexty", fillcolor=BAND_OUTER,
                             line=edge, hoverinfo="skip"))
    # Inner 25–75 cone
    fig.add_trace(go.Scatter(x=d, y=p(bands["p75"]), line=edge, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p25"]), fill="tonexty", fillcolor=BAND_INNER,
                             line=edge, hoverinfo="skip"))
    # Hairline edges trace the envelope - the silhouette of the airflow
    for key in ("p95", "p5"):
        fig.add_trace(go.Scatter(x=d, y=p(bands[key]), hoverinfo="skip",
                                 line=dict(color="rgba(154,123,79,0.45)", width=1,
                                           shape="spline", smoothing=1.0)))
    # Median path - the centreline, drawn last so it sits on top
    fig.add_trace(go.Scatter(
        x=d, y=p(bands["p50"]), name="median",
        line=dict(color=CHARCOAL, width=2.4, shape="spline", smoothing=1.0),
        hovertemplate="Day %{x}<br><b>%{y:.1f}%</b> median<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS_LINE, width=1, dash="dot"))
    fig.update_layout(
        xaxis_title="Trading days", yaxis_title="Cumulative return (%)",
        hovermode="x unified",                 # one clean readout, not five
        transition=dict(duration=420, easing="cubic-in-out"),
    )
    fig = _style_fig(fig, height=340)
    # Aerodynamic axes: no tick spikes, breathing gridlines only
    fig.update_xaxes(ticks="", showspikes=False)
    fig.update_yaxes(ticks="", ticksuffix="%")
    return fig


def surface_chart(density: dict):
    """
    3D surface of how the simulated outcome distribution evolves over the
    horizon - the fan chart's cone re-expressed as a probability surface
    (day x return-bin x density) instead of percentile lines.
    """
    fig = go.Figure(go.Surface(
        x=density["days"], y=density["returns"], z=density["density"].T,
        colorscale=[[0, "#EDE9E3"], [0.5, BRONZE], [1, CHARCOAL]],
        showscale=False,
        hovertemplate="Day %{x} · Return %{y:.0%} · density %{z:.3f}<extra></extra>",
    ))
    scene_axis = dict(gridcolor=GRID, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE,
                      showbackground=True, backgroundcolor="rgba(237,233,227,0.35)")
    fig.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, 'Times New Roman', serif", color=CHARCOAL, size=12),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Trading day", **scene_axis),
            yaxis=dict(title="1-year outcome", tickformat=".0%", **scene_axis),
            zaxis=dict(title="Density", **scene_axis),
        ),
        hoverlabel=dict(bgcolor="#F4F1EA", font=dict(family="Georgia, serif", color=CHARCOAL)),
    )
    return fig


def _seed_particles(density: dict, n_particles: int = 220, seed: int = 42):
    """
    Sample particle anchor points weighted by the density surface itself, so
    the 'drifting particle' overlay clusters where the probability mass
    actually is instead of floating randomly in empty space.
    """
    z = np.asarray(density["density"])            # shape (n_days, n_returns)
    days = np.asarray(density["days"], dtype=float)
    rets = np.asarray(density["returns"], dtype=float)

    w = np.clip(z.flatten(), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / w.size
    rng = np.random.default_rng(seed)
    idx = rng.choice(w.size, size=n_particles, p=w, replace=True)
    day_idx, ret_idx = np.unravel_index(idx, z.shape)

    px = days[day_idx].tolist()
    py = rets[ret_idx].tolist()
    pz = (z[day_idx, ret_idx] * 1.05).tolist()      # sit just above the surface
    return px, py, pz


def living_surface_html(density: dict, height: int = 520, n_particles: int = 220) -> str:
    """
    A 'living' version of the 3D outcome-distribution surface: raw plotly.js
    (bypassing st.plotly_chart's static embed) with a drifting-particle
    overlay and a slow continuous camera auto-rotate, paused while the viewer
    is manually dragging. Same surface/colorscale/hover as surface_chart().
    """
    z = np.asarray(density["density"])
    payload = json.dumps({
        "days": [float(x) for x in density["days"]],
        "rets": [float(x) for x in density["returns"]],
        "z": z.T.tolist(),                          # shape (n_returns, n_days)
        "px": (p := _seed_particles(density, n_particles))[0],
        "py": p[1],
        "pz": p[2],
    })

    return f"""
<div style="background:
      radial-gradient(120% 90% at 50% 0%, #423C33 0%, #2E2A24 58%, #262320 100%);
    border: 1px solid #9A7B4F; border-radius: 14px; padding: 10px 8px 4px;
    box-shadow: inset 0 1px 0 rgba(237,233,227,.08);">
  <div id="living3d" style="width:100%;height:{height - 16}px;"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function() {{
  const data = {payload};
  const surface = {{
    type: 'surface', x: data.days, y: data.rets, z: data.z,
    colorscale: [[0, '#EDE9E3'], [0.5, '#9A7B4F'], [1, '#3F3B35']],
    showscale: false, opacity: 0.96,
    hovertemplate: 'Day %{{x}} · Return %{{y:.0%}} · density %{{z:.3f}}<extra></extra>',
  }};
  const particles = {{
    type: 'scatter3d', mode: 'markers', x: data.px, y: data.py, z: data.pz,
    marker: {{ size: 2.6, color: '#C9A227', opacity: 0.55 }},
    hoverinfo: 'skip',
  }};
  const layout = {{
    height: {height - 16}, margin: {{l:0,r:0,t:10,b:0}},
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: "Georgia, 'Times New Roman', serif", color: '#D9D2C4', size: 12 }},
    scene: {{
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: {{ title: 'Trading day', gridcolor: 'rgba(237,233,227,0.14)',
               backgroundcolor: 'rgba(24,21,18,0.45)', showbackground: true }},
      yaxis: {{ title: '1-year outcome', tickformat: '.0%', gridcolor: 'rgba(237,233,227,0.14)',
               backgroundcolor: 'rgba(24,21,18,0.45)', showbackground: true }},
      zaxis: {{ title: 'Density', gridcolor: 'rgba(237,233,227,0.14)',
               backgroundcolor: 'rgba(24,21,18,0.45)', showbackground: true }},
      camera: {{ eye: {{x: 1.6, y: 1.6, z: 0.9}} }},
    }},
  }};

  Plotly.newPlot('living3d', [surface, particles], layout, {{displayModeBar: false}})
    .then(function(gd) {{
      let t = 0, userInteracting = false, resumeTimer = null;
      const pause = () => {{ userInteracting = true; clearTimeout(resumeTimer); }};
      const resume = () => {{ resumeTimer = setTimeout(() => {{ userInteracting = false; }}, 4000); }};
      gd.addEventListener('mousedown', pause);
      gd.addEventListener('touchstart', pause);
      window.addEventListener('mouseup', resume);
      window.addEventListener('touchend', resume);

      /* Animation, split by cost: the CAMERA glides every frame via
         requestAnimationFrame (60fps - the old 150ms interval stepped it
         at ~7fps, which is exactly what read as choppy), while the
         particle field re-uploads on a slower ~140ms budget. Delta-time
         based, so speed is identical on any refresh rate. */
      let lastT = null, acc = 0, angle = 0;
      const step = function(now) {{
        if (lastT === null) lastT = now;
        const dt = Math.min(now - lastT, 100); lastT = now;
        if (!userInteracting) {{
          angle += dt * 0.00010;                 // one lap ≈ 63s, silk-smooth
          Plotly.relayout('living3d', {{
            'scene.camera.eye.x': 1.6 * Math.cos(angle),
            'scene.camera.eye.y': 1.6 * Math.sin(angle),
          }});
        }}
        acc += dt;
        if (acc >= 140) {{                        // particle drift, budgeted
          acc = 0; t += 1;
          const n = data.px.length;
          const nx = new Array(n), ny = new Array(n), nz = new Array(n);
          for (let i = 0; i < n; i++) {{
            nx[i] = data.px[i] + Math.sin(t * 0.2 + i * 1.7) * 3;
            ny[i] = data.py[i] + Math.cos(t * 0.25 + i * 2.3) * 0.01;
            nz[i] = Math.max(0, data.pz[i] + Math.sin(t * 0.35 + i) * 0.012);
          }}
          Plotly.restyle('living3d', {{x: [nx], y: [ny], z: [nz]}}, [1]);
        }}
      }};

      /* Only animate while this scene is actually on screen. Streamlit keeps
         inactive tab panels mounted, so an ungated interval would repaint the
         surface forever on every other tab - burning CPU and never letting the
         renderer idle. Gate on both intersection and page visibility. */
      /* Respect the OS reduced-motion setting - draw once, hold still. This is
         a hard gate, checked inside shouldRun, because the IntersectionObserver
         callback fires asynchronously and would otherwise restart the loop. */
      const reduced = !!(window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);

      let rafId = null;
      const loop = function(now) {{ step(now); rafId = requestAnimationFrame(loop); }};
      const running = () => rafId !== null;
      const start = function() {{
        if (!running()) {{ lastT = null; rafId = requestAnimationFrame(loop); }} }};
      const stop = function() {{
        if (running()) {{ cancelAnimationFrame(rafId); rafId = null; }} }};
      const shouldRun = function(visible) {{
        (!reduced && visible && document.visibilityState === 'visible') ? start() : stop();
      }};

      if ('IntersectionObserver' in window) {{
        new IntersectionObserver(function(entries) {{
          shouldRun(entries[0].isIntersecting);
        }}, {{ threshold: 0 }}).observe(gd);
      }} else {{
        shouldRun(true);   /* no observer support: run whenever page is visible */
      }}
      document.addEventListener('visibilitychange', function() {{
        shouldRun(gd.getBoundingClientRect().height > 0);
      }});
    }});
}})();
</script>
"""


def panel_head(title: str, subtitle: str = "") -> None:
    """Ruled section lintel inside a tab - replaces bare `###### ` markdown
    headers so every block reads as a titled stone panel, not a run-on wall."""
    sub = f'<span class="s">{subtitle}</span>' if subtitle else ""
    st.markdown(f'<div class="panel-head"><span class="t">{title}</span>{sub}</div>',
                unsafe_allow_html=True)


def read_me(html: str) -> None:
    """Plain-language 'how to read this' block under a chart. Bold key words
    with <b>…</b>. Keeps the honest, defensible captions but makes the
    explanation impossible to miss."""
    st.markdown(f'<div class="read-me">{html}</div>', unsafe_allow_html=True)


def outcome_hist(total_returns, cvar: float):
    """Histogram of simulated 1-year outcomes. The tail the CVaR measures is
    inked in oxblood so the eye lands on the danger, not the middle; every
    bar is a real simulation count - color is annotation, not data."""
    vals = np.asarray(total_returns) * 100.0
    counts, edges = np.histogram(vals, bins=48)
    mids = (edges[:-1] + edges[1:]) / 2
    tail = -cvar * 100.0
    colors = ["#8A3B2E" if m <= tail else BRONZE for m in mids]
    fig = go.Figure(go.Bar(
        x=mids, y=counts, width=(edges[1] - edges[0]) * 0.92,
        marker=dict(color=colors, line=dict(width=0)), opacity=0.92,
        hovertemplate="%{x:.0f}%: %{y} simulations<extra></extra>"))
    fig.add_vline(x=tail, line=dict(color=CHARCOAL, width=2, dash="dash"),
                  annotation_text="CVaR", annotation_position="top left",
                  annotation_font=dict(color=CHARCOAL, size=12))
    fig.add_vline(x=float(np.median(vals)),
                  line=dict(color=BRONZE_DK, width=1, dash="dot"),
                  annotation_text="median", annotation_position="top right",
                  annotation_font=dict(color=BRONZE_DK, size=11))
    fig.update_layout(xaxis_title="1-year return (%)", yaxis_title="Simulations",
                      bargap=0.06)
    return _style_fig(fig, height=280)


def hbar(series: pd.Series, color=BRONZE, pct: bool = False, title_x: str = ""):
    """Themed horizontal bar chart. Bars deepen with magnitude - the biggest
    value wears the darkest bronze - so ranking reads at a glance."""
    x = series.values * (100 if pct else 1)
    span = float(np.max(np.abs(x))) or 1.0
    def _shade(v):  # lerp #CBBB94 (light) -> #8A6A3C (deep) by |value|
        f = abs(v) / span
        r = int(0xCB + (0x8A - 0xCB) * f)
        g = int(0xBB + (0x6A - 0xBB) * f)
        b = int(0x94 + (0x3C - 0x94) * f)
        return f"rgb({r},{g},{b})"
    fig = go.Figure(go.Bar(
        x=x, y=list(series.index), orientation="h",
        marker=dict(color=[_shade(v) for v in x], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.2f}" + ("%" if pct else "") + "<extra></extra>"))
    fig.update_layout(xaxis_title=title_x)
    return _style_fig(fig, height=max(160, 30 * len(series) + 40))


def grit_breakdown_fig(scores: pd.DataFrame):
    """Grouped bar: recovery / consistency / resilience sub-scores per ticker."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(scores.index), x=scores["recovery_score"].values, orientation="h",
        name="recovery", marker=dict(color="#CBBB94"),
        hovertemplate="%{y} recovery: %{x:.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=list(scores.index), x=scores["consistency_score"].values, orientation="h",
        name="consistency", marker=dict(color=BRONZE),
        hovertemplate="%{y} consistency: %{x:.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=list(scores.index), x=scores["resilience_score"].values, orientation="h",
        name="resilience", marker=dict(color=BRONZE_DK),
        hovertemplate="%{y} resilience: %{x:.0f}<extra></extra>"))
    fig = _style_fig(fig, height=max(220, 50 * len(scores)))
    fig.update_layout(
        barmode="group", xaxis_title="score (0–100, relative to this universe)",
        showlegend=True, legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return fig


# ---- Hero: the pitch, not the dashboard ----
with open("assets/logo.svg", "r", encoding="utf-8") as f:
    logo_svg = f.read()

# Architectural plate behind the hero stat deck (replaces the watermark crest).
# assets/facade.jpg - Unsplash (free commercial license, no attribution
# required). Duotoned toward the palette in CSS, so the photo can never clash.
try:
    with open("assets/facade.jpg", "rb") as f:
        _facade_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        # The architecture washes across the WHOLE hero, melting into beige
        # toward the text side - the building emerges from the page's own
        # color. Luminosity blend: the photo keeps only its LIGHT - its hue
        # comes entirely from the base color beneath, which is CITY-STONE
        # BEIGE in the crest's own warm family (not gray, not orange).
        f"<style>.hero-section {{ background-color: #C9AF87; "
        f"background-image: linear-gradient(90deg, "
        f"#EDE9E3 0%, rgba(237,233,227,.96) 40%, rgba(237,233,227,.62) 66%, "
        f"rgba(237,233,227,.22) 100%), "
        f"url(data:image/jpeg;base64,{_facade_b64}); "
        f"background-size: auto, cover; "
        f"background-position: left, right 78%; "
        f"background-blend-mode: normal, luminosity; "
        # Full-bleed: Casper fills the page edge-to-edge and reaches the top;
        # the bottom keeps its hairline + beige gap, the seam before Gotham.
        f"margin: -2.4rem calc(50% - 50vw) 0; "
        f"padding: 64px max(7vw, calc(50vw - 744px)) 48px; }}</style>",
        unsafe_allow_html=True)
except OSError:
    pass  # no photo on disk -> tiles render on the plain field, nothing breaks

# Dark-band plate: the charcoal showcase band gets its own architectural
# photograph (assets/band.jpg - Unsplash, free commercial license): fog-bound
# towers, duotoned near-charcoal in CSS with a scrim baked in so the beige
# text stays the loudest thing on the band. Same graceful fallback.
try:
    with open("assets/band.jpg", "rb") as f:
        _band_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<style>.showcase-row::before {{ background-image: "
        f"linear-gradient(165deg, rgba(59,50,40,.42), rgba(45,39,32,.68)), "
        f"url(data:image/jpeg;base64,{_band_b64}); }}</style>",
        unsafe_allow_html=True)
except OSError:
    pass  # band stays plain charcoal

# Boot veil renders ONLY on the first script run of a session. Streamlit
# reruns the whole script on every interaction (and the freshness ticker),
# which would re-create the veil and restart its fade forever - so after
# the first run it is simply never rendered again.
if not st.session_state.get("_booted"):
    st.session_state["_booted"] = True
    st.markdown("""
<div id="boot-skel">
  <div class="sk-head">
    <div class="sk crest"></div>
    <div class="sk-titles">
      <div class="sk title"></div>
      <div class="sk line"></div>
      <div class="sk line short"></div>
    </div>
  </div>
  <div class="sk-tiles">
    <div class="sk tile"></div><div class="sk tile"></div>
    <div class="sk tile"></div><div class="sk tile"></div>
  </div>
  <div class="sk chart"></div>
  <div class="sk-load">Loading market data</div>
</div>
<style>
  /* First-load only: the intro tiles fly in top-to-down (UFO settle). This
     style block renders solely on the un-booted first pass, so Streamlit
     reruns never replay the arrival. */
  .hero-stats .hstat { animation: ufo-drop .6s cubic-bezier(.2,.9,.25,1) both; }
  .hero-stats .hstat:nth-child(1) { animation-delay: .05s; }
  .hero-stats .hstat:nth-child(2) { animation-delay: .13s; }
  .hero-stats .hstat:nth-child(3) { animation-delay: .21s; }
  .hero-stats .hstat:nth-child(4) { animation-delay: .29s; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-section reveal" id="hero">
  <div class="hero-left">
    <div class="hero-crest">{logo_svg}</div>
    <div class="hero-eyebrow">Meleona &middot; Portfolio Risk Engine</div>
    <h1 class="hero-title"><span class="hline">Grit.</span><span class="hline">Discipline.</span><span class="hline">Evidence.</span></h1>
    <div class="hero-sub">
      A hedge-fund-grade portfolio risk engine - VaR, CVaR, Monte Carlo stress
      testing, and named factor exposures, computed live from real market data.
      But every stock has drawdowns. What sets a name apart is what happens
      after one - that's what we call <strong>grit</strong>.
    </div>
    <a href="#grit-showcase" class="cta-btn">Explore what we do &darr;</a>
  </div>
  <div class="hero-stats">
    <div class="hstat"><div class="n">10,000</div><div class="l">Simulated paths</div></div>
    <div class="hstat"><div class="n">10</div><div class="l">Crises replayed</div></div>
    <div class="hstat"><div class="n">2</div><div class="l">Monte Carlo engines</div></div>
    <div class="hstat"><div class="n">4</div><div class="l">Factor exposures</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Showcase: the Grit Zone innovation, explained before you touch a slider ----
st.markdown("""
<div class="showcase-row reveal">
  <div class="showcase-section" id="grit-showcase" style="position:relative;">
    <div class="engrave scale"><svg viewBox="22 36 28 26" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <g fill="none" stroke="#9A7B4F" stroke-width="1.1" stroke-linecap="round">
        <circle cx="35" cy="46" r="9"/>
        <line x1="30" y1="45" x2="40" y2="45"/>
        <line x1="35" y1="42.5" x2="35" y2="50"/>
        <path d="M28.5,46.5 Q30,49 31.5,46.5"/>
        <path d="M38.5,46.5 Q40,49 41.5,46.5"/>
        <line x1="28.5" y1="46.5" x2="30" y2="44.5"/><line x1="31.5" y1="46.5" x2="30" y2="44.5"/>
        <line x1="38.5" y1="46.5" x2="40" y2="44.5"/><line x1="41.5" y1="46.5" x2="40" y2="44.5"/>
        <circle cx="35" cy="45" r="1.1"/>
      </g>
    </svg></div>
    <div class="showcase-eyebrow">The Innovation</div>
    <h2 class="showcase-title">Introducing the Grit Zone</h2>
    <div class="showcase-body">
      Fear &amp; Greed indices measure market mood. We measure something more
      durable: whether an asset, when it gets knocked down, actually gets back
      up - consistently, across real crises. There's no such thing as a
      perfect stock. Grit isn't about avoiding setbacks - it's about what
      happens after one.
    </div>
    <div class="pillar-row">
      <div class="pillar-card">
        <div class="pillar-label">Recovery</div>
        <div class="pillar-desc">How fast and how completely a name claws
          back from its own drawdowns.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-label">Consistency</div>
        <div class="pillar-desc">The share of rolling 1-year holding periods
          that ended positive.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-label">Resilience</div>
        <div class="pillar-desc">How shallow the drawdown and how fast the
          recovery across real historical crises.</div>
      </div>
    </div>
    <a href="#conviction" class="cta-btn">See the hardest trade &rarr;</a>
  </div>
  <div class="showcase-section" id="conviction">
    <div class="conv-core">
      <div class="showcase-eyebrow">The Conviction</div>
      <h2 class="showcase-title">The hardest trade is the one history rewards</h2>
      <div class="showcase-body">
        Your brain treats a falling portfolio the way it treats a physical threat
        - the panic you feel in a crash is wiring, not weakness. That is the
        emotional problem this engine exists to solve. Not with a slogan: with the
        actual record of every named crisis it stress-tests, computed live from
        market data. Below, what really happened to a buyer on the scariest day of
        each crisis - and on the worst-timed day, the pre-crash peak.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---- Showcase: Crisis Conviction - the emotional problem, answered in numbers ----
@st.cache_data(ttl=6 * 3600, show_spinner="Reading the crisis record…")
def load_conviction_data():
    """Benchmark crisis record + AI-capex recovery race, live from Yahoo."""
    return load_conviction()

try:
    _conv = load_conviction_data()
    _s = _conv["summary"]
    _t1, _t3 = _s["trough_1y_later"], _s["trough_3y_later"]
    _p3 = _s["peak_3y_later"]
    _race = _conv["race"]
    # A race is decided when at least one side recovered; the basket wins a
    # race the benchmark never finished (recovered vs. not within ~3y).
    _decided = _race.dropna(subset=["basket_days", "bench_days"], how="all")
    _bwin = int(((_decided["basket_days"].fillna(np.inf)
                  < _decided["bench_days"].fillna(np.inf))).sum())
    _nrace = int(len(_decided))

    _sl1, _sl2, _sl3, _sl4 = st.columns(4)
    with _sl1:
        st.markdown(f"""<div class="slab"><div class="slab-label">Bought the scariest day</div>
        <div class="slab-num">{round(_t1["pct_positive"] * _t1["n"])} of {_t1["n"]}</div>
        <div class="slab-note">crises were positive one year after the trough
        - median <b>{_t1["median"]:+.0%}</b>.</div></div>""", unsafe_allow_html=True)
    with _sl2:
        st.markdown(f"""<div class="slab"><div class="slab-label">Three years on</div>
        <div class="slab-num">{_t3["median"]:+.0%}</div>
        <div class="slab-note">median gain three years after the scariest day
        ({round(_t3["pct_positive"] * _t3["n"])} of {_t3["n"]} positive).</div></div>""",
                    unsafe_allow_html=True)
    with _sl3:
        st.markdown(f"""<div class="slab"><div class="slab-label">Worst possible timing</div>
        <div class="slab-num">{round(_p3["pct_positive"] * _p3["n"])} of {_p3["n"]}</div>
        <div class="slab-note">crises: even a buyer at the pre-crash <b>peak</b>
        was whole within three years (median {_p3["median"]:+.0%}).</div></div>""",
                    unsafe_allow_html=True)
    with _sl4:
        st.markdown(f"""<div class="slab"><div class="slab-label">The AI-capex race</div>
        <div class="slab-num">{_bwin} of {_nrace}</div>
        <div class="slab-note">crises where heavy compute investors reclaimed
        their pre-crisis level <b>faster</b> than the S&amp;P 500.</div></div>""",
                    unsafe_allow_html=True)
    st.caption(
        "Computed live from Yahoo Finance adjusted closes (S&P 500 via SPY; "
        "AI-capex basket disclosed in the Crisis Conviction tab). Historical "
        "record, not a forecast - full tables, definitions, and honest limits "
        "in the tab below."
    )
except Exception as _exc:  # noqa: BLE001 - landing page must never crash on data
    st.caption(f"Crisis record unavailable right now ({_exc}). "
               "The Crisis Conviction tab retries on load.")

st.markdown("""
<div class="showcase-section reveal" style="padding-top:8px;">
  <a href="#engine" class="cta-btn">Work with an exceptional risk engine &darr;</a>
</div>
<hr class="section-divider">
<div class="engine-heading reveal" id="engine">
  <div class="showcase-eyebrow">The Engine</div>
  <h2 class="showcase-title" style="font-size:26px;">Stress-test any portfolio, live</h2>
</div>
""", unsafe_allow_html=True)

# ---- The cockpit: controls fold into three numbered drawers so the verdict
# leads the section. Widgets still execute when collapsed - zero logic change,
# the reader just isn't bombarded with every dial at once. ----
with st.expander("01 · Universe - which assets", expanded=False):
    preset = st.selectbox("Preset basket", list(PRESETS.keys()), label_visibility="collapsed")

    # Keying the multiselect on the preset name makes it re-initialize with the
    # new default whenever the preset changes - while still letting users add or
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


# Short TTL so the session re-checks the (already freshness-aware, 6h) disk
# cache often and the UI feels snappy -- this does NOT hit Yahoo more often;
# it just re-reads the local parquet faster. See PROGRESS.md "fast polling."
@st.cache_data(ttl=60, show_spinner="Fetching market data…")
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


@st.cache_data(ttl=3600, show_spinner="Scoring the Grit Zone…")
def load_grit(tickers_tuple: tuple[str, ...]):
    """Grit scores need each asset's FULL price history, not just the 2y window
    used for VaR - grit_scores() pulls it separately (see src.grit)."""
    return grit_scores(list(tickers_tuple))


@st.cache_data(ttl=3600, show_spinner="Building the security master…")
def load_security_master(tickers_tuple: tuple[str, ...]):
    """Corporate actions change rarely (not intraday) - a longer TTL is fine."""
    return security_master(list(tickers_tuple))


@st.fragment(run_every="1s")
def _freshness_ticker(fetched_at_iso: str):
    """Live-ticking 'as of Xs ago' -- reruns only this fragment, not the app."""
    fetched = pd.Timestamp(fetched_at_iso)
    now = pd.Timestamp.now(tz=fetched.tzinfo) if fetched.tzinfo else pd.Timestamp.now()
    secs = max(0, int((now - fetched).total_seconds()))
    st.caption(f"⟳ Polling every 60s during market hours · data pulled {secs}s ago.")


# ---- Audit trail: what this run actually did, in order (see Lineage tab) ----
audit_log = []


def _audit(step: str, detail: str) -> None:
    audit_log.append({"step": step, "detail": detail})


try:
    prices = load_universe(tuple(tickers))
except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the user
    st.error(f"Couldn't load market data: {exc}")
    st.stop()

returns = get_returns(prices)
loaded = list(prices.columns)
missing = [t for t in tickers if t not in loaded]
if len(loaded) < 2:
    st.error("Fewer than two symbols returned data. Try different tickers.")
    st.stop()
if missing:
    st.caption(f"Couldn't load: {', '.join(missing)} - skipped.")
_audit("Data fetch", f"{len(loaded)} tickers loaded from {preset!r}: {', '.join(loaded)}"
      + (f" (missing: {', '.join(missing)})" if missing else ""))

# ---- Data-freshness indicator (honest, not a fake real-time feed) ----
health = data_health(prices)
fresh_col, refresh_col = st.columns([5, 1])
with fresh_col:
    fresh = "live" if health["staleness_days"] <= 1 else f"{health['staleness_days']}d old"
    st.caption(f"Data: {health['rows']} trading days · through {health['end']} · {fresh}")
    prov_now = provenance(tickers)
    if prov_now:
        _freshness_ticker(prov_now["fetched_at_utc"])
if refresh_col.button("Refresh", help="Clear cache and re-pull the latest prices."):
    clear_cache(tickers)       # drop disk cache so Yahoo is hit fresh
    st.cache_data.clear()      # drop Streamlit's in-memory cache
    st.rerun()

# ---- Allocation + stress test: one control deck, side by side ----
deck_alloc, deck_stress = st.columns(2, gap="medium")
with deck_alloc, st.expander("02 · Allocation - how capital is weighted",
                             expanded=False):
    COV_LABELS = {
        "Ledoit-Wolf": "Ledoit-Wolf - steady (default)",
        "Sample": "Sample - plain history",
        "EWMA": "EWMA - reactive / panic lens",
    }
    cov_method = st.selectbox(
        "Covariance estimator", ["Ledoit-Wolf", "Sample", "EWMA"],
        format_func=lambda m: COV_LABELS[m],
        help="How the risk matrix is built - it feeds risk parity, vol-targeting, "
             "and the Balance blend. Ledoit-Wolf (default) shrinks noisy history "
             "toward a stable target: steady, always invertible. Sample is plain "
             "history. EWMA (RiskMetrics λ=0.94) weights the last ~2 weeks heavily "
             "and forgets the calm quarter - it flinches at a single bad day. It is "
             "the reactive lens the rest of this product argues against; reach for it "
             "to SEE the panic view, not as your default.")
    cov, cov_info = estimate_covariance(returns, cov_method)  # annualized risk matrix
    st.caption(f"Risk matrix: {cov_info}.")
    if cov_method == "EWMA":
        st.caption(
            "⚠️ **Reactive lens.** EWMA spikes on one bad day (≈11-day half-life, "
            "~90% of its weight in the last month). It embodies exactly the panic "
            "[Crisis Conviction] argues against - shown for contrast, so you can see "
            "how twitchy risk looks, not because the engine recommends reacting."
        )
    acol1, acol2 = st.columns(2)
    method = acol1.radio(
        "Weighting", ["Equal weight", "Risk parity"], label_visibility="collapsed",
        help="Risk parity equalizes each asset's RISK contribution, so no single "
             "name dominates - the Bridgewater All-Weather idea.")
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
_audit("Allocation", f"{method}" + (f", vol-targeted to {target_vol:.0%} "
      f"(leverage {leverage:.2f}x)" if use_vt else ""))

# ---- Stress test: custom parametric shock OR historical regime replay ----
alloc_label = "risk-parity" if method == "Risk parity" else "equal-weight"
alloc_art = "an" if alloc_label[0] in "aeiou" else "a"  # "an equal-weight" / "a risk-parity"
lev_txt = f", levered {leverage:.2f}×" if use_vt else ""

with deck_stress, st.expander("03 · Stress test - shock or replay a crisis",
                              expanded=False):
    engine = st.radio(
        "Return model", ["Bootstrap (empirical)", "Jump-diffusion (Merton)"],
        horizontal=True,
        help="Bootstrap resamples real historical days - it can only replay tails "
             "it has already seen. Jump-diffusion (Merton 1976) adds Poisson jumps "
             "on top of Gaussian diffusion, generating NEW extremes - deeper crashes "
             "and jump clusters - for a fatter, more honest tail.")
    mode = st.selectbox(
        "Scenario", ["Custom shock (sliders)"] + list(HISTORICAL_REGIMES.keys()),
        help="Custom: set your own drawdown and volatility shock. Or replay the "
             "ACTUAL daily returns of a real crisis - real correlations, real "
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
_audit("Stress scenario", scenario_label or
      (f"Custom shock (drawdown {drawdown_shock:+d}%, vol {vol_shock:+d}%)"
       if is_shocked else "None (base case)"))
_audit("Monte Carlo", f"{engine}, 10,000 paths x 252 days -> CVaR {mc['cvar']:.2%}")

# ---- Liquidity-adjusted tail ----
# The CVaR above assumes you're out at the horizon. Widen it for the days it
# actually takes to unwind at 20% of real daily volume (default $1M book). The
# interactive version lives in the Liquidity tab; this is the headline default.
try:
    _adv = load_adv(tuple(tickers)).reindex(loaded).fillna(0.0)
    _dtl = days_to_liquidate(weights, _adv, book_value=1_000_000,
                             participation_rate=0.20)
    lvar = liquidity_adjusted_cvar(mc["cvar"],
                                   liquidity_profile(_dtl)["full_exit_days"])
except Exception:  # noqa: BLE001 - headline must still render if volume feed is down
    lvar = None

# ---- Headline verdict ----
if scenario_label:
    verdict = (
        f"Replaying the actual returns of {scenario_label} "
        f"({len(shocked_returns)} trading days), {alloc_art} {alloc_label} portfolio{lev_txt} "
        f"loses an average of <b>{mc['cvar']:.1%}</b> in the worst 5% of simulated years."
    )
    if excluded:
        verdict += f" *(Excludes {', '.join(excluded)} - not trading in that period.)*"
else:
    verdict = (
        f"In the worst 5% of simulated years, {alloc_art} {alloc_label} portfolio of these "
        f"{len(loaded)} assets{lev_txt} loses an average of <b>{mc['cvar']:.1%}</b>."
    )
    if is_shocked:
        verdict += " *(under the stress scenario applied above)*"

# Only surface the liquidity add-on when it materially fattens the tail
# (multiplier > 1.005 ≈ more than ~2.5 trading days to fully exit).
if lvar and np.isfinite(lvar["lvar"]) and lvar["multiplier"] > 1.005:
    verdict += (
        f" Adjusted for the ~<b>{lvar['full_exit_days']:.0f} trading days</b> "
        f"it takes to fully unwind at 20% of daily volume, that tail widens to "
        f"<b>{lvar['lvar']:.1%}</b>."
    )

# ---- Verdict + the cone of simulated outcomes: one wide row ----
v_col, f_col = st.columns([5, 7], gap="large")
with v_col:
    st.markdown(f"""
<div class="verdict-box">
  <div class="verdict-label">1-Year CVaR (95% confidence)</div>
  <div class="verdict-number">{mc['cvar']:.1%}</div>
  <div class="verdict-sentence">{verdict}</div>
</div>
""", unsafe_allow_html=True)
    # Legend folds away - the verdict number and chart lead; the tutorial
    # is one click for whoever wants it (matches the hide-depth doctrine).
    with st.expander("How to read the cone"):
        st.markdown(
            '<div class="read-me">'
            '<b>How to read the cone.</b> Time runs left to right - one year of '
            'trading days. The dark centreline is the <b>middle outcome</b>: half '
            'the simulations landed above it, half below. The dark inner cone holds '
            'the <b>middle 50%</b> of outcomes; the pale outer cone holds <b>90%</b>. '
            'It widens because uncertainty compounds. Its <b>bottom edge is the '
            'tail</b> the CVaR headline measures. Change any setting and watch the '
            'cone breathe.'
            '</div>', unsafe_allow_html=True)
with f_col:
    st.markdown("""
<div class="engrave line" aria-hidden="true"><svg viewBox="28 40 66 60" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#9A7B4F" stroke-linecap="round">
    <path stroke-width="2.4" d="M86,83 C72,90 54,92 44,84 C37,79 34,70 35,58"/>
    <path stroke-width="1.5" d="M35,58 C35,54 34,50 36,47"/>
    <path stroke-width="1" d="M38,52 L33,50 L39,48"/>
    <path stroke-width="1" d="M36,55 L31,55 L37,51"/>
  </g>
</svg></div>
""", unsafe_allow_html=True)
    st.plotly_chart(fan_chart(mc["path_bands"]), width="stretch", config=PLOTLY_CFG)
    _se_txt = (f" CVaR sampling error: ±{mc['cvar_se']:.2%} "
               f"({mc['n_simulations']:,} paths - a simulated estimate, "
               "not an exact truth)." if np.isfinite(mc.get("cvar_se", float("nan")))
               else "")
    st.caption(
        "Each simulated path compounds a year of daily returns. Hypothetical "
        "distribution, not a forecast - the curves interpolate between real "
        "computed percentiles." + _se_txt
    )

def eigen_factor_panel(cov, weights, returns) -> None:
    """Statistical risk factors panel (eigendecomposition / PCA).

    Separate function so the tab can wrap it in one try/except and
    degrade gracefully, matching the factor-exposures panel pattern.
    """
    fac = eigen_factors(cov)
    pc1_pct = float(fac["variance_explained"][0])
    port_pc1 = pc1_exposure(weights, fac)
    kappa = fac["condition_number"]

    e1, e2, e3 = st.columns(3)
    e1.metric("PC1 - variance explained", f"{pc1_pct:.0f}%",
              help="Share of total universe variance carried by the single "
                   "dominant statistical factor. High = one wave moves "
                   "everything.")
    e2.metric("Your book riding PC1", f"{port_pc1:.0%}",
              help="Share of THIS portfolio's variance on that dominant "
                   "factor - the macro vs idiosyncratic split.")
    e3.metric("Condition number κ", f"{kappa:,.0f}" if np.isfinite(kappa)
              else "∞ (singular)",
              help="λmax/λmin - numerical stability of the risk matrix "
                   "before any inversion. Fragile above ~1e8.")

    # Plain-English translation so a non-quant meets a sentence, not κ.
    st.markdown(
        f"**In plain terms:** one market wave drives about **{pc1_pct:.0f}%** of "
        f"this universe's day-to-day swings, and **{port_pc1:.0%}** of *your* "
        "book's risk rides that single wave. The higher that climbs, the less "
        "your diversification is actually real - in a crash it heads toward 100%.")

    read_me(
        "<b>The rubber sheet.</b> Stretch a rubber sheet and most directions "
        "bend - but a few stretch <i>straight</i>. Those unbending directions "
        "are the <b>eigenvectors</b>: the market's pure risk pathways. How "
        "hard each is stretched is its <b>eigenvalue</b> - the variance that "
        "factor carries. The decomposition untangles the correlation web into "
        "independent (orthogonal) factors, ranked by strength. Honest limit: "
        "these factors are <i>statistical and unlabeled</i> - PC1 with "
        "all-positive loadings reads as the market wave, but naming later "
        "factors is interpretation, not math. In a crisis, PC1's share spikes "
        "toward 100% - the diversification illusion collapsing into one bet.")

    # Scree chart: variance explained per factor + Marcenko-Pastur noise line
    lam = fac["eigenvalues"]
    # sigma2 excludes the top (signal) eigenvalue -- the SAME estimator
    # clip_eigenvalues uses, so this ceiling matches the module logic.
    sigma2 = float(lam[1:].mean()) if len(lam) > 1 else float(lam.mean())
    _, mp_hi = marcenko_pastur_bounds(len(lam), len(returns), sigma2)
    scree = go.Figure()
    scree.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(lam))],
        y=fac["variance_explained"],
        marker=dict(color=[BRONZE_DK if v >= mp_hi else "#CBBB94"
                           for v in lam]),
        hovertemplate="%{x}: %{y:.1f}% of variance<extra></extra>"))
    scree.add_hline(y=float(mp_hi / lam.sum() * 100) if lam.sum() > 0 else 0,
                    line=dict(color="#8A6A3C", width=1, dash="dot"),
                    annotation_text="noise ceiling (Marcenko-Pastur, heuristic)",
                    annotation_font=dict(size=11, color="#8A6A3C"))
    scree = _style_fig(scree, height=300)
    scree.update_layout(yaxis_title="% of total variance", showlegend=False)
    st.plotly_chart(scree, width="stretch", config=PLOTLY_CFG)
    st.caption(
        f"Factors above the dotted line carry more variance than pure noise "
        f"would produce at this sample size (N={len(lam)}, T={len(returns)}). "
        "Heuristic reference at this universe size, not a hard test - the "
        "Ledoit-Wolf estimator is the production defense against inversion "
        "noise. Flip the covariance estimator to EWMA in Engine controls to "
        "see the CURRENT regime's factor structure instead of the 2-year "
        "average.")

    with st.expander("Factor loadings - how each name anchors onto each factor"):
        ld = fac["loadings"]
        lmax = float(np.abs(ld.values).max()) or 1.0
        lfig = go.Figure(go.Heatmap(
            z=ld.values, x=list(ld.columns), y=list(ld.index),
            zmin=-lmax, zmax=lmax,
            colorscale=[[0.0, "#3F3B35"], [0.5, "#EDE9E3"],
                        [0.775, "#C9B48A"], [0.875, "#9A7B4F"],
                        [0.95, "#7A5426"], [1.0, "#5C3D14"]],
            xgap=2, ygap=2,
            hovertemplate="%{y} on %{x}: %{z:+.3f}<extra></extra>",
            colorbar=dict(thickness=10, outlinewidth=0)))
        lfig.update_layout(height=max(260, 34 * len(ld) + 80),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(lfig, width="stretch", config=PLOTLY_CFG)
        st.caption(
            "√λ-scaled eigenvectors, in return units: bronze = the name moves "
            "WITH the factor, charcoal = against it. Sign convention is "
            "deterministic (largest loading forced positive) so a factor "
            "hedge can never silently invert between runs. Neutralizing PC1 "
            "with an index overlay removes the dominant systematic wave "
            "without selling a single position - that is the eigen-hedge "
            "lens, shown here as exposure, not an execution engine.")


# ---- Supporting depth: one tab at a time, not stacked accordions ----
# Twelve tabs on one strip overflow invisibly (the tab-list scrollbar is
# hidden by design) - split into two ruled rows: risk analysis first,
# research & housekeeping second. Nothing removed, everything reachable.
panel_head("Risk & conviction", "The analysis - where the risk lives")
(tab_3d, tab_breakdown, tab_watch, tab_balance, tab_grit,
 tab_conviction) = st.tabs([
    "3D Distribution", "Risk Breakdown", "Correlation Watch", "Balance",
    "Grit Zone", "Crisis Conviction",
])
panel_head("Research & controls", "The workshop - signals, regimes, plumbing")
(tab_signals, tab_regimes, tab_liquidity, tab_secmaster, tab_dq,
 tab_lineage) = st.tabs([
    "Signal Lab", "Regime Atlas", "Liquidity", "Security Master",
    "Data Quality", "Lineage & Audit",
])

with tab_watch:
    # Correlation as a moving picture. A static matrix answers "are these
    # two related on average?" - this tab answers "are they related NOW,
    # and is that relationship eating my diversification?"
    corr_now = correlation_from_cov(covariance_matrix(returns))
    try:
        def_a, def_b, _ = most_correlated_pair(corr_now)
    except Exception:  # noqa: BLE001 - degenerate universe; fall back to first two
        def_a, def_b = loaded[0], loaded[1]

    wc1, wc2, wc3, wc4 = st.columns([2, 2, 2, 2])
    pick_a = wc1.selectbox("Asset A", loaded, index=loaded.index(def_a),
                           key="watch_a")
    pick_b = wc2.selectbox("Asset B", loaded, index=loaded.index(def_b),
                           key="watch_b")
    win = wc3.slider("Rolling window (days)", 10, 63, 21, step=1, key="watch_w",
                     help="21 trading days ≈ one month. Shorter reacts faster "
                          "but is noisier.")
    thresh = wc4.slider("Concentration threshold", 0.50, 0.95, 0.75, step=0.05,
                        key="watch_t",
                        help="Above this, the pair is close to one bet - "
                             "diversification between them is thinning.")

    if pick_a == pick_b:
        st.warning("Pick two different assets - a name is always +1.00 "
                   "correlated with itself.")
    else:
        roll = rolling_correlation(returns, pick_a, pick_b, window=win).dropna()
        static_corr = float(corr_now.loc[pick_a, pick_b])
        latest = float(roll.iloc[-1]) if len(roll) else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{pick_a} × {pick_b} now ({win}d)", f"{latest:+.2f}")
        m2.metric("Full-period average", f"{static_corr:+.2f}")
        m3.metric("Range over history",
                  f"{roll.min():+.2f} … {roll.max():+.2f}" if len(roll) else "-")

        if latest > thresh:
            st.warning(f"**Concentration reading:** {pick_a} and {pick_b} are "
                     f"moving at {latest:+.2f} over the last {win} trading "
                     f"days - above your {thresh:.2f} threshold. Right now "
                     "they are closer to one bet than two.")
        else:
            st.success(f"**Stable:** {pick_a} × {pick_b} at {latest:+.2f} over "
                       f"the last {win} trading days, below your "
                       f"{thresh:.2f} threshold.")

        wfig = go.Figure()
        wfig.add_hrect(y0=thresh, y1=1.0, fillcolor="rgba(154,123,79,0.10)",
                       line_width=0)
        wfig.add_hline(y=thresh, line=dict(color="#8A6A3C", width=1, dash="dot"),
                       annotation_text=f"threshold {thresh:.2f}",
                       annotation_font=dict(size=11, color="#8A6A3C"))
        wfig.add_hline(y=0, line=dict(color="#C4BDAE", width=1))
        wfig.add_trace(go.Scatter(
            x=roll.index, y=roll.values, mode="lines",
            line=dict(color=BRONZE, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}<extra></extra>",
            name=f"{pick_a} × {pick_b}"))
        wfig.update_layout(yaxis=dict(range=[-1, 1], title="correlation"),
                           showlegend=False, height=360)
        st.plotly_chart(wfig, width="stretch", config=PLOTLY_CFG)

        read_me(
            "<b>Covariance vs correlation - same sign, different units.</b> "
            "Both tell you the <i>direction</i> two assets move together. "
            "Covariance is in squared-return units, so its size is unreadable "
            "alone; correlation is covariance divided by both volatilities - "
            "co-movement per unit of risk, locked to −1…+1. The engine "
            "computes it by the matrix identity R = D⁻¹ΣD⁻¹. And it is not a "
            "constant: this line is the relationship <i>moving</i>. Pairs "
            "that average +0.4 can run above +0.9 inside a stress regime - "
            "which is exactly when you need them not to.")

        # --- Defensive simulation: measured, not promised ---
        others = [t for t in loaded if t not in (pick_a, pick_b)]
        if latest > thresh and others:
            dest, dest_corr = least_correlated_to_pair(corr_now,
                                                       (pick_a, pick_b))
            w_shift = defensive_shift(weights, loaded, (pick_a, pick_b),
                                      dest, cut=0.15)
            pr_before = portfolio_daily_returns(returns, weights)
            pr_after = portfolio_daily_returns(returns, w_shift)
            cv_b, cv_a = cvar(pr_before), cvar(pr_after)
            vol_b = float(pr_before.std() * np.sqrt(252))
            vol_a = float(pr_after.std() * np.sqrt(252))

            panel_head("Defensive simulation",
                       f"Cut {pick_a} & {pick_b} by up to 15pts each, "
                       f"move the freed weight into {dest}")
            d1, d2 = st.columns(2)
            d1.metric("Daily CVaR (95%)", f"{cv_a:.2%}",
                      delta=f"{cv_a - cv_b:+.2%} vs current",
                      delta_color="inverse")
            d2.metric("Annualized vol", f"{vol_a:.1%}",
                      delta=f"{vol_a - vol_b:+.1%} vs current",
                      delta_color="inverse")
            verdict_shift = ("reduced" if cv_a < cv_b else
                             "did NOT reduce")
            st.caption(
                f"Measured through the same engine: the shift **{verdict_shift}** "
                f"tail risk on this history. {dest} was chosen as the name "
                f"least correlated to the pair - but its own average "
                f"correlation to them is **{dest_corr:+.2f}**, not zero: "
                "inside one equity universe there is no truly independent "
                "asset, only less-dependent ones. Simulation on historical "
                "returns, not advice; correlations converge toward +1 in "
                "crashes, so measured diversification is a fair-weather "
                "number.")
        elif latest > thresh:
            st.caption("No third asset in this universe to shift into - "
                       "a two-asset book has nowhere defensive to go.")

with tab_3d:
    st.iframe(living_surface_html(mc["path_density"]), height=540)
    st.caption(
        "Simulated (Monte Carlo) distribution of portfolio value over the next "
        "year - the fan chart's cone shown as a probability surface, with a "
        "drifting particle overlay and slow auto-rotate (drag to take over). "
        "Hypothetical, not historical."
    )

with tab_breakdown:
    st.caption(f"Universe ({len(loaded)}): {', '.join(loaded)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median 1-year return", f"{mc['median_return']:+.1%}")
    c2.metric("Probability of loss", f"{mc['prob_loss']:.1%}")
    c3.metric("Worst simulated year", f"{mc['worst_case']:+.1%}")

    # --- Risk-contribution decomposition (where the risk actually lives) ---
    panel_head("Risk contribution by asset", "Where the risk actually lives")
    rc = risk_contributions(weights, cov)
    rc_fig = go.Figure()
    rc_fig.add_trace(go.Bar(
        y=list(rc.index), x=rc["weight"].values * 100, orientation="h",
        name="dollar weight", marker=dict(color="#CBBB94"),
        hovertemplate="%{y} weight: %{x:.1f}%<extra></extra>"))
    rc_fig.add_trace(go.Bar(
        y=list(rc.index), x=rc["risk_pct"].values * 100, orientation="h",
        name="risk share", marker=dict(color=BRONZE_DK),
        hovertemplate="%{y} risk: %{x:.1f}%<extra></extra>"))
    rc_fig = _style_fig(rc_fig, height=max(200, 46 * len(rc)))
    rc_fig.update_layout(
        barmode="group", xaxis_title="% of portfolio", showlegend=True,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)))
    st.plotly_chart(rc_fig, width="stretch", config=PLOTLY_CFG)
    top = rc["risk_pct"].idxmax()
    st.caption(
        f"Share of total portfolio volatility per asset. {top} contributes the most "
        f"risk ({rc.loc[top, 'risk_pct']:.0%}). Equal dollar weight ≠ equal risk - "
        "switch Allocation to Risk parity to flatten these bars."
    )

    panel_head("Risk-adjusted performance", "Sharpe vs the real T-bill rate")
    rf = load_risk_free_rate()
    ann_ret = float(port_returns.mean()) * 252
    ann_vol = float(port_returns.std()) * np.sqrt(252)
    sharpe = sharpe_ratio(port_returns, rf if rf is not None else 0.0)
    s1, s2, s3 = st.columns(3)
    s1.metric("Sharpe ratio", f"{sharpe:.2f}")
    s2.metric("Annualized return", f"{ann_ret:+.1%}")
    s3.metric("Annualized volatility", f"{ann_vol:.1%}")
    rf_txt = (f"{rf:.2%} (13-week T-bill, ^IRX)" if rf is not None
              else "unavailable - Sharpe computed against 0%")
    st.caption(
        f"Sharpe = (annualized return − risk-free) / annualized volatility, on "
        f"the real (unshocked) portfolio. Risk-free rate: {rf_txt}."
    )


    panel_head("Correlation matrix", "Do these names move together?")
    read_me(
        "Each cell is how tightly two names move together: <b>bronze = lockstep "
        "(+1)</b>, <b>beige = independent (0)</b>, <b>charcoal = seesaw (−1)</b>. "
        "A book full of deep bronze has little real diversification - everything "
        "falls at once; charcoal cells are the offsets. The empty upper half is "
        "the same data mirrored, masked so the eye reads each pair once.")
    corr = correlation_matrix(shocked_returns)

    # Lower triangle only - the upper half is a mirror image, masked out.
    cmat = corr.to_numpy(dtype=float, copy=True)
    cmat[np.triu(np.ones_like(cmat, dtype=bool))] = np.nan
    # Hot cells (|corr| >= 0.75, the Watch tab's default threshold) print bold -
    # the eye lands on concentration first.
    ctext = np.where(np.isnan(cmat), "", np.vectorize(
        lambda v: f"<b>{v:.2f}</b>" if abs(v) >= 0.75 else f"{v:.2f}")(
        np.nan_to_num(cmat)))
    hm = go.Figure(go.Heatmap(
        z=cmat, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1,
        # Furnace ramp, palette-native: charcoal seesaw -> beige independent ->
        # bronze warming -> deep molten bronze at lockstep. Heat = concentration,
        # driven by the real correlation value, nothing simulated.
        colorscale=[[0.0, "#3F3B35"], [0.5, "#EDE9E3"], [0.775, "#C9B48A"],
                    [0.875, "#9A7B4F"], [0.95, "#7A5426"], [1.0, "#5C3D14"]],
        text=ctext, texttemplate="%{text}", textfont=dict(size=11),
        hoverongaps=False, xgap=2, ygap=2,
        hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0,
                      tickvals=[-1, 0, 1], ticktext=["−1", "0", "+1"]),
    ))
    hm.update_layout(
        height=max(300, 34 * len(corr) + 90),
        yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom"))
    st.plotly_chart(hm, width="stretch", config=PLOTLY_CFG)

    panel_head("Distribution of simulated 1-year outcomes",
               "Every simulated year, sorted into buckets")
    st.plotly_chart(outcome_hist(mc["total_returns"], mc["cvar"]),
                    width="stretch", config=PLOTLY_CFG)
    if mc.get("engine") == "jump-diffusion":
        jp = mc["jump_params"]
        st.caption(
            f"Merton jump-diffusion: the engine flagged **{jp['n_jumps']} jump days** "
            f"in {jp['n_days']} (moves beyond {jp['k']:.0f}σ), implying "
            f"**~{jp['lambda_daily'] * 252:.1f} jumps/year** on a diffusion vol of "
            f"{jp['sigma_d'] * np.sqrt(252):.0%}. Poisson jumps let the tail run "
            "deeper than any single historical day - a fatter, more honest crash."
        )

    # --- Deep dive: the three statistical-test-heavy panels fold behind one
    #     click so a cold viewer meets the intuitive charts (returns, risk
    #     contribution, correlation) first, and the model-validation depth
    #     second - "lead with one number", don't wall them with seven charts. ---
    with st.expander("Deep dive - model validation & factor structure"):
        # --- VaR methods + backtest (validates the model, not just reports it) ---
        panel_head("Value at Risk - methods & backtest", "The daily loss line, and whether it holds up")
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
        panel_head("Factor exposures", "What systematic bets is this book taking?")
        try:
            fx = factor_exposures(port_returns)
            st.plotly_chart(hbar(pd.Series(fx["betas"]), color=BRONZE, title_x="beta"),
                            width="stretch", config=PLOTLY_CFG)
            st.caption(
                f"Market beta {fx['betas']['Market']:+.2f} · "
                f"R-squared {fx['r_squared']:.0%} · "
                f"annualized alpha {fx['alpha_annual']:+.1%}. "
                "Size/Value/Momentum are tilts vs. broad market (ETF-proxy factors)."
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Factor exposures unavailable: {exc}")

        # --- Statistical risk factors (eigendecomposition / PCA) ---
        panel_head("Statistical risk factors",
                   "Eigendecomposition - how many independent bets is this book?")
        try:
            eigen_factor_panel(cov, weights, returns)
        except Exception as exc:  # noqa: BLE001 - degrade like the panel above
            st.caption(f"Statistical risk factors unavailable: {exc}")

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

with tab_balance:
    st.caption(
        "The second solution. Crisis Conviction argues you shouldn't panic-sell; "
        "this argues you shouldn't have to bet the outcome on being right. Pick an "
        "asset you hold, and the engine ranks every other name in your universe by "
        "how it moves against it - a negatively-correlated partner offsets part of "
        "the anchor's swings. This is diversification, computed from real covariance, "
        "not a story about the future."
    )
    if len(loaded) < 2:
        st.caption("Balance needs at least two assets in the universe.")
    else:
        try:
            corr_b = correlation_matrix(returns)
            default_anchor = loaded[int(np.argmax(weights))]  # your biggest position
            anchor = st.selectbox(
                "Anchor (the position you want to balance)", loaded,
                index=loaded.index(default_anchor))
            ranked = rank_hedges(corr_b, anchor)

            # Classify by correlation STRENGTH, not just sign: a name at −0.01
            # is independent, not a hedge, and must not be painted as one.
            HEDGE, INDEP = -0.20, 0.20   # bands: <−0.2 offsets · ±0.2 independent

            def _hedge_color(v):
                if v < HEDGE:
                    return "#3F6B3F"          # green - genuinely moves against
                if v > INDEP:
                    return "#8A3B2E"          # red - moves with, no protection
                return "#9A7B4F"              # bronze - independent, not a hedge

            panel_head("Balancers",
                       f"How every other name moves relative to {anchor}")
            hedge_fig = go.Figure(go.Bar(
                x=ranked.values, y=list(ranked.index), orientation="h",
                marker=dict(color=[_hedge_color(v) for v in ranked.values],
                            line=dict(width=0)),
                hovertemplate="%{y}: correlation %{x:.2f}<extra></extra>"))
            hedge_fig.add_vline(x=0, line=dict(color=AXIS_LINE, width=1))
            hedge_fig = _style_fig(hedge_fig, height=max(160, 34 * len(ranked) + 40))
            hedge_fig.update_layout(xaxis_title="correlation with " + anchor)
            st.plotly_chart(hedge_fig, width="stretch", config=PLOTLY_CFG)
            st.markdown(
                '<div class="read-me"><b>Green</b> = moves <b>against</b> the anchor '
                '(correlation below −0.2) - a true offset. <b>Bronze</b> = roughly '
                '<b>independent</b> (±0.2): it diversifies but does not cancel the '
                "anchor's moves. <b>Red</b> = moves <b>with</b> it - no protection. "
                'In a single-sector basket (all tech, say) you often find no green at '
                'all - everything rises and falls together.</div>',
                unsafe_allow_html=True)

            best = ranked.index[0]
            best_corr = float(ranked.iloc[0])
            pair = min_variance_pair(cov, anchor, best)
            # Honest verdict: is this actually a hedge, or just the least-bad?
            if best_corr < HEDGE:
                kind = (f"**{best}** genuinely moves against {anchor} "
                        f"(correlation {best_corr:+.2f}) - a real hedge.")
            elif best_corr <= INDEP:
                kind = (f"No true hedge in this universe: **{best}** is the most "
                        f"**independent** name (correlation {best_corr:+.2f}), not a "
                        f"mirror. Blending it *diversifies* {anchor} - it does not "
                        f"offset it. A genuine hedge would need an asset from outside "
                        f"this basket (bonds, gold, cash).")
            else:
                kind = (f"Everything here moves **together**: even the least-correlated "
                        f"name (**{best}**, {best_corr:+.2f}) still rises and falls with "
                        f"{anchor}. This basket cannot hedge itself - a real offset "
                        f"needs an asset from a different sector or asset class.")

            panel_head("The balanced pair",
                       f"{anchor} paired with {best}, at minimum-variance weights")
            b1, b2, b3 = st.columns(3)
            b1.metric(f"Hold {anchor}", f"{pair['w_anchor']:.0%}")
            b2.metric(f"Hold {best}", f"{pair['w_hedge']:.0%}")
            b3.metric("Volatility cut", f"−{pair['vol_reduction']:.0%}")
            st.caption(
                f"{kind} Blending **{pair['w_anchor']:.0%} {anchor}** with "
                f"**{pair['w_hedge']:.0%} {best}** takes the pair's annual volatility "
                f"from **{pair['anchor_vol']:.1%}** ({anchor} alone) down to "
                f"**{pair['blended_vol']:.1%}** - a {pair['vol_reduction']:.0%} "
                "reduction, from diversification. Long-only minimum-variance weights."
            )
            # Caveat kept one click away but signposted in the title, so the
            # honesty is never buried - just not competing with the number.
            with st.expander("The honest limit - what this does NOT do"):
                st.markdown(
                    '<div class="read-me"><b>The honest limit - read this.</b> '
                    'Correlations are historical and <b>unstable</b>. In a real crash '
                    'they converge toward +1: almost everything falls together, and a '
                    'hedge that worked in calm markets fades exactly when you need it '
                    'most. This tab lowers <b>ordinary</b> volatility; it does not make a '
                    'portfolio crisis-proof. It is the counterweight to Crisis Conviction '
                    '- hold your nerve, and structure so being wrong costs less.</div>',
                    unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001 - never crash the tab
            st.caption(f"Balance unavailable for this universe: {exc}")

with tab_grit:
    st.caption(
        "Fear & Greed measures market MOOD. Grit measures something different: "
        "when a name gets knocked down, does it get back up - consistently, "
        "across real crises? There's no such thing as a perfect stock; every "
        "name here has drawdowns. This ranks your chosen universe by how much "
        "perseverance each name's OWN price history has actually shown."
    )
    try:
        grit = load_grit(tuple(loaded))
        gscores = grit["scores"]
        if gscores.empty:
            st.caption(
                "Not enough price history in this universe to score grit "
                f"(need ≥{MIN_HISTORY_DAYS} trading days per name)."
            )
        else:
            st.plotly_chart(
                hbar(gscores["grit_score"], color=BRONZE_DK, title_x="Grit Score (0–100)"),
                width="stretch", config=PLOTLY_CFG,
            )
            read_me(
                "<b>Longer bar = grittier.</b> The name at the top has, across its "
                "own history, bounced back from drawdowns the fastest and most "
                "reliably. Score is <b>0–100 relative to this basket</b> - it ranks "
                "these names against each other, not against the whole market.")
            grittiest = gscores.index[0]
            g = gscores.loc[grittiest]
            st.caption(
                f"**{grittiest}** ranks grittiest here: recovered "
                f"{g['pct_recovered']:.0%} of its own drawdowns "
                f"(median {g['median_recovery_days']:.0f} trading days to claw "
                f"back), stayed positive over {g['consistency']:.0%} of rolling "
                f"1-year holding periods, and lived through "
                f"{g['n_regimes_survived']:.0f} of the named crisis windows above."
            )

            panel_head("Grit breakdown", "Recovery · consistency · resilience")
            st.plotly_chart(grit_breakdown_fig(gscores), width="stretch", config=PLOTLY_CFG)
            st.caption(
                "Recovery: speed and completeness of clawing back from its own "
                "drawdowns (≥5%). Consistency: share of rolling 1-year holding "
                "periods that ended positive. Resilience: how shallow the "
                "drawdown and how fast the recovery across the real historical "
                "crisis windows this name actually traded through. Each bar is "
                "RANKED RELATIVE to the other names in this universe, not an "
                "absolute score - swap in a different basket and the numbers move."
            )

            if grit["excluded"]:
                st.caption(
                    f"*Excluded for insufficient history (<{MIN_HISTORY_DAYS} "
                    f"trading days): {', '.join(grit['excluded'])}.*"
                )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Grit Zone unavailable: {exc}")

# ---- Crisis Conviction: the emotional problem, priced ----
with tab_conviction:
    st.caption(
        "Buying during a crisis feels impossible because the brain treats "
        "financial loss like a physical threat - that panic is wiring, not "
        "weakness. This tab doesn't argue with the feeling. It prices it: "
        "for every named crisis this engine stress-tests, here is what "
        "actually happened next, computed live from market data."
    )
    try:
        conv = load_conviction_data()
        ctab, summ, race = conv["table"], conv["summary"], conv["race"]
        t1 = summ["trough_1y_later"]

        h1, h2, h3 = st.columns(3)
        h1.metric("Positive 1y after the trough",
                  f"{round(t1['pct_positive'] * t1['n'])} / {t1['n']} crises")
        h2.metric("Median 1y gain from the trough", f"{t1['median']:+.0%}")
        h3.metric("Median crash depth", f"{ctab['depth'].median():.0%}")

        panel_head("What a buyer actually got", "Crisis by crisis, best day vs. worst day to buy")
        show = ctab.copy()
        show.columns = ["Crisis", "Trough date", "Crash depth",
                        "Peak buy, 1y later", "Trough buy, 1y later",
                        "Peak buy, 3y later", "Trough buy, 3y later"]
        show = show[["Crisis", "Trough date", "Crash depth",
                     "Trough buy, 1y later", "Trough buy, 3y later",
                     "Peak buy, 1y later", "Peak buy, 3y later"]]
        pct_cols = [c for c in show.columns if c not in ("Crisis", "Trough date")]

        def _tone(v):
            if pd.isna(v):
                return "color: #8A8172;"
            return "color: #3F6B3F;" if v > 0 else "color: #8A3B2E;"

        # The 3 metrics above carry the message; the full 10x7 grid folds
        # so a cold viewer isn't hit with 70 raw percentages up front.
        with st.expander("Show the full table - every crisis, row by row"):
            st.dataframe(
                show.style.format({c: "{:+.0%}" for c in pct_cols}, na_rep="-")
                    .format({"Crash depth": "{:.0%}"})
                    .map(_tone, subset=pct_cols[1:]),
                width="stretch", hide_index=True)
            st.markdown(
                '<div class="read-me">'
                '<b>How to read this.</b> Each row is a real crisis. '
                '<b>Trough buy</b>: you bought the S&amp;P 500 (SPY) on the single '
                'scariest day - the exact bottom. <b>Peak buy</b>: you bought at '
                'the pre-crash top - the worst-timed entry possible. The columns '
                'show where that money stood 1 and 3 trading-years later. '
                '“-” means the crisis is too recent for that horizon: excluded, '
                'not estimated.'
                '</div>', unsafe_allow_html=True)
            st.caption(
                "Nobody can time the exact trough - that row measures the "
                "direction of the edge, not an executable strategy. That's why "
                "the peak row sits beside it: even the worst-timed buyer was "
                "usually whole within three years. The one honest exception is "
                "the dot-com peak - three years wasn't enough."
            )

        # --- The AI-capex recovery race ---
        panel_head("The recovery race", "Heavy-compute investors vs. the broad market")
        st.caption(
            f"The thesis: companies pouring capital into compute and AI "
            f"infrastructure ({', '.join(AI_CAPEX_BASKET)}, equal-weight) "
            f"recover from crises faster than the broad market. That is a "
            f"HYPOTHESIS - here is the actual record, crisis by crisis: "
            f"trading days from each side's trough back to its own "
            f"pre-crisis level."
        )
        rr = race.dropna(subset=["basket_days", "bench_days"], how="all")
        cap = RECOVERY_HORIZON_DAYS
        race_fig = go.Figure()
        race_fig.add_trace(go.Bar(
            y=rr["crisis"], x=rr["bench_days"].fillna(cap), orientation="h",
            name="S&P 500 (SPY)", marker=dict(color="#CBBB94"),
            text=[("not within 3y" if pd.isna(v) else f"{v:.0f}d")
                  for v in rr["bench_days"]],
            textposition="outside", textfont=dict(size=11),
            hovertemplate="%{y} - market: %{text}<extra></extra>"))
        race_fig.add_trace(go.Bar(
            y=rr["crisis"], x=rr["basket_days"].fillna(cap), orientation="h",
            name="AI-capex basket", marker=dict(color=BRONZE_DK),
            text=[("not within 3y" if pd.isna(v) else f"{v:.0f}d")
                  for v in rr["basket_days"]],
            textposition="outside", textfont=dict(size=11),
            hovertemplate="%{y} - basket: %{text}<extra></extra>"))
        race_fig = _style_fig(race_fig, height=max(340, 56 * len(rr) + 70))
        race_fig.update_layout(
            barmode="group", xaxis_title="trading days to reclaim pre-crisis level",
            showlegend=True,
            # headroom so "not within 3y" outside-labels never clip
            xaxis=dict(range=[0, cap * 1.18]),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)))
        st.plotly_chart(race_fig, width="stretch", config=PLOTLY_CFG)

        decided = rr
        bwin = int((decided["basket_days"].fillna(np.inf)
                    < decided["bench_days"].fillna(np.inf)).sum())
        st.markdown(
            f'<div class="read-me"><b>How to read this.</b> Shorter bar = '
            f'faster recovery. The basket got back up faster in '
            f'<b>{bwin} of {len(decided)}</b> crises. Where a bar says '
            f'"not within 3y", that side never reclaimed its pre-crisis '
            f'level inside ~3 trading years - shown, not hidden.</div>',
            unsafe_allow_html=True)
        st.caption(
            "*Honest limits: the basket carries today's \"AI capex\" label - "
            "in 2008 these names were simply large-cap tech, and the record "
            "shown is theirs regardless of the label. Members that hadn't "
            "IPO'd by a crisis are excluded from that race, not back-filled "
            "(member count varies by crisis). Survivorship is real: this "
            "basket is named WITH hindsight. One benchmark, one basket, "
            "hindsight throughout - evidence for a thesis, not proof. "
            "Educational analysis, not investment advice.*"
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Crisis Conviction unavailable: {exc}")

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


with tab_liquidity:
    lc1, lc2 = st.columns(2)
    book = lc1.number_input(
        "Portfolio size ($)", min_value=10_000, max_value=5_000_000_000,
        value=1_000_000, step=100_000,
        help="Total dollars invested. Position sizes - and so the days to unwind "
             "them - scale from this.")
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

        # Liquidity-adjusted tail - how the headline CVaR fattens once the days
        # it takes to unwind THIS book at THESE sliders are priced in.
        lv = liquidity_adjusted_cvar(mc["cvar"], prof["full_exit_days"])
        if np.isfinite(lv["lvar"]):
            d1, d2 = st.columns(2)
            d1.metric("Headline CVaR (95%)", f"{lv['cvar']:.1%}")
            d2.metric("Liquidity-adjusted CVaR", f"{lv['lvar']:.1%}",
                      delta=f"+{(lv['multiplier'] - 1):.0%} for the unwind",
                      delta_color="inverse")
            read_me(
                "<b>The tail you can't trade out of.</b> The headline CVaR "
                "assumes you're flat at the horizon. This widens it by "
                "√(1 + exit-days/252) - the Basel liquidity-horizon convention - "
                "to cover the extra days the market can move against you while "
                "you're still unwinding. Prices market exposure over the unwind, "
                "not the spread you pay to trade. A one-day-liquid book is barely "
                "penalised; a name you'd be stuck holding carries a fatter tail.")

        chart_days = dtl["days"].replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        if not chart_days.empty:
            st.plotly_chart(hbar(chart_days, color=BRONZE, title_x="days to liquidate"),
                            width="stretch", config=PLOTLY_CFG)
            read_me(
                "<b>Longer bar = harder to sell fast.</b> Each bar is how many "
                "trading days it would take to fully exit that position without "
                "being more than your chosen slice of its daily volume. Short bars "
                "are liquid; a long bar is a name you could get stuck holding in a "
                "rush for the door.")

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
                "(e.g. FX/futures on Yahoo) - excluded, not estimated.*"
            )
        st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Liquidity data unavailable: {exc}")

with tab_secmaster:
    st.caption(
        "A security master maps each ticker to stable identifiers and surfaces "
        "the real corporate-action events (splits, dividends) already folded "
        "into the adjusted-close prices used everywhere else in this engine - "
        "nothing here changes a risk number, it makes the underlying events "
        "auditable instead of silently absorbed."
    )
    try:
        sm = load_security_master(tuple(loaded))
        st.dataframe(sm, width="stretch")
        missing_isin = sm[sm["isin"] == "unavailable"].index.tolist()
        if missing_isin:
            st.caption(
                f"*ISIN unavailable on the free feed for: {', '.join(missing_isin)}. "
                "SEDOL/CUSIP and full merger history need a paid reference-data "
                "vendor (Bloomberg, Refinitiv) - not fabricated here.*"
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Security master unavailable: {exc}")

with tab_dq:
    st.caption(
        "Every price pull runs through an automated validation gate before "
        "any risk number is computed from it - schema checks, positivity, "
        "coverage, staleness, and an extreme-move flag. This validates "
        "structure and plausibility, not truth: it catches a malformed or "
        "implausible feed, not a wrong-but-plausible number."
    )
    report = validate_prices(prices)
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for c in report["checks"]:
        st.caption(f"{icon[c['status']]} **{c['check']}** - {c['message']}")
    verdict_dq = "PASS" if report["passed"] else "FAIL"
    panel_head("Overall data-quality gate", f"This feed: {verdict_dq}")

with tab_lineage:
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
            "Yahoo Finance. Every figure above is computed from this source by "
            "the engine's own code - no value originates from a language model. "
            "Use Refresh to re-pull and update this timestamp."
        )
    else:
        st.caption("Provenance record appears after the first live fetch.")

    panel_head("This run's audit trail", "Every step this session actually took, in order")
    st.caption(
        "Every step this run took, in order - session-scoped (rebuilt fresh "
        "each rerun, not persisted across sessions). A full compliance system "
        "would append this to durable storage; this is the same concept at "
        "the scale this engine actually operates at."
    )
    st.dataframe(pd.DataFrame(audit_log), width="stretch", hide_index=True)

# ---- Signal Lab: does a simple signal actually carry information? ----
with tab_signals:
    # Method greeting folds to one line so a cold viewer meets the three
    # headline numbers first, not a paragraph of academic definition.
    with st.expander("What the Information Coefficient measures"):
        st.caption(
            "The information coefficient (IC) is the daily cross-sectional Spearman "
            "rank correlation between a signal's ranking of this universe and the "
            "forward returns that actually followed. Demo signal: 60-day momentum "
            "skipping the most recent 5 days (to avoid short-term reversal), scored "
            "against 5-day forward returns - computed from the same live price "
            "history as everything above."
        )
    try:
        SIG_HORIZON = 5
        ic = daily_ic(momentum_signal(prices),
                      forward_returns(prices, horizon=SIG_HORIZON))
        summ = ic_summary(ic)

        if summ["n_days"] < 30 or not np.isfinite(summ["t_stat"]):
            st.caption(
                "Not enough overlapping history in this universe to evaluate "
                f"the signal ({summ['n_days']} usable days - need at least 30)."
            )
        else:
            i1, i2, i3 = st.columns(3)
            i1.metric("Mean daily IC", f"{summ['mean_ic']:+.3f}")
            i2.metric("t-statistic", f"{summ['t_stat']:.2f}")
            i3.metric("Hit rate (IC > 0)", f"{summ['hit_rate']:.0%}")

            t = summ["t_stat"]
            if t >= 3:
                bar_txt = ("clears both the textbook t > 2 bar and the stricter "
                           "t > 3 multiple-testing bar of Harvey, Liu & Zhu (2016)")
            elif t >= 2:
                bar_txt = ("clears the textbook t > 2 bar but NOT the t > 3 bar "
                           "Harvey, Liu & Zhu (2016) argue for once you account "
                           "for the thousands of signals the industry has already "
                           "tested - by that stricter standard, unproven")
            else:
                bar_txt = ("clears neither the textbook t > 2 bar nor the "
                           "stricter t > 3 multiple-testing bar of Harvey, Liu "
                           "& Zhu (2016) - statistically indistinguishable from "
                           "no skill on this sample")
            st.caption(
                f"In-sample, this momentum signal's mean IC of "
                f"{summ['mean_ic']:+.3f} (t = {t:.2f}, scored over "
                f"{summ['n_days']} days) **{bar_txt}**."
            )

            roll = ic.rolling(63).mean()
            ic_fig = go.Figure()
            ic_fig.add_trace(go.Scatter(
                x=ic.index, y=ic.values, mode="lines", name="daily IC",
                line=dict(color="#CBBB94", width=1),
                hovertemplate="%{x|%b %d, %Y}: IC %{y:+.2f}<extra>daily</extra>"))
            ic_fig.add_trace(go.Scatter(
                x=roll.index, y=roll.values, mode="lines", name="63-day mean",
                line=dict(color=BRONZE_DK, width=2.2),
                hovertemplate="%{x|%b %d, %Y}: %{y:+.3f}<extra>63-day mean</extra>"))
            ic_fig.add_hline(y=0, line=dict(color=AXIS_LINE, width=1, dash="dot"))
            ic_fig = _style_fig(ic_fig, height=280)
            ic_fig.update_layout(
                yaxis_title="Spearman IC", showlegend=True,
                legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)))
            st.plotly_chart(ic_fig, width="stretch", config=PLOTLY_CFG)
            st.caption(
                "Daily IC is noisy by nature - the 63-day rolling mean is the "
                "signal's actual pulse. Above zero: the ranking carried "
                "information that quarter; below: it was actively wrong."
            )

            panel_head("Grinold's fundamental law", "IR = IC × √breadth")
            rebalances = 252 / SIG_HORIZON
            raw_breadth = len(loaded) * rebalances
            n_eff = effective_breadth(returns)
            eff_breadth = n_eff * rebalances
            g1, g2 = st.columns(2)
            g1.metric(f"IR at raw breadth ({raw_breadth:.0f} bets/yr)",
                      f"{fundamental_law_ir(summ['mean_ic'], raw_breadth):.2f}")
            g2.metric(f"IR at effective breadth ({eff_breadth:.0f} bets/yr)",
                      f"{fundamental_law_ir(summ['mean_ic'], eff_breadth):.2f}")
            st.caption(
                f"Raw breadth counts {len(loaded)} names × {rebalances:.0f} "
                f"rebalances a year as independent bets, but average pairwise "
                f"correlation collapses these {len(loaded)} names to about "
                f"**{n_eff:.1f} independent bets** - correlated stocks are "
                f"largely the same bet taken twice, so the honest IR is the "
                f"smaller one."
            )

            st.caption(
                "*Disclosures: everything here is IN-SAMPLE on the loaded "
                "history - the signal is scored on the same data used to "
                "evaluate it. Momentum is a demo signal, not a recommendation. "
                "No transaction costs or market impact. Published signals decay "
                "out of sample. Educational analysis, not investment advice.*"
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Signal Lab unavailable: {exc}")

# ---- Regime Atlas: Wasserstein k-means on full return distributions ----
with tab_regimes:
    # Method + citation fold to one line so the current-regime verdict leads.
    with st.expander("Method & source - Wasserstein regime clustering"):
        st.caption(
            "Reproduces Horvath, Issa & Muguruza (2021), *Clustering Market "
            "Regimes using the Wasserstein Distance*: every 20-day window of this "
            "portfolio's daily returns becomes an empirical distribution, and "
            "k-means clusters those whole distributions (via the 1-D optimal-"
            "transport closed form) rather than summary features - so regimes "
            "that share volatility but differ in tails or skew still separate."
        )
    try:
        REG_WINDOW, REG_STEP = 20, 5
        k_reg = st.selectbox("Number of regimes (k)", [2, 3, 4], index=1)
        Q_reg, reg_ends = rolling_windows(port_returns,
                                          window=REG_WINDOW, step=REG_STEP)
        if Q_reg.shape[0] < max(30, k_reg * 5):
            st.caption(
                f"Not enough portfolio history to cluster regimes "
                f"({Q_reg.shape[0]} windows - need at least {max(30, k_reg * 5)})."
            )
        else:
            reg_labels = vol_ordered_labels(
                Q_reg, wasserstein_kmeans(Q_reg, k=k_reg)[0])
            reg_rows = regime_stats(Q_reg, reg_labels)
            cur = reg_rows[int(reg_labels[-1])]
            REGIME_WORDS = {2: ["calm", "turbulent"],
                            3: ["calm", "transitional", "turbulent"],
                            4: ["calm", "mild", "stressed", "turbulent"]}
            word = REGIME_WORDS[k_reg][cur["label"]]
            r1, r2, r3 = st.columns(3)
            r1.metric("Current regime",
                      f"{cur['label'] + 1} of {k_reg} - {word}")
            r2.metric("Regime ann. vol", f"{cur['ann_vol']:.1%}")
            r3.metric("Regime CVaR (95%)", f"{cur['cvar_95']:.2%}")

            # regime timeline: one dot per window end-date, shaded by regime
            REG_COLORS = ["#CBBB94", "#B8946A", "#8A6A3C", "#5A4526"][:k_reg]
            reg_fig = go.Figure()
            for j in range(k_reg):
                mask = reg_labels == j
                reg_fig.add_trace(go.Scatter(
                    x=reg_ends[mask], y=reg_labels[mask] + 1, mode="markers",
                    name=f"regime {j + 1} ({REGIME_WORDS[k_reg][j]})",
                    marker=dict(color=REG_COLORS[j], size=7, symbol="square"),
                    hovertemplate="%{x|%b %d, %Y}<extra>regime "
                                  f"{j + 1}</extra>"))
            reg_fig = _style_fig(reg_fig, height=220)
            reg_fig.update_layout(
                yaxis=dict(title="regime", dtick=1,
                           range=[0.5, k_reg + 0.5]),
                showlegend=True,
                legend=dict(orientation="h", y=1.2, x=0, font=dict(size=11)))
            st.plotly_chart(reg_fig, width="stretch", config=PLOTLY_CFG)

            # Two dense grids fold together; the plain-English "sticky %"
            # caption below stays visible as the actual takeaway.
            P_reg = transition_matrix(reg_labels, k_reg)
            with st.expander("Cluster detail - profiles & transition matrix"):
                panel_head("Regime profiles", "Vol, skew, tail per cluster")
                reg_table = pd.DataFrame(reg_rows).set_index("label")
                reg_table.index = [f"regime {i + 1}" for i in reg_table.index]
                reg_table.columns = ["windows", "ann. vol", "mean daily",
                                     "skew", "CVaR 95%"]
                st.dataframe(reg_table.style.format({
                    "ann. vol": "{:.1%}", "mean daily": "{:+.4%}",
                    "skew": "{:+.2f}", "CVaR 95%": "{:.2%}"}),
                    width="stretch")

                panel_head("Transition matrix", "Where the next window goes")
                pt = pd.DataFrame(
                    P_reg,
                    index=[f"from {i + 1}" for i in range(k_reg)],
                    columns=[f"to {i + 1}" for i in range(k_reg)])
                st.dataframe(pt.style.format("{:.0%}"), width="stretch")
            stay = float(np.mean(np.diag(P_reg)))
            st.caption(
                f"Transition matrix, estimated from consecutive windows: "
                f"regimes are sticky - on average a {stay:.0%} chance the "
                f"next window stays in the current regime. Labels are "
                f"in-sample statistical clusters over {Q_reg.shape[0]} "
                f"windows ({REG_WINDOW}-day, step {REG_STEP}), ordered "
                f"calm→turbulent by volatility; k is a user choice, not "
                f"estimated. Educational reproduction of published research, "
                f"not investment advice."
            )
    except Exception as exc:  # graceful, like the other tabs
        st.caption(f"Regime Atlas unavailable for this universe: {exc}")

# ---- MCAP-style closing band: where-to-next rail + honest copyright bar ----
st.markdown("""
<div class="meleona-footer">
  <div class="f-rail">
    <a class="f-box" href="#grit-showcase"><span class="f-num">01</span>The Grit Zone
      <small>Resilience ranked from real drawdowns - recovery, consistency, crisis behavior.</small></a>
    <a class="f-box" href="#conviction"><span class="f-num">02</span>Crisis Conviction
      <small>The hardest trade, priced from the actual record of ten crises.</small></a>
    <a class="f-box" href="#engine"><span class="f-num">03</span>The Engine
      <small>Stress-test any universe live - allocation, scenarios, CVaR verdict.</small></a>
  </div>
  <div class="f-bar">
    <div>Meleona &middot; Portfolio Risk Engine &middot; &copy; 2026 John Nguyen</div>
    <div>Live end-of-day data: Yahoo Finance &middot; Educational analysis, not investment advice</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Book-glide: eased anchor scrolling on the REAL scroll container ----
# Streamlit scrolls its own <section>, so `scroll-behavior` on <html> never
# fires - anchor clicks teleported. This zero-height component reaches into
# the parent document, intercepts CTA anchor clicks, and drives a 1.1s
# eased glide (easeInOutCubic) with a mid-scroll arrival animation on the
# destination - a page turn, not a teleport. Guarded so Streamlit reruns
# never stack duplicate listeners.
components.html("""
<script>
(function() {
  const P = window.parent.document;
  if (P.__meleonaGlide) return;          // rerun guard: bind once per page
  P.__meleonaGlide = true;
  const ease = t => t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;
  /* Streamlit's scroll container has moved between releases - never trust
     a hardcoded selector. Walk UP from the destination to the first
     ancestor that really scrolls (proved by a nudge test). */
  function findScroller(el) {
    let n = el.parentElement;
    while (n) {
      /* Demand a REAL page scroller (hundreds of px of travel). A tiny
         accidental overflow (e.g. a full-bleed band adding a few px to an
         inner container) must not hijack the walk - that bug froze every
         CTA at 1px of movement. */
      if (n.scrollHeight > n.clientHeight + 120) {
        /* The scroller carries `scroll-behavior: smooth` (our CSS fallback),
           which makes a programmatic scrollTop ASYNC - an immediate read-back
           sees no movement and this test wrongly rejected the real scroller.
           Force instant behavior for the probe, restore after. */
        const prevSB = n.style.scrollBehavior;
        n.style.scrollBehavior = 'auto';
        const was = n.scrollTop;
        n.scrollTop = was + 1;
        const ok = n.scrollTop !== was;
        n.scrollTop = was;
        n.style.scrollBehavior = prevSB;
        if (ok) return n;
      }
      n = n.parentElement;
    }
    return P.scrollingElement;
  }
  function targetY(scroller, el) {
    return el.getBoundingClientRect().top -
           scroller.getBoundingClientRect().top + scroller.scrollTop - 26;
  }
  function glide(scroller, el, dur, settled) {
    /* Our rAF drives every frame - the scroller's own smooth behavior would
       fight it (each scrollTo becoming its own animation). Instant while we
       fly, restored when we land. */
    const prevSB = scroller.style.scrollBehavior;
    scroller.style.scrollBehavior = 'auto';
    const y0 = scroller.scrollTop, d = targetY(scroller, el) - y0,
          t0 = performance.now();
    (function f(now) {
      const p = Math.min(1, (now - t0) / dur);
      scroller.scrollTo(0, y0 + d * ease(p));
      if (p < 1) { requestAnimationFrame(f); return; }
      scroller.style.scrollBehavior = prevSB;
      /* landing check: if the page shifted mid-flight (a chart mounted,
         a rerun repainted), re-aim once with a short corrective glide -
         the reader always ends ON the section the button promised. */
      const drift = targetY(scroller, el) - scroller.scrollTop;
      if (!settled && Math.abs(drift) > 4) glide(scroller, el, 320, true);
    })(t0);
  }
  P.addEventListener('click', function(e) {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const el = P.getElementById(a.getAttribute('href').slice(1));
    if (!el) return;
    const scroller = findScroller(el);
    if (!scroller) return;               // nothing scrolls: let native run
    e.preventDefault(); e.stopPropagation();
    glide(scroller, el, 1100, false);
    // destination rises into place as the glide lands
    const dest = el.clientHeight === 0 ? el.nextElementSibling : el;
    if (dest) {
      dest.style.animation = 'none'; void dest.offsetWidth;
      dest.style.animation = 'section-arrive .9s cubic-bezier(.16,1,.3,1) .45s both';
    }
    /* Arrival theatrics, one per destination: the EARTHBENDER launch fires
       only for the first CTA (#grit-showcase); #conviction gets the quiet
       engraver's ring instead - same band, different gesture. */
    const href = a.getAttribute('href');
    const row = el.closest('.showcase-row');
    if (row && href === '#grit-showcase') {
      row.classList.remove('band-arrive'); void row.offsetWidth;
      row.classList.add('band-arrive');
    }
    if (href === '#conviction') {
      el.classList.remove('ring-arrive'); void el.offsetWidth;
      el.classList.add('ring-arrive');
    }
  }, true);
})();
</script>
""", height=0)
