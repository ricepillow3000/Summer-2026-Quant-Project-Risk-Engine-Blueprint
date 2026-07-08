"""
Meleona — Streamlit entry point.

Design philosophy:
A risk desk doesn't hand a PM eight charts and say "figure it out." It leads
with one verdict and one number. Everything else is detail you open on demand.

Phase V:
The universe is now chosen by the viewer, not hard-coded. Anyone can load a
preset basket (equities, sector ETFs, FX, futures) or type their own symbols,
so the engine speaks to any audience — not just one watchlist.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

st.set_page_config(page_title="Meleona", layout="centered")

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
st.markdown("""
<style>
/* ============================================================
   "THE TEARSHEET" — private-bank editorial design language.
   Doctrine: sharp edges (no rounded pills), hairline bronze rules,
   extreme type contrast (huge serif numerals vs tiny tracked labels),
   numbered ruled sections. Numbers are king; craft gives it soul.
   ============================================================ */
html, body, [class*="css"] { font-family: Georgia, 'Times New Roman', serif; }
h1, h2, h3 { color: #3F3B35; font-weight: 400; letter-spacing: -0.01em; }
/* Sharpen the whole app — kill Streamlit's default rounded corners */
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

/* HERO VERDICT — the editorial centerpiece. No card: a stat framed by
   bronze hairlines, the one number that owns the page. */
.verdict-box { background: transparent; border: none;
    border-top: 2px solid #9A7B4F; border-bottom: 1px solid #C4BDAE;
    padding: 30px 4px 36px; margin: 12px 0 44px; }
.verdict-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.22em; text-transform: uppercase; color: #9A7B4F; }
.verdict-number { font-size: 96px; color: #3F3B35; margin: 8px 0 4px;
    line-height: 1; letter-spacing: -0.035em; font-weight: 400; }
.verdict-sentence { font-size: 18px; color: #54504A; line-height: 1.6; max-width: 580px; }

/* Numbered section eyebrow — editorial ledger markers (01 — UNIVERSE) */
.sec-mark { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.24em; text-transform: uppercase; color: #9A7B4F;
    border-top: 1px solid #C4BDAE; padding-top: 14px; margin: 36px 0 14px;
    display: flex; align-items: baseline; gap: 12px; }
.sec-mark b { color: #B7A98E; font-weight: 400; }
.panel-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.16em; text-transform: uppercase; color: #8A7E6A;
    margin-bottom: 4px; }

/* Slider — squared thumb, hairline track */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #C4BDAE !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #9A7B4F !important; border-radius: 0 !important;
    border: 2px solid #F4F1EA !important;
    box-shadow: 0 1px 3px rgba(63,59,53,0.25) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #3F3B35 !important; font-family: 'Helvetica Neue', sans-serif !important;
    font-size: 12px !important; }

/* Expander — flat cream, sharp, hairline */
[data-testid="stExpander"] { border: 1px solid #C4BDAE !important;
    background: #ECE7DD !important; }

/* ---- Presentation flow: hero, showcase, CTAs, scroll reveal ---- */
html { scroll-behavior: smooth; }
@keyframes meleona-rise { from { opacity: 0; transform: translateY(24px); }
                          to   { opacity: 1; transform: translateY(0); } }
.reveal { animation: meleona-rise linear both;
          animation-timeline: view(); animation-range: entry 0% cover 30%; }

.hero-section { min-height: 88vh; display: flex; flex-direction: column;
    justify-content: center; align-items: flex-start; text-align: left;
    padding: 56px 8px; gap: 18px; border-bottom: 1px solid #C4BDAE; }
.hero-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.32em; text-transform: uppercase; color: #9A7B4F; }
.hero-title { font-size: 96px !important; color: #3F3B35;
    line-height: 0.98 !important; margin: 2px 0; letter-spacing: -0.035em; }
.hero-sub { font-size: 21px; color: #54504A; max-width: 640px; line-height: 1.6; }

/* CTA — sharp charcoal slab, bronze on hover */
.cta-btn { display: inline-block; margin-top: 14px; padding: 15px 34px;
    background: #3F3B35; color: #F4F1EA !important; text-decoration: none !important;
    border-radius: 0; font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.16em; text-transform: uppercase;
    transition: background 0.25s ease, letter-spacing 0.25s ease; }
.cta-btn:hover { background: #9A7B4F; letter-spacing: 0.2em; }

.showcase-section { padding: 96px 8px 64px; text-align: left; display: flex;
    flex-direction: column; align-items: flex-start; gap: 18px; }
.showcase-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.28em; text-transform: uppercase; color: #9A7B4F; }
.showcase-title { font-size: 52px; color: #3F3B35; margin: 0; font-weight: 400;
    letter-spacing: -0.025em; line-height: 1.05; }
.showcase-body { font-size: 16px; color: #54504A; max-width: 620px; line-height: 1.6; }

/* Pillars — no cards: ledger columns divided by bronze hairlines */
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

.section-divider { border: none; border-top: 1px solid #C4BDAE; margin: 8px 0 36px; }
.engine-heading { text-align: left; padding: 4px 0 22px; }

/* ============================================================
   TABS — "the gatehouse". Each label is a stone lintel: generous
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
/* bronze rule grows from the centre — no jump, no flash */
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
   THE KEEP — stone-slab surfaces. Sharp corners, hairline mortar,
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

/* Ruled panel header inside a tab — the lintel over each block */
.panel-head { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    border-top: 1px solid #C4BDAE; padding-top: 13px; margin: 34px 0 14px; }
.panel-head .t { font-family: 'Helvetica Neue', sans-serif; font-size: 11.5px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #3F3B35;
    white-space: nowrap; }
.panel-head .s { font-size: 13px; color: #8A8172; line-height: 1.5; }

/* Charts glide in with the panel instead of popping */
[data-testid="stPlotlyChart"], [data-testid="stIFrame"] {
    animation: panel-settle .5s cubic-bezier(.16,1,.3,1) both;
    animation-delay: .06s; }

/* Metric slabs pick up the same masonry */
[data-testid="stMetric"] { background: #F1EDE5; border: 1px solid #D4CDBF;
    border-top: 2px solid #9A7B4F; padding: 16px 18px 12px;
    transition: transform .3s cubic-bezier(.16,1,.3,1), border-color .3s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #9A7B4F; }

/* Buttons: cut stone, bronze on press */
.stButton>button { font-family: 'Helvetica Neue', sans-serif; font-size: 11.5px;
    letter-spacing: 0.16em; text-transform: uppercase; border: 1px solid #C4BDAE;
    background: #F1EDE5; color: #3F3B35;
    transition: background .25s ease, border-color .25s ease, letter-spacing .25s ease; }
.stButton>button:hover { background: #3F3B35; color: #F4F1EA;
    border-color: #3F3B35; letter-spacing: 0.2em; }

@media (prefers-reduced-motion: reduce) {
    *, *::after { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)

# ---- Themed Plotly palette + chart helpers (institutional beige/bronze) ----
BRONZE = "#9A7B4F"
BRONZE_DK = "#8A6A3C"
CHARCOAL = "#3F3B35"
BAND_OUTER = "rgba(154,123,79,0.14)"   # light bronze — 5–95 percentile cone
BAND_INNER = "rgba(154,123,79,0.30)"   # medium bronze — 25–75 percentile cone
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
        hoverlabel=dict(bgcolor="#F4F1EA", font=dict(family="Georgia, serif", color=CHARCOAL)),
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
    point is a real computed percentile from the simulation — the spline only
    interpolates *between* those points instead of connecting them with jagged
    straight segments. Hover reports the true underlying value, so nothing is
    smoothed away from the numbers themselves; only the ink between them.
    """
    d = bands["days"]

    def p(a):
        return np.asarray(a) * 100.0

    # One curve style for every edge of the cone — laminar, not stair-stepped.
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
    # Hairline edges trace the envelope — the silhouette of the airflow
    for key in ("p95", "p5"):
        fig.add_trace(go.Scatter(x=d, y=p(bands[key]), hoverinfo="skip",
                                 line=dict(color="rgba(154,123,79,0.45)", width=1,
                                           shape="spline", smoothing=1.0)))
    # Median path — the centreline, drawn last so it sits on top
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
    horizon — the fan chart's cone re-expressed as a probability surface
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
<div id="living3d" style="width:100%;height:{height}px;"></div>
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
    height: {height}, margin: {{l:0,r:0,t:10,b:0}},
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {{ family: "Georgia, 'Times New Roman', serif", color: '#3F3B35', size: 12 }},
    scene: {{
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: {{ title: 'Trading day', gridcolor: 'rgba(63,59,53,0.12)',
               backgroundcolor: 'rgba(237,233,227,0.35)', showbackground: true }},
      yaxis: {{ title: '1-year outcome', tickformat: '.0%', gridcolor: 'rgba(63,59,53,0.12)',
               backgroundcolor: 'rgba(237,233,227,0.35)', showbackground: true }},
      zaxis: {{ title: 'Density', gridcolor: 'rgba(63,59,53,0.12)',
               backgroundcolor: 'rgba(237,233,227,0.35)', showbackground: true }},
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

      /* One animation step: drift the particles, ease the camera round. */
      const step = function() {{
        t += 1;
        const n = data.px.length;
        const nx = new Array(n), ny = new Array(n), nz = new Array(n);
        for (let i = 0; i < n; i++) {{
          nx[i] = data.px[i] + Math.sin(t * 0.2 + i * 1.7) * 3;
          ny[i] = data.py[i] + Math.cos(t * 0.25 + i * 2.3) * 0.01;
          nz[i] = Math.max(0, data.pz[i] + Math.sin(t * 0.35 + i) * 0.012);
        }}
        Plotly.restyle('living3d', {{x: [nx], y: [ny], z: [nz]}}, [1]);
        if (!userInteracting) {{
          const angle = t * 0.01;
          Plotly.relayout('living3d', {{
            'scene.camera.eye.x': 1.6 * Math.cos(angle),
            'scene.camera.eye.y': 1.6 * Math.sin(angle),
          }});
        }}
      }};

      /* Only animate while this scene is actually on screen. Streamlit keeps
         inactive tab panels mounted, so an ungated interval would repaint the
         surface forever on every other tab — burning CPU and never letting the
         renderer idle. Gate on both intersection and page visibility. */
      /* Respect the OS reduced-motion setting — draw once, hold still. This is
         a hard gate, checked inside shouldRun, because the IntersectionObserver
         callback fires asynchronously and would otherwise restart the loop. */
      const reduced = !!(window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);

      let timer = null;
      const running = () => timer !== null;
      const start = function() {{ if (!running()) timer = setInterval(step, 150); }};
      const stop = function() {{ if (running()) {{ clearInterval(timer); timer = null; }} }};
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
    """Ruled section lintel inside a tab — replaces bare `###### ` markdown
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
    """Themed histogram of simulated 1-year outcomes with the CVaR line marked."""
    fig = go.Figure(go.Histogram(x=np.asarray(total_returns) * 100, nbinsx=48,
                                 marker=dict(color=BRONZE, line=dict(width=0)), opacity=0.9))
    fig.add_vline(x=-cvar * 100, line=dict(color=CHARCOAL, width=2, dash="dash"),
                  annotation_text="CVaR", annotation_position="top left",
                  annotation_font=dict(color=CHARCOAL, size=12))
    fig.update_layout(xaxis_title="1-year return (%)", yaxis_title="Simulations")
    return _style_fig(fig, height=280)


def hbar(series: pd.Series, color=BRONZE, pct: bool = False, title_x: str = ""):
    """Themed horizontal bar chart for a single labeled series (factors, etc.)."""
    x = series.values * (100 if pct else 1)
    fig = go.Figure(go.Bar(
        x=x, y=list(series.index), orientation="h",
        marker=dict(color=color),
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

st.markdown(f"""
<div class="hero-section reveal" id="hero">
  <div style="width:84px; height:84px;">{logo_svg}</div>
  <div class="hero-eyebrow">Pride &middot; Integrity</div>
  <h1 class="hero-title">Meleona</h1>
  <div class="hero-sub">
    A hedge-fund-grade portfolio risk engine — VaR, CVaR, Monte Carlo stress
    testing, and named factor exposures, computed live from real market data.
    But every stock has drawdowns. What sets a name apart is what happens
    after one — that's what we call <strong>grit</strong>.
  </div>
  <a href="#grit-showcase" class="cta-btn">Explore what we do &darr;</a>
</div>
""", unsafe_allow_html=True)

# ---- Showcase: the Grit Zone innovation, explained before you touch a slider ----
st.markdown("""
<div class="showcase-section reveal" id="grit-showcase">
  <div class="showcase-eyebrow">The Innovation</div>
  <h2 class="showcase-title">Introducing the Grit Zone</h2>
  <div class="showcase-body">
    Fear &amp; Greed indices measure market MOOD. We measure something more
    durable: whether an asset, when it gets knocked down, actually gets back
    up &mdash; consistently, across real crises. There's no such thing as a
    perfect stock. Grit isn't about avoiding setbacks &mdash; it's about what
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
  <a href="#conviction" class="cta-btn">See the hardest trade &darr;</a>
</div>
""", unsafe_allow_html=True)


# ---- Showcase: Crisis Conviction — the emotional problem, answered in numbers ----
@st.cache_data(ttl=6 * 3600, show_spinner="Reading the crisis record…")
def load_conviction_data():
    """Benchmark crisis record + AI-capex recovery race, live from Yahoo."""
    return load_conviction()


st.markdown("""
<div class="showcase-section reveal" id="conviction" style="padding-bottom:24px;">
  <div class="showcase-eyebrow">The Conviction</div>
  <h2 class="showcase-title">The hardest trade is the one history rewards</h2>
  <div class="showcase-body">
    Your brain treats a falling portfolio the way it treats a physical threat
    &mdash; the panic you feel in a crash is wiring, not weakness. That is the
    emotional problem this engine exists to solve. Not with a slogan: with the
    actual record of every named crisis it stress-tests, computed live from
    market data. Below, what really happened to a buyer on the scariest day of
    each crisis &mdash; and on the worst-timed day, the pre-crash peak.
  </div>
</div>
""", unsafe_allow_html=True)

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
        — median <b>{_t1["median"]:+.0%}</b>.</div></div>""", unsafe_allow_html=True)
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
        "record, not a forecast — full tables, definitions, and honest limits "
        "in the tab below."
    )
except Exception as _exc:  # noqa: BLE001 — landing page must never crash on data
    st.caption(f"Crisis record unavailable right now ({_exc}). "
               "The Crisis Conviction tab retries on load.")

st.markdown("""
<div class="showcase-section reveal" style="padding-top:8px;">
  <a href="#engine" class="cta-btn">Work with an exceptional risk engine &darr;</a>
</div>
<hr class="section-divider">
<div id="engine"></div>
<div class="engine-heading reveal">
  <div class="showcase-eyebrow">The Engine</div>
  <h2 class="showcase-title" style="font-size:26px;">Stress-test any portfolio, live</h2>
</div>
""", unsafe_allow_html=True)

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
    used for VaR — grit_scores() pulls it separately (see src.grit)."""
    return grit_scores(list(tickers_tuple))


@st.cache_data(ttl=3600, show_spinner="Building the security master…")
def load_security_master(tickers_tuple: tuple[str, ...]):
    """Corporate actions change rarely (not intraday) — a longer TTL is fine."""
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
_audit("Allocation", f"{method}" + (f", vol-targeted to {target_vol:.0%} "
      f"(leverage {leverage:.2f}x)" if use_vt else ""))

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
_audit("Stress scenario", scenario_label or
      (f"Custom shock (drawdown {drawdown_shock:+d}%, vol {vol_shock:+d}%)"
       if is_shocked else "None (base case)"))
_audit("Monte Carlo", f"{engine}, 10,000 paths x 252 days -> CVaR {mc['cvar']:.2%}")

# ---- Headline verdict ----
if scenario_label:
    verdict = (
        f"Replaying the actual returns of {scenario_label} "
        f"({len(shocked_returns)} trading days), a {alloc_label} portfolio{lev_txt} "
        f"loses an average of <b>{mc['cvar']:.1%}</b> in the worst 5% of simulated years."
    )
    if excluded:
        verdict += f" *(Excludes {', '.join(excluded)} — not trading in that period.)*"
else:
    verdict = (
        f"In the worst 5% of simulated years, a {alloc_label} portfolio of these "
        f"{len(loaded)} assets{lev_txt} loses an average of <b>{mc['cvar']:.1%}</b>."
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

# ---- Hero visual: the cone of simulated outcomes ----
st.plotly_chart(fan_chart(mc["path_bands"]), width="stretch", config=PLOTLY_CFG)
st.markdown(
    '<div class="read-me">'
    '<b>How to read this.</b> Time runs left to right — one year of trading days. '
    'The dark centreline is the <b>middle outcome</b>: half the simulations landed '
    'above it, half below. The dark inner cone holds the <b>middle 50%</b> of '
    'outcomes; the pale outer cone holds <b>90%</b>. The cone widens because '
    'uncertainty compounds the further out you look. Its <b>bottom edge is the '
    'tail</b> the CVaR headline above measures. Change any setting and watch the '
    'cone breathe.'
    '</div>', unsafe_allow_html=True)
st.caption(
    "Each simulated path compounds a year of daily returns. Hypothetical "
    "distribution, not a forecast — the curves interpolate between real computed "
    "percentiles."
)

# ---- Supporting depth: one tab at a time, not stacked accordions ----
(tab_3d, tab_breakdown, tab_grit, tab_conviction, tab_liquidity,
 tab_secmaster, tab_dq, tab_lineage, tab_signals, tab_regimes) = st.tabs([
    "3D Distribution", "Risk Breakdown", "Grit Zone", "Crisis Conviction",
    "Liquidity", "Security Master", "Data Quality", "Lineage & Audit",
    "Signal Lab", "Regime Atlas",
])

with tab_3d:
    st.iframe(living_surface_html(mc["path_density"]), height=540)
    st.caption(
        "Simulated (Monte Carlo) distribution of portfolio value over the next "
        "year — the fan chart's cone shown as a probability surface, with a "
        "drifting particle overlay and slow auto-rotate (drag to take over). "
        "Hypothetical, not historical."
    )

with tab_breakdown:
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
        f"risk ({rc.loc[top, 'risk_pct']:.0%}). Equal dollar weight ≠ equal risk — "
        "switch Allocation to Risk parity to flatten these bars."
    )

    panel_head("Correlation matrix", "Do these names move together?")
    read_me(
        "Each cell is how tightly two names move together, from <b>0</b> (independent) "
        "to <b>1</b> (lockstep). <b>Darker = more correlated.</b> A book full of dark "
        "cells has little real diversification — everything falls at once.")
    corr = correlation_matrix(shocked_returns)

    def beige_scale(val):
        # higher correlation -> deeper warm gray, no matplotlib needed
        shade = int(245 - max(0.0, min(1.0, val)) * 90)
        text = "#4A4640" if val < 0.7 else "#FFFFFF"
        return f"background-color: rgb({shade},{shade-6},{shade-14}); color: {text};"

    st.dataframe(corr.style.format("{:.2f}").map(beige_scale))

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
            "deeper than any single historical day — a fatter, more honest crash."
        )

    # --- VaR methods + backtest (validates the model, not just reports it) ---
    panel_head("Value at Risk — methods & backtest", "The daily loss line, and whether it holds up")
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

with tab_grit:
    st.caption(
        "Fear & Greed measures market MOOD. Grit measures something different: "
        "when a name gets knocked down, does it get back up — consistently, "
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
                "reliably. Score is <b>0–100 relative to this basket</b> — it ranks "
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
                "absolute score — swap in a different basket and the numbers move."
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
        "financial loss like a physical threat — that panic is wiring, not "
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

        st.dataframe(
            show.style.format({c: "{:+.0%}" for c in pct_cols}, na_rep="—")
                .format({"Crash depth": "{:.0%}"})
                .map(_tone, subset=pct_cols[1:]),
            width="stretch", hide_index=True)
        st.markdown(
            '<div class="read-me">'
            '<b>How to read this.</b> Each row is a real crisis. '
            '<b>Trough buy</b>: you bought the S&amp;P 500 (SPY) on the single '
            'scariest day — the exact bottom. <b>Peak buy</b>: you bought at '
            'the pre-crash top — the worst-timed entry possible. The columns '
            'show where that money stood 1 and 3 trading-years later. '
            '“—” means the crisis is too recent for that horizon: excluded, '
            'not estimated.'
            '</div>', unsafe_allow_html=True)
        st.caption(
            "Nobody can time the exact trough — that row measures the "
            "direction of the edge, not an executable strategy. That's why "
            "the peak row sits beside it: even the worst-timed buyer was "
            "usually whole within three years. The one honest exception is "
            "the dot-com peak — three years wasn't enough."
        )

        # --- The AI-capex recovery race ---
        panel_head("The recovery race", "Heavy-compute investors vs. the broad market")
        st.caption(
            f"The thesis: companies pouring capital into compute and AI "
            f"infrastructure ({', '.join(AI_CAPEX_BASKET)}, equal-weight) "
            f"recover from crises faster than the broad market. That is a "
            f"HYPOTHESIS — here is the actual record, crisis by crisis: "
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
            hovertemplate="%{y} — market: %{text}<extra></extra>"))
        race_fig.add_trace(go.Bar(
            y=rr["crisis"], x=rr["basket_days"].fillna(cap), orientation="h",
            name="AI-capex basket", marker=dict(color=BRONZE_DK),
            text=[("not within 3y" if pd.isna(v) else f"{v:.0f}d")
                  for v in rr["basket_days"]],
            textposition="outside", textfont=dict(size=11),
            hovertemplate="%{y} — basket: %{text}<extra></extra>"))
        race_fig = _style_fig(race_fig, height=max(300, 40 * len(rr) + 60))
        race_fig.update_layout(
            barmode="group", xaxis_title="trading days to reclaim pre-crisis level",
            showlegend=True,
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
            f'level inside ~3 trading years — shown, not hidden.</div>',
            unsafe_allow_html=True)
        st.caption(
            "*Honest limits: the basket carries today's \"AI capex\" label — "
            "in 2008 these names were simply large-cap tech, and the record "
            "shown is theirs regardless of the label. Members that hadn't "
            "IPO'd by a crisis are excluded from that race, not back-filled "
            "(member count varies by crisis). Survivorship is real: this "
            "basket is named WITH hindsight. One benchmark, one basket, "
            "hindsight throughout — evidence for a thesis, not proof. "
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
                "(e.g. FX/futures on Yahoo) — excluded, not estimated.*"
            )
        st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Liquidity data unavailable: {exc}")

with tab_secmaster:
    st.caption(
        "A security master maps each ticker to stable identifiers and surfaces "
        "the real corporate-action events (splits, dividends) already folded "
        "into the adjusted-close prices used everywhere else in this engine — "
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
                "vendor (Bloomberg, Refinitiv) — not fabricated here.*"
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Security master unavailable: {exc}")

with tab_dq:
    st.caption(
        "Every price pull runs through an automated validation gate before "
        "any risk number is computed from it — schema checks, positivity, "
        "coverage, staleness, and an extreme-move flag. This validates "
        "structure and plausibility, not truth: it catches a malformed or "
        "implausible feed, not a wrong-but-plausible number."
    )
    report = validate_prices(prices)
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for c in report["checks"]:
        st.caption(f"{icon[c['status']]} **{c['check']}** — {c['message']}")
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
            "the engine's own code — no value originates from a language model. "
            "Use Refresh to re-pull and update this timestamp."
        )
    else:
        st.caption("Provenance record appears after the first live fetch.")

    panel_head("This run's audit trail", "Every step this session actually took, in order")
    st.caption(
        "Every step this run took, in order — session-scoped (rebuilt fresh "
        "each rerun, not persisted across sessions). A full compliance system "
        "would append this to durable storage; this is the same concept at "
        "the scale this engine actually operates at."
    )
    st.dataframe(pd.DataFrame(audit_log), width="stretch", hide_index=True)

# ---- Signal Lab: does a simple signal actually carry information? ----
with tab_signals:
    st.caption(
        "The information coefficient (IC) is the daily cross-sectional Spearman "
        "rank correlation between a signal's ranking of this universe and the "
        "forward returns that actually followed. Demo signal: 60-day momentum "
        "skipping the most recent 5 days (to avoid short-term reversal), scored "
        "against 5-day forward returns — computed from the same live price "
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
                f"the signal ({summ['n_days']} usable days — need at least 30)."
            )
        else:
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Mean daily IC", f"{summ['mean_ic']:+.3f}")
            i2.metric("t-statistic", f"{summ['t_stat']:.2f}")
            i3.metric("Hit rate (IC > 0)", f"{summ['hit_rate']:.0%}")
            i4.metric("Days scored", f"{summ['n_days']}")

            t = summ["t_stat"]
            if t >= 3:
                bar_txt = ("clears both the textbook t > 2 bar and the stricter "
                           "t > 3 multiple-testing bar of Harvey, Liu & Zhu (2016)")
            elif t >= 2:
                bar_txt = ("clears the textbook t > 2 bar but NOT the t > 3 bar "
                           "Harvey, Liu & Zhu (2016) argue for once you account "
                           "for the thousands of signals the industry has already "
                           "tested — by that stricter standard, unproven")
            else:
                bar_txt = ("clears neither the textbook t > 2 bar nor the "
                           "stricter t > 3 multiple-testing bar of Harvey, Liu "
                           "& Zhu (2016) — statistically indistinguishable from "
                           "no skill on this sample")
            st.caption(
                f"In-sample, this momentum signal's mean IC of "
                f"{summ['mean_ic']:+.3f} (t = {t:.2f}) **{bar_txt}**."
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
                "Daily IC is noisy by nature — the 63-day rolling mean is the "
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
                f"**{n_eff:.1f} independent bets** — correlated stocks are "
                f"largely the same bet taken twice, so the honest IR is the "
                f"smaller one."
            )

            st.caption(
                "*Disclosures: everything here is IN-SAMPLE on the loaded "
                "history — the signal is scored on the same data used to "
                "evaluate it. Momentum is a demo signal, not a recommendation. "
                "No transaction costs or market impact. Published signals decay "
                "out of sample. Educational analysis, not investment advice.*"
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Signal Lab unavailable: {exc}")

# ---- Regime Atlas: Wasserstein k-means on full return distributions ----
with tab_regimes:
    st.caption(
        "Reproduces Horvath, Issa & Muguruza (2021), *Clustering Market "
        "Regimes using the Wasserstein Distance*: every 20-day window of this "
        "portfolio's daily returns becomes an empirical distribution, and "
        "k-means clusters those whole distributions (via the 1-D optimal-"
        "transport closed form) rather than summary features — so regimes "
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
                f"({Q_reg.shape[0]} windows — need at least {max(30, k_reg * 5)})."
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
                      f"{cur['label'] + 1} of {k_reg} — {word}")
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

            reg_table = pd.DataFrame(reg_rows).set_index("label")
            reg_table.index = [f"regime {i + 1}" for i in reg_table.index]
            reg_table.columns = ["windows", "ann. vol", "mean daily",
                                 "skew", "CVaR 95%"]
            st.dataframe(reg_table.style.format({
                "ann. vol": "{:.1%}", "mean daily": "{:+.4%}",
                "skew": "{:+.2f}", "CVaR 95%": "{:.2%}"}),
                width="stretch")

            P_reg = transition_matrix(reg_labels, k_reg)
            pt = pd.DataFrame(
                P_reg,
                index=[f"from {i + 1}" for i in range(k_reg)],
                columns=[f"to {i + 1}" for i in range(k_reg)])
            st.dataframe(pt.style.format("{:.0%}"), width="stretch")
            stay = float(np.mean(np.diag(P_reg)))
            st.caption(
                f"Transition matrix, estimated from consecutive windows: "
                f"regimes are sticky — on average a {stay:.0%} chance the "
                f"next window stays in the current regime. Labels are "
                f"in-sample statistical clusters over {Q_reg.shape[0]} "
                f"windows ({REG_WINDOW}-day, step {REG_STEP}), ordered "
                f"calm→turbulent by volatility; k is a user choice, not "
                f"estimated. Educational reproduction of published research, "
                f"not investment advice."
            )
    except Exception as exc:  # graceful, like the other tabs
        st.caption(f"Regime Atlas unavailable for this universe: {exc}")
