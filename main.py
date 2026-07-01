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

st.set_page_config(page_title="Meleona", layout="centered")

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

/* ---- Presentation flow: hero, showcase, CTAs, scroll reveal ---- */
html { scroll-behavior: smooth; }

@keyframes meleona-rise { from { opacity: 0; transform: translateY(24px); }
                          to   { opacity: 1; transform: translateY(0); } }
.reveal { animation: meleona-rise linear both;
          animation-timeline: view();
          animation-range: entry 0% cover 30%; }

.hero-section { min-height: 74vh; display: flex; flex-direction: column;
    justify-content: center; align-items: center; text-align: center;
    padding: 40px 16px; gap: 16px; }
.hero-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #8A6E45; }
.hero-title { font-size: 56px; color: #3F3B35; line-height: 1.05; margin: 4px 0; }
.hero-sub { font-size: 19px; color: #54504A; max-width: 620px; line-height: 1.55; }

.cta-btn { display: inline-block; margin-top: 10px; padding: 14px 30px;
    background: #3F3B35; color: #F4F1EA !important; text-decoration: none !important;
    border-radius: 4px; font-family: 'Helvetica Neue', sans-serif; font-size: 13px;
    letter-spacing: 0.1em; text-transform: uppercase;
    transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
    box-shadow: 0 2px 8px rgba(63,59,53,0.18); }
.cta-btn:hover { transform: translateY(-2px); background: #2E2B27;
    box-shadow: 0 6px 16px rgba(63,59,53,0.28); }

.showcase-section { padding: 56px 8px 40px; text-align: center; display: flex;
    flex-direction: column; align-items: center; gap: 14px; }
.showcase-eyebrow { font-family: 'Helvetica Neue', sans-serif; font-size: 12px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #8A6E45; }
.showcase-title { font-size: 34px; color: #3F3B35; margin: 0; font-weight: 400; }
.showcase-body { font-size: 16px; color: #54504A; max-width: 640px; line-height: 1.6; }

.pillar-row { display: flex; gap: 18px; flex-wrap: wrap; justify-content: center;
    margin-top: 8px; }
.pillar-card { background: #F4F1EA; border: 1px solid #BFB8A9; border-radius: 8px;
    padding: 20px 22px; width: 210px; text-align: left;
    transition: transform 0.25s ease, box-shadow 0.25s ease; }
.pillar-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(63,59,53,0.15); }
.pillar-label { font-family: 'Helvetica Neue', sans-serif; font-size: 11px;
    letter-spacing: 0.12em; text-transform: uppercase; color: #9A7B4F; margin-bottom: 6px; }
.pillar-desc { font-size: 13.5px; color: #524E47; line-height: 1.5; }

.section-divider { border: none; border-top: 1px solid #BFB8A9; margin: 8px 0 36px;
    opacity: 0.6; }
.engine-heading { text-align: center; padding: 4px 0 26px; }

/* Tabs — quieter than a stack of accordions */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; }
[data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'Helvetica Neue', sans-serif;
    font-size: 12.5px; letter-spacing: 0.04em; color: #7A6E5A; }
[data-testid="stTabs"] [aria-selected="true"] { color: #3F3B35 !important; }
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
    """Monte Carlo outcome cone: median path + 25–75 and 5–95 percentile bands."""
    d = bands["days"]

    def p(a):
        return np.asarray(a) * 100.0

    fig = go.Figure()
    # Outer 5–95 cone (draw upper first, then lower with fill-to-previous)
    fig.add_trace(go.Scatter(x=d, y=p(bands["p95"]), line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p5"]), fill="tonexty", fillcolor=BAND_OUTER,
                             line=dict(width=0), hoverinfo="skip"))
    # Inner 25–75 cone
    fig.add_trace(go.Scatter(x=d, y=p(bands["p75"]), line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p25"]), fill="tonexty", fillcolor=BAND_INNER,
                             line=dict(width=0), hoverinfo="skip"))
    # Median path
    fig.add_trace(go.Scatter(x=d, y=p(bands["p50"]), line=dict(color=CHARCOAL, width=2.2),
                             hovertemplate="Day %{x}: %{y:.1f}%<extra>median</extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS_LINE, width=1, dash="dot"))
    fig.update_layout(xaxis_title="Trading days", yaxis_title="Cumulative return (%)")
    return _style_fig(fig, height=300)


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

      setInterval(function() {{
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
      }}, 150);
    }});
}})();
</script>
"""


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

# ---- Hero visual: the cone of simulated outcomes ----
st.plotly_chart(fan_chart(mc["path_bands"]), width="stretch", config=PLOTLY_CFG)
st.caption(
    "Each simulated path compounds a year of daily returns. The dark line is the "
    "median outcome; the shaded cones are the 25–75% and 5–95% ranges — the lower "
    "edge is the tail the CVaR above measures. Change any setting to watch the cone move."
)

# ---- Supporting depth: one tab at a time, not stacked accordions ----
(tab_3d, tab_breakdown, tab_grit, tab_liquidity,
 tab_secmaster, tab_dq, tab_lineage) = st.tabs([
    "3D Distribution", "Risk Breakdown", "Grit Zone", "Liquidity",
    "Security Master", "Data Quality", "Lineage & Audit",
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
    st.markdown("###### Risk contribution by asset")
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

    st.markdown("###### Correlation matrix")
    corr = correlation_matrix(shocked_returns)

    def beige_scale(val):
        # higher correlation -> deeper warm gray, no matplotlib needed
        shade = int(245 - max(0.0, min(1.0, val)) * 90)
        text = "#4A4640" if val < 0.7 else "#FFFFFF"
        return f"background-color: rgb({shade},{shade-6},{shade-14}); color: {text};"

    st.dataframe(corr.style.format("{:.2f}").map(beige_scale))

    st.markdown("###### Distribution of simulated 1-year outcomes")
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

            st.markdown("###### Grit breakdown: recovery, consistency, resilience")
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
    st.markdown(f"###### Overall gate: **{verdict_dq}**")

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

    st.markdown("###### This run's audit trail")
    st.caption(
        "Every step this run took, in order — session-scoped (rebuilt fresh "
        "each rerun, not persisted across sessions). A full compliance system "
        "would append this to durable storage; this is the same concept at "
        "the scale this engine actually operates at."
    )
    st.dataframe(pd.DataFrame(audit_log), width="stretch", hide_index=True)
