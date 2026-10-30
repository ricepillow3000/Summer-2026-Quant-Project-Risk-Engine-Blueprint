"""
Meliona - Streamlit entry point.

Design philosophy:
A risk desk doesn't hand a PM eight charts and say "figure it out." It leads
with one verdict and one number. Everything else is detail you open on demand.

Phase V:
The universe is now chosen by the viewer, not hard-coded. Anyone can load a
preset basket (equities, sector ETFs, FX, futures) or type their own symbols,
so the engine speaks to any audience - not just one watchlist.
"""

import base64
from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.app_session import AppSession
import plotly.graph_objects as go

from src.ingestion import (
    fetch_prices, get_returns, data_health, provenance, CACHE_MAX_AGE_HOURS,
    average_dollar_volume, fetch_dollar_volume, fetch_risk_free_rate,
    PRESETS, valid_ticker, NoDataError, flat_columns,
    MAX_UNIVERSE, MIN_ROWS, SUGGESTIONS,
)
from src.analytics import correlation_matrix, covariance_matrix
from src.risk import (
    monte_carlo, jump_diffusion_mc, parametric_var, var_backtest, sharpe_ratio,
    cvar, var, portfolio_daily_returns, mcneil_frey_tail,
)
from src.comovement import (
    correlation_from_cov, rolling_correlation, most_correlated_pair,
    defensive_shift, least_correlated_to_pair, cluster_order,
)
from src.factors import factor_exposures
from src.narrative import (DEFAULT_PARTICIPATION, build_book, headline, role_label,
                           role_note, Role)
from src.strategies import risk_contributions, risk_parity_weights, vol_target
from src.hedge import min_variance_pair, rank_hedges
from src.pairing import (anchor_rank, backtest_pair, crisis_cushion,
                         es_confidence_interval, pair_weights, regime_labels,
                         tail_gap, DEFENSIVE_ANCHOR_TICKERS)
from src.covariance import estimate_covariance, RISKMETRICS_LAMBDA, TRADING_DAYS
from src.topology import anchor_frame, build_map_payload, war_room_html
from src.observability import (host_setting, log_incident, log_session_start,
                               new_session_ref, setup_logging)
from src.eigenrisk import eigen_factors, marcenko_pastur_bounds, pc1_exposure
from src.scenarios import HISTORICAL_REGIMES, replay_returns
from hero import N_PATHS, hero_html
from src.liquidity import (days_to_liquidate, liquidity_profile,
                           liquidity_adjusted_cvar)
from src.grit import grit_scores, MIN_HISTORY_DAYS
from src.data_quality import validate_prices
from src.conviction import (
    load_conviction, AI_CAPEX_BASKET, HORIZONS, RECOVERY_HORIZON_DAYS, race_days,
    race_record,
)

# Title and icon match the <head> tools/public_files.py serves to crawlers and
# link previews, so a shared link and the browser tab say the same thing.
st.set_page_config(page_title="Meliona - Portfolio Risk Engine",
                   page_icon=str(Path(__file__).parent / "public" / "favicon.png"),
                   layout="wide")


# toolbarMode="viewer" hides the "Clear cache" menu, but the websocket message
# behind it is still honoured: Streamlit's AppSession answers a `clear_cache`
# BackMsg from ANY client by emptying st.cache_data and st.cache_resource for
# every visitor in this process. 2026-10-15 security audit: a scripted client
# looping that message pins the single process recomputing cold pages. This
# app has no visitor action that should clear shared state, so the handler
# becomes a no-op; operators restart the process or run tools.warm_cache.
# The name is Streamlit-private and checked against the pinned ~=1.66.1
# (requirements.txt) by test_a_crafted_clear_cache_message_clears_nothing.
# Re-assigning on every rerun is idempotent.
def _ignore_clear_cache_request(self) -> None:
    return None


AppSession._handle_clear_cache_request = _ignore_clear_cache_request

# ---- Operations: kill switch, crash wire, session reference ----------------
# SUPPORT_EMAIL is the one place the contact address lives; the footer, the
# maintenance page, the feedback link and the privacy panel all read it from
# here. A project mailbox rather than a personal one: it is published on a
# public page, so it will attract spam, and a dedicated address can be
# forwarded, filtered or handed on without touching a personal inbox.
# MELIONA_SUPPORT_EMAIL overrides it per environment without a redeploy.
SUPPORT_EMAIL = host_setting("SUPPORT_EMAIL", "meleona.support@gmail.com")

# Release channel, for the phased rollout in RUNBOOK.md section 5. Setting
# MELIONA_CHANNEL=beta on the preview/beta deploy marks the page as a beta and
# invites feedback; production leaves it unset and shows nothing.
RELEASE_CHANNEL = host_setting("CHANNEL").strip().lower()

# Kill switch. Set MELIONA_MAINTENANCE=1 in the host's environment and the next
# page load serves this notice instead of the engine - no redeploy, no code
# change, and it takes effect for everyone at once. MELIONA_MAINTENANCE_NOTE
# optionally carries a one-line reason for visitors.
if host_setting("MAINTENANCE").strip().lower() in {"1", "true", "yes", "on"}:
    st.title("Meliona is briefly offline")
    st.write(host_setting("MAINTENANCE_NOTE")
             or "The engine is paused for maintenance. Please check back shortly.")
    st.caption(f"Questions: {SUPPORT_EMAIL}")
    st.stop()

# Crash wire: attach the rotating log handler once, mint a per-session
# reference, and record the session start. The reference shows in the footer so
# a visitor reporting a problem can quote it and the operator can grep for that
# exact session - the same correlation a hosted crash reporter gives, without
# shipping anything about the visitor to a third party.
setup_logging()
if "session_ref" not in st.session_state:
    st.session_state.session_ref = new_session_ref()
    log_session_start(st.session_state.session_ref)
SESSION_REF = st.session_state.session_ref


def safe_err(where: str, exc: BaseException) -> str:
    """What a visitor may see about a failure: the exception TYPE and their
    session reference - never str(exc).

    str(exc) routinely carries the absolute server path (FileNotFoundError and
    PermissionError on a cache file both embed it), which is precisely what
    `showErrorDetails = false` in .streamlit/config.toml exists to keep off the
    page. The full text and traceback still reach the operator through
    log_incident(); the ref is how support ties the two together.
    """
    log_incident(SESSION_REF, where, exc)
    return f"{type(exc).__name__} - quote {SESSION_REF} if you report this"

# One prefilled mailto, reused by the beta banner and the privacy panel: the
# subject already carries the session reference, so a report arrives with the
# one thing that makes it diagnosable (see RUNBOOK.md section 3).
FEEDBACK_LINK = (f"mailto:{SUPPORT_EMAIL}?subject=Meliona%20feedback%20"
                 f"%E2%80%93%20session%20{SESSION_REF}")

if RELEASE_CHANNEL == "beta":
    st.info(
        f"**Beta.** This is a pre-release deployment - features and numbers "
        f"may change, and data is live *end-of-day*, never real-time. Found "
        f"something wrong? [Email the session]({FEEDBACK_LINK}) - reference "
        f"`{SESSION_REF}` is already in the subject line.",
        icon=":material/science:",
    )

# ---- Minimal institutional styling ----
# Page background, slider color, and expander shade are set in .streamlit/config.toml.
def load_css() -> None:
    """The design system, loaded once from static/app.css instead of 778 lines
    of inline <style> re-sent on every rerun."""
    css = (Path(__file__).parent / "static" / "app.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()

# ---- Book-glide: eased anchor scrolling on the REAL scroll container ----
# Streamlit scrolls its own <section>, so `scroll-behavior` on <html> never
# fires - anchor clicks teleported. This zero-height component reaches into
# the parent document, intercepts in-page anchor clicks, and drives one eased
# glide to the destination - a scroll, not a teleport and not a slide change.
# tools/deep_scan.mjs times every button's flight frame by frame.
#
# HERE, right after load_css(), not at the end of the script (John, 2026-10-19:
# "sometimes the button does not transition"): at the end, every load left the
# buttons live but unwired for 2.7-4.6 s, and a run ending in st.stop() never
# sent the frame - Streamlit removed it, its listeners died, and the old
# bind-once guard kept the next frame from binding until a reload. A new frame
# now replaces the old listeners instead (no duplicates either way).
#
# Through st.iframe (a same-origin srcdoc frame), not the deprecated
# components.html, which warned on every run. st.iframe refuses height 0, so
# the frame is 1px inside a keyed container static/app.css folds to nothing.
def _script_frame(script: str) -> None:
    with st.container(key="glide"):
        st.iframe(script, height=1)


_script_frame("""
<script>
(function() {
  const P = window.parent.document;
  const prev = P.__melionaGlide;         // a frame from an earlier run, maybe dead
  if (prev && prev.onTab) {
    P.removeEventListener('click', prev.onTab, true);
    P.removeEventListener('click', prev.onLink, true);
    if (prev.onYield) ['wheel', 'touchstart', 'keydown'].forEach(function (t) {
      P.removeEventListener(t, prev.onYield, true);
    });
    if (prev.onScroll) P.removeEventListener('scroll', prev.onScroll, true);
  }

  /* Charts inside a tab or expander are rendered while their container is
     HIDDEN, so it measures 0px wide and Plotly falls back to its 700px
     default - which then sits ~90px short inside a 790px box and leaves a
     gap. Plotly's responsive mode only re-lays out on a WINDOW resize, and
     revealing a tab fires no such event, so the stale width sticks.
     Dispatching one resize after the reveal makes every chart re-fit its
     real container. Verified: 700px -> 790px exact fit. */
  function onTab(e) {
    if (e.target.closest && e.target.closest('[role="tab"], summary')) {
      /* Twice on purpose: the first nudge catches the common case, the second
         covers containers still settling their width when the first fires
         (measured a chart mid-settle at 700px in a 755px box). Two resize
         events are cheap - Plotly only re-lays out figures whose box moved. */
      [140, 620].forEach(function(d) {
        setTimeout(function() {
          window.parent.dispatchEvent(new Event('resize'));
        }, d);
      });
    }
  }
  /* easeInOutSine: the flattest speed profile of the standard in-out curves.
     Its fastest frame is ~1.6x the average; the quart it replaced peaked at
     4x, which on the footer's 6000px flights meant ~270px per frame mid-way -
     a lurch, not a glide. Both ends still settle softly. */
  const ease = t => -(Math.cos(Math.PI * t) - 1) / 2;
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
        /* Nudge BOTH ways. At the bottom of the page - exactly where the
           footer's links live - a +1 nudge cannot move, the real scroller
           failed the test, and the footer links were dead (deep scan). */
        const was = n.scrollTop;
        n.scrollTop = was + 1;
        if (n.scrollTop === was) n.scrollTop = was - 1;
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
  /* Every frame's scroll re-sent the Risk Topology map's five canvases to the
     compositor, even 4300px off screen: 42% of the main thread at 4x CPU,
     20 fps (John, 2026-10-19: "very laggy"). While a flight runs, the map's
     box skips rendering when off screen - and renders as usual if the flight
     passes it. Only while flying: a permanent rule would pause the map's own
     loops and, on a phone, clip its full-bleed frame (paint containment), so
     phones (<= 640px, where that bleed lives) are left alone. */
  function quietMap(on) {
    const m = P.querySelector('.st-key-risk-map');
    if (!m) return;
    const q = on && window.parent.innerWidth > 640;
    m.style.contentVisibility = q ? 'auto' : '';
    m.style.containIntrinsicBlockSize = q ? 'auto 820px' : '';
  }
  /* Hand scrolling back: the stylesheet's smooth behaviour (never a saved
     inline value - a second flight would save the first one's 'auto'). */
  function land(scroller) {
    scroller.style.scrollBehavior = '';
    scroller.style.overflowAnchor = '';
    quietMap(false);
    P.__melionaStop = null;
    P.__melionaScroller = null;
    P.__melionaDir = 0;
  }
  /* Every way a flight or a wait ends goes through its stop: a newer press
     used to bump the flight number and leave the old flight's 'auto'
     scrolling and the map's skipped rendering behind (Council B's leak
     probe, 2026-10-22). */
  function supersede() {
    if (P.__melionaStop) P.__melionaStop();
  }
  /* Is Streamlit still writing the page? Its running icon (the one app.css
     draws "Updating..." from) is mounted exactly while a run is in flight. */
  const pageRunning = () => !!P.querySelector('[data-testid="stStatusWidgetRunningIcon"]');
  function glide(scroller, el) {
    /* One flight at a time: a new press, or the reader's own wheel / touch /
       key (onYield), bumps the flight number and the old frames stop. Two
       presses used to fly two flights at once - a 59px jump and a wrong
       landing. */
    supersede();
    const id = P.__melionaFlight = (P.__melionaFlight || 0) + 1;
    P.__melionaFlightAt = performance.now();
    P.__melionaStop = function () { land(scroller); };
    P.__melionaScroller = scroller;
    /* Our rAF drives every frame - the scroller's own smooth behavior would
       fight it (each scrollTo becoming its own animation). Instant while we
       fly, handed back when we land. */
    scroller.style.scrollBehavior = 'auto';
    /* No scroll anchoring while we fly: the flight follows its target itself,
       and an anchoring jump would read as the reader scrolling (onScroll). */
    scroller.style.overflowAnchor = 'none';
    quietMap(true);
    /* Calm over speed: 1500ms on a slow-out curve reads as being LED
       somewhere, where a fast jump reads as a slide change. */
    const dur = 1500, y0 = scroller.scrollTop, t0 = performance.now();
    /* Progress advances at most 50 ms per frame. Driven by the clock, one
       frame that came late (charts mounting while the page builds) covered
       66-81% of the flight - a jump, not a glide. */
    let aim = targetY(scroller, el), last = t0, run = 0, done = 0, grew = t0;
    let floor = scroller.scrollHeight - scroller.clientHeight;
    const dir = Math.sign(aim - y0) || 1;
    P.__melionaDir = dir;
    /* No frame moves faster than the curve's own peak (pi/2 the average
       speed, so 2x with margin). A page still loading is shorter than the
       flight: scrollTo clamped the eased position at its bottom and the next
       mount released it all at once - 138-230px of a 475px flight in one
       frame, 6 of 8 presses during load (ME-04). Now a page that grows under
       the flight is caught up at the flight's own pace. */
    const vmax = 2 * Math.max(Math.abs(aim - y0), 600) / dur;
    (function f(now) {
      if (id !== P.__melionaFlight) return;   // a newer flight, or the reader took over
      const dt = Math.min(now - last, 50); run += dt; last = now;
      const p = Math.min(1, run / dur);
      const tgt = targetY(scroller, el), cur = scroller.scrollTop;
      /* Re-aim against the page's CURRENT bottom every frame. */
      const max = scroller.scrollHeight - scroller.clientHeight;
      if (max > floor + 1) { floor = max; grew = now; }
      /* The destination can still be moving: the showcase row rises 24px on
         its scroll-driven reveal while we fly, so a target read once at the
         click landed 8-11px past it and crawled back. Follow it - at most 2px
         a frame further away, so a chart mounting mid-flight cannot yank the
         page; at once when it comes closer (a rerun repainting the page above
         it), so the flight never runs past it (1 of 4 basket switches
         overshot "Skip to the risk map" by 220px). */
      const d = tgt - aim;
      if (p >= 1 || Math.sign(d) !== dir) aim = tgt;
      else aim += Math.min(2, Math.abs(d)) * dir;
      const cap = vmax * dt + 4;
      let step = (p < 1 ? y0 + (aim - y0) * ease(p) : aim) - cur;
      /* Behind the curve (the page was too short) or past its end: close in
         at no more than peak speed, softly over the last stretch - but by a
         whole pixel at least: scrollTop snaps to device pixels, and a 0.5px
         step was rounded away, so a flight 3px short of its target kept
         re-asserting itself for 10 s (it dragged the page back 1 time in 10
         when the next press moved it). */
      if (p >= 1 || Math.abs(step) > cap)
        step = Math.sign(aim - cur) * Math.min(cap, Math.max(1, Math.abs(aim - cur) * 0.15));
      if (Math.sign(step) !== dir && Math.sign(tgt - cur) === dir) step = 0;   // never back away from a target still ahead
      scroller.scrollTo(0, Math.max(0, Math.min(cur + step, max)));
      P.__melionaSetY = scroller.scrollTop;    // where WE put it (onScroll)
      if (p < 1) { requestAnimationFrame(f); return; }
      /* Settle until the reader is ON the section the button promised. One
         320 ms correction was not enough on a page still loading: the hero
         CTA stopped 223px short at the bottom of a 1371px page, 8 of 8
         (ME-04). While the target sits below the page's end, wait for the
         page to grow: until its footer (the last thing the run writes) has
         arrived, or while it still grows - at most 10 s. A quiet spell alone
         is no sign of the end: the cold harness paused 3 s between mounts
         and the flight landed 416px short just before the page grew. And
         the 10 s is a floor, not a cap, while the run goes on: on a slow cold
         start the page below a late section (#analysis) kept growing for
         longer than that (ME-05, third review). */
      if (!done) done = now;
      const on = Math.abs(tgt - scroller.scrollTop) <= 2;
      const waiting = tgt > max + 2 &&
                      (!P.querySelector('.meliona-footer') || now - grew < 3000);
      if (!on && (now - done < 10000 || (waiting && pageRunning())) &&
          (waiting || Math.abs(Math.max(0, Math.min(tgt, max)) - scroller.scrollTop) > 2)) {
        requestAnimationFrame(f); return;
      }
      land(scroller);
    })(t0);
  }
  /* The reader takes over: a wheel, a touch or a scroll key ends the flight
     where it is, instead of being pulled back each frame (8 reversals per
     wheel gesture). But a trackpad flick keeps sending momentum wheel events
     for a second or two after the press, in the same direction and
     decaying: that is the gesture the reader already finished, not a new
     one. Forgiving only the first 300 ms cancelled 30 of 30 flights pressed
     during a flick (ME-04). Only that tail is forgiven: the gesture running
     at the press, in the flight's own direction, under 250 ms apart (the
     stream paused 168 ms across the press itself in headless Chromium),
     within 2 s, each delta no bigger than the last - and only once the
     stream has SHRUNK at least once since the press. A steady mouse wheel
     (100px a notch) never shrinks: forgiven as "no bigger", it dragged a
     reader wheeling down 2.7k px UP by a footer press (Council B). Requiring
     every delta to shrink fixed that but cancelled real tails, whose integer
     deltas repeat (2, 2, 1, 1: momentum 0/10, Council B's second review).
     Anything else - a reversal, a pause, a harder wheel, a wheel that has
     never shrunk, one against the flight - yields at once. */
  const SCROLL_KEYS = ['PageUp', 'PageDown', 'ArrowUp', 'ArrowDown', 'Home', 'End', ' '];
  let wheelAt = 0, wheelDy = 0, tail = false, shrunk = false, forgivenAt = 0;
  function yieldNow() {
    tail = false;
    P.__melionaFlight = (P.__melionaFlight || 0) + 1;
    P.__melionaStop();
  }
  function onYield(e) {
    if (e.type === 'wheel') {
      const t = performance.now();
      const same = t - wheelAt < 250 && Math.sign(e.deltaY) === Math.sign(wheelDy);
      const noBigger = same && Math.abs(e.deltaY) <= Math.abs(wheelDy);
      if (same && Math.abs(e.deltaY) < Math.abs(wheelDy)) shrunk = true;   // reset at every press
      wheelAt = t; wheelDy = e.deltaY;
      if (!P.__melionaStop) return;
      if (tail && noBigger && shrunk && Math.sign(e.deltaY) === P.__melionaDir &&
          t - P.__melionaPressAt < 2000) { forgivenAt = t; return; }
    }
    if (!P.__melionaStop) return;
    if (e.type === 'keydown' && SCROLL_KEYS.indexOf(e.key) < 0) return;
    yieldNow();
  }
  /* A scrollbar drag sends no wheel, touch or key - only scroll events. A
     scroll the flight did not make is the reader taking over: on a page still
     growing the settle phase re-asserted the target for up to 10 s and pulled
     a dragged scrollbar back (Council B). Not counted: the forgiven flick's
     own scrolling (300 ms after its last event), and the page shrinking under
     the flight and clamping it. */
  function onScroll(e) {
    const sc = P.__melionaScroller;
    if (!P.__melionaStop || !sc || e.target !== sc || P.__melionaSetY === undefined) return;
    const y = sc.scrollTop;
    if (Math.abs(y - P.__melionaSetY) <= 2) return;
    if (performance.now() - forgivenAt < 300) { P.__melionaSetY = y; return; }
    if (y >= sc.scrollHeight - sc.clientHeight - 2 && P.__melionaSetY > y) return;
    yieldNow();
  }
  function go(a, el) {
    const scroller = findScroller(el);
    /* Reduced motion: land at once, same place. A CSS media query cannot stop
       a JavaScript flight, so ask on every click (the setting can change
       while the page is open). */
    if (window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      supersede();
      scroller.style.scrollBehavior = 'auto';
      scroller.scrollTo(0, targetY(scroller, el));
      scroller.style.scrollBehavior = '';
      return;
    }
    glide(scroller, el);
  }
  /* A press can come before its section exists: #analysis is written after
     the whole engine (and #crisis-record was, until it moved ahead of its
     data), so during a cold load both buttons fell through to the browser's
     hash jump, which scrolled nowhere and never followed - dead 8 of 8
     (ME-04). Keep the press, wait for the section, then fly.
     How long: as long as the page is still being written. A fixed 10 s was
     not enough on a real cold start - the Live audit (run 39737935470) saw
     "Skip to the analysis" give up and do nothing, and offline with 4 s pulls
     the section came 12.3-15.7 s after the press, 0 of 2 landed. The wait now
     lasts while Streamlit's running sign is up (the icon app.css draws its
     "Updating..." from), and 10 s from the press in any case; once the run has
     ended (no running sign for 2 s) with the section still absent, nothing is
     left that could write it and the wait ends. The reader ends it at once:
     a wheel, a touch or a scroll key (onYield), or a scroll they make
     (onScroll, measured from where the page stood at the press). No anchor
     can be written early instead: #analysis has to sit right above the tab
     strip, which only exists once the engine has run. */
  function waitFor(a, id) {
    supersede();                          // a flight still running lands first
    const w = P.__melionaFlight = (P.__melionaFlight || 0) + 1;
    const scroller = findScroller(a);
    P.__melionaStop = function () { P.__melionaStop = null; P.__melionaScroller = null; };   // nothing else to restore
    P.__melionaScroller = scroller;       // so a scroll the reader makes while it waits ends the wait
    P.__melionaSetY = scroller.scrollTop;
    const t0 = performance.now();
    let idleSince = null;
    (function poll() {
      if (w !== P.__melionaFlight) return;   // a newer press, or the reader took over
      const el = P.getElementById(id);
      if (el) { go(a, el); return; }
      const now = performance.now();
      idleSince = pageRunning() ? null : (idleSince === null ? now : idleSince);
      if (now - t0 < 10000 || idleSince === null || now - idleSince < 2000) setTimeout(poll, 100);
      else P.__melionaStop();             // the run is over and the section never came
    })();
  }
  function onLink(e) {
    const a = e.target.closest('a[href^="#"]');
    const id = a ? a.getAttribute('href').slice(1) : '';
    if (!id) return;                     // a bare '#': nothing to fly to
    e.preventDefault(); e.stopPropagation();
    P.__melionaPressAt = performance.now();
    tail = P.__melionaPressAt - wheelAt < 250;   // a flick still running at the press
    shrunk = false;                              // ...that has yet to shrink since it
    const el = P.getElementById(id);
    if (el) go(a, el); else waitFor(a, id);
    /* release the arrow so it falls ahead of the page it is pulling - every
       link that carries one, the quiet skip link too */
    if (a.querySelector('.cta-arrow')) {
      a.classList.remove('is-launching'); void a.offsetWidth;
      a.classList.add('is-launching');
      setTimeout(function () { a.classList.remove('is-launching'); }, 1250);
    }
    /* No arrival theatre on the destination. It used to drop to 25% opacity
       and slide 26px the instant the button was pressed, then rise in as the
       glide landed (plus a photo "launch" on the band and a frame on
       Conviction) - a slide change, not a scroll, and the deep scan measured
       the flash on every button. The section is simply there when you
       arrive; the glide is the whole transition. */
  }
  P.addEventListener('click', onTab, true);
  P.addEventListener('click', onLink, true);
  ['wheel', 'touchstart', 'keydown'].forEach(function (t) {
    P.addEventListener(t, onYield, { capture: true, passive: true });
  });
  P.addEventListener('scroll', onScroll, { capture: true, passive: true });
  P.__melionaGlide = { onTab: onTab, onLink: onLink, onYield: onYield, onScroll: onScroll };
})();
/* The crest's rings ripple while a pointer rests on it (static/app.css). On
   leave, each ring finishes the pass it is in and stops at rest - a ripple
   that settles rather than snapping back - and the outer rings, a beat behind,
   carry the last wave outward. Same rebind rule as the glide above. */
(function() {
  const P = window.parent.document;
  const prev = P.__melionaRipple;
  if (prev) {
    P.removeEventListener('pointerover', prev.enter, true);
    P.removeEventListener('pointerout', prev.leave, true);
  }
  function crestOf(e) {
    const c = e.target.closest && e.target.closest('.hero-crest');
    return c && !c.contains(e.relatedTarget) ? c : null;   /* crossing its edge */
  }
  function enter(e) {
    const c = crestOf(e); if (!c) return;
    c.classList.add('rippling');
    /* Back before the last wave ended: a ring that already settled is no
       longer listed, so wake the ones kept from the leave too - each resumes
       from rest, where its last pass ended. */
    (c.__rings || []).concat(c.getAnimations({ subtree: true })).forEach(function(a) {
      a.effect.updateTiming({ iterations: Infinity });
    });
  }
  function leave(e) {
    const c = crestOf(e); if (!c) return;
    const runs = c.__rings = c.getAnimations({ subtree: true });
    if (!runs.length) { c.classList.remove('rippling'); return; }
    Promise.all(runs.map(function(a) {
      const pass = a.effect.getComputedTiming().currentIteration || 0;
      a.effect.updateTiming({ iterations: pass + 1 });
      return a.finished;
    })).then(function() {
      if (!c.matches(':hover')) { c.classList.remove('rippling'); c.__rings = null; }
    }, function() {});
  }
  P.addEventListener('pointerover', enter, true);
  P.addEventListener('pointerout', leave, true);
  P.__melionaRipple = { enter: enter, leave: leave };
})();
</script>
""")

# ---- GLOSS LAYER (2026-07-09) ----
# Institutional sheen over the matte editorial base: specular highlights,
# layered depth, and a few one-pass arrival glints. Same Citadel palette -
# no new hues, no rounded pills. Every effect degrades to the matte base
# under prefers-reduced-motion (guard at the end).
# ---- BREATHING ROOM (2026-07-15, Breez-inspired spacing pass) ----
# The engine section reads as a wall when panels touch. Generous, consistent
# vertical rhythm: every ruled section gets air above it, tab strips get a
# quiet margin, and expanders stop crowding the panel above them. Space,
# not emptiness - nothing removed.
# ---- ROUND 3 (2026-07-09) ----
# Softened geometry (loosen the border-radius:0 doctrine into a restrained
# radius scale + organic/circular accents), glossy chart halos + glowing bars,
# a whisper of scroll-motion-blur on decorative layers, and UFO tile arrivals.
# Still institutional: radii are gentle (10-26px), no candy.
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
# ---- Themed Plotly palette + chart helpers (institutional beige/bronze) ----
BRONZE = "#9A7B4F"
BRONZE_DK = "#8A6A3C"
CHARCOAL = "#3F3B35"
BAND_OUTER = "rgba(154,123,79,0.14)"   # light bronze - 5–95 percentile cone
BAND_INNER = "rgba(154,123,79,0.30)"   # medium bronze - 25–75 percentile cone
GRID = "rgba(63,59,53,0.12)"
AXIS_LINE = "rgba(63,59,53,0.28)"

# responsive=True is not cosmetic. Plotly defaults it to FALSE, which means a
# figure keeps whatever pixel width it first laid out at and never re-fits when
# its container settles, an expander opens, a column re-flows or the window
# resizes. That is how charts end up hanging outside their box - measured here
# at 643px of SVG inside a 429px column. With it on, Plotly re-lays out on
# every container resize, so charts stay inside their designated spot.
PLOTLY_CFG = {"displayModeBar": False, "staticPlot": False, "responsive": True}


def _style_fig(fig, height: int = 300):
    """Apply the calm serif/beige institutional theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        # Charts inside st.tabs are rendered while their tab is HIDDEN, so the
        # container measures 0px wide and Plotly falls back to its 700px
        # default - which then sits 90px short inside a 790px box, leaving a
        # gap on the right. autosize makes the figure take its width from the
        # container instead of that default, so it fills the box on whichever
        # tab it lands in. Pairs with responsive=True in PLOTLY_CFG: autosize
        # decides the initial width, responsive keeps it correct on resize.
        autosize=True,
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
    """Histogram of simulated 1-year outcomes. Bars at or beyond the CVaR
    average (not the whole VaR tail - the measured tail starts at the 5th
    percentile; CVaR is that tail's mean, which sits deeper) are inked in
    oxblood so the eye lands on the danger; every bar is a real simulation
    count - color is annotation, not data."""
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


def hbar(series: pd.Series, pct: bool = False, title_x: str = "",
         top_first: bool = False):
    """Themed horizontal bar chart. Bars deepen with magnitude - the biggest
    value wears the darkest bronze - so ranking reads at a glance.

    Plotly draws the FIRST category at the BOTTOM of a horizontal bar chart.
    Pass top_first=True for an already-ranked series so the leader lands where
    the reader - and the caption - expects it."""
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
    fig = _style_fig(fig, height=max(160, 30 * len(series) + 40))
    if top_first:
        fig.update_yaxes(autorange="reversed")
    return fig


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


# Anchored on this file, not the working directory: `streamlit run` from any
# other folder used to crash here on the very first line of the hero.
ASSETS = Path(__file__).parent / "assets"
STATIC = Path(__file__).parent / "static"   # served at /app/static/

# ---- Hero: the pitch, not the dashboard ----
with open(ASSETS / "logo.svg", "r", encoding="utf-8") as f:
    logo_svg = f.read()


# The short-viewport hero tiers live at the END of static/app.css (search
# "SHORT-VIEWPORT HERO"): the boot skeleton in index.html needs them from
# the first frame, before anything here is sent.


# The hero's words and markup live in hero.py, shared with the boot skeleton
# tools/public_files.py writes into index.html.


# The boot skeleton is painted by index.html itself (tools/public_files.py),
# from the first frame: sent from here it could only arrive once Streamlit's
# JavaScript had started, and the first 2.5 s (laptop) to 4.4 s (phone) were a
# plain screen (2026-10-19). This slot keeps the first-load tile fly-in, which
# must render ONLY on the first script run of a session - Streamlit reruns the
# whole script on every interaction, which would replay it forever.
if not st.session_state.get("_booted"):
    st.session_state["_booted"] = True
    st.markdown(
        """<style>
  /* First-load only: the intro tiles fly in top-to-down (UFO settle). This
     style block renders solely on the un-booted first pass, so Streamlit
     reruns never replay the arrival. Scoped to #hero so the skeleton's own
     tiles hold still. */
  #hero .hero-stats .hstat { animation: ufo-drop .6s cubic-bezier(.2,.9,.25,1) both; }
  #hero .hero-stats .hstat:nth-child(1) { animation-delay: .05s; }
  #hero .hero-stats .hstat:nth-child(2) { animation-delay: .13s; }
  #hero .hero-stats .hstat:nth-child(3) { animation-delay: .21s; }
  #hero .hero-stats .hstat:nth-child(4) { animation-delay: .29s; }
</style>""", unsafe_allow_html=True)
else:
    # Hold the veil's slot. Streamlit keys every element by its position, so
    # dropping one element on the second run shifted everything below it and
    # every drawer (st.expander) re-mounted closed - the first control a
    # visitor changed snapped shut under their cursor (recruiter walkthrough).
    st.markdown("", unsafe_allow_html=True)

# Architectural plate behind the hero stat deck (replaces the watermark crest).
# static/facade.jpg - Unsplash (free commercial license, no attribution
# required). Duotoned toward the palette in CSS, so the photo can never clash.
# Served as a file from static/ (config.toml enableStaticServing): inlined as
# base64 it put ~3.5 MB on the websocket AHEAD of the boot skeleton, which then
# arrived with the hero it stands in for (2026-10-19). Paint only - the hero's
# layout (full-bleed margin, padding, ledger frame) lives in app.css, so the
# skeleton has the hero's shape before this arrives. A missing photo holds the
# slot, so nothing below moves.
if (STATIC / "facade.jpg").is_file():
    st.markdown(
        # The architecture washes across the WHOLE hero, melting into beige
        # toward the text side - the building emerges from the page's own
        # color. Luminosity blend: the photo keeps only its LIGHT - its hue
        # comes entirely from the base color beneath, which is CITY-STONE
        # BEIGE in the crest's own warm family (not gray, not orange).
        "<style>.hero-section { background-color: #C9AF87; "
        "background-image: linear-gradient(90deg, "
        "#EDE9E3 0%, rgba(237,233,227,.96) 40%, rgba(237,233,227,.62) 66%, "
        "rgba(237,233,227,.22) 100%), url(app/static/facade.jpg); "
        "background-size: auto, cover; "
        "background-position: left, right 78%; "
        "background-blend-mode: normal, luminosity; }</style>",
        unsafe_allow_html=True)
else:
    st.markdown("", unsafe_allow_html=True)

# Dark-band plate: the charcoal showcase band gets its own architectural
# photograph (static/band.jpg - Unsplash, free commercial license): fog-bound
# towers, duotoned near-charcoal in CSS with a scrim baked in so the beige
# text stays the loudest thing on the band. A file, not base64 (see above).
if (STATIC / "band.jpg").is_file():
    st.markdown(
        "<style>.showcase-row::before { background-image: "
        "linear-gradient(165deg, rgba(59,50,40,.42), rgba(45,39,32,.68)), "
        "url(app/static/band.jpg); }</style>",
        unsafe_allow_html=True)
else:
    st.markdown("", unsafe_allow_html=True)  # band stays plain charcoal

# The hero lifts its own skeleton when it arrives - state, not a timer. The
# veil used to fade on a fixed 2.1 s clock whether or not anything was ready,
# and under reduced motion (no animation runs) a timer never fires at all.
_LIFT_SKELETON = ("<style>#pre-boot,#boot-skel{opacity:0;visibility:hidden}"
                  "#boot-skel *{animation:none!important}</style>")
st.markdown(hero_html(crest=logo_svg) + _LIFT_SKELETON, unsafe_allow_html=True)

# ---- Showcase: the Grit Zone innovation, explained before you touch a slider ----
st.markdown("""
<div class="showcase-row reveal">
  <div class="showcase-section" id="grit-showcase" style="position:relative;">
    <div class="engrave scale"><svg viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <!-- the crest's hexagon and its X, cut in outline (assets/logo.svg) -->
      <g fill="none" stroke="#9A7B4F" stroke-width="1.1" stroke-linejoin="round">
        <circle r="44"/>
        <polygon points="0,-35 27.358,-19.205 0,-3.409 -27.358,-19.205"/>
        <polygon points="0,-2.309 4,0 0,2.309 -4,0"/>
        <polygon points="0,35 27.358,19.205 0,3.409 -27.358,19.205"/>
        <polygon points="-5.905,0 -30.311,-14.091 -30.311,14.091"/>
        <polygon points="5.905,0 30.311,-14.091 30.311,14.091"/>
      </g>
    </svg></div>
    <div class="showcase-eyebrow">The Innovation</div>
    <h2 class="showcase-title">The Grit Zone</h2>
    <div class="showcase-body">
      Fear &amp; Greed indices measure market mood. We measure something more
      durable: whether an asset, once knocked down, actually gets back up,
      consistently, across real crises.
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
  </div>
  <div class="showcase-section" id="conviction">
    <div class="conv-core">
      <div class="showcase-eyebrow">The Conviction</div>
      <h2 class="showcase-title" style="font-size:34px;">The hardest trade is the one history rewards</h2>
      <div class="showcase-body">
        Panic in a crash is wiring, not weakness. This engine answers it with
        evidence, not a slogan: the actual record of every named crisis it
        stress-tests, computed live. Below is what a buyer earned on the scariest
        day of each crisis, and at the worst-timed entry, the pre-crash peak.
      </div>
      <a href="#crisis-record" class="cta-btn on-dark" style="margin-top:22px;">
        See the crisis record<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span></a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---- Showcase: Crisis Conviction - the emotional problem, answered in numbers ----
@st.cache_data(ttl=6 * 3600, show_spinner="Reading the crisis record…")
def load_conviction_data():
    """Benchmark crisis record + AI-capex recovery race, live from Yahoo."""
    return load_conviction()

# Landing target for "See the crisis record" and footer 02. Zero-height marker
# rather than a heading, so the section keeps its current typography - the
# glide script scrolls to whatever element carries the id. Written BEFORE the
# data loads, whatever happens to it: inside the try it existed only once the
# record arrived, so during a slow load the press had to wait, and if the load
# failed footer 02 waited 10 s and went nowhere (Council B, 2026-10-22). Now
# both land on the record's place - the slabs, or the honest "unavailable".
st.markdown('<div id="crisis-record" style="scroll-margin-top:18px;"></div>',
            unsafe_allow_html=True)

try:
    _conv = load_conviction_data()
    _s = _conv["summary"]
    _t1, _t3 = _s["trough_1y_later"], _s["trough_3y_later"]
    _p3 = _s["peak_3y_later"]
    # Counted once, in src/conviction.race_record, for the slab and the tab.
    _rec = race_record(_conv["race"])
    _bwin, _nrace = _rec["wins"], _rec["n"]


    _sl1, _sl2, _sl3, _sl4 = st.columns(4)
    with _sl1:
        st.markdown(f"""<div class="slab"><div class="slab-label">Bought the scariest day</div>
        <div class="slab-num">{round(_t1["pct_positive"] * _t1["n"])} of {_t1["n"]}</div>
        <div class="slab-note">crises were positive one year after the trough
        - median return <b>{_t1["median"]:+.0%}</b>.</div></div>""", unsafe_allow_html=True)
    with _sl2:
        st.markdown(f"""<div class="slab"><div class="slab-label">Three years on</div>
        <div class="slab-num">{_t3["median"]:+.0%}</div>
        <div class="slab-note">median return three years after the scariest day
        ({round(_t3["pct_positive"] * _t3["n"])} of {_t3["n"]} positive).</div></div>""",
                    unsafe_allow_html=True)
    with _sl3:
        st.markdown(f"""<div class="slab"><div class="slab-label">Worst possible timing</div>
        <div class="slab-num">{round(_p3["pct_positive"] * _p3["n"])} of {_p3["n"]}</div>
        <div class="slab-note">crises: a buyer at the pre-crash <b>peak</b>
        was ahead three years later (median return {_p3["median"]:+.0%}).</div></div>""",
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
    # This CTA used to sit at the end of the narrative block ABOVE - which put
    # it before the four evidence boxes and this caption, so clicking "from the
    # record to the allocation" jumped 346px PAST the record it names and the
    # reader had to scroll back up to find it. Moved here, after the evidence,
    # so the button is the last thing in its own section and skips nothing.
    st.markdown(
        '<div style="text-align:center;margin:18px 0 4px;">'
        '<a href="#the-slice" class="cta-btn">'
        'From the record to the allocation<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span></a></div>',
        unsafe_allow_html=True)
except Exception as _exc:  # noqa: BLE001 - landing page must never crash on data
    st.caption(f"Crisis record unavailable right now ({safe_err('crisis-record-unavailable-right-now', _exc)}). "
               "The Crisis Conviction tab retries on load.")

# ---- Gungnir - the slice: reading ends, allocation begins ----
# assets/gungnir.png is generated in-repo (procedural topographic contours
# from a single scalar field + the bound-rune engraving) - born at 3200px,
# no upscaling involved, so it can never look like a stretched photo.
try:
    with open(ASSETS / "gungnir.png", "rb") as _gf:
        _gungnir_b64 = base64.b64encode(_gf.read()).decode()
    _gungnir_bg = (
        "background-image:linear-gradient(90deg, rgba(244,241,234,0) 0%, "
        "rgba(244,241,234,.22) 36%, rgba(244,241,234,.9) 64%, #F4F1EA 82%), "
        f"url(data:image/png;base64,{_gungnir_b64});")
except Exception:  # noqa: BLE001 - asset missing: plain band, honest fallback
    _gungnir_bg = ""
st.markdown(
    f'<div id="the-slice"></div>'
    f'<style>.st-key-gungnir_zone::before {{ {_gungnir_bg} }}</style>',
    unsafe_allow_html=True)
_gungnir_zone = st.container(key="gungnir_zone")
with _gungnir_zone:
    st.markdown(
        '<div class="gungnir-head">'
        '<div class="showcase-eyebrow">The Slice</div>'
        '<h2 class="showcase-title" style="font-size:30px;">From evidence to allocation.</h2>'
        '<div class="gungnir-sub">Everything above was the case. Everything below is the tool: '
        'choose a side, pick a universe.</div>'
        '</div>',
        unsafe_allow_html=True)

# ---- Direction: the same honest math from either side of the trade ----
# Council pass 7: bearish mode models a SYNTHETIC DAILY-REBALANCED SHORT
# (negated daily returns). Symmetric statistics (covariance, correlation,
# eigenvectors, volatility) are mathematically identical either way and are
# never relabeled; only asymmetric ones (tails, drawdowns, Monte Carlo,
# factor betas, Bon Voyage) genuinely recompute.
with _gungnir_zone:
    _dir_col = st.columns([1.2, 2.6, 1.2])[1]
    with _dir_col:
        direction = st.radio(
            "Which side of the trade are you on?",
            ["Bullish - long the book", "Bearish - short the book"],
            horizontal=True, key="bv_direction",
            help="Bearish mode re-runs every tail metric, Monte Carlo path and "
                 "the defensive pairing on a synthetic daily-rebalanced short "
                 "of the same assets. Borrow fees, margin interest and buy-ins "
                 "are NOT modeled - real short results are worse than shown.")
bearish = direction.startswith("Bearish")
_bv_sk_a, _bv_sk_b = ("the short position", "the squeeze cushion") if bearish \
    else ("the high-flyer", "the steady cushion")
# Geometry note: the tether line runs CIRCUMFERENCE to CIRCUMFERENCE - its
# endpoints sit on each circle's edge plus a 4px breath, never inside. All
# labels are sized against the chord width where they sit, so nothing
# overflows a circle or collides with a role caption.
# The sketch keeps the left half; the right half carries the plaque: the
# two-asset risk-parity identity the sketch is drawing. Volatility is
# symmetric under negation (Council pass 7), so the plaque holds verbatim
# on either side of the trade and never relabels.
with _gungnir_zone:
    _gg_sketch, _gg_plaque = st.columns([1.05, 1], gap="large")
    with _gg_sketch:
        st.markdown(
        f'<div class="showcase-section reveal" style="padding-top:2px;text-align:center;">'
        # The pairing drawn as the equation it actually is. w1*s1 = w2*s2 says
        # two blocks of (capital x volatility) have EQUAL AREA, so that is the
        # picture: one wide-and-short, one narrow-and-tall, same area, sharing
        # a baseline. The old two-circles-and-a-line was decorative - the radii
        # carried no quantity and the line asserted a relationship instead of
        # showing one. Here the geometry is exact: 120x36 and 40x108 are both
        # 4320, a true 3:1, which is the plaque's own sentence ("three times as
        # volatile gets one third the capital") made visible.
        # Schematic on purpose - no ticks, no numbers, qualitative axes only,
        # so it can never be mistaken for a live chart of the user's book.
        f'<svg viewBox="0 0 360 200" width="460" height="256" xmlns="http://www.w3.org/2000/svg" style="opacity:.96;max-width:92vw;height:auto;">'
        f'<line x1="34" y1="152" x2="330" y2="152" stroke="#C4BDAE" stroke-width="1"/>'
        f'<line x1="34" y1="24" x2="34" y2="152" stroke="#C4BDAE" stroke-width="1"/>'
        f'<rect x="60" y="116" width="120" height="36" fill="rgba(63,59,53,.10)" stroke="#3F3B35" stroke-width="1.2"/>'
        f'<rect x="248" y="44" width="40" height="108" fill="rgba(154,123,79,.20)" stroke="#8A6A3C" stroke-width="1.4"/>'
        f'<text x="214" y="105" text-anchor="middle" font-family="Georgia" font-size="17" fill="#8A6A3C">=</text>'
        f'<text x="42" y="17" font-family="Helvetica Neue" font-size="7.5" letter-spacing="1.5" fill="#9A7B4F">VOLATILITY</text>'
        f'<text x="330" y="167" text-anchor="end" font-family="Helvetica Neue" font-size="7.5" letter-spacing="1.5" fill="#9A7B4F">CAPITAL</text>'
        f'<text x="120" y="172" text-anchor="middle" font-family="Helvetica Neue" font-size="8.5" letter-spacing="1.6" fill="#9A7B4F">02</text>'
        f'<text x="120" y="190" text-anchor="middle" font-family="Georgia" font-size="11.5" fill="#6B6459">{_bv_sk_b}</text>'
        f'<text x="268" y="172" text-anchor="middle" font-family="Helvetica Neue" font-size="8.5" letter-spacing="1.6" fill="#9A7B4F">01</text>'
        f'<text x="268" y="190" text-anchor="middle" font-family="Georgia" font-size="11.5" fill="#6B6459">{_bv_sk_a}</text>'
        f'</svg>'
        f'<div style="font-family:Georgia;font-size:12.5px;color:#5D574D;max-width:560px;margin:2px auto 0;">'
        f'{"Shorting flips the sketch: block 01 is the position you are against, block 02 a correlated long that cushions the squeeze. Equal areas, equal risk share. Short losses can exceed 100%; borrow costs are not modeled." if bearish else "Block 01 is narrow and tall - little capital, high volatility. Block 02 is wide and short. Equal areas, so each name carries the same share of the risk."}'
        f'</div>'
        f'<a href="#engine" class="cta-btn" style="margin-top:14px;">'
        f'To the engine<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span></a>'
        f'</div>',
        unsafe_allow_html=True)
    # Overwhelm audit (2026-10-29): the sketch and the equation stay - they draw
    # the flyer-and-cushion pairing, the linkage the product is built on. The
    # dense paragraph under them (equal risk contribution, correlation
    # cancelling, the covariance matrix) went: the engine below does that work.
    with _gg_plaque:
        st.markdown(
        f'<div class="gungnir-plaque reveal">'
        f'<div class="gplaque-eyebrow">The math behind the sketch</div>'
        f'<div class="gplaque-eq">'
        f'<span class="gplaque-term"><span class="t">w<sub>1</sub>&sigma;<sub>1</sub></span>'
        f'<span class="c">block 01</span></span>'
        f'<span class="gplaque-eqs">=</span>'
        f'<span class="gplaque-term"><span class="t">w<sub>2</sub>&sigma;<sub>2</sub></span>'
        f'<span class="c">block 02</span></span>'
        f'</div>'
        f'<div class="gplaque-read">The steadier name carries more of the money, '
        f'so both names carry the same share of the risk.</div>'
        f'</div>',
        unsafe_allow_html=True)

st.markdown(
    f'<hr class="section-divider">'
    f'<div class="engine-heading reveal" id="engine">'
    f'<div class="showcase-eyebrow">The Engine</div>'
    f'<h2 class="showcase-title" style="font-size:26px;">Stress-test any '
    f'{"short book" if bearish else "portfolio"}, live</h2>'
    # "the analysis", not "the risk map": it lands on the analysis tab strip,
    # whichever tab the reader left open (ME-04). Styled by .cta-link.
    f'<a href="#analysis" class="cta-link">'
    f'Skip to the analysis<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span></a>'
    f'</div>',
    unsafe_allow_html=True)

# ---- Orientation: what a first-time visitor needs before the dials ----
# A walkthrough as a non-specialist found the controls arrive before any
# statement of what the page is FOR. Three sentences, then the engine.
st.markdown(
    '<div class="orient reveal">'
    '<div class="orient-t">New here? Read this first</div>'
    '<ol class="orient-l">'
    '<li><b>Pick what you own.</b> Choose a starter basket below, or type any '
    'ticker - the box takes any symbol Yahoo Finance carries.</li>'
    '<li><b>Break it on purpose.</b> Replay a real crisis, or move the sliders, '
    'and watch the headline number move.</li>'
    '<li><b>Read one number.</b> The verdict answers a single question: in the '
    'worst 5% of simulated years, how much of this portfolio goes away? '
    'Everything below it is the working behind that answer.</li>'
    '</ol>'
    '<div class="orient-f">Nothing here is advice, and no figure is invented: '
    'every number is computed from live end-of-day prices, and every tab shows '
    'its own method.</div>'
    '</div>',
    unsafe_allow_html=True)

# ---- The cockpit: controls fold into three numbered drawers so the verdict
# leads the section. Widgets still execute when collapsed - zero logic change,
# the reader just isn't bombarded with every dial at once. ----
with st.expander("01 · Universe - which assets", expanded=False):
    preset = st.selectbox("Preset basket", list(PRESETS.keys()), label_visibility="collapsed")

    # Keying the multiselect on the preset name makes it re-initialize with the
    # new default whenever the preset changes - while still letting users add or
    # remove individual symbols (accept_new_options allows arbitrary tickers).
    # The full suggestion catalogue, not just names that happen to sit in a
    # preset: a walkthrough found a visitor's own holding (NFLX, AMD, BAC)
    # was absent from every dropdown even though the box accepts any symbol.
    suggestions = SUGGESTIONS
    chosen = st.multiselect(
        "Tickers to analyze",
        options=suggestions,
        default=PRESETS[preset],
        key=f"tickers__{preset}",
        accept_new_options=True,
        help="Pick a preset above, or add any Yahoo Finance symbol "
             "(e.g. BRK-B, EURUSD=X for FX, GC=F for gold futures).",
    )

_raw_syms = sorted({t.strip().upper() for t in chosen if t.strip()})
# The ticker box takes free text, and ticker names get interpolated into
# unsafe_allow_html blocks downstream (the headline verdict names excluded
# symbols inside a <div>). Reject anything not shaped like a Yahoo symbol
# HERE, and say so, rather than silently dropping it - a user who typos a
# symbol deserves to know why it vanished.
tickers = [t for t in _raw_syms if valid_ticker(t)]
_rejected = [t for t in _raw_syms if not valid_ticker(t)]
# Notices that come and go with the basket share ONE slot, so the tab strip
# below never changes position (test_the_tab_strip_keeps_its_slot_whatever_the_basket).
_notices = st.container()
if _rejected:
    _notices.warning(
        "Ignored "
        + ", ".join(f"`{t[:24]}`" for t in _rejected)
        + " - not a valid Yahoo Finance symbol. Symbols are letters, digits "
          "and `. - = ^` only (e.g. `BRK-B`, `EURUSD=X`, `GC=F`, `^IRX`)."
    )
if len(tickers) > MAX_UNIVERSE:
    # Complexity budget, enforced where the visitor can SEE it. Every extra
    # symbol is another Yahoo leg, another covariance row, another full-history
    # Grit pull and another map unit - on a single-process public deploy that
    # cost lands on every other visitor. src/ingestion.py caps the funnel too,
    # so a caller that skips this check still cannot exceed the budget; this
    # branch exists so the cap is disclosed rather than silently applied.
    _notices.warning(
        f"This engine analyzes up to **{MAX_UNIVERSE}** symbols at once - a "
        "deliberate compute budget on a public, single-process deployment. "
        f"Using the first {MAX_UNIVERSE} alphabetically; dropped "
        + ", ".join(f"`{html.escape(t)}`" for t in tickers[MAX_UNIVERSE:])
        + "."
    )
    tickers = tickers[:MAX_UNIVERSE]
if len(tickers) < 2:
    st.warning("Add at least two symbols to analyze a portfolio.")
    st.stop()


# Short TTL so the session re-checks the (already freshness-aware, 6h) disk
# cache often and the UI feels snappy -- this does NOT hit Yahoo more often;
# it just re-reads the local parquet faster. See PROGRESS.md "fast polling."
#
# max_entries on every cache keyed by visitor input: st.cache_data keeps one
# entry per argument tuple until its ttl, so without a cap a script cycling
# tickers or slider values grows process memory without bound. 2026-10-15
# security audit. 64 for small results, 32 for frames and bootstraps - well
# above what honest traffic reuses within a ttl.
@st.cache_data(ttl=60, max_entries=64, show_spinner="Fetching market data…")
def load_universe(tickers_tuple: tuple[str, ...], period: str = "2y"):
    return fetch_prices(list(tickers_tuple), period=period)


@st.cache_data(ttl=3600, max_entries=32, show_spinner="Fitting tail model…")
def load_tail_fit(tickers_tuple: tuple[str, ...], weights_tuple: tuple[float, ...],
                  bearish_flag: bool):
    """
    GPD tail fit on a LONG history, deliberately not the 2-year window the
    rest of the app uses. Peaks-over-threshold needs enough points ABOVE the
    threshold: 2y of daily data leaves ~25 exceedances at the 95% level, under
    the 50 the asymptotics require, so the fit would (correctly) refuse. 10y
    gives ~126. Cached because the xi bootstrap (200 refits) still costs
    ~0.6s.

    Names without the full history are dropped and the remaining weights are
    rescaled to preserve the book's original gross notional - so the fit
    describes the same leverage, on the names that actually have the history.
    """
    # Unaligned on purpose: an aligned pull cut every name to the youngest
    # listing, so one 2021 IPO turned "10 years" into 5.1 and nothing was ever
    # named as dropped (2026-10-17 council). A week of slack absorbs holidays.
    px = fetch_prices(list(tickers_tuple), period="10y", align=False)
    first = px.notna().idxmax()
    px = px.loc[:, first <= first.min() + pd.Timedelta(days=7)].dropna()
    rets = px.pct_change().dropna()
    keep = [t for t in tickers_tuple if t in rets.columns]
    if len(keep) < 2:
        return None, 0, [], 0.0
    w = pd.Series(weights_tuple, index=list(tickers_tuple))[keep]
    gross = float(np.abs(weights_tuple).sum())
    if float(w.abs().sum()) == 0:
        return None, 0, [], 0.0
    w = w * (gross / float(w.abs().sum()))
    pr = rets[keep] @ w
    if bearish_flag:
        pr = -pr
    dropped = [t for t in tickers_tuple if t not in keep]
    # Conditional EVT: the GPD is fitted to EWMA-standardised residuals, not
    # raw returns, because peaks-over-threshold assumes i.i.d. exceedances and
    # the Christoffersen test above exists to show they are not. Returns the
    # unconditional xi alongside, so the clustering bias is visible.
    years = (pr.index[-1] - pr.index[0]).days / 365.25 if len(pr) else 0.0
    return mcneil_frey_tail(pr), int(len(pr)), dropped, years


@st.cache_data(ttl=3600, max_entries=64, show_spinner="Loading volume data…")
def load_adv(tickers_tuple: tuple[str, ...]):
    """Average daily dollar volume per ticker (recent 3-month lookback)."""
    return average_dollar_volume(list(tickers_tuple))


@st.cache_data(ttl=3600, max_entries=64, show_spinner="Loading volume history…")
def load_dollar_volume(tickers_tuple: tuple[str, ...], period: str = "2y"):
    """DAILY dollar volume per ticker. The Book needs the whole series, not a
    mean: its liquidity gate runs on the 5th percentile of the window, so that
    eligibility is a quiet-tape claim rather than an average-tape one."""
    return fetch_dollar_volume(list(tickers_tuple), period=period)


@st.cache_data(ttl=3600, max_entries=32, show_spinner="Assembling the book…")
def load_book(px, dv, book_value: float, participation_rate: float,
              grit_floor: bool, anchors: tuple[str, ...], grit: dict,
              fallback_grit: dict | None = None):
    """Cached because Streamlit re-executes this whole script on EVERY widget
    interaction, and build_book is not cheap: risk_parity_weights runs 10,000
    coordinate-descent iterations and grit_scores replays every named crisis
    window per ticker. Uncached, dragging the participation slider recomputed
    the entire book on each frame - one visitor holding a slider could pin the
    single process this app deploys as. The cache key is exactly the eight
    inputs that change the answer."""
    return build_book(px, dv, book_value=book_value,
                      participation_rate=participation_rate,
                      flyer_grit_floor=grit_floor,
                      fallback_anchors=set(anchors) or None, grit=grit,
                      fallback_grit=fallback_grit)


@st.cache_data(ttl=3600, max_entries=32, show_spinner="Reading the defensive shelf…")
def load_anchor_shelf(anchors: tuple[str, ...], period: str = "2y"):
    """Prices and volume for the defensive names The Book may fall back on when
    the chosen universe contains nothing that thins a tail."""
    return (fetch_prices(list(anchors), period=period, align=False),
            fetch_dollar_volume(list(anchors), period=period))


@st.cache_data(ttl=3600, show_spinner="Reading the 13-week T-bill rate…")
def load_risk_free_rate():
    """Latest 13-week T-bill yield (^IRX) as an annual decimal, or None."""
    return fetch_risk_free_rate()


# The wait is the full-history pull, not the score (grit maths ~0.2 s): name
# what is happening so a first visit reads as working, not stalled - in a few
# calm words, no excuse for the wait (John, 2026-10-19).
@st.cache_data(ttl=3600, max_entries=32,
               show_spinner="Reading full price histories for the grit scores…")
def load_grit(tickers_tuple: tuple[str, ...]):
    """Grit scores need each asset's FULL price history, not just the 2y window
    used for VaR - grit_scores() pulls it separately (see src.grit)."""
    return grit_scores(list(tickers_tuple))


# Ticks once a MINUTE, not once a second. Stress test 2026-10-15: each tick
# costs the single server process ~60ms, so 15 idle tabs at 1 Hz held it at
# 93% of a core doing nothing - and every new visitor queued behind them.
# The data is end-of-day; a minute counter loses nothing a second one said.
@st.fragment(run_every="60s")
def _freshness_ticker(fetched_at_iso: str, live: bool):
    """Live-ticking 'as of N min ago' -- reruns only this fragment, not the app."""
    fetched = pd.Timestamp(fetched_at_iso)
    now = pd.Timestamp.now(tz=fetched.tzinfo) if fetched.tzinfo else pd.Timestamp.now()
    mins = max(0, int((now - fetched).total_seconds()) // 60)
    ago = "under a minute ago" if mins == 0 else f"{mins} min ago"
    # Nothing polls: this fragment only re-renders the counter. Yahoo is hit
    # again only once the disk copy is older than CACHE_MAX_AGE_HOURS - the old
    # "quotes cached 60s" wording described the in-memory re-read, not a pull.
    # The dot (ME-11) says what the stamp says, at a glance: bronze, pulsing
    # once each time this fragment ticks, while the data is live (its last row
    # at most a day old); still and grey once it is stale. Never a "real-time"
    # claim - it beats with the minute counter, not with the market.
    dot = ('<span class="fresh-dot is-live" role="img" aria-label="Data is current"></span>' if live
           else '<span class="fresh-dot is-stale" role="img" aria-label="Data is stale"></span>')
    st.caption(f"{dot} Data pulled {ago} · re-pulled from Yahoo once older "
               f"than {CACHE_MAX_AGE_HOURS:g}h, on your next interaction - not on a timer.",
               unsafe_allow_html=True)


# ---- Audit trail: what this run actually did, in order (see Lineage tab) ----
audit_log = []


def _audit(step: str, detail: str) -> None:
    audit_log.append({"step": step, "detail": detail})


try:
    prices = load_universe(tuple(tickers))
except NoDataError as exc:
    # A visitor's typo is not a crash: say what to do, not "quote a reference".
    # NoDataError messages are written for visitors (src/ingestion.py).
    log_incident(SESSION_REF, "load_universe", exc)
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the user
    st.error(f"Couldn't load market data: {safe_err('couldn-t-load-market-data', exc)}")
    st.stop()

# ---- Short-history guard --------------------------------------------------
# Aligning a basket intersects every member onto their COMMON trading days, so
# one young or delisted name silently truncates everything. Found 2026-08-26 in
# a walkthrough: EA carried 17 days of history after going private, which cut a
# ten-name basket to 20 aligned rows - and the engine went on to render a year
# of risk, and a 21-day rolling correlation, from it. The data-quality gate had
# already returned passed=False; nothing was reading it.
#
# Drop the offender rather than the basket, and say which one and why.
if len(prices) < MIN_ROWS and len(tickers) > 1:
    try:
        _unaligned = fetch_prices(tickers, period="2y", align=False)
        _coverage = _unaligned.notna().sum()
        _short = [t for t in _unaligned.columns if int(_coverage[t]) < MIN_ROWS]
        _keep = [t for t in tickers if t not in _short]
        if _short and len(_keep) >= 2:
            _notices.warning(
                "Excluded "
                + ", ".join(f"`{html.escape(t)}` ({int(_coverage[t])} trading days)"
                            for t in _short)
                + f" - each has less than the {MIN_ROWS} days of history this "
                "engine needs, and aligning the basket to their history would "
                "have cut every other name down with them. Analysing the "
                f"remaining {len(_keep)}."
            )
            tickers = _keep
            prices = load_universe(tuple(tickers))
    except Exception as _short_exc:  # noqa: BLE001 - keep the original frame
        log_incident(SESSION_REF, "short_history_guard", _short_exc)

if len(prices) < MIN_ROWS:
    st.error(
        f"This basket only has {len(prices)} trading days in common - too few "
        f"to estimate risk from (this engine needs {MIN_ROWS}). That usually "
        "means one member is newly listed, delisted, or trades on a different "
        "calendar. Remove the newest name and try again."
    )
    st.stop()

returns = get_returns(prices)
# Zero-variance guard. A price that never moves (a money-market fund such as
# VMFXX holds $1.00) has no correlation and no risk-parity weight: with the
# Sample or EWMA estimator the ERC solver divided by its zero variance, the
# headline read "nan%" and the page crashed further down. Drop it and say why,
# the same way the short-history guard above treats a name it cannot use.
_flat = flat_columns(returns)
if _flat:
    _notices.warning(
        "Excluded " + ", ".join(f"`{html.escape(t)}`" for t in _flat)
        + " - its price did not move once in this window (cash and money-market "
        "funds look like this), so it carries no measurable risk and has no "
        "correlation to estimate. The analysis runs on the remaining names."
    )
    prices = prices.drop(columns=_flat)
    returns = returns.drop(columns=_flat)
loaded = list(prices.columns)
missing = [t for t in tickers if t not in loaded and t not in _flat]
if len(loaded) < 2:
    st.error("Fewer than two usable symbols"
             + (f" (nothing for {', '.join(missing)})" if missing else "")
             + ". Check the spelling or try different tickers.")
    st.stop()
if missing:
    _notices.caption(f"Couldn't load: {', '.join(missing)} - skipped.")
_audit("Data fetch", f"{len(loaded)} tickers loaded from {preset!r}: {', '.join(loaded)}"
      + (f" (missing: {', '.join(missing)})" if missing else ""))

# ---- Data-freshness indicator (honest, not a fake real-time feed) ----
health = data_health(prices)
# No visitor "Refresh" button. It used to call st.cache_data.clear(), which
# empties every cached result for EVERY visitor in this single process and
# deletes the disk cache the spend cap falls back on - an operator action
# handed to anonymous viewers. Data is end-of-day and the disk cache already
# re-pulls past CACHE_MAX_AGE_HOURS; an operator forces a pull with
# `python -m tools.warm_cache --drill` (RUNBOOK.md).
fresh = "live" if health["staleness_days"] <= 1 else f"{health['staleness_days']}d old"
# Prices are counted in trading days, returns in daily returns: 504 prices
# give 503 returns, and one noun with two counts read as a contradiction.
st.caption(f"Data: {health['rows']} trading days of prices ({len(returns)} daily returns)"
           f" · through {health['end']} · {fresh}")
prov_now = provenance(tickers)
if prov_now:
    _freshness_ticker(prov_now["fetched_at_utc"], live=fresh == "live")
else:
    st.empty()  # hold the ticker's slot: the tab strip below must not move

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
        st.markdown(
            # the palette's warning mark in a Data Quality row, not a yellow
            # emoji in a 60%-opacity caption (2.63:1, Council B)
            '<div class="dq-row"><span class="dq-mark" role="img" aria-label="Warning" '
            'style="color:#6A5030;font-weight:700;">!</span> '
            "<b>Reactive lens.</b> EWMA spikes on one bad day (≈11-day half-life; "
            "at λ=0.94 about 73% of its weight sits in the last month, and it "
            "takes ~37 trading days to reach 90%). It embodies exactly the panic "
            "[Crisis Conviction] argues against - shown for contrast, so you can see "
            "how twitchy risk looks, not because the engine recommends reacting.</div>",
            unsafe_allow_html=True,
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
if bearish:
    # Synthetic daily-rebalanced short of the whole book: negate the
    # PORTFOLIO return stream. Covariance/correlation/eigen panels keep the
    # raw asset returns - those statistics are sign-invariant (council: a
    # "bearish correlation matrix" would be a fabricated distinction).
    port_returns = -port_returns
_audit("Allocation", f"{method}" + (f", vol-targeted to {target_vol:.0%} "
      f"(leverage {leverage:.2f}x)" if use_vt else ""))

# ---- Stress test: custom parametric shock OR historical regime replay ----
alloc_label = "risk-parity" if method == "Risk parity" else "equal-weight"
if bearish:
    alloc_label = f"short {alloc_label}"
alloc_art = "an" if alloc_label[0] in "aeiou" else "a"  # "an equal-weight" / "a short ..."
lev_txt = f", levered {leverage:.2f}×" if use_vt else ""

with deck_stress, st.expander("03 · Stress test - shock or replay a crisis",
                              expanded=False):
    engine = st.radio(
        "Return model", ["Bootstrap (empirical)", "Jump-diffusion (Merton)"],
        horizontal=True,
        help="Bootstrap resamples real historical days - it can only replay tails "
             "it has already seen. Jump-diffusion (Merton 1976) adds Poisson jumps "
             "on top of Gaussian diffusion, so a simulated day can fall further "
             "than any day in the sample. Switch between the two to compare "
             "their tails on this book.")
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
        # Filled once the replay is built: the nominal window claimed the whole
        # crisis even when a late lister had cut the replay short.
        replay_caption = st.empty()

# Build the return distribution + weights the simulation will sample from.
if mode == "Custom shock (sliders)":
    shocked_returns = returns.copy()
    if drawdown_shock != 0:
        # Multiplicative so the year compounds to the labelled drawdown: the
        # additive -0.5/252 compounded to -39%, not the -50% on the slider.
        shocked_returns = (1 + shocked_returns) * (1 + drawdown_shock / 100) ** (1 / TRADING_DAYS) - 1
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
        log_incident(SESSION_REF, f"replay_returns/{mode}", exc)
        st.error(f"Couldn't load history for {mode}: {safe_err('couldn-t-load-history-for-mode', exc)}")
        st.stop()
    sim_assets = list(shocked_returns.columns)
    excluded = [t for t in loaded if t not in sim_assets]
    if len(sim_assets) < 2:
        st.warning(f"Too few of your assets traded during {mode}. Try another scenario.")
        st.stop()
    _verb = "Calibrating to" if engine.startswith("Jump") else "Replaying"  # nothing replays under JD
    replay_caption.caption(f"{_verb} actual market returns from {shocked_returns.index[0].date()} to {shocked_returns.index[-1].date()}.")
    idx = [loaded.index(a) for a in sim_assets]
    sim_weights = weights[idx]
    sim_weights = sim_weights / sim_weights.sum() * weights.sum()  # preserve exposure
    is_shocked = True
    scenario_label = mode

use_jd = engine.startswith("Jump-diffusion")


@st.cache_data(ttl=3600, max_entries=64, show_spinner=f"Simulating {N_PATHS:,} paths…")
def run_monte_carlo(rets: pd.DataFrame, w: np.ndarray, jump: bool) -> dict:
    """Both engines are seeded (seed=42), so a cached result is bit-identical
    to a fresh one. Uncached, the 10,000-path run (0.2-0.5s of CPU) repeated on
    EVERY widget click anywhere on the page, for every visitor. ~0.2MB per
    entry, so the 64-entry cap bounds the cache at roughly 15MB."""
    fn = jump_diffusion_mc if jump else monte_carlo
    return fn(rets, w, n_simulations=N_PATHS, horizon_days=TRADING_DAYS)


# The same rule for the other pure functions of (data, settings) a new visitor
# paid for on every page view. Profiled on a warm server: a NEW session still
# spent ~3s of CPU on these (ES bootstrap 0.8s, map payload 0.7s, and a since-removed signal IC 1.6s),
# and the process serves sessions one at a time, so 10 simultaneous visitors
# queued ~35s for the headline. Both are deterministic (the bootstrap is
# seeded), so caching changes no number.
@st.cache_data(ttl=3600, max_entries=32, show_spinner="Drawing the Risk Topology map…")
def cached_map_payload(rets: pd.DataFrame, w: np.ndarray, names: tuple, short: bool) -> dict:
    return build_map_payload(rets, w, list(names), short)


@st.cache_data(ttl=3600, max_entries=64, show_spinner=False)
def cached_es_ci(r: pd.Series) -> tuple:
    return es_confidence_interval(r)


if bearish:
    # Negate BEFORE calibration/resampling so jump-diffusion calibrates its
    # jumps on the short's own tail (an asset's melt-UP is the short's crash).
    shocked_returns = -shocked_returns
# A day that loses the whole book or more. A volatility shock on a levered book
# can make one (Options audit, 2026-10-30: a short S&P-sector book, EWMA risk
# parity at 26% target vol, a 260% vol shock - April 2025's rally day became a
# loss past 100%). A year cannot be compounded past a wipe-out: the bootstrap
# turned the book's value negative and the jump model took the log of it. Said,
# then stopped, like a crisis too few names traded through.
_worst_day = float((shocked_returns.to_numpy() @ np.asarray(sim_weights, dtype=float)).min())
if _worst_day <= -1:
    st.warning(
        f"At these settings one day of this book loses {-_worst_day:.0%} - more than "
        "everything it holds - so there is no year to simulate past it. Lower the "
        "volatility shock or the target volatility.")
    st.stop()
mc = run_monte_carlo(shocked_returns, np.asarray(sim_weights, dtype=float), use_jd)
_audit("Stress scenario", scenario_label or
      (f"Custom shock (drawdown {drawdown_shock:+d}%, vol {vol_shock:+d}%)"
       if is_shocked else "None (base case)"))
_audit("Monte Carlo", f"{engine}, {mc['n_simulations']:,} paths x {mc['horizon_days']} days -> CVaR {mc['cvar']:.2%}")

# ---- Liquidity-adjusted tail ----
# The CVaR above assumes you're out at the horizon. Widen it for the days it
# actually takes to unwind at DEFAULT_PARTICIPATION of real daily volume, for
# a HEADLINE_BOOK book - named in the sentence, since the days turn on it. The
# interactive version lives in the Liquidity tab; this is the headline default.
HEADLINE_BOOK = 1_000_000
try:
    _adv = load_adv(tuple(tickers)).reindex(loaded).fillna(0.0)
    _dtl = days_to_liquidate(weights, _adv, book_value=HEADLINE_BOOK,
                             participation_rate=DEFAULT_PARTICIPATION)
    lvar = liquidity_adjusted_cvar(mc["cvar"],
                                   liquidity_profile(_dtl)["full_exit_days"])
except Exception:  # noqa: BLE001 - headline must still render if volume feed is down
    lvar = None

# ---- Headline verdict ----
# CVaR is a LOSS: positive means the worst 5% of years lose money. It goes
# negative when even those years end up ahead - a crisis replay over a window
# this book rallied through, or a short book replaying a crash - and "loses an
# average of -57.6%" (found by the recruiter walkthrough) contradicts itself.
_tail = (f"loses an average of <b>{mc['cvar']:.1%}</b>" if mc["cvar"] > 0 else
         f"still ends up an average of <b>{-mc['cvar']:+.1%}</b> - this window gives "
         f"the book no tail loss to measure")
if scenario_label:
    # Jump-diffusion replays nothing: it fits a model to the window. Either
    # way the N-day window becomes a year, and the sentence says so.
    verdict = (
        f"With a jump-diffusion model calibrated to the actual returns of "
        f"{scenario_label} ({len(shocked_returns)} trading days) and run for a "
        f"year, " if use_jd else
        f"Replaying the actual returns of {scenario_label} "
        f"({len(shocked_returns)} trading days), resampled into a year, "
    ) + (f"{alloc_art} {alloc_label} portfolio{lev_txt} "
         f"{_tail} in the worst 5% of simulated years.")
    if excluded:
        # Ticker names are user-supplied and this string is rendered with
        # unsafe_allow_html. VALID_TICKER already forbids "<", so this escape
        # is the second line, not the first - defence that survives someone
        # widening the regex later.
        verdict += (" *(Excludes "
                    + ", ".join(html.escape(t) for t in excluded)
                    + " - not trading for the whole period, or priced at or below zero in it.)*")
else:
    verdict = (
        f"In the worst 5% of simulated years, {alloc_art} {alloc_label} portfolio of these "
        f"{len(loaded)} assets{lev_txt} {_tail}."
    )
    if is_shocked:
        verdict += " *(under the stress scenario applied above)*"
if bearish:
    verdict += (
        " <span style='font-size:12px;color:#6A512E;'>Synthetic daily-"
        "rebalanced short: borrow fees, margin interest and buy-ins are not "
        "modeled - real short results are worse. Short losses can exceed "
        "100% of capital.</span>"
    )

# Only surface the liquidity add-on when it materially fattens the tail
# (multiplier > 1.005 ≈ more than ~2.5 trading days to fully exit). A CVaR
# <= 0 has no tail loss to widen; the line read "widens to" a gain.
if (mc["cvar"] > 0 and lvar and np.isfinite(lvar["lvar"])
        and lvar["multiplier"] > 1.005):
    verdict += (
        f" Adjusted for the ~<b>{lvar['full_exit_days']:.0f} trading days</b> "
        f"it takes to fully unwind a ${HEADLINE_BOOK:,.0f} book at "
        f"{DEFAULT_PARTICIPATION:.0%} of each name's average daily dollar volume, "
        f"that tail widens to "
        f"<b>{lvar['lvar']:.1%}</b>."
    )

# ---- Headline accents (ME-11): each one a state this run computed ----
# The orientation promises "watch the headline number move"; a number that
# changes in place is easy to miss. A hairline 0-100% scale (no loss to the
# whole book) carries a tick at this CVaR and a faint one at the last
# DIFFERENT headline this session showed, and the move is said in words.
# Marks name the two states that change what the number means: a shock or
# crisis replay applied above, and the short side. All inside the verdict's
# own markdown, so nothing above the tab strip gains an element.
def cvar_scale_html(now: float, prev: float | None) -> str:
    """The headline's scale, its ticks and the move in words.

    Everything is read from the headlines AS DISPLAYED (one decimal): the raw
    values 22.540% and 22.480% both read "22.5%", and the page said "-0.1
    points vs your last setting" between two equal numbers (Council B,
    2026-10-22). Equal displays say nothing. A CVaR past 100% (a leveraged or
    short book can lose more than it holds) sits at the scale's end, and the
    end says so instead of clamping silently."""
    shown = lambda v: float(f"{v:.1%}"[:-1])                 # exactly what the headline prints
    on_scale = lambda v: min(max(shown(v), 0.0), 100.0)      # a gain sits at 0
    beyond = shown(now) > 100
    moved = None if prev is None else round(shown(now) - shown(prev), 1)
    return (
        '<div class="cvar-scale" aria-hidden="true">'
        + (f'<span class="cvar-tick prev" style="left:{on_scale(prev):.1f}%"></span>'
           if prev is not None else "")
        + f'<span class="cvar-tick now{" beyond" if beyond else ""}" style="left:{on_scale(now):.1f}%;'
          f'--from:{on_scale(now if prev is None else prev):.1f}%"></span></div>'
        '<div class="cvar-scale-ends" aria-hidden="true"><span>0%</span>'
        + ('<span class="beyond">beyond the whole book &#8250;</span>' if beyond
           else '<span>the whole book</span>') + '</div>'
        + (f'<div class="cvar-delta">{"+" if moved > 0 else chr(0x2212)}{abs(moved):.1f} '
           f'points vs your last setting</div>' if moved else ""))


# The last DIFFERENT headline this session displayed (compared as displayed).
_cv = float(mc["cvar"])
if st.session_state.get("_cvar_now") is None or f"{st.session_state['_cvar_now']:.1%}" != f"{_cv:.1%}":
    st.session_state["_cvar_prev"] = st.session_state.get("_cvar_now")
    st.session_state["_cvar_now"] = _cv
_scale = cvar_scale_html(_cv, st.session_state.get("_cvar_prev"))
_marks = "".join(
    f' <span class="state-mark" title="{why}">{name}</span>'
    for name, why, on in (
        ("Stressed", "A shock or a crisis replay is applied in 03 · Stress test", is_shocked),
        ("Short book", "The bearish side: a synthetic daily-rebalanced short", bearish))
    if on)

# ---- Verdict + the cone of simulated outcomes: one wide row ----
v_col, f_col = st.columns([5, 7], gap="large")
with v_col:
    st.markdown(f"""
<div class="verdict-box">
  <div class="verdict-label">Bad-year loss · 1-year CVaR (95%){_marks}</div>
  <div class="verdict-number">{mc['cvar']:.1%}</div>
  {_scale}
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
            'frontier of the worst 5%</b>; the CVaR headline is the average loss '
            'beyond that edge, so the CVaR number sits deeper than the cone shows. '
            'Change any setting and watch the cone breathe.'
            '</div>', unsafe_allow_html=True)
with f_col:
    st.markdown("""
<div class="engrave line" aria-hidden="true"><svg viewBox="-60 -4 66 60" xmlns="http://www.w3.org/2000/svg">
  <!-- a quarter of the crest's three rings, fading outward (assets/logo.svg) -->
  <g fill="none" stroke="#9A7B4F">
    <circle r="47.2" stroke-width="2.4"/>
    <circle r="51.4" stroke-width="1.5" stroke-opacity=".6"/>
    <circle r="55.6" stroke-width="1" stroke-opacity=".3"/>
  </g>
</svg></div>
""", unsafe_allow_html=True)
    st.plotly_chart(fan_chart(mc["path_bands"]), width="stretch", config=PLOTLY_CFG)
    # Jump-diffusion resamples nothing, and a replay under 30 days falls back to
    # i.i.d. (block_days=1): both used to claim "crash weeks stay intact".
    _se_txt = (f" CVaR sampling error: ±{mc['cvar_se']:.2%} "
               f"({mc['n_simulations']:,} paths, "
               + ("parametric Merton paths" if mc.get("engine") == "jump-diffusion"
                  else f"resampled in blocks averaging {mc['block_days']} days"
                  if mc.get("block_days", 1) > 1
                  else "resampled day by day (window too short for blocks)")
               + " - a simulated estimate, not an exact truth)."
               if np.isfinite(mc.get("cvar_se", float("nan"))) else "")
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
              help="Share of total universe VARIANCE carried by the single "
                   "dominant statistical factor. Read it as where the risk "
                   "sits, not as proof that everything moves together: this "
                   "runs on the covariance matrix, so one unusually volatile "
                   "name can carry PC1 on its own size alone, with no "
                   "co-movement behind it. The correlation matrix on the "
                   "Correlation Watch tab is what answers 'do they move "
                   "together'.")
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
    # EWMA weights carry (1+lam)/(1-lam) ~ 32 effective days, not 503: at
    # T=503 the ceiling sat 1.95x too low and painted noise as signal.
    mp_t = ((1 + RISKMETRICS_LAMBDA) / (1 - RISKMETRICS_LAMBDA)
            if cov_method == "EWMA" else len(returns))
    _, mp_hi = marcenko_pastur_bounds(len(lam), mp_t, sigma2)
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
        f"would produce at this sample size (N={len(lam)}, T={mp_t:.0f}"
        f"{' effective days under EWMA' if cov_method == 'EWMA' else ''}). "
        "Heuristic reference at this universe size, not a hard test - the "
        "Ledoit-Wolf estimator is the production defense against inversion "
        "noise. Flip the covariance estimator to EWMA in 02 · Allocation to "
        "see the CURRENT regime's factor structure instead of the "
        f"average over {len(returns)} daily returns.")

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
# ---- The Book: the guided arc, deliberately AHEAD of the tab strip ----
# The tabs answer sixteen questions and leave the reader to assemble them.
# This answers them in order, and each answer narrows the next: book size sets
# what each name can absorb, that sets who is eligible, that sets who can lead,
# and only then is anything paired. See src/narrative.py.
st.markdown('<div id="the-book"></div>', unsafe_allow_html=True)
panel_head("The Book",
           "Capital first, then the names that lead it, then what sits beside them")
_bk1, _bk2 = st.columns(2)
_book_size = _bk1.number_input(
    "Book size ($)", min_value=10_000, max_value=5_000_000_000,
    value=1_000_000, step=100_000, key="book_size",
    help="The parameter the whole section turns on. Raise it and names drop "
         "out - not because they got worse, but because you can no longer "
         "leave them.")
_book_part = _bk2.slider(
    "Max daily participation (% of volume)", 5, 50,
    round(DEFAULT_PARTICIPATION * 100), step=5, key="book_part",
    help="How much of a name's daily dollar volume you would be before your own "
         "trading moves the price. Risk desks use ~10-20%.") / 100
# Overwhelm audit (2026-10-29): the book leads with its paragraph and the
# names that carry it - the leaders (grit) and their anchors (the pairing).
# The names with no role, a run of near-identical sentences, fold into one
# drawer with the two finer settings - same widgets, same numbers.
_bk_box = st.container()
_bk_head = _bk_box.empty()
_bk_lead = _bk_box.container()
_bk_more = _bk_box.expander("The other names, line by line - and two finer settings")
_bk3, _bk4 = _bk_more.columns(2)
_bk_floor = _bk3.toggle(
    "Flyers must be in the grittiest half", value=False, key="book_floor",
    help="Off: a recovery record only has to EXIST, and momentum does the "
         "ranking. On: a leader must also rank in the grittiest half. Both "
         "readings of 'grittiest high-flyer' are defensible - this shows the "
         "difference rather than deciding it for you.")
_bk_fb = _bk4.toggle(
    "Allow a defensive anchor from outside the universe", value=True,
    key="book_fallback",
    help="A basket of mega-caps contains nothing that thins a tail. When "
         "nothing inside your universe cushions, this lets the engine reach "
         "for TLT/GLD/XLP/XLU/USMV - and say on screen that it did.")

# One slot, whatever the basket: The Book writes one card per name, and
# Streamlit keys elements by position, so a card count that changed with the
# basket moved the tab strip below it and left the old map on screen, faded,
# beside the new one (John, 2026-10-19). The cards live in the drawer above.
with _bk_more:
    try:
        _bk_px = prices
        _bk_dv = load_dollar_volume(tuple(tickers)).reindex(
            columns=loaded).reindex(prices.index)
        _bk_anchors = None
        if _bk_fb:
            _a_px, _a_dv = load_anchor_shelf(tuple(DEFENSIVE_ANCHOR_TICKERS))
            _keep = [c for c in _a_px.columns if c not in loaded]
            if _keep:
                _bk_px = pd.concat(
                    [prices, _a_px[_keep].reindex(prices.index)], axis=1)
                _bk_dv = pd.concat(
                    [_bk_dv, _a_dv[_keep].reindex(prices.index)], axis=1)
                _bk_anchors = set(_keep)
        # Grit from FULL history over the visitor's own names - the Grit Zone
        # tab's exact call - not from the 2y `prices`, where no named crisis
        # falls (NVDA 6.9 here vs 73.2 in the tab, council 2026-10-17), and not
        # ranked against the defensive shelf (V 67 here vs 61 in the tab,
        # ME-08). The shelf's grit (basket plus shelf) only orders the shelf
        # when an outside anchor is screened; no sentence prints it.
        # Missing volume days stay NaN: build_book's mean and 5th percentile
        # skip them. Zero-filled, 4 gaps in 63 days dragged the quiet-tape
        # volume to 0 and printed "reports no volume on this feed" (ME-09).
        _the_book = load_book(
            _bk_px, _bk_dv, float(_book_size), _book_part,
            _bk_floor, tuple(sorted(_bk_anchors or ())),
            load_grit(tuple(loaded)),
            load_grit(tuple(_bk_px.columns)) if _bk_anchors else None)

        _bk_head.markdown(
            f'<div style="background:#F1EDE5;border:1px solid #C4BDAE;'
            f'border-top:2px solid #9A7B4F;border-radius:12px;padding:18px 22px;'
            f'margin:6px 0 18px;font-family:Georgia,serif;font-size:17px;'
            f'color:#3F3B35;line-height:1.5;">{headline(_the_book)}</div>',
            unsafe_allow_html=True)

        # Border tone carries the role (decorative, so bronze is allowed); the
        # LABEL is words, so it uses the sanctioned on-beige text tone written as a
        # LITERAL. test_text_colours_meet_wcag_aa scans main.py for `color:#RRGGBB`
        # and an f-string placeholder hides the value from it - #8A6A3C sat at
        # ~3.1:1 here unseen because it arrived through a variable.
        # The chip's words come from narrative.role_label: a name with no role
        # can still hold capital, so "LEFT OUT" is only for a name holding none.
        _CHIP = {Role.FLYER: ("#8A6A3C", "#F6F2EA"),
                 Role.CUSHION: ("#3F3B35", "#EFEAE0"),
                 Role.EXCLUDED: ("#6B6459", "#EDE9E3")}
        for _d in _the_book:
            (_col, _bg), _lbl = _CHIP[_d.role], role_label(_d)
            # A NO ROLE card shows what it holds, so "still held" can be
            # checked on the card; on its own line, so the 86px chip wraps
            # rather than cuts at 360px.
            _note = role_note(_d)
            if _note:
                _lbl += (f'<br><span class="role-note" style="letter-spacing:.06em;">'
                         f'{_note}</span>')
            _flag = (' &middot; <span style="color:#6A5030;">outside the chosen '
                     'universe</span>' if _d.is_fallback_anchor else "")
            # Redundant on a card that was already excluded FOR capacity - the
            # sentence there already names the breakpoint.
            _bind = (' &middot; <span style="color:#6A5030;">liquidity set this '
                     'size, not risk</span>'
                     if _d.liquidity_binding and _d.role is not Role.EXCLUDED else "")
            # leaders and anchors stay on the page; the rest is in the drawer
            (_bk_lead if _d.role in (Role.FLYER, Role.CUSHION) else st).markdown(
                f'<div style="display:flex;gap:14px;align-items:flex-start;'
                f'background:{_bg};border-left:3px solid {_col};border-radius:8px;'
                f'padding:12px 16px;margin-bottom:8px;">'
                f'<div style="flex:0 0 86px;font-family:Helvetica Neue,sans-serif;'
                f'font-size:9.5px;letter-spacing:.16em;color:#6A5030;'
                f'padding-top:3px;">{_lbl}</div>'
                f'<div style="flex:1;font-family:Georgia,serif;font-size:14px;'
                f'color:#3F3B35;line-height:1.55;">{_d.sentence()}{_flag}{_bind}</div>'
                f'</div>',
                unsafe_allow_html=True)

        read_me(
            "<b>Read it top to bottom.</b> Every line is a slot-filled template "
            "over measured numbers - the renderer is handed the dossier and no "
            "price data, so it cannot say anything the engine did not compute. "
            "<b>NO ROLE</b> names are still held, at the size risk parity and "
            "liquidity give them, just without a lead or anchor role; "
            "<b>LEFT OUT</b> names hold nothing. "
            "Change the book size and watch names leave: that is the liquidity "
            "constraint doing real work, not a caption about it.")
    except Exception as _bk_exc:  # noqa: BLE001 - the page must never die on data
        # safe_err, not str(exc): the message can carry a server path (see safe_err).
        _bk_head.info(
            f"The Book needs price and volume history for this universe "
            f"({safe_err('the-book', _bk_exc)}). No placeholder book is shown.")

st.markdown('<div id="analysis"></div>', unsafe_allow_html=True)
panel_head("Risk & conviction", "The analysis - where the risk lives")
(tab_3d, tab_breakdown, tab_liquidity, tab_watch, tab_balance, tab_grit,
 tab_conviction) = st.tabs([
    "Risk Topology", "Risk Breakdown", "Liquidity", "Correlation Watch",
    "Balance", "Grit Zone", "Crisis Conviction",
])
# Overwhelm audit (2026-10-29): the research workshop (Signal Lab, Regime
# Atlas, Security Master) asked no question a visitor has to decide on, and it
# stood open by default. What remains here is the provenance a reader can
# audit - folded, one click away, never in the way.
with st.expander("Where the numbers come from - data checks and audit trail"):
    tab_dq, tab_lineage = st.tabs(["Data Quality", "Lineage & Audit"])

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
        m1.metric(f"{pick_a} × {pick_b} now ({win}d)",
                  f"{latest:+.2f}" if len(roll) else "-")
        # Not an average of the rolling series - it is the single Pearson
        # correlation over the whole sample, which is a different number.
        m2.metric("Full-period correlation", f"{static_corr:+.2f}")
        m3.metric("Range over history",
                  f"{roll.min():+.2f} … {roll.max():+.2f}" if len(roll) else "-")

        # 60 prices leave 59 returns while the slider reaches 63: an empty
        # rolling series used to read as "Stable ... at +nan".
        if not len(roll) or not np.isfinite(latest):
            st.info(f"Not enough history for a {win}-day window - shorten the "
                    "rolling window to get a reading.")
        elif latest > thresh:
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
            if bearish:
                # Every other tail number on this page is computed on the
                # negated book (see the port_returns branch above). This panel
                # was not, so in bearish mode it reported the LONG book's CVaR
                # under a short book's heading.
                pr_before, pr_after = -pr_before, -pr_after
            cv_b, cv_a = cvar(pr_before), cvar(pr_after)
            vol_b = float(pr_before.std() * np.sqrt(TRADING_DAYS))
            vol_a = float(pr_after.std() * np.sqrt(TRADING_DAYS))

            panel_head("Defensive simulation",
                       f"Cut {pick_a} & {pick_b} by up to 15pts each, "
                       f"move the freed weight into {dest}")
            d1, d2 = st.columns(2)
            d1.metric("Daily CVaR (95%)", f"{cv_a:.2%}",
                      delta=f"{(cv_a - cv_b) * 100:+.2f} points vs current",
                      delta_color="inverse")
            d2.metric("Annualized vol", f"{vol_a:.1%}",
                      delta=f"{(vol_a - vol_b) * 100:+.1f} points vs current",
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
    # Risk Topology - the Monte Carlo re-drawn as a strategy map. Every
    # number in the payload traces to yfinance history through src/ code:
    # positions from rolling windows, dynamics from the OU calibration,
    # pair statistics from the real-history backtest. Nothing typed in.
    try:
        _wr_payload = cached_map_payload(returns, np.asarray(weights, dtype=float),
                                         tuple(loaded), bearish)
        # width is explicit, not left to the default. On streamlit 1.63.0 the
        # element container for an st.iframe inside a tab collapsed to 16px -
        # the map rendered fully (tiles, canvas, payload all present in the
        # srcdoc) into a sliver nobody could see. 1.62.0 stretched by default.
        # its own keyed container (.st-key-risk-map): the glide skips the
        # map's rendering while it flies past it off screen (see quietMap)
        with st.container(key="risk-map"):
            st.iframe(war_room_html(_wr_payload), width="stretch", height=820)
        st.caption(
            "The Monte Carlo as a strategy map. The grid is (beta, realized "
            "vol) - beta is how far the book moves when the market moves one "
            "point, so 1.0 tracks the market, above 1.0 swings harder; "
            "realized vol is how much it has actually been moving. The lit "
            "terrain holds 68% of simulated 30-day end-states, "
            "night falls at 99.7%, and the perimeter marks the disclosed "
            "hazard policy. Tracers are simulated state paths from dynamics "
            "calibrated on this book's actual history; linkage statistics "
            "replay real history. Simulated estimates, not forecasts."
        )
    except Exception as _wr_exc:  # noqa: BLE001 - never fake the map
        st.info(
            "Risk Topology needs live market history to calibrate "
            f"({safe_err('risk-topology', _wr_exc)}). No demo numbers are "
            "shown in the product - reload when data is available."
        )

with tab_breakdown:
    st.caption(f"Universe ({len(loaded)}): {', '.join(loaded)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median 1-year return", f"{mc['median_return']:+.1%}")
    c2.metric("Probability of loss", f"{mc['prob_loss']:.1%}")
    c3.metric("Worst simulated year", f"{mc['worst_case']:+.1%}")

    # --- Risk-contribution decomposition (where the risk actually lives) ---
    panel_head("Risk contribution by asset", "Where the risk actually lives")
    # base_weights, NOT the vol-targeted ones: risk_pct is scale-invariant and
    # always sums to 100%, so plotting LEVERED dollar weights beside it put two
    # different denominators under one "% of portfolio" axis - at 0.6x leverage
    # every weight bar shrank and read as a risk/weight gap that was pure
    # leverage. Leverage scales the whole book and cannot change the split.
    rc = risk_contributions(base_weights, cov)
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
        + (f" Both bars are shares of the book itself; the {leverage:.2f}× "
           "vol-target overlay scales the whole book and does not change this "
           "split." if use_vt else "")
    )

    panel_head("Risk-adjusted performance", "Sharpe vs the real T-bill rate")
    rf = load_risk_free_rate()
    ann_ret = float(port_returns.mean()) * TRADING_DAYS
    ann_vol = float(port_returns.std()) * np.sqrt(TRADING_DAYS)
    # Unlevered book (base_weights): on the vol-targeted weights Sharpe was
    # (L*mu - rf)/(L*sigma) - idle cash at 0, borrowing free - so the slider
    # moved it 1.615 -> 0.975 for the same strategy. With cash and financing at
    # rf, Sharpe is leverage-invariant (Sharpe 1994). Council 2026-10-17.
    sharpe = sharpe_ratio((-1 if bearish else 1) * (returns @ base_weights),
                          rf if rf is not None else 0.0)
    s1, s2, s3 = st.columns(3)
    s1.metric("Sharpe ratio", f"{sharpe:.2f}")
    s2.metric("Annualized return", f"{ann_ret:+.1%}")
    s3.metric("Annualized volatility", f"{ann_vol:.1%}")
    rf_txt = (f"{rf:.2%} (13-week T-bill, ^IRX)" if rf is not None
              else "unavailable - Sharpe computed against 0%")
    st.caption(
        f"Sharpe = (annualized return − risk-free) / annualized volatility, on "
        f"the real (unshocked) "
        f"{'synthetic short book (daily-rebalanced; borrow costs not modeled)' if bearish else 'portfolio'}. "
        f"Risk-free rate: {rf_txt}."
        + (" Sharpe is taken on the unlevered book: the vol-target overlay only "
           "rescales it, and with idle cash and borrowing at the risk-free rate "
           "rescaling leaves Sharpe unchanged. Return and volatility beside it "
           "are the levered book's." if use_vt else "")
    )


    panel_head("Correlation matrix", "Do these names move together?")
    read_me(
        "Each cell is how tightly two names move together: <b>bronze = lockstep "
        "(+1)</b>, <b>beige = independent (0)</b>, <b>charcoal = seesaw (−1)</b>. "
        "A book full of deep bronze has little real diversification - everything "
        "falls at once; charcoal cells are the offsets. The empty upper half is "
        "the same data mirrored, masked so the eye reads each pair once. "
        "Names are ordered by a correlation tree, so the ones that move "
        "together sit next to each other and each group reads as one block. "
        "Sample: the returns AFTER the stress-test choice above - under a "
        "crisis replay this matrix is that crisis window, not the "
        f"{len(returns)} daily returns the Correlation Watch tab averages.")
    corr = correlation_matrix(shocked_returns)
    # Reorder BEFORE the mask and the bold text are built, so both follow the
    # new order. In input order a block of names that move together is
    # scattered across the grid; a permutation changes no number.
    order = cluster_order(corr)
    corr = corr.loc[order, order]

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
            f"**~{jp['lambda_daily'] * TRADING_DAYS:.1f} jumps/year** on a diffusion vol of "
            f"{jp['sigma_d'] * np.sqrt(TRADING_DAYS):.0%}. Poisson jumps let the tail run "
            "further than any day in the sample; switch the return model to the "
            "bootstrap to compare the two tails on this book."
        )

    # --- Deep dive: the three statistical-test-heavy panels fold behind one
    #     click so a cold viewer meets the intuitive charts (returns, risk
    #     contribution, correlation) first, and the model-validation depth
    #     second - "lead with one number", don't wall them with seven charts. ---
    with st.expander("Deep dive - model validation & factor structure"):
        # --- VaR methods + backtest (validates the model, not just reports it) ---
        panel_head("Value at Risk - methods & backtest", "The daily loss line, and whether it holds up")
        hist_var = var(port_returns)
        bt = var_backtest(port_returns)
        _chr = bt.get("christoffersen", {})
        # Two dimensions, two words. Kupiec answers "how many?", Christoffersen
        # answers "did they arrive together?" - a model can pass one and fail
        # the other, and clustering is the more dangerous failure.
        _clump = ("n/a" if _chr.get("passed_ind") is None
                  else "Independent" if _chr["passed_ind"] else "CLUSTERED")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Historical VaR (95%)", f"{hist_var:.2%}")
        v2.metric("Parametric VaR (95%)", f"{parametric_var(port_returns):.2%}")
        v3.metric("VaR breaches", f"{bt['breaches']} / {bt['expected_breaches']:.0f} exp.")
        v4.metric("Breach timing", _clump,
                  help="Kupiec counts breaches; this asks whether they arrived "
                       "independently or bunched together. Breaches that cluster "
                       "mean the model holds in calm markets and fails in stress - "
                       "exactly when the number matters. 'n/a' means too few "
                       "breaches to estimate it, not a pass.")
        verdict_word = ("passes" if bt["passed"] else
                        "fails" if bt["passed"] is False else "cannot be computed for")
        _kupiec_clause = (
            f"the model's breach rate of {bt['observed_rate']:.1%} is "
            "statistically consistent with the 5% it claims" if bt["passed"]
            else f"the model's breach rate of {bt['observed_rate']:.1%} is "
                 "statistically INCONSISTENT with the 5% it claims - the VaR "
                 "model misstates its own tail on this sample")
        # Christoffersen. Reported separately because it answers a different
        # question, and because "n/a" must never read as a pass.
        if _chr.get("lr_ind") is None:
            _cc_clause = (
                f"**Breach timing:** only {bt['breaches']} breach(es) in "
                f"{bt['n']} days - too few consecutive-day transitions to "
                "estimate P(breach | breach yesterday), so the Christoffersen "
                "independence test is not defined here. Not a pass: undefined."
            )
        else:
            _ind_word = "independent" if _chr["passed_ind"] else "CLUSTERED"
            _cc_word = "passes" if _chr["passed_cc"] else "fails"
            _cc_clause = (
                f"**Breach timing (Christoffersen 1998):** breaches look "
                f"{_ind_word} - P(breach | breach yesterday) = "
                f"{_chr['pi11']:.1%} vs {_chr['pi01']:.1%} otherwise; "
                f"LR_ind = {_chr['lr_ind']} against a 3.84 critical value "
                f"(p = {_chr['p_ind']:.3f}). Combined conditional coverage "
                f"{_cc_word}: LR_cc = LR_uc + LR_ind = {_chr['lr_cc']} "
                f"against 5.99 on 2 dof (p = {_chr['p_cc']:.3f}). "
                + ("Count and timing both hold up."
                   if _chr["passed_cc"] else
                   "A model can pass on count and still fail here - clustered "
                   "breaches mean it works in calm markets and breaks in stress.")
            )
        if not bt["testable"]:
            # Walk-forward needs an estimation window plus at least one day to
            # hold out. Say so rather than render a verdict from no days.
            st.caption(
                f"Not enough history for a walk-forward check: "
                f"{len(port_returns)} days is under the {bt['window']}-day "
                "estimation window, so there are no out-of-sample days to "
                "test. Widen the date range."
            )
        else:
            st.caption(
                "**What this is:** a report card on the risk model itself. "
                "It counts how often the real loss was worse than the model's "
                "own daily prediction, and asks whether that count is close "
                "enough to what the model promised. Too many misses means the "
                "model understates risk; far too few means it is needlessly "
                "gloomy. "
                f"Daily VaR check {verdict_word} the Kupiec "
                f"proportion-of-failures test (Kupiec 1995; "
                f"LR = {bt['kupiec_lr']}, 95% critical = 3.84): "
                f"{_kupiec_clause}. Walk-forward: each day's VaR is estimated "
                f"from the {bt['window']} trading days strictly before it, then "
                f"tested against that day's actual return - {bt['n']} "
                "out-of-sample days."
            )
            st.caption(_cc_clause)

        # --- Tail shape (EVT) - the question CVaR cannot answer -------------
        panel_head("Tail shape - how bad can it get",
                   "What the average of the tail still doesn't tell you")
        try:
            _tf, _tf_days, _tf_dropped, _tf_years = load_tail_fit(
                tuple(loaded), tuple(float(x) for x in weights), bool(bearish))
        except Exception as exc:  # noqa: BLE001 - never take the page down
            _tf, _tf_days, _tf_dropped, _tf_years = None, 0, [], 0.0
            st.caption(f"Tail model unavailable: {safe_err('tail-model-unavailable', exc)}")

        if _tf is not None and _tf["fitted"]:
            _xi, _ci = _tf["xi"], _tf["xi_ci"]
            t1, t2, t3 = st.columns(3)
            t1.metric("Tail index ξ (filtered)", f"{_xi:+.2f}",
                      help="Shape of the loss tail from a Generalized Pareto "
                           "fit on VOLATILITY-STANDARDISED residuals, so it "
                           "measures how heavy the shocks really are rather "
                           "than how bunched they arrived. ξ>0 = heavy tail "
                           "with NO finite worst case; ξ<0 would mean a "
                           "bounded tail. Bigger ξ = fatter.")
            t2.metric("Worst case",
                      "None exists" if _xi >= 0 else f"{_tf['finite_endpoint_return']:.1%}",
                      help="A finite maximum loss exists only if ξ<0. For "
                           "equities ξ is essentially always >0, so the honest "
                           "answer is that no finite worst case exists - the "
                           "only hard floor is -100% from limited liability.")
            _es999 = _tf["conditional"].get(0.999, {}).get("es")
            t3.metric("ES 99.9% tomorrow",
                      "undefined" if _es999 is None else f"{_es999:.1%}",
                      help="Average loss on a 1-in-1000 day, for TOMORROW: the "
                           "fitted residual tail rescaled by the current "
                           "volatility forecast. It moves with the market's "
                           "state, unlike a static full-sample number. "
                           "'undefined' means ξ≥1 - the mean itself does not "
                           "exist, so no ES can.")

            _ci_txt = ("" if _ci is None else
                       f" (95% bootstrap interval {_ci[0]:+.2f} to {_ci[1]:+.2f})")
            _shape = ("**no finite worst case exists** - the tail decays "
                      "polynomially, so there is no loss level the model rules "
                      "out. The only hard floor is -100%."
                      if _xi >= 0 else
                      f"the fit implies a bounded tail ending at "
                      f"{_tf['finite_endpoint_return']:.1%}. Treat that with suspicion: "
                      "equity losses almost always fit ξ>0, so a negative ξ on "
                      "this sample is more likely estimation noise than a real bound.")
            _broken = [k for k, v in _tf["moments_finite"].items() if not v]
            st.caption(
                f"Fitted a Generalized Pareto to the "
                f"{_tf['n_exceedances']} losses above the "
                f"{_tf['threshold_quantile']:.0%} threshold "
                f"({_tf['threshold_return']:.2%} of the book at today's volatility; "
                f"{_tf['threshold_z']:.2f} in standardised units), drawn from "
                f"{_tf_days} daily returns "
                f"(~{_tf_years:.1f} years - the {len(returns)} daily returns used elsewhere leave too few "
                f"exceedances to fit honestly). **ξ = {_xi:+.2f}**{_ci_txt}: {_shape}"
                + (f" At this ξ the {', '.join(_broken)} of the loss "
                   f"distribution {'is' if len(_broken) == 1 else 'are'} "
                   "infinite, so any statistic relying on "
                   f"{'it' if len(_broken) == 1 else 'them'} is meaningless here."
                   if _broken else "")
                + (f" *{', '.join(_tf_dropped)} lack{'s' if len(_tf_dropped) == 1 else ''}"
                   " the full history and were excluded from this fit.*"
                   if _tf_dropped else "")
            )
            _uxi, _snext = _tf.get("unconditional_xi"), _tf.get("sigma_next")
            if _uxi is not None:
                _gap = _uxi - _xi
                st.caption(
                    f"**Conditional EVT (McNeil-Frey 2000).** Peaks-over-threshold "
                    f"assumes exceedances are independent, and the Christoffersen "
                    f"test above is there to check exactly that - so the GPD here "
                    f"is fitted to **volatility-standardised residuals**, not raw "
                    f"returns. Fitting the raw series instead gives ξ = {_uxi:+.2f} "
                    f"against {_xi:+.2f} filtered, a gap of {_gap:+.2f}: that "
                    f"difference is volatility clustering masquerading as a fatter "
                    f"tail, and it is what an unconditional fit would have "
                    f"reported as danger. Volatility is filtered with EWMA "
                    f"(λ={_tf.get('lam', 0.94)}, the RiskMetrics daily decay), "
                    f"which is IGARCH(1,1) with λ fixed - so **no volatility "
                    f"parameter is fitted to this sample** and the filter cannot "
                    f"overfit it. The paper specifies AR(1)-GARCH(1,1); this is "
                    f"the deliberate simplification. Tomorrow's numbers rescale "
                    f"the residual tail by the current volatility forecast "
                    f"σ = {_snext:.2%} per day."
                )
            st.caption(
                "Honest limits: ξ is estimated by maximum likelihood on "
                "exceedances over a 95% threshold, which is known to sit a "
                "little BELOW the asymptotic value - so read the gap between "
                "the two fits, which is robust, rather than either level as "
                "exact. The interval is a percentile bootstrap over "
                "exceedances: it captures estimation noise, not the risk of "
                "having picked the wrong threshold. The conditional mean is "
                "taken as zero, since at a one-day horizon equity drift is "
                "~2 orders of magnitude below daily volatility."
            )
        elif _tf is not None:
            st.caption(
                f"No tail fit: {_tf['reason']}. Extreme Value Theory needs "
                "enough observations above the threshold before it says "
                "anything - reporting a shape from fewer would be a guess "
                "wearing a parameter's name."
            )

        # --- Named factor exposures ---
        panel_head("Factor exposures", "What systematic bets is this book taking?")
        try:
            fx = factor_exposures(port_returns)
            st.plotly_chart(hbar(pd.Series(fx["betas"]), title_x="beta"),
                            width="stretch", config=PLOTLY_CFG)
            st.caption(
                f"Market beta {fx['betas']['Market']:+.2f} · "
                f"R-squared {fx['r_squared']:.0%} · "
                f"annualized alpha {fx['alpha_annual']:+.1%} ({fx['alpha_basis']}). "
                "Size (IWM-SPY) and Momentum (MTUM-SPY) are tilts vs. the broad market; "
                "Value is IWD-IWF, value minus growth, not versus the market. "
                "ETF-proxy factors, not the academic research series."
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Factor exposures unavailable: {safe_err('factor-exposures-unavailable', exc)}")

        # --- Statistical risk factors (eigendecomposition / PCA) ---
        panel_head("Statistical risk factors",
                   "Eigendecomposition - how many independent bets is this book?")
        try:
            eigen_factor_panel(cov, weights, returns)
        except Exception as exc:  # noqa: BLE001 - degrade like the panel above
            st.caption(f"Statistical risk factors unavailable: {safe_err('statistical-risk-factors-unavailable', exc)}")

source_txt = f"the {scenario_label} window" if scenario_label else (
    f"the {len(shocked_returns)} daily returns loaded "
    f"({shocked_returns.index[0].date()} to {shocked_returns.index[-1].date()})")
engine_txt = (
    "Merton jump-diffusion (Poisson jumps on Gaussian diffusion), calibrated to"
    if use_jd else "bootstrap Monte Carlo, resampled from"
)
st.caption(
    f"Methodology: {mc['n_simulations']:,}-path {engine_txt} {source_txt}, over a {mc['horizon_days']}-day horizon. "
    f"{alloc_label.capitalize()} allocation{lev_txt}."
)

with tab_balance:
    st.caption(
        "Pick a name you hold; the engine ranks every other name in your universe "
        "by how it moves against it. A negatively-correlated partner offsets part "
        "of the anchor's swings - diversification computed from real covariance, "
        "not a story about the future."
    )
    if len(loaded) < 2:
        st.caption("Balance needs at least two assets in the universe.")
    else:
        try:
            # The blend below is sized on the selected estimator's `cov`, so the
            # partner is ranked on the same matrix: under EWMA the sample
            # correlation differed by up to 0.47 on the offline market (ME-09).
            corr_b = correlation_from_cov(cov)
            # Keyed, with the default seeded only when no valid choice is
            # stored: an index= that followed argmax(weights) re-created the
            # widget on every Weighting change and silently replaced the
            # visitor's anchor (council 2026-10-17).
            if st.session_state.get("balance_anchor") not in loaded:
                st.session_state["balance_anchor"] = loaded[int(np.argmax(weights))]  # your biggest position
            anchor = st.selectbox(
                "Anchor (the position you want to balance)", loaded,
                key="balance_anchor")
            ranked = rank_hedges(corr_b, anchor)

            # Classify by correlation STRENGTH, not just sign: a name at −0.01
            # is independent, not a hedge, and must not be painted as one.
            HEDGE, INDEP = -0.20, 0.20   # bands: <−0.2 offsets · ±0.2 independent

            def _hedge_color(v):
                if v < HEDGE:
                    return "#33582F"          # green - genuinely moves against
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
                "reduction, from diversification. Long-only minimum-variance weights, "
                f"on the {cov_method} risk matrix (02 · Allocation)."
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

            # ---- BON VOYAGE: long-only defensive pairing ----------------
            # "What goes up must come down. We can't cap the fall; we
            # measure the cushion." Council pass 6: cushion never cap,
            # ES@97.5 never CVaR@99, ranks never promises.
            @st.cache_data(ttl=6 * 3600, max_entries=32, show_spinner="Reading the defensive shelf…")
            def _bv_anchor_returns(universe_key: str):
                """Universe returns joined with defensive ETF candidates
                (Treasuries/gold/staples/utilities/min-vol) - an equity
                basket usually cannot anchor itself."""
                return anchor_frame(returns)

            @st.cache_data(ttl=6 * 3600, max_entries=64,
                           show_spinner="Replaying this pair through each named crisis…")
            def _bv_crisis(a: str, b: str, w: float, short_a: bool = False):
                return crisis_cushion(a, b, w, short_a=short_a)

            @st.cache_data(ttl=6 * 3600, max_entries=32,
                           show_spinner="Reading full price histories to rank the anchors…")
            def _bv_grit(tickers: tuple):
                """Grit scores for anchor candidates - the engine's founding
                question ("who gets back up?") IS the Circle 2 screen."""
                return grit_scores(list(tickers))["scores"]["grit_score"]

            panel_head("The defensive pair",
                       "What goes up must come down: tether a high-flyer "
                       "to an anchor and measure the cushion")
            try:
                bv_rets = _bv_anchor_returns(",".join(sorted(loaded)))
            except Exception:  # noqa: BLE001 - offline: universe only
                bv_rets = returns
            vols = returns.std() * np.sqrt(TRADING_DAYS)
            flyer = st.selectbox(
                "High-flyer (Circle 1 - the volatile conviction position)",
                loaded, index=loaded.index(vols.idxmax()),
                help="Defaults to the most volatile name in your universe.")
            try:                                   # grit feeds the anchor rank
                bv_grit = _bv_grit(tuple(sorted(bv_rets.columns)))
            except Exception:  # noqa: BLE001 - grit needs full-history fetch
                bv_grit = None
            # Bearish: Circle 1 becomes a synthetic daily-rebalanced SHORT of
            # the flyer (negated column); every downstream formula is reused
            # unchanged. The anchor screen flips to squeeze-cushion logic.
            bv_frame = bv_rets.copy()
            if bearish:
                bv_frame[flyer] = -bv_frame[flyer]
            bv_ranked = anchor_rank(bv_frame, flyer, grit=bv_grit,
                                    direction="short" if bearish else "long")
            bv_anchor = bv_ranked.index[0]
            tg = tail_gap(bv_frame, flyer, bv_anchor)
            ci_lo, ci_hi = cached_es_ci(bv_frame[flyer])
            pw = pair_weights(bv_frame[flyer], bv_frame[bv_anchor])
            bt = backtest_pair(bv_frame[flyer], bv_frame[bv_anchor], pw["w_a"])
            phase_now = regime_labels(
                (1 + bv_frame[flyer]).cumprod(), tg["gap"]).iloc[-1]
            _ph_col = {"Tether": "#33582F", "Descent": "#8A3B2E",
                       "Rotation": "#9A7B4F"}[phase_now]
            # Circle-and-line drawn to John's sketch: the high-flyer rides
            # top-RIGHT, the steady anchor sits bottom-LEFT, joined by the
            # diagonal safety line. Two-in-one motif, no new chart. Kept
            # contiguous - st.markdown truncates at blank lines.
            _bv_role_a = ("THE SHORT" if bearish else "THE HIGH-FLYER")
            _bv_role_b = ("SQUEEZE CUSHION" if bearish else "THE STEADY ANCHOR")
            # Inverse-vol weights give the anchor under half when the flyer is
            # the steadier leg (Index core SPY/GLD: 47%), so who holds the
            # money is read off the weights, not assumed (2026-10-17 council).
            _bv_money = (
                f'Why the anchor holds most of the capital: equal-risk split - '
                f'{flyer} runs {pw["vol_a"]:.0%} realized annual vol to {bv_anchor}\'s '
                f'{pw["vol_b"]:.0%}, so each dollar of {flyer} carries '
                f'{pw["vol_a"] / pw["vol_b"]:.1f}x the risk. Each leg carries the '
                f'same share of the RISK; the anchor holds most of the MONEY.'
                if pw["w_b"] >= 0.5 else
                f'Why {flyer} holds more of the capital: equal-risk split - '
                f'{flyer} runs {pw["vol_a"]:.0%} realized annual vol to {bv_anchor}\'s '
                f'{pw["vol_b"]:.0%}, so here {flyer} is the steadier leg and each '
                f'dollar of it carries {pw["vol_a"] / pw["vol_b"]:.1f}x the risk. '
                f'Each leg carries the same share of the RISK; {flyer} holds more '
                f'of the MONEY.')
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:44px;padding:26px 10px 12px;flex-wrap:wrap;">'
                f'<div style="flex:1 1 100%;max-width:980px;margin:0 auto;background:#F1EDE5;'
                f'border:1px solid #C4BDAE;border-top:2px solid #9A7B4F;border-radius:14px;'
                f'padding:26px 18px 14px;box-shadow:0 1px 2px rgba(63,59,53,.05),'
                f'0 12px 30px -22px rgba(63,59,53,.4);">'
                f'<svg viewBox="0 0 420 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;">'
                # Ledger corner ticks - the same double-rule frame grammar as
                # the section bands.
                f'<path d="M8 30 V8 H30 M390 8 H412 V30 M412 270 V292 H390 M30 292 H8 V270" fill="none" stroke="rgba(154,123,79,.5)" stroke-width="1.5"/>'
                # The safety line: bronze, dashed, circumference to
                # circumference (endpoints sit r+6 from each center - the
                # old line ran 38px INSIDE Circle 1).
                f'<line x1="129" y1="208" x2="270" y2="101" stroke="#9A7B4F" stroke-width="2" stroke-dasharray="7 5"/>'
                f'<g transform="rotate(-37 200 155)">'
                f'<rect x="154" y="142" width="92" height="26" rx="2" fill="#F1EDE5" stroke="rgba(154,123,79,.55)" stroke-width="1"/>'
                f'<text x="200" y="159" text-anchor="middle" font-family="Helvetica Neue" font-size="12" letter-spacing="1.5" fill="#8A6A3C">GAP {tg["gap"]:.1%}</text>'
                f'</g>'
                # Circle 1 - double ring, warm glossy fill (crest treatment).
                f'<circle cx="316" cy="66" r="52" fill="#F6F2EA" stroke="#8A6A3C" stroke-width="2"/>'
                f'<circle cx="316" cy="66" r="45" fill="rgba(154,123,79,.10)" stroke="rgba(154,123,79,.4)" stroke-width="1"/>'
                f'<text x="316" y="62" text-anchor="middle" font-family="Georgia" font-size="{"17" if len(flyer) < 5 else "14"}" fill="#3F3B35">{"-" + flyer if bearish else flyer}</text>'
                f'<text x="316" y="80" text-anchor="middle" font-family="Helvetica Neue" font-size="10" letter-spacing="1.4" fill="#8A6A3C">{pw["w_a"]:.0%} CAPITAL</text>'
                f'<text x="316" y="136" text-anchor="middle" font-family="Helvetica Neue" font-size="10.5" letter-spacing="1.8" fill="#6B6459">{_bv_role_a}</text>'
                # Circle 2 - the one we are: charcoal-inked, steady.
                f'<circle cx="92" cy="236" r="40" fill="#EFEAE0" stroke="#3F3B35" stroke-width="2"/>'
                f'<circle cx="92" cy="236" r="34" fill="rgba(63,59,53,.07)" stroke="rgba(63,59,53,.3)" stroke-width="1"/>'
                f'<text x="92" y="233" text-anchor="middle" font-family="Georgia" font-size="{"15" if len(bv_anchor) < 5 else "12"}" fill="#3F3B35">{bv_anchor}</text>'
                f'<text x="92" y="249" text-anchor="middle" font-family="Helvetica Neue" font-size="8" letter-spacing="1" fill="#6B6459">{pw["w_b"]:.0%} CAPITAL</text>'
                f'<text x="104" y="287" text-anchor="middle" font-family="Helvetica Neue" font-size="10.5" letter-spacing="1.8" fill="#6B6459">{_bv_role_b}</text>'
                # Phase chip under the line.
                f'<rect x="150" y="228" width="118" height="24" rx="12" fill="none" stroke="{_ph_col}" stroke-width="1.3"/>'
                f'<circle cx="166" cy="240" r="4" fill="{_ph_col}"/>'
                f'<text x="216" y="244" text-anchor="middle" font-family="Helvetica Neue" font-size="10.5" letter-spacing="1.8" fill="{_ph_col}">{phase_now.upper()}</text>'
                f'</svg>'
                f'<div style="font-family:\'Helvetica Neue\',sans-serif;font-size:10px;'
                f'letter-spacing:.22em;text-transform:uppercase;color:#6A5030;'
                f'text-align:center;margin-top:10px;">Defensive pair &middot; '
                f'{"the short and its squeeze cushion" if bearish else "what goes up must come down"}'
                f' &middot; sizes are illustrative; the numbers carry the quantities</div>'
                f'</div>'
                f'<div style="flex:1 1 100%;min-width:0;max-width:980px;margin:0 auto;">'
                f'<div style="font-family:Georgia;font-size:15px;color:#3F3B35;line-height:1.55;">'
                # A fall is its size beside the verb: "fell -36%" reads as a rise.
                f'Over the {bt["n_days"]} daily returns loaded, {"shorting" if bearish else "holding"} '
                f'<b>{flyer}</b> alone fell <b>{abs(bt["max_dd_solo"]):.0%}</b> at its worst'
                f'{" (a short bleeds when the name rallies - the squeeze)" if bearish else ""}; '
                f'tethered to {"a correlated long in " if bearish else ""}'
                f'<b>{bv_anchor}</b> at risk-parity weights '
                f'(<b>{pw["w_b"]:.0%}</b> anchor / <b>{pw["w_a"]:.0%}</b> flyer, '
                f'rebalanced monthly) the fall was <b>{abs(bt["max_dd_pair"]):.0%}</b> - '
                # a gap between two losses, in percentage points: "+17%"
                # read as a 17% cut (John, 2026-10-19), so say the unit and
                # the direction in words
                + (f'<b>{bt["cushion"] * 100:.0f} percentage points shallower</b>. '
                   'A cushion, not a cap.</div>' if bt["cushion"] >= 0 else
                   f'<b>{-bt["cushion"] * 100:.0f} percentage points deeper</b>: '
                   'no cushion in this history.</div>') +
                f'<div style="font-family:\'Helvetica Neue\',sans-serif;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#5D574D;margin-top:8px;">'
                f'Safety line (tail gap): ES 97.5% {tg["es_a"]:.1%} vs {tg["es_b"]:.1%} &middot; '
                f'{"short " if bearish else ""}{flyer} ES 90% CI {ci_lo:.1%}-{ci_hi:.1%} &middot; current phase '
                f'<span style="color:{_ph_col};font-weight:600;">{phase_now}</span></div>'
                f'<div style="font-family:Georgia;font-size:12.5px;color:#5D574D;margin-top:6px;">'
                f'{_bv_money}</div>'
                f'</div></div>',
                unsafe_allow_html=True)
            with st.expander("The real backtest - pair vs solo, and the crisis record"):
                bv_fig = go.Figure()
                bv_fig.add_scatter(x=bt["solo_path"].index,
                                   y=(bt["solo_path"] - 1) * 100,
                                   name=f"{'short ' if bearish else ''}{flyer} alone",
                                   line=dict(color="#8A3B2E", width=1.4))
                bv_fig.add_scatter(x=bt["pair_path"].index,
                                   y=(bt["pair_path"] - 1) * 100,
                                   name="defensive pair",
                                   line=dict(color=BRONZE, width=2))
                bv_fig = _style_fig(bv_fig, height=280)
                bv_fig.update_layout(yaxis_title="cumulative return (%)",
                                     legend=dict(orientation="h", y=1.08))
                st.plotly_chart(bv_fig, width="stretch", config=PLOTLY_CFG)
                st.caption(
                    f"Real daily returns, {bt['n_days']} of them, monthly "
                    f"rebalancing, no lookahead. Pair vol {bt['ann_vol_pair']:.1%} "
                    f"vs {bt['ann_vol_solo']:.1%} solo; total return "
                    f"{bt['total_return_pair']:+.0%} vs {bt['total_return_solo']:+.0%}. "
                    "In-sample description of one history, not a forecast.")
                try:
                    cc = _bv_crisis(flyer, bv_anchor, round(pw["w_a"], 3),
                                    bearish)
                    if len(cc):
                        panel_head("Crisis cushion record",
                                   "The same pair replayed through real crises "
                                   "(both assets must have traded)")
                        # the cushion is a gap in percentage points, not a %
                        # change: shown as points, positive = the pair fell less
                        st.dataframe(
                            cc.assign(cushion=cc["cushion"] * 100).rename(columns={
                                "solo_dd": f"{'short ' if bearish else ''}{flyer} alone",
                                "pair_dd": "pair", "cushion": "cushion (points)",
                                "days": "days"})
                              .style.format({f"{'short ' if bearish else ''}{flyer} alone": "{:.1%}",
                                             "pair": "{:.1%}",
                                             "cushion (points)": "{:+.1f}",
                                             "days": "{:.0f}"}),
                            width="stretch")
                        st.caption(
                            "Cushion = how many percentage points less the pair "
                            "fell than the flyer alone (negative: it fell more). "
                            "Crises where either asset had not yet listed are "
                            "omitted, not guessed. A small or negative cushion "
                            "is the honest signature of correlations converging "
                            "in that crisis.")
                except Exception:  # noqa: BLE001 - crisis replay needs network
                    st.caption("Crisis replay unavailable offline.")
                st.markdown(
                    '<div class="read-me"><b>What this is - and is not.</b> '
                    'Long-only defensive pairing (a core-satellite blend), not '
                    '"pairs trading" - there is no short leg and no cointegration '
                    'bet. The anchor is chosen by rank (lowest correlation to the '
                    "universe's dominant risk factor, low volatility, shallow "
                    'tail, highest Grit score - the engine\'s founding question, '
                    '"who gets back up?", IS the Circle 2 screen - from your '
                    'universe plus Treasury/gold/staples/utilities/min-vol ETFs) '
                    '- ranked, never promised. Weights are risk parity: each leg '
                    'contributes equal risk, which is why the steadier leg '
                    'holds more of the capital. Losses '
                    'are <b>cushioned, not capped</b>: a long-only pair has no '
                    'floor above zero, and crisis correlations converge toward '
                    '+1 exactly when protection matters most. The Tether / '
                    'Descent / Rotation phases are a descriptive regime study '
                    'on past prices - not a trading signal. Educational '
                    'analysis, not investment advice.</div>',
                    unsafe_allow_html=True)
                if bearish:
                    st.caption(
                        "Bearish read of grit: LOW grit describes past "
                        "fragility - it is not a short signal. Past losers "
                        "are often crowded shorts with high borrow cost and "
                        "the sharpest squeezes; the asset that keeps getting "
                        "back up (HIGH grit) is the short-seller's nightmare. "
                        "Synthetic daily-rebalanced short; borrow fees, "
                        "margin and buy-ins not modeled - real short results "
                        "are worse. Short losses can exceed 100%.")
        except Exception as exc:  # noqa: BLE001 - never crash the tab
            st.caption(f"Balance unavailable for this universe: {safe_err('balance-unavailable-for-this-universe', exc)}")

with tab_grit:
    st.caption(
        "Grit ranks your universe by how each name recovered from its own "
        "drawdowns: how fast, how completely, and how consistently across real "
        "crises. Every name here has drawdowns; this measures what happened next."
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
                hbar(gscores["grit_score"],
                     title_x="Grit Score (0–100)", top_first=True),
                width="stretch", config=PLOTLY_CFG,
            )
            read_me(
                "<b>Longer bar = grittier.</b> The name at the top has, across its "
                "own history, bounced back from drawdowns the fastest and most "
                "reliably. Score is <b>0–100 relative to this basket</b> - it ranks "
                "these names against each other, not against the whole market.")
            grittiest = gscores.index[0]
            g = gscores.loc[grittiest]
            # NaN here means "no drawdown episode deep enough to measure",
            # which is an unknown record - never render it as a perfect one.
            if pd.isna(g["pct_recovered"]):
                _rec = ("has not yet had a drawdown deep enough to time a "
                        "recovery from")
            else:
                _med = g["median_recovery_days"]
                _rec = (f"recovered {g['pct_recovered']:.0%} of its own drawdowns"
                        + ("" if pd.isna(_med) else
                           f" (median {_med:.0f} trading days to claw back)"))
            st.caption(
                f"**{grittiest}** ranks grittiest here: {_rec}, stayed positive "
                f"over {g['consistency']:.0%} of rolling 1-year holding "
                f"periods, and lived through {g['n_regimes_survived']:.0f} of "
                f"the named crisis windows above."
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
        st.caption(f"Grit Zone unavailable: {safe_err('grit-zone-unavailable', exc)}")

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
        h2.metric("Median 1y return from the trough", f"{t1['median']:+.0%}")
        h3.metric("Median crash depth", f"{ctab['depth'].median():.0%}")

        panel_head("What a buyer actually got", "Crisis by crisis, best day vs. worst day to buy")
        # By key, not position: a reordered table can never put one entry's
        # returns under the other's heading.
        show = ctab.rename(columns={
            "crisis": "Crisis", "trough_date": "Trough date", "depth": "Crash depth",
            **{f"{e}_{h}": f"{e.capitalize()} buy, {h}"
               for e in ("peak", "trough") for h in HORIZONS}})
        show = show[["Crisis", "Trough date", "Crash depth",
                     "Trough buy, 1y later", "Trough buy, 3y later",
                     "Peak buy, 1y later", "Peak buy, 3y later"]]
        pct_cols = [c for c in show.columns if c not in ("Crisis", "Trough date")]

        def _tone(v):
            if pd.isna(v):
                # 2.43:1 before - a "not available" cell still has to be readable
                return "color: #5D574D;"
            return "color: #33582F;" if v > 0 else "color: #8A3B2E;"

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
            # The exceptions are READ from the table, never written in: which
            # crises leave a peak buyer behind depends on the live series, and
            # a hand-typed "the one exception is ..." goes stale silently.
            _p3y = pd.to_numeric(show["Peak buy, 3y later"], errors="coerce")
            _behind = show.loc[_p3y < 0, "Crisis"].tolist()
            st.caption(
                "Nobody can time the exact trough - that row measures the "
                "direction of the edge, not an executable strategy. That's why "
                "the peak row sits beside it: the worst-timed buyer was ahead "
                f"three years later in {int((_p3y > 0).sum())} of "
                f"{int(_p3y.notna().sum())} crises with three years of data"
                + (f". The honest exception{'s' if len(_behind) > 1 else ''}: "
                   f"{', '.join(_behind)} - three years wasn't enough."
                   if _behind else ".")
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
        # Raced: the basket traded through it and at least one side fell.
        rrec = race_record(race)
        rr = rrec["decided"]
        cap = RECOVERY_HORIZON_DAYS

        def _lbl(days, fell):
            if not fell:
                return "never fell"
            return "not within 3y" if pd.isna(days) else f"{days:.0f}d"

        race_fig = go.Figure()
        for col, flag, nm, colr, who in (
            ("bench_days", "bench_fell", "S&P 500 (SPY)", "#CBBB94", "market"),
            ("basket_days", "basket_fell", "AI-capex basket", BRONZE_DK, "basket"),
        ):
            race_fig.add_trace(go.Bar(
                y=rr["crisis"],
                x=[race_days(d, f, cap) for d, f in zip(rr[col], rr[flag])],
                orientation="h", name=nm, marker=dict(color=colr),
                text=[_lbl(d, f) for d, f in zip(rr[col], rr[flag])],
                textposition="outside", textfont=dict(size=11),
                hovertemplate="%{y} - " + who + ": %{text}<extra></extra>"))
        race_fig = _style_fig(race_fig, height=max(340, 56 * len(rr) + 70))
        race_fig.update_layout(
            barmode="group", xaxis_title="trading days to reclaim pre-crisis level",
            showlegend=True,
            # headroom so "not within 3y" outside-labels never clip
            xaxis=dict(range=[0, cap * 1.18]),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)))
        st.plotly_chart(race_fig, width="stretch", config=PLOTLY_CFG)

        st.markdown(
            f'<div class="read-me"><b>How to read this.</b> Shorter bar = '
            f'faster recovery. The basket got back up faster in '
            f'<b>{rrec["wins"]} of {rrec["n"]}</b> crises. Where a bar says '
            f'"not within 3y", that side never reclaimed its pre-crisis '
            f'level inside ~3 trading years; "never fell" means it did not '
            f'decline at all inside that window, so it had nothing to '
            f'reclaim - shown, not hidden.'
            + (f' Left out: {", ".join(rrec["left_out"])} - none of the '
               f'basket\'s names was trading yet, so there was no race to '
               f'count.' if rrec["left_out"] else '')
            + '</div>',
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
        st.caption(f"Crisis Conviction unavailable: {safe_err('crisis-conviction-unavailable', exc)}")

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
        "Max daily participation (% of ADV)", 5, 50,
        round(DEFAULT_PARTICIPATION * 100), step=5,
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
        if mc["cvar"] <= 0:
            # Was a "+X% for the unwind" delta on a figure that was a gain.
            read_me(
                "<b>No tail loss to widen.</b> The worst 5% of simulated years "
                f"still end up ahead on average (headline CVaR {mc['cvar']:.1%}), "
                "so there is no tail loss for the unwind to widen and no "
                "liquidity adjustment is applied.")
        elif np.isfinite(lv["lvar"]):
            d1, d2 = st.columns(2)
            d1.metric("Headline CVaR (95%)", f"{lv['cvar']:.1%}")
            # The delta stays short: st.metric keeps it on one line and cuts it
            # with an ellipsis, and "+0% relative, for the unwind" was cut on a
            # 360px phone (device check, Live audit 2026-10-22). What it is
            # relative to lives in the help.
            d2.metric("Liquidity-adjusted CVaR", f"{lv['lvar']:.1%}",
                      delta=f"+{(lv['multiplier'] - 1):.0%} relative",
                      delta_color="inverse",
                      help="The change is relative to the headline CVaR: +2% means "
                           "1.02 times it, the extra tail for the unwind at this "
                           "book size and participation.")
            read_me(
                "<b>The tail you can't trade out of.</b> The headline CVaR "
                "assumes you're flat at the horizon. This widens it by "
                f"√(1 + exit-days/{TRADING_DAYS}) - the Basel liquidity-horizon convention - "
                "to cover the extra days the market can move against you while "
                "you're still unwinding. Prices market exposure over the unwind, "
                "not the spread you pay to trade. A one-day-liquid book is barely "
                "penalised; a name you'd be stuck holding carries a fatter tail.")

        chart_days = dtl["days"].replace([np.inf, -np.inf], np.nan).dropna().sort_values()
        if not chart_days.empty:
            st.plotly_chart(hbar(chart_days, title_x="days to liquidate"),
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
        if (mc["cvar"] > 0 and np.isfinite(lv["multiplier"])
                and lv["multiplier"] - 1 < 0.005):
            # Every mega-cap/ETF preset exits in well under a day, so the tail
            # adjustment rounds to +0%. Say why, or a reader assumes the model
            # is dead rather than the book being genuinely liquid.
            caption += (
                "At this size the participation cap never binds, so the tail is "
                "unchanged - a liquid book, not a dormant model. The *Small-cap "
                "liquidity stress* preset, or a larger book above, makes it bite. "
            )
        if prof["no_volume"]:
            caption += (
                f"*No volume feed for {', '.join(prof['no_volume'])} "
                "(e.g. FX/futures on Yahoo) - excluded, not estimated.*"
            )
        st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Liquidity data unavailable: {safe_err('liquidity-data-unavailable', exc)}")

with tab_dq:
    st.caption(
        "Every price pull runs through an automated validation gate before "
        "any risk number is computed from it - schema checks, positivity, "
        "coverage, staleness, and an extreme-move flag. This validates "
        "structure and plausibility, not truth: it catches a malformed or "
        "implausible feed, not a wrong-but-plausible number."
    )
    report = validate_prices(prices)
    # Palette marks, not colour emoji - the emoji were the only bright colours
    # on the page (ME-04). Each mark is a text tone that clears AA on beige;
    # the glyph carries the meaning, the weight the urgency. A row, not an
    # st.caption: Streamlit draws captions at 60% opacity, which would take
    # the marks (and the results they mark) below AA.
    # role/aria-label: a screen reader says "Pass", not "check mark" or
    # "multiplication sign" (Council B).
    mark = {"PASS": '<span class="dq-mark" role="img" aria-label="Pass" style="color:#6A5030;">&#10003;</span>',
            "WARN": '<span class="dq-mark" role="img" aria-label="Warning" style="color:#6A5030;font-weight:700;">!</span>',
            "FAIL": '<span class="dq-mark" role="img" aria-label="Fail" style="color:#3F3B35;font-weight:700;">&times;</span>'}
    for c in report["checks"]:
        st.markdown(f'<div class="dq-row">{mark[c["status"]]} <b>{html.escape(c["check"])}</b>'
                    f' - {html.escape(c["message"])}</div>', unsafe_allow_html=True)
    verdict_dq = "PASS" if report["passed"] else "FAIL"
    panel_head("Overall data-quality gate", f"This feed: {verdict_dq}")

with tab_lineage:
    prov = provenance(tickers)
    if prov:
        # Each name is pulled on its own, so a basket can span several pulls:
        # say so rather than print one time for all of them.
        _newest = prov.get("fetched_newest_utc", prov["fetched_at_utc"])
        _fetched = (prov["fetched_at_utc"] if _newest == prov["fetched_at_utc"] else
                    f"{prov['fetched_at_utc']} → {_newest} (each name is pulled "
                    f"on its own and re-pulled after {CACHE_MAX_AGE_HOURS:g} hours)")
        st.markdown(
            f"- **Source:** {prov['source']}\n"
            f"- **Fetched (UTC):** {_fetched}\n"
            f"- **Symbols:** {', '.join(prov['symbols'])}\n"
            f"- **Coverage:** {prov['start']} → {prov['end']} "
            f"({prov['rows']} trading days)\n"
            f"- **Library:** yfinance {prov['yfinance_version']}"
        )
        st.caption(
            "Prices are live end-of-day adjusted closes, pulled on demand from "
            "Yahoo Finance. Every figure above is computed from this source by "
            "the engine's own code - no value originates from a language model. "
            f"A copy older than {CACHE_MAX_AGE_HOURS:g} hours is re-pulled "
            "automatically on the next interaction."
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

    # ---- Your data: the honest version of a "delete my account" control ----
    panel_head("Your data", "What this app holds about you, and how to clear it")
    st.markdown(
        "**There is no account to delete, because there is no account.** "
        "Meliona has no sign-up, no login, no password, no cookie beyond the "
        "one Streamlit uses to keep this browser tab connected, and no "
        "analytics or advertising SDK of any kind.\n\n"
        f"- **Held in this session (memory only, gone when you close the tab):** "
        f"the tickers you chose, the widget settings above, and the reference "
        f"`{SESSION_REF}`, which is random and is not linked to you.\n"
        "- **Held on the server:** cached Yahoo Finance price files, named by a "
        "hash of the ticker set. They are public market data, shared by every "
        "visitor who picks the same basket, and contain nothing about you.\n"
        "- **Logs:** errors and a session-start line, tagged with the reference "
        "above so a support email can be matched to a stack trace. They rotate "
        "away on size and carry no personal data.\n"
        "- **Sent to third parties:** nothing about you. The server fetches "
        "prices from Yahoo Finance and, for ISIN lookups, Business Insider - "
        "your browser talks only to this app.\n\n"
        f"Questions, or want the logs for your reference purged early? Email "
        f"[{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL}) and quote `{SESSION_REF}` "
        f"- or [report this session]({FEEDBACK_LINK}) with the reference "
        f"already filled in."
    )
    if st.button("Clear this session's data", key="wipe_session"):
        # Deliberately session-scoped: it clears what THIS browser holds. It
        # does not clear the shared market-data cache - that is an operator
        # action, and visitors are viewers here (see .streamlit/config.toml).
        keep = st.session_state.session_ref
        st.session_state.clear()
        st.session_state.session_ref = keep
        st.success("Session settings cleared. Reloading with defaults.")
        st.rerun()

# ---- MCAP-style closing band: where-to-next rail + honest copyright bar ----
st.markdown("""
<div class="meliona-footer">
  <div class="f-rail">
    <a class="f-box" href="#grit-showcase"><span class="f-num">01</span>The Grit Zone
      <small>Resilience ranked from real drawdowns - recovery, consistency, crisis behavior.</small></a>
    <a class="f-box" href="#crisis-record"><span class="f-num">02</span>Crisis Conviction
      <small>The hardest trade, priced from the actual record of __NCRISES__ named crises.</small></a>
    <a class="f-box" href="#engine"><span class="f-num">03</span>The Engine
      <small>Stress-test any universe live - allocation, scenarios, CVaR verdict.</small></a>
  </div>
  <div class="f-bar">
    <div>Meliona &middot; Portfolio Risk Engine &middot; &copy; 2026 John Nguyen</div>
    <div>Live end-of-day data: Yahoo Finance &middot; Educational analysis, not investment advice</div>
  </div>
  <div class="f-bar">
    <div>Support: <a href="mailto:__SUPPORT__?subject=Meliona%20%E2%80%93%20session%20__REF__">__SUPPORT__</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/PRIVACY.md">Privacy</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/TERMS.md">Terms</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/ACCESSIBILITY.md">Accessibility</a></div>
    <div>No accounts &middot; no tracking &middot; session reference <code>__REF__</code> (quote it when you write in)</div>
  </div>
  <div class="f-bar">
    <div>Built for WCAG 2.2 AA. The Risk Topology map is drawn on canvas and carries a
      text description; charts are accompanied by their numbers in text. Known gaps are
      listed in the accessibility statement.</div>
    <div>Hit a barrier? Email <a href="mailto:__SUPPORT__?subject=Meliona%20accessibility%20%E2%80%93%20session%20__REF__">__SUPPORT__</a>
      with the reference above - barriers are treated as defects, not feature requests.</div>
  </div>
</div>
""".replace("__SUPPORT__", SUPPORT_EMAIL).replace("__REF__", SESSION_REF)
   .replace("__NCRISES__", str(len(HISTORICAL_REGIMES))),
    unsafe_allow_html=True)
