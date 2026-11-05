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

from pathlib import Path
import re
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
from src.risk import monte_carlo, jump_diffusion_mc, sharpe_ratio
from src.narrative import (DEFAULT_PARTICIPATION, build_book, headline, role_label,
                           role_note, Role)
from src.strategies import risk_contributions, risk_parity_weights, vol_target
from src.pairing import DEFENSIVE_ANCHOR_TICKERS
from src.covariance import estimate_covariance, TRADING_DAYS
from src.topology import build_map_payload, war_room_html
from src.observability import (count_preset, count_session, host_setting, log_incident,
                               log_session_start, new_session_ref, setup_logging)
from src.scenarios import HISTORICAL_REGIMES, replay_returns
from hero import (N_PATHS, CONTACT_MAX, CONTACT_NAME_MAX, CONTACT_TOPICS, RESPONSE_TIME,
                  RESPONSE_WINDOW, about_html, contact_mailto, crest_engraving, faq_html,
                  hero_html, profile_marks)
from src.liquidity import (days_to_liquidate, liquidity_profile,
                           liquidity_adjusted_cvar)
from src.grit import grit_scores, MIN_HISTORY_DAYS
from src.data_quality import validate_prices
from src.conviction import load_conviction, HORIZONS

# Title and icon match the <head> tools/public_files.py serves to crawlers and
# link previews, so a shared link and the browser tab say the same thing.
st.set_page_config(page_title="Meliona · Portfolio Risk Engine",
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
# John's support phone (2026-11-04), shown as a tap-to-call link beside the
# email. It is published on a public page, so it will be scraped like the
# email: MELIONA_SUPPORT_PHONE on the host replaces it, and "off" hides it,
# both without a redeploy.
SUPPORT_PHONE = host_setting("SUPPORT_PHONE", "770-265-9930").strip()
if SUPPORT_PHONE.lower() in {"off", "none", "0", ""}:
    SUPPORT_PHONE = ""
_digits = re.sub(r"\D", "", SUPPORT_PHONE)
SUPPORT_TEL = f"+1{_digits}" if len(_digits) == 10 else (f"+{_digits}" if _digits else "")

# What a visitor reads when Yahoo returns nothing for a preset basket. Those
# names are known good, so the feed, not the spelling, is the likely cause.
FEED_DOWN_MSG = ("Yahoo Finance returned no prices for this preset basket, so the "
                 "market-data feed is probably unreachable right now. Try again in "
                 "a minute.")

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
    st.caption(f"Questions: {SUPPORT_EMAIL}" + (f" or {SUPPORT_PHONE}" if SUPPORT_PHONE else ""))
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
    count_session()            # +1 to today's anonymous total (PRIVACY.md "Analytics")
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
    return f"{type(exc).__name__}; quote {SESSION_REF} if you report this"


def _md_plain(text: str) -> str:
    """A visitor's own words as plain text inside markdown. Every ASCII
    punctuation mark is backslash-escaped, so a typed "[x](y)" or "**x**"
    shows as typed instead of becoming a link or bold. The rejected tickers
    used to sit in `code spans` for the same reason (2026-11-04: the code
    font read as a stray terminal on the page)."""
    import re as _re
    return _re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", text)

# One prefilled mailto, reused by the beta banner and the privacy panel: the
# subject already carries the session reference, so a report arrives with the
# one thing that makes it diagnosable (see RUNBOOK.md section 3).
FEEDBACK_LINK = (f"mailto:{SUPPORT_EMAIL}?subject=Meliona%20feedback%20"
                 f"%E2%80%93%20session%20{SESSION_REF}")

if RELEASE_CHANNEL == "beta":
    st.info(
        f"Beta: a pre-release deployment. Features and numbers may change, and "
        f"the data is end-of-day, never real-time. Found something wrong? "
        f"[Email this session]({FEEDBACK_LINK}); its reference, {SESSION_REF}, "
        f"is already in the subject line.",
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
  /* This frame only carries the script: out of the Tab order and out of the
     accessibility tree, or a keyboard visitor lands on an invisible stop. */
  if (window.frameElement) {
    window.frameElement.setAttribute('tabindex', '-1');
    window.frameElement.setAttribute('aria-hidden', 'true');
  }
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
      /* The destination can still be moving: a block above it can grow or
         shift while we fly (it rose 24px on a scroll-driven reveal until
         2026-10-30), so a target read once at the click landed 8-11px past it
         and crawled back. Follow it - at most 2px
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
      /* Hold the landing while a forgiven flick is still arriving (onYield
         forgives its tail for 2 s from the press). The flight lasts 1.5 s; a
         tail that outlived it scrolled the page on past the section once the
         flight let go - "Explore what we do" ended 73px past it, 1 of 2
         presses on a loaded CI runner (Live audit 41015789077, step 8c). */
      if (now - forgivenAt < 250) { requestAnimationFrame(f); return; }
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
/* Page motion lives in static/reveal.js, which <head> loads before the first
   paint on the deployed page (tools/public_files.py). Where the head was not
   patched (a local run), load it from here: it leaves what is already on
   screen as drawn and reveals only what is below. */
(function() {
  const W = window.parent;
  if (W.__melionaReveal || W.document.getElementById('meliona-reveal')) return;
  const s = W.document.createElement('script');
  s.id = 'meliona-reveal'; s.src = 'app/static/reveal.js';
  W.document.head.appendChild(s);
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
  /* the rings' own waves only: the crest's first-load lift is an animation
     in the same subtree, and must never be made to loop */
  function waves(c) {
    return c.getAnimations({ subtree: true }).filter(function(a) {
      return a.animationName === 'crest-ripple';
    });
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
    (c.__rings || []).concat(waves(c)).forEach(function(a) {
      a.effect.updateTiming({ iterations: Infinity });
    });
  }
  function leave(e) {
    const c = crestOf(e); if (!c) return;
    const runs = c.__rings = waves(c);
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

# ---- Themed Plotly palette + chart helpers (the share banner's system) ----
# Charts sit on the flat paper with no fill of their own, hairline axes and
# three inks only: charcoal, bronze and the crest's gold (ME-28, 2026-11-05).
BRONZE = "#9A7B4F"
CHARCOAL = "#3F3B35"
STONE = "#2E2A25"          # the banner's charcoal block
GOLD = "#C6A26B"           # the crest's gold
GRID = "rgba(63,59,53,0.08)"
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
        legend_font=dict(color="#5D574D"),
        showlegend=False,
        bargap=0.12,
        # the readout is a small receipt: charcoal stone, paper words
        hoverlabel=dict(bgcolor=STONE, bordercolor=STONE,
                        font=dict(family="Georgia, serif", color="#EDE9E3", size=13)),
        transition=dict(duration=380, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS_LINE, linewidth=1,
                     ticks="outside", tickcolor=AXIS_LINE, tickfont=dict(color="#5D574D", size=12),
                     title_font=dict(color="#5D574D", size=13))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor=AXIS_LINE,
                     zerolinewidth=1, linecolor=AXIS_LINE, linewidth=1,
                     tickfont=dict(color="#5D574D", size=12), title_font=dict(color="#5D574D", size=13))
    return fig


def fan_chart(bands: dict):
    """
    Monte Carlo outcome cone: median path + 25-75 and 5-95 percentile bands.

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
    # Outer 5–95 cone (draw upper first, then lower with fill-to-previous).
    # Trace order is read by the audits (p95, p5, p75, p25 at 0..3).
    fig.add_trace(go.Scatter(x=d, y=p(bands["p95"]), name="p95", line=edge, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p5"]), name="p5", fill="tonexty",
                             fillcolor="rgba(63,59,53,0.05)", line=edge, hoverinfo="skip"))
    # Inner 25–75 cone
    fig.add_trace(go.Scatter(x=d, y=p(bands["p75"]), name="p75", line=edge, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=p(bands["p25"]), name="p25", fill="tonexty",
                             fillcolor="rgba(63,59,53,0.09)", line=edge, hoverinfo="skip"))
    # Hairline edges trace the envelope
    for key in ("p95", "p5"):
        fig.add_trace(go.Scatter(x=d, y=p(bands[key]), hoverinfo="skip",
                                 line=dict(color="rgba(63,59,53,0.45)", width=1,
                                           shape="spline", smoothing=1.0)))

    # The hover readout: all five levels on the hovered day, best at the top.
    # A unified hover lists traces last-first, so the readouts go in
    # worst-first, with the median in the middle.
    def readout(key, label):
        return go.Scatter(x=d, y=p(bands[key]), name=f"{key} readout", line=dict(width=0),
                          hovertemplate="%{y:.1f}% " + label + "<extra></extra>")

    fig.add_trace(readout("p5", "· worst 5 in 100 start here"))
    fig.add_trace(readout("p25", "· bottom of the middle half"))
    # Median path - the centreline, drawn after the fills so it sits on top
    fig.add_trace(go.Scatter(
        x=d, y=p(bands["p50"]), name="median",
        line=dict(color=CHARCOAL, width=2.0, shape="spline", smoothing=1.0),
        hovertemplate="<b>%{y:.1f}%</b> median, trading day %{x}<extra></extra>"))
    fig.add_trace(readout("p75", "· top of the middle half"))
    fig.add_trace(readout("p95", "· best 5 in 100 start here"))
    fig.add_hline(y=0, line=dict(color=AXIS_LINE, width=1, dash="dot"))
    fig.update_layout(
        xaxis_title="Trading days", yaxis_title="Total return so far (%)",
        hovermode="x unified",                 # one readout: all five levels on that day
        transition=dict(duration=420, easing="cubic-in-out"),
        meta={"meliona": "cone"},              # the audits find the cone by this
    )
    fig = _style_fig(fig, height=360)
    fig.update_layout(hoverlabel=dict(namelength=0))
    # Aerodynamic axes: no tick spikes, breathing gridlines only
    fig.update_xaxes(ticks="", showspikes=False)
    fig.update_yaxes(ticks="", ticksuffix="%")
    return fig


def panel_head(title: str, subtitle: str = "") -> None:
    """A section heading in the share banner's system (ME-28): the title as a
    caps kicker with its charcoal bar, the subtitle as the serif line under
    it, then the annual-report rule - a hairline with a charcoal key that
    draws itself across as the section comes up the screen."""
    sub = f'<span class="s">{subtitle}</span>' if subtitle else ""
    st.markdown(f'<div class="panel-head"><span class="t">{title}</span>{sub}'
                f'<span class="ar-rule"></span></div>', unsafe_allow_html=True)


def read_me(html: str) -> None:
    """Plain-language 'how to read this' block under a chart. Bold key words
    with <b>…</b>. Keeps the honest, defensible captions but makes the
    explanation impossible to miss."""
    st.markdown(f'<div class="read-me">{html}</div>', unsafe_allow_html=True)


def hbar(series: pd.Series, pct: bool = False, title_x: str = "",
         top_first: bool = False):
    """Themed horizontal bar chart. Bars deepen with magnitude - the biggest
    value wears the darkest bronze - so ranking reads at a glance.

    Plotly draws the FIRST category at the BOTTOM of a horizontal bar chart.
    Pass top_first=True for an already-ranked series so the leader lands where
    the reader - and the caption - expects it."""
    x = series.values * (100 if pct else 1)
    span = float(np.max(np.abs(x))) or 1.0
    def _shade(v):  # lerp the crest's gold #C6A26B (light) -> stone #2E2A25 (deep) by |value|
        f = abs(v) / span
        r = int(0xC6 + (0x2E - 0xC6) * f)
        g = int(0xA2 + (0x2A - 0xA2) * f)
        b = int(0x6B + (0x25 - 0x6B) * f)
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
        name="recovery", marker=dict(color=GOLD),
        hovertemplate="%{y} recovery: %{x:.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=list(scores.index), x=scores["consistency_score"].values, orientation="h",
        name="consistency", marker=dict(color=BRONZE),
        hovertemplate="%{y} consistency: %{x:.0f}<extra></extra>"))
    fig.add_trace(go.Bar(
        y=list(scores.index), x=scores["resilience_score"].values, orientation="h",
        name="resilience", marker=dict(color=STONE),
        hovertemplate="%{y} resilience: %{x:.0f}<extra></extra>"))
    fig = _style_fig(fig, height=max(220, 50 * len(scores)))
    fig.update_layout(
        barmode="group", xaxis_title_text="score (0–100, relative to this universe)",
        showlegend=True, legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return fig


# Anchored on this file, not the working directory: `streamlit run` from any
# other folder used to crash here on the very first line of the hero.
ASSETS = Path(__file__).parent / "assets"

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
  /* First-load only (keyframes in app.css, "ARRIVAL"): the crest and the
     four tiles lift from face-down to face-forward, the headline and its
     line slide in from the left, one after another. This style block
     renders solely on the un-booted first pass, so Streamlit reruns never
     replay the arrival. It starts .55s in, as the boot veil (which holds .3s
     and fades over .55s) is half gone - earlier, it played behind the veil.
     Scoped to #hero so the skeleton's copies hold still;
     fill "backwards", so once landed the hover states own the transform. */
  #hero .hero-crest { animation: lift-face .9s cubic-bezier(.2,.8,.2,1) .6s backwards; }
  #hero .hline { animation: from-left .95s cubic-bezier(.16,1,.3,1) backwards; }
  #hero .hline:nth-child(1) { animation-delay: .73s; }
  #hero .hline:nth-child(2) { animation-delay: .85s; }
  #hero .hero-sub { animation: from-left .95s cubic-bezier(.16,1,.3,1) 1.01s backwards; }
  #hero .cta-btn { animation: from-left .95s cubic-bezier(.16,1,.3,1) 1.15s backwards; }
  #hero .hero-ctas .cta-link { animation: from-left .95s cubic-bezier(.16,1,.3,1) 1.27s backwards; }
  #hero .hstat { animation: lift-face .85s cubic-bezier(.2,.8,.2,1) backwards; }
  #hero .hstat:nth-child(1) { animation-delay: .89s; }
  #hero .hstat:nth-child(2) { animation-delay: .99s; }
  #hero .hstat:nth-child(3) { animation-delay: 1.09s; }
  #hero .hstat:nth-child(4) { animation-delay: 1.19s; }
</style>""", unsafe_allow_html=True)
else:
    # Hold the veil's slot. Streamlit keys every element by its position, so
    # dropping one element on the second run shifted everything below it and
    # every drawer (st.expander) re-mounted closed - the first control a
    # visitor changed snapped shut under their cursor (recruiter walkthrough).
    st.markdown("", unsafe_allow_html=True)

# (2026-11-05, ME-28: the hero and band photographs went - John chose flat
# charcoal and beige blocks with the crest's faint engraving, exactly like the
# share banner. Their two <style> slots went with them: the hero's full-bleed
# margin in app.css counts the blocks above it, see
# test_the_skeleton_paints_first_and_the_hero_lifts_it.)

# The hero lifts its own skeleton when it arrives - state, not a timer. The
# veil used to fade on a fixed 2.1 s clock whether or not anything was ready,
# and under reduced motion (no animation runs) a timer never fires at all.
_LIFT_SKELETON = ("<style>#pre-boot,#boot-skel{opacity:0;visibility:hidden}"
                  "#boot-skel *{animation:none!important}</style>")
st.markdown(hero_html(crest=logo_svg) + _LIFT_SKELETON, unsafe_allow_html=True)

# ---- Showcase: the Grit Zone innovation, explained before you touch a slider ----
st.markdown("""
<div class="showcase-row">
  __ENGRAVING__
  <div class="showcase-section" id="grit-showcase">
    <div class="kicker">Scored on each name's full price history</div>
    <h2 class="showcase-title">The Grit Zone</h2>
    <div class="ar-rule"></div>
    <div class="showcase-body">
      Grit asks one question: after a stock falls, does it get back up, and
      does it do that in crisis after crisis?
    </div>
    <div class="pillar-row">
      <div class="pillar-card">
        <div class="pillar-label">Recovery</div>
        <div class="pillar-desc">How fast, and how fully, a stock wins back
          what it lost in each fall of 5% or more.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-label">Consistency</div>
        <div class="pillar-desc">How often a year of holding it ended in a
          gain, counted over every one-year stretch.</div>
      </div>
      <div class="pillar-card">
        <div class="pillar-label">Resilience</div>
        <div class="pillar-desc">How little it fell, and how fast it came
          back, in the real crises we replay.</div>
      </div>
    </div>
  </div>
  <div class="showcase-section" id="conviction">
    <div class="conv-core">
      <div class="kicker">S&amp;P 500 via SPY, computed live</div>
      <h2 class="showcase-title">The hardest trade is the one history rewards</h2>
      <div class="ar-rule"></div>
      <div class="showcase-body">
        Buying in a crash feels wrong. Here is what a buyer of the S&amp;P 500
        earned in each crisis we replay: once at the bottom, the lowest close
        of the crisis, and once at the worst moment, the peak before the
        fall.
      </div>
      <a href="#crisis-record" class="cta-btn on-dark" style="margin-top:22px;">
        See the crisis record<span class="cta-arrow" aria-hidden="true"><svg viewBox="0 0 16 30" fill="none" xmlns="http://www.w3.org/2000/svg"><path class="shaft" d="M8 1 V22" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path class="head" d="M2.5 16.5 L8 22.5 L13.5 16.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></span></a>
    </div>
  </div>
</div>
""".replace("__ENGRAVING__", crest_engraving()), unsafe_allow_html=True)


# ---- Showcase: Crisis Conviction - the emotional problem, answered in numbers ----
@st.cache_data(ttl=6 * 3600, show_spinner="Reading the crisis record…")
def load_conviction_data():
    """Benchmark crisis record, live from Yahoo."""
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

    _sl1, _sl2, _sl3 = st.columns(3)
    with _sl1:
        st.markdown(f"""<div class="slab"><div class="slab-label">Bought at the bottom</div>
        <div class="slab-num">{round(_t1["pct_positive"] * _t1["n"])} of {_t1["n"]}</div>
        <div class="slab-note">crises showed a gain one year later. Typical
        (median) return: {_t1["median"]:+.0%}.</div></div>""", unsafe_allow_html=True)
    with _sl2:
        st.markdown(f"""<div class="slab"><div class="slab-label">Three years on</div>
        <div class="slab-num">{_t3["median"]:+.0%}</div>
        <div class="slab-note">typical (median) return three years after the
        bottom. {round(_t3["pct_positive"] * _t3["n"])} of {_t3["n"]} showed a gain.</div></div>""",
                    unsafe_allow_html=True)
    with _sl3:
        st.markdown(f"""<div class="slab"><div class="slab-label">Worst possible timing</div>
        <div class="slab-num">{round(_p3["pct_positive"] * _p3["n"])} of {_p3["n"]}</div>
        <div class="slab-note">crises: a buyer at the peak before the fall
        was ahead three years later. Typical (median) return: {_p3["median"]:+.0%}.</div></div>""",
                    unsafe_allow_html=True)
    st.caption(
        "From Yahoo Finance adjusted closing prices of SPY, a fund that tracks the "
        "S&P 500. A record of the past, not a forecast. The Crisis Conviction tab "
        "below has every crisis, row by row."
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
# Flat paper (ME-28, 2026-11-05): the topographic plate and the faded mark
# behind the zone went with the photographs.
st.markdown('<div id="the-slice"></div>', unsafe_allow_html=True)
_gungnir_zone = st.container(key="gungnir_zone")
with _gungnir_zone:
    st.markdown(
        '<div class="gungnir-head">'
        '<div class="kicker">Equal risk, two names</div>'
        '<h2 class="showcase-title">From evidence to allocation</h2>'
        '<div class="ar-rule"></div>'
        '<div class="gungnir-sub">Above is the evidence. Below is the tool: '
        'choose a side, then choose your stocks.</div>'
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
            ["Bullish: long the book", "Bearish: short the book"],
            horizontal=True, key="bv_direction",
            help="Bearish runs every number as if you had sold the same stocks "
                 "short, reset each day. Borrowing fees, margin interest and "
                 "forced buy-backs are not included, so a real short would do "
                 "worse than shown.")
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
        f'<div class="showcase-section" style="padding-top:2px;text-align:center;">'
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
        f'<text x="214" y="105" text-anchor="middle" font-family="Georgia" font-size="17" fill="#6A5030">=</text>'
        f'<text x="42" y="17" font-family="Georgia" font-size="10" fill="#6A5030">volatility</text>'
        f'<text x="330" y="168" text-anchor="end" font-family="Georgia" font-size="10" fill="#6A5030">capital</text>'
        f'<text x="120" y="184" text-anchor="middle" font-family="Georgia" font-size="11.5" fill="#5D574D">{_bv_sk_b}</text>'
        f'<text x="268" y="184" text-anchor="middle" font-family="Georgia" font-size="11.5" fill="#5D574D">{_bv_sk_a}</text>'
        f'</svg>'
        f'<div style="font-family:Georgia;font-size:13px;color:#5D574D;max-width:560px;margin:2px auto 0;">'
        f'{"Shorting flips the sketch: the tall block is the stock you bet against, the wide one a stock you own that tends to rise with it, softening a sudden jump. The areas match, so each carries the same share of the risk. A short can lose more than 100%, and borrowing costs are not included." if bearish else "The flyer is narrow and tall: little money, big price swings. The cushion is wide and short: more money, small swings. The areas match, so each name carries the same share of the risk."}'
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
        f'<div class="gungnir-plaque">'
        f'<div class="gplaque-eq">'
        f'<span class="gplaque-term"><span class="t">w<sub>1</sub>&sigma;<sub>1</sub></span>'
        f'<span class="c">flyer</span></span>'
        f'<span class="gplaque-eqs">=</span>'
        f'<span class="gplaque-term"><span class="t">w<sub>2</sub>&sigma;<sub>2</sub></span>'
        f'<span class="c">cushion</span></span>'
        f'</div>'
        f'<div class="gplaque-read">Money in a name (w) times how much its price '
        f'swings (&sigma;, volatility) is the same for both: a stock that swings '
        f'three times as much gets a third of the money.</div>'
        f'</div>',
        unsafe_allow_html=True)

st.markdown(
    f'<div class="engine-heading" id="engine">'
    f'<div class="kicker">Bad-year loss &middot; real crises &middot; simulated years</div>'
    f'<h2 class="showcase-title">Test any '
    f'{"short book" if bearish else "portfolio"} against a bad year</h2><div class="ar-rule"></div>'
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
    '<div class="orient">'
    '<div class="orient-t" role="heading" aria-level="3">How to use this page</div>'
    '<ol class="orient-l">'
    '<li>Pick what you own: choose a starter basket below, or type any ticker '
    'Yahoo Finance carries.</li>'
    '<li>Replay a real crisis or move the sliders, and watch the big number '
    'move.</li>'
    '<li>Read the big number. It answers one question: in a bad year, the '
    'worst 5 of every 100 simulated years, how much does this portfolio lose '
    'on average? Everything below it shows the working.</li>'
    '</ol>'
    '<div class="orient-f">Nothing here is advice. Every number is computed '
    'from live end-of-day prices, and every tab shows its method.</div>'
    '</div>',
    unsafe_allow_html=True)

# ---- The cockpit: controls fold into three drawers so the verdict
# leads the section. Widgets still execute when collapsed - zero logic change,
# the reader just isn't bombarded with every dial at once. ----
with st.expander("Portfolio: which names to test", expanded=False):
    # The preset rides in the address (?preset=...), so a reload or a shared
    # link opens the same basket (UX check, 2026-11-04: every choice was lost
    # on reload). Read once per session, and only a name in PRESETS is taken:
    # the address is visitor input, and a closed set is the whole check.
    _preset_names = list(PRESETS.keys())
    if "preset" not in st.session_state:
        _asked = st.query_params.get("preset")
        if _asked in PRESETS:
            st.session_state["preset"] = _asked
    preset = st.selectbox("Preset basket", _preset_names, key="preset",
                          label_visibility="collapsed")
    if preset == _preset_names[0]:
        if "preset" in st.query_params:
            del st.query_params["preset"]
    elif st.query_params.get("preset") != preset:
        st.query_params["preset"] = preset
    # Usage: each preset basket a session opens counts once for that session,
    # not once per rerun. A name from PRESETS only - never the typed tickers.
    _opened = st.session_state.setdefault("_usage_presets", set())
    if preset not in _opened:
        _opened.add(preset)
        count_preset(preset, PRESETS)

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
        help="Pick a starter basket above, or add any Yahoo Finance symbol "
             "(for example BRK-B, EURUSD=X for a currency, GC=F for gold).",
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
        + ", ".join(_md_plain(t[:24]) for t in _rejected)
        + ": not a valid Yahoo Finance symbol. A symbol uses letters, digits "
          "and the characters . - = ^ only, for example BRK-B, EURUSD=X, "
          "GC=F or ^IRX."
    )
if len(tickers) > MAX_UNIVERSE:
    # Complexity budget, enforced where the visitor can SEE it. Every extra
    # symbol is another Yahoo leg, another covariance row, another full-history
    # Grit pull and another map unit - on a single-process public deploy that
    # cost lands on every other visitor. src/ingestion.py caps the funnel too,
    # so a caller that skips this check still cannot exceed the budget; this
    # branch exists so the cap is disclosed rather than silently applied.
    _notices.warning(
        f"This engine analyzes up to {MAX_UNIVERSE} symbols at once, a "
        "deliberate compute budget on a public, single-process deployment. "
        f"Using the first {MAX_UNIVERSE} alphabetically; dropped "
        + ", ".join(_md_plain(t) for t in tickers[MAX_UNIVERSE:])
        + "."
    )
    tickers = tickers[:MAX_UNIVERSE]
if len(tickers) < 2:
    st.warning("Add at least two symbols to test a portfolio, or pick a "
               "starter basket in the Portfolio drawer above.")
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
               f"than {CACHE_MAX_AGE_HOURS:g}h, on your next interaction, not on a timer.",
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
    # Every name came from a preset, so nothing was typed and "check the
    # spelling" points the wrong way: the feed is the likely cause.
    st.error(FEED_DOWN_MSG if set(tickers) <= set(PRESETS[preset]) else str(exc))
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
                + ", ".join(f"{_md_plain(t)} ({int(_coverage[t])} trading days)"
                            for t in _short)
                + f": each has less than the {MIN_ROWS} days of history this "
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
        f"This basket only has {len(prices)} trading days in common, too few "
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
        "Excluded " + ", ".join(_md_plain(t) for t in _flat)
        + ": its price did not move once in this window (cash and money-market "
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
             + (". Yahoo Finance may be unreachable right now; try again in a minute."
                if set(tickers) <= set(PRESETS[preset]) else
                ". Check the spelling or try different tickers."))
    st.stop()
if missing:
    _notices.caption(f"Couldn't load {', '.join(missing)}; skipped.")
_audit("Data fetch", f"{len(loaded)} tickers loaded from the {preset} preset: {', '.join(loaded)}"
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
with deck_alloc, st.expander("Weights: how the money is split",
                             expanded=False):
    COV_LABELS = {
        "Ledoit-Wolf": "Ledoit-Wolf: steady (default)",
        "Sample": "Sample: plain history",
        "EWMA": "EWMA: reactive / panic lens",
    }
    cov_method = st.selectbox(
        "Covariance estimator", ["Ledoit-Wolf", "Sample", "EWMA"],
        format_func=lambda m: COV_LABELS[m],
        help="How the engine measures the way these names move together (their "
             "covariance), which sets the weights. Ledoit-Wolf (default) pulls "
             "noisy history toward a steadier estimate. Sample uses history as it "
             "is. EWMA (RiskMetrics, λ=0.94) leans hard on the last few weeks, so "
             "one bad day moves it a lot: use it to see the panic view, not as a "
             "default.")
    cov, cov_info = estimate_covariance(returns, cov_method)  # annualized risk matrix
    st.caption(f"How the names move together: {cov_info}.")
    if cov_method == "EWMA":
        st.markdown(
            # the palette's warning mark in a Data Quality row, not a yellow
            # emoji in a 60%-opacity caption (2.63:1, Council B)
            '<div class="dq-row"><span class="dq-mark" role="img" aria-label="Warning" '
            'style="color:#6A5030;font-weight:700;">!</span> '
            "Panic view: EWMA jumps on one bad day. Half its weight sits in the "
            "last ≈11 trading days, about 73% in the last month, and 90% in the "
            "last ~37 (λ=0.94). It is the panic the Crisis Conviction tab argues "
            "against, shown so you can see how jumpy risk looks; the engine does "
            "not suggest acting on it.</div>",
            unsafe_allow_html=True,
        )
    acol1, acol2 = st.columns(2)
    method = acol1.radio(
        "Weighting", ["Equal weight", "Risk parity"], label_visibility="collapsed",
        help="Equal weight puts the same money in each name. Risk parity gives "
             "each name the same share of the risk, so calmer names get more money "
             "and no single name dominates (the idea behind Bridgewater's "
             "All Weather fund).")
    use_vt = acol2.checkbox(
        "Target volatility", help="Scales the whole portfolio up or down so its "
        "yearly price swings (volatility) match the target. Below 1× invests less "
        "than the full amount; above 1× borrows to invest more.")
    target_vol = acol2.slider("Target yearly volatility (%)", 5, 30, 10, step=1,
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

with deck_stress, st.expander("Stress test: shock or replay a crisis",
                              expanded=False):
    engine = st.radio(
        "Return model", ["Bootstrap (empirical)", "Jump-diffusion (Merton)"],
        horizontal=True,
        help="Bootstrap builds each simulated year from stretches of real past "
             "days, so it can only repeat falls it has already seen. Jump-diffusion "
             "(Merton 1976) adds rare, sudden jumps to ordinary daily moves, so a "
             "simulated day can fall further than any real one. Switch between "
             "the two to compare their worst years on this book.")
    mode = st.selectbox(
        "Scenario", ["Custom shock (sliders)"] + list(HISTORICAL_REGIMES.keys()),
        help="Custom: set your own market fall and extra turbulence. Or replay "
             "the actual daily returns of a real crisis, with the way these names "
             "really moved together in it.")
    if mode == "Custom shock (sliders)":
        col1, col2 = st.columns(2)
        drawdown_shock = col1.slider(
            "Market drawdown shock", -50, 0, 0, step=5,
            help="Lowers every past daily return so the year as a whole falls "
                 "by this much, before the simulation runs.")
        vol_shock = col2.slider(
            "Volatility shock", 0, 300, 0, step=10,
            help="Widens the daily ups and downs by this much, to mimic a "
                 "more turbulent market.")
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
        f"At these settings one day of this book loses {-_worst_day:.0%}, more than "
        "everything it holds, so there is no year to simulate past it. Lower the "
        "volatility shock or the target volatility.")
    st.stop()
mc = run_monte_carlo(shocked_returns, np.asarray(sim_weights, dtype=float), use_jd)
_audit("Stress scenario", scenario_label or
      (f"Custom shock (drawdown {drawdown_shock:+d}%, vol {vol_shock:+d}%)"
       if is_shocked else "None (base case)"))
_audit("Monte Carlo", f"{engine}, {mc['n_simulations']:,} paths of {mc['horizon_days']} days, CVaR {mc['cvar']:.2%}")

# ---- Liquidity-adjusted tail ----
# The CVaR above assumes you're out at the horizon. Widen it for the days it
# actually takes to unwind at DEFAULT_PARTICIPATION of real daily volume, for
# a HEADLINE_BOOK book - named in the sentence, since the days turn on it. The
# Book below takes its own size and participation; this is the headline default.
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
_tail = (f"loses <b>{mc['cvar']:.1%}</b> on average" if mc["cvar"] > 0 else
         f"still ends up <b>{-mc['cvar']:+.1%}</b> on average (this window gives "
         f"the book no bad-year loss to measure)")
if scenario_label:
    # Jump-diffusion replays nothing: it fits a model to the window. Either
    # way the N-day window becomes a year, and the sentence says so.
    verdict = (
        f"With a jump model calibrated to the actual returns of "
        f"{scenario_label} ({len(shocked_returns)} trading days) and run for a "
        f"year, " if use_jd else
        f"Replaying the actual returns of {scenario_label} "
        f"({len(shocked_returns)} trading days), resampled into a year, "
    ) + (f"{alloc_art} {alloc_label} portfolio{lev_txt} "
         f"{_tail} in a bad year (the worst 5 of every 100 simulated years).")
    if excluded:
        # Ticker names are user-supplied and this string is rendered with
        # unsafe_allow_html. VALID_TICKER already forbids "<", so this escape
        # is the second line, not the first - defence that survives someone
        # widening the regex later.
        verdict += (" (Excludes "
                    + ", ".join(html.escape(t) for t in excluded)
                    + ": it did not trade through the whole window, or its price fell to zero or below.)")
else:
    verdict = (
        f"In a bad year (the worst 5 of every 100 simulated years), {alloc_art} "
        f"{alloc_label} portfolio of these {len(loaded)} names{lev_txt} {_tail}."
    )
    if is_shocked:
        verdict += " (under the stress scenario set above)"
if bearish:
    verdict += (
        " <span style='font-size:13px;color:#6A5030;'>A simulated short, reset "
        "each day. Borrowing fees, margin interest and forced buy-backs are not "
        "included, so a real short would do worse. A short can lose more than "
        "100% of the money put in.</span>"
    )

# Only surface the liquidity add-on when it materially fattens the tail
# (multiplier > 1.005 ≈ more than ~2.5 trading days to fully exit). A CVaR
# <= 0 has no tail loss to widen; the line read "widens to" a gain.
if (mc["cvar"] > 0 and lvar and np.isfinite(lvar["lvar"])
        and lvar["multiplier"] > 1.005):
    verdict += (
        f" Selling all of a ${HEADLINE_BOOK:,.0f} book takes about "
        f"<b>{lvar['full_exit_days']:.0f} trading days</b> when you trade at most "
        f"{DEFAULT_PARTICIPATION:.0%} of each name's average daily dollar volume; "
        f"counting those days, the bad-year loss (the tail) widens to "
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
        ("Stressed", "A shock or a crisis replay is set in the Stress test drawer", is_shocked),
        ("Short book", "The bearish side: a simulated short, reset each day", bearish))
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
            'Time runs left to right over one year of trading days. The dark line '
            'is the middle outcome (the median): half the simulated years end above '
            'it, half below. The darker band holds the middle half of all years, '
            'the pale band 9 in 10. The cone widens because small daily moves add '
            'up. Its bottom edge is where the worst 5 in 100 years begin; the big '
            'number is the average loss below that edge, so it sits lower than the '
            'cone shows.'
            '</div>', unsafe_allow_html=True)
with f_col:
    st.plotly_chart(fan_chart(mc["path_bands"]), width="stretch", config=PLOTLY_CFG)
    # Jump-diffusion resamples nothing, and a replay under 30 days falls back to
    # i.i.d. (block_days=1): both used to claim "crash weeks stay intact".
    _se_txt = (f" Chance alone could move the bad-year loss by about "
               f"±{mc['cvar_se']:.2%} (its sampling error; {mc['n_simulations']:,} "
               f"simulated years, "
               + ("drawn from the jump model, not from real days"
                  if mc.get("engine") == "jump-diffusion"
                  else f"resampled from runs of real days averaging {mc['block_days']} days"
                  if mc.get("block_days", 1) > 1
                  else "resampled one real day at a time (window too short for runs)")
               + "). It is an estimate, not an exact value."
               if np.isfinite(mc.get("cvar_se", float("nan"))) else "")
    st.caption(
        "Each simulated year chains together a year of daily returns. A range of "
        "what could happen, not a forecast. The curves are smoothed between "
        "computed points." + _se_txt
    )

# ---- The Book: the guided arc, deliberately AHEAD of the tab strip ----
# The tabs answer sixteen questions and leave the reader to assemble them.
# This answers them in order, and each answer narrows the next: book size sets
# what each name can absorb, that sets who is eligible, that sets who can lead,
# and only then is anything paired. See src/narrative.py.
st.markdown('<div id="the-book"></div>', unsafe_allow_html=True)
panel_head("The Book",
           "The names that lead, and the steady names beside them")
_bk1, _bk2 = st.columns(2)
_book_size = _bk1.number_input(
    "Book size ($)", min_value=10_000, max_value=5_000_000_000,
    value=1_000_000, step=100_000, key="book_size",
    help="How much money the portfolio (the book) holds. This whole section "
         "turns on it: raise it and names drop out, because a big position "
         "cannot be sold quickly.")
_book_part = _bk2.slider(
    "Most of a day's volume you trade (%)", 5, 50,
    round(DEFAULT_PARTICIPATION * 100), step=5, key="book_part",
    help="The most of a stock's daily trading you would make up when selling. "
         "Trade more than that and your own selling starts to move the price. "
         "Risk desks often use 10 to 20%.") / 100
# Overwhelm audit (2026-10-29): the book leads with its paragraph and the
# names that carry it - the leaders (grit) and their anchors (the pairing).
# The names with no role, a run of near-identical sentences, fold into one
# drawer with the two finer settings - same widgets, same numbers.
_bk_box = st.container()
_bk_head = _bk_box.empty()
_bk_lead = _bk_box.container()
_bk_more = _bk_box.expander("The other names, line by line, and two finer settings")
_bk3, _bk4 = _bk_more.columns(2)
_bk_floor = _bk3.toggle(
    "Flyers must be in the grittiest half", value=False, key="book_floor",
    help="Off: a leader only needs a recovery record, and its recent climb "
         "sets the order. On: a leader must also be in the grittiest half. "
         "Both readings of 'grittiest high-flyer' are fair; this shows the "
         "difference and leaves the choice to you.")
_bk_fb = _bk4.toggle(
    "Allow a steady anchor from outside your list", value=True,
    key="book_fallback",
    help="A basket of big tech names may hold nothing calm enough to soften "
         "its worst days. When nothing in your list does, this lets the engine "
         "add one of TLT, GLD, XLP, XLU or USMV (Treasury bonds, gold, staples, "
         "utilities, low-swing stocks) and say on screen that it did.")

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

        _bk_head.markdown(f'<div class="book-lede">{headline(_the_book)}</div>',
                          unsafe_allow_html=True)

        # Ruled rows (ME-28): each name is a row under a hairline whose key -
        # the 3px mark over the role column - carries the role's tone
        # (decorative, so the light bronze may draw it); the words take the
        # AA text tones in app.css (.book-role, .book-text). The chip's words
        # come from narrative.role_label: a name with no role can still hold
        # capital, so "Left out" is only for a name holding none.
        _ROW = {Role.FLYER: "is-flyer", Role.CUSHION: "is-cushion", Role.EXCLUDED: "is-excluded"}
        for _d in _the_book:
            _lbl = role_label(_d)
            # A "No role" card shows what it holds, so "still held" can be
            # checked on the card; on its own line, so the role column wraps
            # rather than cuts at 360px.
            _note = role_note(_d)
            if _note:
                _lbl += f'<br><span class="role-note">{_note}</span>'
            _flag = (' &middot; <span class="book-flag">from outside your '
                     'list</span>' if _d.is_fallback_anchor else "")
            # Redundant on a card that was already excluded FOR capacity - the
            # sentence there already names the breakpoint.
            _bind = (' &middot; <span class="book-flag">trading volume, not '
                     'risk, set this size</span>'
                     if _d.liquidity_binding and _d.role is not Role.EXCLUDED else "")
            # leaders and anchors stay on the page; the rest is in the drawer
            (_bk_lead if _d.role in (Role.FLYER, Role.CUSHION) else st).markdown(
                f'<div class="book-row {_ROW[_d.role]}"><div class="book-role">{_lbl}</div>'
                f'<div class="book-text">{_d.sentence()}{_flag}{_bind}</div></div>',
                unsafe_allow_html=True)

        read_me(
            "Read it top to bottom. Each sentence is a fixed template filled in "
            "with the engine's own numbers; the text never sees the prices, so it "
            "cannot say anything the engine did not compute. Names marked No role "
            "are still held, at the size equal-risk weighting and trading volume "
            "allow, but neither lead nor anchor; names marked Left out hold nothing. Raise "
            "the book size and names drop out once they can no longer be sold "
            "quickly enough.")
    except Exception as _bk_exc:  # noqa: BLE001 - the page must never die on data
        # safe_err, not str(exc): the message can carry a server path (see safe_err).
        _bk_head.info(
            f"The Book needs price and volume history for this universe "
            f"({safe_err('the-book', _bk_exc)}). No placeholder book is shown.")

st.markdown('<div id="analysis"></div>', unsafe_allow_html=True)
panel_head("Risk & conviction", "The working: where the risk sits and how names held up")
tab_3d, tab_breakdown, tab_grit, tab_conviction = st.tabs([
    "Risk Topology", "Risk Breakdown", "Grit Zone", "Crisis Conviction",
])
# Story cut (John, 2026-11-05): the page tells one story - a bad year's loss,
# what real crashes did, which names hold up, how anchors carry flyers - so
# Liquidity, Correlation Watch, Balance, the AI-capex race and the model-
# validation drawer left the strip. src/ keeps their libraries.
# Overwhelm audit (2026-10-29): the research workshop (Signal Lab, Regime
# Atlas, Security Master) asked no question a visitor has to decide on, and it
# stood open by default. What remains here is the provenance a reader can
# audit - folded, one click away, never in the way.
with st.expander("Where the numbers come from: data checks and audit trail"):
    tab_dq, tab_lineage = st.tabs(["Data Quality", "Lineage & Audit"])

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
            "The simulation drawn as a map: across, how much the book moves with "
            "the market (beta); up, how much it swings (volatility). The lit ground "
            "holds 68 in 100 of the places the book could be in 30 days, and night "
            "falls at 99.7 in 100; the outer line is a stated danger limit, not an "
            "estimate. The paths are fitted to this book's own history. Estimates, "
            "not forecasts. The Explain button on the map defines every term."
        )
    except Exception as _wr_exc:  # noqa: BLE001 - never fake the map
        st.info(
            "The Risk Topology map needs live market history "
            f"({safe_err('risk-topology', _wr_exc)}). No stand-in numbers are "
            "shown; reload when the data is back."
        )

with tab_breakdown:
    st.caption(f"Universe ({len(loaded)}): {', '.join(loaded)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Median 1-year return", f"{mc['median_return']:+.1%}")
    c2.metric("Chance of a losing year", f"{mc['prob_loss']:.1%}")
    c3.metric("Worst simulated year", f"{mc['worst_case']:+.1%}")

    # --- Risk-contribution decomposition (where the risk actually lives) ---
    panel_head("Risk by name", "How much of the book's swings each name causes")
    # base_weights, NOT the vol-targeted ones: risk_pct is scale-invariant and
    # always sums to 100%, so plotting LEVERED dollar weights beside it put two
    # different denominators under one "% of portfolio" axis - at 0.6x leverage
    # every weight bar shrank and read as a risk/weight gap that was pure
    # leverage. Leverage scales the whole book and cannot change the split.
    rc = risk_contributions(base_weights, cov)
    rc_fig = go.Figure()
    rc_fig.add_trace(go.Bar(
        y=list(rc.index), x=rc["weight"].values * 100, orientation="h",
        name="dollar weight", marker=dict(color=GOLD),
        hovertemplate="%{y} weight: %{x:.1f}%<extra></extra>"))
    rc_fig.add_trace(go.Bar(
        y=list(rc.index), x=rc["risk_pct"].values * 100, orientation="h",
        name="risk share", marker=dict(color=STONE),
        hovertemplate="%{y} risk: %{x:.1f}%<extra></extra>"))
    rc_fig = _style_fig(rc_fig, height=max(200, 46 * len(rc)))
    rc_fig.update_layout(
        barmode="group", xaxis_title_text="% of portfolio", showlegend=True,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)))
    st.plotly_chart(rc_fig, width="stretch", config=PLOTLY_CFG)
    top = rc["risk_pct"].idxmax()
    st.caption(
        f"Gold: the share of the money in each name. Dark: its share of the "
        f"portfolio's swings (volatility). {top} causes the most "
        f"({rc.loc[top, 'risk_pct']:.0%}). Equal money is not equal risk; switch "
        "Weights to Risk parity to even out the dark bars."
        + (f" Both bars are shares of the book itself; the {leverage:.2f}× "
           "volatility target scales the whole book and does not change this "
           "split." if use_vt else "")
    )

    panel_head("Return for the risk taken", "The Sharpe ratio, against the real T-bill rate")
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
    s2.metric("Return per year", f"{ann_ret:+.1%}")
    s3.metric("Volatility per year", f"{ann_vol:.1%}")
    rf_txt = (f"{rf:.2%} (13-week T-bill, ^IRX)" if rf is not None
              else "unavailable, so Sharpe is computed against 0%")
    st.caption(
        f"Sharpe = (return per year − the risk-free rate) / volatility per year: "
        f"how much return each unit of swing earned, on the real, unstressed "
        f"{'simulated short book (reset daily; borrowing costs not included)' if bearish else 'portfolio'}. "
        f"Risk-free rate: {rf_txt}."
        + (" Sharpe is taken on the unlevered book, before the volatility target "
           "scales it: with idle cash and borrowing at the risk-free rate, scaling "
           "leaves Sharpe unchanged. Return and volatility beside it are the "
           "scaled book's." if use_vt else "")
    )

    if mc.get("engine") == "jump-diffusion":
        jp = mc["jump_params"]
        st.caption(
            f"Merton jump-diffusion: the engine found {jp['n_jumps']} jump days "
            f"in {jp['n_days']} (moves beyond {jp['k']:.0f} standard deviations), "
            f"about {jp['lambda_daily'] * TRADING_DAYS:.1f} jumps a year on top of "
            f"ordinary swings of {jp['sigma_d'] * np.sqrt(TRADING_DAYS):.0%} a year. "
            "Random jumps let a simulated day fall further than any real one; "
            "switch the return model to the bootstrap to compare the two on this "
            "book."
        )

source_txt = f"the {scenario_label} window" if scenario_label else (
    f"the {len(shocked_returns)} daily returns loaded "
    f"({shocked_returns.index[0].date()} to {shocked_returns.index[-1].date()})")
engine_txt = (
    "a Merton jump-diffusion model (ordinary daily moves plus rare random jumps), calibrated to"
    if use_jd else "bootstrap Monte Carlo, redrawing real days at random from"
)
st.caption(
    f"Methodology: {mc['n_simulations']:,} simulated years of {mc['horizon_days']} trading days each, "
    f"from {engine_txt} {source_txt}. Weighting: {alloc_label}{lev_txt}."
)

with tab_grit:
    st.caption(
        "Grit ranks your names by how each one came back from its own falls: "
        "how fast, how fully, and how reliably across real crises. Every name "
        "here has fallen; this measures what happened next."
    )
    try:
        grit = load_grit(tuple(loaded))
        gscores = grit["scores"]
        if gscores.empty:
            st.caption(
                "Not enough price history in these names to score grit "
                f"(each needs at least {MIN_HISTORY_DAYS} trading days)."
            )
        else:
            st.plotly_chart(
                hbar(gscores["grit_score"],
                     title_x="Grit score (0–100)", top_first=True),
                width="stretch", config=PLOTLY_CFG,
            )
            read_me(
                "Longer bar, grittier name. The name at the top has, over its own "
                "history, come back from its falls the fastest and most reliably. "
                "Scores run 0 to 100 within this basket: they rank these names "
                "against each other, not against the whole market.")
            grittiest = gscores.index[0]
            g = gscores.loc[grittiest]
            # NaN here means "no drawdown episode deep enough to measure",
            # which is an unknown record - never render it as a perfect one.
            if pd.isna(g["pct_recovered"]):
                _rec = ("has not yet fallen far enough to time a "
                        "recovery from")
            else:
                _med = g["median_recovery_days"]
                _rec = (f"won back the loss in {g['pct_recovered']:.0%} of its own falls"
                        + ("" if pd.isna(_med) else
                           f" (half took {_med:.0f} trading days or fewer)"))
            st.caption(
                f"{grittiest} ranks grittiest here: it {_rec}, ended "
                f"{g['consistency']:.0%} of its one-year stretches with a gain, "
                f"and lived through {g['n_regimes_survived']:.0f} of the named "
                f"crises."
            )

            panel_head("Grit, part by part", "Recovery · consistency · resilience")
            st.plotly_chart(grit_breakdown_fig(gscores), width="stretch", config=PLOTLY_CFG)
            st.caption(
                "Each bar ranks a name against the others in this basket, not on "
                "a fixed scale, so a different basket moves every number. Recovery "
                "counts falls of 5% or more; resilience uses only the crises the "
                "name traded through."
            )

            if grit["excluded"]:
                st.caption(
                    f"Left out for too little history (under {MIN_HISTORY_DAYS} "
                    f"trading days): {', '.join(grit['excluded'])}."
                )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Grit Zone unavailable: {safe_err('grit-zone-unavailable', exc)}")

# The three crises the case studies tell, by their names in
# HISTORICAL_REGIMES (test_the_case_studies_read_the_live_record).
CASE_STUDIES = ("COVID-19 crash (2020)", "2022 bear market", "Global Financial Crisis (2008)")


def _long_date(d) -> str:
    """23 March 2020 - built by hand: %-d is not portable to Windows."""
    return f"{d.day} {d:%B %Y}"


def case_row(r) -> str:
    """One case study as a ruled row: the crisis, then three cells - the
    fall, the way back and the bottom-day buyer a year on - each a label,
    the number and one line of what it means."""
    def cell(label, value, note):
        return (f'<div class="case-cell"><div class="case-k">{label}</div>'
                f'<div class="case-v">{value}</div><div class="case-n">{note}</div></div>')
    days, back = r.get("recovery_days"), r.get("recovery_date")
    year = r.get("trough_1y later")
    way = (cell("The way back", f"{int(days)} days",
                f"trading days to close at the old peak again, on {_long_date(back)}")
           if days is not None and not pd.isna(days) else
           cell("The way back", "Not back",
                "no close at the old peak within three trading years of the bottom, "
                "as far as the record runs"))
    later = (cell("A year later", f"{year:+.0%}", "for a buyer on the bottom day")
             if year is not None and not pd.isna(year) else
             cell("A year later", "Too recent", "less than a trading year of record after the bottom"))
    return (f'<div class="case-row"><div class="case-name">{html.escape(str(r["crisis"]))}</div>'
            + cell("The fall", f"{r['depth']:.0%}",
                   f"peak to bottom; the bottom was {_long_date(r['trough_date'])}")
            + way + later + "</div>")


# ---- Crisis Conviction: the emotional problem, priced ----
with tab_conviction:
    st.caption(
        "For every crisis we replay, here is what actually happened next, "
        "worked out from live market data."
    )
    try:
        conv = load_conviction_data()
        ctab, summ = conv["table"], conv["summary"]
        t1 = summ["trough_1y_later"]

        h1, h2, h3 = st.columns(3)
        h1.metric("Up one year after the bottom",
                  f"{round(t1['pct_positive'] * t1['n'])} / {t1['n']} crises")
        h2.metric("Typical return a year after the bottom", f"{t1['median']:+.0%}")
        h3.metric("Typical fall, peak to bottom", f"{ctab['depth'].median():.0%}")

        # Case studies (ME-28 part 4): three named crises told as one ruled
        # row each - the fall, the way back, a bottom-day buyer a year on.
        # Every figure is read from this live table; a crisis the record does
        # not cover is skipped, never filled in.
        panel_head("Case studies", "Three crashes, and the way back")
        _cases = [ctab[ctab["crisis"] == c].iloc[0] for c in CASE_STUDIES
                  if (ctab["crisis"] == c).any()]
        if _cases:
            st.markdown('<div class="case-list">' + "".join(case_row(r) for r in _cases)
                        + "</div>", unsafe_allow_html=True)
            st.caption(
                "Read from the record of SPY, a fund that tracks the S&P 500 (adjusted "
                "closes). The way back counts trading days from the lowest close to the "
                "first close at or above the peak before the fall.")

        panel_head("What a buyer actually got", "Crisis by crisis: the best day to buy and the worst")
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
        with st.expander("Show the full table: every crisis, row by row"):
            st.dataframe(
                show.style.format({c: "{:+.0%}" for c in pct_cols}, na_rep="-")
                    .format({"Crash depth": "{:.0%}"})
                    .map(_tone, subset=pct_cols[1:]),
                width="stretch", hide_index=True)
            st.markdown(
                '<div class="read-me">'
                'Each row is a real crisis. Trough buy: you bought the S&amp;P '
                '500 (SPY) at the lowest close of the crisis, the exact bottom. '
                'Peak buy: you bought at the top before the fall, the worst '
                'possible timing. The columns show where that money stood 1 and 3 '
                'trading years later. "-" means the crisis is too recent for that '
                'span: left out, not estimated.'
                '</div>', unsafe_allow_html=True)
            # The exceptions are READ from the table, never written in: which
            # crises leave a peak buyer behind depends on the live series, and
            # a hand-typed "the one exception is ..." goes stale silently.
            _p3y = pd.to_numeric(show["Peak buy, 3y later"], errors="coerce")
            _behind = show.loc[_p3y < 0, "Crisis"].tolist()
            st.caption(
                "Nobody can time the exact bottom, so the trough columns show which "
                "way the odds lean, not a strategy you could follow. That is why "
                "the peak columns sit beside them: the worst-timed buyer was ahead "
                f"three years later in {int((_p3y > 0).sum())} of "
                f"{int(_p3y.notna().sum())} crises with three years of data"
                + (f". The exception{'s' if len(_behind) > 1 else ''}: "
                   f"{', '.join(_behind)}, where three years was not enough."
                   if _behind else ".")
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Crisis Conviction unavailable: {safe_err('crisis-conviction-unavailable', exc)}")

# src/data_quality.py names its checks for code ("schema.duplicate_dates");
# a visitor reads plain words. The ids stay in the engine and its tests.
DQ_LABELS = {
    "schema.index": "Dates", "schema.duplicate_dates": "Duplicate dates",
    "schema.sorted": "Date order", "schema.dtype": "Numbers only",
    "positivity.non_positive_prices": "Positive prices",
    "coverage.min_rows": "Length of history", "freshness.staleness": "Freshness",
    "coverage.calendar_gaps": "Missing days", "sanity.extreme_moves": "Extreme moves",
}
DQ_MESSAGES = {"DatetimeIndex confirmed.": "Every row carries a date.",
               "Index is not a DatetimeIndex.": "The rows do not carry dates."}

with tab_dq:
    st.caption(
        "Every price download is checked before any risk number is worked out "
        "from it: dates present and in order, prices above zero, enough history, "
        "missing days, how recent it is, and any unusually large one-day move. These checks catch a "
        "broken or implausible feed; they cannot catch a wrong number that "
        "looks plausible."
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
        _name = DQ_LABELS.get(c["check"], c["check"])
        _says = DQ_MESSAGES.get(c["message"], c["message"])
        st.markdown(f'<div class="dq-row">{mark[c["status"]]} {html.escape(_name)}'
                    f': {html.escape(_says)}</div>', unsafe_allow_html=True)
    verdict_dq = "passed" if report["passed"] else "failed"
    panel_head("Data checks, overall", f"This feed {verdict_dq} the checks")

with tab_lineage:
    prov = provenance(tickers)
    if prov:
        # Each name is pulled on its own, so a basket can span several pulls:
        # say so rather than print one time for all of them.
        _newest = prov.get("fetched_newest_utc", prov["fetched_at_utc"])
        _fetched = (prov["fetched_at_utc"] if _newest == prov["fetched_at_utc"] else
                    f"{prov['fetched_at_utc']} to {_newest} (each name is pulled "
                    f"on its own and re-pulled after {CACHE_MAX_AGE_HOURS:g} hours)")
        st.markdown(
            f"- Source: {prov['source']}\n"
            f"- Fetched (UTC): {_fetched}\n"
            f"- Symbols: {', '.join(prov['symbols'])}\n"
            f"- Coverage: {prov['start']} to {prov['end']} "
            f"({prov['rows']} trading days)\n"
            f"- Library: yfinance {prov['yfinance_version']}"
        )
        st.caption(
            "Prices are live end-of-day adjusted closes, pulled on demand from "
            "Yahoo Finance. Every figure above is computed from this source by "
            "the engine's own code; no value originates from a language model. "
            f"A copy older than {CACHE_MAX_AGE_HOURS:g} hours is re-pulled "
            "automatically on the next interaction."
        )
    else:
        st.caption("Provenance record appears after the first live fetch.")

    panel_head("This run's audit trail", "Every step this session took, in order")
    st.caption(
        "Kept for this visit only: rebuilt on every change, never stored. A "
        "full compliance system would write it to permanent storage; this is "
        "the same idea at this engine's scale."
    )
    st.dataframe(pd.DataFrame(audit_log), width="stretch", hide_index=True)

    # ---- Your data: what stands in for a "delete my account" control ----
    panel_head("Your data", "What this app holds about you, and how to clear it")
    st.markdown(
        "There is no account to delete: Meliona has no sign-up, login or "
        "password, and no analytics or advertising SDK. It stores at most three "
        "things in your browser: Streamlit's anti-forgery cookie and your light "
        "or dark theme, and, once you press OK on the cookie notice, a note that "
        "you have seen it (details in the privacy note).\n\n"
        f"- This session, in memory only and gone when you close the tab: the "
        f"tickers you chose, the settings above, the preset in the address bar, "
        f"and the random reference {SESSION_REF}, which is not linked to you.\n"
        "- The server: cached Yahoo Finance price files, named by a hash of the "
        "ticker set. They are public market data, shared by every visitor who "
        "picks the same basket, and hold nothing about you.\n"
        "- Logs: errors and a session-start line, tagged with the reference "
        "above so a support email can be matched to a stack trace. They rotate "
        "away on size and hold no personal data. Once an hour the server also "
        "writes one line of anonymous totals: how many sessions started today "
        "and how many opened each preset basket, with no reference in it.\n"
        "- Third parties: nothing about you. The server fetches prices from "
        "Yahoo Finance only, and your browser talks only to this app.\n\n"
        f"To have the logs for your reference purged early, email "
        f"[{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL}) and quote {SESSION_REF}, "
        f"or [report this session]({FEEDBACK_LINK}) with the reference already "
        f"filled in."
    )
    if st.button("Clear this session's data", key="wipe_session"):
        # Deliberately session-scoped: it clears what THIS browser holds. It
        # does not clear the shared market-data cache - that is an operator
        # action, and visitors are viewers here (see .streamlit/config.toml).
        keep = st.session_state.session_ref
        opened = st.session_state.get("_usage_presets", set())
        st.session_state.clear()
        st.query_params.clear()           # the preset in the address too
        st.session_state.session_ref = keep
        st.session_state["_usage_presets"] = opened   # still one session: no recount
        st.session_state["_cleared"] = True
        st.rerun()
    # Shown after the reload: a message written just before st.rerun() is
    # wiped before anyone can read it (2026-11-04).
    if st.session_state.pop("_cleared", False):
        st.success("Done: this session's settings are cleared and the page is back "
                   "to its defaults.")

# ---- The builder, five questions, and a way to write in (ME-28 part 4) ----
# All below the tab strip, so nothing here moves an element above it.
st.markdown(about_html(), unsafe_allow_html=True)
st.markdown(faq_html(), unsafe_allow_html=True)

# Contact. There is no mail server and no database, so nothing is sent from
# here: the form checks the words, then prepares an email the visitor sends
# from their own app (hero.contact_mailto). Their text is held only while this
# visit lasts, is never printed into the page, never logged, never stored.
st.markdown(
    '<div class="contact-head" id="contact"><div class="kicker">Write to John</div>'
    '<h2 class="showcase-title">Contact</h2><div class="ar-rule"></div>'
    f'<p class="contact-sub">{RESPONSE_TIME} Questions, feedback, recruiting or a '
    'problem with a number: write it here and your own email app sends it.</p></div>',
    unsafe_allow_html=True)
_contact_slot = st.container(key="contact_zone")
with _contact_slot:
    if st.session_state.get("_contact_mail"):
        # The thank-you state: one step is left, and it is the visitor's own.
        st.markdown(
            '<div class="thanks-card" role="status"><div class="thanks-t">Thank you</div>'
            '<p class="thanks-p">Your email is ready. Open your email app with the button '
            'below and press send: nothing has left this page yet, and nothing you wrote '
            'is kept.</p>'
            '<div class="thanks-row"><span>Response time</span>'
            f'<b>{RESPONSE_WINDOW}</b></div>'
            '<div class="thanks-row"><span>Your session reference</span>'
            f'<b>{SESSION_REF}</b></div>'
            f'<div class="thanks-row"><span>Or write to</span><b><a href="mailto:{SUPPORT_EMAIL}">'
            f'{SUPPORT_EMAIL}</a></b></div></div>',
            unsafe_allow_html=True)
        _ty1, _ty2 = st.columns([1, 1], width=720)
        with _ty1:
            st.link_button("Open your email app", st.session_state["_contact_mail"],
                           type="primary", width="stretch")
        with _ty2:
            if st.button("Write another message", key="contact_again", width="stretch"):
                st.session_state.pop("_contact_mail", None)
                st.rerun()
        st.markdown('<p class="thanks-keep">Want the reply details for later? Keep '
                    '<a href="/thanks.html" target="_blank" rel="noopener">a page you can '
                    'bookmark</a>.</p>', unsafe_allow_html=True)
    else:
        _form_col, _aside_col = st.columns([3, 2], gap="large")
        with _form_col:
            with st.form("contact_form", clear_on_submit=False, border=False):
                _c_name = st.text_input("Your name (optional)", max_chars=CONTACT_NAME_MAX,
                                        key="contact_name")
                _c_topic = st.selectbox("Topic", CONTACT_TOPICS, key="contact_topic")
                _c_msg = st.text_area("Message", max_chars=CONTACT_MAX, height=160,
                                      key="contact_message",
                                      placeholder="What would you like to say?")
                _c_go = st.form_submit_button("Prepare my email", type="primary",
                                              key="contact_send")
            if _c_go:
                _mail, _why = contact_mailto(SUPPORT_EMAIL, _c_name, _c_topic, _c_msg, SESSION_REF)
                if _why:
                    st.error(_why, icon=":material/error:")
                else:
                    st.session_state["_contact_mail"] = _mail
                    st.rerun()
        with _aside_col:
            st.markdown(
                '<div class="contact-aside"><div class="thanks-row"><span>Response time</span>'
                f'<b>{RESPONSE_WINDOW}</b></div>'
                f'<div class="thanks-row"><span>Email</span><b><a href="mailto:{SUPPORT_EMAIL}">'
                f'{SUPPORT_EMAIL}</a></b></div>'
                '<p class="contact-note">Nothing is sent from this page. The button opens an '
                'email in your own app with your words filled in, so you see exactly what '
                f'goes out. The limit is {CONTACT_MAX:,} characters.</p></div>',
                unsafe_allow_html=True)

# ---- Closing band: three links onward, then copyright, support, accessibility ----
st.markdown("""
<div class="meliona-footer">
  <div class="f-rail">
    <a class="f-box" href="#grit-showcase">The Grit Zone
      <small>Which names get back up after a fall.</small></a>
    <a class="f-box" href="#crisis-record">Crisis Conviction
      <small>What buyers earned in __NCRISES__ real crises.</small></a>
    <a class="f-box" href="#engine">The engine
      <small>Test any basket against a bad year, from the weights to the big number.</small></a>
  </div>
  <div class="f-bar">
    <div>Meliona &middot; Portfolio Risk Engine &middot; &copy; 2026 John Nguyen</div>
    <div class="f-marks">__MARKS__</div>
  </div>
  <div class="f-bar">
    <div>Live end-of-day data: Yahoo Finance &middot; Educational analysis, not investment advice</div>
  </div>
  <div class="f-bar">
    <div class="f-links">Support: <a href="mailto:__SUPPORT__?subject=Meliona%20%E2%80%93%20session%20__REF__">__SUPPORT__</a>__PHONE__
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/PRIVACY.md">Privacy</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/TERMS.md">Terms</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/ACCESSIBILITY.md">Accessibility</a>
      &middot; <a href="https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint">Source code on GitHub</a></div>
    <div>No accounts &middot; no tracking &middot; session reference <span class="f-ref">__REF__</span> (quote it when you write in)</div>
  </div>
  <div class="f-bar">
    <div>Built for WCAG 2.2 AA. The Risk Topology map is drawn on canvas and carries a
      text description; charts are accompanied by their numbers in text. Known gaps are
      listed in the accessibility statement.</div>
    <div>Hit a barrier? Email <a href="mailto:__SUPPORT__?subject=Meliona%20accessibility%20%E2%80%93%20session%20__REF__">__SUPPORT__</a>
      with the reference above; a barrier is fixed as a defect.</div>
  </div>
</div>
""".replace("__MARKS__", profile_marks("f-mark")).replace("__SUPPORT__", SUPPORT_EMAIL)
   .replace("__REF__", SESSION_REF).replace(
    "__PHONE__", f' &middot; <a href="tel:{SUPPORT_TEL}">{SUPPORT_PHONE}</a>' if SUPPORT_PHONE else "")
   .replace("__NCRISES__", str(len(HISTORICAL_REGIMES))),
    unsafe_allow_html=True)
