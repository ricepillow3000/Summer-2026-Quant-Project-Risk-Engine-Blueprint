/* Meliona motion: how the page arrives as you scroll.
 *
 *   1. Reveals - each block fades in and slides up (headings slide in from the
 *      left) the first time it enters the screen. An IntersectionObserver
 *      reports the moment it arrives; nothing listens to every scrolled pixel.
 *      Blocks that arrive together are staggered top to bottom, left to right.
 *   2. Scrubs - decorative layers (the band's photo, the bronze lintel over each
 *      panel) follow the scroll position itself. app.css does this with scroll
 *      timelines, off the main thread; where a browser has none, this file
 *      feeds the same position as --scrub, and only while the layer is on
 *      screen.
 *   3. Plates - the heavy photos behind a band start downloading 600px before
 *      the band reaches the screen, and fade in once there. Words come first.
 *
 * Only opacity and transform ever change: the compositor (GPU) animates them,
 * and neither moves the layout, so nothing on the page shifts or reflows.
 * The hidden state lives under html.reveal-on, which the page's <head> sets
 * before the first paint (tools/public_files.py) - no flash of the finished
 * page. Without JavaScript, or with reduced motion, that class is never set
 * and every block is simply there. Distance, duration and easing live in
 * app.css (search "REVEAL").
 */
(function () {
  'use strict';
  var W = window, D = document, H = D.documentElement;
  if (W.__melionaReveal) return;                      // one copy per page

  var STAGGER_MS = 60;    // gap between blocks that arrive together: higher = slower cascade
  var MAX_STEPS = 8;      // a long list waits at most MAX_STEPS x STAGGER_MS

  // What reveals. Our own blocks and Streamlit's (metrics, charts, tables) -
  // tagged here, as each one is added to the page, before it is painted.
  var ITEMS = ['.reveal-item', '.showcase-eyebrow', '.showcase-title', '.showcase-body',
    '.gungnir-sub', '.pillar-card', '.slab', '.cta-btn', '.cta-link', '.sec-mark',
    '.panel-head', '.verdict-box', '.gungnir-plaque', '.orient', '.read-me', '.f-box',
    '[data-testid="stMetric"]', '[data-testid="stPlotlyChart"]',
    '[data-testid="stDataFrame"]'].join(',');
  var LEFT = '.showcase-eyebrow, .showcase-title, .showcase-body, .gungnir-sub, .sec-mark';
  var LIFT = '.pillar-card';
  // Never animated: the hero (it has its own first-load arrival), the boot
  // skeleton, the map (its own probe measures it), and any section a button
  // flies to - the glide lands on it, and it must not fade as it lands
  // (deep_scan: "destination never fades").
  var SKIP = '#hero, #boot-skel, .st-key-risk-map';
  var DEST = '#grit-showcase, #conviction, #crisis-record, #the-slice, #engine, #analysis, #the-book';
  var SCRUB = '.showcase-row, .panel-head';
  // Each photo plate, and the class on <html> that attaches it (app.css,
  // main.py). On <html>, not the band: Streamlit may rewrite a block's own
  // classes on a rerun, and the photo would go with them.
  var PLATES = { '.showcase-row': 'has-band', '.st-key-gungnir_zone': 'has-gungnir' };
  var PLATE = Object.keys(PLATES).join(',');
  var ALL = [ITEMS, SCRUB, PLATE].join(',');
  function attach(el) {
    for (var sel in PLATES) if (el.matches(sel)) H.classList.add(PLATES[sel]);
  }

  var calm = !('IntersectionObserver' in W) ||
    (W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var api = W.__melionaReveal = { revealAll: function () {} };
  if (calm) { H.classList.remove('reveal-on'); return; }   // everything as drawn, photos at once

  // ---- 1. reveals ----------------------------------------------------------
  var seen = new IntersectionObserver(function (entries) {
    entries.filter(function (e) { return e.isIntersecting; })
      // reading order = document order: a row of metrics left to right, a
      // heading before its line. (Their boxes are still mid-transform, so
      // on-screen positions would mis-sort a tilted card against a slid one.)
      .sort(function (a, b) {
        return a.target.compareDocumentPosition(b.target) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
      })
      .forEach(function (e, i) { show(e.target, Math.min(i, MAX_STEPS) * STAGGER_MS); });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0 });  // fires once 8% inside the bottom edge

  function show(el, delay) {
    seen.unobserve(el);
    el.style.setProperty('--reveal-delay', delay + 'ms');
    el.classList.add('is-visible');
    var end = function (ev) {
      if (ev && (ev.target !== el || ev.propertyName !== 'transform')) return;
      done(el);
    };
    el.addEventListener('transitionend', end);
    el.__rvTimer = setTimeout(end, delay + 1500);     // a tab hidden mid-way never ends
  }
  // Landed: hand the block back to its own styles (hover lifts, transitions)
  // and drop will-change, so the GPU layer it held is released.
  function done(el) {
    clearTimeout(el.__rvTimer);
    el.classList.remove('reveal-item', 'reveal-item--left', 'reveal-item--lift', 'is-visible');
    el.style.removeProperty('--reveal-delay');
  }
  api.revealAll = function () {                       // the end state, at once (layout probes)
    [].forEach.call(D.querySelectorAll('.reveal-item'), function (el) { seen.unobserve(el); done(el); });
  };

  // ---- 2. scrubs (fallback only) -------------------------------------------
  var native = W.CSS && CSS.supports && CSS.supports('animation-timeline: view()');
  var live = [], frame = 0;
  function paint() {
    frame = 0;
    var vh = W.innerHeight;
    live.forEach(function (el) {                      // 0 = just below the screen, 1 = just above
      var r = el.getBoundingClientRect();
      var p = Math.min(1, Math.max(0, (vh - r.top) / (vh + r.height)));
      el.style.setProperty('--scrub', p.toFixed(3));
    });
  }
  function onScroll() { if (!frame) frame = W.requestAnimationFrame(paint); }
  var onScreen = native ? null : new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var i = live.indexOf(e.target);
      if (e.isIntersecting && i < 0) live.push(e.target);
      if (!e.isIntersecting && i >= 0) live.splice(i, 1);
    });
    // the scroll listener exists only while a scrub layer is on screen
    D.removeEventListener('scroll', onScroll, true);
    if (live.length) { D.addEventListener('scroll', onScroll, { capture: true, passive: true }); onScroll(); }
  });

  // ---- 3. plates -------------------------------------------------------------
  var near = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      near.unobserve(e.target);
      attach(e.target);                               // the photo downloads now, then fades up
    });
  }, { rootMargin: '600px 0px' });

  // ---- tagging -----------------------------------------------------------------
  // If this file arrived after the page had painted (no <head> snippet: a
  // local run), what is already on screen stays put; only what is below joins.
  var late = !H.classList.contains('reveal-on');
  function inView(el) {
    var r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < W.innerHeight && r.width > 0;
  }
  function scan(root) {
    if (root.nodeType !== 1) return;
    var found = root.matches(ALL) ? [root] : [];
    found.push.apply(found, root.querySelectorAll(ALL));
    found.forEach(function (el) {
      if (el.__rv || el.closest(SKIP)) return;
      el.__rv = 1;
      if (el.matches(PLATE)) { if (late) attach(el); else near.observe(el); }
      if (el.matches(SCRUB) && onScreen) onScreen.observe(el);
      if (!el.matches(ITEMS) || el.matches(DEST) || el.querySelector(DEST)) return;
      var parent = el.parentElement && el.parentElement.closest('.reveal-item');
      if (parent || (late && inView(el))) return;     // never animate inside an animating block
      el.classList.add('reveal-item');
      if (el.matches(LEFT)) el.classList.add('reveal-item--left');
      else if (el.matches(LIFT)) el.classList.add('reveal-item--lift');
      seen.observe(el);
    });
  }
  scan(D.body);
  late = false;
  H.classList.add('reveal-on');
  // Streamlit draws the page after this runs, and redraws parts of it on every
  // rerun: each new node is tagged in the same task it is added, so it is
  // already hidden when the browser first paints it.
  new MutationObserver(function (muts) {
    for (var i = 0; i < muts.length; i++) {
      var add = muts[i].addedNodes;
      for (var j = 0; j < add.length; j++) scan(add[j]);
    }
  }).observe(D.body, { childList: true, subtree: true });
})();
