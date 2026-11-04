/* Meliona motion: how the page arrives as you scroll.
 *
 *   1. Reveals - every block on the page arrives the first time it enters the
 *      screen, from where it sits (John, 2026-11-02): a block on the left
 *      third slides in from the left, one on the right third from the right,
 *      and one in the middle turns face up like a dealt card (a wide one tips
 *      up off its bottom edge, a narrow one turns on its spine). An
 *      IntersectionObserver reports the moment it arrives; nothing listens to
 *      every scrolled pixel. Blocks that arrive together are staggered top to
 *      bottom, left to right.
 *   2. Scrubs - decorative layers (the band's photo, the bronze lintel over each
 *      panel) follow the scroll position itself. app.css does this with scroll
 *      timelines, off the main thread; where a browser has none, this file
 *      feeds the same position as --scrub, and only while the layer is on
 *      screen.
 *   3. Plates - the heavy photos behind a band start downloading 600px before
 *      the band reaches the screen, and fade in once there. Words come first.
 *   4. Glide - a mouse wheel or trackpad eases the page to where it was sent
 *      instead of stepping it there. It stands aside while a button's glide
 *      flies, inside anything that scrolls on its own (a table, a list), and
 *      for touch and keys, which keep the browser's own scrolling.
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

  var STAGGER_MS = 90;    // gap between blocks that arrive together: higher = slower cascade
  var MAX_STEPS = 8;      // a long list waits at most MAX_STEPS x STAGGER_MS

  // What reveals. Our own blocks and Streamlit's (metrics, charts, tables) by
  // name; every other block on the page - a control, a caption, a card, a
  // line of the Book - as Streamlit's own leaf container. Tagged as each one
  // is added to the page, before it is painted.
  var NAMED = ['.reveal-item', '.showcase-eyebrow', '.showcase-title', '.showcase-body',
    '.gungnir-sub', '.pillar-card', '.slab', '.cta-btn', '.cta-link', '.sec-mark',
    '.panel-head', '.verdict-box', '.gungnir-plaque', '.orient', '.read-me', '.f-box',
    '[data-testid="stMetric"]', '[data-testid="stPlotlyChart"]',
    '[data-testid="stDataFrame"]'].join(',');
  var LEAF = '[data-testid="stElementContainer"]';
  var ITEMS = NAMED + ',' + LEAF;
  // Never animated: the hero (it has its own first-load arrival), the boot
  // skeleton, the map (its own probe measures it), the glide's script frame,
  // and any section a button flies to - the glide lands on it, and it must
  // not fade as it lands (deep_scan: "destination never fades").
  var SKIP = '#hero, #boot-skel, .st-key-risk-map, .st-key-glide';
  var DEST = '#grit-showcase, #conviction, #crisis-record, #the-slice, #engine, #analysis, #the-book';
  // Where a block sits decides how it arrives: its centre in the left or right
  // share of the page slides in from that side; in the middle it turns over.
  var SIDE = 0.42;        // left of 42% (or right of 58%) of the page's width
  var WIDE = 0.6;         // wider than 60% of the page: tips up, not turns
  var CARD = 0.45;        // a card is narrower than 45% of the page; a wider block in the middle tips up
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
  // The cone's wind tunnel (static/windtunnel.js) comes with the motion: one
  // entry point, so the head and main.py's fallback both bring it.
  if (!D.getElementById('meliona-wind')) {
    var wt = D.createElement('script'), me = D.getElementById('meliona-reveal');
    wt.id = 'meliona-wind';
    wt.src = (me && me.src ? me.src : 'app/static/reveal.js').replace(/reveal\.js(\?.*)?$/, 'windtunnel.js');
    D.head.appendChild(wt);
  }

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
    var how = side(el);
    el.classList.add('reveal-item--' + how);          // jumps there while still invisible
    // A card turns on its spine with the eye 3x its width away, never closer:
    // nearer, its front edge swings past the eye and it draws as a sliver.
    el.style.setProperty('--reveal-depth', Math.max(1000, 3 * el.offsetWidth) + 'px');
    void el.offsetWidth;                               // ...and starts from there
    el.style.setProperty('--reveal-delay', delay + 'ms');
    el.classList.add('is-visible');
    var end = function (ev) {
      if (ev && ev.target !== el) return;
      if (ev && ev.type === 'transitionend' && ev.propertyName !== 'transform') return;
      done(el);
    };
    el.addEventListener('transitionend', end);
    el.addEventListener('transitioncancel', end);     // interrupted: land, never hang mid-way
    el.__rvTimer = setTimeout(end, delay + 2200);     // a tab hidden mid-way never ends
    // A block whose content is redrawn while it arrives (Streamlit fills a
    // placeholder after a rerun: the Book's sentence, 2026-11-04) changes
    // size mid-move. Its move was chosen for the old size, so it lands at
    // once instead of finishing a turn meant for a different block.
    el.__rvW = el.offsetWidth; el.__rvH = el.offsetHeight;
    if (grew) grew.observe(el);
  }
  var grew = W.ResizeObserver ? new ResizeObserver(function (entries) {
    entries.forEach(function (e) {
      var el = e.target, w = el.offsetWidth, h = el.offsetHeight;
      if (!el.classList.contains('is-visible')) { grew.unobserve(el); return; }
      if (Math.abs(w - el.__rvW) > 24 || Math.abs(h - el.__rvH) > 24) done(el);
    });
  }) : null;
  // Measured as it arrives, not when tagged: Streamlit lays a block out after
  // adding it, and a tab or drawer opened later sits wherever it opens. The
  // hidden state only lifts and shrinks about the centre, so the centre read
  // here is where the block will land.
  // Widths are layout widths (offsetWidth): the waiting state's transform
  // must not make a block look narrower than it is. A block with no size yet
  // (an empty placeholder) tips up: it can never be mistaken for a card.
  function side(el) {
    var sc = D.querySelector(SCROLLER) || D.body;
    var page = sc.getBoundingClientRect(), pw = sc.clientWidth || page.width;
    var w = el.offsetWidth, r = el.getBoundingClientRect();
    if (!pw || !w || !el.offsetHeight) return 'flip';
    if (w > WIDE * pw) return 'flip';
    var f = ((r.left + r.right) / 2 - page.left) / pw;
    if (f >= SIDE && f <= 1 - SIDE && w > CARD * pw) return 'flip';
    return f < SIDE ? 'left' : f > 1 - SIDE ? 'right' : 'card';
  }
  // Landed: hand the block back to its own styles (hover lifts, transitions)
  // and drop will-change, so the GPU layer it held is released.
  function done(el) {
    clearTimeout(el.__rvTimer);
    if (grew) grew.unobserve(el);
    el.classList.remove('reveal-item', 'reveal-item--left', 'reveal-item--right', 'reveal-item--flip',
                        'reveal-item--card', 'is-visible');
    el.style.removeProperty('--reveal-delay');
    el.style.removeProperty('--reveal-depth');
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

  // ---- 4. glide ----------------------------------------------------------------
  // The wheel sets where the page is going; each frame closes a share of the
  // distance (GLIDE_EASE), so a notch eases in and a long spin coasts out.
  var GLIDE_EASE = 0.09;  // share of the remaining distance per frame: lower = longer coast
  var SCROLLER = '[data-testid="stMain"]';
  var aim = null, put = null, run = 0;
  function scrolls(n, dy) {                            // can this inner box take the wheel itself?
    var o = W.getComputedStyle(n).overflowY;
    if (o !== 'auto' && o !== 'scroll') return false;
    return dy > 0 ? n.scrollTop + n.clientHeight < n.scrollHeight - 1 : n.scrollTop > 0;
  }
  function coast() {
    var sc = D.querySelector(SCROLLER);
    // someone else moved the page (a button's glide, a key, the scrollbar):
    // theirs wins, and this wheel's aim is dropped
    if (!sc || aim === null || (put !== null && Math.abs(sc.scrollTop - put) > 2)) { aim = put = null; run = 0; return; }
    var cur = sc.scrollTop, next = cur + (aim - cur) * GLIDE_EASE;
    if (Math.abs(aim - next) < 0.5) next = aim;
    sc.scrollTop = next;
    put = sc.scrollTop;
    if (put === aim || Math.abs(put - cur) < 0.05 && next !== aim) { aim = put = null; run = 0; return; }
    run = W.requestAnimationFrame(coast);
  }
  W.addEventListener('wheel', function (e) {
    if (e.ctrlKey || e.defaultPrevented || !e.deltaY || W.__melionaStop) return;   // zoom; a glide in flight
    var sc = D.querySelector(SCROLLER);
    if (!sc || !sc.contains(e.target)) return;
    for (var n = e.target; n && n !== sc; n = n.parentElement) if (scrolls(n, e.deltaY)) return;
    e.preventDefault();
    var dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaMode === 2 ? e.deltaY * sc.clientHeight : e.deltaY;
    var max = sc.scrollHeight - sc.clientHeight;
    aim = Math.max(0, Math.min((aim === null ? sc.scrollTop : aim) + dy, max));
    if (!run) { put = sc.scrollTop; sc.style.scrollBehavior = 'auto'; run = W.requestAnimationFrame(coast); }
  }, { passive: false });

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
      if (!el.matches(ITEMS) || el.matches(DEST) || el.querySelector(DEST) || el.querySelector(SKIP)) return;
      // a container block arrives through its parts: its own leaves, or the
      // named blocks inside it (a row of cards keeps its card-by-card deal)
      if (!el.matches(NAMED) && (el.querySelector(LEAF) || el.querySelector(NAMED))) return;
      var parent = el.parentElement && el.parentElement.closest('.reveal-item');
      if (parent || (late && inView(el))) return;     // never animate inside an animating block
      el.classList.add('reveal-item');
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
