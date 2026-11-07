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
 *   2. Rules - the charcoal key of each section heading's rule draws itself
 *      across once, as the heading arrives (app.css). It used to follow the
 *      scroll position, so a reader who stopped with a heading low on the
 *      screen saw its rule frozen half-drawn (John, 2026-11-06).
 *   3. Glide - a mouse wheel or trackpad tows the page, like a boat on a line, to where it was sent
 *      instead of stepping it there. It stands aside while a button's glide
 *      flies, inside anything that scrolls on its own (a table, a list), and
 *      for touch and keys, which keep the browser's own scrolling. Pushed
 *      past either end of the page it stretches, with resistance, and eases
 *      back when let go: a rubber band, never a bounce.
 *   4. Popovers - a dropdown or a help tip grows out of the control that
 *      opened it and shrinks back into it (--pop-origin), and a menu's rows
 *      arrive 30ms apart. 5. Masks - John's photographs are uncovered from
 *      their cap down as they arrive. app.css "MOTION" holds the timings.
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
  var MAX_STEPS = 5;      // a long list waits at most MAX_STEPS x STAGGER_MS
                          // (8 held the last of a batch back 720ms, still hidden on screen)

  // What reveals. Our own blocks and Streamlit's (metrics, charts, tables) by
  // name; every other block on the page - a control, a caption, a card, a
  // line of the Book - as Streamlit's own leaf container. Tagged as each one
  // is added to the page, before it is painted.
  var NAMED = ['.reveal-item', '.showcase-title', '.showcase-body',
    '.gungnir-sub', '.pillar-card', '.slab', '.cta-btn', '.cta-link',
    '.panel-head', '.verdict-box', '.gungnir-plaque', '.orient', '.read-me', '.f-box',
    '[data-testid="stMetric"]', '[data-testid="stPlotlyChart"]',
    '[data-testid="stDataFrame"]'].join(',');
  var LEAF = '[data-testid="stElementContainer"]';
  var ITEMS = NAMED + ',' + LEAF;
  // Never animated: the hero (it has its own first-load arrival), the boot
  // skeleton, the map (its own probe measures it), the glide's script frame,
  // and any section a button flies to - the glide lands on it, and it must
  // not fade as it lands (deep_scan: "destination never fades").
  // The thank-you receipt too: it confirms with its own pulse (app.css MOTION).
  var SKIP = '#hero, #boot-skel, .st-key-risk-map, .st-key-glide, .thanks-card';
  var DEST = '#grit-showcase, #conviction, #crisis-record, #the-slice, #engine, #analysis, #the-book, #contact';
  // What the reader opens - a drawer's body, a tab's panel - is simply there
  // when it opens: an arrival inside it is motion the reader did not ask for,
  // and it is where iPhones froze blocks mid-turn (John's screenshots,
  // 2026-11-05: the Weights and Stress test drawers, the Book's drawer).
  var OPENED = '[data-testid="stExpanderDetails"], [data-testid="stTabPanel"]';
  // Where a block sits decides how it arrives: its centre in the left or right
  // share of the page slides in from that side; in the middle it turns over.
  var SIDE = 0.42;        // left of 42% (or right of 58%) of the page's width
  var WIDE = 0.6;         // wider than 60% of the page: tips up, not turns
  var CARD = 0.45;        // a card is narrower than 45% of the page; a wider block in the middle tips up
  var LEAD = '15%';       // an arrival starts this far below the screen's bottom edge

  var calm = !('IntersectionObserver' in W) ||
    (W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var api = W.__melionaReveal = { revealAll: function () {} };
  // The bottom dock - the phone shortcut and the cookie notice
  // (static/dock.js) - comes with this file: one entry point, so the head and
  // main.py's fallback both bring it. Under reduced motion it loads too (the
  // dock shows without its slide).
  [['meliona-dock', 'dock.js']].forEach(function (f) {
    var id = f[0], file = f[1];
    if (D.getElementById(id)) return;
    var s = D.createElement('script'), me = D.getElementById('meliona-reveal');
    s.id = id;
    s.src = (me && me.src ? me.src : 'app/static/reveal.js').replace(/reveal\.js(\?.*)?$/, file);
    D.head.appendChild(s);
  });
  // A press before the glide script exists. main.py's glide ships in a frame
  // that loads after the first links are drawn - "Skip to the analysis" was
  // on screen 229ms before it, so a quick press fell to the browser's own
  // jump to a section not written yet, which went nowhere (Live audit 8c
  // early, four times on 2026-11-05: moved 5px, ended 2400px short). This
  // file is in the <head>, there first: it keeps the press on the page
  // (__melionaPending, the same record the glide resumes a wait from), and
  // the glide flies it as it arrives. Should the glide never come, the
  // browser's own jump is made once the section exists.
  D.addEventListener('click', function (e) {
    if (D.__melionaGlide || e.defaultPrevented || e.button || e.metaKey || e.ctrlKey ||
        e.shiftKey || e.altKey) return;
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    var id = a ? a.getAttribute('href').slice(1) : '';
    if (!id) return;
    e.preventDefault();
    var p = D.__melionaPending = { a: a, id: id, ms: Date.now() };
    (function wait() {
      if (D.__melionaGlide || D.__melionaPending !== p) return;   // the glide took it over
      var el = D.getElementById(id);
      if (el && Date.now() - p.ms > 3000) { D.__melionaPending = null; el.scrollIntoView(); return; }
      if (Date.now() - p.ms < 180000) setTimeout(wait, 100);
    })();
  }, true);
  if (calm) { H.classList.remove('reveal-on'); return; }   // everything as drawn
  // Flat motion on WebKit (every iPhone and iPad browser, and Safari: only
  // WebKit has GestureEvent) and on any touch screen: the same arrivals as
  // slides and rises, with no perspective turn. WebKit left 3D-turned blocks
  // frozen part-way, squeezed and sideways (John's iPhone, 2026-11-05); a
  // phone gains nothing from a turn anyway - every block is the full width.
  var flat = 'GestureEvent' in W || !!(W.matchMedia && W.matchMedia('(pointer: coarse)').matches);
  if (flat) H.classList.add('reveal-flat');

  // ---- 1. reveals ----------------------------------------------------------
  function arrive(entries) {
    entries.filter(function (e) { return e.isIntersecting; })
      // reading order = document order: a row of metrics left to right, a
      // heading before its line. (Their boxes are still mid-transform, so
      // on-screen positions would mis-sort a tilted card against a slid one.)
      .sort(function (a, b) {
        return a.target.compareDocumentPosition(b.target) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
      })
      .forEach(function (e, i) { show(e.target, Math.min(i, MAX_STEPS) * STAGGER_MS); });
  }
  // Fires as a block's slot comes within LEAD of the bottom edge, so it is
  // already arriving when the reader first sees it. It used to wait until it
  // was 8% INSIDE the edge (and its waiting state sits 46px low): a slow
  // scroll stopped with a heading, a card or the contact form's button
  // missing or out of place at the bottom of the screen for up to 3s,
  // until a scroll brought it "into full view" (John, 2026-11-06).
  // The root is the page's own scroller (stMain), not the window: a margin
  // only widens the root, and stMain clips everything below it first, so on
  // the window the lead counted for nothing. Streamlit draws stMain after
  // this file runs: until then the window stands in, and when stMain comes
  // (or is ever redrawn) the blocks still waiting move to a fresh observer.
  var seen = null, seenRoot;
  function observer() {
    var sc = D.querySelector(SCROLLER);
    if (seen && seenRoot === sc) return seen;
    var old = seen;
    seenRoot = sc;
    seen = new IntersectionObserver(arrive, { root: sc, rootMargin: '0px 0px ' + LEAD + ' 0px', threshold: 0 });
    if (old) {
      old.disconnect();
      [].forEach.call(D.querySelectorAll('.reveal-item:not(.is-visible)'), function (el) { seen.observe(el); });
    }
    return seen;
  }

  function show(el, delay) {
    observer().unobserve(el);
    var how = side(el);
    el.classList.add('reveal-item--' + how);          // jumps there while still invisible
    // A card turns on its spine with the eye 3x its width away, never closer:
    // nearer, its front edge swings past the eye and it draws as a sliver.
    el.style.setProperty('--reveal-depth', Math.max(1000, 3 * el.offsetWidth) + 'px');
    void el.offsetWidth;                               // ...and starts from there
    el.style.setProperty('--reveal-delay', delay + 'ms');
    el.classList.add('is-visible');
    var end = function (ev) {
      if (ev && (ev.target !== el || ev.pseudoElement)) return;   // a heading's rule ends on its own clock
      if (ev && ev.type === 'transitionend' && ev.propertyName !== 'transform') return;
      done(el);
    };
    el.addEventListener('transitionend', end);
    el.addEventListener('transitioncancel', end);     // interrupted: land, never hang mid-way
    el.__rvTimer = setTimeout(end, delay + 2200);     // a tab hidden mid-way never ends
    // A CARD whose content is redrawn while it turns (Streamlit fills a
    // placeholder after a rerun: the Book's sentence, 2026-11-04) changes
    // size mid-turn. Its turn was chosen for the old size, so it lands at
    // once instead of finishing a turn meant for a different block. Only a
    // card: a slide or a tip-up stays right whatever the size, and on a cold
    // load nearly every block grows while it arrives - landing those at once
    // made the whole page appear without its motion (John, 2026-11-04).
    el.__rvW = el.offsetWidth; el.__rvH = el.offsetHeight;
    if (grew && how === 'card') grew.observe(el);
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
    [].forEach.call(D.querySelectorAll('.reveal-item'), function (el) { observer().unobserve(el); done(el); });
    [].forEach.call(D.querySelectorAll('.mask-reveal'), function (el) { veiled.unobserve(el.parentElement); unveil(el); });
  };

  // ---- masks: the photographs ----------------------------------------------
  // Uncovered from the cap down by a clip-path mask as they arrive, once; the
  // classes go when it lands, so nothing stays clipped or on its own layer.
  // Its frame (the <figure>) is what is watched: a box its own mask hides
  // entirely never counts as on screen.
  var MASK = '.about-photo picture, .story-photo picture';
  var veiled = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      var el = e.target.__mask;
      veiled.unobserve(e.target);
      if (!el || !el.classList.contains('mask-reveal')) return;
      el.style.setProperty('--mask-delay', '120ms');
      void el.offsetWidth;
      el.classList.add('is-unmasked');
      el.addEventListener('transitionend', function (ev) { if (ev.target === el) unveil(el); });
      el.__mkTimer = setTimeout(function () { unveil(el); }, 1600);
    });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0 });
  function unveil(el) {
    clearTimeout(el.__mkTimer);
    el.classList.remove('mask-reveal', 'is-unmasked');
    el.style.removeProperty('--mask-delay');
  }

  // ---- popovers: from the trigger, back to the trigger ----------------------
  // react-aria draws each dropdown and help tip in a portal at the end of
  // <body>. Its panel scales from --pop-origin: the point of the panel's edge
  // nearest the control that opened it (the focused field for a menu, the
  // hovered mark for a tip), so it grows out of that control and, on close,
  // shrinks back into it. Measured in the frame before the first paint, and
  // again as react-aria settles the popover's position (a tip is first drawn
  // at the page's corner, then moved beside its mark).
  var POP = '[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectDropdown"], [role="tooltip"][data-rac]';
  var hovered = null;
  D.addEventListener('pointerover', function (e) { hovered = e.target; }, { capture: true, passive: true });
  function aimPop(pop) {
    var panel = pop.firstElementChild;
    while (panel && panel.tagName === 'SPAN') panel = panel.nextElementSibling;   // react-aria's focus-scope markers
    if (!panel || pop.hasAttribute('data-exiting')) return;   // closing: it shrinks back to where it came from
    var tip = pop.getAttribute('role') === 'tooltip';
    var trig = tip ? hovered && hovered.closest && hovered.closest('[data-testid="stTooltipHoverTarget"], button, a')
                   : D.activeElement && D.activeElement.closest('[data-testid="stSelectbox"], [data-testid="stMultiSelect"]');
    var t = trig && trig.getBoundingClientRect && trig.getBoundingClientRect();
    var r = panel.getBoundingClientRect();
    if (!t || !r.width || !r.height) return;
    var x = Math.max(0, Math.min(r.width, (t.left + t.right) / 2 - r.left));
    var y = (t.top + t.bottom) / 2 < (r.top + r.bottom) / 2 ? 0 : r.height;   // the trigger sits above, or below
    panel.style.setProperty('--pop-origin', x.toFixed(0) + 'px ' + y.toFixed(0) + 'px');
    if (tip || pop.__rows) return;
    pop.__rows = 1;
    // the rows: the first eight are numbered for their 30ms stagger, for this opening only
    var rows = pop.querySelectorAll('[role="option"]');
    for (var i = 0; i < rows.length && i < 8; i++) rows[i].style.setProperty('--row', i);
    pop.classList.add('pop-rows');
    setTimeout(function () { pop.classList.remove('pop-rows'); }, 520);
  }

  // ---- 3. glide ----------------------------------------------------------------
  // A boat on a tow line (John, 2026-11-04: "heavier ... like a boat being
  // tugged", not static). The wheel moves the tug - where the page is going
  // (aim). The page is the hull: a mass pulled toward the aim by a critically
  // damped spring, so it gathers way instead of jumping, keeps its momentum
  // through the next notch of the wheel, and comes to rest on the aim without
  // overshooting or bouncing back. Integrated on the frame's own clock, so a
  // fast and a slow screen move the hull alike.
  var TOW = 3.8;          // the tow line's natural frequency (rad/s): lower = heavier hull; ~1.7 s to settle
                          // (John, 2026-11-06: "heavy ... slightly delayed ... more fluid": 4.6 -> 3.8)
  var SCROLLER = '[data-testid="stMain"]';
  var aim = null, put = null, run = 0, pos = 0, vel = 0, last = 0;
  function scrolls(n, dy) {                            // can this inner box take the wheel itself?
    var o = (n.ownerDocument.defaultView || W).getComputedStyle(n).overflowY;
    if (o !== 'auto' && o !== 'scroll') return false;
    return dy > 0 ? n.scrollTop + n.clientHeight < n.scrollHeight - 1 : n.scrollTop > 0;
  }
  function stop() { aim = put = null; run = 0; vel = 0; last = 0; }
  // A button's glide is flying: main.py's glide script keeps its stop handle
  // on this document (its P). The old check read the window and never saw it.
  function gliding() { return !!(D.__melionaStop || W.__melionaStop); }
  function coast(now) {
    var sc = D.querySelector(SCROLLER);
    // someone else moved the page (a button's glide, a key, the scrollbar):
    // theirs wins, and this wheel's aim and way are dropped at once
    if (!sc || aim === null || gliding() || (put !== null && Math.abs(sc.scrollTop - put) > 2)) { stop(); return; }
    var dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60, max = sc.scrollHeight - sc.clientHeight;
    last = now;
    for (var i = 0; i < 4; i++) {                      // the tow line's pull, less the water's drag
      var h = dt / 4, acc = TOW * TOW * (aim - pos) - 2 * TOW * vel;
      vel += acc * h; pos += vel * h;
    }
    if (pos < 0 || pos > max) { pos = Math.max(0, Math.min(pos, max)); vel = 0; }   // the page's ends are the quay
    if (Math.abs(aim - pos) < 0.5 && Math.abs(vel) < 8) { pos = aim; vel = 0; }
    sc.scrollTop = pos;
    put = sc.scrollTop;
    if (pos === aim && vel === 0) { stop(); return; }
    run = W.requestAnimationFrame(coast);
  }
  // ---- the rubber band at the page's ends ---------------------------------
  // Taking the wheel took the browser's own elastic ends with it. Pushed past
  // the top or the bottom, the page now gives with resistance - it follows
  // the hand at once (never tweened), less and less the further it is pushed,
  // to at most BAND px - and eases back when the hand lets go, on a
  // critically damped spring: no overshoot, no bounce.
  var BAND = 72;          // the furthest the page gives past an end (px)
  var RETURN = 16;        // how briskly it comes back (rad/s): ~0.35 s
  var BODY = '[data-testid="stMainBlockContainer"]';
  var give = 0, gv = 0, gRun = 0, gLast = 0, gAt = 0;
  function stretch(px) {                               // pull (raw px) -> give (px shown)
    var a = Math.abs(px);
    return (px < 0 ? -1 : 1) * BAND * (1 - 1 / (a / (BAND * 1.8) + 1));
  }
  function unstretch(g) {                              // give -> the pull that shows it
    var a = Math.min(Math.abs(g) / BAND, 0.999);
    return (g < 0 ? -1 : 1) * BAND * 1.8 * (1 / (1 - a) - 1);
  }
  function paintGive() {
    var b = D.querySelector(BODY);
    if (!b) return;
    b.style.transform = Math.abs(give) < 0.25 ? '' : 'translate3d(0,' + (-give).toFixed(2) + 'px,0)';
  }
  function settleGive(now) {
    gRun = 0;
    if (now - gAt < 90) { gRun = W.requestAnimationFrame(settleGive); return; }   // the hand is still on it
    var dt = gLast ? Math.min(0.05, (now - gLast) / 1000) : 1 / 60;
    gLast = now;
    for (var i = 0; i < 4; i++) { var h = dt / 4; gv += (-RETURN * RETURN * give - 2 * RETURN * gv) * h; give += gv * h; }
    if (Math.abs(give) < 0.25 && Math.abs(gv) < 4) { give = gv = 0; gLast = 0; paintGive(); return; }
    paintGive();
    gRun = W.requestAnimationFrame(settleGive);
  }
  function band(dy) {                                  // a push past the end, or back from it
    // from where the page stands now - mid-return included - so it never jumps
    var pull = (give ? unstretch(give) : 0) + dy;
    if (give && Math.sign(pull) !== Math.sign(give)) pull = 0;   // pulled back through the end
    give = stretch(pull); gv = 0; gLast = 0; gAt = performance.now();
    paintGive();
    if (!gRun) gRun = W.requestAnimationFrame(settleGive);
  }

  function onWheel(e) {
    if (e.ctrlKey || e.defaultPrevented || !e.deltaY || gliding()) return;   // zoom; a glide in flight (it reads the wheel itself)
    var sc = D.querySelector(SCROLLER), doc = e.target.ownerDocument || D;
    // A wheel inside a frame (the Risk Topology map) is the page's wheel at
    // the frame's place - once nothing inside the frame takes it: a panel
    // that scrolls, or the frame's own page on a phone.
    var at = doc === D ? e.target : doc.defaultView && doc.defaultView.frameElement;
    if (!sc || !at || !sc.contains(at)) return;
    if (doc !== D) {
      for (var m = e.target; m && m.nodeType === 1; m = m.parentElement) if (scrolls(m, e.deltaY)) return;
      var se = doc.scrollingElement;
      if (se && (e.deltaY > 0 ? se.scrollTop + se.clientHeight < se.scrollHeight - 1 : se.scrollTop > 0)) return;
    }
    for (var n = at; n && n !== sc; n = n.parentElement) if (scrolls(n, e.deltaY)) return;
    e.preventDefault();
    var dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaMode === 2 ? e.deltaY * sc.clientHeight : e.deltaY;
    var max = sc.scrollHeight - sc.clientHeight;
    // At rest on an end and pushed on past it - or already stretched: the band
    // takes the wheel (pulling back first undoes the stretch).
    var atTop = sc.scrollTop <= 0 && dy < 0, atEnd = sc.scrollTop >= max - 1 && dy > 0;
    if (give || (!run && (atTop || atEnd) && max > 0)) { band(dy); return; }
    aim = Math.max(0, Math.min((aim === null ? sc.scrollTop : aim) + dy, max));
    if (!run) { pos = put = sc.scrollTop; vel = 0; last = 0; sc.style.scrollBehavior = 'auto'; run = W.requestAnimationFrame(coast); }
  }
  W.addEventListener('wheel', onWheel, { passive: false });
  // The map is a same-origin frame, and a wheel over a frame goes to the
  // frame's own window, never this one: the page fell back to the browser's
  // step scroll there, the one stretch of the page that did not glide (John,
  // 2026-11-06). Each frame's window gets the same listener, again after
  // every load (Streamlit rewrites the map's srcdoc on a rerun, a new window).
  function hookFrame(f) {
    if (f.__tow) return;
    f.__tow = 1;
    var hook = function () {
      try {
        var fw = f.contentWindow;
        if (fw && !fw.__melionaTow) { fw.__melionaTow = 1; fw.addEventListener('wheel', onWheel, { passive: false }); }
      } catch (x) { /* a cross-origin frame keeps its own wheel */ }
    };
    f.addEventListener('load', hook);
    hook();
  }

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
    (root.tagName === 'IFRAME' ? [root] : [].slice.call(root.querySelectorAll('iframe'))).forEach(hookFrame);
    (root.matches(POP) ? [root] : [].slice.call(root.querySelectorAll(POP))).forEach(function (pop) {
      W.requestAnimationFrame(function () {
        aimPop(pop);
        W.requestAnimationFrame(function () { aimPop(pop); });
      });
      [60, 140, 260].forEach(function (ms) { setTimeout(function () { aimPop(pop); }, ms); });
    });
    var masks = root.matches(MASK) ? [root] : [].slice.call(root.querySelectorAll(MASK));
    masks.forEach(function (el) {
      if (el.__mk || el.closest(DEST) || (late && inView(el))) return;
      el.__mk = 1;
      el.classList.add('mask-reveal');
      el.parentElement.__mask = el;
      veiled.observe(el.parentElement);
    });
    var found = root.matches(ITEMS) ? [root] : [];
    found.push.apply(found, root.querySelectorAll(ITEMS));
    found.forEach(function (el) {
      if (el.__rv || el.closest(SKIP) || el.closest(OPENED)) return;
      el.__rv = 1;
      if (el.matches(DEST) || el.querySelector(DEST) || el.querySelector(SKIP)) return;
      // a container block arrives through its parts: its own leaves, or the
      // named blocks inside it (a row of cards keeps its card-by-card deal)
      if (!el.matches(NAMED) && (el.querySelector(LEAF) || el.querySelector(NAMED))) return;
      var parent = el.parentElement && el.parentElement.closest('.reveal-item');
      if (parent || (late && inView(el))) return;     // never animate inside an animating block
      el.classList.add('reveal-item');
      observer().observe(el);
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
