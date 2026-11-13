/* Meliona motion: how the page arrives as you scroll.
 *
 *   1. Reveals - every block on the page arrives the first time it comes on
 *      screen, where the reader can see it (John, 2026-11-13: scrolling down
 *      he saw no arrivals at all - they played during the load and below the
 *      screen's edge). Headings slide in from the left; on a wide screen a
 *      block beside another comes in from its own side; the rest rise. A
 *      fade carries each one; nothing turns or zooms. An
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
 *      for touch and keys, which keep the browser's own scrolling. The page's
 *      ends are hard: pushed past the top it stays put, no gap shows.
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
    '.about-text', '.about-find', '.story-text', '.faq-section > .kicker', '.faq-row',
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
  // An arrival starts once its block is FOOT inside the screen's bottom edge,
  // so the reader sees it move; one that sits on screen DWELL_MS without
  // getting there (a reader who stopped with it low) starts then, so nothing
  // waits on screen. It used to start 15% BELOW the edge: 0.6s later it had
  // landed, before the reader saw it (92% of blocks, John 2026-11-13).
  var FOOT = '-10%';
  var DWELL_MS = 250;
  var SLACK = '32px';     // at least app.css --reveal-y
  // Where each block comes from. Headings from the left. On a screen wide
  // enough to set things side by side, a block in the first of a row of
  // columns from the left, in the last from the right, the text beside
  // John's photographs from the right; everything else rises. A phone
  // stacks its columns, so there it only rises (and its headings still
  // come from the left, which can never open a sideways scroll).
  var HEAD = '.panel-head, .showcase-title, .faq-section > .kicker, [data-testid="stHeading"]';
  var BESIDE = '.about-text, .story-text';
  var wide = W.matchMedia ? W.matchMedia('(min-width: 761px)') : { matches: true };
  function way(el) {
    if (el.matches(HEAD) || el.querySelector(HEAD + ', h1, h2, h3')) return 'rv-l';
    if (!wide.matches) return '';
    if (el.matches(BESIDE)) return 'rv-r';
    var col = el.closest('[data-testid="stColumn"]');
    var row = col && col.parentElement && [].filter.call(col.parentElement.children, function (c) {
      return c.matches('[data-testid="stColumn"]');
    });
    if (!row || row.length < 2) return '';
    return col === row[0] ? 'rv-l' : col === row[row.length - 1] ? 'rv-r' : '';
  }

  var calm = !('IntersectionObserver' in W) ||
    (W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var api = W.__melionaReveal = { revealAll: function () {} };
  // iOS Safari only applies :active (the press) when the page listens for
  // touches; an empty passive listener is the standard switch.
  D.addEventListener('touchstart', function () {}, { passive: true });
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
  // Optimistic update (John, 2026-11-09): a changed setting is answered at
  // once. Streamlit shows its running sign only 500ms into a run, so a
  // reader's change looked ignored for half a second; html.m-busy lights the
  // same rail and "Updating..." tag (app.css) the moment the change is made,
  // and drops it when the run ends - or at once if no run follows.
  var busyT = 0;
  function busy() {
    var t0 = Date.now(), seen = false;
    H.classList.add('m-busy');
    clearInterval(busyT);
    busyT = setInterval(function () {
      var app = D.querySelector('.stApp');
      var on = !!D.querySelector('[data-testid="stStatusWidgetRunningIcon"]') ||
               (app && app.getAttribute('data-test-script-state') === 'running');
      if (on) seen = true;
      if ((seen && !on) || (!seen && Date.now() - t0 > 900) || Date.now() - t0 > 30000) {
        H.classList.remove('m-busy'); clearInterval(busyT);
      }
    }, 80);
  }
  D.addEventListener('change', function (e) {
    if (e.target.closest && e.target.closest('[data-testid="stMain"]')) busy();
  }, true);
  // react-aria's radios, toggles and segmented buttons change on a press and
  // fire no native 'change', so a press on a control that is not already
  // the chosen one counts too (and a slider when it is let go).
  var CONTROL = '[role="radiogroup"] label, [data-testid="stCheckbox"] label, [data-testid="stToggle"] label, ' +
    '[role="option"], .stButton>button, .stFormSubmitButton>button, [data-testid="stButtonGroup"] button, ' +
    '[data-testid="stSegmentedControl"] button, [data-testid="stPills"] button';
  D.addEventListener('click', function (e) {
    var c = e.target.closest && e.target.closest(CONTROL);
    if (!c || c.querySelector('input[type="radio"]:checked') || c.getAttribute('aria-selected') === 'true' ||
        c.getAttribute('aria-checked') === 'true' || c.getAttribute('aria-pressed') === 'true') return;
    if (c.closest('[data-testid="stMain"], [data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectDropdown"]')) busy();
  }, true);
  D.addEventListener('pointerup', function (e) {
    if (e.target.closest && e.target.closest('[data-testid="stSlider"]')) busy();
  }, true);
  if (calm) { H.classList.remove('reveal-on'); return; }   // everything as drawn

  // ---- 1. reveals ----------------------------------------------------------
  // The first load: Streamlit writes the blocks below the hero BEFORE the
  // hero's slot is filled, so for a moment they sit on a short first screen
  // (under the boot skeleton) and their arrival played there; the hero then
  // pushed them far down, already landed (15 of 22, John 2026-11-13). So no
  // arrival plays until the first run has ended - or the reader moves, or
  // 12s pass - and then whatever is on screen arrives.
  var ready = false;
  function release() {
    if (ready) return;
    ready = true;
    W.requestAnimationFrame(function () { W.requestAnimationFrame(sweep); });
  }
  function sweep() {
    var sc = D.querySelector(SCROLLER), bottom = sc ? sc.getBoundingClientRect().bottom : W.innerHeight;
    var on = [].filter.call(D.querySelectorAll('.reveal-item:not(.is-visible)'), function (el) {
      var r = el.getBoundingClientRect();
      return r.height > 0 && r.top < bottom && r.bottom > 0;
    });
    on.forEach(function (el, i) { show(el, Math.min(i, MAX_STEPS) * STAGGER_MS); });
  }
  new MutationObserver(function () {
    var app = D.querySelector('.stApp');
    if (app && app.getAttribute('data-test-script-state') === 'notRunning') release();
  }).observe(D.documentElement, { attributes: true, subtree: true, attributeFilter: ['data-test-script-state'] });
  ['wheel', 'touchstart', 'keydown', 'pointerdown'].forEach(function (t) {
    D.addEventListener(t, release, { capture: true, passive: true });
  });
  setTimeout(release, 12000);

  function arrive(entries) {
    if (!ready) return;
    entries.filter(function (e) { return e.isIntersecting && !e.target.classList.contains('is-visible'); })
      // reading order = document order: a row of metrics left to right, a
      // heading before its line. (Their boxes are still mid-transform, so
      // on-screen positions would mis-sort them.)
      .sort(function (a, b) {
        return a.target.compareDocumentPosition(b.target) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
      })
      .forEach(function (e, i) { show(e.target, Math.min(i, MAX_STEPS) * STAGGER_MS); });
  }
  // A block on screen that has not reached FOOT (the reader stopped with it
  // low) starts after DWELL_MS: John, 2026-11-06, saw a heading, a card and
  // the contact form's button missing at the screen's foot for up to 3s.
  function linger(entries) {
    entries.forEach(function (e) {
      var el = e.target;
      clearTimeout(el.__dw);
      if (e.isIntersecting) el.__dw = setTimeout(function () {
        if (ready && el.classList.contains('reveal-item') && !el.classList.contains('is-visible')) show(el, 0);
      }, DWELL_MS);
    });
  }
  // The root is the page's own scroller (stMain), not the window: a margin
  // only changes the root's box, and stMain clips everything below it first.
  // Streamlit draws stMain after this file runs: until then the window
  // stands in, and when stMain comes (or is ever redrawn) the blocks still
  // waiting move to fresh observers.
  var seen = null, edge = null, seenRoot;
  function observer() {
    var sc = D.querySelector(SCROLLER);
    if (seen && seenRoot === sc) return seen;
    var old = seen, oldEdge = edge;
    seenRoot = sc;
    seen = new IntersectionObserver(arrive, { root: sc, rootMargin: '0px 0px ' + FOOT + ' 0px', threshold: 0 });
    // a rising block waits --reveal-y (28px) below its place, and an observer
    // reads that shifted box: the margin lets its place count, not its box
    edge = new IntersectionObserver(linger, { root: sc, rootMargin: '0px 0px ' + SLACK + ' 0px', threshold: 0 });
    if (old) {
      old.disconnect(); oldEdge.disconnect();
      [].forEach.call(D.querySelectorAll('.reveal-item:not(.is-visible)'), watch);
    }
    return seen;
  }
  function watch(el) { observer().observe(el); edge.observe(el); }
  function unwatch(el) { observer().unobserve(el); edge.unobserve(el); clearTimeout(el.__dw); }

  function show(el, delay) {
    unwatch(el);
    void el.offsetWidth;                               // starts from its waiting state
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
  }
  // Landed: hand the block back to its own styles (hover lifts, transitions)
  // and drop will-change, so the GPU layer it held is released.
  function done(el) {
    clearTimeout(el.__rvTimer);
    el.classList.remove('reveal-item', 'is-visible', 'rv-l', 'rv-r');
    el.style.removeProperty('--reveal-delay');
  }
  api.revealAll = function () {                       // the end state, at once (layout probes)
    landCounts();
    [].forEach.call(D.querySelectorAll('.reveal-item'), function (el) { unwatch(el); done(el); });
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

  // ---- origin-aware hover ---------------------------------------------------
  // A button's hover colour opens from the point the pointer came in at, and
  // closes toward where it left (app.css: the ::after circle at --mx/--my).
  var HOVERS = '.stButton>button, .stFormSubmitButton>button, .stLinkButton>a, .cta-btn';
  function aimHover(e) {
    var el = e.target.closest && e.target.closest(HOVERS);
    if (!el || (e.relatedTarget && el.contains(e.relatedTarget))) return;   // moving inside it
    var r = el.getBoundingClientRect();
    if (!r.width || !r.height) return;
    el.style.setProperty('--mx', Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)).toFixed(1) + '%');
    el.style.setProperty('--my', Math.max(0, Math.min(100, (e.clientY - r.top) / r.height * 100)).toFixed(1) + '%');
  }
  D.addEventListener('pointerover', aimHover, { capture: true, passive: true });
  D.addEventListener('pointerout', aimHover, { capture: true, passive: true });

  // ---- count-up: a headline number counts to its value as it arrives -------
  // Only the page's own HTML numbers (the verdict, the hero's receipt), never
  // a Streamlit-owned text node. It counts from the last value that place
  // showed (0 the first time) and always ends on the server's exact text,
  // kept in data-count-final; revealAll() lands every count at once.
  var COUNT = '.verdict-number, .hstat .n', NUM = /^(\D*?)([-\u2212]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\D*)$/;
  var lastNum = {}, counts = [];
  function countUp(el) {
    var final = el.textContent.trim(), m = NUM.exec(final);
    el.setAttribute('data-count-final', final);
    if (!m) return;
    var dec = m[4] ? m[4].length - 1 : 0, sign = m[2] ? -1 : 1, commas = m[3].indexOf(',') >= 0;
    var to = sign * parseFloat(m[3].replace(/,/g, '') + (m[4] || ''));
    var key = el.className + '|' + [].indexOf.call(D.querySelectorAll(COUNT), el);
    var from = key in lastNum ? lastNum[key] : 0;
    lastNum[key] = to;
    if (from === to) return;
    var fmt = function (v) {
      var s = Math.abs(v).toFixed(dec), i = s.split('.');
      if (commas) i[0] = i[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      return m[1] + (v < 0 ? (m[2] || '-') : '') + i.join('.') + m[5];
    };
    var c = { el: el, final: final, t0: 0 };
    counts.push(c);
    el.textContent = fmt(from);                          // starts where it counts from, even under the loading screen
    (function step(now) {
      if (c.done) return;
      if (D.getElementById('boot-skel') && getComputedStyle(D.getElementById('boot-skel')).display !== 'none' &&
          getComputedStyle(D.getElementById('boot-skel')).opacity > 0.05) { W.requestAnimationFrame(step); return; }
      if (!c.t0) c.t0 = now;
      var p = Math.min(1, (now - c.t0) / 900), e = 1 - Math.pow(1 - p, 3);
      if (!el.isConnected) { c.done = true; return; }       // Streamlit redrew it: the new node counts on its own
      el.textContent = p < 1 ? fmt(from + (to - from) * e) : final;
      if (p < 1) W.requestAnimationFrame(step); else c.done = true;
    })(performance.now());
  }
  function landCounts() { counts.forEach(function (c) { if (!c.done) { c.done = true; if (c.el.isConnected) c.el.textContent = c.final; } }); }
  var counting = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) { counting.unobserve(e.target); countUp(e.target); } });
  }, { threshold: 0.6 });

  // ---- 3. glide ----------------------------------------------------------------
  // A boat on a tow line (John, 2026-11-04: "heavier ... like a boat being
  // tugged", not static). The wheel moves the tug - where the page is going
  // (aim). The page is the hull: a mass pulled toward the aim by a critically
  // damped spring, so it gathers way instead of jumping, keeps its momentum
  // through the next notch of the wheel, and comes to rest on the aim without
  // overshooting or bouncing back. Integrated on the frame's own clock, so a
  // fast and a slow screen move the hull alike.
  var TOW = 10;           // the tow line's natural frequency (rad/s): lower = heavier hull; ~0.6 s to settle
                          // (John, 2026-11-09: 3.8 drifted ~1.7 s after the wheel stopped and made him
                          // motion sick; smooth, but done by the time the eye looks for the line)
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
      // the frame's own page takes the wheel only if it can scroll: on a
      // desktop the map's body is overflow: hidden, and a hover tip past its
      // bottom made it look taller - the browser then stepped the whole page
      // 100px in one frame (Live audit 8b, 2026-11-12)
      var se = doc.scrollingElement, fv = doc.defaultView;
      var held = fv && doc.body && /hidden|clip/.test(fv.getComputedStyle(doc.body).overflowY + fv.getComputedStyle(doc.documentElement).overflowY);
      if (se && !held && (e.deltaY > 0 ? se.scrollTop + se.clientHeight < se.scrollHeight - 1 : se.scrollTop > 0)) return;
    }
    for (var n = at; n && n !== sc; n = n.parentElement) if (scrolls(n, e.deltaY)) return;
    e.preventDefault();
    var dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaMode === 2 ? e.deltaY * sc.clientHeight : e.deltaY;
    var max = sc.scrollHeight - sc.clientHeight;
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
    (root.matches(COUNT) ? [root] : [].slice.call(root.querySelectorAll(COUNT))).forEach(function (el) {
      if (el.__ct) return;
      el.__ct = 1;
      el.setAttribute('data-count-final', el.textContent.trim());
      counting.observe(el);
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
      var from = way(el);
      if (from) el.classList.add(from);
      el.classList.add('reveal-item');
      watch(el);
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
