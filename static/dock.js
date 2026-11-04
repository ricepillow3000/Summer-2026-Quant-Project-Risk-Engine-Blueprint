/* Meliona dock: the two things that sit at the bottom of the screen.
 *
 *   1. The phone shortcut (John, 2026-11-04). On a phone (640px and under) a
 *      slim charcoal bar with one button, "Run the risk engine", appears once
 *      the hero's own button has scrolled off the top, and goes again as soon
 *      as the engine (#engine) or anything below it is on screen - so it never
 *      sits over the verdict, the Book or the tabs. It is an in-page link, so
 *      main.py's glide flies it like every other button: no second scroll
 *      routine lives here. Never on a wider screen.
 *   2. The cookie notice. Informational, not a consent wall: the site's only
 *      cookie is Streamlit's anti-forgery _streamlit_xsrf (strictly necessary)
 *      and nothing tracks. Shown until "OK" is pressed; the press is kept in
 *      localStorage under KEY. If storage is blocked the notice just shows
 *      again next visit. No JavaScript, no notice: it is a convenience.
 *
 * Both live in ONE fixed stack, the notice above the bar, so they can never
 * overlap each other. The stack is added to <body>, outside Streamlit's tree:
 * it adds no element to the script's run, so nothing above the tab strip
 * moves (CLAUDE.md, the position rule). Two observers decide when the bar
 * shows - nothing listens to every scrolled pixel. Layout probes press OK
 * through window.__melionaNotice.dismiss(), as they land reveals through
 * window.__melionaReveal.revealAll(). Styles: static/app.css, "DOCK".
 */
(function () {
  'use strict';
  var W = window, D = document;
  if (W.__melionaNotice || !D.body) return;          // one copy per page

  var KEY = 'meliona-cookie-notice';
  var PRIVACY = 'https://github.com/ricepillow3000/Summer-2026-Quant-Project-Risk-Engine-Blueprint/blob/main/PRIVACY.md';
  var PHONE = W.matchMedia('(max-width: 640px)');
  var HERO_CTA = '#hero .cta-btn', ENGINE = 'engine';

  var stack = D.createElement('div');
  stack.className = 'm-stack';

  // ---- 2. the notice --------------------------------------------------------
  var said = false;
  try { said = W.localStorage.getItem(KEY) === 'ok'; } catch (e) { /* blocked: show it */ }
  var notice = null;
  if (!said) {
    notice = D.createElement('section');
    notice.className = 'm-notice';
    notice.setAttribute('aria-label', 'Cookie notice');
    notice.innerHTML = '<p>This site uses one cookie, needed for security, and no tracking. ' +
      'Details in the <a href="' + PRIVACY + '">privacy policy</a>.</p>' +
      '<button type="button" class="m-notice-ok">OK</button>';
    notice.querySelector('button').addEventListener('click', dismiss);
    stack.appendChild(notice);
  }
  function dismiss() {
    try { W.localStorage.setItem(KEY, 'ok'); } catch (e) { /* blocked: it shows again next visit */ }
    if (notice) { notice.remove(); notice = null; }
  }

  // ---- 1. the phone shortcut ------------------------------------------------
  var dock = D.createElement('nav');
  dock.className = 'm-dock';
  dock.setAttribute('aria-label', 'Shortcut');
  dock.hidden = true;
  dock.innerHTML = '<a class="m-dock-btn" href="#' + ENGINE + '" ' +
    'aria-label="Run the risk engine: go to the engine section">Run the risk engine</a>';
  stack.appendChild(dock);
  // First in the body, so the keyboard meets the notice and the shortcut at
  // the top of the Tab order, beside Streamlit's own skip link - not after
  // every control on the page.
  D.body.insertBefore(stack, D.body.firstChild);

  var heroCta = null, engine = null;
  var heroGone = false;     // the hero's button has left by the TOP of the screen
  var engineAhead = true;   // #engine is still below the screen (or not written yet)
  function update() {
    var show = PHONE.matches && heroGone && engineAhead;
    if (dock.hidden === !show) return;
    dock.hidden = !show;
    stack.classList.toggle('has-dock', show);
  }
  // No observer (a very old browser): the bar stays down, the notice still works.
  var io = !('IntersectionObserver' in W) ? null : new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var r = e.boundingClientRect, vh = e.rootBounds ? e.rootBounds.bottom : W.innerHeight;
      if (e.target === heroCta) heroGone = !e.isIntersecting && r.bottom <= 0;
      if (e.target === engine) engineAhead = !e.isIntersecting && r.top >= vh;
    });
    update();
  });
  // Streamlit writes the page after this runs and may redraw a block on a
  // rerun: whenever the page changes, follow the CURRENT hero button and
  // engine heading (two lookups, once a frame at most).
  function bind() {
    frame = 0;
    if (!io) return;
    var h = D.querySelector(HERO_CTA), g = D.getElementById(ENGINE);
    if (h !== heroCta) {
      if (heroCta) io.unobserve(heroCta);
      heroCta = h; heroGone = false;
      if (h) io.observe(h);
    }
    if (g !== engine) {
      if (engine) io.unobserve(engine);
      engine = g; engineAhead = true;
      if (g) io.observe(g);
    }
    update();
  }
  var frame = 0;
  new MutationObserver(function () {
    if (!frame) frame = W.requestAnimationFrame(bind);
  }).observe(D.body, { childList: true, subtree: true });
  if (PHONE.addEventListener) PHONE.addEventListener('change', update);
  bind();

  W.__melionaNotice = {
    dismiss: dismiss,                                  // press OK (layout probes)
    shown: function () { return !!notice; },
    dock: function () { return !dock.hidden; }         // is the phone shortcut up?
  };
})();
