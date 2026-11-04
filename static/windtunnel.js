/* Meliona wind tunnel: smoke blown through the cone of simulated outcomes.
 *
 * John, 2026-11-04: the cumulative-return cone should look like it sits in a
 * wind tunnel - smoke blown through it, many colours, fully interactive - and
 * the colours "must not be random: they have to correlate with the engine and
 * the cumulative return, along with its trading days".
 *
 * What is drawn, and why it is honest:
 *   - Each smoke particle IS a percentile level u between 5% and 95%, never a
 *     free point. At trading day x it sits exactly where the simulation put
 *     that level: the computed 5th/25th/50th/75th/95th percentile paths are
 *     read from the chart itself (Plotly's _fullData), and a level between two
 *     of them is interpolated between those two. So the smoke can never leave
 *     the cone, and its height on any day is a return the engine produced.
 *   - Its colour is the cumulative return at that height, from the ONE scale
 *     main.py's RETURN_SCALE defines and ships in the figure's layout.meta -
 *     the same scale the band fills use and the key under the chart shows.
 *   - Particles are seeded evenly across the 5-95% levels, so the smoke is as
 *     dense in each slice of probability as in any other.
 *   - The wind blows along the trading days (a year in about 3.5 s). Eddies
 *     only move a particle between percentile levels, so a wisp that curls is
 *     still always on the cone.
 *
 * Interactive: the pointer is an obstacle the smoke parts around (in level
 * space, so it still never leaves the cone); a click sends a gust through.
 * Plotly's own hover readout keeps working underneath - the canvas takes no
 * pointer events.
 *
 * Cost: one canvas over the plot area, drawn only while the chart is on
 * screen and the tab is visible; nothing runs under reduced motion.
 * window.__melionaWind.check() re-derives every particle's return from its
 * drawn pixel (through Plotly's own axis) and compares the colour it was given.
 */
(function () {
  'use strict';
  var W = window, D = document;
  if (W.__melionaWind) return;
  var api = W.__melionaWind = { tunnels: [], check: function () { return null; } };
  if (W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // ---- tuning -------------------------------------------------------------
  var PER_PX = 1 / 170;      // particles per square px of plot area
  var MIN_P = 320, MAX_P = 1500;
  var WIND = 72;             // trading days per second the smoke travels
  var GUST = 0.32;           // +/- share of WIND one particle's own speed varies
  var TRAIL = 22;            // positions kept per particle: the length of a wisp
  var PUFFS = 70;            // soft haze puffs riding the same levels
  var POINTER_R = 70;        // px: how far the pointer parts the smoke
  var LO = 0.05, HI = 0.95;  // the cone's computed range
  var BUCKETS = 120;         // colour steps across the scale (about 1% of return each)
  // two eddy modes (amplitude a in level-days, wavenumbers along days and
  // across levels, drift speed w): a slow roll and a quicker curl on top
  // (a slow roll carries a particle about 0.10 of a level per second, the
  // curl about 0.05; along the days they add at most ~18 and ~9 days/s)
  var MODES = [{ a: 2.6, kx: 0.035, ks: 2 * Math.PI, w: 0.9, ph: 0.0 },
               { a: 0.5, kx: 0.090, ks: 5 * Math.PI, w: 1.7, ph: 2.1 }];
  var LEVELS = [[0.05, 'p5'], [0.25, 'p25'], [0.50, 'median'], [0.75, 'p75'], [0.95, 'p95']];

  // ---- the scale: return (%) -> colour, from the figure --------------------
  function hex(c) { return [1, 3, 5].map(function (i) { return parseInt(c.slice(i, i + 2), 16); }); }
  function makeScale(knots) {
    var k = knots.map(function (s) { return { r: +s[0], c: hex(s[1]) }; });
    var lo = k[0].r, hi = k[k.length - 1].r;
    function rgbAt(ret) {
      if (ret <= lo) return k[0].c;
      if (ret >= hi) return k[k.length - 1].c;
      for (var i = 1; i < k.length; i++) if (ret <= k[i].r) {
        var a = k[i - 1], b = k[i], f = (ret - a.r) / (b.r - a.r);
        return [0, 1, 2].map(function (j) { return a.c[j] + (b.c[j] - a.c[j]) * f; });
      }
      return k[k.length - 1].c;
    }
    var css = [];
    for (var i = 0; i < BUCKETS; i++) {
      var c = rgbAt(lo + (i + 0.5) / BUCKETS * (hi - lo));
      css.push('rgb(' + c.map(Math.round).join(',') + ')');
    }
    function bucket(ret) {
      return Math.max(0, Math.min(BUCKETS - 1, Math.floor((ret - lo) / (hi - lo) * BUCKETS)));
    }
    return { rgbAt: rgbAt, css: css, bucket: bucket, lo: lo, hi: hi };
  }

  // ---- one tunnel per cone on the page ---------------------------------------
  function Tunnel(gd) {
    var T = this;
    T.gd = gd; gd.__wind = T;
    T.cv = D.createElement('canvas');
    T.cv.className = 'wind-tunnel';
    T.cv.setAttribute('aria-hidden', 'true');
    T.cv.style.cssText = 'position:absolute;pointer-events:none;';   // page order puts it under the hover layer
    T.ctx = T.cv.getContext('2d');
    T.parts = []; T.puffs = []; T.ptr = null; T.gust = 0; T.on = false; T.last = 0; T.raf = 0; T.t = 0;
    T.read();
    gd.on && gd.on('plotly_afterplot', function () { T.read(); });
    gd.addEventListener('mousemove', function (e) {
      var b = T.cv.getBoundingClientRect();
      T.ptr = { x: e.clientX - b.left, y: e.clientY - b.top };
    });
    gd.addEventListener('mouseleave', function () { T.ptr = null; });
    gd.addEventListener('pointerdown', function () { T.gust = 1; });
    T.io = new IntersectionObserver(function (es) {
      T.visible = es[0].isIntersecting; T.wake();
    }, { threshold: 0.05 });
    T.io.observe(gd);
    api.tunnels.push(T);
  }

  // Read the cone and the axes from the chart itself, after every redraw: a
  // new setting re-renders the figure, and the smoke follows the new cone.
  Tunnel.prototype.read = function () {
    var gd = this.gd, fl = gd._fullLayout, fd = gd._fullData, meta = metaOf(gd);
    if (!fl || !fd || !fl.xaxis || !fl.yaxis || !meta || !meta.scale) { this.ok = false; return; }
    var q = {};
    for (var i = 0; i < fd.length; i++) if (fd[i].name && !(fd[i].name in q)) q[fd[i].name] = fd[i];
    var days = q.median && Array.from(q.median.x || []);
    var lv = LEVELS.map(function (L) { return q[L[1]] ? Array.from(q[L[1]].y || []) : null; });
    if (!days || days.length < 2 || lv.some(function (a) { return !a || a.length !== days.length; })) { this.ok = false; return; }
    this.days = days; this.lv = lv;
    this.scale = makeScale(meta.scale);
    this.xa = fl.xaxis; this.ya = fl.yaxis; this.sz = fl._size;
    // the canvas covers the plot area only, above the fills, below the hover layer
    var box = gd.querySelector('.svg-container');
    if (box && this.cv.parentNode !== box) {
      var svgs = box.querySelectorAll('svg.main-svg');
      box.insertBefore(this.cv, svgs[1] || null);
    }
    var dpr = Math.min(2, W.devicePixelRatio || 1), s = this.sz;
    this.cv.style.left = s.l + 'px'; this.cv.style.top = s.t + 'px';
    this.cv.style.width = s.w + 'px'; this.cv.style.height = s.h + 'px';
    if (this.cv.width !== Math.round(s.w * dpr) || this.cv.height !== Math.round(s.h * dpr)) {
      this.cv.width = Math.round(s.w * dpr); this.cv.height = Math.round(s.h * dpr);
    }
    this.dpr = dpr;
    this.ok = s.w > 40 && s.h > 40;
    var want = Math.max(MIN_P, Math.min(MAX_P, Math.round(s.w * s.h * PER_PX)));
    while (this.parts.length < want) this.parts.push(this.spawn(true));
    this.parts.length = want;
    while (this.puffs.length < PUFFS) this.puffs.push(this.spawn(true, true));
    this.wake();
  };

  // A new particle enters at the tunnel mouth (or, the first time, anywhere
  // to the left of it, so the first gust sweeps through as a front).
  Tunnel.prototype.spawn = function (first, puff) {
    var d0 = this.days ? this.days[0] : 1, span = this.days ? this.days[this.days.length - 1] - d0 : 250;
    return {
      u: LO + Math.random() * (HI - LO),
      x: d0 - (first ? Math.random() * span * 1.1 : Math.random() * span * 0.08),
      v: 1 + (Math.random() * 2 - 1) * GUST,
      ph: Math.random() * 1000,
      tr: [], puff: !!puff, r: puff ? 14 + Math.random() * 30 : 0
    };
  };

  // The return (%) of percentile level u on trading day x: linear between the
  // two computed days, then linear between the two computed levels around u.
  Tunnel.prototype.valueAt = function (x, u) {
    var d = this.days, n = d.length, i = 1;
    var fx = (x - d[0]) / (d[n - 1] - d[0]) * (n - 1);
    var j = Math.max(0, Math.min(n - 2, Math.floor(fx))), f = Math.max(0, Math.min(1, fx - j));
    while (i < LEVELS.length - 1 && u > LEVELS[i][0]) i++;
    var a = LEVELS[i - 1][0], b = LEVELS[i][0], g = (u - a) / (b - a);
    var va = this.lv[i - 1][j] + (this.lv[i - 1][j + 1] - this.lv[i - 1][j]) * f;
    var vb = this.lv[i][j] + (this.lv[i][j + 1] - this.lv[i][j]) * f;
    return va + (vb - va) * g;
  };
  Tunnel.prototype.px = function (x) { return this.xa.l2p(x); };
  Tunnel.prototype.py = function (v) { return this.ya.l2p(v); };

  Tunnel.prototype.wake = function () {
    var run = this.ok && this.visible && !D.hidden && this.gd.isConnected;
    if (run && !this.raf) { var T = this; this.last = 0; this.raf = W.requestAnimationFrame(function (t) { T.frame(t); }); }
    if (!run && this.raf) { W.cancelAnimationFrame(this.raf); this.raf = 0; }
  };

  Tunnel.prototype.frame = function (now) {
    var T = this;
    T.raf = 0;
    if (!T.gd.isConnected) { T.io.disconnect(); api.tunnels.splice(api.tunnels.indexOf(T), 1); return; }
    var dt = T.last ? Math.min(0.05, (now - T.last) / 1000) : 0.016;
    T.last = now; T.t += dt;
    T.step(dt); T.draw();
    T.wake();
  };

  Tunnel.prototype.step = function (dt) {
    var T = this, d = T.days, end = d[d.length - 1], t = T.t, ptr = T.ptr;
    var boost = 1 + 2.2 * T.gust;
    T.gust = Math.max(0, T.gust - dt * 1.4);
    var all = T.parts.concat(T.puffs);
    for (var k = 0; k < all.length; k++) {
      var p = all[k];
      // Eddies from a stream function psi(x, s, t) over (trading day, level):
      // the flow it makes is incompressible, so the smoke never bunches into
      // lanes - each slice of probability keeps its share of smoke - and psi
      // is flat along the 5% and 95% edges, so nothing is pushed past them.
      var s = (p.u - LO) / (HI - LO), du = 0, dxe = 0;
      for (var m = 0; m < MODES.length; m++) {
        var M = MODES[m], ph = M.kx * p.x - M.w * t + M.ph;
        var sinS = Math.sin(M.ks * s), cosS = Math.cos(M.ks * s);
        du += -M.a * M.kx * Math.cos(ph) * sinS / (HI - LO) * dt;     // -dpsi/dx, in levels
        dxe += M.a * M.ks * Math.sin(ph) * cosS / (HI - LO) * dt;     // +dpsi/dlevel, in days
      }
      var slow = 1;
      if (ptr && p.x >= d[0]) {
        var sx = T.px(p.x), sy = T.py(T.valueAt(p.x, p.u));
        var dx = sx - ptr.x, dy = sy - ptr.y, dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < POINTER_R) {
          var push = (1 - dist / POINTER_R);
          du += (dy < 0 ? 1 : -1) * push * push * 0.9 * dt;   // above the pointer: up a level
          slow = 1 - 0.55 * push;                              // the obstacle holds it back
        }
      }
      p.u += du;
      if (p.u < LO) p.u = LO + (LO - p.u);                 // only the pointer can reach an edge
      if (p.u > HI) p.u = HI - (p.u - HI);
      p.x += WIND * p.v * boost * slow * dt + dxe;
      if (p.x > end) {
        var n = T.spawn(false, p.puff);
        p.u = n.u; p.x = n.x; p.v = n.v; p.tr.length = 0;
      }
      if (!p.puff && p.x >= d[0]) {
        p.tr.push(p.x, p.u);
        if (p.tr.length > TRAIL * 2) p.tr.splice(0, 2);
      }
    }
  };

  Tunnel.prototype.draw = function () {
    var T = this, c = T.ctx, dpr = T.dpr, d0 = T.days[0], sc = T.scale;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, T.sz.w, T.sz.h);
    // everything is drawn inside the computed 5-95% cone and nowhere else:
    // the haze's soft edge is clipped at the cone's own outline
    var D5 = T.lv[0], D95 = T.lv[4], dd = T.days;
    c.save();
    c.beginPath();
    for (var e = 0; e < dd.length; e++) c[e ? 'lineTo' : 'moveTo'](T.px(dd[e]), T.py(D95[e]));
    for (var f = dd.length - 1; f >= 0; f--) c.lineTo(T.px(dd[f]), T.py(D5[f]));
    c.closePath();
    c.clip();
    // haze: soft puffs, the volume of the smoke
    for (var h = 0; h < T.puffs.length; h++) {
      var q = T.puffs[h];
      if (q.x < d0) continue;
      var v = T.valueAt(q.x, q.u), x = T.px(q.x), y = T.py(v), rgb = sc.rgbAt(v).map(Math.round).join(',');
      var g = c.createRadialGradient(x, y, 0, x, y, q.r);
      g.addColorStop(0, 'rgba(' + rgb + ',0.13)');
      g.addColorStop(1, 'rgba(' + rgb + ',0)');
      c.fillStyle = g;
      c.beginPath(); c.arc(x, y, q.r, 0, 6.2832); c.fill();
    }
    // wisps: each a short trail, fading toward its tail, grouped by colour so
    // a frame is a few hundred strokes, not thousands
    c.lineCap = 'round';
    var STEPS = 5;
    for (var s = 0; s < STEPS; s++) {
      var paths = {};
      for (var k = 0; k < T.parts.length; k++) {
        var p = T.parts[k], tr = p.tr, n = tr.length / 2;
        if (n < 2) continue;
        var from = Math.floor((n - 1) * s / STEPS), to = Math.floor((n - 1) * (s + 1) / STEPS);
        if (to <= from) continue;
        var vHead = T.valueAt(tr[2 * to], tr[2 * to + 1]);
        var b = sc.bucket(vHead), list = paths[b] || (paths[b] = []);
        list.push(p, from, to);
      }
      c.globalAlpha = 0.06 + 0.46 * Math.pow((s + 1) / STEPS, 1.8);
      c.lineWidth = 0.6 + 1.1 * (s + 1) / STEPS;
      for (var bk in paths) {
        var L = paths[bk];
        c.strokeStyle = sc.css[bk];
        c.beginPath();
        for (var m = 0; m < L.length; m += 3) {
          var pp = L[m].tr;
          for (var z = L[m + 1]; z <= L[m + 2]; z++) {
            var X = T.px(pp[2 * z]), Y = T.py(T.valueAt(pp[2 * z], pp[2 * z + 1]));
            if (z === L[m + 1]) c.moveTo(X, Y); else c.lineTo(X, Y);
          }
        }
        c.stroke();
      }
    }
    c.globalAlpha = 1;
    c.restore();
  };

  // Self-check for probes: every drawn head, its return read back from its
  // PIXEL through Plotly's own axis, the colour bucket that return belongs
  // to, against the bucket it was painted with; and whether it lies inside
  // the computed 5-95% band on its day.
  api.check = function () {
    var out = { particles: 0, colourMismatch: 0, outsideCone: 0, minRet: Infinity, maxRet: -Infinity,
                p5End: null, p95End: null };
    api.tunnels.forEach(function (T) {
      if (!T.ok) return;
      // the cone the smoke is riding right now (a probe compares it with the chart's)
      out.p5End = T.lv[0][T.lv[0].length - 1]; out.p95End = T.lv[4][T.lv[4].length - 1];
      T.parts.forEach(function (p) {
        var n = p.tr.length / 2;
        if (n < 2) return;
        var x = p.tr[2 * (n - 1)], u = p.tr[2 * (n - 1) + 1];
        var y = T.py(T.valueAt(x, u)), back = T.ya.p2l(y);
        var lo = T.valueAt(x, LO), hi = T.valueAt(x, HI);
        out.particles++;
        // one bucket of slack: a pixel round-trip can land a hair across a bucket edge
        if (Math.abs(T.scale.bucket(back) - T.scale.bucket(T.valueAt(x, u))) > 1) out.colourMismatch++;
        // Plotly rounds pixels, so a read-back may sit up to half a pixel off
        var halfPx = Math.abs(T.ya.p2l(0) - T.ya.p2l(1)) / 2;
        if (back < Math.min(lo, hi) - halfPx || back > Math.max(lo, hi) + halfPx) out.outsideCone++;
        out.minRet = Math.min(out.minRet, back); out.maxRet = Math.max(out.maxRet, back);
      });
    });
    return out;
  };

  function metaOf(gd) { return (gd._fullLayout && gd._fullLayout.meta) || (gd.layout && gd.layout.meta) || null; }

  // ---- finding the cone ----------------------------------------------------------
  function find() {
    var plots = D.querySelectorAll('.js-plotly-plot');
    for (var i = 0; i < plots.length; i++) {
      var gd = plots[i], m = metaOf(gd);
      if (gd.__wind) { if (!gd.__wind.cv.isConnected) gd.__wind.read(); continue; }
      if (m && m.meliona === 'cone') new Tunnel(gd);
    }
  }
  var queued = 0;
  new MutationObserver(function () {
    if (!queued) queued = W.requestAnimationFrame(function () { queued = 0; find(); });
  }).observe(D.body, { childList: true, subtree: true });
  D.addEventListener('visibilitychange', function () { api.tunnels.forEach(function (T) { T.wake(); }); });
  W.addEventListener('resize', function () { api.tunnels.forEach(function (T) { T.read(); }); });
  find();
})();
