/* Meliona wind tunnel: a lattice Boltzmann flow around the cone's computed
 * percentile paths, drawn the way FluidX3D draws a flow.
 *
 * John, 2026-11-04: "Do not generate a static or generic looping animation
 * that mimics a fluid ... Every line, color, velocity vector, and animation
 * timestep must be directly driven by and deterministically bound to the
 * provided data array ... an exact, high-performance scientific CFD
 * calculation mirroring the precise engineering look of FluidX3D." No glass,
 * no gloss, no glow, no blur: a scientific render.
 *
 * The calculation (FluidX3D's own method, github.com/ProjectPhysX/FluidX3D):
 *   - D2Q9 lattice Boltzmann, BGK collision, on a grid laid over the plot
 *     area. Nine distribution functions per cell, streamed (pull scheme) and
 *     relaxed toward equilibrium with relaxation time TAU each step; density
 *     and velocity are their moments. No noise, no random numbers, no timers:
 *     the state after step n is a pure function of the data and n.
 *   - The DATA are the walls. The chart's five computed percentile paths
 *     (5th, 25th, median, 75th, 95th, read from Plotly's _fullData, the same
 *     arrays the hover prints) are rasterised into the grid as solid,
 *     non-permeable cells; the flow bounces back off them (halfway
 *     bounce-back) and has to compress, shear and turn around their shape.
 *     Air enters on the left at the inflow speed and leaves on the right;
 *     the domain's top and bottom rows are free stream (held at the inflow,
 *     the far-field boundary of an open test section), so the air can pass
 *     above and below the cone and is squeezed around it.
 *   - The CLOCK is the data's. One solver step moves inflow-speed air U_IN of
 *     a cell; the plot spans the cone's trading days, so
 *     STEPS_PER_DAY = columns / (U_IN x trading days): air at the inflow speed
 *     crosses the plot in exactly one simulated year. The day being replayed
 *     is step / STEPS_PER_DAY, and the inflow on that day is
 *     U_IN x (1 - U_SWING + U_SWING x w), where w is how fast the computed
 *     5-95% band widens that day relative to its fastest day: a calm year
 *     blows steady, a shocked one gusts where its cone opens fastest. Each
 *     displayed frame advances a fixed STEPS_PER_FRAME; how fast the screen
 *     refreshes never changes what is computed.
 *
 * The picture (FluidX3D's look, on Meliona's page instead of black):
 *   - each cell's colour is a fixed, stepped (not blended) function of its
 *     simulated speed relative to the inflow: at rest fully transparent, so
 *     the page shows through; then charcoal, deep blue, blue, teal. The steps
 *     come from main.py (FLOW_SPEED_STEPS) in the figure's layout.meta, the
 *     same table the key under the chart prints;
 *   - a wall cell's colour is the surface pressure the flow puts on it, the
 *     pressure coefficient Cp = (p - p_inflow) / (rho U^2 / 2), as FluidX3D
 *     colours a boundary by the force on it: charcoal, then crimson, then
 *     vivid red at the last step. Red therefore exists ONLY on the data
 *     paths, exactly where the flow collides with them;
 *   - streamlines (FluidX3D's graphics_streamline): seeded at the data's own
 *     nodes - every SEED_DAYS trading days along each percentile path, just
 *     either side of it - and at the inflow, integrated one cell a step along
 *     the computed velocity, stopping at a wall or where the flow is still;
 *     one-pixel lines in the same speed steps;
 *   - a faint mesh of the lattice's cells behind it all, and a 1px wireframe
 *     box: the simulation domain.
 *
 * "Verification Overlay" (a button on the chart): arrows of the computed
 * velocity at grid intersections with the speed printed beside them, the
 * highest-pressure wall cells marked with their Cp, the fastest cell marked
 * with its speed, and the solver's state (step, trading day, inflow, tau,
 * Reynolds number). window.__melionaWind.check() proves the same on the
 * rendered cells: every cell's colour against its own computed value, no red
 * off a wall, the walls on the data, and the peak pressure painted red.
 *
 * Performance: a graphics card runs it live (stopping off screen, in a hidden
 * tab, while the reader scrolls or a button's glide flies); under reduced
 * motion the same calculation is computed in small slices and shown as a
 * still frame. A software renderer (SwiftShader, llvmpipe) or no WebGL2: the
 * chart alone - an emulated GPU also draws the page, and the solve froze it.
 * tools/flow_check.mjs runs and proves the solve in CI.
 */
(function () {
  'use strict';
  var W = window, D = document;
  if (W.__melionaWind) return;
  var api = W.__melionaWind = { tunnels: [], check: function () { return null; }, renderer: null, overlay: false };
  var reduced = !!(W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ---- the solver's constants -------------------------------------------------
  var LBM = {
    COLS: 320, COLS_STILL: 130, // lattice cells across the plot
    TAU: 0.54,                  // BGK relaxation time: viscosity nu = (TAU - 0.5) / 3
    U_IN: 0.07,                 // inflow speed, lattice units (Mach 0.12: keeps the lattice's compressibility error small)
    U_SWING: 0.15,              // the inflow's share that follows the cone's widening on the replayed day
    STEPS_PER_FRAME: 6,         // solver steps per displayed frame (live)
    STILL_TRANSITS: 1.6,        // a still frame: the inflow crosses the plot this many times
    STILL_CHUNK: 10,            // ... at most this many steps per idle frame
    LINE_CELLS: 1.6,            // a percentile path's wall thickness, in cells
    SEED_DAYS: 5,               // a streamline seed every this many trading days along each path
    SEED_OFFSET: 2.5,           // ... this many cells off the wall, on either side
    SEED_ROWS: 3,               // ... and one every this many cells down the inflow
    H: 0.5,                     // streamline integration step, in cells (second-order, Heun)
    LEN: 520,                   // streamline length limit, in steps
    STREAM_EVERY: 6,            // live: retrace the streamlines every this many frames
    ARROW_CELLS: 16,            // Verification Overlay: an arrow every this many cells
    STILL_MS: 320,              // hold still this long after the reader's last scroll input
    PAD_IN: 24, PAD_OUT: 16     // computed but not drawn: a lead-in before day 1 and a run-out after the last day,
                                // so the inflow and the outlet sit away from the data (as in a real test section)
  };
  var LEVELS = ['p5', 'p25', 'median', 'p75', 'p95'];

  function hex(c) { return [1, 3, 5].map(function (i) { return parseInt(c.slice(i, i + 2), 16); }); }
  function metaOf(gd) { return (gd._fullLayout && gd._fullLayout.meta) || (gd.layout && gd.layout.meta) || null; }

  // ---- WebGL2 with float render targets, and whether it is a real GPU ----------
  var GL_PROBE = null;
  function glSupport() {
    if (GL_PROBE !== null) return GL_PROBE;
    GL_PROBE = false;
    try {
      var c = D.createElement('canvas'), gl = c.getContext('webgl2');
      if (gl && gl.getExtension('EXT_color_buffer_float')) {
        var ri = gl.getExtension('WEBGL_debug_renderer_info');
        var name = String(ri ? gl.getParameter(ri.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
        GL_PROBE = { soft: /swiftshader|llvmpipe|softpipe|software|basic render/i.test(name), name: name };
      }
      var lose = gl && gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    } catch (e) { GL_PROBE = false; }
    api.renderer = GL_PROBE ? GL_PROBE.name : 'none';
    return GL_PROBE;
  }

  // ---- shaders -------------------------------------------------------------------
  var VS = '#version 300 es\nin vec2 aPos;out vec2 vUv;void main(){vUv=aPos*.5+.5;gl_Position=vec4(aPos,0.,1.);}';
  // One D2Q9 step: pull-stream (bounce back off the walls), inflow on the
  // left and free stream on the top and bottom rows, a pressure outlet on the
  // right (reference density, the arriving flow's velocity), BGK
  // collision. Textures hold POST-collision distributions: A = f0..f3,
  // B = f4..f7, C = (f8, density, ux, uy). A wall cell stores, in C.y, the mean
  // density of its fluid neighbours (the pressure the flow puts on it).
  var LBM_FS = '#version 300 es\nprecision highp float;precision highp sampler2D;in vec2 vUv;' +
    'layout(location=0) out vec4 oA;layout(location=1) out vec4 oB;layout(location=2) out vec4 oC;' +
    'uniform sampler2D A,B,C,S;uniform vec2 texel;uniform float tau,uin,init;' +
    'const vec2 E[9]=vec2[9](vec2(0,0),vec2(1,0),vec2(0,1),vec2(-1,0),vec2(0,-1),vec2(1,1),vec2(-1,1),vec2(-1,-1),vec2(1,-1));' +
    'const float WT[9]=float[9](4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.);' +
    'const int OPP[9]=int[9](0,3,4,1,2,7,8,5,6);' +
    'float fAt(int i,vec2 p){if(i<4)return texture(A,p)[i];if(i<8)return texture(B,p)[i-4];return texture(C,p).x;}' +
    'bool wall(vec2 p){return texture(S,p).r>.5;}' +
    'bool outY(vec2 p){return p.y<0.||p.y>1.;}' +
    'float feq(int i,float r,vec2 u){float eu=dot(E[i],u);return WT[i]*r*(1.+3.*eu+4.5*eu*eu-1.5*dot(u,u));}' +
    'void main(){' +
    'if(wall(vUv)){float rs=0.,n=0.;for(int i=1;i<9;i++){vec2 q=vUv+E[i]*texel;' +
    'if(q.x<0.||q.x>1.||outY(q)||wall(q))continue;rs=max(rs,texture(C,q).y);n+=1.;}' +   // the side the flow presses on
    'oA=vec4(0.);oB=vec4(0.);oC=vec4(0.,n>0.?rs:1.,0.,0.);return;}' +
    'float f[9];float rho=0.;vec2 u=vec2(0.);' +
    'if(init>.5||vUv.x<texel.x||vUv.y<texel.y||vUv.y>1.-texel.y){rho=1.;u=vec2(uin,0.);for(int i=0;i<9;i++)f[i]=feq(i,rho,u);}' +   // start, the inflow column, and the free stream above and below
    'else if(vUv.x>1.-texel.x){rho=1.;u=texture(C,vUv-vec2(texel.x,0.)).zw;for(int i=0;i<9;i++)f[i]=feq(i,rho,u);}' +   // pressure outlet: reference pressure, the flow's own velocity
    'else{for(int i=0;i<9;i++){vec2 q=vUv-E[i]*texel;' +
    'if(outY(q)||wall(q))f[i]=fAt(OPP[i],vUv);else f[i]=fAt(i,q);}' +                             // halfway bounce-back
    'for(int i=0;i<9;i++){rho+=f[i];u+=f[i]*E[i];}u/=rho;}' +
    'for(int i=0;i<9;i++)f[i]-=(f[i]-feq(i,rho,u))/tau;' +
    'oA=vec4(f[0],f[1],f[2],f[3]);oB=vec4(f[4],f[5],f[6],f[7]);oC=vec4(f[8],rho,u);}';
  // The colour of the air: the computed speed, interpolated bilinearly between
  // the four nearest lattice nodes (fluid nodes only), then the figure's own
  // STEPPED table - the steps' edges follow the field smoothly, never the
  // cells. Wall cells are drawn as the data paths themselves (the line layer).
  var SHOW_FS = '#version 300 es\nprecision highp float;precision highp sampler2D;in vec2 vUv;out vec4 o;' +
    'uniform sampler2D C,S;uniform vec2 grid,plot;uniform float uref;uniform float sT[6];uniform vec4 sC[6];' +
    'void main(){vec2 g=vec2(vUv.x*plot.x+plot.y,vUv.y*grid.y)-.5;vec2 i0=floor(g),fr=g-i0;float sp=0.,wt=0.;' +
    'for(int k=0;k<4;k++){vec2 o2=vec2(float(k&1),float(k>>1));ivec2 ij=ivec2(clamp(i0+o2,vec2(0.),grid-1.));' +
    'float w=(o2.x>.5?fr.x:1.-fr.x)*(o2.y>.5?fr.y:1.-fr.y);' +
    'if(texelFetch(S,ij,0).r>.5){continue;}sp+=w*length(texelFetch(C,ij,0).zw);wt+=w;}' +
    'if(wt<.25){o=vec4(0.);return;}' +
    'float s=sp/wt/uref;vec4 k=sC[5];for(int i=0;i<6;i++){if(s<sT[i]){k=sC[i];break;}}' +
    'o=vec4(k.rgb*k.a,k.a);}';

  // ---- one tunnel per cone -----------------------------------------------------------
  function Tunnel(gd, still) {
    var T = this;
    T.gd = gd; gd.__wind = T; T.still = still; T.mode = still ? 'lbm-still' : 'lbm';
    T.cv = mk('wind-tunnel');
    var gl = T.gl = T.cv.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) { T.failed = true; return; }
    T.lbm = prog(gl, LBM_FS); T.show = prog(gl, SHOW_FS);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    T.grid = mk('wind-tunnel-grid'); T.flow = mk('wind-tunnel-flow'); T.over = mk('wind-tunnel-verify');
    T.btn = D.createElement('button');
    T.btn.type = 'button'; T.btn.className = 'wt-verify'; T.btn.textContent = 'Verification Overlay';
    T.btn.setAttribute('aria-pressed', String(api.overlay));
    T.btn.title = 'Show the solver\'s numbers: velocity arrows with speeds, wall pressures, and the solver\'s state';
    T.btn.addEventListener('click', function () {
      api.overlay = !api.overlay;
      api.tunnels.forEach(function (t) { t.btn.setAttribute('aria-pressed', String(api.overlay)); t.drawOverlay(); });
    });
    T.frames = 0; T.steps = 0; T.raf = 0; T.ok = false;
    T.read();
    if (gd.on) gd.on('plotly_afterplot', function () { T.read(); });
    T.io = new IntersectionObserver(function (es) { T.visible = es[0].isIntersecting; T.wake(); }, { threshold: 0.05 });
    T.io.observe(gd);
    api.tunnels.push(T);
  }
  function mk(cls) {
    var c = D.createElement('canvas');
    c.className = cls; c.setAttribute('aria-hidden', 'true');
    c.style.cssText = 'position:absolute;pointer-events:none;';
    return c;
  }
  function prog(gl, fs) {
    function sh(type, s) {
      var o = gl.createShader(type); gl.shaderSource(o, s); gl.compileShader(o);
      if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o));
      return o;
    }
    var p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'aPos'); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    var u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < n; i++) { var name = gl.getActiveUniform(p, i).name; u[name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, name); }
    return { p: p, u: u };
  }
  function tex(gl, w, h, internal, format, type, data) {
    var t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  // a lattice state: three float textures behind one framebuffer (MRT)
  Tunnel.prototype.state = function (w, h) {
    var gl = this.gl, s = { A: tex(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT), B: tex(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT),
                            C: tex(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT), fb: gl.createFramebuffer() };
    gl.bindFramebuffer(gl.FRAMEBUFFER, s.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, s.A, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, s.B, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, s.C, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    return s;
  };
  Tunnel.prototype.free = function () {
    var gl = this.gl, L = this.L;
    if (!L) return;
    [L.cur, L.nxt].forEach(function (s) { [s.A, s.B, s.C].forEach(function (t) { gl.deleteTexture(t); }); gl.deleteFramebuffer(s.fb); });
    if (L.S) gl.deleteTexture(L.S);
    if (L.chk) { gl.deleteTexture(L.chk.t); gl.deleteFramebuffer(L.chk.fb); }
    this.L = null;
  };

  // The data, every time the chart is drawn: the five computed paths, the axes,
  // the colour tables. A new cone gets new walls and a fresh lattice.
  Tunnel.prototype.read = function () {
    var T = this, gd = T.gd, fl = gd._fullLayout, fd = gd._fullData, meta = metaOf(gd);
    if (T.failed) return;
    if (!fl || !fd || !fl.xaxis || !fl.yaxis || !meta || !meta.flow) { T.ok = false; return; }
    var q = {};
    for (var i = 0; i < fd.length; i++) if (fd[i].name && !(fd[i].name in q)) q[fd[i].name] = fd[i];
    var days = q.median && Array.from(q.median.x || []);
    var lv = LEVELS.map(function (k) { return q[k] ? Array.from(q[k].y || []) : null; });
    if (!days || days.length < 2 || lv.some(function (a) { return !a || a.length !== days.length; })) { T.ok = false; return; }
    var s = fl._size;
    var key = JSON.stringify([days.length, lv.map(function (a) { return [a[0], a[a.length >> 1], a[a.length - 1]]; }), s.w, s.h, s.l, s.t]);
    T.days = days; T.lv = lv; T.xa = fl.xaxis; T.ya = fl.yaxis; T.sz = s;
    T.ramp = meta.flow;
    var box = gd.querySelector('.svg-container');
    if (!box) { T.ok = false; return; }
    var svgs = box.querySelectorAll('svg.main-svg'), top = svgs[1] || null;
    [T.grid, T.cv, T.flow, T.over].forEach(function (c) { box.insertBefore(c, top); });   // under Plotly's hover layer
    if (T.btn.parentNode !== box) box.appendChild(T.btn);
    var dpr = Math.min(2, W.devicePixelRatio || 1);
    [T.grid, T.cv, T.flow, T.over].forEach(function (c) {
      c.style.left = s.l + 'px'; c.style.top = s.t + 'px'; c.style.width = s.w + 'px'; c.style.height = s.h + 'px';
    });
    T.btn.style.cssText = 'position:absolute;z-index:3;top:' + Math.max(2, s.t - 30) + 'px;right:' + s.r + 'px;';
    T.dpr = dpr;
    T.ok = s.w > 40 && s.h > 40;
    if (!T.ok) return;
    if (key === T.key && T.L) { T.layers(); return; }  // same cone, same size: keep the running flow
    T.key = key;
    var pc = T.still ? LBM.COLS_STILL : LBM.COLS, rows = Math.max(32, Math.round(pc * s.h / s.w));
    var cols = pc + LBM.PAD_IN + LBM.PAD_OUT;
    T.free();
    T.L = { cols: cols, pc: pc, pad: LBM.PAD_IN, rows: rows, cur: T.state(cols, rows), nxt: T.state(cols, rows), S: null, chk: null };
    T.sx = s.w / pc; T.sy = s.h / rows;                // plot pixels per cell
    T.cv.width = Math.round(s.w * dpr); T.cv.height = Math.round(s.h * dpr);   // full resolution: no cells on screen
    T.walls(cols, rows);
    T.clock();
    T.steps = 0; T.done = false; T.lastState = null;
    T.step(true);                                         // equilibrium at the inflow everywhere
    T.layers();
    T.draw();
    T.wake();
  };

  // The percentile paths as walls, at the lattice's resolution: drawn by the
  // browser's 2D rasteriser from the chart's own pixel coordinates.
  Tunnel.prototype.walls = function (cols, rows) {
    var T = this, gl = T.gl, c = D.createElement('canvas'), x = c.getContext('2d'), sz = T.sz, d = T.days;
    c.width = cols; c.height = rows;
    var sx = T.L.pc / sz.w, sy = rows / sz.h, pad = T.L.pad;
    x.strokeStyle = '#fff'; x.lineWidth = LBM.LINE_CELLS; x.lineJoin = 'round'; x.lineCap = 'round';
    T.lv.forEach(function (path) {
      x.beginPath();
      for (var i = 0; i < d.length; i++) x[i ? 'lineTo' : 'moveTo'](T.xa.l2p(d[i]) * sx + pad, T.ya.l2p(path[i]) * sy);
      x.stroke();
    });
    var img = x.getImageData(0, 0, cols, rows).data, solid = new Uint8Array(cols * rows), up = new Uint8Array(cols * rows * 4);
    for (var r = 0; r < rows; r++) for (var k = 0; k < cols; k++) {
      var on = img[4 * (r * cols + k) + 3] > 64 ? 1 : 0;       // covered by a path: a wall
      var gy = rows - 1 - r;                                  // GL rows count up
      solid[gy * cols + k] = on;
      up[4 * (gy * cols + k)] = on ? 255 : 0; up[4 * (gy * cols + k) + 3] = 255;
    }
    T.solid = solid;
    T.L.S = tex(gl, cols, rows, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, up);
  };

  // The clock from the data: steps per trading day, and the inflow on each day
  // from how fast the computed 5-95% band widens that day.
  Tunnel.prototype.clock = function () {
    var T = this, d = T.days, n = d.length, lo = T.lv[0], hi = T.lv[4], wid = [], wmax = 0;
    for (var i = 0; i < n; i++) {
      var a = Math.max(0, i - 2), b = Math.min(n - 1, i + 2);
      var w = b > a ? ((hi[b] - lo[b]) - (hi[a] - lo[a])) / (d[b] - d[a]) : 0;
      wid.push(Math.max(0, w)); wmax = Math.max(wmax, w);
    }
    T.widen = wid; T.widenMax = wmax;
    T.stepsPerDay = T.L.pc / (LBM.U_IN * n);
  };
  Tunnel.prototype.dayAt = function (step) { return Math.floor(step / this.stepsPerDay) % this.days.length; };
  Tunnel.prototype.inflowAt = function (step) {
    var w = this.widenMax > 0 ? this.widen[this.dayAt(step)] / this.widenMax : 0;
    return LBM.U_IN * (1 - LBM.U_SWING + LBM.U_SWING * w);
  };

  Tunnel.prototype.pass = function (P, out, w, h, texs, uni) {
    var gl = this.gl, unit = 0;
    gl.useProgram(P.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out);
    gl.viewport(0, 0, w, h);
    for (var k in texs) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texs[k]);
      if (P.u[k]) gl.uniform1i(P.u[k], unit);
      unit++;
    }
    for (var j in uni) {
      var v = uni[j], loc = P.u[j];
      if (!loc) continue;
      if (typeof v === 'number') gl.uniform1f(loc, v);
      else if (v.length === 2) gl.uniform2fv(loc, v);
      else if (v.f1) gl.uniform1fv(loc, v.f1);
      else if (v.f4) gl.uniform4fv(loc, v.f4);
    }
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  Tunnel.prototype.step = function (init) {
    var T = this, L = T.L;
    T.pass(T.lbm, L.nxt.fb, L.cols, L.rows, { A: L.cur.A, B: L.cur.B, C: L.cur.C, S: L.S },
           { texel: [1 / L.cols, 1 / L.rows], tau: LBM.TAU, uin: T.inflowAt(T.steps), init: init ? 1 : 0 });
    var t = L.cur; L.cur = L.nxt; L.nxt = t;
    if (!init) T.steps++;
  };
  Tunnel.prototype.rampUniforms = function () {
    var R = this.ramp, sT = [], sC = [], wT = [], wC = [];
    R.speed.forEach(function (b) { var c = hex(b[1]); sT.push(b[0]); sC.push(c[0] / 255, c[1] / 255, c[2] / 255, b[2]); });
    R.wall.forEach(function (b) { var c = hex(b[1]); wT.push(b[0]); wC.push(c[0] / 255, c[1] / 255, c[2] / 255, 1); });
    return { uref: LBM.U_IN, grid: [this.L.cols, this.L.rows], plot: [this.L.pc, this.L.pad], sT: { f1: sT }, sC: { f4: sC } };
  };
  Tunnel.prototype.draw = function () {
    var T = this, L = T.L, gl = T.gl;
    gl.disable(gl.BLEND);
    T.pass(T.show, null, T.cv.width, T.cv.height, { C: L.cur.C, S: L.S }, T.rampUniforms());
    T.frames++;
  };
  // the computed state, back on the CPU (density and velocity per cell)
  Tunnel.prototype.readState = function () {
    var T = this, L = T.L, gl = T.gl, n = L.cols * L.rows * 4;
    if (!T.cBuf || T.cBuf.length !== n) T.cBuf = new Float32Array(n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, L.cur.fb);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(0, 0, L.cols, L.rows, gl.RGBA, gl.FLOAT, T.cBuf);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    return T.cBuf;
  };
  // the same stepped tables on the CPU (for streamlines, the overlay and the proof)
  Tunnel.prototype.band = function (table, v) {
    for (var i = 0; i < table.length; i++) if (v < table[i][0]) return i;
    return table.length - 1;
  };
  // Cp = (p - p_ref) / (rho U^2 / 2), with p = rho/3 (lattice units) and p_ref
  // the mean pressure where the air leaves (the outflow column): the
  // reference an engineer reads a wind tunnel's pressures against. The cone
  // blocks much of the tunnel, so the air ahead of it is pushed above the
  // outflow pressure and Cp can pass 1 (a confined duct, not open air).
  Tunnel.prototype.cp = function (rho) { return (rho - this.rhoRef) / 3 / (0.5 * LBM.U_IN * LBM.U_IN); };
  Tunnel.prototype.reference = function (u) {
    var L = this.L, s = 0, n = 0, c = L.cols - 2;
    for (var r = 1; r < L.rows - 1; r++) { var j = r * L.cols + c; if (!this.solid[j]) { s += u[4 * j + 1]; n++; } }
    this.rhoRef = n ? s / n : 1;
  };
  // the pressure on a path at one of its data points: the wall cells within a
  // cell and a half of it, their highest Cp (the side the flow hits)
  Tunnel.prototype.pathCp = function (u, X, Y) {
    var L = this.L, cx = X / this.sx + L.pad, cy = L.rows - Y / this.sy, best = null;
    for (var dy = -2; dy <= 2; dy++) for (var dx = -2; dx <= 2; dx++) {
      var xx = Math.floor(cx) + dx, yy = Math.floor(cy) + dy;
      if (xx < 0 || yy < 0 || xx >= L.cols || yy >= L.rows) continue;
      if ((xx + 0.5 - cx) * (xx + 0.5 - cx) + (yy + 0.5 - cy) * (yy + 0.5 - cy) > 2.25) continue;
      var j = yy * L.cols + xx;
      if (this.solid[j]) { var v = this.cp(u[4 * j + 1]); if (best === null || v > best) best = v; }
    }
    return best;
  };

  // The faint lattice mesh and the domain's 1px wireframe box (static per cone).
  Tunnel.prototype.layers = function () {
    var T = this, sz = T.sz, L = T.L, dpr = T.dpr;
    [T.grid, T.flow, T.over].forEach(function (c) {
      var w = Math.round(sz.w * dpr), h = Math.round(sz.h * dpr);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    });
    var x = T.grid.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    x.strokeStyle = 'rgba(63,59,53,0.07)'; x.lineWidth = 1 / dpr; x.beginPath();
    var cw = T.sx, ch = T.sy, every = cw < 3 ? 2 : 1;      // a line per cell (per two when cells are tiny)
    for (var i = 0; i <= L.pc; i += every) { var X = Math.round(i * cw * dpr) / dpr + 0.5 / dpr; x.moveTo(X, 0); x.lineTo(X, sz.h); }
    for (var j = 0; j <= L.rows; j += every) { var Y = Math.round(j * ch * dpr) / dpr + 0.5 / dpr; x.moveTo(0, Y); x.lineTo(sz.w, Y); }
    x.stroke();
    x.strokeStyle = '#3F3B35'; x.lineWidth = 1; x.strokeRect(0.5, 0.5, sz.w - 1, sz.h - 1);   // the domain
    T.drawOverlay();
  };

  // FluidX3D's streamlines, seeded at the data's nodes, then the data paths
  // themselves, drawn as smooth lines through their exact points and coloured
  // segment by segment by the pressure the flow puts on them.
  Tunnel.prototype.vel = function (u, px, py) {      // bilinear velocity at a point (cell units), null in a wall
    var L = this.L, cols = L.cols, rows = L.rows, gx = px - 0.5, gy = py - 0.5, i0 = Math.floor(gx), j0 = Math.floor(gy);
    var fx = gx - i0, fy = gy - j0, vx = 0, vy = 0, wt = 0;
    for (var k = 0; k < 4; k++) {
      var i = Math.min(cols - 1, Math.max(0, i0 + (k & 1))), j = Math.min(rows - 1, Math.max(0, j0 + (k >> 1)));
      var w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy), c = j * cols + i;
      if (this.solid[c]) continue;
      vx += w * u[4 * c + 2]; vy += w * u[4 * c + 3]; wt += w;
    }
    if (wt < 0.5) return null;
    return [vx / wt, vy / wt];
  };
  Tunnel.prototype.streams = function (cbuf) {
    var T = this, L = T.L, sz = T.sz, dpr = T.dpr, cols = L.cols, rows = L.rows, u = cbuf || T.readState();
    var sx = T.sx, sy = T.sy, pad = L.pad, x = T.flow.getContext('2d'), R = T.ramp.speed, H = LBM.H;
    T.reference(u);
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    x.lineWidth = 0.75; x.lineCap = 'round'; x.lineJoin = 'round';
    var paths = {}, seeds = [];
    for (var k = 0; k < T.lv.length; k++) for (var i = 0; i < T.days.length; i += LBM.SEED_DAYS) {
      var cx = T.xa.l2p(T.days[i]) / sx + pad, cy = rows - T.ya.l2p(T.lv[k][i]) / sy;
      seeds.push([cx, cy + LBM.SEED_OFFSET], [cx, cy - LBM.SEED_OFFSET]);
    }
    for (var r = 1; r < rows - 1; r += LBM.SEED_ROWS) seeds.push([1.0, r + 0.5]);     // the inflow, in the lead-in
    seeds.forEach(function (sd) {
      var px = sd[0], py = sd[1];
      for (var l = 0; l < LBM.LEN; l++) {
        if (px < 0 || py < 0 || px > cols || py > rows) break;
        var a = T.vel(u, px, py);
        if (!a) break;
        var al = Math.sqrt(a[0] * a[0] + a[1] * a[1]);
        if (al < 0.1 * LBM.U_IN) break;                                        // FluidX3D: stop where the flow is still
        var mx = px + H * a[0] / al, my = py + H * a[1] / al, bq = T.vel(u, mx, my);   // Heun: average the two slopes
        if (!bq) break;
        var bl = Math.sqrt(bq[0] * bq[0] + bq[1] * bq[1]) || al;
        var dx = 0.5 * (a[0] / al + bq[0] / bl), dy = 0.5 * (a[1] / al + bq[1] / bl), dl = Math.sqrt(dx * dx + dy * dy) || 1;
        var nx = px + H * dx / dl, ny = py + H * dy / dl, b = T.band(R, al / LBM.U_IN);
        if (b > 0) (paths[b] || (paths[b] = [])).push((px - pad) * sx, sz.h - py * sy, (nx - pad) * sx, sz.h - ny * sy);
        px = nx; py = ny;
      }
    });
    for (var bk in paths) {
      var Lp = paths[bk]; x.strokeStyle = R[bk][1]; x.globalAlpha = Math.min(1, R[bk][2] + 0.2); x.beginPath();
      for (var m = 0; m < Lp.length; m += 4) { x.moveTo(Lp[m], Lp[m + 1]); x.lineTo(Lp[m + 2], Lp[m + 3]); }
      x.stroke();
    }
    x.globalAlpha = 1;
    // the data paths, through their exact points, each segment in its pressure step
    var W2 = T.ramp.wall, segs = [];
    x.lineWidth = 2; x.lineCap = 'butt';                // square ends: a short segment keeps its own colour
    for (var q = 0; q < T.lv.length; q++) for (var d = 0; d + 1 < T.days.length; d++) {
      var X0 = T.xa.l2p(T.days[d]), Y0 = T.ya.l2p(T.lv[q][d]), X1 = T.xa.l2p(T.days[d + 1]), Y1 = T.ya.l2p(T.lv[q][d + 1]);
      var cpv = T.pathCp(u, (X0 + X1) / 2, (Y0 + Y1) / 2), wb = cpv === null ? 0 : T.band(W2, cpv);
      segs.push([q, d, wb, cpv]);
      x.strokeStyle = W2[wb][1]; x.beginPath(); x.moveTo(X0, Y0); x.lineTo(X1, Y1); x.stroke();
    }
    T.segs = segs;
    T.lastState = u;
    if (api.overlay) T.drawOverlay(u);
  };

  // Verification Overlay: the solver's own numbers on the picture.
  Tunnel.prototype.drawOverlay = function (cbuf) {
    var T = this, x = T.over.getContext('2d'), sz = T.sz, dpr = T.dpr, L = T.L;
    if (!L) return;
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    if (!api.overlay) return;
    var u = cbuf || T.lastState || T.readState(), cols = L.cols, rows = L.rows, sx = T.sx, sy = T.sy, pad = L.pad;
    x.font = '9px ui-monospace, Menlo, Consolas, monospace'; x.textBaseline = 'middle';
    var label = function (s, X, Y, col) {
      var w = x.measureText(s).width + 4;
      x.fillStyle = 'rgba(237,233,227,0.9)'; x.fillRect(X, Y - 6, w, 12);
      x.fillStyle = col; x.fillText(s, X + 2, Y);
    };
    // velocity arrows at grid intersections, the speed (x inflow) beside them
    var g = LBM.ARROW_CELLS, fastest = { s: -1 }, wallTop = [];
    for (var cy = g / 2; cy < rows; cy += g) for (var cx = pad + g / 2; cx < pad + L.pc; cx += g) {
      var i = cy * cols + cx;
      if (T.solid[i]) continue;
      var ux = u[4 * i + 2], uy = u[4 * i + 3], s = Math.sqrt(ux * ux + uy * uy) / LBM.U_IN;
      if (s < T.ramp.speed[0][0]) continue;            // air at rest: no arrow (it is transparent too)
      var X = (cx - pad + 0.5) * sx, Y = sz.h - (cy + 0.5) * sy, len = Math.min(1.6, s) * 9, ang = Math.atan2(-uy, ux);
      var tx = X + Math.cos(ang) * len, ty = Y + Math.sin(ang) * len;
      x.strokeStyle = '#1C1A17'; x.fillStyle = '#1C1A17'; x.lineWidth = 1;
      x.beginPath(); x.moveTo(X, Y); x.lineTo(tx, ty); x.stroke();
      x.beginPath(); x.moveTo(tx, ty);
      x.lineTo(tx + Math.cos(ang + 2.6) * 3.2, ty + Math.sin(ang + 2.6) * 3.2);
      x.lineTo(tx + Math.cos(ang - 2.6) * 3.2, ty + Math.sin(ang - 2.6) * 3.2);
      x.fill();
      if (((cx - pad - g / 2) / g + (cy - g / 2) / g) % 2 === 0) label(s.toFixed(2), X + 3, Y + 7, '#1C1A17');
    }
    for (var j = 0; j < cols * rows; j++) {
      if (T.solid[j] || j % cols < pad || j % cols >= pad + L.pc) continue;   // the plot's own cells
      var sp = Math.sqrt(u[4 * j + 2] * u[4 * j + 2] + u[4 * j + 3] * u[4 * j + 3]) / LBM.U_IN;
      if (sp > fastest.s) fastest = { s: sp, j: j };
    }
    // the highest pressures on the data paths: where the flow hits them hardest (the red)
    (T.segs || []).forEach(function (sg) { if (sg[3] !== null) wallTop.push([sg[3], sg[0], sg[1]]); });
    wallTop.sort(function (a, b) { return b[0] - a[0]; });
    var marked = [];
    for (var t = 0; t < wallTop.length && marked.length < 5; t++) {
      var pq = wallTop[t][1], pd = wallTop[t][2];
      var wx = (T.xa.l2p(T.days[pd]) + T.xa.l2p(T.days[pd + 1])) / 2, wy = (T.ya.l2p(T.lv[pq][pd]) + T.ya.l2p(T.lv[pq][pd + 1])) / 2;
      if (marked.some(function (m) { return Math.abs(m[0] - wx) < 60 && Math.abs(m[1] - wy) < 18; })) continue;
      marked.push([wx, wy]);
      x.strokeStyle = '#B00020'; x.lineWidth = 1; x.strokeRect(wx - 3.5, wy - 3.5, 7, 7);
      label('Cp ' + wallTop[t][0].toFixed(2), wx + 6, wy - 8, '#9E0B1F');
    }
    if (fastest.s > 0) {
      var fx = (fastest.j % cols - pad + 0.5) * sx, fy = sz.h - (Math.floor(fastest.j / cols) + 0.5) * sy;
      x.strokeStyle = '#0B5F8A'; x.lineWidth = 1; x.beginPath(); x.arc(fx, fy, 4, 0, 6.2832); x.stroke();
      label('max |u| ' + fastest.s.toFixed(2) + ' U', fx + 6, fy + 8, '#0B4F73');
    }
    var nu = (LBM.TAU - 0.5) / 3, day = T.dayAt(T.steps);
    label('D2Q9 lattice Boltzmann · ' + L.pc + ' x ' + rows + ' cells (+' + (cols - L.pc) + ' columns off-plot) · step ' + T.steps.toLocaleString('en-US') +
          ' · trading day ' + T.days[day] + ' · inflow ' + (T.inflowAt(T.steps) / LBM.U_IN).toFixed(3) + ' U · tau ' + LBM.TAU +
          ' · Re ' + Math.round(LBM.U_IN * L.pc / nu).toLocaleString('en-US'), 4, 9, '#1C1A17');
  };

  // ---- running ---------------------------------------------------------------------
  Tunnel.prototype.wake = function () {
    var run = this.ok && !this.done && this.visible && !D.hidden && this.gd.isConnected;
    if (run && !this.raf) { var T = this; this.raf = W.requestAnimationFrame(function (t) { T.frame(t); }); }
    if (!run && this.raf) { W.cancelAnimationFrame(this.raf); this.raf = 0; }
  };
  Tunnel.prototype.frame = function (now) {
    var T = this;
    T.raf = 0;
    if (!T.gd.isConnected) { prune(); return; }
    // the page's own motion first: hold while a glide flies or the reader scrolls
    if (W.__melionaStop || now - inputAt < LBM.STILL_MS) { T.wake(); return; }
    if (T.still) {
      var goal = Math.round(LBM.STILL_TRANSITS * T.L.pc / LBM.U_IN);
      for (var i = 0; i < LBM.STILL_CHUNK && T.steps < goal; i++) T.step();
      if (T.steps >= goal) { T.draw(); T.streams(); T.done = true; }
    } else {
      for (var k = 0; k < LBM.STEPS_PER_FRAME; k++) T.step();
      T.draw();
      if (T.frames % LBM.STREAM_EVERY === 0) T.streams();
    }
    T.wake();
  };

  // ---- the proof, on the rendered pixels -----------------------------------------
  // (1) The air: draw it once more with one pixel at each lattice node, read it
  //     and the computed state back, and compare every node's colour with the
  //     stepped table applied to its own computed speed (a wall node: nothing).
  // (2) The data paths: at the middle of every drawn segment, read the screen
  //     pixel and compare it with the pressure step of that segment's computed
  //     Cp (segments where two paths meet, as at day 1, are skipped).
  // (3) Red: no red pixel anywhere on the picture farther than 3px from a data
  //     path; and the highest pressure on the paths is painted red when it
  //     reaches the red step.
  // (4) The walls are the data: each path's point every 5 trading days sits on
  //     (or within one cell of) a wall cell.
  Tunnel.prototype.verify = function (out) {
    var T = this, gl = T.gl, L = T.L;
    if (!L) return;
    if (!T.lastState) T.streams();
    var u = T.lastState;
    if (!L.chk) {
      var tx = tex(gl, L.cols, L.rows, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx, 0);
      L.chk = { t: tx, fb: fb };
    }
    // the state the picture was drawn from goes back to the GPU's view: draw (1) from the same state
    var n = L.cols * L.rows, px = new Uint8Array(n * 4), cur = T.readState();
    var ru = T.rampUniforms(); ru.plot = [L.cols, 0];     // every node of the grid, lead-in and run-out included
    T.pass(T.show, L.chk.fb, L.cols, L.rows, { C: L.cur.C, S: L.S }, ru);
    gl.bindFramebuffer(gl.FRAMEBUFFER, L.chk.fb);
    gl.readPixels(0, 0, L.cols, L.rows, gl.RGBA, gl.UNSIGNED_BYTE, px);
    var R = T.ramp, red = R.wall.length - 1, redRGB = hex(R.wall[red][1]);
    for (var j = 0; j < n; j++) {
      var w;
      if (T.solid[j]) w = [0, 0, 0, 0];
      else {
        var v = Math.sqrt(cur[4 * j + 2] * cur[4 * j + 2] + cur[4 * j + 3] * cur[4 * j + 3]) / LBM.U_IN, e = R.speed[T.band(R.speed, v)], c = hex(e[1]);
        w = [c[0] * e[2], c[1] * e[2], c[2] * e[2], 255 * e[2]];
        out.maxSpeed = Math.max(out.maxSpeed, v);
      }
      out.particles++;
      if (Math.abs(px[4 * j] - w[0]) > 2 || Math.abs(px[4 * j + 1] - w[1]) > 2 || Math.abs(px[4 * j + 2] - w[2]) > 2 || Math.abs(px[4 * j + 3] - w[3]) > 2) {
        out.colourMismatch++;
        if ((out.worst = out.worst || []).length < 4) out.worst.push({ cell: j, wall: !!T.solid[j], got: Array.from(px.slice(4 * j, 4 * j + 4)), want: w.map(Math.round) });
      }
      if (Math.abs(px[4 * j] - redRGB[0]) < 40 && px[4 * j + 1] < 60 && px[4 * j + 2] < 80 && px[4 * j + 3] > 0) out.redOffWall++;   // the air is never red
    }
    // (2) and (3) on the drawn picture
    var dpr = T.dpr, cw = T.flow.width, chh = T.flow.height, img = T.flow.getContext('2d').getImageData(0, 0, cw, chh).data;
    var pts = [];
    for (var q = 0; q < T.lv.length; q++) for (var d = 0; d < T.days.length; d++) pts.push([T.xa.l2p(T.days[d]), T.ya.l2p(T.lv[q][d]), q]);
    var near = function (X, Y, r, skipQ) {
      for (var k = 0; k < pts.length; k++) if (pts[k][2] !== skipQ && Math.abs(pts[k][0] - X) <= r && Math.abs(pts[k][1] - Y) <= r) return true;
      return false;
    };
    var peak = null;
    (T.segs || []).forEach(function (sg) {
      var q2 = sg[0], d2 = sg[1];
      var X = (T.xa.l2p(T.days[d2]) + T.xa.l2p(T.days[d2 + 1])) / 2, Y = (T.ya.l2p(T.lv[q2][d2]) + T.ya.l2p(T.lv[q2][d2 + 1])) / 2;
      if (sg[3] !== null && (!peak || sg[3] > peak.cp)) peak = { cp: sg[3], X: X, Y: Y, b: sg[2] };
      if (near(X, Y, 6, q2) || X < 3 || Y < 3 || X > T.sz.w - 3 || Y > T.sz.h - 3) return;   // shared pixels with another path, or the domain's edge
      var c2 = hex(R.wall[sg[2]][1]), k2 = 4 * (Math.round(Y * dpr) * cw + Math.round(X * dpr)), found = false;
      for (var oy = -Math.ceil(dpr); oy <= Math.ceil(dpr) && !found; oy++) for (var ox = -Math.ceil(dpr); ox <= Math.ceil(dpr) && !found; ox++) {
        var kk = 4 * ((Math.round(Y * dpr) + oy) * cw + Math.round(X * dpr) + ox);   // the line's core pixel, within 1px
        if (Math.abs(img[kk] - c2[0]) <= 8 && Math.abs(img[kk + 1] - c2[1]) <= 8 && Math.abs(img[kk + 2] - c2[2]) <= 8 && img[kk + 3] > 230) found = true;
      }
      out.wallChecks++;
      if (!found) {
        out.wallMismatch++;
        if ((out.wallWorst = out.wallWorst || []).length < 3) out.wallWorst.push({ path: LEVELS[q2], day: T.days[d2], cp: sg[3], got: Array.from(img.slice(k2, k2 + 3)), want: c2 });
      }
    });
    for (var yy = 0; yy < chh; yy += 2) for (var xx = 0; xx < cw; xx += 2) {
      var k3 = 4 * (yy * cw + xx);
      if (img[k3 + 3] > 200 && Math.abs(img[k3] - redRGB[0]) < 30 && Math.abs(img[k3 + 1] - redRGB[1]) < 30 && Math.abs(img[k3 + 2] - redRGB[2]) < 30 &&
          !near(xx / dpr, yy / dpr, 3, -1)) out.redOffWall++;
    }
    if (peak) {
      out.maxCp = +peak.cp.toFixed(3);
      var k4 = 4 * (Math.round(peak.Y * dpr) * cw + Math.round(peak.X * dpr));
      out.peakIsRed = peak.cp < R.wall[red - 1][0] ? null :
        (peak.b === red && Math.abs(img[k4] - redRGB[0]) <= 8 && Math.abs(img[k4 + 1] - redRGB[1]) <= 8 && Math.abs(img[k4 + 2] - redRGB[2]) <= 8);
    }
    // (4)
    var sx = T.sx, sy = T.sy;
    for (var k5 = 0; k5 < T.lv.length; k5++) for (var i5 = 0; i5 < T.days.length; i5 += 5) {
      var cx = Math.floor(T.xa.l2p(T.days[i5]) / sx + L.pad), cy = L.rows - 1 - Math.floor(T.ya.l2p(T.lv[k5][i5]) / sy), hit = false;
      for (var dy = -1; dy <= 1 && !hit; dy++) for (var dx = -1; dx <= 1 && !hit; dx++) {
        var x6 = cx + dx, y6 = cy + dy;
        if (x6 >= 0 && y6 >= 0 && x6 < L.cols && y6 < L.rows && T.solid[y6 * L.cols + x6]) hit = true;
      }
      out.obstacleChecks++;
      if (!hit) out.obstacleMiss++;
    }
    out.maxSpeed = +out.maxSpeed.toFixed(3);
    out.steps = T.steps; out.day = T.days[T.dayAt(T.steps)]; out.stepsPerDay = +T.stepsPerDay.toFixed(3);
    out.inflow = +(T.inflowAt(T.steps) / LBM.U_IN).toFixed(4); out.rhoRef = +T.rhoRef.toFixed(5);
  };
  api.check = function () {
    prune();
    var out = { mode: null, particles: 0, colourMismatch: 0, redOffWall: 0, obstacleMiss: 0, obstacleChecks: 0,
                wallChecks: 0, wallMismatch: 0,
                maxSpeed: 0, maxCp: null, peakIsRed: null, p5End: null, p95End: null, frames: 0 };
    api.tunnels.forEach(function (T) {
      if (!T.ok || !T.L) return;
      out.mode = T.mode; out.frames = T.frames;
      out.p5End = T.lv[0][T.lv[0].length - 1]; out.p95End = T.lv[4][T.lv[4].length - 1];
      T.verify(out);
    });
    return out;
  };

  // ---- finding the cone ----------------------------------------------------------
  // A chart Streamlit replaced takes its tunnel with it: stopped, its GPU
  // memory freed, and out of check().
  function prune() {
    api.tunnels = api.tunnels.filter(function (T) {
      if (T.gd.isConnected) return true;
      if (T.raf) W.cancelAnimationFrame(T.raf);
      if (T.io) T.io.disconnect();
      T.free();
      var lose = T.gl && T.gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
      return false;
    });
  }
  function find() {
    prune();
    var g = glSupport();
    // No GPU, no flow: a software renderer (SwiftShader, llvmpipe) runs the
    // browser's own compositing on the same emulated GPU, and the solve froze
    // the whole page on CI's machines (Live audit, 2026-11-04). The chart is
    // complete without it. tools/flow_check.mjs proves the solve in CI.
    if (!g || g.soft) { api.status = g ? 'software renderer: flow not run' : 'no WebGL2: flow not run'; return; }
    var plots = D.querySelectorAll('.js-plotly-plot');
    for (var i = 0; i < plots.length; i++) {
      var gd = plots[i], m = metaOf(gd);
      if (gd.__wind) { if (!gd.__wind.cv.isConnected) gd.__wind.read(); continue; }
      if (m && m.meliona === 'cone') new Tunnel(gd, reduced);
    }
  }
  var queued = 0;
  new MutationObserver(function () {
    if (!queued) queued = W.requestAnimationFrame(function () { queued = 0; find(); });
  }).observe(D.body, { childList: true, subtree: true });
  var inputAt = -Infinity;
  ['wheel', 'touchmove', 'scroll'].forEach(function (k) {
    W.addEventListener(k, function () { inputAt = performance.now(); }, { capture: true, passive: true });
  });
  D.addEventListener('visibilitychange', function () { api.tunnels.forEach(function (T) { T.wake(); }); });
  W.addEventListener('resize', function () { api.tunnels.forEach(function (T) { T.read(); }); });
  find();
})();
