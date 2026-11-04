/* Meliona wind tunnel: a lattice Boltzmann flow around the cone's computed
 * percentile paths, drawn the way FluidX3D draws a velocity field.
 *
 * John, 2026-11-04: "Every line, color, velocity vector, and animation
 * timestep must be directly driven by and deterministically bound to the
 * provided data array"; FluidX3D's look, "grid cells with over 100,000 time
 * steps", on the site's own background; no glass, gloss, glow or blur; not
 * pixelated; transparent at rest; red only where the flow hits the data; a
 * Verification Overlay. "Each color, cell, line, animation and where it's
 * going reflects the chart and the data in the chart."
 *
 * The calculation (FluidX3D's own method, github.com/ProjectPhysX/FluidX3D):
 *   - D2Q9 lattice Boltzmann, BGK collision, on a lattice of about 100,000+
 *     cells laid over the plot (a cell is about CELL_PX screen pixels) and
 *     reaching PAD_Y of the plot's height beyond it above and below - open air,
 *     so the cone is a body in a stream rather than a plug in a pipe. Nine
 *     distribution functions per cell, streamed (pull scheme) and relaxed
 *     toward equilibrium each step, with FluidX3D's Smagorinsky-Lilly subgrid
 *     model (SUBGRID) raising the relaxation time where the flow shears, so a
 *     thin viscosity stays stable. Density and velocity are the moments. No
 *     noise, no random numbers, no timers in the physics: the state after
 *     step n is a pure function of the data and n.
 *   - The DATA are the walls. The chart's five computed percentile paths
 *     (5th, 25th, median, 75th, 95th, read from Plotly's _fullData, the same
 *     arrays the hover prints) are rasterised into the lattice as solid cells;
 *     the flow bounces back off them (halfway bounce-back) and has to
 *     compress, shear and turn around their shape. Air enters on the left at
 *     the inflow speed and leaves on the right; the top and bottom rows are
 *     free stream (an open test section). A lead-in and a run-out are computed
 *     off-plot, so the boundaries sit away from the data.
 *   - The CLOCK is the data's. STEPS_PER_DAY = columns / (U_IN x trading
 *     days): air at the inflow speed crosses the plot in exactly one simulated
 *     year, so the trading day being replayed is step / STEPS_PER_DAY and the
 *     air over day d is the air that has travelled d days' worth of chart.
 *     The inflow on that day is U_IN x (1 - U_SWING + U_SWING x w), w = how
 *     fast the computed 5-95% band widens that day against its fastest day: a
 *     calm year blows steady, a shocked one gusts where its cone opens. The
 *     step counter runs past 100,000 in about a minute; how fast the screen
 *     refreshes changes only how many steps are shown per frame, never what a
 *     step computes.
 *
 * The picture (FluidX3D's VIS_FIELD, on Meliona's page instead of black):
 *   - THE GRID IS THE CHART'S AXES. Its columns are trading days (one cell =
 *     ND trading days, ND the smallest of 1, 2, 3, 5, 10, 21 that leaves room
 *     for a line) and its rows are steps of cumulative return (one cell = DR
 *     percent, a round step, rows aligned on 0%). A faint mesh draws the
 *     cells; the overlay prints ND and DR.
 *   - ONE LINE PER CELL, as FluidX3D's graphics_field kernel draws one line
 *     per lattice point (draw_line(p - u/2|u|, p + u/2|u|)): centred in its
 *     cell, pointing where the computed air at that (trading day, return
 *     level) is going, coloured by its speed. The velocity is interpolated
 *     between the four nearest lattice nodes (fluid nodes only).
 *   - THE COLOURS ARE FLUIDX3D'S OWN SCALE. colorscale_rainbow (kernel.cpp:
 *     black, violet, blue, cyan, green, yellow, red from slow to fast), of
 *     which the air takes only the cool part (black to green, `top`): the hot
 *     end - red - is left to the data. FluidX3D does not draw a lattice point
 *     whose scaled speed is under 0.1 (`rest`): still air is transparent; up
 *     to `fade`, a line fades into the page as FluidX3D's dark slow colours
 *     fade into its black ground. `shade` darkens the scale for a light page.
 *     All of it is main.py's FLOW_AIR, in the figure's layout.meta, the same
 *     table the key under the chart prints.
 *   - TRACER PARTICLES (FluidX3D's PARTICLES: integrate_particles, with its
 *     particle_boundary_force): one per grid row - one return level - is
 *     released at the inflow every ND trading days, and every solver step
 *     moves it by the computed velocity where it is. In undisturbed air they
 *     ride one grid cell apart, through the cells' centres; where the cone
 *     bends, slows or squeezes the air, they leave the grid. Each is coloured
 *     like the line of the air it is in.
 *   - THE DATA PATHS are drawn as smooth lines through their exact points,
 *     each segment in main.py's FLOW_WALL_STEPS step by the pressure the flow
 *     puts on it (Cp against the free stream passing the same trading day) -
 *     as FluidX3D colours a boundary by the force on it. Red therefore exists
 *     ONLY on the data, exactly where the flow runs into it.
 *   - A line of solver state in the corner, as FluidX3D prints its own: the
 *     lattice, the step, and the trading day being replayed.
 *
 * "Verification Overlay" (a button on the chart): the grid's size in days and
 * percent, arrowheads and printed speeds on a sample of cells, the solid
 * lattice cells' outline (FluidX3D's VIS_FLAG_LATTICE), the highest-pressure
 * wall points with their Cp, the fastest cell, and the solver's state (inflow,
 * tau, Reynolds number). window.__melionaWind.check() proves the same on the
 * rendered output: every line's colour and direction against its own
 * computed velocity, every line on its trading day and return level, every
 * particle's step against FluidX3D's integrate_particles recomputed on the
 * CPU and its colour against the air it is in, no red off a wall, the walls
 * on the data, and the peak pressure painted red.
 *
 * Performance - the page's own motion first (John, 2026-11-04: on the live
 * site the tunnel cost the arrivals and the buttons). Nothing is built while
 * the page loads: a cone's tunnel is built only once the cone comes within a
 * screen of the view, in idle time, with no glide, arrival or scroll moving.
 * Its programs compile in the background (KHR_parallel_shader_compile), and
 * the live loop reads the GPU back through a fence, never a blocking
 * readPixels. It runs only on screen, in a visible tab, and holds while the
 * reader scrolls, a button's glide flies or a block arrives; a slow card
 * shows fewer steps per frame, and one that cannot keep up even then keeps a
 * still frame. Under reduced motion the same calculation is computed in
 * small slices and shown as a still frame. A software renderer
 * (SwiftShader, llvmpipe) or no WebGL2: the chart alone - an emulated GPU also
 * draws the page, and the solve froze it. tools/flow_check.mjs runs and
 * proves the solve in CI.
 */
(function () {
  'use strict';
  var W = window, D = document;
  if (W.__melionaWind || !('IntersectionObserver' in W)) return;   // an old browser: the chart alone
  var api = W.__melionaWind = { tunnels: [], check: function () { return null; }, renderer: null, overlay: false };
  var reduced = !!(W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ---- the solver's constants -------------------------------------------------
  var LBM = {
    CELL_PX: 1.75,              // plot pixels per lattice cell (about 100,000+ cells on a desktop plot)
    COLS_MIN: 160, COLS_MAX: 760, COLS_STILL: 300,   // lattice columns across the plot (live / still frame)
    TAU: 0.52,                  // BGK relaxation time: viscosity nu = (TAU - 0.5) / 3, raised locally by the
                                // Smagorinsky-Lilly subgrid model where the flow shears (FluidX3D's SUBGRID)
    U_IN: 0.07,                 // inflow speed, lattice units (Mach 0.12: keeps the lattice's compressibility error small)
    U_SWING: 0.15,              // the inflow's share that follows the cone's widening on the replayed day
    SPIN_UP: 0.5,               // the tunnel starts still and the inflow rises linearly over this share of a
                                // transit (an impulsive start rings the closed cone like a bell)
    STEPS_PER_FRAME: 24,        // solver steps shown per displayed frame (live) ...
    STEPS_MIN: 4,               // ... fewer, down to this, on a card that cannot keep up
    STILL_TRANSITS: 1.6,        // a still frame: the inflow crosses the plot this many times
    STILL_CHUNK: 12,            // ... at most this many steps per idle frame
    LINE_CELLS: 1.6,            // a percentile path's wall thickness, in lattice cells
    CELL_MIN_PX: 6,             // a drawn grid cell is at least this many pixels on a side
    DAY_STEPS: [1, 2, 3, 5, 10, 21],                        // grid columns: trading days per cell
    RET_STEPS: [0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50],   // grid rows: percent of return per cell
    LINE_FILL: 0.84,            // a cell's line spans this share of the cell's shorter side
    WALL_EVERY: 8,              // live: re-measure the pressure on the paths every this many frames
    LABEL_EVERY: 6,             // Verification Overlay: a printed speed every this many cells each way
    STILL_MS: 320,              // hold still this long after the reader's last scroll input
    PAD_IN: 24, PAD_OUT: 16,    // computed but not drawn: a lead-in before day 1 and a run-out after the last day,
    PAD_Y: 0.4,                 // ... and this share of the plot's height above and below it: open air, so the
                                // cone is a body in a stream, not a plug in a pipe
    PARTICLE_PX: 2.6,           // a tracer particle's diameter, in pixels
    PARTICLE_LIFE: 1.6          // a particle is recycled after this many inflow transits of the whole lattice
  };
  var LEVELS = ['p5', 'p25', 'median', 'p75', 'p95'];

  // a button's glide is flying: main.py's glide script keeps its stop handle
  // on this page's document (its P), not on the window
  function gliding() { return !!(D.__melionaStop || W.__melionaStop); }
  // the page is moving: a glide, an arrival, or the reader's own scroll
  function busy() {
    var R = W.__melionaReveal;
    return gliding() || (R && R.moving && R.moving() > 0) || performance.now() - inputAt < LBM.STILL_MS;
  }
  function hex(c) { return [1, 3, 5].map(function (i) { return parseInt(c.slice(i, i + 2), 16); }); }
  function metaOf(gd) { return (gd._fullLayout && gd._fullLayout.meta) || (gd.layout && gd.layout.meta) || null; }

  // FluidX3D's colorscale_rainbow (src/kernel.cpp), line for line: x in [0, 1]
  // -> black, violet, blue, cyan, green, yellow, red.
  function rainbow(x) {
    x = Math.min(6, Math.max(0, 6 * (1 - x)));
    var r = 0, g = 0, b = 0;
    if (x < 1.2) { r = 1; g = x * 0.83333333; }
    else if (x < 2) { r = 2.5 - x * 1.25; g = 1; }
    else if (x < 3) { g = 1; b = x - 2; }
    else if (x < 4) { g = 4 - x; b = 1; }
    else if (x < 5) { r = x * 0.4 - 1.6; b = 3 - x * 0.5; }
    else { r = 2.4 - x * 0.4; b = 3 - x * 0.5; }
    return [r, g, b];
  }
  // a cell's line: [r, g, b, a] in 0..1 (straight alpha) for a speed s x inflow
  function airColour(A, s) {
    var q = s / A.full, a = A.alpha * Math.min(1, Math.max(0, (q - A.rest) / (A.fade - A.rest)));
    var c = rainbow(A.top * Math.min(1, q));
    return [A.shade * c[0], A.shade * c[1], A.shade * c[2], a];
  }

  // ---- WebGL2 with float render targets, and whether it is a real GPU ----------
  // One context per cone, made once and asked about its renderer: no
  // throwaway probe context (making a context is not free).
  var CTX = { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false };
  function glContext(cv) {
    var gl = null, name = 'none';
    try { gl = cv.getContext('webgl2', CTX); } catch (e) { gl = null; }
    if (gl && gl.getExtension('EXT_color_buffer_float')) {
      var ri = gl.getExtension('WEBGL_debug_renderer_info');
      name = String(ri ? gl.getParameter(ri.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    } else gl = null;
    api.renderer = name;
    return { gl: gl, soft: /swiftshader|llvmpipe|softpipe|software|basic render/i.test(name) };
  }

  // ---- shaders -------------------------------------------------------------------
  var VS = '#version 300 es\nin vec2 aPos;out vec2 vUv;void main(){vUv=aPos*.5+.5;gl_Position=vec4(aPos,0.,1.);}';
  // One D2Q9 step: pull-stream (bounce back off the walls), inflow on the
  // left and free stream on the top and bottom rows, a pressure outlet on the
  // right (reference density, the arriving flow's velocity), BGK
  // collision. Textures hold POST-collision distributions: A = f0..f3,
  // B = f4..f7, C = (f8, density, ux, uy). A wall cell stores, in C.y, the
  // highest density of its MOVING fluid neighbours (the pressure the flow puts
  // on it), or -1 where only still air touches it: still air is not drawn,
  // and it does not count as the flow hitting the data either.
  var LBM_FS = '#version 300 es\nprecision highp float;precision highp sampler2D;in vec2 vUv;' +
    'layout(location=0) out vec4 oA;layout(location=1) out vec4 oB;layout(location=2) out vec4 oC;' +
    'uniform sampler2D A,B,C,S;uniform vec2 texel;uniform float tau,uin,init,ustill;' +
    'const vec2 E[9]=vec2[9](vec2(0,0),vec2(1,0),vec2(0,1),vec2(-1,0),vec2(0,-1),vec2(1,1),vec2(-1,1),vec2(-1,-1),vec2(1,-1));' +
    'const float WT[9]=float[9](4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.);' +
    'const int OPP[9]=int[9](0,3,4,1,2,7,8,5,6);' +
    'float fAt(int i,vec2 p){if(i<4)return texture(A,p)[i];if(i<8)return texture(B,p)[i-4];return texture(C,p).x;}' +
    'bool wall(vec2 p){return texture(S,p).r>.5;}' +
    'bool outY(vec2 p){return p.y<0.||p.y>1.;}' +
    'float feq(int i,float r,vec2 u){float eu=dot(E[i],u);return WT[i]*r*(1.+3.*eu+4.5*eu*eu-1.5*dot(u,u));}' +
    'void main(){' +
    'if(wall(vUv)){float rs=0.,n=0.;for(int i=1;i<9;i++){vec2 q=vUv+E[i]*texel;' +
    'if(q.x<0.||q.x>1.||outY(q)||wall(q))continue;vec4 cq=texture(C,q);if(length(cq.zw)<ustill)continue;rs=max(rs,cq.y);n+=1.;}' +   // the side the flow presses on
    'oA=vec4(0.);oB=vec4(0.);oC=vec4(0.,n>0.?rs:-1.,0.,0.);return;}' +
    'float f[9];float rho=0.;vec2 u=vec2(0.);' +
    'if(init>.5){rho=1.;for(int i=0;i<9;i++)f[i]=feq(i,rho,u);}' +                               // start: still air everywhere
    'else if(vUv.x<texel.x||vUv.y<texel.y||vUv.y>1.-texel.y){rho=1.;u=vec2(uin,0.);for(int i=0;i<9;i++)f[i]=feq(i,rho,u);}' +   // the inflow column, and the free stream above and below
    'else if(vUv.x>1.-texel.x){rho=1.;u=texture(C,vUv-vec2(texel.x,0.)).zw;for(int i=0;i<9;i++)f[i]=feq(i,rho,u);}' +   // pressure outlet: reference pressure, the flow's own velocity
    'else{for(int i=0;i<9;i++){vec2 q=vUv-E[i]*texel;' +
    'if(outY(q)||wall(q))f[i]=fAt(OPP[i],vUv);else f[i]=fAt(i,q);}' +                             // halfway bounce-back
    'for(int i=0;i<9;i++){rho+=f[i];u+=f[i]*E[i];}u/=rho;}' +
    'float fe[9];float hxx=0.,hyy=0.,hxy=0.;for(int i=0;i<9;i++){fe[i]=feq(i,rho,u);float fn=f[i]-fe[i];hxx+=E[i].x*E[i].x*fn;hyy+=E[i].y*E[i].y*fn;hxy+=E[i].x*E[i].y*fn;}' +
    'float te=.5*(tau+sqrt(tau*tau+.76421222*sqrt(hxx*hxx+hyy*hyy+2.*hxy*hxy)/rho));' +   // Smagorinsky-Lilly: FluidX3D's constant
    'for(int i=0;i<9;i++)f[i]-=(f[i]-fe[i])/te;' +
    'oA=vec4(f[0],f[1],f[2],f[3]);oB=vec4(f[4],f[5],f[6],f[7]);oC=vec4(f[8],rho,u);}';
  // FluidX3D's graphics_field, one line per grid cell. aN = (lattice x, lattice
  // y [cell units, y up], grid column, grid row). The velocity is the bilinear
  // mix of the four nearest FLUID lattice nodes (none if under half the weight
  // is fluid). mode 0 draws the line (premultiplied); mode 1 writes the cell's
  // colour, mode 2 its direction and speed, one pixel per cell, for check().
  var NEEDLE_VS = '#version 300 es\nprecision highp float;precision highp sampler2D;' +
    'in vec4 aN;out vec4 vC;' +
    'uniform sampler2D C,S;uniform vec2 grid,plot,cell,ng;uniform float pad,ytop,half_,uref,full,rest,fade,top,shade,alpha;uniform int mode;' +
    'vec3 rainbow(float x){x=clamp(6.*(1.-x),0.,6.);float r=0.,g=0.,b=0.;' +
    'if(x<1.2){r=1.;g=x*.83333333;}else if(x<2.){r=2.5-x*1.25;g=1.;}else if(x<3.){g=1.;b=x-2.;}' +
    'else if(x<4.){g=4.-x;b=1.;}else if(x<5.){r=x*.4-1.6;b=3.-x*.5;}else{r=2.4-x*.4;b=3.-x*.5;}return vec3(r,g,b);}' +
    'void main(){vec2 g=aN.xy-.5;vec2 i0=floor(g),fr=g-i0;vec2 v=vec2(0.);float wt=0.;' +
    'for(int k=0;k<4;k++){vec2 o=vec2(float(k&1),float(k>>1));ivec2 ij=ivec2(clamp(i0+o,vec2(0.),grid-1.));' +
    'float w=(o.x>.5?fr.x:1.-fr.x)*(o.y>.5?fr.y:1.-fr.y);if(texelFetch(S,ij,0).r>.5)continue;v+=w*texelFetch(C,ij,0).zw;wt+=w;}' +
    'float s=0.;vec2 d=vec2(0.);bool fl=wt>=.5;if(fl){v/=wt;float l=length(v);s=l/uref;if(l>0.)d=v/l;}' +
    'float q=s/full;float a=fl?alpha*clamp((q-rest)/(fade-rest),0.,1.):0.;vec3 c=shade*rainbow(top*min(1.,q));' +
    'if(mode==0){vec2 X=vec2((aN.x-pad)*cell.x,(ytop-aN.y)*cell.y)+(gl_VertexID==0?-1.:1.)*half_*vec2(d.x,-d.y);' +
    'gl_Position=vec4(X.x/plot.x*2.-1.,1.-X.y/plot.y*2.,0.,1.);vC=vec4(c*a,a);}' +
    'else{gl_PointSize=1.;gl_Position=vec4((aN.z+.5)/ng.x*2.-1.,(aN.w+.5)/ng.y*2.-1.,0.,1.);' +
    'vC=mode==1?vec4(c,a):vec4(d*.5+.5,min(1.,s/4.),fl?1.:0.);}}';
  var NEEDLE_FS = '#version 300 es\nprecision highp float;in vec4 vC;out vec4 o;void main(){o=vC;}';
  // FluidX3D's tracer particles (PARTICLES, integrate_particles): each solver
  // step a particle moves by the velocity interpolated at its position
  // (interpolate_u: the four nearest nodes, a wall node's velocity is zero),
  // plus particle_boundary_force, which steers it off a wall rather than into
  // it. P = (x, y, 0, alive) in lattice cell units. One particle per grid row
  // (a return level) is released at the inflow every ND trading days: in
  // undisturbed air they ride one grid cell apart, so where the cone bends the
  // air, they leave the grid. `born` is the slot released this step (-1: none).
  var PART_FS = '#version 300 es\nprecision highp float;precision highp sampler2D;out vec4 o;' +
    'uniform sampler2D P,C,S;uniform vec2 grid;uniform float born,y0,dy;' +
    'void main(){ivec2 id=ivec2(gl_FragCoord.xy);vec4 q=texelFetch(P,id,0);' +
    'if(float(id.x)==born){o=vec4(1.,y0+float(id.y)*dy,0.,1.);return;}' +
    'if(q.w<.5){o=q;return;}' +
    'vec2 g=q.xy-.5;vec2 i0=floor(g),fr=g-i0;vec2 un=vec2(0.),bf=vec2(0.);float bd=2.;' +
    'for(int k=0;k<4;k++){vec2 c=vec2(float(k&1),float(k>>1));ivec2 ij=ivec2(clamp(i0+c,vec2(0.),grid-1.));' +
    'un+=(c.x>.5?fr.x:1.-fr.x)*(c.y>.5?fr.y:1.-fr.y)*texelFetch(C,ij,0).zw;' +
    'if(texelFetch(S,ij,0).r>.5){bf+=vec2(.5)-c;bd=min(bd,length(fr-c));}}' +
    'if(bd<1.&&length(bf)>0.)un+=length(un)*normalize(bf);' +
    'vec2 p=q.xy+un;o=vec4(p,0.,(p.x<grid.x-1.&&p.y>0.&&p.y<grid.y)?1.:0.);}';
  // the particles on screen: a disc coloured like the line of the air it is
  // in (mode 0), or one pixel per particle with that colour (mode 1, check())
  var PSHOW_VS = '#version 300 es\nprecision highp float;precision highp sampler2D;out vec4 vC;' +
    'uniform sampler2D P,C,S;uniform vec2 grid,plot,cell,ng;uniform float pad,ytop,psize,uref,full,rest,fade,top,shade,alpha;uniform int mode;' +
    'vec3 rainbow(float x){x=clamp(6.*(1.-x),0.,6.);float r=0.,g=0.,b=0.;' +
    'if(x<1.2){r=1.;g=x*.83333333;}else if(x<2.){r=2.5-x*1.25;g=1.;}else if(x<3.){g=1.;b=x-2.;}' +
    'else if(x<4.){g=4.-x;b=1.;}else if(x<5.){r=x*.4-1.6;b=3.-x*.5;}else{r=2.4-x*.4;b=3.-x*.5;}return vec3(r,g,b);}' +
    'void main(){int n=int(ng.x);ivec2 id=ivec2(gl_VertexID%n,gl_VertexID/n);vec4 q=texelFetch(P,id,0);' +
    'vec2 g=q.xy-.5;vec2 i0=floor(g),fr=g-i0;vec2 v=vec2(0.);float wt=0.;' +
    'for(int k=0;k<4;k++){vec2 o=vec2(float(k&1),float(k>>1));ivec2 ij=ivec2(clamp(i0+o,vec2(0.),grid-1.));' +
    'float w=(o.x>.5?fr.x:1.-fr.x)*(o.y>.5?fr.y:1.-fr.y);if(texelFetch(S,ij,0).r>.5)continue;v+=w*texelFetch(C,ij,0).zw;wt+=w;}' +
    'float s=wt>=.5?length(v/wt)/uref:0.;float qq=s/full;' +
    'vec2 X=vec2((q.x-pad)*cell.x,(ytop-q.y)*cell.y);' +
    'bool vis=q.w>.5&&wt>=.5&&X.x>=0.&&X.x<=plot.x&&X.y>=0.&&X.y<=plot.y;' +
    'float a=vis?alpha*clamp((qq-rest)/(fade-rest),0.,1.):0.;vec3 c=shade*rainbow(top*min(1.,qq));' +
    'if(mode==0){gl_PointSize=psize;gl_Position=a>0.?vec4(X.x/plot.x*2.-1.,1.-X.y/plot.y*2.,0.,1.):vec4(2.,2.,2.,1.);vC=vec4(c*a,a);}' +
    'else{gl_PointSize=1.;gl_Position=vec4((float(id.x)+.5)/ng.x*2.-1.,(float(id.y)+.5)/ng.y*2.-1.,0.,1.);vC=vec4(c,a);}}';
  var PSHOW_FS = '#version 300 es\nprecision highp float;in vec4 vC;out vec4 o;uniform float psize;' +
    'void main(){float r=length(gl_PointCoord-.5)*psize;o=vC*clamp(psize*.5-r+.5,0.,1.);}';

  // ---- one tunnel per cone -----------------------------------------------------------
  function Tunnel(gd, still, cv, gl) {
    var T = this;
    T.gd = gd; gd.__wind = T; T.still = still; T.mode = still ? 'lbm-still' : 'lbm';
    T.mesh = mk('wind-tunnel-grid'); T.cv = cv; T.flow = mk('wind-tunnel-flow'); T.over = mk('wind-tunnel-verify');
    T.gl = gl;
    T.par = gl.getExtension('KHR_parallel_shader_compile');
    T.lbm = prog(gl, VS, LBM_FS); T.ndl = prog(gl, NEEDLE_VS, NEEDLE_FS);
    T.adv = prog(gl, VS, PART_FS); T.pshow = prog(gl, PSHOW_VS, PSHOW_FS);
    T.compiling = true;
    T.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, T.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    T.nbuf = gl.createBuffer();
    T.btn = D.createElement('button');
    T.btn.type = 'button'; T.btn.className = 'wt-verify'; T.btn.textContent = 'Verification Overlay';
    T.btn.setAttribute('aria-pressed', String(api.overlay));
    T.btn.title = 'Show the solver\'s numbers: the grid in days and percent, speeds, wall pressures, and the solver\'s state';
    T.btn.addEventListener('click', function () {
      api.overlay = !api.overlay;
      api.tunnels.forEach(function (t) { t.btn.setAttribute('aria-pressed', String(api.overlay)); t.drawOverlay(); });
    });
    T.frames = 0; T.steps = 0; T.raf = 0; T.ok = false; T.spf = LBM.STEPS_PER_FRAME; T.slow = 0; T.lastFrame = 0;
    (function poll() {                                   // the programs, then the lattice: nothing waits on the driver
      if (!gd.isConnected) return;
      var progs = [T.lbm, T.ndl, T.adv, T.pshow];
      if (!progs.every(function (P) { return ready(gl, P, T.par); }) || busy()) { W.requestAnimationFrame(poll); return; }
      try { progs.forEach(function (P) { finish(gl, P); }); } catch (e) { T.failed = true; api.status = 'shader: ' + e.message; return; }
      T.compiling = false;
      T.read();
      T.wake();
    })();
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
  // Programs are compiled in the background (KHR_parallel_shader_compile):
  // asking a program for its status before the driver has finished blocks
  // the page until it has, and on Windows (ANGLE) these shaders take long
  // enough to freeze every button and arrival on the page (John,
  // 2026-11-04). ready() is asked once per frame; finish() reads the
  // uniforms only when the driver says the program is done.
  function prog(gl, vs, fs) {
    function sh(type, s) { var o = gl.createShader(type); gl.shaderSource(o, s); gl.compileShader(o); return o; }
    var p = gl.createProgram(), a = sh(gl.VERTEX_SHADER, vs), b = sh(gl.FRAGMENT_SHADER, fs);
    gl.attachShader(p, a); gl.attachShader(p, b);
    if (vs === VS) gl.bindAttribLocation(p, 0, 'aPos'); else if (vs === NEEDLE_VS) gl.bindAttribLocation(p, 0, 'aN');
    gl.linkProgram(p);
    return { p: p, u: null, sh: [a, b] };
  }
  function ready(gl, P, par) { return !par || gl.getProgramParameter(P.p, par.COMPLETION_STATUS_KHR); }
  function finish(gl, P) {
    if (!gl.getProgramParameter(P.p, gl.LINK_STATUS)) {
      throw new Error(P.sh.map(function (o) { return gl.getShaderInfoLog(o); }).join(' ') + gl.getProgramInfoLog(P.p));
    }
    var u = {}, n = gl.getProgramParameter(P.p, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < n; i++) { var name = gl.getActiveUniform(P.p, i).name; u[name.replace(/\[0\]$/, '')] = gl.getUniformLocation(P.p, name); }
    P.u = u;
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
    if (L.pchk) { gl.deleteTexture(L.pchk.t); gl.deleteFramebuffer(L.pchk.fb); }
    if (L.P) [L.P.cur, L.P.nxt].forEach(function (b) { gl.deleteTexture(b.t); gl.deleteFramebuffer(b.fb); });
    if (this.rb) { gl.deleteSync(this.rb.sync); this.rb = null; }
    this.L = null;
  };

  // The data, every time the chart is drawn: the five computed paths, the axes,
  // the colour tables. A new cone gets new walls, a new grid and a fresh lattice.
  Tunnel.prototype.read = function () {
    var T = this, gd = T.gd, fl = gd._fullLayout, fd = gd._fullData, meta = metaOf(gd);
    if (T.failed || T.compiling) return;
    if (!fl || !fd || !fl.xaxis || !fl.yaxis || !meta || !meta.flow) { T.ok = false; return; }
    var q = {};
    for (var i = 0; i < fd.length; i++) if (fd[i].name && !(fd[i].name in q)) q[fd[i].name] = fd[i];
    var days = q.median && Array.from(q.median.x || []);
    var lv = LEVELS.map(function (k) { return q[k] ? Array.from(q[k].y || []) : null; });
    if (!days || days.length < 2 || lv.some(function (a) { return !a || a.length !== days.length; })) { T.ok = false; return; }
    var s = fl._size;
    var key = JSON.stringify([days.length, lv.map(function (a) { return [a[0], a[a.length >> 1], a[a.length - 1]]; }),
                              s.w, s.h, s.l, s.t, fl.yaxis.range, fl.xaxis.range]);
    T.days = days; T.lv = lv; T.xa = fl.xaxis; T.ya = fl.yaxis; T.sz = s;
    T.air = meta.flow.air; T.wallSteps = meta.flow.wall;
    var box = gd.querySelector('.svg-container');
    if (!box) { T.ok = false; return; }
    var svgs = box.querySelectorAll('svg.main-svg'), topSvg = svgs[1] || null;
    [T.mesh, T.cv, T.flow, T.over].forEach(function (c) { box.insertBefore(c, topSvg); });   // under Plotly's hover layer
    if (T.btn.parentNode !== box) box.appendChild(T.btn);
    var dpr = Math.min(2, W.devicePixelRatio || 1);
    [T.mesh, T.cv, T.flow, T.over].forEach(function (c) {
      c.style.left = s.l + 'px'; c.style.top = s.t + 'px'; c.style.width = s.w + 'px'; c.style.height = s.h + 'px';
    });
    T.btn.style.cssText = 'position:absolute;z-index:3;top:' + Math.max(2, s.t - 30) + 'px;right:' + s.r + 'px;';
    T.dpr = dpr;
    T.ok = s.w > 40 && s.h > 40;
    if (!T.ok) return;
    if (key === T.key && T.L) { T.layers(); return; }  // same cone, same size: keep the running flow
    T.key = key;
    var pc = Math.max(LBM.COLS_MIN, Math.min(T.still ? LBM.COLS_STILL : LBM.COLS_MAX, Math.round(s.w / LBM.CELL_PX)));
    var rp = Math.max(32, Math.round(pc * s.h / s.w)), py = Math.round(rp * LBM.PAD_Y);
    var rows = rp + 2 * py, cols = pc + LBM.PAD_IN + LBM.PAD_OUT;
    T.free();
    T.L = { cols: cols, pc: pc, pad: LBM.PAD_IN, rows: rows, rp: rp, py: py, ytop: py + rp,
            cur: T.state(cols, rows), nxt: T.state(cols, rows), S: null, chk: null };
    T.sx = s.w / pc; T.sy = s.h / rp;                  // plot pixels per lattice cell
    T.cv.width = Math.round(s.w * dpr); T.cv.height = Math.round(s.h * dpr);
    T.walls(cols, rows);
    T.cells();
    T.clock();
    T.seed();
    T.steps = 0; T.done = false; T.lastState = null; T.segs = null; T.born = -1;
    T.step(true);                                         // still air everywhere
    T.layers();
    T.draw();
    T.paint(null);                                        // the paths in their rest colour until the first reading
    T.wake();
  };

  // The percentile paths as walls, at the lattice's resolution: drawn by the
  // browser's 2D rasteriser from the chart's own pixel coordinates.
  Tunnel.prototype.walls = function (cols, rows) {
    var T = this, gl = T.gl, c = D.createElement('canvas'), x = c.getContext('2d'), sz = T.sz, d = T.days;
    c.width = cols; c.height = rows;
    var sx = T.L.pc / sz.w, sy = T.L.rp / sz.h, pad = T.L.pad, top = T.L.py;
    x.strokeStyle = '#fff'; x.lineWidth = LBM.LINE_CELLS; x.lineJoin = 'round'; x.lineCap = 'round';
    T.lv.forEach(function (path) {
      x.beginPath();
      for (var i = 0; i < d.length; i++) x[i ? 'lineTo' : 'moveTo'](T.xa.l2p(d[i]) * sx + pad, T.ya.l2p(path[i]) * sy + top);
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

  // The drawn grid, from the chart's axes: columns of ND trading days, rows of
  // DR percent of return (aligned on 0%), each the smallest round step that
  // leaves CELL_MIN_PX for its line. One line per cell, at the cell's centre.
  Tunnel.prototype.cells = function () {
    var T = this, sz = T.sz, d = T.days, n = d.length, L = T.L;
    var pxDay = Math.abs(T.xa.l2p(d[n - 1]) - T.xa.l2p(d[0])) / (d[n - 1] - d[0]);
    var pxPct = Math.abs(T.ya.l2p(1) - T.ya.l2p(0));
    var nd = LBM.DAY_STEPS.find(function (k) { return k * pxDay >= LBM.CELL_MIN_PX; }) || LBM.DAY_STEPS[LBM.DAY_STEPS.length - 1];
    var dr = LBM.RET_STEPS.find(function (k) { return k * pxPct >= LBM.CELL_MIN_PX; }) || LBM.RET_STEPS[LBM.RET_STEPS.length - 1];
    var yr = T.ya.range.slice().sort(function (a, b) { return a - b; });
    var cols = [], rowsv = [];
    for (var c = d[0]; c + nd <= d[n - 1] + 1e-9; c += nd) cols.push(c + nd / 2);           // day columns, from the first day
    for (var r = Math.floor(yr[0] / dr); (r + 1) * dr <= yr[1] + 1e-9; r++) if (r * dr >= yr[0] - 1e-9) rowsv.push((r + 0.5) * dr);
    var data = new Float32Array(cols.length * rowsv.length * 4), list = [];
    for (var j = 0; j < rowsv.length; j++) for (var i = 0; i < cols.length; i++) {
      var X = T.xa.l2p(cols[i]), Y = T.ya.l2p(rowsv[j]), k = 4 * (j * cols.length + i);
      data[k] = X / T.sx + L.pad; data[k + 1] = L.ytop - Y / T.sy; data[k + 2] = i; data[k + 3] = j;
      list.push({ day: cols[i], ret: rowsv[j], X: X, Y: Y });
    }
    T.grid = { nd: nd, dr: dr, cols: cols, rows: rowsv, data: data, list: list,
               half: 0.5 * LBM.LINE_FILL * Math.min(nd * pxDay, dr * pxPct) };
    var gl = T.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, T.nbuf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
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
  // The tracer particles: K slots per grid row. Slot floor(step / release)
  // mod K of every row is released at the inflow when that number ticks over,
  // release = ND trading days of steps; K covers PARTICLE_LIFE transits, so a
  // slot is reused only after its particle has left (or stalled in still air).
  Tunnel.prototype.seed = function () {
    var T = this, gl = T.gl, L = T.L, G = T.grid, nc = G.cols.length, nr = G.rows.length;
    T.release = G.nd * T.stepsPerDay;
    if (!nc || !nr) return;
    var K = Math.ceil(LBM.PARTICLE_LIFE * L.cols / LBM.U_IN / T.release);
    var buf = function () {
      var t = tex(gl, K, nr, gl.RGBA32F, gl.RGBA, gl.FLOAT, new Float32Array(K * nr * 4)), fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      return { t: t, fb: fb };
    };
    L.P = { cur: buf(), nxt: buf(), K: K, nr: nr, y0: G.data[1], dy: nr > 1 ? G.data[4 * nc + 1] - G.data[1] : 0 };
  };
  Tunnel.prototype.dayAt = function (step) { return Math.floor(step / this.stepsPerDay) % this.days.length; };
  Tunnel.prototype.inflowAt = function (step) {
    var w = this.widenMax > 0 ? this.widen[this.dayAt(step)] / this.widenMax : 0;
    return LBM.U_IN * (1 - LBM.U_SWING + LBM.U_SWING * w) * Math.min(1, step / (LBM.SPIN_UP * this.L.pc / LBM.U_IN));
  };

  Tunnel.prototype.bind = function (P, texs, uni) {
    var gl = this.gl, unit = 0;
    gl.useProgram(P.p);
    for (var k in texs) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texs[k]);
      if (P.u[k]) gl.uniform1i(P.u[k], unit);
      unit++;
    }
    for (var j in uni) {
      var v = uni[j], loc = P.u[j];
      if (!loc) continue;
      if (j === 'mode') gl.uniform1i(loc, v);
      else if (typeof v === 'number') gl.uniform1f(loc, v);
      else gl.uniform2fv(loc, v);
    }
  };
  Tunnel.prototype.step = function (init) {
    var T = this, L = T.L, gl = T.gl;
    T.bind(T.lbm, { A: L.cur.A, B: L.cur.B, C: L.cur.C, S: L.S },
           { texel: [1 / L.cols, 1 / L.rows], tau: LBM.TAU, uin: T.inflowAt(T.steps), init: init ? 1 : 0,
             ustill: T.air.rest * T.air.full * LBM.U_IN });
    gl.bindFramebuffer(gl.FRAMEBUFFER, L.nxt.fb);
    gl.viewport(0, 0, L.cols, L.rows);
    gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, T.quad);
    gl.enableVertexAttribArray(0); gl.vertexAttribDivisor(0, 0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    var t = L.cur; L.cur = L.nxt; L.nxt = t;
    if (init) return;
    T.steps++;
    if (L.P) {
      var b = Math.floor(T.steps / T.release), b0 = Math.floor((T.steps - 1) / T.release);
      T.advect(L.P.nxt.fb, b !== b0 ? b % L.P.K : -1);
      var pt = L.P.cur; L.P.cur = L.P.nxt; L.P.nxt = pt;
    }
  };
  // one step of the particles, into `fb` (born: the slot released now, or -1)
  Tunnel.prototype.advect = function (fb, born) {
    var T = this, L = T.L, gl = T.gl, P = L.P;
    T.bind(T.adv, { P: P.cur.t, C: L.cur.C, S: L.S }, { grid: [L.cols, L.rows], born: born, y0: P.y0, dy: P.dy });
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, P.K, P.nr);
    gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, T.quad);
    gl.enableVertexAttribArray(0); gl.vertexAttribDivisor(0, 0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  // the particles: mode 0 on screen (over the lines, no clear); 1 into the check target
  Tunnel.prototype.particles = function (mode, fb, w, h) {
    var T = this, L = T.L, gl = T.gl, A = T.air, P = L.P;
    if (!P) return;
    T.bind(T.pshow, { P: P.cur.t, C: L.cur.C, S: L.S }, {
      grid: [L.cols, L.rows], plot: [T.sz.w, T.sz.h], cell: [T.sx, T.sy], ng: [P.K, P.nr], pad: L.pad, ytop: L.ytop,
      psize: mode === 0 ? LBM.PARTICLE_PX * T.dpr : 1, uref: LBM.U_IN, full: A.full, rest: A.rest, fade: A.fade, top: A.top,
      shade: A.shade, alpha: A.alpha, mode: mode });
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    if (mode === 0) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); }
    else { gl.disable(gl.BLEND); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    gl.disableVertexAttribArray(0);
    gl.drawArrays(gl.POINTS, 0, P.K * P.nr);
    gl.disable(gl.BLEND);
  };
  // FluidX3D's integrate_particles for one particle, here on the CPU (check())
  Tunnel.prototype.advectCPU = function (u, x, y) {
    var L = this.L, cols = L.cols, rows = L.rows, gx = x - 0.5, gy = y - 0.5, i0 = Math.floor(gx), j0 = Math.floor(gy);
    var fx = gx - i0, fy = gy - j0, ux = 0, uy = 0, bx = 0, by = 0, bd = 2;
    for (var k = 0; k < 4; k++) {
      var ci = k & 1, cj = k >> 1, i = Math.min(cols - 1, Math.max(0, i0 + ci)), j = Math.min(rows - 1, Math.max(0, j0 + cj));
      var w = (ci ? fx : 1 - fx) * (cj ? fy : 1 - fy), c = j * cols + i;
      ux += w * u[4 * c + 2]; uy += w * u[4 * c + 3];
      if (this.solid[c]) { bx += 0.5 - ci; by += 0.5 - cj; bd = Math.min(bd, Math.hypot(fx - ci, fy - cj)); }
    }
    var bl = Math.hypot(bx, by);
    if (bd < 1 && bl > 0) { var l = Math.hypot(ux, uy); ux += l * bx / bl; uy += l * by / bl; }
    var px = x + ux, py = y + uy;
    return [px, py, (px < cols - 1 && py > 0 && py < rows) ? 1 : 0];
  };
  // the cells' lines: mode 0 on screen; 1 and 2 into the check target
  Tunnel.prototype.needles = function (mode, fb, w, h) {
    var T = this, L = T.L, gl = T.gl, A = T.air, G = T.grid;
    T.bind(T.ndl, { C: L.cur.C, S: L.S }, {
      grid: [L.cols, L.rows], plot: [T.sz.w, T.sz.h], cell: [T.sx, T.sy], ng: [G.cols.length, G.rows.length],
      pad: L.pad, ytop: L.ytop, half_: G.half, uref: LBM.U_IN, full: A.full, rest: A.rest, fade: A.fade, top: A.top, shade: A.shade,
      alpha: A.alpha, mode: mode });
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (mode === 0) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); } else gl.disable(gl.BLEND);
    gl.bindBuffer(gl.ARRAY_BUFFER, T.nbuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(0, 1);
    var N = G.cols.length * G.rows.length;
    if (mode === 0) gl.drawArraysInstanced(gl.LINES, 0, 2, N); else gl.drawArraysInstanced(gl.POINTS, 0, 1, N);
    gl.vertexAttribDivisor(0, 0);
    gl.disable(gl.BLEND);
  };
  Tunnel.prototype.draw = function () {
    this.needles(0, null, this.cv.width, this.cv.height);
    this.particles(0, null, this.cv.width, this.cv.height);
    this.frames++;
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
  // the wall table on the CPU (for the paths, the overlay and the proof)
  Tunnel.prototype.band = function (table, v) {
    for (var i = 0; i < table.length; i++) if (v < table[i][0]) return i;
    return table.length - 1;
  };
  // Cp = (p - p_inf) / (rho U^2 / 2), with p = rho/3 (lattice units) and
  // p_inf the free stream's own pressure at the same column (the same trading
  // day): the mean over the open air above and below the plot, away from the
  // cone. Against the free stream passing it, a path reads positive where the
  // air runs into it and negative where the air pulls away.
  Tunnel.prototype.cp = function (rho, col) { return (rho - this.rhoInf[col]) / 3 / (0.5 * LBM.U_IN * LBM.U_IN); };
  Tunnel.prototype.reference = function (u) {
    var L = this.L, ref = this.rhoInf = new Float64Array(L.cols), band = Math.max(1, L.py >> 1);
    for (var c = 0; c < L.cols; c++) {
      var s = 0, n = 0;
      for (var b = 1; b <= band; b++) [b, L.rows - 1 - b].forEach(function (r) { var j = r * L.cols + c; if (!this.solid[j]) { s += u[4 * j + 1]; n++; } }, this);
      ref[c] = n ? s / n : 1;
    }
  };
  // the pressure on a path at one of its points: the wall cells within a cell
  // and a half of it that moving air touches, their highest Cp (the side the
  // flow hits); null where only still air touches it (drawn charcoal)
  Tunnel.prototype.pathCp = function (u, X, Y) {
    var L = this.L, cx = X / this.sx + L.pad, cy = L.ytop - Y / this.sy, best = null;
    for (var dy = -2; dy <= 2; dy++) for (var dx = -2; dx <= 2; dx++) {
      var xx = Math.floor(cx) + dx, yy = Math.floor(cy) + dy;
      if (xx < 0 || yy < 0 || xx >= L.cols || yy >= L.rows) continue;
      if ((xx + 0.5 - cx) * (xx + 0.5 - cx) + (yy + 0.5 - cy) * (yy + 0.5 - cy) > 2.25) continue;
      var j = yy * L.cols + xx;
      if (this.solid[j] && u[4 * j + 1] > 0) { var v = this.cp(u[4 * j + 1], xx); if (best === null || v > best) best = v; }
    }
    return best;
  };
  // the velocity at a point (cell units, y up): the same bilinear mix of fluid
  // nodes as the line shader; null where under half the weight is fluid
  Tunnel.prototype.vel = function (u, px, py) {
    var L = this.L, cols = L.cols, rows = L.rows, gx = px - 0.5, gy = py - 0.5, i0 = Math.floor(gx), j0 = Math.floor(gy);
    var fx = gx - i0, fy = gy - j0, vx = 0, vy = 0, wt = 0;
    for (var k = 0; k < 4; k++) {
      var i = Math.min(cols - 1, Math.max(0, i0 + (k & 1))), j = Math.min(rows - 1, Math.max(0, j0 + (k >> 1)));
      var w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy), c = j * cols + i;
      if (this.solid[c]) continue;
      vx += w * u[4 * c + 2]; vy += w * u[4 * c + 3]; wt += w;
    }
    return { wt: wt, v: wt >= 0.5 ? [vx / wt, vy / wt] : null };
  };

  // The mesh of the drawn grid (trading-day columns, return rows) and the
  // domain's 1px box: fixed per cone.
  Tunnel.prototype.layers = function () {
    var T = this, sz = T.sz, dpr = T.dpr, G = T.grid;
    [T.mesh, T.flow, T.over].forEach(function (c) {
      var w = Math.round(sz.w * dpr), h = Math.round(sz.h * dpr);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    });
    if (!G || !G.cols.length || !G.rows.length) return;
    var x = T.mesh.getContext('2d'), snap = function (v) { return Math.round(v * dpr) / dpr + 0.5 / dpr; };
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    x.strokeStyle = 'rgba(63,59,53,0.08)'; x.lineWidth = 1 / dpr; x.beginPath();
    G.cols.forEach(function (c) { var X = snap(T.xa.l2p(c - G.nd / 2)); x.moveTo(X, 0); x.lineTo(X, sz.h); });
    var last = G.cols[G.cols.length - 1] + G.nd / 2, XL = snap(T.xa.l2p(last)); x.moveTo(XL, 0); x.lineTo(XL, sz.h);
    G.rows.forEach(function (r) { var Y = snap(T.ya.l2p(r - G.dr / 2)); x.moveTo(0, Y); x.lineTo(sz.w, Y); });
    var top = G.rows[G.rows.length - 1] + G.dr / 2, YT = snap(T.ya.l2p(top)); x.moveTo(0, YT); x.lineTo(sz.w, YT);
    x.stroke();
    x.strokeStyle = '#3F3B35'; x.lineWidth = 1; x.strokeRect(0.5, 0.5, sz.w - 1, sz.h - 1);   // the domain
    T.drawOverlay();
  };

  // The data paths through their exact points, each segment in its pressure
  // step, and the solver's state line. Reads the computed state back.
  Tunnel.prototype.paint = function (cbuf) {
    var T = this, sz = T.sz, dpr = T.dpr, u = cbuf === undefined ? T.readState() : cbuf, x = T.flow.getContext('2d');
    if (u) T.reference(u);
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    var W2 = T.wallSteps, segs = [];
    x.lineWidth = 2; x.lineCap = 'butt';                // square ends: a short segment keeps its own colour
    for (var q = 0; q < T.lv.length; q++) for (var d = 0; d + 1 < T.days.length; d++) {
      var X0 = T.xa.l2p(T.days[d]), Y0 = T.ya.l2p(T.lv[q][d]), X1 = T.xa.l2p(T.days[d + 1]), Y1 = T.ya.l2p(T.lv[q][d + 1]);
      var cpv = u ? T.pathCp(u, (X0 + X1) / 2, (Y0 + Y1) / 2) : null, wb = cpv === null ? 0 : T.band(W2, cpv);
      segs.push([q, d, wb, cpv]);
      x.strokeStyle = W2[wb][1]; x.beginPath(); x.moveTo(X0, Y0); x.lineTo(X1, Y1); x.stroke();
    }
    T.segs = segs;
    if (u) T.lastState = u;
    T.hud(x);
    if (api.overlay && u) T.drawOverlay(u);
  };
  // The live reading, without stopping the page: the state is copied into a
  // pixel-pack buffer on the GPU and fetched once a fence says the copy is
  // done (a plain readPixels waits for every queued step and freezes the
  // page while it waits).
  Tunnel.prototype.readAsync = function () {
    var T = this, L = T.L, gl = T.gl, n = L.cols * L.rows * 4;
    if (T.rb) return;
    if (!T.pbo || T.pboN !== n) {
      if (T.pbo) gl.deleteBuffer(T.pbo);
      T.pbo = gl.createBuffer(); T.pboN = n;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, T.pbo); gl.bufferData(gl.PIXEL_PACK_BUFFER, n * 4, gl.STREAM_READ);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, L.cur.fb);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, T.pbo);
    gl.readPixels(0, 0, L.cols, L.rows, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    T.rb = { sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), n: n, key: T.key };
    gl.flush();
  };
  Tunnel.prototype.pollRead = function () {
    var T = this, gl = T.gl, rb = T.rb;
    if (!rb) return;
    if (gl.clientWaitSync(rb.sync, 0, 0) === gl.TIMEOUT_EXPIRED) return;
    gl.deleteSync(rb.sync); T.rb = null;
    if (rb.key !== T.key || rb.n !== T.pboN) return;       // a new cone since: the reading is stale
    if (!T.aBuf || T.aBuf.length !== rb.n) T.aBuf = new Float32Array(rb.n);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, T.pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, T.aBuf);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    T.paint(T.aBuf);
  };
  // FluidX3D prints its solver state on the frame; so does this
  Tunnel.prototype.hud = function (x) {
    var T = this, L = T.L, n = T.days.length, parts = ['D2Q9 lattice Boltzmann', (L.cols * L.rows).toLocaleString('en-US') + ' cells',
      'step ' + T.steps.toLocaleString('en-US'), 'trading day ' + T.days[T.dayAt(T.steps)] + ' of ' + T.days[n - 1]], lines = [], line = '';
    x.font = '10px ui-monospace, Menlo, Consolas, monospace'; x.textBaseline = 'middle';
    parts.forEach(function (pt) {                              // wrapped to the plot's width
      var next = line ? line + '  ' + pt : pt;
      if (line && x.measureText(next).width > T.sz.w - 16) { lines.push(line); line = pt; } else line = next;
    });
    lines.push(line);
    T.hudW = Math.max.apply(null, lines.map(function (l) { return x.measureText(l).width; })) + 8;
    T.hudH = 4 + 13 * lines.length + 2;
    x.fillStyle = 'rgba(237,233,227,0.92)'; x.fillRect(4, 4, T.hudW, T.hudH - 4);
    x.fillStyle = '#3F3B35';
    lines.forEach(function (l, i) { x.fillText(l, 8, 11 + 13 * i); });
  };

  // Verification Overlay: the solver's own numbers on the picture.
  Tunnel.prototype.drawOverlay = function (cbuf) {
    var T = this, x = T.over.getContext('2d'), sz = T.sz, dpr = T.dpr, L = T.L, G = T.grid;
    if (!L || !G || !G.list.length) return;
    x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, sz.w, sz.h);
    if (!api.overlay) return;
    var u = cbuf || T.lastState || T.readState(), cols = L.cols, rows = L.rows, sx = T.sx, sy = T.sy, pad = L.pad;
    x.font = '9px ui-monospace, Menlo, Consolas, monospace'; x.textBaseline = 'middle';
    var label = function (s, X, Y, col) {
      var w = x.measureText(s).width + 4;
      x.fillStyle = 'rgba(237,233,227,0.92)'; x.fillRect(X, Y - 6, w, 12);
      x.fillStyle = col; x.fillText(s, X + 2, Y);
    };
    // the solid lattice cells' outline: the walls exactly as the solver sees them (FluidX3D's VIS_FLAG_LATTICE)
    x.strokeStyle = 'rgba(28,26,23,0.55)'; x.lineWidth = 1 / dpr; x.beginPath();
    for (var gy = L.py; gy < L.ytop; gy++) for (var gx = pad; gx < pad + L.pc; gx++) {
      if (!T.solid[gy * cols + gx]) continue;
      var X0 = (gx - pad) * sx, X1 = X0 + sx, Y1 = (L.ytop - gy) * sy, Y0 = Y1 - sy;
      if (gx + 1 >= cols || !T.solid[gy * cols + gx + 1]) { x.moveTo(X1, Y0); x.lineTo(X1, Y1); }
      if (!T.solid[gy * cols + gx - 1]) { x.moveTo(X0, Y0); x.lineTo(X0, Y1); }
      if (gy + 1 >= rows || !T.solid[(gy + 1) * cols + gx]) { x.moveTo(X0, Y0); x.lineTo(X1, Y0); }
      if (gy < 1 || !T.solid[(gy - 1) * cols + gx]) { x.moveTo(X0, Y1); x.lineTo(X1, Y1); }
    }
    x.stroke();
    // arrowheads and printed speeds (x inflow) on a sample of the grid's cells
    var nc = G.cols.length, A = T.air;
    for (var j = Math.floor(LBM.LABEL_EVERY / 2); j < G.rows.length; j += LBM.LABEL_EVERY) {
      for (var i = Math.floor(LBM.LABEL_EVERY / 2); i < nc; i += LBM.LABEL_EVERY) {
        var c = G.list[j * nc + i], k = 4 * (j * nc + i), vv = T.vel(u, G.data[k], G.data[k + 1]).v;
        if (!vv || c.Y < (T.hudH || 19) + 46) continue;        // clear of the state lines
        var s = Math.sqrt(vv[0] * vv[0] + vv[1] * vv[1]) / LBM.U_IN;
        if (airColour(A, s)[3] <= 0) continue;                 // still air: no line, no label
        var ang = Math.atan2(-vv[1], vv[0]), tx = c.X + Math.cos(ang) * G.half, ty = c.Y + Math.sin(ang) * G.half;
        x.fillStyle = '#1C1A17'; x.beginPath(); x.moveTo(tx, ty);
        x.lineTo(tx + Math.cos(ang + 2.6) * 3.2, ty + Math.sin(ang + 2.6) * 3.2);
        x.lineTo(tx + Math.cos(ang - 2.6) * 3.2, ty + Math.sin(ang - 2.6) * 3.2);
        x.fill();
        label(s.toFixed(2), c.X + 3, c.Y + 7, '#1C1A17');
      }
    }
    var fastest = { s: -1 };
    for (var m = 0; m < cols * rows; m++) {
      var mr = Math.floor(m / cols);
      if (T.solid[m] || m % cols < pad || m % cols >= pad + L.pc || mr < L.py || mr >= L.ytop) continue;   // the plot's own cells
      var sp = Math.sqrt(u[4 * m + 2] * u[4 * m + 2] + u[4 * m + 3] * u[4 * m + 3]) / LBM.U_IN;
      if (sp > fastest.s) fastest = { s: sp, j: m };
    }
    // the highest pressures on the data paths: where the flow hits them hardest (the red)
    var wallTop = [], marked = [];
    (T.segs || []).forEach(function (sg) { if (sg[3] !== null) wallTop.push([sg[3], sg[0], sg[1]]); });
    wallTop.sort(function (a, b) { return b[0] - a[0]; });
    for (var t = 0; t < wallTop.length && marked.length < 5; t++) {
      var pq = wallTop[t][1], pd = wallTop[t][2];
      var wx = (T.xa.l2p(T.days[pd]) + T.xa.l2p(T.days[pd + 1])) / 2, wy = (T.ya.l2p(T.lv[pq][pd]) + T.ya.l2p(T.lv[pq][pd + 1])) / 2;
      if (marked.some(function (mk2) { return Math.abs(mk2[0] - wx) < 60 && Math.abs(mk2[1] - wy) < 18; })) continue;
      marked.push([wx, wy]);
      x.strokeStyle = '#B00020'; x.lineWidth = 1; x.strokeRect(wx - 3.5, wy - 3.5, 7, 7);
      label('Cp ' + wallTop[t][0].toFixed(2) + ' on ' + LEVELS[pq] + ', day ' + T.days[pd], wx + 6, wy - 8, '#9E0B1F');
    }
    if (fastest.s > 0) {
      var fx = (fastest.j % cols - pad + 0.5) * sx, fy = (L.ytop - Math.floor(fastest.j / cols) - 0.5) * sy;
      x.strokeStyle = '#0B5F8A'; x.lineWidth = 1; x.beginPath(); x.arc(fx, fy, 4, 0, 6.2832); x.stroke();
      label('max |u| ' + fastest.s.toFixed(2) + ' U', fx + 6, fy + 8, '#0B4F73');
    }
    var nu = (LBM.TAU - 0.5) / 3, parts = [
      'grid cell = ' + G.nd + ' trading day' + (G.nd > 1 ? 's' : '') + ' x ' + G.dr + '% return',
      'lattice ' + L.pc + ' x ' + L.rp + ' on the plot (+' + (cols - L.pc) + ' columns, +' + (rows - L.rp) + ' rows off it)',
      T.stepsPerDay.toFixed(2) + ' steps per trading day', 'inflow ' + (T.inflowAt(T.steps) / LBM.U_IN).toFixed(3) + ' U',
      'tau ' + LBM.TAU + ' + subgrid', 'Re ' + Math.round(LBM.U_IN * L.pc / nu).toLocaleString('en-US')];
    var line = '', ly = (T.hudH || 19) + 8;                  // wrapped to the plot's width, under the state line
    parts.forEach(function (pt) {
      var next = line ? line + '  ·  ' + pt : pt;
      if (line && x.measureText(next).width > sz.w - 16) { label(line, 4, ly, '#1C1A17'); ly += 13; line = pt; } else line = next;
    });
    if (line) label(line, 4, ly, '#1C1A17');
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
    var R = W.__melionaReveal;
    if (gliding() || now - inputAt < LBM.STILL_MS || (R && R.moving && R.moving() > 0)) { T.lastFrame = 0; T.wake(); return; }
    if (T.still) {
      var goal = Math.round(LBM.STILL_TRANSITS * T.L.pc / LBM.U_IN);
      for (var i = 0; i < LBM.STILL_CHUNK && T.steps < goal; i++) T.step();
      if (T.steps >= goal) { T.draw(); T.paint(); T.done = true; }
    } else {
      // a card that cannot keep up shows fewer steps per frame (never changes what a step computes)
      // ...and a card that cannot keep up even then keeps one still frame:
      // the page's own motion comes first.
      if (T.lastFrame && now - T.lastFrame > 24) {
        if (++T.slow >= 6) {
          if (T.spf === LBM.STEPS_MIN) { T.mode = 'lbm-still'; T.done = true; T.draw(); T.readAsync(); T.settle(); return; }
          T.spf = Math.max(LBM.STEPS_MIN, Math.round(T.spf * 0.7)); T.slow = 0;
        }
      } else T.slow = 0;
      T.lastFrame = now;
      for (var k = 0; k < T.spf; k++) T.step();
      T.draw();
      T.pollRead();
      if (T.frames % LBM.WALL_EVERY === 0) T.readAsync();
    }
    T.wake();
  };

  // a stopped tunnel's last reading still lands on its paths, without waiting
  Tunnel.prototype.settle = function () {
    var T = this;
    if (!T.rb) return;
    T.pollRead();
    if (T.rb) W.setTimeout(function () { T.settle(); }, 50);
  };

  // ---- the proof, on the rendered output -------------------------------------------
  // (1) The cells' lines: the line shader writes, one pixel per grid cell, the
  //     colour it draws (mode 1) and the direction it points (mode 2); both are
  //     read back with the computed state and compared with FluidX3D's scale
  //     applied to the cell's own velocity, computed again here on the CPU.
  // (2) The grid is the chart's: every line's centre sits on its trading day
  //     and return level (the axes' own pixel mapping).
  // (3) The data paths: at the middle of every drawn segment, read the screen
  //     pixel and compare it with the pressure step of that segment's computed
  //     Cp (segments where two paths meet, as at day 1, are skipped).
  // (4) Red: no red pixel on the air's lines, none on the path layer farther
  //     than 3px from a data path, and the highest pressure on the paths is
  //     painted red when it reaches the red step.
  // (5) The walls are the data: each path's point every 5 trading days sits on
  //     (or within one cell of) a wall cell.
  function isRed(r, g, b) { return r > 180 && g < 80 && b < 90; }
  Tunnel.prototype.verify = function (out) {
    var T = this, gl = T.gl, L = T.L, G = T.grid, A = T.air;
    if (!L) return;
    var u = T.readState().slice();
    T.paint(u);
    var nc = G.cols.length, nr = G.rows.length, N = nc * nr;
    if (!L.chk || L.chk.w !== nc || L.chk.h !== nr) {
      if (L.chk) { gl.deleteTexture(L.chk.t); gl.deleteFramebuffer(L.chk.fb); }
      var tx = tex(gl, nc, nr, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx, 0);
      L.chk = { t: tx, fb: fb, w: nc, h: nr };
    }
    var col = new Uint8Array(N * 4), dir = new Uint8Array(N * 4);
    T.needles(1, L.chk.fb, nc, nr); gl.bindFramebuffer(gl.FRAMEBUFFER, L.chk.fb); gl.readPixels(0, 0, nc, nr, gl.RGBA, gl.UNSIGNED_BYTE, col);
    T.needles(2, L.chk.fb, nc, nr); gl.bindFramebuffer(gl.FRAMEBUFFER, L.chk.fb); gl.readPixels(0, 0, nc, nr, gl.RGBA, gl.UNSIGNED_BYTE, dir);
    var q8 = function (v) { return Math.round(Math.min(1, Math.max(0, v)) * 255); };
    for (var j = 0; j < N; j++) {
      var k = 4 * j, r = T.vel(u, G.data[k], G.data[k + 1]);
      if (Math.abs(r.wt - 0.5) < 1e-3) continue;                 // on the fluid/wall cut-off itself: either answer is right
      out.lines++;
      var want, wdir;
      if (!r.v) { want = [null, null, null, 0]; wdir = [null, null, null, 0]; }
      else {
        var l = Math.sqrt(r.v[0] * r.v[0] + r.v[1] * r.v[1]), s = l / LBM.U_IN, c = airColour(A, s);
        want = [q8(c[0]), q8(c[1]), q8(c[2]), q8(c[3])];
        wdir = l > 0 ? [q8(r.v[0] / l * 0.5 + 0.5), q8(r.v[1] / l * 0.5 + 0.5), q8(Math.min(1, s / 4)), 255] : [null, null, null, 255];
        out.maxSpeed = Math.max(out.maxSpeed, s);
        if (want[3] > 0) out.drawn++;
      }
      var bad = false;
      for (var e = 0; e < 4; e++) {
        if (want[e] !== null && Math.abs(col[k + e] - want[e]) > 2) bad = true;
        if (wdir[e] !== null && Math.abs(dir[k + e] - wdir[e]) > 2) bad = true;
      }
      if (want[3] > 0 && isRed(want[0], want[1], want[2])) out.redOffWall++;
      if (bad) {
        out.colourMismatch++;
        if ((out.worst = out.worst || []).length < 4) out.worst.push({ cell: j, got: Array.from(col.slice(k, k + 4)), want: want,
                                                                      gotDir: Array.from(dir.slice(k, k + 4)), wantDir: wdir });
      }
      // (2) on the chart's axes: centred on its trading day and return level
      var cell = G.list[j];
      out.gridChecks++;
      if (Math.abs(T.xa.l2p(cell.day) - (G.data[k] - L.pad) * T.sx) > 0.01 || Math.abs(T.ya.l2p(cell.ret) - (L.ytop - G.data[k + 1]) * T.sy) > 0.01) out.gridMiss++;
    }
    out.cells = N; out.gridDays = G.nd; out.gridPct = G.dr;
    // (6) the particles: each one's colour against the air it is in, and one
    //     step of its motion against FluidX3D's integrate_particles recomputed
    //     here (into the spare buffer: the running state is left as it is)
    var P = L.P;
    if (P) {
      var K = P.K, NP = K * P.nr;
      if (!L.pchk) {
        var tp = tex(gl, K, P.nr, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), fbp = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbp);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tp, 0);
        L.pchk = { t: tp, fb: fbp };
      }
      var pcol = new Uint8Array(NP * 4), p0 = new Float32Array(NP * 4), p1 = new Float32Array(NP * 4);
      T.particles(1, L.pchk.fb, K, P.nr); gl.bindFramebuffer(gl.FRAMEBUFFER, L.pchk.fb); gl.readPixels(0, 0, K, P.nr, gl.RGBA, gl.UNSIGNED_BYTE, pcol);
      gl.bindFramebuffer(gl.FRAMEBUFFER, P.cur.fb); gl.readPixels(0, 0, K, P.nr, gl.RGBA, gl.FLOAT, p0);
      T.advect(P.nxt.fb, -1); gl.bindFramebuffer(gl.FRAMEBUFFER, P.nxt.fb); gl.readPixels(0, 0, K, P.nr, gl.RGBA, gl.FLOAT, p1);
      for (var pi = 0; pi < NP; pi++) {
        var b4 = 4 * pi;
        if (p0[b4 + 3] < 0.5) continue;
        out.particlesAlive++;
        var e1 = T.advectCPU(u, p0[b4], p0[b4 + 1]);
        if (Math.abs(e1[0] - p1[b4]) > 1e-3 || Math.abs(e1[1] - p1[b4 + 1]) > 1e-3 || e1[2] !== (p1[b4 + 3] > 0.5 ? 1 : 0)) {
          out.particleMoveMismatch++;
          if ((out.moveWorst = out.moveWorst || []).length < 3) out.moveWorst.push({ from: [p0[b4], p0[b4 + 1]], got: [p1[b4], p1[b4 + 1]], want: e1 });
        }
        var rp = T.vel(u, p0[b4], p0[b4 + 1]);
        if (Math.abs(rp.wt - 0.5) < 1e-3) continue;
        var PX = (p0[b4] - L.pad) * T.sx, PY = (L.ytop - p0[b4 + 1]) * T.sy;
        var sp2 = rp.v ? Math.hypot(rp.v[0], rp.v[1]) / LBM.U_IN : 0, c3 = airColour(A, sp2);
        var vis = !!rp.v && PX >= 0 && PX <= T.sz.w && PY >= 0 && PY <= T.sz.h;
        var wantP = [q8(c3[0]), q8(c3[1]), q8(c3[2]), vis ? q8(c3[3]) : 0];
        if (vis && wantP[3] > 0) out.particlesShown++;
        var badP = false;
        for (var e2 = 0; e2 < 4; e2++) if (Math.abs(pcol[b4 + e2] - wantP[e2]) > 2) badP = true;
        if (badP) {
          out.particleColourMismatch++;
          if ((out.pWorst = out.pWorst || []).length < 3) out.pWorst.push({ got: Array.from(pcol.slice(b4, b4 + 4)), want: wantP });
        }
      }
    }
    // (4) the air's lines on screen, drawn now and read back before the frame is shown
    T.needles(0, null, T.cv.width, T.cv.height);
    T.particles(0, null, T.cv.width, T.cv.height);
    var cw0 = T.cv.width, ch0 = T.cv.height, scr = new Uint8Array(cw0 * ch0 * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, cw0, ch0, gl.RGBA, gl.UNSIGNED_BYTE, scr);
    for (var p = 0; p < cw0 * ch0; p++) {
      var a = scr[4 * p + 3];
      if (a > 40 && isRed(scr[4 * p] * 255 / a, scr[4 * p + 1] * 255 / a, scr[4 * p + 2] * 255 / a)) out.redOffWall++;
      if (a > 0) out.inkPixels++;
    }
    // (3) and (4) on the path layer
    var R = T.wallSteps, red = R.length - 1, redRGB = hex(R[red][1]);
    var dpr = T.dpr, cw = T.flow.width, chh = T.flow.height, img = T.flow.getContext('2d').getImageData(0, 0, cw, chh).data;
    var pts = [];
    for (var qq = 0; qq < T.lv.length; qq++) for (var dd = 0; dd < T.days.length; dd++) pts.push([T.xa.l2p(T.days[dd]), T.ya.l2p(T.lv[qq][dd]), qq]);
    var near = function (X, Y, rr, skipQ) {
      for (var z = 0; z < pts.length; z++) if (pts[z][2] !== skipQ && Math.abs(pts[z][0] - X) <= rr && Math.abs(pts[z][1] - Y) <= rr) return true;
      return false;
    };
    var hudBox = function (X, Y) { return X < (T.hudW || 0) + 10 && Y < (T.hudH || 0) + 4; };        // the state line's own box
    var peak = null;
    out.cpByPath = {};
    (T.segs || []).forEach(function (sg) { if (sg[3] !== null) out.cpByPath[LEVELS[sg[0]]] = Math.max(out.cpByPath[LEVELS[sg[0]]] ?? -1e9, +sg[3].toFixed(3)); });
    (T.segs || []).forEach(function (sg) {
      var q2 = sg[0], d2 = sg[1];
      var X = (T.xa.l2p(T.days[d2]) + T.xa.l2p(T.days[d2 + 1])) / 2, Y = (T.ya.l2p(T.lv[q2][d2]) + T.ya.l2p(T.lv[q2][d2 + 1])) / 2;
      if (sg[3] !== null && (!peak || sg[3] > peak.cp)) peak = { cp: sg[3], X: X, Y: Y, b: sg[2] };
      if (near(X, Y, 6, q2) || X < 3 || Y < 3 || X > T.sz.w - 3 || Y > T.sz.h - 3 || hudBox(X, Y)) return;   // shared pixels, the domain's edge, the state line
      var c2 = hex(R[sg[2]][1]), cyp = Math.floor(Y * dpr), cxp = Math.floor(X * dpr), found = false;
      for (var oy = -Math.ceil(dpr); oy <= Math.ceil(dpr) && !found; oy++) for (var ox = -Math.ceil(dpr); ox <= Math.ceil(dpr) && !found; ox++) {
        var kk = 4 * ((cyp + oy) * cw + cxp + ox);   // the line's core pixel, within 1px
        if (Math.abs(img[kk] - c2[0]) <= 8 && Math.abs(img[kk + 1] - c2[1]) <= 8 && Math.abs(img[kk + 2] - c2[2]) <= 8 && img[kk + 3] > 230) found = true;
      }
      out.wallChecks++;
      if (!found) {
        out.wallMismatch++;
        if ((out.wallWorst = out.wallWorst || []).length < 3) out.wallWorst.push({ path: LEVELS[q2], day: T.days[d2], cp: sg[3], want: c2 });
      }
    });
    for (var yy = 0; yy < chh; yy += 2) for (var xx = 0; xx < cw; xx += 2) {
      var k3 = 4 * (yy * cw + xx);
      if (img[k3 + 3] > 200 && Math.abs(img[k3] - redRGB[0]) < 30 && Math.abs(img[k3 + 1] - redRGB[1]) < 30 && Math.abs(img[k3 + 2] - redRGB[2]) < 30 &&
          !near(xx / dpr, yy / dpr, 3, -1)) out.redOffWall++;
    }
    if (peak) {
      out.maxCp = +peak.cp.toFixed(3);
      var found2 = false;
      for (var o2 = -1; o2 <= 1 && !found2; o2++) for (var o3 = -1; o3 <= 1 && !found2; o3++) {
        var k4 = 4 * ((Math.floor(peak.Y * dpr) + o2) * cw + Math.floor(peak.X * dpr) + o3);
        if (Math.abs(img[k4] - redRGB[0]) <= 8 && Math.abs(img[k4 + 1] - redRGB[1]) <= 8 && Math.abs(img[k4 + 2] - redRGB[2]) <= 8) found2 = true;
      }
      out.peakIsRed = peak.cp < R[red - 1][0] ? null : (peak.b === red && found2);
    }
    // (5)
    for (var k5 = 0; k5 < T.lv.length; k5++) for (var i5 = 0; i5 < T.days.length; i5 += 5) {
      var cx = Math.floor(T.xa.l2p(T.days[i5]) / T.sx + L.pad), cy = Math.floor(L.ytop - T.ya.l2p(T.lv[k5][i5]) / T.sy), hit = false;
      for (var dy = -1; dy <= 1 && !hit; dy++) for (var dx = -1; dx <= 1 && !hit; dx++) {
        var x6 = cx + dx, y6 = cy + dy;
        if (x6 >= 0 && y6 >= 0 && x6 < L.cols && y6 < L.rows && T.solid[y6 * L.cols + x6]) hit = true;
      }
      out.obstacleChecks++;
      if (!hit) out.obstacleMiss++;
    }
    out.maxSpeed = +out.maxSpeed.toFixed(3);
    out.lattice = L.cols * L.rows;
    out.steps = T.steps; out.day = T.days[T.dayAt(T.steps)]; out.stepsPerDay = +T.stepsPerDay.toFixed(3);
    out.inflow = +(T.inflowAt(T.steps) / LBM.U_IN).toFixed(4); out.rhoInfOutlet = +T.rhoInf[L.cols - 2].toFixed(5);
  };
  api.check = function () {
    prune();
    var out = { mode: null, lines: 0, drawn: 0, cells: 0, colourMismatch: 0, gridChecks: 0, gridMiss: 0, redOffWall: 0,
                particlesAlive: 0, particlesShown: 0, particleMoveMismatch: 0, particleColourMismatch: 0,
                obstacleMiss: 0, obstacleChecks: 0, wallChecks: 0, wallMismatch: 0, inkPixels: 0, lattice: 0,
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
  // Nothing is built while the page loads: a cone is watched, and its tunnel
  // (the WebGL context, the programs, the lattice) is built only once the
  // cone comes within a screen of the view, when the browser is idle and no
  // button's glide or block's arrival is moving - the page's own motion
  // first (John, 2026-11-04: the build at load froze the arrivals and the
  // buttons).
  var near = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { near.unobserve(e.target); build(e.target); } });
  }, { rootMargin: '100% 0px 100% 0px' });
  function build(gd) {
    var go = function () {
      if (!gd.isConnected || gd.__wind) return;
      if (busy()) { setTimeout(function () { build(gd); }, 300); return; }
      var cv = mk('wind-tunnel'), g = glContext(cv);
      // No GPU, no flow: a software renderer (SwiftShader, llvmpipe) runs the
      // browser's own compositing on the same emulated GPU, and the solve froze
      // the whole page on CI's machines (Live audit, 2026-11-04). The chart is
      // complete without it. tools/flow_check.mjs proves the solve in CI.
      if (!g.gl || g.soft) {
        api.status = g.gl ? 'software renderer: flow not run' : 'no WebGL2: flow not run';
        var lose = g.gl && g.gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
        return;
      }
      new Tunnel(gd, reduced, cv, g.gl);
    };
    if (W.requestIdleCallback) W.requestIdleCallback(go, { timeout: 1500 }); else setTimeout(go, 200);
  }
  function find() {
    prune();
    var plots = D.querySelectorAll('.js-plotly-plot');
    for (var i = 0; i < plots.length; i++) {
      var gd = plots[i];
      if (gd.__wind) { if (!gd.__wind.compiling && !gd.__wind.cv.isConnected) gd.__wind.read(); continue; }
      if (gd.__windSeen) continue;
      var m = metaOf(gd);
      if (m && m.meliona === 'cone') { gd.__windSeen = true; near.observe(gd); }
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
