// The cone's flow (static/windtunnel.js), proven on the real page.
//
//   node tools/flow_check.mjs <url> [outdir] [steps]
//
// The page runs its lattice Boltzmann solve only on a hardware GPU; a CI
// runner has a software one, where the page leaves the flow out. Here, and
// only here, the browser is told it has a hardware GPU (WEBGL_debug_renderer_
// info's renderer string), the solver is held off its own frame loop and
// stepped a fixed number of times by this script, and the page's own check()
// compares the picture with the computed state: every lattice node's colour
// against its computed speed's step, every path segment's pixel against its
// pressure step, no red pixel off a data path, every path point on a wall, the
// highest path pressure red, and the walls on THIS cone's 5th/95th paths. Then
// the same after a volatility shock (new data, new walls). Exit 1 on any FAIL.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, out = 'flow-shots', stepsArg = '1800'] = process.argv;
const STEPS = +stepsArg;
const fs = await import('node:fs');
fs.mkdirSync(out, { recursive: true });
let fails = 0;
const ok = (c, name, note = '') => { if (!c) fails++; console.log(`${c ? 'PASS' : 'FAIL'}  ${name}${note ? ' - ' + note : ''}`); };
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.addInitScript(() => {
  const g = WebGL2RenderingContext.prototype.getParameter;
  WebGL2RenderingContext.prototype.getParameter = function (k) { return k === 0x9246 ? 'ANGLE (flow_check hardware stand-in)' : g.call(this, k); };
});
const errs = []; p.on('pageerror', e => errs.push(String(e)));
await p.goto(url); await p.waitForSelector('.verdict-number', { timeout: 300000 });
const cone = () => p.waitForFunction(() => [...document.querySelectorAll('.js-plotly-plot')].some(g => (g._fullLayout?.meta || {}).meliona === 'cone'), null, { timeout: 180000 });
const centre = () => p.evaluate(() => { const g = [...document.querySelectorAll('.js-plotly-plot')].find(g => (g._fullLayout?.meta || {}).meliona === 'cone');
  const sc = document.querySelector('[data-testid="stMain"]'); sc.style.scrollBehavior = 'auto'; sc.scrollTop += g.getBoundingClientRect().top - 300; });
const chart = () => p.evaluate(() => { const g = [...document.querySelectorAll('.js-plotly-plot')].find(g => (g._fullLayout?.meta || {}).meliona === 'cone');
  const d = g._fullData; return { p5: d[1].y[d[1].y.length - 1], p95: d[0].y[d[0].y.length - 1] }; });
async function run(tag) {
  await cone(); await centre(); await p.waitForTimeout(800); await centre();
  await p.evaluate(() => window.__melionaReveal?.revealAll());
  await p.waitForFunction(() => { const t = window.__melionaWind?.tunnels; const T = t && t[t.length - 1];
    if (!T || !T.L || !T.gd.isConnected) return false; T.visible = false; T.wake(); return true; }, null, { timeout: 120000 });
  for (let done = 0; done < STEPS; done += 100)                     // in slices, so the page stays responsive
    await p.evaluate(n => { const t = window.__melionaWind.tunnels, T = t[t.length - 1]; for (let i = 0; i < n; i++) T.step(); }, Math.min(100, STEPS - done));
  await p.evaluate(() => { const t = window.__melionaWind.tunnels, T = t[t.length - 1]; T.draw(); T.streams(); });
  const w = await p.evaluate(() => window.__melionaWind.check()), c = await chart();
  console.log(`INFO  ${tag}: ${JSON.stringify({ mode: w.mode, steps: w.steps, day: w.day, stepsPerDay: w.stepsPerDay, maxSpeed: w.maxSpeed, maxCp: w.maxCp })}`);
  ok(w.mode === 'lbm' && w.steps >= STEPS, `${tag}: the solver ran ${STEPS} steps`, `${w.steps}`);
  ok(Math.abs(w.p5End - c.p5) < 1e-9 && Math.abs(w.p95End - c.p95) < 1e-9, `${tag}: the walls are this cone's paths`, `5th ${w.p5End?.toFixed(3)} / 95th ${w.p95End?.toFixed(3)}`);
  ok(w.obstacleMiss === 0, `${tag}: every path point sits on a wall`, `${w.obstacleMiss} of ${w.obstacleChecks}`);
  ok(w.particles > 1000 && w.colourMismatch === 0, `${tag}: every node's colour is its computed speed's step`, `${w.colourMismatch} of ${w.particles}`);
  ok(w.wallChecks > 500 && w.wallMismatch === 0, `${tag}: every path segment's colour is its computed pressure's step`, `${w.wallMismatch} of ${w.wallChecks}`);
  ok(w.redOffWall === 0, `${tag}: red only on the data paths`, `${w.redOffWall} red pixels off them`);
  ok(w.peakIsRed !== false, `${tag}: the highest path pressure is painted red`, `Cp ${w.maxCp}`);
  ok(w.maxSpeed > 0.5, `${tag}: the air is moving`, `max ${w.maxSpeed} x inflow`);
  const box = await p.locator('[data-testid="stPlotlyChart"]').filter({ has: p.locator('canvas.wind-tunnel') }).first().boundingBox();
  await p.screenshot({ path: `${out}/${tag}.png`, clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
  await p.locator('button.wt-verify').click(); await p.waitForTimeout(200);
  await p.screenshot({ path: `${out}/${tag}-overlay.png`, clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
  await p.locator('button.wt-verify').click();
}
await run('baseline');
await p.locator('summary').filter({ hasText: 'Stress test' }).first().click(); await p.waitForTimeout(1200);
const vol = p.locator('[data-testid="stSlider"]').filter({ hasText: 'Volatility shock' }).first();
const vb = await vol.boundingBox(); await p.mouse.click(vb.x + vb.width * 0.5, vb.y + vb.height - 14);
await p.waitForTimeout(6000);
await run('volatility-shock');
ok(errs.length === 0, 'no script errors', errs.slice(0, 3).join(' | '));
await b.close();
console.log(fails ? `\n${fails} FAIL` : '\nevery flow check passes');
process.exit(fails ? 1 : 0);
