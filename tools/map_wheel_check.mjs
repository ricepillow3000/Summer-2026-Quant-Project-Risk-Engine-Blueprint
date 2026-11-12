// The mouse wheel glides over the Risk Topology map as it does everywhere else.
//
//   node tools/map_wheel_check.mjs <url>
//
// John, 2026-11-06: "Throughout the whole site, the scrolling was nice and
// smooth, but once it got to the map, it went back to the static state."
// The map is a same-origin frame, and a wheel over a frame goes to the
// frame's window, never the page's: static/reveal.js's tow never saw it, so
// the page took the browser's step scroll there (one notch: 100px in a single
// frame, against ~60 frames anywhere else). reveal.js now listens in every
// frame's window (hookFrame) and tows the page from there.
//
// At 1440x900, 1280x720 and 1920x1080, with the real mouse wheel:
//   notch   one notch over the map moves the page in many small frames
//   run     fourteen notches that carry the map under a still pointer never
//           move the page more than 40px in one frame
//   panel   a map panel that scrolls on its own still takes the wheel first
//           (checked where one overflows)
// Exit code 1 on any failure.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
let pw;
try {
  pw = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
} catch {
  try {
    const at = createRequire(import.meta.url).resolve('playwright', { paths: [process.env.NODE_PATH || ''] });
    pw = await import(pathToFileURL(at).href);
  } catch {
    console.error('playwright not installed');
    process.exit(3);
  }
}
const { chromium } = pw.chromium ? pw : pw.default;
const [,, url] = process.argv;
if (!url) { console.error('usage: node tools/map_wheel_check.mjs <url>'); process.exit(2); }
const SIZES = [[1440, 900], [1280, 720], [1920, 1080]];
const MAP = '.st-key-risk-map iframe';
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
let fails = 0;
const ok = (pass, what) => { console.log(`${pass ? 'PASS' : 'FAIL'}  ${what}`); if (!pass) fails++; };

for (const [W, H] of SIZES) {
  const p = await (await b.newContext({ viewport: { width: W, height: H } })).newPage();
  await p.goto(url);
  await p.waitForSelector('.meliona-footer', { timeout: 300000 });
  await p.waitForTimeout(1500);
  await p.evaluate(() => window.__melionaReveal && window.__melionaReveal.revealAll());
  const tab = p.getByRole('tab', { name: /risk topology/i }).first();
  await tab.scrollIntoViewIfNeeded(); await tab.click();
  await p.waitForSelector(MAP, { timeout: 120000 });
  await p.waitForTimeout(2500);
  const place = (offset) => p.evaluate(([sel, off]) => {
    const m = document.querySelector(sel), sc = document.querySelector('[data-testid="stMain"]');
    sc.style.scrollBehavior = 'auto'; sc.scrollTop += m.getBoundingClientRect().top - off;
  }, [MAP, offset]);
  const record = (ms) => p.evaluate(([ms, sel]) => {
    const sc = document.querySelector('[data-testid="stMain"]'); window.__tr = []; window.__tx = []; const t0 = performance.now();
    (function f(now) {
      window.__tr.push(sc.scrollTop);
      // what else changed on the frame of a big step: the page's height, the
      // map frame's window (a rerun rewrites its srcdoc) and whether it still has the tow
      const m = document.querySelector(sel), fw = m && m.contentWindow;
      window.__tx.push({ t: Math.round(now - t0), h: sc.scrollHeight, top: m ? Math.round(m.getBoundingClientRect().top) : null,
        tow: !!(fw && fw.__melionaTow), busy: document.documentElement.classList.contains('m-busy') });
      if (performance.now() - t0 < ms) requestAnimationFrame(f);
    })(performance.now());
  }, [ms, MAP]);
  const trace = () => p.evaluate(() => { const t = window.__tr, d = t.slice(1).map((y, i) => y - t[i]);
    return { moved: t.at(-1) - t[0], frames: d.filter(x => Math.abs(x) > 0.5).length, maxStep: Math.max(...d.map(Math.abs)) }; });

  // notch: the pointer over the map's plot
  await place(40); await p.waitForTimeout(1200);
  const box = await p.evaluate((sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return [r.left + r.width * 0.35, Math.min(r.top + r.height * 0.45, innerHeight - 60)]; }, MAP);
  await p.mouse.move(box[0], box[1]); await p.waitForTimeout(300);
  await record(1500); await p.mouse.wheel(0, 100); await p.waitForTimeout(1700);
  const n = await trace();
  ok(n.moved > 50 && n.frames >= 8, `${W}x${H} notch: one notch over the map moved the page ${Math.round(n.moved)}px over ${n.frames} frames`);

  // run: the map comes up under a still pointer
  await place(H - 100); await p.waitForTimeout(1200);
  await p.mouse.move(W * 0.4, H * 0.5); await p.waitForTimeout(200);
  await record(4500);
  for (let i = 0; i < 14; i++) { await p.mouse.wheel(0, 100); await p.waitForTimeout(90); }
  await p.waitForTimeout(3500);
  const r = await trace();
  const crossed = await p.evaluate(([sel, y]) => document.querySelector(sel).getBoundingClientRect().top < y, [MAP, H * 0.5]);
  ok(crossed && r.maxStep <= 40, `${W}x${H} run: the map passed under the pointer, largest single-frame step ${Math.round(r.maxStep)}px over ${r.frames} frames`);
  if (r.maxStep > 40) console.log(`::error title=map wheel ${W}x${H}::big step `, JSON.stringify(await p.evaluate(() => {
    const t = window.__tr, x = window.__tx; let k = 1;
    for (let i = 1; i < t.length; i++) if (Math.abs(t[i] - t[i - 1]) > Math.abs(t[k] - t[k - 1])) k = i;
    return x.slice(Math.max(0, k - 3), k + 3).map((e, j) => ({ ...e, y: Math.round(t[Math.max(0, k - 3) + j]) }));
  })));

  // panel: a map panel that can scroll takes the wheel before the page
  const at = await p.evaluate((sel) => {
    const f = document.querySelector(sel), d = f.contentDocument;
    const el = d && [...d.querySelectorAll('.side, .detail')].find(e => e.scrollHeight > e.clientHeight + 4);
    if (!el) return null;
    el.scrollTop = 0; el.dataset.wheelProbe = '1';
    const sc = document.querySelector('[data-testid="stMain"]');
    sc.scrollTop += f.getBoundingClientRect().top + el.getBoundingClientRect().top - innerHeight / 3;
    return 1;
  }, MAP);
  if (!at) { console.log(`  --  ${W}x${H} panel: no map panel overflows at this size`); await p.close(); continue; }
  await p.waitForTimeout(800);
  const pt = await p.evaluate((sel) => { const f = document.querySelector(sel).getBoundingClientRect(), r = document.querySelector(sel).contentDocument.querySelector('[data-wheel-probe]').getBoundingClientRect(); return [f.left + r.left + r.width / 2, f.top + r.top + Math.min(r.height / 2, 120)]; }, MAP);
  await p.mouse.move(pt[0], pt[1]); await p.waitForTimeout(200);
  const state = () => p.evaluate((sel) => [document.querySelector('[data-testid="stMain"]').scrollTop, document.querySelector(sel).contentDocument.querySelector('[data-wheel-probe]').scrollTop], MAP);
  const s0 = await state(); await p.mouse.wheel(0, 60); await p.waitForTimeout(700); const s1 = await state();
  ok(s1[1] > s0[1] && Math.abs(s1[0] - s0[0]) < 2, `${W}x${H} panel: the panel scrolled ${Math.round(s1[1] - s0[1])}px, the page ${Math.round(s1[0] - s0[0])}px`);
  await p.close();
}
await b.close();
console.log(fails ? `${fails} map wheel check(s) failed` : 'the wheel glides over the map');
process.exit(fails ? 1 : 0);
