/*
  Map layout probe - renders the SHIPPED Risk Topology page in Chromium at the
  height main.py gives its iframe (820px) and the widths a visitor can have,
  and reports what a person would hit: glass panels on top of each other or of
  the plot, the bottom chips covering the beta tick labels, buttons that are
  painted but cannot be clicked, a plot too small to read, a canvas stretched
  to a width it was not drawn for, and sideways scrolling.

  It needs playwright (npm i -g playwright && npx playwright install chromium);
  without it it exits 3 so the caller can skip, the way tests/map_probe.mjs's
  caller skips without node.

  usage: node tests/map_layout.mjs <page.html> [width,width,...]
*/
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

// 1456 and 1220 sit inside bands where the HUD tiles once spilled under the inspector while every round width passed
const [pagePath, widthArg = '1920,1456,1440,1280,1220,1180,1024,1000,790,600,390'] = process.argv.slice(2);
let pw;
try {
  pw = await import('playwright');
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
const browser = await chromium.launch().catch(() => chromium.launch({ channel: 'msedge' }));
const out = [];
for (const width of widthArg.split(',').map(Number)) {
  const page = await browser.newPage({ viewport: { width, height: 820 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(pathToFileURL(pagePath).href);
  await page.waitForFunction(() => typeof S !== 'undefined' && S.cloud);
  await page.waitForTimeout(400);
  const m = await page.evaluate(() => {
    const box = (el) => { const r = el.getBoundingClientRect(); return { l: r.left, t: r.top, r: r.right, b: r.bottom }; };
    const hit = (a, b) => Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
    const panels = { side: '#sidePanel', hud: '.hud', inspector: '#inspPanel', chips: '.chipbar' };
    const P = Object.fromEntries(Object.entries(panels).map(([k, s]) => [k, box(document.querySelector(s))]));
    const found = [];
    const names = Object.keys(P);
    names.forEach((a, i) => names.slice(i + 1).forEach((b) => {
      const area = hit(P[a], P[b]);
      if (area > 4) found.push(`${a} overlaps ${b} (${Math.round(area)} px²)`);
    }));
    // a panel's content spilling out of the panel (it paints over whatever is next to it)
    for (const k of ['side', 'hud', 'inspector']) {
      for (const el of document.querySelector(panels[k]).querySelectorAll('.tile, .chip, .verdict, .runbtn, .explbtn, .srow, .dcell')) {
        const e = box(el);
        if (e.r > P[k].r + 1 || e.l < P[k].l - 1 || e.b > P[k].b + 1) found.push(`${k}: ${el.className} spills out of its panel`);
      }
    }
    // the plot, in page coordinates: nothing may lie on it, and the chips may not cover its tick labels
    const cv = document.getElementById('cvBase').getBoundingClientRect();
    const sx = cv.width / W, sy = cv.height / H;
    const plot = { l: cv.left + PAD.l * sx, r: cv.left + (W - PAD.r) * sx, t: cv.top + PAD.t * sy, b: cv.top + (H - PAD.b) * sy };
    for (const k of ['side', 'hud', 'inspector', 'chips']) if (hit(plot, P[k]) > 4) found.push(`${k} lies on the plot`);
    const ticks = { l: plot.l - 20, r: plot.r + 20, t: cv.top + (H - PAD.b + 5) * sy, b: cv.top + (H - PAD.b + 20) * sy };
    if (hit(ticks, P.chips) > 4) found.push('chips cover the beta tick labels');
    // every control must take a click where it is painted
    for (const id of ['btnRun', 'btnExplain', 'selCal', 'tgAtmo', 'tgMarks', 'tgLinks']) {
      const el = document.getElementById(id);
      const target = el.type === 'checkbox' ? el.closest('label') : el;
      target.scrollIntoView({ block: 'center' });
      const r = target.getBoundingClientRect();
      const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      if (!top || !(target === top || target.contains(top))) found.push(`#${id} is covered and cannot be clicked`);
    }
    window.scrollTo(0, 0);
    const pw_ = (W - PAD.l - PAD.r) * sx, ph = (H - PAD.t - PAD.b) * sy;
    if (pw_ < 260 || ph < 260) found.push(`plot too small to read: ${Math.round(pw_)} x ${Math.round(ph)} px`);
    if (Math.abs(cv.width - W) > 1 || Math.abs(cv.height - H) > 1) found.push(`canvas drawn at ${W}x${H} but shown at ${Math.round(cv.width)}x${Math.round(cv.height)} (stretched)`);
    // stacked (narrow) layout: the x-axis must be on the frame's first screen, not below the fold of the iframe
    if (getComputedStyle(document.querySelector('.stage')).position !== 'absolute' && plot.b + 20 > innerHeight)
      found.push(`x-axis below the fold (${Math.round(plot.b)}px of ${innerHeight})`);
    // explain mode reads the MAP: a pointer outside it (below the plot, over the stacked panels) shows nothing
    const wasExplain = S.explain;
    S.explain = true;
    document.getElementById('stage').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: cv.left + cv.width / 2, clientY: cv.bottom + 40 }));
    if (document.getElementById('tip').style.display === 'block') found.push('explain tooltip shows outside the map');
    document.getElementById('tip').style.display = 'none';
    S.explain = wasExplain;
    const sideways = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    if (sideways > 0) found.push(`scrolls sideways by ${sideways} px`);
    return { plot: [Math.round(pw_), Math.round(ph)], found };
  });
  out.push({ width, ...m, errors });
  await page.close();
}
await browser.close();
console.log(JSON.stringify(out));
