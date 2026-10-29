// Every caption clears AA - measured from what the visitor sees.
//
//   node tools/caption_contrast_check.mjs <url> [width=1440]
//
// test_text_colours_meet_wcag_aa reads colours out of the stylesheet, and the
// stylesheet said captions were #524E47 (6.5:1 on the panel). The page drew
// them at 60% opacity - Streamlit's own caption style - so the ink a visitor
// got was a blend of that colour and the ground (ME-06 follow-up, 2026-10-22).
// For every caption (and the small metric labels beside them) in every tab,
// this takes the ground from screenshot pixels around the text, blends the
// computed text colour into it through every ancestor's opacity, and fails on
// anything under 4.5:1. The darkest glyph pixel is printed beside it as a
// cross-check (anti-aliasing keeps thin serif stems lighter than the ink).
// Exit code 1 on any failure.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, width = '1440'] = process.argv;
const W = +width;
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const page = await (await b.newContext({ viewport: { width: W, height: 900 },
  ...(W < 600 ? { isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : {}) })).newPage();
await page.goto(url);
await page.waitForSelector('.meliona-footer', { timeout: 300000 });
await page.waitForTimeout(2500);
await page.evaluate(() => { document.querySelector('[data-testid="stMain"]').style.scrollBehavior = 'auto'; });

const lum = ([r, g, bl]) => { const c = [r, g, bl].map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; };
const ratio = (a, z) => { const [x, y] = [lum(a), lum(z)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

const SEL = '[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"]';
const seen = new Map();
async function measureVisible(where) {
  const n = await page.locator(SEL).count();
  for (let i = 0; i < n; i++) {
    const l = page.locator(SEL).nth(i);
    const info = await l.evaluate(e => {
      const r = e.getBoundingClientRect();
      if (!r.width || !r.height || !e.offsetParent) return null;
      const t = e.innerText.trim(); if (!t) return null;
      // the colour the words are drawn in, and every opacity above them
      const ink = e.querySelector('p') || e;
      const cs = getComputedStyle(ink);
      let op = 1; for (let n = ink; n; n = n.parentElement) op *= parseFloat(getComputedStyle(n).opacity);
      return { t: t.slice(0, 50), color: cs.color, op, kind: e.dataset.testid };
    });
    if (!info || seen.has(info.t)) continue;
    await l.evaluate(e => e.scrollIntoView({ block: 'center' }));
    await page.waitForTimeout(60);
    const box = await l.boundingBox();
    if (!box || box.width < 4) continue;
    const png = (await page.screenshot({ clip: { x: box.x, y: box.y, width: Math.min(box.width, W - box.x), height: box.height } })).toString('base64');
    const px = await page.evaluate(async src => {
      const img = new Image(); img.src = 'data:image/png;base64,' + src; await img.decode();
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      const x = c.getContext('2d'); x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, img.width, img.height).data, all = [];
      for (let i = 0; i < d.length; i += 16) all.push([d[i], d[i + 1], d[i + 2]]);
      all.sort((p, q) => (p[0] + p[1] + p[2]) - (q[0] + q[1] + q[2]));
      return { ground: all[Math.floor(all.length * 0.9)], darkest: all[0] };
    }, png);
    const fg = (info.color.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    const eff = fg.map((v, k) => v * info.op + px.ground[k] * (1 - info.op));
    seen.set(info.t, { ...info, where, ground: px.ground, eff: eff.map(Math.round),
      ratio: ratio(eff, px.ground), glyph: ratio(px.darkest, px.ground) });
  }
}
await measureVisible('page');
const tabs = await page.getByRole('tab').all();
for (const t of tabs) {
  await t.evaluate(e => e.scrollIntoView({ block: 'center' }));
  await t.click().catch(() => {});
  await page.waitForTimeout(900);
  await measureVisible(`tab ${(await t.innerText()).trim()}`);
}
await b.close();

const rows = [...seen.values()];
const fails = rows.filter(r => r.ratio < 4.5);
const by = k => rows.filter(r => r.kind === k);
for (const k of ['stCaptionContainer', 'stMetricLabel']) {
  const r = by(k).map(x => x.ratio).sort((p, q) => p - q);
  if (r.length) console.log(`${k.padEnd(19)} n=${String(r.length).padStart(3)}  min ${r[0].toFixed(2)}:1  median ${r[r.length >> 1].toFixed(2)}:1  (opacity ${[...new Set(by(k).map(x => x.op.toFixed(2)))].join('/')})`);
}
for (const f of fails.slice(0, 12))
  console.log(`FAIL ${f.kind} ${f.ratio.toFixed(2)}:1 (glyph ${f.glyph.toFixed(2)}:1) ink ${f.color} x${f.op.toFixed(2)} on rgb(${f.ground}) [${f.where}] "${f.t}"`);
console.log(fails.length ? `${fails.length} of ${rows.length} under 4.5:1` : `all ${rows.length} clear 4.5:1`);
process.exit(fails.length ? 1 : 0);
