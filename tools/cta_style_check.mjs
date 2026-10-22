// Buttons and marks look and act alike.
//
//   node tools/cta_style_check.mjs <url> [width=1440]
//
// The 2026-10-22 hunt (ME-04) measured the page's in-page links one by one
// and found the outliers a visitor feels without naming: "Skip to the risk
// map" had no hover state and an arrow that never drew, the footer boxes and
// that link wore Streamlit's rounded box-shadow ring while the CTAs wore a
// square bronze outline, the focus outline on the dark band measured 2.82:1,
// and "See the crisis record" stretched to 587px beside 221-428px siblings.
// For every in-page link this reads the computed style at rest, on hover and
// on keyboard focus, and the real pixels around the focus ring, then fails on:
//   - a CTA in another type family or case than the others;
//   - a corner radius on any of them (the CTAs are hard rectangles);
//   - a hover that changes nothing;
//   - an arrow that does not draw on hover and on focus;
//   - a focus ring that is not the one shared style (2px solid outline, no
//     box-shadow ring), or that clears less than 3:1 against the ground
//     around it (WCAG 2.2 1.4.11, measured from a screenshot, not assumed);
//   - a CTA much wider than its words (a flex parent stretching it).
// Exit code 1 on any failure.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, width = '1440'] = process.argv;
const LINKS = [
  ['#hero .cta-btn', 'hero CTA', 'cta'],
  ['a.cta-btn[href="#crisis-record"]', 'See the crisis record', 'cta'],
  ['a.cta-btn[href="#the-slice"]', 'From the record to the allocation', 'cta'],
  ['a.cta-btn[href="#engine"]', 'To the engine', 'cta'],
  ['.engine-heading a[href="#analysis"]', 'Skip link', 'cta'],
  ['.meleona-footer a.f-box >> nth=0', 'Footer 01', 'box'],
  ['.meleona-footer a.f-box >> nth=1', 'Footer 02', 'box'],
  ['.meleona-footer a.f-box >> nth=2', 'Footer 03', 'box'],
];
const W = +width, touch = W < 600;
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const page = await (await b.newContext({ viewport: { width: W, height: 900 },
  ...(touch ? { isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : {}) })).newPage();
await page.goto(url);
await page.waitForSelector('.meleona-footer', { timeout: 300000 });
await page.waitForTimeout(2500);

const PROPS = ['fontFamily', 'textTransform', 'borderTopLeftRadius', 'backgroundColor', 'color',
  'borderBottomColor', 'outlineStyle', 'outlineWidth', 'outlineColor', 'outlineOffset', 'boxShadow'];
const read = l => l.evaluate((a, props) => {
  const cs = getComputedStyle(a), o = {};
  props.forEach(p => { o[p] = cs[p]; });
  const sh = a.querySelector('.cta-arrow .shaft');
  o.arrow = sh ? getComputedStyle(sh).strokeDashoffset : null;
  const r = a.getBoundingClientRect(); o.w = Math.round(r.width);
  // the words' own width, to tell a stretched button from a long label
  const range = document.createRange(); range.selectNodeContents(a);
  o.words = Math.round(range.getBoundingClientRect().width);
  return o;
}, PROPS);
const lum = ([r, g, bl]) => { const c = [r, g, bl].map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; };
const ratio = (a, z) => { const [x, y] = [lum(a), lum(z)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const rgb = s => (s.match(/[\d.]+/g) || []).slice(0, 3).map(Number);

// pixels just outside the ring: 3px beyond outline-offset + width, at the
// middle of each side, read from a real screenshot
async function groundAround(l, o) {
  const box = await l.boundingBox();
  const pad = parseFloat(o.outlineOffset) + parseFloat(o.outlineWidth) + 3;
  const clip = { x: Math.max(0, box.x - pad - 2), y: Math.max(0, box.y - pad - 2),
                 width: box.width + 2 * pad + 4, height: box.height + 2 * pad + 4 };
  const png = (await page.screenshot({ clip })).toString('base64');
  return page.evaluate(async ([src, w, h]) => {
    const img = new Image(); img.src = 'data:image/png;base64,' + src; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const x = c.getContext('2d'); x.drawImage(img, 0, 0);
    const sx = img.width / w, sy = img.height / h;
    const at = (px, py) => [...x.getImageData(Math.round(px * sx), Math.round(py * sy), 1, 1).data].slice(0, 3);
    // 15 pixels along each side, the median by brightness: a glyph of the
    // paragraph above must not pass for the ground
    const side = f => { const v = []; for (let i = 1; i <= 15; i++) v.push(f(i / 16));
      v.sort((p, q) => (p[0] + p[1] + p[2]) - (q[0] + q[1] + q[2])); return v[7]; };
    return [side(t => at(1, t * h)), side(t => at(w - 2, t * h)),
            side(t => at(t * w, 1)), side(t => at(t * w, h - 2))];
  }, [png, clip.width, clip.height]);
}

const rows = [], fails = [];
const fail = (label, why) => fails.push(`${label}: ${why}`);
// the page scrolls smoothly (app.css): a box read mid-scroll puts the pixel
// samples on the wrong spot, so this probe scrolls instantly
await page.evaluate(() => { document.querySelector('[data-testid="stMain"]').style.scrollBehavior = 'auto'; });
await page.keyboard.press('Tab');                 // later focus() calls count as keyboard focus
for (const [sel, label, kind] of LINKS) {
  const l = page.locator(sel).first();
  if (!(await l.count())) { fail(label, 'not on the page'); continue; }
  await l.evaluate(a => a.scrollIntoView({ block: 'center' }));
  await page.mouse.move(2, 2); await page.waitForTimeout(450);
  const rest = await read(l);
  await l.hover(); await page.waitForTimeout(950);
  const hover = await read(l);
  await page.mouse.move(2, 2); await l.focus(); await page.waitForTimeout(950);
  const focus = await read(l);
  const ground = await groundAround(l, focus);
  const ring = rgb(focus.outlineColor);
  const worst = Math.min(...ground.map(g => ratio(ring, g)));
  rows.push({ label, kind, rest, hover, focus, ring: focus.outlineColor, worst: +worst.toFixed(2) });
  await l.evaluate(a => a.blur());

  if (rest.borderTopLeftRadius !== '0px') fail(label, `corner radius ${rest.borderTopLeftRadius}`);
  // a touch screen has no hover to judge; focus still counts (keyboards pair with phones)
  if (!touch && ['backgroundColor', 'color', 'borderBottomColor'].every(p => rest[p] === hover[p]))
    fail(label, 'hover changes nothing');
  if (rest.arrow !== null && ((!touch && hover.arrow !== '0px') || focus.arrow !== '0px'))
    fail(label, `arrow does not draw (hover ${hover.arrow}, focus ${focus.arrow})`);
  if (focus.outlineStyle !== 'solid' || focus.outlineWidth !== '2px' || focus.boxShadow !== 'none')
    fail(label, `focus ring is not the shared style: ${focus.outlineStyle} ${focus.outlineWidth}, shadow ${focus.boxShadow}`);
  if (worst < 3) fail(label, `focus ring ${focus.outlineColor} clears only ${worst.toFixed(2)}:1 against the ground around it`);
  if (kind === 'cta' && rest.w > rest.words + 2 * 34 + 2 * 11 + 60)
    fail(label, `${rest.w}px wide for ${rest.words}px of words - stretched by its parent`);
}
const ctas = rows.filter(r => r.kind === 'cta');
for (const p of ['fontFamily', 'textTransform']) {
  const seen = [...new Set(ctas.map(r => r.rest[p]))];
  if (seen.length > 1) fail('CTAs', `${p} differs: ${ctas.map(r => `${r.label}=${r.rest[p]}`).join(', ')}`);
}
const offsets = [...new Set(rows.map(r => r.focus.outlineOffset))];
if (offsets.length > 1) fail('focus rings', `outline offsets differ: ${offsets.join(', ')}`);

for (const r of rows)
  console.log(`${r.label.padEnd(36)} ${r.rest.fontFamily.split(',')[0].padEnd(16)} radius ${r.rest.borderTopLeftRadius.padEnd(4)} `
    + `w ${String(r.rest.w).padStart(4)} (words ${r.rest.words})  hover ${['backgroundColor', 'color', 'borderBottomColor'].some(p => r.rest[p] !== r.hover[p]) ? 'yes' : 'NO '}  `
    + `arrow ${r.rest.arrow === null ? '-' : `${r.hover.arrow}/${r.focus.arrow}`}  ring ${r.focus.outlineStyle} ${r.focus.outlineWidth} +${r.focus.outlineOffset} `
    + `${r.ring} shadow ${r.focus.boxShadow === 'none' ? 'none' : 'YES'}  ${r.worst}:1`);
for (const f of fails) console.log('FAIL', f);
console.log(fails.length ? `${fails.length} failures` : 'every link looks and acts alike');
await b.close();
process.exit(fails.length ? 1 : 0);
