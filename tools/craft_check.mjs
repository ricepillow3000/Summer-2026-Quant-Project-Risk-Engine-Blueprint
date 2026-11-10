// The page's craft, measured where the visitor sees it.
//
//   node tools/craft_check.mjs <url>
//
// John, 2026-11-09: "make the website feel more immersive, alive, less AI and
// human" - kerning, optical center, tabular numbers, layout shift, tracking,
// optimistic update, x-height, touch target, leading, origin-aware animation,
// OKLCH, widows, safe area, stagger, line length, affordance, combo box,
// interruptible animation and shared element transition. Each is a reading of
// the real page in Chromium (desktop 1440x900, then an iPhone-sized touch
// screen), not of the stylesheet's text:
//
//   kerning      body text is kerned and laid out for legibility
//   x-height     body type keeps its x-height if Georgia is missing (font-size-adjust)
//   tabular      every on-screen figure (digits, %, $) sets tabular, lining numbers
//   tracking     display type (>= 30px) sets at its own spacing - never tightened, the
//                generated-page look - and the capital label voice is opened up
//   leading      prose 1.45-1.8, display type 0.9-1.25
//   measure      no prose paragraph runs past 75 characters a line
//   widows       prose wraps 'pretty' and headings 'balance', and no prose
//                paragraph ends on a lone word
//   affordance   links in prose are underlined, every button shows a pointer,
//                and a keyboard focus draws a visible ring
//   oklch        hover, focus and selection tints are mixed in OKLCH
//   optical      the hero's words sit at the optical centre: above the box's middle
//   layout shift cumulative layout shift from load to landing under 0.02
//   stagger      blocks that arrive together leave 90ms apart
//   origin       a button's hover colour rises from where the pointer came in
//   optimistic   a changed setting answers inside 150ms: an updating line at
//                the top, the old numbers kept readable, never blanked
//   combo box    the ticker picker is a combobox: typing filters its list
//   interrupt    a tab picked mid-slide retargets the underline from where it is
//   shared       one underline travels between the tabs, never cut to the next
//   count-up     every number that counts up lands on the server's exact text
//   touch        on a phone every control is at least 44 x 44px
//   safe area    the viewport covers the notch and the page keeps its gutters
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
const { chromium, devices } = pw.chromium ? pw : pw.default;
const [,, url] = process.argv;
if (!url) { console.error('usage: node tools/craft_check.mjs <url>'); process.exit(2); }
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
let fails = 0;
const ok = (pass, what, detail = '') => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${what}${detail ? '  ' + detail : ''}`);
  if (!pass) fails++;
};
async function open(ctx) {
  const p = await ctx.newPage();
  // Layout shift as the visitor sees it: from the moment the loading screen
  // (#boot-skel, a full-screen opaque layer) lifts. The page assembles under
  // it on purpose - blocks arrive as Streamlit streams them - and a move
  // under an opaque layer is nothing anyone can see; everything after the
  // lift counts. Both totals are printed.
  await p.addInitScript(() => {
    window.__cls = 0; window.__clsHidden = 0;
    const covered = () => { const s = document.getElementById('boot-skel');
      if (!s) return document.readyState === 'loading';
      const cs = getComputedStyle(s); return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.95; };
    try {
      new PerformanceObserver(l => l.getEntries().forEach(e => { if (e.hadRecentInput) return;
        if (covered()) window.__clsHidden += e.value; else window.__cls += e.value; }))
        .observe({ type: 'layout-shift', buffered: true });
    } catch (x) { /* no layout-shift API */ }
  });
  await p.goto(url);
  await p.waitForSelector('.meliona-footer', { timeout: 300000 });
  await p.waitForTimeout(2000);
  return p;
}

// ---------------------------------------------------------------- desktop
const desk = await b.newContext({ viewport: { width: 1440, height: 900 } });
const p = await open(desk);

// optical centre and layout shift on the page as it first lands
const optical = await p.evaluate(() => {
  const hero = document.querySelector('[data-testid="stMain"] .hero-section');
  const copy = hero && hero.querySelector('.hero-title');
  if (!hero || !copy) return null;
  // the part of the hero the visitor sees: it bleeds past the top of the screen
  const r = hero.getBoundingClientRect(), h = { top: Math.max(0, r.top), height: r.bottom - Math.max(0, r.top) };
  // the words' block: from the crest (or title) to the last line under it
  const parts = [...hero.querySelectorAll('.hero-mark, .hero-title, .hero-sub, .hero-ctas')].map(e => e.getBoundingClientRect()).filter(r => r.height);
  const top = Math.min(...parts.map(r => r.top)), bottom = Math.max(...parts.map(r => r.bottom));
  return { lift: ((h.top + h.height / 2) - (top + bottom) / 2) / h.height, h: h.height };
});
ok(optical && optical.lift >= 0.015 && optical.lift <= 0.09, 'optical: the hero\'s words sit above the middle of their box',
  optical ? `${(optical.lift * 100).toFixed(1)}% of ${Math.round(optical.h)}px above centre` : 'no hero');

// scroll the whole page down like a reader so every block arrives, then measure the landed page
await p.mouse.move(1410, 450);
for (let i = 0; i < 160; i++) {
  await p.mouse.wheel(0, 240); await p.waitForTimeout(45);
  const done = await p.evaluate(() => { const s = document.querySelector('[data-testid="stMain"]'); return s.scrollTop >= s.scrollHeight - s.clientHeight - 2; });
  if (done) break;
}
await p.waitForTimeout(2500);
const cls = await p.evaluate(() => [window.__cls, window.__clsHidden]);
ok(cls[0] < 0.02, 'layout shift: nothing the visitor can see jumps, from the first look to the last block', `CLS ${cls[0].toFixed(4)} (under the loading screen: ${cls[1].toFixed(4)})`);
await p.evaluate(() => { window.__melionaReveal && window.__melionaReveal.revealAll(); document.querySelector('[data-testid="stMain"]').scrollTop = 0; });
await p.waitForTimeout(800);

const type = await p.evaluate(() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; };
  const main = document.querySelector('[data-testid="stMain"]');
  const bodyCs = getComputedStyle(document.body);
  const out = { kern: bodyCs.fontKerning, render: bodyCs.textRendering, adjust: bodyCs.fontSizeAdjust };
  // every leaf that shows figures
  const figs = [];
  main.querySelectorAll('*').forEach(e => {
    if (e.children.length || !vis(e) || e.closest('svg, canvas, iframe, [data-testid="stDataFrame"], .stCode, code')) return;
    const t = (e.textContent || '').trim();
    if (!/\d/.test(t) || t.length > 24 || !/^[\s$€£+\-−–.,:%x()/0-9A-Za-z·]+$/.test(t)) return;
    if ((t.match(/\d/g) || []).length < Math.max(1, t.replace(/\s/g, '').length * 0.4)) return;
    const v = getComputedStyle(e).fontVariantNumeric;
    figs.push({ t, ok: /tabular-nums/.test(v) && /lining-nums/.test(v) });
  });
  out.figs = figs;
  // display type and the label voice
  const disp = [], labels = [];
  main.querySelectorAll('h1, h2, h3, .hero-title, .showcase-title, .panel-head .s, .verdict-number, p, span, div').forEach(e => {
    if (!vis(e) || ![...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) return;
    const cs = getComputedStyle(e), fs = parseFloat(cs.fontSize), ls = cs.letterSpacing === 'normal' ? 0 : parseFloat(cs.letterSpacing);
    if (fs >= 30) disp.push({ t: e.textContent.trim().slice(0, 30), em: ls / fs, lh: parseFloat(cs.lineHeight) / fs });
    if (cs.textTransform === 'uppercase' && fs <= 15) labels.push({ t: e.textContent.trim().slice(0, 30), em: ls / fs });
  });
  out.disp = disp; out.labels = labels;
  // prose: paragraphs of real reading length
  const prose = [];
  main.querySelectorAll('p, .story-p, .faq-a, .book-lede, .book-text, .about-bio, .hero-sub, .contact-sub').forEach(e => {
    if (!vis(e) || e.closest('[data-testid="stExpander"] details:not([open])')) return;
    const t = (e.textContent || '').trim();
    if (t.length < 160) return;
    const cs = getComputedStyle(e), fs = parseFloat(cs.fontSize);
    // the measure: the paragraph's width in its own '0' widths
    // CSS's own ch, measured inside the paragraph (its font, size and figure style)
    const probe = document.createElement('span');
    probe.style.cssText = 'display:inline-block;width:50ch;height:0;position:absolute;visibility:hidden';
    e.appendChild(probe); const ch = probe.getBoundingClientRect().width / 50; probe.remove();
    // a widow: the last word alone on the last line
    const tw = document.createTreeWalker(e, NodeFilter.SHOW_TEXT), tn = [];  // in reading order
    for (let n = tw.nextNode(); n; n = tw.nextNode()) if (n.textContent.trim()) tn.push(n);
    let widow = false;
    if (tn.length) {
      const last = tn[tn.length - 1], s = last.textContent.replace(/\s+$/, ''), i = s.search(/\S+$/);
      if (i > 0) {
        const r1 = document.createRange(); r1.setStart(last, i); r1.setEnd(last, s.length);
        const j = s.slice(0, i).replace(/\s+$/, '').search(/\S+$/);
        if (j >= 0) { const r0 = document.createRange(); r0.setStart(last, j); r0.setEnd(last, j + 1);
          widow = r1.getBoundingClientRect().top - r0.getBoundingClientRect().top > fs * 0.6; }
      }
    }
    prose.push({ t: t.slice(0, 40), chars: e.getBoundingClientRect().width / ch, lh: parseFloat(cs.lineHeight) / fs,
      wrap: cs.textWrapStyle || cs.textWrap || '', widow, size: fs });
  });
  out.prose = prose;
  const heads = [...main.querySelectorAll('.hero-title, .showcase-title, .panel-head .s, h1, h2, h3')].filter(vis)
    .map(e => ({ t: e.textContent.trim().slice(0, 30), wrap: getComputedStyle(e).textWrapStyle || getComputedStyle(e).textWrap || '' }));
  out.heads = heads;
  // a short value is never broken: a slider's figure stays on one line
  out.thumbs = [...main.querySelectorAll('[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] > *')].filter(vis).map(e => {
    const tops = new Set();
    const w = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      if (!n.textContent.trim()) continue;
      const r = document.createRange(); r.selectNodeContents(n);
      [...r.getClientRects()].filter(q => q.width > 0).forEach(q => tops.add(Math.round(q.top)));
    }
    return { t: e.textContent.trim(), lines: tops.size };
  });
  // links in prose and the pointer on buttons
  out.links = [...main.querySelectorAll('p a, .story-p a, .faq-a a, .about-bio a, .contact-sub a, .contact-note a')].filter(vis)
    .map(a => ({ t: a.textContent.trim().slice(0, 30), u: /underline/.test(getComputedStyle(a).textDecorationLine) }));
  out.buttons = [...main.querySelectorAll('button, [role="tab"], summary, .cta-btn, .cta-link, a[href]')].filter(vis)
    .map(e => ({ t: (e.textContent || e.getAttribute('aria-label') || '').trim().slice(0, 30), c: getComputedStyle(e).cursor }))
    .filter(x => x.c !== 'pointer');
  // OKLCH in the stylesheet's states
  let oklch = 0;
  for (const sh of document.styleSheets) { try { for (const r of sh.cssRules) if (/oklch/.test(r.cssText)) oklch++; } catch (x) { /* cross-origin sheet */ } }
  out.oklch = oklch;
  const vp = document.querySelector('meta[name="viewport"]');
  out.viewport = vp ? vp.content : '';
  return out;
});
ok(type.kern === 'normal' && /optimizelegibility/i.test(type.render), 'kerning: body text is kerned and set for legibility', `${type.kern} / ${type.render}`);
ok(type.adjust && type.adjust !== 'none', 'x-height: body type keeps its x-height on a fallback font', `font-size-adjust ${type.adjust}`);
const badFig = type.figs.filter(f => !f.ok);
ok(type.figs.length > 10 && !badFig.length, `tabular: ${type.figs.length} on-screen figures set tabular, lining numbers`,
  badFig.slice(0, 4).map(f => `"${f.t}"`).join(' '));
const looseDisp = type.disp.filter(d => Math.abs(d.em) > 0.001);
ok(type.disp.length && !looseDisp.length, `tracking: ${type.disp.length} display lines at their own spacing`, looseDisp.slice(0, 3).map(d => `"${d.t}" ${d.em.toFixed(3)}em`).join(' '));
const tightLab = type.labels.filter(l => l.em < 0.1);
ok(type.labels.length && !tightLab.length, `tracking: ${type.labels.length} capital labels opened to >= .1em`, tightLab.slice(0, 3).map(l => `"${l.t}" ${l.em.toFixed(3)}em`).join(' '));
const badLead = type.prose.filter(x => x.lh < 1.45 || x.lh > 1.8).concat(type.disp.filter(d => d.lh < 0.9 || d.lh > 1.25));
ok(!badLead.length, `leading: prose 1.45-1.8, display 0.9-1.25`, badLead.slice(0, 3).map(x => `"${x.t}" ${x.lh.toFixed(2)}`).join(' '));
const long = type.prose.filter(x => x.chars > 75);
const broken = type.thumbs.filter(x => x.lines > 1);
ok(type.thumbs.length && !broken.length, `wrapping: ${type.thumbs.length} slider figures each on one line`, broken.map(x => `"${x.t}" on ${x.lines} lines`).join(' '));
ok(type.prose.length && !long.length, `measure: ${type.prose.length} prose paragraphs at <= 75 characters a line`, long.slice(0, 3).map(x => `"${x.t}" ${Math.round(x.chars)}ch`).join(' '));
const unwrapped = type.prose.filter(x => !/pretty|balance/.test(x.wrap)).concat(type.heads.filter(h => !/balance/.test(h.wrap)));
ok(!unwrapped.length, 'widows: prose wraps pretty, headings balance', unwrapped.slice(0, 3).map(x => `"${x.t}" ${x.wrap || 'wrap'}`).join(' '));
const widows = type.prose.filter(x => x.widow);
ok(!widows.length, `widows: no prose paragraph ends on a lone word`, widows.slice(0, 3).map(x => `"${x.t}"`).join(' '));
const plain = type.links.filter(l => !l.u);
ok(!plain.length, `affordance: ${type.links.length} links in prose are underlined`, plain.slice(0, 3).map(l => `"${l.t}"`).join(' '));
ok(!type.buttons.length, 'affordance: every button and link shows a pointer', type.buttons.slice(0, 4).map(x => `"${x.t}"`).join(' '));
ok(type.oklch >= 3, 'oklch: hover, focus and selection tints are mixed in OKLCH', `${type.oklch} rules`);
ok(/viewport-fit=cover/.test(type.viewport), 'safe area: the viewport covers the notch', type.viewport);

// focus ring
await p.evaluate(() => document.querySelector('[data-testid="stMain"]').focus());
const ring = await p.evaluate(async () => {
  const btn = [...document.querySelectorAll('[data-testid="stMain"] button')].find(b => b.getBoundingClientRect().width > 40);
  if (!btn) return null;
  btn.scrollIntoView({ block: 'center' });
  return true;
});
let ringOk = false;
for (let i = 0; i < 40 && ring; i++) {
  await p.keyboard.press('Tab');
  const r = await p.evaluate(() => { const a = document.activeElement; if (!a || a === document.body) return null;
    const cs = getComputedStyle(a); return { tag: a.tagName, w: parseFloat(cs.outlineWidth), s: cs.outlineStyle, sh: cs.boxShadow }; });
  if (r && r.tag === 'BUTTON') { ringOk = (r.s !== 'none' && r.w >= 2) || /rgb/.test(r.sh); console.log(`      focus on a button: outline ${r.s} ${r.w}px`); break; }
}
ok(ringOk, 'affordance: a keyboard focus draws a visible ring');

// stagger: blocks that arrive together leave 90ms apart
const stagger = await p.evaluate(async () => {
  const sc = document.querySelector('[data-testid="stMain"]');
  const seen = [];
  const mo = new MutationObserver(ms => ms.forEach(m => { const e = m.target; if (e.classList && e.classList.contains('is-visible')) {
    const d = parseFloat(e.style.getPropertyValue('--reveal-delay')); if (!isNaN(d) && !seen.includes(d)) seen.push(d); } }));
  mo.observe(document.body, { attributes: true, attributeFilter: ['class'], subtree: true });
  return { start: sc.scrollTop };
});
await p.reload(); await p.waitForSelector('.meliona-footer', { timeout: 300000 }); await p.waitForTimeout(1500);
const delays = await p.evaluate(async () => {
  const seen = new Set();
  const mo = new MutationObserver(ms => ms.forEach(m => { const e = m.target;
    if (e.classList && e.classList.contains('is-visible')) seen.add(parseFloat(e.style.getPropertyValue('--reveal-delay'))); }));
  mo.observe(document.body, { attributes: true, attributeFilter: ['class'], subtree: true });
  const sc = document.querySelector('[data-testid="stMain"]'); sc.style.scrollBehavior = 'auto';
  sc.scrollTop = sc.scrollHeight * 0.45;
  await new Promise(r => setTimeout(r, 900));
  mo.disconnect();
  return [...seen].filter(x => !isNaN(x)).sort((a, b) => a - b);
});
ok(delays.length >= 3 && delays.slice(1).every((d, i) => Math.abs(d - delays[i] - 90) < 1), 'stagger: blocks that arrive together leave 90ms apart', delays.join(','));
void stagger;

// origin-aware hover: the colour rises from where the pointer came in
await p.evaluate(() => window.__melionaReveal && window.__melionaReveal.revealAll());
const cta = p.locator('.cta-btn:visible').first();
await cta.scrollIntoViewIfNeeded(); await p.waitForTimeout(400);
const cb = await cta.boundingBox();
await p.mouse.move(cb.x - 30, cb.y + cb.height * 0.8); await p.waitForTimeout(200);
await p.mouse.move(cb.x + 6, cb.y + cb.height * 0.8, { steps: 4 }); await p.waitForTimeout(60);
const origin = await cta.evaluate(e => ({ x: e.style.getPropertyValue('--mx'), y: e.style.getPropertyValue('--my'), clip: getComputedStyle(e, '::after').clipPath }));
ok(parseFloat(origin.x) < 30 && parseFloat(origin.y) > 50 && /circle/.test(origin.clip),
  'origin: a button\'s hover colour rises from where the pointer came in', `--mx ${origin.x} --my ${origin.y} ${origin.clip.slice(0, 40)}`);
await p.mouse.move(5, 5);

// count-up: every number that counts lands on the server's exact text
const counts = await p.evaluate(async () => {
  await new Promise(r => setTimeout(r, 1800));
  return [...document.querySelectorAll('[data-count-final]')].map(e => ({ shown: e.textContent.trim(), final: e.getAttribute('data-count-final') }));
});
const off = counts.filter(c => c.shown !== c.final);
ok(counts.length >= 1 && !off.length, `count-up: ${counts.length} numbers counted up and landed on their exact values`, off.slice(0, 3).map(c => `${c.shown}!=${c.final}`).join(' '));

// tabs: one underline slides, and a second pick mid-slide continues from
// where the underline is - frame by frame, no step larger than the slide's own
const tabs = p.locator('[data-testid="stTab"]');
if (await tabs.count() >= 3) {
  await tabs.nth(0).scrollIntoViewIfNeeded(); await tabs.nth(0).click(); await p.waitForTimeout(800);
  await p.evaluate(() => {
    const list = document.querySelector('[data-testid="stTab"]').closest('[role="tablist"]');
    window.__tr = []; window.__press = [];
    list.addEventListener('pointerdown', () => window.__press.push(performance.now()), true);
    const t0 = performance.now();
    (function f() { const i = list.querySelector('.react-aria-SelectionIndicator');
      if (i) window.__tr.push([performance.now(), i.getBoundingClientRect().left]);
      if (performance.now() - t0 < 1600) requestAnimationFrame(f); })();
  });
  await tabs.nth(2).click(); await p.waitForTimeout(90);
  await tabs.nth(1).click(); await p.waitForTimeout(1300);
  const r = await p.evaluate(() => {
    const tr = window.__tr, second = window.__press[1];
    const steps = tr.slice(1).map((x, i) => [x[0], Math.abs(x[1] - tr[i][1])]);
    const before = steps.filter(s => s[0] < second).map(s => s[1]);
    const after = steps.filter(s => s[0] >= second).map(s => s[1]);
    return { firstMax: Math.max(...before), afterMax: Math.max(...after), n: tr.length, ones: document.querySelectorAll('[role="tablist"] .react-aria-SelectionIndicator').length,
      path: tr.filter((x, i) => i % 3 === 0).map(x => Math.round(x[1])).join(',') };
  });
  ok(r.afterMax <= Math.max(30, r.firstMax * 1.25), 'interrupt: a tab picked mid-slide continues from where the underline is',
    `largest frame step ${Math.round(r.afterMax)}px after the second pick (the slide's own: ${Math.round(r.firstMax)}px) ${r.path}`);
  ok(r.ones >= 1, 'shared: one underline travels between the tabs', `${r.ones} underline(s) on the strip`);
}

// combo box: typing filters the ticker list
const combo = await p.evaluate(() => {
  const ms = [...document.querySelectorAll('[data-testid="stMultiSelect"]')][0];
  const input = ms && ms.querySelector('input[role="combobox"], [role="combobox"]');
  return input ? { role: input.getAttribute('role'), ph: (ms.innerText || '').slice(0, 80) } : null;
});
if (combo) {
  await p.evaluate(() => document.querySelectorAll('[data-testid="stExpander"] details:not([open]) > summary').forEach(sm => {
    if (sm.parentElement.querySelector('[data-testid="stMultiSelect"]')) sm.click(); }));
  await p.waitForTimeout(800);
  const inp = p.locator('[data-testid="stMultiSelect"] [role="combobox"]').first();
  await inp.scrollIntoViewIfNeeded(); await inp.click(); await p.waitForTimeout(400);
  const all = await p.locator('[data-testid="stMultiSelectDropdown"] [role="option"]').count();
  await p.keyboard.type('JP', { delay: 60 }); await p.waitForTimeout(500);
  const opts = await p.locator('[data-testid="stMultiSelectDropdown"] [role="option"]').allInnerTexts();
  await p.keyboard.press('Escape');
  const names = opts.filter(o => !/^Select \d+ match/i.test(o));
  ok(names.length > 0 && opts.length < all && names.every(o => /JP/i.test(o)), 'combo box: typing filters the ticker list', `${all} -> ${names.length}: ${names.slice(0, 3).join(' | ')}`);
} else ok(false, 'combo box: the ticker picker is a combobox');

// optimistic update: a changed setting answers inside 150ms, old numbers stay readable
await p.evaluate(() => window.__melionaReveal && window.__melionaReveal.revealAll());
const radio = p.locator('[data-testid="stRadio"]:visible [role="radiogroup"] label');
const rN = await radio.count();
let optim = null;
for (let i = 0; i < rN && !optim; i++) {
  const r = radio.nth(i);
  if (await r.evaluate(e => !!e.querySelector('input:checked'))) continue;
  await r.scrollIntoViewIfNeeded(); await p.waitForTimeout(300);
  await r.click();
  const t0 = Date.now(); let seen = null;
  while (Date.now() - t0 < 600) {
    const s = await p.evaluate(() => { const rail = getComputedStyle(document.body, '::before');
      const on = rail.content !== 'none' && rail.position === 'fixed';
      const stale = [...document.querySelectorAll('[data-stale="true"]')].map(e => parseFloat(getComputedStyle(e).opacity));
      return { on: !!on, minStale: stale.length ? Math.min(...stale) : 1 }; });
    if (s.on) { seen = { ms: Date.now() - t0, minStale: s.minStale }; break; }
    await p.waitForTimeout(15);
  }
  optim = seen || { ms: null };
  await p.waitForTimeout(6000);
}
ok(optim && optim.ms !== null && optim.ms <= 150, 'optimistic: a changed setting shows its updating line inside 150ms', optim ? `${optim.ms}ms` : 'no radio');
ok(optim && optim.minStale >= 0.5, 'optimistic: the old numbers stay readable while the new ones are worked out', optim ? `stale opacity >= ${optim.minStale}` : '');
await desk.close();

// ---------------------------------------------------------------- phone
const phone = await b.newContext({ ...devices['iPhone 13'] });
const q = await open(phone);
await q.evaluate(() => window.__melionaReveal && window.__melionaReveal.revealAll());
const touch = await q.evaluate(() => {
  const main = document.querySelector('[data-testid="stMain"]');
  const small = [];
  main.querySelectorAll('button, [role="tab"], [role="button"], summary, a[href], input[type="checkbox"], input[type="radio"], [role="combobox"], [role="slider"], [data-testid="stTooltipHoverTarget"]').forEach(e => {
    const r = e.getBoundingClientRect(); if (!r.width || !r.height || getComputedStyle(e).visibility === 'hidden') return;
    if (e.closest('p, li, .story-p, .faq-a, .about-bio, .contact-sub, .contact-note, .caption, [data-testid="stCaptionContainer"]') && e.tagName === 'A') return;  // a link inside a sentence (WCAG 2.5.8 exception)
    // the control's own box, or the label/hit area that answers for it
    const hit = e.closest('label') || e;
    const h = hit.getBoundingClientRect();
    const w = Math.max(r.width, h.width), ht = Math.max(r.height, h.height);
    if (w < 44 || ht < 44) small.push(`${(e.textContent || e.getAttribute('aria-label') || e.tagName).trim().slice(0, 24)} ${Math.round(w)}x${Math.round(ht)}`);
  });
  const pad = getComputedStyle(document.querySelector('[data-testid="stMainBlockContainer"]'));
  return { small, padL: parseFloat(pad.paddingLeft), padR: parseFloat(pad.paddingRight) };
});
ok(!touch.small.length, 'touch: every control on a phone is at least 44 x 44px', touch.small.slice(0, 6).join(' | '));
ok(touch.padL >= 16 && touch.padR >= 16, 'safe area: the page keeps 16px gutters on a phone', `${touch.padL}px / ${touch.padR}px`);
await phone.close();
await b.close();
console.log(fails ? `${fails} craft check(s) failed` : 'the page\'s craft holds');
process.exit(fails ? 1 : 0);
