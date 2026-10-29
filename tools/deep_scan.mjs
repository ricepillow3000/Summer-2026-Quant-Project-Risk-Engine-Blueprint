// Every inch of the page as a visitor meets it, then every control the engine has.
//
//   node tools/deep_scan.mjs <url> [outdir]        DEPTH=full: every crisis x both engines
//
// Where recruiter_walk.mjs asks "does a visitor's first five minutes work?",
// this asks "does EVERYTHING work, and does it agree with itself?":
//   1. boot      - console errors, failed requests and bad HTTP codes from the first
//                  millisecond; when the skeleton, hero and headline arrive
//   2. glide     - every in-page button: does it land on its section, and is the
//                  flight smooth (no teleport frames, no dropped frames, no reversal,
//                  no slide-change flash on the destination)?
//   3. surface   - every rendered word: unformatted placeholders, raw markup, NaN,
//                  truncated metric values, broken images, charts wider than their box
//   4. Gungnir   - the sketch's two blocks really are equal areas; the side switch
//                  relabels the sketch and leaves the plaque's maths alone
//   5. engine    - every preset loads; every allocation and estimator; vol targeting
//                  hits its target; every crisis replay names itself; the cone, the
//                  histogram and the headline agree; the same settings give the same number
//   6. book      - names leave as the book grows; the headline counts match the cards
//   7. tabs      - all nine render; every drawer opens; Balance for every flyer
//   8. map       - Risk Topology: Run x3 under each calibration, long and short; ES >= VaR,
//                  finite, breach a probability; overlays, explain mode, hover, pairings;
//                  no demo wording on a live payload
// Each check prints PASS / FAIL / WARN / INFO. Exit code is non-zero on any FAIL.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, outdir = '.'] = process.argv;
const FULL = process.env.DEPTH === 'full';
// Option lists come from the code (the dropdowns render lazily, so reading them
// back from the page misses whatever is scrolled out of the list):
//   SCAN_PRESETS / SCAN_SCENARIOS = JSON arrays, see the Live audit workflow.
const fromEnv = k => { try { return JSON.parse(process.env[k] || 'null'); } catch (e) { return null; } };
const PRESET_LIST = fromEnv('SCAN_PRESETS'), SCENARIO_LIST = fromEnv('SCAN_SCENARIOS');
const LIMIT = 120000;
const results = [];
const say = (status, name, note = '') => {
  results.push({ status, name, note });
  console.log(`${status.padEnd(4)}  ${name}${note ? ' - ' + note : ''}`);
};
const ok = (cond, name, note = '') => say(cond ? 'PASS' : 'FAIL', name, note);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.setDefaultTimeout(LIMIT);

// ---- 1. boot: listen before the first byte ----------------------------------
const consoleErrors = [], badResponses = [], failedRequests = [];
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 200)); });
page.on('pageerror', e => consoleErrors.push('pageerror: ' + String(e).slice(0, 200)));
page.on('response', r => { if (r.status() >= 400) badResponses.push(`${r.status()} ${r.url().slice(0, 120)}`); });
page.on('requestfailed', r => {
  const why = r.failure()?.errorText || '';
  // A navigation away aborts in-flight fetches; that is not a defect.
  if (!/ERR_ABORTED/.test(why)) failedRequests.push(`${why} ${r.url().slice(0, 120)}`);
});

// A rerun is finished when the status widget is gone AND no element is still
// marked stale - checked twice, because the widget can blink off mid-run.
const quiet = () => !document.querySelector('[data-testid="stStatusWidget"] *') &&
                    !document.querySelector('[data-stale="true"]');
async function settle() {
  for (let i = 0; i < 2; i++) {
    await page.waitForFunction(quiet, null, { timeout: LIMIT }).catch(() => {});
    await page.waitForTimeout(400);
  }
}
const verdict = () => page.textContent('.verdict-number').then(t => t.trim()).catch(() => '');
const sentence = () => page.textContent('.verdict-sentence').catch(() => '');
const pct = s => parseFloat(String(s).replace(/[^\d.+-]/g, ''));
const shot = name => page.screenshot({ path: `${outdir}/scan-${name}.png` }).catch(() => {});
async function step(name, fn) {
  try { await fn(); } catch (e) {
    // Keep Playwright's "waiting for ..." line: it names the element that never came.
    const lines = String(e).split('\n').filter(l => l.trim());
    say('FAIL', name, 'threw: ' + [lines[0], ...lines.filter(l => /waiting for|intercepts|not visible|resolved to/.test(l)).slice(0, 2)]
      .join(' / ').slice(0, 320));
  }
}
async function newVerdict(prev) {
  await page.waitForFunction(p => {
    const v = document.querySelector('.verdict-number')?.textContent.trim();
    return v && v !== p;
  }, prev, { timeout: LIMIT }).catch(() => {});
  await settle();
  return verdict();
}
// Long option lists are virtualised, so type to filter as a visitor would.
// A script types faster than the list opens and keystrokes can land before
// it does; fill() sets the whole filter at once, and a missed open retries.
async function pick(label, option) {
  const box = page.getByRole('combobox', { name: label }).first();
  for (let attempt = 1; ; attempt++) {
    try {
      await box.scrollIntoViewIfNeeded();
      await box.click();
      await box.fill(option);
      const opt = page.getByRole('option', { name: option, exact: true }).first();
      await opt.waitFor({ state: 'visible', timeout: 10000 });
      await opt.click({ timeout: 10000 }).catch(() => page.keyboard.press('Enter'));
      await page.keyboard.press('Escape');
      // A click that did not land leaves the old value: Escape snaps the box
      // back and nothing reruns (live audit run 30 waited 90s on that, twice).
      if (await box.inputValue() !== option) throw new Error(`"${option}" did not take in "${label}"`);
      return;
    } catch (e) {
      await page.keyboard.press('Escape');
      if (attempt === 3) throw e;
    }
  }
}
// Radios, checkboxes and toggles by their own label - a bare getByText can
// land on a hidden copy of the same words and wait forever.
const esc = t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
async function choose(text) {
  const l = page.locator('[data-testid="stRadio"] label, [data-testid="stCheckbox"] label')
    .filter({ hasText: new RegExp('^\\s*' + esc(text) + '\\s*$') }).first();
  await l.scrollIntoViewIfNeeded();
  await l.click();
}
async function openDrawer(label) {
  const s = page.locator('summary', { hasText: label }).first();
  await s.scrollIntoViewIfNeeded();
  if (!(await s.evaluate(el => el.parentElement.open))) await s.click();
}
const metric = label => page.evaluate(l => {
  const m = [...document.querySelectorAll('[data-testid="stMetric"]')]
    .find(e => (e.querySelector('[data-testid="stMetricLabel"]')?.textContent || '').trim().startsWith(l));
  return m ? m.querySelector('[data-testid="stMetricValue"]').textContent.trim() : null;
}, label);
// Plotly figures carry their real arrays on the element; read the chart, not a screenshot.
const figure = which => page.evaluate(w => {
  for (const gd of document.querySelectorAll('.js-plotly-plot')) {
    const d = gd.data || [];
    const hit = w === 'fan' ? d.some(t => t.name === 'median')
      : w === 'hist' ? d.some(t => /simulations/.test(t.hovertemplate || ''))
      : w === 'rc' ? d.some(t => t.name === 'risk share') : false;
    // _fullData, not data: plotly.js 6 ships arrays base64-encoded ({dtype, bdata})
    // in .data and decodes them into _fullData.
    if (hit) return { data: gd._fullData.map(t => ({ name: t.name, x: Array.from(t.x || []), y: Array.from(t.y || []) })),
                      shapes: (gd.layout.shapes || []).map(s => ({ x0: s.x0, y0: s.y0 })) };
  }
  return null;
}, which);
const last = a => a[a.length - 1];

// The cone, the histogram and the headline are three views of ONE simulation.
// If they disagree, one of them is lying.
async function coneAgrees(tag) {
  const v = pct(await verdict());
  const fan = await figure('fan'), hist = await figure('hist');
  if (!fan || !hist) return say('FAIL', `${tag}: cone/histogram present`, 'chart missing');
  const p5 = last(fan.data[1].y), p50 = last(fan.data.find(t => t.name === 'median').y);
  const allFinite = fan.data.every(t => t.y.every(Number.isFinite));
  ok(allFinite, `${tag}: every cone value is finite`);
  // CVaR is the mean loss beyond the 5% frontier, so it sits at or beyond the cone's floor.
  ok(v >= -p5 - 0.06, `${tag}: CVaR ${v}% sits at or beyond the cone floor (VaR ${(-p5).toFixed(1)}%)`);
  const cvLine = hist.shapes[0]?.x0;
  ok(Math.abs(-cvLine - v) <= 0.06, `${tag}: histogram CVaR line = headline`, `line ${(-cvLine).toFixed(2)}% vs ${v}%`);
  const med = pct(await metric('Median 1-year return'));
  if (Number.isFinite(med)) ok(Math.abs(med - p50) <= 0.06, `${tag}: Risk Breakdown median = cone centreline`,
                               `${med}% vs ${p50.toFixed(2)}%`);
  return { v, p5, p50 };
}

// ---- boot timeline -----------------------------------------------------------
const t0 = Date.now();
await page.goto(url, { waitUntil: 'commit' });
const when = {};
await Promise.all([
  page.waitForSelector('#pre-boot, #boot-skel, #hero', { state: 'attached' }).then(() => { when.first = Date.now() - t0; }).catch(() => {}),
  page.waitForSelector('#hero .cta-btn').then(() => { when.hero = Date.now() - t0; }).catch(() => {}),
  page.waitForSelector('.verdict-number').then(() => { when.headline = Date.now() - t0; }).catch(() => {}),
]);
await settle();
say('INFO', 'boot timeline', `first paint ${when.first}ms, hero ${when.hero}ms, headline ${when.headline}ms`);
ok(/^\d+(\.\d+)?%$/.test(await verdict()), 'headline is a percentage', await verdict());
await shot('01-landing');

// ---- 2. glide: every in-page button -------------------------------------------
const scrollerSel = '[data-testid="stMain"]';
async function glideCheck(linkSel, label) {
  const link = page.locator(linkSel).first();
  if (!(await link.count())) return say('FAIL', `button "${label}"`, 'not on the page');
  const href = await link.getAttribute('href');
  // Start where a reader would: the button on screen, page at rest.
  await link.evaluate((a, s) => {
    const sc = document.querySelector(s); const prev = sc.style.scrollBehavior;
    sc.style.scrollBehavior = 'auto';
    sc.scrollTop += a.getBoundingClientRect().top - sc.getBoundingClientRect().top - 300;
    sc.style.scrollBehavior = prev;
  }, scrollerSel);
  await page.waitForTimeout(700);
  await page.evaluate(([s, id]) => {
    const sc = document.querySelector(s), dest = document.getElementById(id);
    window.__glide = [];
    const t0 = performance.now();
    const f = now => {
      const cs = getComputedStyle(dest);
      window.__glide.push({ t: now - t0, top: sc.scrollTop, dy: dest.getBoundingClientRect().top - sc.getBoundingClientRect().top,
                            op: parseFloat(cs.opacity),
                            anims: document.getAnimations().filter(a => a.playState === 'running')
                              .map(a => a.animationName).filter(Boolean) });
      if (now - t0 < 2600) requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  }, [scrollerSel, href.slice(1)]);
  const hashBefore = await page.evaluate(() => location.hash);
  await link.click();
  await page.waitForTimeout(2900);
  const hashAfter = await page.evaluate(() => location.hash);
  const s = await page.evaluate(() => window.__glide);
  const dist = Math.abs(last(s).top - s[0].top);
  // The sampler's rAF runs before the glide's in each frame, so sample k shows
  // the position the glide set one frame earlier: step k's movement belongs
  // to the PREVIOUS interval. Pair them that way, or 17/33ms frame jitter
  // reads as speed surges that never happened.
  const steps = s.slice(1).map((p, i) => ({ d: p.top - s[i].top, dt: p.t - s[i].t,
                                            span: i ? s[i].t - s[i - 1].t : p.t - s[i].t }));
  const maxStep = Math.max(...steps.map(x => Math.abs(x.d)));
  const maxRate = Math.max(...steps.map(x => Math.abs(x.d) / Math.max(x.span, 1)));
  const moving = steps.filter(x => Math.abs(x.d) > 0.5);
  const dropped = moving.filter(x => x.dt > 50).length;
  const longest = Math.max(0, ...moving.map(x => x.dt));
  const dir = Math.sign(last(s).top - s[0].top);
  const reversals = moving.filter(x => Math.sign(x.d) === -dir && Math.abs(x.d) > 2).length;
  const minOp = Math.min(...s.map(p => p.op));
  const effects = [...new Set(s.flatMap(p => p.anims))].filter(n => n !== 'sk-sheen');
  const landed = Math.abs(last(s).dy) <= 60;
  const flight = moving.length ? Math.round(moving.reduce((a, x) => a + x.dt, 0)) : 0;
  // Smoothness: how far the fastest frame runs ahead of the average one.
  // easeInOutSine is ~1.6x; a 4x curve covers a long page in a lurch.
  const surge = flight ? maxRate / (dist / flight) : 0;
  ok(landed, `button "${label}" lands on ${href}`, `section top ${last(s).dy.toFixed(0)}px from the top of the view`);
  // The browser's own anchor jump (the glide never bound, or its listener
  // died) passes every frame check below: the scroller's smooth behaviour
  // turns it into a ~400 ms scroll. It sets the hash; the glide never does,
  // and the glide takes 1.5 s (John, 2026-10-19: "does not transition").
  ok(hashAfter === hashBefore && (dist < 80 || flight >= 1000),
     `button "${label}" is the glide, not the browser's jump`,
     `hash ${hashBefore || '(none)'} -> ${hashAfter || '(none)'}, flight ${flight}ms`);
  ok(dist < 80 || maxStep / dist <= 0.25, `button "${label}" glides (no teleport frame)`,
     `largest single frame ${maxStep.toFixed(0)}px of ${dist.toFixed(0)}px, flight ${flight}ms`);
  // A frame over 200ms is a visible stutter; a few over 50ms is headless
  // Chromium's software renderer, not the page (no long tasks were found).
  ok(longest <= 200 && dropped <= Math.max(2, moving.length * 0.1), `button "${label}" holds its frame rate`,
     `${dropped} of ${moving.length} frames over 50ms, longest ${longest.toFixed(0)}ms`);
  if (dist >= 80) say(surge <= 2.5 ? 'PASS' : 'WARN', `button "${label}" keeps an even pace`,
                      `peak speed ${surge.toFixed(1)}x the average (${(maxRate * 16.7).toFixed(0)}px per 60Hz frame)`);
  ok(reversals === 0, `button "${label}" never scrolls backwards mid-flight`, `${reversals} reversals`);
  ok(minOp >= 0.9, `button "${label}" destination never fades out (no slide-change flash)`,
     `lowest opacity ${minOp.toFixed(2)}`);
  if (effects.length) say('INFO', `button "${label}" animations during the flight`, effects.join(', '));
}
await step('glide', async () => {
  for (const [sel, label] of [
    ['#hero .cta-btn', 'Explore what we do'],
    ['a.cta-btn[href="#crisis-record"]', 'See the crisis record'],
    ['a.cta-btn[href="#the-slice"]', 'From the record to the allocation'],
    ['a.cta-btn[href="#engine"]', 'To the engine'],
    ['a[href="#analysis"]', 'Skip to the analysis'],
    ['.meleona-footer a[href="#grit-showcase"]', 'Footer: The Grit Zone'],
    ['.meleona-footer a[href="#crisis-record"]', 'Footer: Crisis Conviction'],
    ['.meleona-footer a[href="#engine"]', 'Footer: The Engine'],
  ]) { await glideCheck(sel, label); }
});

// ---- 3. surface: every word, image and chart on the page ---------------------
async function surface(tag) {
  const f = await page.evaluate(() => {
    const main = document.querySelector('[data-testid="stMain"]');
    const heads = [...main.querySelectorAll('h1, h2, .panel-head .t, summary, .showcase-eyebrow, [data-testid="stMetricLabel"]')];
    const where = el => {
      let w = '';
      for (const h of heads) {
        if (h.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) w = h.textContent.trim().slice(0, 40);
      }
      return w || 'top';
    };
    const out = { text: [], trunc: [], img: [], charts: [], cover: [] };
    const txt = main.innerText;
    const pats = [[/\*\*[^*\n]{1,40}\*\*/g, 'raw markdown'], [/<\/?(?:b|i|div|span|br|small)\b[^>]*>/g, 'raw HTML'],
                  [/\{[A-Za-z_][\w[\]'".]*\}/g, 'unformatted placeholder'], [/\bNaN\b|\bnan%|\b-?inf%|\bundefined\b|\[object \w+\]/g, 'not-a-number'],
                  [/\bNone\b(?! exists)/g, 'Python None']];
    for (const [re, kind] of pats) for (const m of txt.match(re) || []) out.text.push(`${kind}: "${m}"`);
    for (const v of main.querySelectorAll('[data-testid="stMetricValue"], [data-testid="stMetricLabel"]')) {
      if (v.offsetParent && v.scrollWidth > v.clientWidth + 1)
        out.trunc.push(`${where(v)}: "${v.textContent.trim()}" (${v.scrollWidth}px in ${v.clientWidth}px)`);
    }
    for (const im of document.images) if (im.offsetParent && !(im.complete && im.naturalWidth > 0)) out.img.push(im.src.slice(0, 80));
    for (const gd of main.querySelectorAll('.js-plotly-plot')) {
      if (!gd.offsetParent) continue;
      const svg = gd.querySelector('.main-svg'), box = gd.parentElement.getBoundingClientRect();
      if (svg && svg.getBoundingClientRect().width > box.width + 2)
        out.charts.push(`${where(gd)}: chart ${svg.getBoundingClientRect().width.toFixed(0)}px in a ${box.width.toFixed(0)}px box`);
    }
    // Every visible button and link must be the thing under its own centre.
    for (const b of main.querySelectorAll('a.cta-btn, button, [role="tab"], summary')) {
      const r = b.getBoundingClientRect();
      // Toolbar icons that only appear on hover are not on screen to cover.
      if (!b.checkVisibility({ opacityProperty: true, visibilityProperty: true }) ||
          b.closest('[data-testid="stElementToolbar"]') ||
          r.width < 2 || r.bottom < 0 || r.top > innerHeight) continue;
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      if (hit && !b.contains(hit) && !hit.contains(b)) out.cover.push(`${where(b)}: "${b.textContent.trim().slice(0, 30)}" covered by ${hit.tagName}.${hit.className}`.slice(0, 160));
    }
    // Can the visitor actually scroll sideways? Clipped overflow is not a scroll.
    const ox = getComputedStyle(main).overflowX;
    const se = document.scrollingElement;
    out.sideways = (!/hidden|clip/.test(ox) && main.scrollWidth > main.clientWidth + 1) ||
                   se.scrollWidth > se.clientWidth + 1;
    return out;
  });
  ok(!f.text.length, `${tag}: no raw markup, placeholders or NaN in the text`, f.text.slice(0, 4).join(' | '));
  ok(!f.trunc.length, `${tag}: no metric cut off by its box`, f.trunc.slice(0, 4).join(' | '));
  ok(!f.img.length, `${tag}: every image loads`, f.img.join(' | '));
  ok(!f.charts.length, `${tag}: every chart fits its box`, f.charts.slice(0, 3).join(' | '));
  ok(!f.cover.length, `${tag}: no control is covered by another element`, f.cover.slice(0, 3).join(' | '));
  ok(!f.sideways, `${tag}: no sideways scroll`);
}
// Walk the page top to bottom, a screen at a time, scanning what is on screen.
await step('surface', async () => {
  const n = await page.evaluate(s => document.querySelector(s).scrollHeight, scrollerSel);
  for (let y = 0, i = 0; y < n; y += 800, i++) {
    await page.evaluate(([s, y]) => { const sc = document.querySelector(s); sc.style.scrollBehavior = 'auto'; sc.scrollTop = y; }, [scrollerSel, y]);
    await page.waitForTimeout(450);
    if (i % 3 === 0) await shot(`02-screen-${String(i).padStart(2, '0')}`);
  }
  await surface('landing page');
});

// ---- 4. Gungnir -----------------------------------------------------------------
await step('gungnir', async () => {
  const g = await page.evaluate(() => {
    const svg = document.querySelector('.st-key-gungnir_zone svg[viewBox="0 0 360 200"]');
    const rects = [...svg.querySelectorAll('rect')].map(r => ({ w: +r.getAttribute('width'), h: +r.getAttribute('height') }));
    return { rects, labels: [...svg.querySelectorAll('text')].map(t => t.textContent),
             plaque: document.querySelector('.gungnir-plaque').textContent };
  });
  const [a, b] = g.rects;
  ok(a.w * a.h === b.w * b.h, 'Gungnir: the two blocks have equal area (w1 s1 = w2 s2)', `${a.w}x${a.h} vs ${b.w}x${b.h}`);
  ok(Math.max(a.w / b.w, b.w / a.w) === Math.max(a.h / b.h, b.h / a.h) && Math.max(a.w / b.w, b.w / a.w) === 3,
     'Gungnir: the sketch is the plaque\'s 3:1 sentence', 'three times as volatile gets one third the capital');
  ok(/w1σ1=w2σ2|w1σ1\s*=\s*w2σ2/.test(g.plaque.replace(/block 0[12]/g, '').replace(/\s+/g, '')), 'Gungnir: plaque states w1σ1 = w2σ2');
  ok(g.labels.includes('the high-flyer') && g.labels.includes('the steady cushion'), 'Gungnir: long-side labels on the sketch');
});

// ---- 5. engine ------------------------------------------------------------------
let base = null;
await step('engine baseline', async () => {
  base = await coneAgrees('baseline');
  const again = await verdict();
  ok(again === `${base.v}%` || pct(again) === base.v, 'baseline: the headline is stable without input', again);
});

await step('allocation', async () => {
  await openDrawer('02 · Allocation');
  for (const est of ['Sample - plain history', 'EWMA - reactive / panic lens', 'Ledoit-Wolf - steady (default)']) {
    const prev = await verdict();
    await pick('Covariance estimator', est);
    await settle();
    const v = await verdict();
    ok(/^\d+(\.\d+)?%$/.test(v), `allocation: estimator "${est.split(' -')[0]}" gives a finite headline`, `${prev} -> ${v}`);
  }
  // Risk parity must make every risk share equal - that is its definition.
  let prev = await verdict();
  await choose('Risk parity');
  const rp = await newVerdict(prev);
  const rc = await figure('rc');
  const shares = rc.data.find(t => t.name === 'risk share').x;
  const spread = Math.max(...shares) - Math.min(...shares);
  ok(spread <= 1.0, 'allocation: risk parity equalises every risk share', `shares ${shares.map(s => s.toFixed(1)).join('/')}%`);
  await coneAgrees('risk parity');
  say('INFO', 'allocation: risk parity headline', `${prev} -> ${rp}`);
  prev = rp;
  await choose('Equal weight');
  await newVerdict(prev);
  const ew = (await figure('rc')).data.find(t => t.name === 'dollar weight').x;
  ok(Math.max(...ew) - Math.min(...ew) <= 0.05, 'allocation: equal weight gives equal dollar weights', `${ew.map(s => s.toFixed(1)).join('/')}%`);
  // Vol targeting: the realised vol of the scaled book should land on the target.
  prev = await verdict();
  await choose('Target volatility');
  await newVerdict(prev);
  await settle();
  const vol = pct(await metric('Annualized volatility'));
  const s = await sentence();
  ok(Math.abs(vol - 10) <= 1.5, 'allocation: vol target 10% lands near 10% realised', `${vol}% realised`);
  ok(/levered \d+\.\d\d×/.test(s), 'allocation: the verdict names the leverage', (s.match(/levered [\d.]+×/) || [''])[0]);
  prev = await verdict();
  await choose('Target volatility');
  const back = await newVerdict(prev);
  ok(pct(back) === base.v, 'allocation: switching everything back restores the baseline exactly', `${back} vs ${base.v}%`);
});

await step('stress test', async () => {
  await openDrawer('03 · Stress test');
  // A deeper drawdown shock must not make the tail lighter.
  const slider = page.getByRole('slider', { name: 'Market drawdown shock' }).first();
  await slider.scrollIntoViewIfNeeded();
  let prev = await verdict();
  await slider.focus();
  for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowLeft');
  const shocked = await newVerdict(prev);
  ok(pct(shocked) > base.v, 'stress: a -30% drawdown shock deepens the tail', `${base.v}% -> ${shocked}`);
  ok(/under the stress scenario/.test(await sentence()), 'stress: the verdict says a shock is applied');
  await coneAgrees('custom shock');
  prev = shocked;
  await slider.focus();
  for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowRight');
  const unshocked = await newVerdict(prev);
  ok(pct(unshocked) === base.v, 'stress: removing the shock restores the baseline exactly', unshocked);

  if (!SCENARIO_LIST) return say('FAIL', 'stress: crisis list', 'SCAN_SCENARIOS not set');
  const scenarios = SCENARIO_LIST;
  const engines = FULL ? ['Bootstrap (empirical)', 'Jump-diffusion (Merton)'] : ['Bootstrap (empirical)'];
  for (const eng of engines) {
    await choose(eng);
    await settle();
    for (const sc of scenarios) {
      prev = await verdict();
      await pick('Scenario', sc);
      await page.waitForFunction(n => (document.querySelector('.verdict-sentence')?.textContent || '').includes(n),
                                 sc, { timeout: LIMIT }).catch(() => {});
      await settle();
      const sen = await sentence(), v = await verdict();
      const days = (sen.match(/\((\d+) trading days\)/) || [])[1];
      // The verb follows the engine (ME-07): jump-diffusion replays nothing.
      const jd = eng.startsWith('Jump');
      const said = jd ? `calibrated to the actual returns of ${sc}` : `Replaying the actual returns of ${sc}`;
      ok(sen.includes(said) && /^-?\d+(\.\d+)?%$/.test(v),
         `stress [${eng.split(' ')[0]}]: ${sc} replays and names itself`, `CVaR ${v}, ${days} trading days`
         + (/Excludes/.test(sen) ? `, ${sen.match(/Excludes ([^-]+)-/)?.[1].trim()} excluded and disclosed` : ''));
      const cap = await page.locator('[data-testid="stCaptionContainer"]', { hasText: jd ? 'Calibrating to actual market returns from' : 'Replaying actual market returns from' }).first().textContent().catch(() => '');
      ok(/\d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d/.test(cap), `stress: ${sc} window shown`, cap.trim());
      await coneAgrees(`${sc}`);
    }
  }
  await pick('Scenario', 'Custom shock (sliders)');
  await choose('Bootstrap (empirical)');
  await settle();
  ok(pct(await verdict()) === base.v, 'stress: back to the base case restores the baseline exactly', await verdict());
  await shot('03-engine');
});

await step('every preset', async () => {
  await openDrawer('01 · Universe');
  if (!PRESET_LIST) return say('FAIL', 'presets: list', 'SCAN_PRESETS not set');
  const names = PRESET_LIST;
  const list = FULL ? names : names.slice(0, 4);
  say('INFO', 'presets on offer', `${names.length}; scanning ${list.length}`);
  for (const p of list.slice(1).concat(list[0])) {
    const prev = await verdict();
    await pick('Preset basket', p);
    const v = await newVerdict(prev);
    const errs = await page.locator('[data-testid="stAlertContentError"], [data-testid="stException"]').count();
    const data = await page.locator('[data-testid="stCaptionContainer"]', { hasText: 'daily returns) · through' }).first().textContent().catch(() => '');
    const through = (data.match(/through (\d{4}-\d\d-\d\d)/) || [])[1];
    const age = through ? (Date.now() - Date.parse(through)) / 864e5 : NaN;
    const warns = await page.locator('[data-testid="stAlertContentWarning"]').allTextContents();
    ok(!errs && /^-?\d+(\.\d+)?%$/.test(v), `preset "${p}" loads`, `CVaR ${v}, ${data.trim()}`);
    ok(age <= 6, `preset "${p}" data is current`, `through ${through}`);
    // Warnings here are the product talking (a concentration reading, a
    // disclosed exclusion) - reported, not judged; errors are counted above.
    if (warns.length) say('INFO', `preset "${p}" notices`, warns.map(w => w.slice(0, 140)).join(' | '));
    if (FULL) await mapQuick(p);
  }
});

// ---- 6. the book ------------------------------------------------------------------
async function bookState() {
  return page.evaluate(() => {
    // A dossier card is the flex row whose first cell is its role label.
    const ROLES = ['LEADS', 'ANCHOR', 'NO ROLE', 'LEFT OUT'];
    const cards = [...document.querySelectorAll('[data-testid="stMarkdownContainer"] div')]
      .filter(d => d.children.length === 2 && ROLES.includes(
        (d.firstElementChild.textContent || '').trim()
          .replace(d.firstElementChild.querySelector('.role-note')?.textContent.trim() || '', '').trim()));
    // the role cell holds the label, and on a NO ROLE card the weight it holds
    const note = c => c.firstElementChild?.querySelector('.role-note')?.textContent.trim() || '';
    const lbl = c => (c.firstElementChild?.textContent || '').trim().replace(note(c), '').trim();
    const head = [...document.querySelectorAll('[data-testid="stMarkdownContainer"] div')]
      .map(d => d.textContent).find(t => /^\$[\d,]+ book - \d+ of your \d+ names/.test(t)) || '';
    return { leads: cards.filter(c => lbl(c) === 'LEADS').length, anchors: cards.filter(c => lbl(c) === 'ANCHOR').length,
             held: cards.filter(c => lbl(c) === 'NO ROLE').length,
             // every NO ROLE card states its weight, and no other card does
             noteOk: cards.every(c => (lbl(c) === 'NO ROLE')
               === /^(\d+\.\d%|<0\.1%) of the book$/.test(note(c))),
             out: cards.filter(c => lbl(c) === 'LEFT OUT').length, total: cards.length, head,
             outside: cards.filter(c => /outside the chosen universe/.test(c.textContent)).length };
  });
}
await step('the book', async () => {
  // "n of N names hold a position worth keeping" counts the LIQUIDITY gate:
  // an eligible name can still hold no role for ranking below the cut, so
  // leaders + anchors <= eligible <= N. N is the visitor's own names (ME-08):
  // an outside anchor's card is on screen but not one of them, so it counts
  // in neither N nor the anchors that must fit inside the eligible names.
  const counts = b => (b.head.match(/(\d+) of your (\d+) names/) || []).slice(1).map(Number);
  const own = b => b.total - b.outside, ownAnchors = b => b.anchors - b.outside;
  const cards = b => `cards: ${b.leads} lead, ${b.anchors} anchor, ${b.held} held without a role, ${b.out} left out, ${b.outside} from outside`;
  // Leaders and anchors sit under the headline; the other cards and the two
  // finer settings sit in one drawer below them.
  await openDrawer('The other names, line by line');
  const b0 = await bookState();
  const [e0, n0] = counts(b0);
  ok(b0.total > 0, 'book: dossier cards render', cards(b0));
  ok(b0.noteOk, 'book: every card without a role states the weight it holds', cards(b0));
  ok(n0 === own(b0) && b0.leads + ownAnchors(b0) <= e0 && e0 <= n0, 'book: the headline counts agree with the cards',
     `${b0.head.slice(0, 70)} | ${cards(b0)}`);
  const size = page.locator('input[aria-label="Book size ($)"]').first();
  await size.scrollIntoViewIfNeeded();
  let prevElig = e0;
  for (const v of ['100000000', '5000000000']) {
    await size.fill(v); await size.press('Enter');
    await page.waitForFunction(d => [...document.querySelectorAll('div')].some(e => e.textContent.startsWith(d + ' book')),
                               '$' + Number(v).toLocaleString('en-US'), { timeout: LIMIT }).catch(() => {});
    await settle();
    const b = await bookState();
    const [e, n] = counts(b);
    const $v = '$' + Number(v).toLocaleString('en-US');
    ok(e <= prevElig, `book: at ${$v} names only leave, never join`, `${prevElig} -> ${e} eligible`);
    ok(n === own(b) && b.leads + ownAnchors(b) <= e, `book: headline still agrees with the cards at ${$v}`, cards(b));
    prevElig = e;
  }
  await size.fill('1000000'); await size.press('Enter'); await settle();
  await choose('Flyers must be in the grittiest half'); await settle();
  const bf = await bookState();
  ok(bf.leads <= b0.leads, 'book: the grit floor can only narrow the leaders', `${b0.leads} -> ${bf.leads}`);
  await choose('Flyers must be in the grittiest half'); await settle();
  await choose('Allow a defensive anchor from outside the universe'); await settle();
  const bn = await bookState();
  ok(bn.outside === 0, 'book: with the outside anchor off, nothing from outside appears', `${bn.outside} outside cards`);
  await choose('Allow a defensive anchor from outside the universe'); await settle();
});

// ---- 7. tabs, drawers, Balance ----------------------------------------------------
const TABS = ['Risk Topology', 'Risk Breakdown', 'Liquidity', 'Correlation Watch', 'Balance',
              'Grit Zone', 'Crisis Conviction', 'Data Quality', 'Lineage & Audit'];
// The last two live in the folded provenance drawer under the strip.
const PROVENANCE = ['Data Quality', 'Lineage & Audit'];
async function openTab(name) {
  if (PROVENANCE.includes(name)) await openDrawer('Where the numbers come from');
  const t = page.getByRole('tab', { name, exact: true }).first();
  await t.scrollIntoViewIfNeeded(); await t.click(); await page.waitForTimeout(900);
}
await step('tabs', async () => {
  for (const name of TABS) {
    await openTab(name);
    const len = await page.evaluate(n => {
      const t = [...document.querySelectorAll('[role="tab"]')].find(e => e.innerText.trim() === n);
      const p = t && document.getElementById(t.getAttribute('aria-controls'));
      return p ? p.innerText.trim().length + p.querySelectorAll('iframe, canvas, svg').length * 100 : 0;
    }, name);
    ok(len > 80, `tab "${name}" renders`, `${len} chars`);
    await surface(`tab ${name}`);
    await shot(`04-tab-${name.replace(/\W+/g, '-')}`);
  }
  // Every drawer on the page opens and has something inside.
  const n = await page.locator('summary').count();
  let empty = [];
  for (let i = 0; i < n; i++) {
    const s = page.locator('summary').nth(i);
    if (!(await s.isVisible())) continue;
    const label = (await s.textContent()).trim().slice(0, 50);
    if (!(await s.evaluate(el => el.parentElement.open))) { await s.click(); await page.waitForTimeout(300); }
    const len = await s.evaluate(el => el.parentElement.innerText.length - el.innerText.length);
    if (len < 40) empty.push(label);
  }
  ok(!empty.length, 'every visible drawer opens onto content', empty.join(' | '));
});

await step('balance, every flyer', async () => {
  await openTab('Balance');
  const box = page.getByRole('combobox', { name: /High-flyer/ }).first();
  await box.scrollIntoViewIfNeeded(); await box.click();
  const flyers = (await page.getByRole('option').allTextContents()).map(s => s.trim());
  await page.keyboard.press('Escape');
  for (const f of (FULL ? flyers : flyers.slice(0, 3))) {
    await pick('High-flyer (Circle 1 - the volatile conviction position)', f); await settle();
    const t = await page.evaluate(() => [...document.querySelectorAll('svg text')].map(e => e.textContent).find(s => /^GAP /.test(s)) || '');
    const line = await page.evaluate(() => [...document.querySelectorAll('div')].map(d => d.textContent)
      .find(s => /^Over the \d+ daily returns loaded/.test(s)) || '');
    ok(/^GAP -?\d+(\.\d+)?%$/.test(t) && /cushion/.test(line), `balance: flyer ${f} pairs and measures its cushion`,
       `${t}; ${line.slice(0, 110)}`);
  }
});

// ---- 8. Risk Topology ----------------------------------------------------------------
async function mapFrame() {
  for (const f of page.frames()) if (await f.$('#cvTrail').catch(() => null)) return f;
  return null;
}
async function mapRun(f) {
  await f.click('#btnRun');
  await f.waitForFunction(() => !document.getElementById('btnRun').disabled, null, { timeout: LIMIT });
  return f.evaluate(() => ({ es: document.getElementById('hudES').textContent, var95: document.getElementById('hudVaR').textContent,
                             breach: document.getElementById('hudTail').textContent }));
}
function inkOf(f) {
  return f.evaluate(() => [...document.querySelectorAll('canvas')].map(c => {
    try { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let s = 0;
          for (let i = 3; i < d.length; i += 97) s += d[i]; return s; } catch (e) { return -1; }
  }));
}
// Every universe gets its own map: open it, let it boot, and read what it says.
async function mapQuick(preset) {
  await openTab('Risk Topology');
  const f = await mapFrame();
  if (!f) return say('FAIL', `map for "${preset}"`, 'no map frame');
  await f.waitForFunction(() => !document.getElementById('btnRun').disabled && /\d/.test(document.getElementById('hudES').textContent),
                          null, { timeout: LIMIT }).catch(() => {});
  const m = await f.evaluate(() => ({ es: document.getElementById('hudES').textContent, v: document.getElementById('hudVaR').textContent,
    br: document.getElementById('hudTail').textContent, prov: document.getElementById('chipProv').textContent,
    body: document.body.innerText, units: (document.getElementById('mapSummary').textContent.match(/plots (\d+) assets/) || [])[1] }));
  const ink = (await inkOf(f)).reduce((a, x) => a + Math.max(x, 0), 0);
  const es = pct(m.es), v = pct(m.v), br = pct(m.br);
  ok(Number.isFinite(es) && Math.abs(es) >= Math.abs(v) && br >= 0 && br <= 100 && ink > 0
     && !/demo/i.test(m.prov) && !/\bNaN\b|undefined|Infinity/.test(m.body),
     `map for "${preset}" boots live and finite`, `ES ${m.es}, VaR ${m.v}, breach ${m.br}, ${m.units} assets plotted`);
}
async function mapAudit(side) {
  await openTab('Risk Topology');
  const f = await mapFrame();
  if (!f) return say('FAIL', `map [${side}]: frame present`, 'no #cvTrail');
  // The map's own frame: the glide's zero-height frame comes first on the page.
  await (await f.frameElement()).scrollIntoViewIfNeeded().catch(() => {});
  const words = await f.evaluate(() => ({ chips: document.getElementById('chipProv').textContent,
    tag: document.getElementById('provTag').textContent, foot: document.getElementById('paramsFoot').textContent,
    body: document.body.innerText }));
  ok(!/demo/i.test(words.chips + words.tag + words.foot), `map [${side}]: no demo wording on the live map`,
     `${words.chips} | ${words.foot.slice(0, 80)}`);
  ok(!/\bNaN\b|undefined|Infinity/.test(words.body), `map [${side}]: no NaN/undefined on the map`);
  const runs = {};
  for (const cal of ['calm', 'base', 'stress']) {
    await f.selectOption('#selCal', cal);
    await f.waitForFunction(() => !document.getElementById('btnRun').disabled, null, { timeout: LIMIT });
    runs[cal] = [];
    for (let i = 0; i < 3; i++) {
      const r = await mapRun(f);
      const es = pct(r.es), v = pct(r.var95), br = pct(r.breach);
      runs[cal].push(es);
      ok(Number.isFinite(es) && Number.isFinite(v) && Math.abs(es) >= Math.abs(v) && br >= 0 && br <= 100,
         `map [${side}] ${cal} run ${i + 1}: ES >= VaR, breach a probability`, `ES ${r.es}, VaR ${r.var95}, breach ${r.breach}`);
    }
    const sp = Math.max(...runs[cal].map(Math.abs)) - Math.min(...runs[cal].map(Math.abs));
    ok(sp <= 1.0, `map [${side}] ${cal}: three runs agree within Monte Carlo noise`, `spread ${sp.toFixed(1)}pp`);
  }
  const mean = c => runs[c].reduce((a, x) => a + Math.abs(x), 0) / 3;
  const order = mean('calm') <= mean('base') + 0.3 && mean('base') <= mean('stress') + 0.3;
  say(order ? 'PASS' : 'WARN', `map [${side}]: calm <= baseline <= stressed`,
      `${mean('calm').toFixed(1)} / ${mean('base').toFixed(1)} / ${mean('stress').toFixed(1)}% ES`);
  await f.selectOption('#selCal', 'base');
  await f.waitForFunction(() => !document.getElementById('btnRun').disabled, null, { timeout: LIMIT });
  // Overlays: each toggle must change the picture, and switching back must restore it.
  for (const id of ['tgAtmo', 'tgMarks', 'tgLinks']) {
    const before = await inkOf(f);
    await f.click(`label:has(#${id}) .pip`);
    await page.waitForTimeout(250);
    const off = await inkOf(f);
    await f.click(`label:has(#${id}) .pip`);
    await page.waitForTimeout(250);
    const on = await inkOf(f);
    ok(JSON.stringify(before) !== JSON.stringify(off) && JSON.stringify(before) === JSON.stringify(on),
       `map [${side}]: overlay ${id} toggles off and back`);
  }
  // Hover and explain.
  const box = await f.evaluate(() => { const r = document.getElementById('stage').getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  const fr = await (await f.frameElement()).boundingBox();
  const cx = fr.x + box.x + box.w * 0.45, cy = fr.y + box.y + box.h * 0.5;
  await page.mouse.move(cx, cy); await page.waitForTimeout(250);
  const tip = await f.evaluate(() => getComputedStyle(document.getElementById('tip')).display !== 'none'
    ? document.getElementById('tip').innerText : '');
  ok(/β/.test(tip) && /VaR95/.test(tip), `map [${side}]: hover reads the terrain`, tip.replace(/\s+/g, ' ').slice(0, 120));
  // A click alone must show something: the button once only switched on
  // hover-reading over the plot, so a click with no hover looked dead (John,
  // 2026-10-19). The glossary opens under it, inside the frame's view.
  await f.click('#btnExplain'); await page.waitForTimeout(200);
  const gl = await f.evaluate(() => {
    const g = document.getElementById('glossary'), r = g.getBoundingClientRect();
    const side = document.getElementById('sidePanel').getBoundingClientRect();
    return { open: !g.hidden && r.height > 40, inView: r.top < Math.min(innerHeight, side.bottom) - 40 && r.bottom > 0,
             e: document.getElementById('btnExplain').getAttribute('aria-expanded'),
             words: g.innerText.slice(0, 60) };
  });
  ok(gl.open && gl.inView && gl.e === 'true', `map [${side}]: a click on Explain (no hover) opens the glossary in view`,
     `${gl.words.replace(/\s+/g, ' ')}`);
  await page.mouse.move(cx + 3, cy + 3); await page.waitForTimeout(250);
  const ex = await f.evaluate(() => ({ t: document.getElementById('tip').innerText }));
  ok(/explain mode/i.test(ex.t), `map [${side}]: explain mode names what is under the cursor`, ex.t.replace(/\s+/g, ' ').slice(0, 120));
  await f.click('#btnExplain');
  const shut = await f.evaluate(() => document.getElementById('glossary').hidden);
  ok(shut, `map [${side}]: a second click closes the glossary`);
  // Every pairing in the league table opens its own detail with finite numbers.
  const rows = await f.locator('.srow').count();
  for (let i = 0; i < rows; i++) {
    await f.locator('.srow').nth(i).click(); await page.waitForTimeout(200);
    const d = await f.evaluate(() => document.getElementById('detail').innerText);
    ok(!/NaN|undefined|Infinity/.test(d) && /ES 97\.5/.test(d), `map [${side}]: pairing ${i + 1} of ${rows} opens a finite detail`,
       d.split('\n').slice(0, 3).join(' ').slice(0, 90));
  }
  await shot(`05-map-${side}`);
}
await step('map long', () => mapAudit('long'));

await step('short side', async () => {
  const prev = await verdict();
  await choose('Bearish - short the book');
  await page.waitForFunction(() => /short/i.test(document.querySelector('.verdict-sentence')?.textContent || ''));
  await settle();
  const g = await page.evaluate(() => [...document.querySelectorAll('.st-key-gungnir_zone svg text')].map(t => t.textContent));
  ok(g.includes('the short position') && g.includes('the squeeze cushion'), 'short: the Gungnir sketch relabels');
  ok(/short book/.test(await page.textContent('#engine')), 'short: the engine heading says short book');
  say('INFO', 'short: headline', `${prev} -> ${await verdict()}`);
  await coneAgrees('short book');
  await surface('short book');
});
await step('map short', () => mapAudit('short'));

// ---- wrap up ---------------------------------------------------------------------------
await step('clear session', async () => {
  await openTab('Lineage & Audit');
  await page.getByRole('button', { name: "Clear this session's data" }).click();
  await page.waitForSelector('.verdict-number'); await settle();
  ok(/^\d+(\.\d+)?%$/.test(await verdict()), 'clearing the session reloads the defaults', await verdict());
});
const noise = s => /favicon|Failed to load resource: the server responded with a status of 404/.test(s);
ok(!consoleErrors.filter(e => !noise(e)).length, 'no console or script errors from boot to finish', consoleErrors.slice(0, 4).join(' | '));
ok(!badResponses.length, 'no request answered with an error status', badResponses.slice(0, 4).join(' | '));
ok(!failedRequests.length, 'no request failed outright', failedRequests.slice(0, 4).join(' | '));
await browser.close();

const n = s => results.filter(r => r.status === s).length;
console.log(`\n${results.length} checks: ${n('PASS')} pass, ${n('FAIL')} fail, ${n('WARN')} warn, ${n('INFO')} info`);
process.exit(n('FAIL') ? 1 : 0);
