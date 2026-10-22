// Every button glide lands - also while the page is still loading.
//
//   node tools/glide_load_check.mjs <url> [early|momentum|reader|steady|rest|all] [repeats]
//
// deep_scan.mjs times each button on a settled page, where every glide passed
// (480/480 on the offline harness, ME-04). The failures lived elsewhere:
//   early     a fresh visit, each button pressed 200 ms after it appears, while
//             the script is still running: targets not written yet (dead
//             presses), a page still growing under the flight (one-frame leaps,
//             stops at the bottom of a short page);
//   momentum  a trackpad's flick still arriving after the press (the flight was
//             cancelled 30/30);
//   reader    a swipe back mid-flight still stops the flight at once;
//   steady    a steady mouse wheel turned through the press stops it too;
//   rest      the settled page, so a fix for the two above breaks nothing.
// Each press is sampled every frame. A press fails when it is DEAD (never
// moves), TELEPORTS (one frame covers >25% of the flight), or STOPS SHORT
// (the section is not within 60px of the top of the view when it ends).
// Exit code 1 on any failure. Point it at a cold server for "early" to mean
// anything: a warm cache renders the whole page before the first press.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, which = 'all', reps = '4'] = process.argv;
const N = +reps;
const SC = '[data-testid="stMain"]';
const LINKS = [
  ['#hero .cta-btn', 'Explore what we do'],
  ['a.cta-btn[href="#crisis-record"]', 'See the crisis record'],
  ['a.cta-btn[href="#the-slice"]', 'From the record to the allocation'],
  ['a.cta-btn[href="#engine"]', 'To the engine'],
  ['a[href="#analysis"]', 'Skip to the analysis'],
  ['.meleona-footer a[href="#grit-showcase"]', 'Footer 01'],
  ['.meleona-footer a[href="#crisis-record"]', 'Footer 02'],
  ['.meleona-footer a[href="#engine"]', 'Footer 03'],
];
const LOADING = LINKS.slice(0, 5);     // the footer exists only once the run has ended

const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const ctx = await b.newContext({ viewport: { width: 1440, height: 900 } });
const tally = {}, fails = [];

async function sample(page, id, ms) {
  await page.evaluate(([s, id, ms]) => {
    window.__g = []; const t0 = performance.now();
    const f = now => {
      const sc = document.querySelector(s), d = document.getElementById(id);
      window.__g.push({ t: now - t0, top: sc.scrollTop,
                        dy: d ? d.getBoundingClientRect().top - sc.getBoundingClientRect().top : null });
      if (now - t0 < ms) requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  }, [SC, id, ms]);
}
function judge(s) {
  const dist = Math.abs(s[s.length - 1].top - s[0].top);
  const steps = s.slice(1).map((p, i) => Math.abs(p.top - s[i].top));
  const maxStep = Math.max(0, ...steps);
  const end = s[s.length - 1].dy, first = s.find(p => p.dy !== null);
  if (first && Math.abs(first.dy) <= 60 && dist < 5) return { v: 'ok', dist, maxStep, end };
  if (dist < 5) return { v: 'DEAD', dist, maxStep, end };
  if (dist >= 80 && maxStep / dist > 0.25) return { v: 'TELEPORT', dist, maxStep, end };
  if (end === null || Math.abs(end) > 60) return { v: 'SHORT', dist, maxStep, end };
  return { v: 'ok', dist, maxStep, end };
}
async function press(page, sel, label, mode, ms, act) {
  const href = await page.locator(sel).first().getAttribute('href');
  // Start where a reader would: the button on screen, 300px down the view.
  await page.locator(sel).first().evaluate((a, s) => {
    const sc = document.querySelector(s); sc.style.scrollBehavior = 'auto';
    sc.scrollTop += a.getBoundingClientRect().top - sc.getBoundingClientRect().top - 300;
    sc.style.scrollBehavior = '';
  }, SC);
  await page.waitForTimeout(150);
  await sample(page, href.slice(1), ms);
  await act(page.locator(sel).first());
  await page.waitForTimeout(ms + 100);
  const r = judge(await page.evaluate(() => window.__g));
  const k = `${mode.padEnd(8)} ${label}`;
  tally[k] = tally[k] || { ok: 0, DEAD: 0, TELEPORT: 0, SHORT: 0 };
  tally[k][r.v]++;
  if (r.v !== 'ok') fails.push({ mode, label, ...r });
  await page.evaluate(() => history.replaceState(null, '', location.pathname));
}
async function settled(page) {
  await page.goto(url);
  await page.waitForSelector('.meleona-footer', { timeout: 300000 });
  await page.waitForFunction(() => !!document.__meleonaGlide, null, { timeout: 60000 });
  await page.waitForTimeout(1500);
}

// ONLY=3 (or a label's words) presses one button: the Live audit restarts the
// app on an empty cache before each one, so every early press is cold.
const only = process.env.ONLY;
const pick = list => only ? list.filter((x, i) => String(i + 1) === only || x[1].includes(only)) : list;
if (which === 'early' || which === 'all') {
  for (const [sel, label] of pick(LOADING)) for (let i = 0; i < N; i++) {
    const page = await ctx.newPage();
    await page.goto(url);
    await page.waitForSelector(sel, { timeout: 120000 });
    await page.waitForTimeout(200);
    // a press may wait for its section, then fly: give it 12 s to land
    await press(page, sel, label, 'early', 12000, l => l.click({ force: true }));
    await page.close();
  }
}
if (which === 'momentum' || which === 'all') {
  const page = await ctx.newPage(); await settled(page);
  for (const [sel, label] of LOADING) for (let i = 0; i < N; i++) {
    await press(page, sel, label, 'momentum', 3200, async l => {
      // a flick, then the press while its decaying tail keeps arriving (~1 s)
      const box = await l.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      let d = 24;
      for (let k = 0; k < 4; k++) { await page.mouse.wheel(0, d); await page.waitForTimeout(16); }
      // a hand presses where the button is now, without Playwright's wait
      // for it to stop moving (that wait put a pause into the flick no
      // trackpad ever makes)
      const now = await l.boundingBox();
      await page.mouse.click(now.x + now.width / 2, now.y + now.height / 2);
      // a real tail: decaying, in WHOLE pixels, so the small deltas repeat
      // (2, 2, 1, 1) - float deltas hid that the strict-decay rule cancelled
      // every real tail (Council B, 2026-10-22)
      for (let k = 0; k < 50 && d >= 0.6; k++) { d *= 0.93; await page.mouse.wheel(0, Math.max(1, Math.round(d))); await page.waitForTimeout(16); }
    });
  }
  await page.close();
}
if (which === 'reader' || which === 'all') {
  // the other side of momentum: a reader who swipes back mid-flight must get
  // the page at once - the flight stops instead of landing
  const page = await ctx.newPage(); await settled(page);
  for (const [sel, label] of LOADING) for (let i = 0; i < N; i++) {
    const href = await page.locator(sel).first().getAttribute('href');
    await page.locator(sel).first().evaluate((a, s) => {
      const sc = document.querySelector(s); sc.style.scrollBehavior = 'auto';
      sc.scrollTop += a.getBoundingClientRect().top - sc.getBoundingClientRect().top - 300;
      sc.style.scrollBehavior = '';
    }, SC);
    await page.waitForTimeout(400);
    await page.locator(sel).first().click();
    await page.waitForTimeout(500);
    for (let k = 0; k < 3; k++) { await page.mouse.wheel(0, -60); await page.waitForTimeout(16); }
    await page.waitForTimeout(2000);
    const dy = await page.evaluate(([s, id]) => document.getElementById(id).getBoundingClientRect().top
      - document.querySelector(s).getBoundingClientRect().top, [SC, href.slice(1)]);
    const k = `reader   ${label}`;
    tally[k] = tally[k] || { ok: 0, DEAD: 0, TELEPORT: 0, SHORT: 0 };
    // landed anyway = the reader was overruled
    if (Math.abs(dy - 26) > 40) tally[k].ok++; else { tally[k].DEAD++; fails.push({ mode: 'reader', label, v: 'IGNORED-THE-READER', dy }); }
  }
  await page.close();
}
if (which === 'steady' || which === 'all') {
  // A mouse wheel turned at a steady 100px a notch is the reader scrolling,
  // never a trackpad's decaying tail (Council B, 2026-10-22: a press during a
  // steady wheel flew 2.7k px UP against it). The flight must end by the
  // second notch after the press, and the page must never move against the
  // wheel. Footer targets sit above (the wheel opposes the flight); the CTAs'
  // sit below (same direction, but not decaying).
  const page = await ctx.newPage(); await settled(page);
  for (const [sel, label] of LINKS) for (let i = 0; i < N; i++) {
    const l = page.locator(sel).first();
    await l.evaluate((a, s) => {
      const sc = document.querySelector(s); sc.style.scrollBehavior = 'auto';
      sc.scrollTop += a.getBoundingClientRect().top - sc.getBoundingClientRect().top - 500;
      sc.style.scrollBehavior = '';
    }, SC);
    await page.waitForTimeout(400);
    let box = await l.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    for (let k = 0; k < 3; k++) { await page.mouse.wheel(0, 100); await page.waitForTimeout(100); }
    box = await l.boundingBox();
    await page.evaluate(s => { window.__ys = []; const f = () => { window.__ys.push(document.querySelector(s).scrollTop);
      if (window.__ys.length < 240) requestAnimationFrame(f); }; requestAnimationFrame(f); }, SC);
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    for (let k = 0; k < 2; k++) { await page.mouse.wheel(0, 100); await page.waitForTimeout(100); }
    const flying = await page.evaluate(() => !!document.__meleonaStop);
    for (let k = 0; k < 10; k++) { await page.mouse.wheel(0, 100); await page.waitForTimeout(100); }
    await page.waitForTimeout(1500);
    const ys = await page.evaluate(() => window.__ys);
    // after the flight has had two notches to yield, count frames that moved UP
    const up = ys.slice(20).reduce((n, y, k, a) => n + (k && y < a[k - 1] - 3 ? 1 : 0), 0);
    const k = `steady   ${label}`;
    tally[k] = tally[k] || { ok: 0, DEAD: 0, TELEPORT: 0, SHORT: 0 };
    if (!flying && up === 0) tally[k].ok++;
    else { tally[k].DEAD++; fails.push({ mode: 'steady', label, v: 'OVERRULED-THE-WHEEL', stillFlying: flying, framesAgainstWheel: up }); }
    await page.evaluate(() => history.replaceState(null, '', location.pathname));
  }
  await page.close();
}
if (which === 'rest' || which === 'all') {
  const page = await ctx.newPage(); await settled(page);
  for (const [sel, label] of LINKS) for (let i = 0; i < N; i++)
    await press(page, sel, label, 'rest', 3200, l => l.click());
  await page.close();
}
await b.close();
for (const [k, t] of Object.entries(tally))
  console.log(`${k.padEnd(46)} ok ${t.ok}  dead ${t.DEAD}  teleport ${t.TELEPORT}  short ${t.SHORT}`);
for (const f of fails) console.log('FAIL', JSON.stringify(f));
const n = Object.values(tally).reduce((a, t) => a + t.ok + t.DEAD + t.TELEPORT + t.SHORT, 0);
console.log(`${n - fails.length}/${n} presses passed`);
process.exit(fails.length ? 1 : 0);
