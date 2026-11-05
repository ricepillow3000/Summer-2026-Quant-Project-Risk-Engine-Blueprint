// What a visitor's browser sees, checked - on any deployment of the app.
//
//   node tools/site_check.mjs <url> [outdir]
//
// At a desktop and a phone viewport: the boot skeleton outlines the real hero
// (every crest/title/sub/button/tile box within 2px of the element it stands
// in for), Streamlit's top bar stays hidden, the headline is a finite
// percentage, the data is current, nothing on the page reads NaN/inf, the
// page does not scroll sideways, and no script error fires. The bottom dock
// (static/dock.js): the cookie notice shows on a first visit, clear of the
// main CTA; on the phone the "Run the risk engine" bar comes up past the hero,
// never under or over the notice, glides to the engine and goes there. Screenshots land
// in outdir. Exit code 1 on any failure.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, outdir = '.'] = process.argv;
const fails = [];
const ok = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) fails.push(msg); };

const PAIRS = ['.hero-crest', '.hero-title', '.hero-sub', '.cta-btn',
               '.hstat:nth-child(1)', '.hstat:nth-child(2)', '.hstat:nth-child(3)', '.hstat:nth-child(4)'];

// The bottom dock's boxes (static/dock.js), as the visitor sees them: null
// when not drawn.
function dockBoxes() {
  const b = s => { const e = document.querySelector(s); if (!e) return null;
    const r = e.getBoundingClientRect(); return r.width && r.height ? { l: r.left, t: r.top, r: r.right, b: r.bottom } : null; };
  return { notice: b('.m-notice'), ok: b('.m-notice-ok'), dock: b('.m-dock:not([hidden])'),
           btn: b('.m-dock:not([hidden]) .m-dock-btn'), cta: b('#hero .cta-btn') };
}
const meet = (a, b) => !!(a && b && a.l < b.r - 0.5 && b.l < a.r - 0.5 && a.t < b.b - 0.5 && b.t < a.b - 0.5);
const inside = (a, w, h) => !!(a && a.l >= -0.5 && a.t >= -0.5 && a.r <= w + 0.5 && a.b <= h + 0.5);
const tap = a => !!(a && a.r - a.l >= 44 && a.b - a.t >= 44);
const box = a => (a ? `${Math.round(a.l)},${Math.round(a.t)} ${Math.round(a.r - a.l)}x${Math.round(a.b - a.t)}` : 'none');

const browser = await chromium.launch();
for (const [label, w, h] of [['desktop', 1440, 900], ['phone', 390, 844]]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  page.setDefaultTimeout(180000);
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  const t0 = Date.now();
  await page.goto(url);
  await page.waitForSelector('#boot-skel .hero-title', { state: 'attached' });
  await page.screenshot({ path: `${outdir}/${label}-skeleton.png` });
  await page.waitForSelector('#hero .hero-title');
  const heroMs = Date.now() - t0;
  console.log(`INFO  ${label}: hero painted ${heroMs}ms after navigation`);
  // First Contentful Paint: the hero's boot skeleton, painted from index.html
  // (tools/public_files.py) before Streamlit's JavaScript; the hero follows.
  const fcp = await page.evaluate(() => performance.getEntriesByName('first-contentful-paint')[0]?.startTime ?? -1);
  console.log(`INFO  ${label}: first contentful paint ${fcp.toFixed(0)}ms`);
  if (process.env.SEO) ok(fcp >= 0 && fcp < 2000, `${label}: first branded paint under 2s (FCP ${fcp.toFixed(0)}ms)`);
  await page.waitForTimeout(3000);              // tile fly-in settles
  const gaps = await page.evaluate(sels => sels.map(s => {
    const a = document.querySelector(`#boot-skel ${s}`)?.getBoundingClientRect();
    const b = document.querySelector(`#hero ${s}`)?.getBoundingClientRect();
    if (!a || !b) return [s, null];
    return [s, Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y),
                        Math.abs(a.width - b.width), Math.abs(a.height - b.height))];
  }), PAIRS);
  const missing = gaps.filter(g => g[1] === null).map(g => g[0]);
  const worst = Math.max(...gaps.filter(g => g[1] !== null).map(g => g[1]));
  ok(!missing.length && worst <= 2,
     `${label}: skeleton outlines the hero (worst box gap ${worst.toFixed(1)}px${missing.length ? `, missing ${missing}` : ''})`);
  const header = await page.evaluate(() => {
    const el = document.querySelector('[data-testid="stHeader"]');
    return el ? getComputedStyle(el).display : 'absent';
  });
  ok(header === 'none' || header === 'absent', `${label}: Streamlit top bar hidden (${header})`);
  const crestTop = await page.evaluate(() => document.querySelector('#hero .hero-crest').getBoundingClientRect().top);
  ok(crestTop >= 0, `${label}: crest fully on screen (top ${crestTop.toFixed(0)}px)`);
  const cta = await page.evaluate(() => document.querySelector('#hero .cta-btn').getBoundingClientRect().bottom);
  ok(cta <= h, `${label}: main CTA above the fold (bottom ${cta.toFixed(0)}px of ${h})`);
  // The dock (static/dock.js) is new on 2026-11-04. A branch push checks the
  // DEPLOYED site with the old build still live, so step 5 passes DOCK=optional
  // there; this commit's own app (step 8) and main always require it.
  const dockHere = await page.evaluate(() => !!window.__melionaNotice);
  if (!dockHere && process.env.DOCK === 'optional') {
    console.log(`INFO  ${label}: no cookie notice or shortcut bar on this deployment yet (DOCK=optional)`);
  } else {
    // The cookie notice (static/dock.js) shows on a first visit. It informs and
    // asks nothing, so it must never stand between a visitor and the main button:
    // measured here, on the first screen, before it is answered.
    const first = await page.evaluate(dockBoxes);
    ok(!!first.notice, `${label}: cookie notice shown on a first visit`);
    ok(inside(first.notice, w, h), `${label}: cookie notice inside the screen`);
    ok(!meet(first.notice, first.cta), `${label}: cookie notice clear of the main CTA (CTA ${box(first.cta)}, notice ${box(first.notice)})`);
    ok(tap(first.ok), `${label}: the notice's OK is a 44px target (${box(first.ok)})`);
    ok(!first.dock, `${label}: no shortcut bar while the hero's own button is on screen`);
    const noAlt = await page.evaluate(() => [...document.querySelectorAll('img')].filter(i => !i.hasAttribute('alt')).length);
    ok(noAlt === 0, `${label}: every <img> has alt text (${noAlt} missing)`);
    await page.screenshot({ path: `${outdir}/${label}-hero.png` });

    await page.waitForSelector('.verdict-number');
    const verdict = (await page.textContent('.verdict-number')).trim();
    ok(/^\d+(\.\d+)?%$/.test(verdict), `${label}: headline CVaR is a finite percentage (${verdict})`);
    const through = await page.evaluate(() => (document.body.innerText.match(/through (\d{4}-\d{2}-\d{2})/) || [])[1]);
    const ageDays = through ? (Date.now() - Date.parse(through)) / 864e5 : Infinity;
    ok(ageDays <= 6, `${label}: data is current (through ${through}, ${ageDays.toFixed(1)} days)`);
    const text = await page.evaluate(() => document.body.innerText);
    const bad = text.match(/\b(nan|[-+]?inf)\b/gi);
    ok(!bad, `${label}: no NaN/inf in visible text${bad ? ` (${bad.slice(0, 3)})` : ''}`);
    // The phone shortcut (static/dock.js): up once the hero's button has left the
    // top of the screen, never on a wider screen, under the notice and never on
    // it; pressed, the page's own glide flies it to the engine, and it goes as
    // soon as the engine is on screen - it never sits over the verdict.
    await page.evaluate(() => {
      const sc = document.querySelector('[data-testid="stMain"]'), c = document.querySelector('#hero .cta-btn');
      sc.style.scrollBehavior = 'auto'; sc.scrollTop += c.getBoundingClientRect().bottom + 60; sc.style.scrollBehavior = '';
    });
    await page.waitForTimeout(900);
    const mid = await page.evaluate(dockBoxes);
    if (label === 'phone') {
      ok(!!mid.dock, 'phone: shortcut bar up once the hero button has scrolled off');
      ok(inside(mid.dock, w, h) && mid.dock.b >= h - 0.5, `phone: shortcut bar on the bottom edge (${box(mid.dock)})`);
      ok(tap(mid.btn), `phone: shortcut button is a 44px target (${box(mid.btn)})`);
      ok(!!mid.notice && !meet(mid.notice, mid.dock), `phone: cookie notice sits above the shortcut, never on it (notice ${box(mid.notice)}, bar ${box(mid.dock)})`);
      await page.screenshot({ path: `${outdir}/${label}-dock.png` });
      await page.click('.m-dock-btn');
      await page.waitForFunction(() => !document.__melionaStop && !window.__melionaStop, null, { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(2500);
      const after = await page.evaluate(() => ({ top: document.getElementById('engine').getBoundingClientRect().top,
        hash: location.hash, up: !!window.__melionaNotice?.dock() }));
      ok(after.top >= 0 && after.top <= 80 && !after.hash, `phone: the shortcut glides to the engine (heading ${after.top.toFixed(0)}px from the top${after.hash ? `, hash ${after.hash}` : ''})`);
      ok(!after.up, 'phone: shortcut gone once the engine is on screen');
      await page.click('.m-notice-ok');
      const said = await page.evaluate(() => ({ shown: !!document.querySelector('.m-notice'),
        kept: (() => { try { return localStorage.getItem('meliona-cookie-notice'); } catch (e) { return 'blocked'; } })() }));
      ok(!said.shown && said.kept === 'ok', `phone: OK closes the notice and remembers it (${said.kept})`);
    } else {
      ok(!mid.dock, `${label}: no shortcut bar on a wide screen`);
      await page.evaluate(() => window.__melionaNotice?.dismiss());
    }
  }
  // The footer renders at the end of the script run, after the headline.
  const contact = await page.waitForFunction(
    () => /meleona\.support@gmail\.com/.test(document.body.textContent), null, { timeout: 120000 })
    .then(() => true, () => false);
  ok(contact, `${label}: real support contact shown on the page`);
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  ok(sideways <= 1, `${label}: no sideways scroll (${sideways}px)`);
  if (label === 'desktop') {
    // Risk Topology (first tab): its agents - one unit per asset, simulated
    // tracer paths, end-state dots - must be drawn from a finite payload.
    // The "Universe (N): A, B, ..." caption sits in a closed tab, so read the element itself.
    const cap = await page.evaluate(() => [...document.querySelectorAll('[data-testid="stCaptionContainer"]')]
      .map(e => e.textContent.trim()).find(t => /^Universe \(\d+\): /.test(t)) || '');
    const universe = cap.replace(/^Universe \(\d+\): /, '').split(', ').map(x => x.trim()).filter(Boolean).sort();
    let map = null;
    for (let i = 0; i < 60 && !map; i++) {
      for (const f of page.frames()) {
        if (await f.$('#cvTrail').catch(() => null)) { map = f; break; }
      }
      if (!map) await page.waitForTimeout(1000);
    }
    ok(!!map, 'desktop: Risk Topology map frame rendered');
    if (map) {
      await page.waitForTimeout(6000);          // one tracer pass
      const m = await map.evaluate(() => {
        let bad = 0;
        const walk = v => { if (typeof v === 'number') { if (!isFinite(v)) bad++; }
                            else if (v && typeof v === 'object') Object.values(v).forEach(walk); };
        walk(DEMO);
        const ink = id => { const c = document.getElementById(id);
          const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n / (d.length / 4); };
        return { units: DEMO.assets.filter(a => a.f === 'Universe').map(a => a.t).sort(), bad,
                 base: ink('cvBase'), trail: ink('cvTrail') };
      });
      ok(universe.length > 1 && m.units.join() === universe.join(),
         `desktop: map units are the visitor's own tickers, no demo data (${m.units.join(' ')} vs ${universe.join(' ')})`);
      ok(m.bad === 0, `desktop: every number in the map payload is finite (${m.bad} non-finite)`);
      ok(m.base > 0.5 && m.trail > 0, `desktop: terrain and tracer paths drawn (terrain ${(m.base * 100).toFixed(0)}%, tracers ${(m.trail * 100).toFixed(1)}% of canvas)`);
    }
  }
  if (label === 'phone') {
    // The map on an iPhone (2026-10-17): it shipped in a 278px frame inside the
    // page's 56px gutters, scrolled sideways and put its x-axis below the
    // screen. Strict on this commit's app (SEO) and on main (EXPECT_BUILD);
    // a branch run against the not-yet-deployed site only reports.
    const strict = !!(process.env.SEO || process.env.EXPECT_BUILD);
    const check = (cond, msg) => (strict ? ok(cond, msg) : console.log(`${cond ? 'INFO ' : 'INFO  (not yet deployed)'} ${msg}`));
    let map = null;
    for (let i = 0; i < 60 && !map; i++) {
      for (const f of page.frames()) {
        if (await f.$('#cvTrail').catch(() => null)) { map = f; break; }
      }
      if (!map) await page.waitForTimeout(1000);
    }
    check(!!map, 'phone: Risk Topology map frame rendered');
    if (map) {
      const box = await (await map.frameElement()).boundingBox();
      check(box && box.width >= w - 32, `phone: map frame uses the screen (${Math.round(box ? box.width : 0)}px of ${w})`);
      await map.waitForFunction(() => typeof S !== 'undefined' && S.cloud, null, { timeout: 60000 }).catch(() => null);
      const m = await map.evaluate(() => {
        const cv = document.getElementById('cvBase').getBoundingClientRect();
        return { sideways: document.documentElement.scrollWidth - document.documentElement.clientWidth,
                 axis: Math.round(cv.top + (H - PAD.b) * cv.height / H) };
      });
      check(m.sideways <= 0, `phone: nothing in the map scrolls sideways (${m.sideways}px)`);
      check(m.axis + 20 <= 700, `phone: the map's x-axis is on the first screen (${m.axis}px into the frame)`);
    }
  }
  await page.locator('.verdict-number').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${outdir}/${label}-verdict.png` });
  ok(!errors.length, `${label}: no script errors${errors.length ? ` (${errors.slice(0, 2)})` : ''}`);
  await page.close();
}
// Which commit is this host serving? tools/public_files.py stamps it into the
// raw HTML; the Live audit sets EXPECT_BUILD on main, where the deployed site
// must be serving the commit under test (a host that quietly kept an old build
// cost two sessions in September).
const get = async p => { const r = await fetch(url.replace(/\/$/, '') + p); return [r.status, r.headers.get('content-type') || '', await r.text()]; };
const [hs, , head] = await get('/');
const build = (head.match(/<meta name="meliona-build" content="([^"]*)"/) || [])[1] || '(no build tag)';
console.log(`INFO  build: ${build}`);
if (process.env.EXPECT_BUILD)
  ok(process.env.EXPECT_BUILD.startsWith(build) && build.length >= 7,
     `deployed build is the commit under test (${build} vs ${process.env.EXPECT_BUILD.slice(0, 12)})`);
if (process.env.SEO) {
  // Launch basics a crawler or link preview reads - raw HTML, no JavaScript.
  for (const tag of ['<title>Meliona', 'name="description"', 'property="og:image"', 'property="og:title"', 'rel="canonical"', 'twitter:card'])
    ok(hs === 200 && head.includes(tag), `seo: raw HTML carries ${tag}`);
  const [rs, , robots] = await get('/robots.txt');
  ok(rs === 200 && /Sitemap: https?:\/\//.test(robots), 'seo: /robots.txt names the sitemap');
  const [ss, , site] = await get('/sitemap.xml');
  ok(ss === 200 && site.includes('<urlset'), 'seo: /sitemap.xml served');
  const [os_, ot] = await get('/og-image.png');
  ok(os_ === 200 && ot.startsWith('image/png'), `seo: Open Graph image served (${os_} ${ot})`);
  const [fs, ft] = await get('/favicon.png');
  ok(fs === 200 && ft.startsWith('image/png'), 'seo: favicon served');
  const p404 = await browser.newPage();
  await p404.goto(url.replace(/\/$/, '') + '/no/such/page'); await p404.waitForTimeout(2500);
  ok(/page not found/i.test(await p404.evaluate(() => document.body.innerText)), 'seo: a bad link shows the branded 404, not a blank page');
  await p404.screenshot({ path: `${outdir}/404.png` });
}
await browser.close();
console.log(`\n${fails.length ? fails.length + ' FAILED' : 'all site checks pass'}`);
process.exit(fails.length ? 1 : 0);
