// What a visitor's browser sees, checked - on any deployment of the app.
//
//   node tools/site_check.mjs <url> [outdir]
//
// At a desktop and a phone viewport: the boot skeleton outlines the real hero
// (every crest/title/sub/button/tile box within 2px of the element it stands
// in for), Streamlit's top bar stays hidden, the headline is a finite
// percentage, the data is current, nothing on the page reads NaN/inf, the
// page does not scroll sideways, and no script error fires. Screenshots land
// in outdir. Exit code 1 on any failure.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, outdir = '.'] = process.argv;
const fails = [];
const ok = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) fails.push(msg); };

const PAIRS = ['.hero-crest', '.hero-title', '.hero-sub', '.cta-btn',
               '.hstat:nth-child(1)', '.hstat:nth-child(2)', '.hstat:nth-child(3)', '.hstat:nth-child(4)'];

const browser = await chromium.launch();
for (const [label, w, h] of [['desktop', 1440, 900], ['phone', 390, 844]]) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  page.setDefaultTimeout(180000);
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(url);
  await page.waitForSelector('#boot-skel .hero-title', { state: 'attached' });
  await page.screenshot({ path: `${outdir}/${label}-skeleton.png` });
  await page.waitForSelector('#hero .hero-title');
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
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  ok(sideways <= 1, `${label}: no sideways scroll (${sideways}px)`);
  await page.locator('.verdict-number').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${outdir}/${label}-verdict.png` });
  ok(!errors.length, `${label}: no script errors${errors.length ? ` (${errors.slice(0, 2)})` : ''}`);
  await page.close();
}
await browser.close();
console.log(`\n${fails.length ? fails.length + ' FAILED' : 'all site checks pass'}`);
process.exit(fails.length ? 1 : 0);
