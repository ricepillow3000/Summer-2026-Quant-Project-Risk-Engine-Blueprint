// A data/AI-analyst recruiter's first five minutes on the site, in a real browser.
//
//   node tools/recruiter_walk.mjs <url> [outdir]
//
// Lands, reads the hero, clicks the main CTA, scrolls the whole page, then does
// what a curious analyst does: switches basket, replays a crisis, flips the
// return model and the side of the trade, and opens every tab. Each step is
// timed and checked - an error box, a NaN, a step slower than 90s, or a
// headline that never answers fails the walk. Screenshots land in outdir.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, outdir = '.'] = process.argv;
const fails = [], steps = [];
const LIMIT = 90000;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.setDefaultTimeout(LIMIT);
const pageErrors = [];
page.on('pageerror', e => pageErrors.push(String(e)));

const verdict = () => page.textContent('.verdict-number').then(t => t.trim()).catch(() => '');
const sentence = () => page.textContent('.verdict-sentence').catch(() => '');
async function problems() {
  return page.evaluate(() => {
    const errs = [...document.querySelectorAll('[data-testid="stException"], [data-testid="stAlertContentError"]')]
      .map(e => e.innerText.slice(0, 120));
    const nan = (document.body.innerText.match(/\b(nan|[-+]?inf)\b/gi) || []).slice(0, 3);
    // A loss phrased as a negative number contradicts itself.
    const contra = (document.body.innerText.match(/loses an average of -\d[^ ]*/g) || []).slice(0, 1);
    return { errs: errs.concat(contra.map(c => 'contradiction: ' + c)), nan };
  });
}
async function step(name, fn, expect) {
  const t0 = Date.now();
  let note = '', ok = true;
  try { note = (await fn()) || ''; } catch (e) { ok = false; note = String(e).split('\n')[0]; }
  // Streamlit reruns after an interaction; wait for it to settle.
  await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"] *'), null,
                             { timeout: LIMIT }).catch(() => {});
  if (ok && expect) { const r = await expect(); ok = r === true; if (r !== true) note += ` | ${r}`; }
  const ms = Date.now() - t0;
  const { errs, nan } = await problems();
  if (errs.length) { ok = false; note += ` | error box: ${errs[0]}`; }
  if (nan.length) { ok = false; note += ` | NaN/inf: ${nan}`; }
  if (ms > LIMIT) { ok = false; note += ' | slower than 90s'; }
  steps.push({ name, ms, ok, note });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} (${(ms / 1000).toFixed(1)}s)${note ? ' - ' + note : ''}`);
  if (!ok) fails.push(name);
  await page.screenshot({ path: `${outdir}/walk-${String(steps.length).padStart(2, '0')}.png` });
}
const tab = name => page.getByRole('tab', { name, exact: true }).first();
// Name the element in a timeout, so a failure explains itself.
const tagged = (what, p) => p.catch(e => { throw new Error(`${what}: ${String(e).split('\n')[0]}`); });
async function pick(label, option) {
  // Long option lists are virtualised (off-screen options are not rendered),
  // so type to filter the way a user would, choose, then Escape so no popover
  // is left open over the next control. A script types faster than the list
  // opens - keystrokes landed before it did and the option never came (live
  // audit run 13) - so fill the filter in one go and retry a missed open.
  const box = page.getByRole('combobox', { name: label }).first();
  for (let attempt = 1; ; attempt++) {
    try {
      await box.scrollIntoViewIfNeeded();
      await tagged(`open "${label}"`, box.click());
      await box.fill(option);
      const opt = page.getByRole('option', { name: option, exact: true }).first();
      await tagged(`choose "${option}"`, opt.waitFor({ state: 'visible', timeout: 10000 }));
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
// Open a drawer only if it is shut - a click on an open one would close it.
async function openDrawer(label) {
  const s = page.locator('summary', { hasText: label }).first();
  await s.scrollIntoViewIfNeeded();
  if (!(await s.evaluate(el => el.parentElement.open))) await tagged(`open drawer "${label}"`, s.click());
}

await step('land and read the headline', async () => {
  await page.goto(url);
  await page.waitForSelector('.verdict-number');
  return `CVaR ${await verdict()}`;
}, async () => /^\d+(\.\d+)?%$/.test(await verdict()) || `headline "${await verdict()}"`);

await step('click "Explore what we do"', async () => {
  await page.click('#hero .cta-btn');
  await page.waitForTimeout(2500);
}, async () => {
  const y = await page.evaluate(() => document.querySelector('#grit-showcase').getBoundingClientRect().top);
  return Math.abs(y) < 400 || `showcase at ${y.toFixed(0)}px, CTA did not scroll to it`;
});

await step('scroll the whole page', async () => {
  const n = await page.evaluate(async () => {
    const m = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
    let i = 0;
    for (let y = 0; y < m.scrollHeight; y += 700, i++) {
      m.scrollTo(0, y); await new Promise(r => setTimeout(r, 250));
    }
    m.scrollTo(0, 0); return i;
  });
  return `${n} screens`;
});

const before = await verdict();
await step('switch basket to Semiconductors', async () => {
  await openDrawer('01 · Universe');
  await pick('Preset basket', 'Semiconductors');
  await page.waitForFunction(b => document.querySelector('.verdict-number')?.textContent.trim() !== b, before);
  return `CVaR ${before} -> ${await verdict()}`;
}, async () => {
  // The visitor's FIRST change used to re-mount every drawer closed - the
  // one they were using snapped shut under the cursor.
  const open = await page.locator('summary', { hasText: '01 · Universe' }).first()
    .evaluate(el => el.parentElement.open);
  return open || 'the Universe drawer closed itself after the first change';
});

await step('replay the COVID-19 crash', async () => {
  await openDrawer('03 · Stress test');
  await pick('Scenario', 'COVID-19 crash (2020)');
  await page.waitForFunction(() => /Replaying/.test(document.querySelector('.verdict-sentence')?.textContent || ''));
  return `CVaR ${await verdict()}`;
}, async () => /COVID/.test(await sentence()) || 'verdict does not name the crisis');

await step('switch to the jump-diffusion engine', async () => {
  const b = await verdict();
  // The drawer must still be open after the crisis rerun - it used to snap shut.
  const open = await page.locator('summary', { hasText: '03 · Stress test' }).first()
    .evaluate(el => el.parentElement.open);
  if (!open) throw new Error('the Stress test drawer closed itself after the previous change');
  await page.getByText('Jump-diffusion (Merton)', { exact: true }).first().click();
  await page.waitForFunction(v => document.querySelector('.verdict-number')?.textContent.trim() !== v, b);
  return `CVaR ${b} -> ${await verdict()}`;
});

await step('flip to the short side', async () => {
  await page.getByText('Bearish - short the book', { exact: true }).first().click();
  await page.waitForFunction(() => /short/i.test(document.querySelector('.verdict-sentence')?.textContent || ''));
  return `CVaR ${await verdict()}`;
});

for (const name of ['Risk Topology', 'Risk Breakdown', 'Correlation Watch', 'Balance', 'Grit Zone',
                    'Crisis Conviction', 'Signal Lab', 'Regime Atlas', 'Liquidity', 'Security Master',
                    'Data Quality', 'Lineage & Audit']) {
  await step(`open tab: ${name}`, async () => {
    await tab(name).scrollIntoViewIfNeeded(); await tab(name).click(); await page.waitForTimeout(1200);
  }, async () => {
    const len = await page.evaluate(n => {
      const t = [...document.querySelectorAll('[role="tab"]')].find(e => e.innerText.trim() === n);
      const panel = t && document.getElementById(t.getAttribute('aria-controls'));
      return panel ? panel.innerText.trim().length + panel.querySelectorAll('iframe, canvas, svg').length * 100 : 0;
    }, name);
    return len > 80 || `tab "${name}" looks empty (${len})`;
  });
}

// Presets are warmed after every deploy; a basket a visitor types is not
// (John, 2026-10-20: "make typed-in baskets load faster too"). Three names the
// page offers but no preset holds (test_the_walks_typed_basket_pool_is_in_no_preset),
// drawn fresh each run so a recent run's pull does not make the next one warm.
const TYPED_POOL = ['ACN', 'ADI', 'AEP', 'AIG', 'ALL', 'AMT', 'ANET', 'APD', 'BBY', 'BMY',
                    'BSX', 'CB', 'CCI', 'CI', 'CL', 'CME', 'CMG', 'CMI', 'CSCO', 'CSX',
                    'CVS', 'DAL', 'DD', 'DG', 'DHR', 'DUK', 'EBAY', 'ECL', 'ELV', 'EMR',
                    'ETN', 'EXC', 'F', 'FCX', 'GD', 'GILD'];
const typed = [...TYPED_POOL].sort(() => Math.random() - 0.5).slice(0, 3).sort();
const chips = () => page.evaluate(() =>
  [...document.querySelectorAll('[data-baseweb="tag"]')].map(e => e.innerText.trim()));
async function addTicker(sym) {
  // The ticker box clears itself after a pick, so success is the new chip.
  const box = page.getByRole('combobox', { name: 'Tickers to analyze' }).first();
  for (let attempt = 1; ; attempt++) {
    try {
      await box.scrollIntoViewIfNeeded();
      await tagged('open the ticker box', box.click());
      await box.fill(sym);
      const opt = page.getByRole('option', { name: sym, exact: true }).first();
      await tagged(`choose "${sym}"`, opt.waitFor({ state: 'visible', timeout: 10000 }));
      await opt.click({ timeout: 10000 }).catch(() => page.keyboard.press('Enter'));
      await page.keyboard.press('Escape');
      await tagged(`chip "${sym}"`, page.waitForFunction(s =>
        [...document.querySelectorAll('[data-baseweb="tag"]')].some(e => e.innerText.trim() === s),
        sym, { timeout: 10000 }));
      return;
    } catch (e) {
      await page.keyboard.press('Escape');
      if (attempt === 3) throw e;
    }
  }
}

await step(`type in your own basket (+${typed.join(', ')})`, async () => {
  const b = await verdict();
  await openDrawer('01 · Universe');
  for (const sym of typed) await addTicker(sym);
  return `CVaR ${b} -> ${await verdict()}`;
}, async () => {
  const have = await chips();
  const lost = typed.filter(s => !have.includes(s));
  return !lost.length || `the ticker box lost ${lost.join(', ')}`;
});

await browser.close();
if (pageErrors.length) { fails.push('script errors'); console.log(`FAIL  script errors: ${pageErrors.slice(0, 2)}`); }
const total = steps.reduce((a, s) => a + s.ms, 0) / 1000;
console.log(`\n${steps.length} steps in ${total.toFixed(0)}s - ${fails.length ? fails.length + ' FAILED' : 'recruiter walk clean'}`);
process.exit(fails.length ? 1 : 0);
