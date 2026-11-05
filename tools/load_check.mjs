// Simultaneous visitors, and a restore from nothing (John, 2026-11-04: "test
// simultaneous users and test backup restores").
//
//   node tools/load_check.mjs <url> <visitors> <seconds> [label]
//
// Opens <visitors> separate browser contexts at the same moment - each a new
// visitor with its own session - and waits for every one to show the headline
// verdict (a number like 22.6%). Fails if any visitor sees a server error, a
// failed request to the app, an error box, a refusal from the rate limiter, or
// no headline within <seconds>. Prints each visitor's time to the headline and
// the slowest, which is the number to quote.
//
// The Live audit runs it twice (step 8e): 10 visitors at once on this commit's
// warm app, and 1 visitor on a fresh copy started with an empty data/ and logs/
// - the state a redeploy or a restore from git leaves - which measures the
// time to recover from nothing (the restore drill). Nothing here is a backup
// of user data: the app holds none (LAUNCH_CHECKLIST #10).

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, n = '10', budget = '180', label = 'load'] = process.argv;
const visitors = Number(n), limitMs = Number(budget) * 1000;
if (!url || !(visitors > 0)) {
  console.error('usage: node tools/load_check.mjs <url> <visitors> <seconds> [label]');
  process.exit(2);
}

const browser = await chromium.launch();
const t0 = Date.now();

async function visit(i) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const problems = [];
  let done = false;
  page.on('response', r => {
    if (r.url().startsWith(url) && (r.status() >= 500 || r.status() === 429)) {
      problems.push(`HTTP ${r.status()} ${r.url().slice(url.length) || '/'}`);
    }
  });
  page.on('requestfailed', r => {
    if (r.url().startsWith(url)) problems.push(`failed ${r.url().slice(url.length) || '/'}`);
  });
  page.on('websocket', ws => ws.on('close', () => {
    if (!done) problems.push('websocket closed early');
  }));
  const start = Date.now();
  let seconds = null, verdict = '';
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: limitMs });
    await page.waitForFunction(
      () => /^\d+(\.\d+)?%$/.test(document.querySelector('.verdict-number')?.textContent.trim() || ''),
      null, { timeout: Math.max(1000, limitMs - (Date.now() - t0)) });
    seconds = (Date.now() - start) / 1000;
    verdict = (await page.textContent('.verdict-number')).trim();
    const boxes = await page.$$eval('[data-testid="stException"], [data-testid="stAlertContentError"]',
      els => els.map(e => e.textContent.trim().slice(0, 120)));
    problems.push(...boxes.map(b => `error box: ${b}`));
  } catch (e) {
    problems.push(`no headline within ${budget}s (${String(e.message).split('\n')[0]})`);
  }
  done = true;
  await ctx.close();
  return { i, seconds, verdict, problems };
}

const results = await Promise.all(Array.from({ length: visitors }, (_, i) => visit(i + 1)));
await browser.close();

let failed = 0;
for (const r of results) {
  const ok = r.seconds !== null && r.problems.length === 0;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  visitor ${r.i}: ${r.seconds === null ? 'no headline' :
    `headline ${r.verdict} after ${r.seconds.toFixed(1)}s`}${r.problems.length ? ' - ' + r.problems.join(' | ') : ''}`);
}
const times = results.map(r => r.seconds).filter(s => s !== null).sort((a, b) => a - b);
console.log(`\n${label}: ${visitors} visitor(s) at once, ${visitors - failed} served; ` +
  (times.length ? `median ${times[Math.floor(times.length / 2)].toFixed(1)}s, slowest ${times.at(-1).toFixed(1)}s` : 'none served') +
  ` (budget ${budget}s)`);
process.exit(failed ? 1 : 0);
