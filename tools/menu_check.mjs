// A dropdown opens at once and stays open until the reader closes it.
//
//   node tools/menu_check.mjs <url>
//
// John, 2026-11-07: "the menu screen would lag and automatically close by
// itself". react-aria shuts a menu the moment the page under it scrolls, and
// the wheel's tow (static/reveal.js) keeps the page gliding ~1.5s after the
// last notch: a menu clicked mid-glide, or under a trackpad's resting
// fingers, opened and snapped shut. Even a press that stopped the glide lost
// the menu: the browser reports the last glide frame's scroll ~35ms late,
// after the menu has opened. reveal.js now stops the tow on any press, holds
// the page still while a menu is open, and drops a scroll event that reports
// no movement since the press.
//
// For every dropdown and multi-select on the page (1440x900, real mouse):
//   rest     opens within 400ms and is still open 8s later, untouched
//   glide    clicked 80, 250 and 500ms after two wheel notches: still open 2.5s later
//   trackpad six 1px wheel ticks while open: still open, the page has not moved
//   wheel    a full notch elsewhere while open: still open, the page has not moved
//   close    Escape closes it; so does a click on the page beside it
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
const { chromium } = pw.chromium ? pw : pw.default;
const [,, url] = process.argv;
if (!url) { console.error('usage: node tools/menu_check.mjs <url>'); process.exit(2); }
const MENUS = '[data-testid="stSelectbox"]:visible, [data-testid="stMultiSelect"]:visible';
const POP = '[data-testid="stSelectboxVirtualDropdown"]:not([data-exiting]), [data-testid="stMultiSelectDropdown"]:not([data-exiting])';
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
const p = await (await b.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
await p.goto(url);
await p.waitForSelector('.meliona-footer', { timeout: 300000 });
await p.waitForTimeout(1500);
await p.evaluate(() => window.__melionaReveal && window.__melionaReveal.revealAll());
// the engine's menus live in drawers (Portfolio, Weights, Stress test): open each one that holds a menu
await p.evaluate(() => document.querySelectorAll('[data-testid="stExpander"] details:not([open]) > summary').forEach(sm => {
  if (sm.parentElement.querySelector('[data-testid="stSelectbox"], [data-testid="stMultiSelect"]')) sm.click();
}));
await p.waitForTimeout(1500);
let fails = 0;
const ok = (pass, what) => { console.log(`${pass ? 'PASS' : 'FAIL'}  ${what}`); if (!pass) fails++; };
const isOpen = async () => !!(await p.$(POP));
const y = () => p.evaluate(() => document.querySelector('[data-testid="stMain"]').scrollTop);
const closeAll = async () => { for (let i = 0; i < 3 && await isOpen(); i++) { await p.keyboard.press('Escape'); await p.waitForTimeout(300); } };
const menus = p.locator(MENUS);
const n = await menus.count();
ok(n >= 5, `${n} dropdown(s) on the page with its drawers open (5 expected: basket, names, weights, stress test, contact topic)`);
for (let i = 0; i < n; i++) {
  const m = menus.nth(i);
  const name = (await m.evaluate(e => (e.querySelector('[data-testid="stWidgetLabel"]') || e).innerText.split('\n')[0].trim()) || `menu ${i + 1}`).slice(0, 40);
  // the field itself - never a chosen name's remove button, which would change the basket
  const centre = () => m.evaluate(e => { const t = e.querySelector('[role="combobox"]') || e.querySelector('input') || e; const r = t.getBoundingClientRect(); return [r.left + Math.min(r.width / 2, 120), r.top + r.height / 2]; });
  const home = async () => { await m.evaluate(e => e.scrollIntoView({ block: 'center' })); await p.waitForTimeout(1800); await p.mouse.move(24, 450); };
  // rest
  await home();
  let c = await centre(); const t0 = Date.now(); await p.mouse.click(c[0], c[1]);
  await p.waitForSelector(POP, { timeout: 3000 }).catch(() => {});
  const took = Date.now() - t0;
  await p.waitForTimeout(8000);
  ok(took <= 400 && await isOpen(), `"${name}" rest: opened in ${took}ms, still open 8s later`);
  // trackpad trickle, then a full notch elsewhere
  await p.mouse.move(24, 450);
  let y0 = await y();
  for (let k = 0; k < 6; k++) { await p.mouse.wheel(0, 1); await p.waitForTimeout(40); }
  await p.waitForTimeout(900);
  ok(await isOpen() && Math.abs(await y() - y0) < 1, `"${name}" trackpad: open, page moved ${Math.round(await y() - y0)}px`);
  y0 = await y(); await p.mouse.wheel(0, 100); await p.waitForTimeout(1200);
  ok(await isOpen() && Math.abs(await y() - y0) < 1, `"${name}" wheel elsewhere: open, page moved ${Math.round(await y() - y0)}px`);
  // close: Escape, then a click beside it
  await p.keyboard.press('Escape'); await p.waitForTimeout(400);
  ok(!(await isOpen()), `"${name}" close: Escape closes it`);
  c = await centre(); await p.mouse.click(c[0], c[1]); await p.waitForTimeout(500);
  await p.mouse.click(24, 450); await p.waitForTimeout(500);
  ok(!(await isOpen()), `"${name}" close: a click beside it closes it`);
  await closeAll();
  // glide
  let held = 0;
  for (const after of [80, 250, 500]) {
    await home();
    await p.mouse.wheel(0, 100); await p.waitForTimeout(60); await p.mouse.wheel(0, 100); await p.waitForTimeout(after);
    c = await centre(); await p.mouse.click(c[0], c[1]);
    await p.waitForTimeout(2500);
    if (await isOpen()) held++;
    await closeAll();
  }
  ok(held === 3, `"${name}" glide: clicked mid-glide, still open in ${held}/3`);
}
await b.close();
console.log(fails ? `${fails} menu check(s) failed` : 'every menu opens at once and stays open');
process.exit(fails ? 1 : 0);
