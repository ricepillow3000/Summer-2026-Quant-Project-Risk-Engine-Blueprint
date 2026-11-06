// node tests/glide_wait_probe.mjs <page.html>
// A press waiting for its section, on a stand-in page that carries main.py's
// real glide frame (test_a_waiting_press_survives_the_page_moving_itself
// writes it). Live audit 8c early: the page moved itself 5px while a press
// waited, the wait ended, and "Skip to the analysis" stopped ~2400px short.
// Prints one JSON object of checks; exit 1 if any is false, 3 without playwright.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

let pw;
try { pw = await import('playwright'); } catch {
  try {
    const at = createRequire(import.meta.url).resolve('playwright', { paths: [process.env.NODE_PATH || ''] });
    pw = await import(pathToFileURL(at).href);
  } catch { console.error('playwright not installed'); process.exit(3); }
}
const { chromium } = pw.chromium ? pw : pw.default;
const html = readFileSync(process.argv[2], 'utf8');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });

async function press() {               // a fresh page, the button 300px down the view, pressed
  await page.setContent(html);
  await page.waitForFunction(() => !!document.__melionaGlide);
  await page.waitForTimeout(600);        // past the frame's first 300ms (forgivenAt starts at 0)
  await page.evaluate(() => {
    const sc = document.getElementById('sc'), a = document.getElementById('go');
    sc.scrollTop += a.getBoundingClientRect().top - 300;
    a.click();
  });
  await page.waitForTimeout(150);
}
const waiting = () => page.evaluate(() => !!document.__melionaStop);
const write = () => page.evaluate(() => {   // the section the press waits for arrives
  const s = document.createElement('div'); s.id = 'analysis'; s.style.height = '50px';
  document.getElementById('grow').prepend(s);
});
const top = () => page.evaluate(() => Math.round(document.getElementById('analysis').getBoundingClientRect().top));
const y = () => page.evaluate(() => document.getElementById('sc').scrollTop);

const out = {};
await press();
await page.evaluate(() => { document.getElementById('sc').scrollTop += 5; });   // the page's own move
await page.waitForTimeout(150);
out.page_move_keeps_the_wait = await waiting();
await write(); await page.waitForTimeout(3000);
out.the_press_lands = Math.abs(await top()) <= 60;

await press();                         // a reader dragging the scrollbar while it waits
await page.evaluate(() => {
  const sc = document.getElementById('sc'), o = { bubbles: true, pointerType: 'mouse' };
  sc.dispatchEvent(new PointerEvent('pointerdown', o));
  sc.scrollTop += 200;
});
await page.waitForTimeout(150);
await page.evaluate(() => document.getElementById('sc').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' })));
out.a_drag_ends_the_wait = !(await waiting());
const before = await y();
await write(); await page.waitForTimeout(1500);
out.the_reader_is_not_pulled_back = Math.abs((await y()) - before) <= 2;

await browser.close();
console.log(JSON.stringify(out));
process.exit(Object.values(out).every(Boolean) ? 0 : 1);
