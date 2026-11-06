// A first visit with a keyboard and a mouse wheel never paints a gold block.
//
//   node tools/key_wheel_check.mjs <url> [shots-dir]
//
// John, 2026-11-06 : "when pressing S or any key a giant block from the
// left side would appear slowly, and when I would use my scroll button, the
// yellow block would get bigger and bigger". Cause: Streamlit draws its own
// keyboard-focus ring for the page's scroller as [data-testid="stMain"]
// :focus-visible::before - sticky, the full screen tall. Our reading-progress
// line was ALSO stMain::before, so once a key made the focus visible the two
// merged: a full-screen bronze block, scaled from the left by the scroll.
//
// Done the way a visitor does it: a fresh browser (no storage), a real mouse
// click on the page's empty margin (which focuses the scroller), real key
// presses (S, Tab, Space, a letter), then the real mouse wheel, at 1440x900
// and 1280x720. After each step the screen is photographed and its bronze
// pixels counted: bronze (#9A7B4F) is the rules' colour, hairlines only, so
// more than 3% of the screen in bronze is a block, never a rule.
// Exit code 1 on any failure.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
let pw;
try {
  pw = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
} catch {
  try {   // a global install (npm i -g playwright), found through NODE_PATH or npm's global root
    const at = createRequire(import.meta.url).resolve('playwright', { paths: [process.env.NODE_PATH || ''] });
    pw = await import(pathToFileURL(at).href);
  } catch {
    console.error('playwright not installed');
    process.exit(3);
  }
}
const { chromium } = pw.chromium ? pw : pw.default;
const [,, url, shots = ''] = process.argv;
if (!url) { console.error('usage: node tools/key_wheel_check.mjs <url> [shots-dir]'); process.exit(2); }
if (shots) (await import('node:fs')).mkdirSync(shots, { recursive: true });
const LIMIT = 0.03;
const SIZES = [[1440, 900], [1280, 720]];

const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
// Bronze share of a screenshot, counted in a blank page's canvas.
const counter = await (await b.newContext()).newPage();
async function bronze(png) {
  return counter.evaluate(async (src) => {
    const img = new Image(); img.src = src; await img.decode();
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (Math.abs(d[i] - 154) < 14 && Math.abs(d[i + 1] - 123) < 14 && Math.abs(d[i + 2] - 79) < 14) n++;
    }
    return n / (d.length / 4);
  }, 'data:image/png;base64,' + png.toString('base64'));
}

let fails = 0;
for (const [w, h] of SIZES) {
  const ctx = await b.newContext({ viewport: { width: w, height: h } });
  const p = await ctx.newPage();
  await p.goto(url, { waitUntil: 'load' });
  await p.waitForSelector('#hero', { timeout: 60000 });
  await p.waitForTimeout(4000);
  const step = async (name) => {
    await p.waitForTimeout(1200);
    const png = await p.screenshot();
    if (shots) (await import('node:fs')).writeFileSync(`${shots}/${w}x${h}-${name}.png`, png);
    const share = await bronze(png);
    const ok = share <= LIMIT;
    if (!ok) fails++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${w}x${h} ${name}: ${(share * 100).toFixed(1)}% of the screen is bronze`);
  };
  await step('0-arrived');
  // the page's empty right margin: a click there focuses the scroller
  await p.mouse.click(w - 12, Math.round(h / 2));
  await step('1-clicked');
  await p.keyboard.press('s');
  await step('2-key-s');
  for (const k of ['Tab', 'Space', 'a']) await p.keyboard.press(k);
  await step('2-keys-tab-space-a');
  await p.mouse.move(Math.round(w / 2), Math.round(h / 2));
  for (let i = 0; i < 20; i++) { await p.mouse.wheel(0, 120); await p.waitForTimeout(60); }
  await step('3-wheel');
  await p.mouse.click(w - 12, Math.round(h / 2));
  await p.keyboard.press('s');
  for (let i = 0; i < 10; i++) { await p.mouse.wheel(0, 120); await p.waitForTimeout(60); }
  await step('4-click-key-wheel');
  await ctx.close();
}
await b.close();
console.log(fails ? `${fails} FAIL` : 'all PASS');
process.exit(fails ? 1 : 0);
