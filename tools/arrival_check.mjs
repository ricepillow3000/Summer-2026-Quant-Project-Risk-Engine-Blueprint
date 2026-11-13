// A reader scrolling down SEES each block arrive.
//
//   node tools/arrival_check.mjs <url>
//
// John, 2026-11-13: "when scrolling down, I do not see the animations,
// headers, and body of line text coming in from the left side or the right
// side ... I don't see any transitions." Measured on the page: blocks played
// their arrival during the first load while a short page still held them on
// the first screen (20 of 41, before the hero pushed them down), and the rest
// started 15% below the screen's edge, so 92% had landed (1.3px left) before
// the reader saw them. Every check below failed on that page.
//
// At 1440x900 and 390x844, scrolling with the real mouse wheel:
//   load     no block plays its arrival off the screen, or on a first
//            screen the rest of the page then pushes it off
//   seen     at least 80% of the blocks are still visibly arriving (opacity
//            under .9 or more than 4px from home) when they first come on screen
//   sides    desktop: blocks arrive from the left AND from the right as well
//            as rising (a phone, its columns stacked, only rises and slides
//            from the left)
//   rest     a reader who stops is never left looking at a gap: 1.2s after
//            the wheel stops, no block whose place is on screen still waits
//            (a rising block waits 28px low, so its place, not its box)
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
if (!url) { console.error('usage: node tools/arrival_check.mjs <url>'); process.exit(2); }
const b = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
let fails = 0;
const ok = (name, cond, info = '') => { if (!cond) fails++; console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${info ? '  ' + info : ''}`); };

for (const [w, h] of [[1440, 900], [390, 844]]) {
  const tag = `${w}x${h}`;
  const p = await (await b.newContext({ viewport: { width: w, height: h } })).newPage();
  const errors = [];
  p.on('pageerror', e => errors.push(e.message));
  // every arrival as it starts: where its block sits at that moment
  await p.addInitScript(() => {
    window.__plays = [];
    new MutationObserver(ms => {
      for (const m of ms) {
        const el = m.target;
        if (!el.classList || !el.classList.contains('is-visible') || el.__played) continue;
        el.__played = 1;
        const r = el.getBoundingClientRect();
        window.__plays.push({ el, top: r.top, bottom: r.bottom, h: r.height, vh: innerHeight,
          txt: (el.textContent || '').trim().slice(0, 40) });
      }
    }).observe(document, { attributes: true, attributeFilter: ['class'], subtree: true });
  });
  await p.goto(url, { waitUntil: 'domcontentloaded' });
  await p.waitForSelector('.meliona-footer', { timeout: 240000 });
  await p.waitForTimeout(2500);
  const pre = await p.evaluate(() => ({
    on: document.documentElement.classList.contains('reveal-on'),
    reduced: matchMedia('(prefers-reduced-motion: reduce)').matches,
  }));
  ok(`${tag} motion is on (reveal-on, no reduced motion)`, pre.on && !pre.reduced, JSON.stringify(pre));
  // where each block that played during the load sits once the page has
  // settled: one below the first screen played where nobody could see it
  const atLoad = await p.evaluate(() => window.__plays.map(x => {
    const r = x.el.getBoundingClientRect();
    return { txt: x.txt, h: r.height, top: r.top, bottom: r.bottom, vh: innerHeight, then: x.top, thenOff: x.h > 0 && (x.top >= x.vh - 1 || x.bottom <= 0) };
  }));
  const offLoad = atLoad.filter(x => x.thenOff || (x.h > 0 && (x.top >= x.vh - 1 || x.bottom <= 0)));
  ok(`${tag} load: no block plays its arrival off the screen`, !offLoad.length,
    `${offLoad.length} of ${atLoad.length}${offLoad.length ? ': ' + offLoad.slice(0, 4).map(x => `"${x.txt}" at ${x.top | 0}px`).join('; ') : ''}`);

  // watch each waiting block; note how it looks the frame it first comes on screen
  await p.evaluate(() => {
    window.__first = {};
    let n = 0;
    const loop = () => {
      for (const el of document.querySelectorAll('.reveal-item')) {
        if (!el.__pid) el.__pid = ++n;
        if (window.__first[el.__pid]) continue;
        const r = el.getBoundingClientRect();
        if (!r.width || !r.height || r.top >= innerHeight - 1 || r.bottom <= 0) continue;
        const cs = getComputedStyle(el);
        const m = new DOMMatrixReadOnly(cs.transform === 'none' ? undefined : cs.transform);
        window.__first[el.__pid] = { op: +cs.opacity, dx: m.m41, dy: m.m42, txt: (el.textContent || '').trim().slice(0, 30) };
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  });
  // also each block's direction, read from its waiting state as it starts
  await p.evaluate(() => {
    window.__dirs = { l: 0, r: 0, up: 0 };
    new MutationObserver(ms => {
      for (const m of ms) {
        const el = m.target;
        if (!el.classList || !el.classList.contains('reveal-item') || el.__dir) continue;
        if (el.classList.contains('is-visible')) continue;
        el.__dir = 1;
        const cs = getComputedStyle(el);
        const t = new DOMMatrixReadOnly(cs.transform === 'none' ? undefined : cs.transform);
        window.__dirs[t.m41 < -1 ? 'l' : t.m41 > 1 ? 'r' : 'up']++;
      }
    }).observe(document, { attributes: true, attributeFilter: ['class'], subtree: true });
    for (const el of document.querySelectorAll('.reveal-item:not(.is-visible)')) {
      el.__dir = 1;
      const cs = getComputedStyle(el);
      const t = new DOMMatrixReadOnly(cs.transform === 'none' ? undefined : cs.transform);
      window.__dirs[t.m41 < -1 ? 'l' : t.m41 > 1 ? 'r' : 'up']++;
    }
  });

  await p.mouse.move(w / 2, h / 2);
  const stuck = [];
  let stops = 0;
  for (let i = 0; i < 500; i++) {
    await p.mouse.wheel(0, 100);
    await p.waitForTimeout(140);
    // the reader stops to read every few notches
    if (i % 5 === 4) {
      stops++;
      await p.waitForTimeout(1200);
      // where each waiting block sits in the layout, its waiting offset taken
      // off for the read: one whose home is on screen leaves a gap there
      const waiting = await p.evaluate(() => [...document.querySelectorAll('.reveal-item:not(.is-visible)')]
        .map(el => {
          const tf = el.style.transform, tr = el.style.transition;
          el.style.transition = 'none'; el.style.transform = 'none';
          const r = el.getBoundingClientRect();
          el.style.transform = tf; el.style.transition = tr;
          return { r, txt: (el.textContent || '').trim().slice(0, 30) };
        })
        .filter(({ r }) => r.height > 0 && r.top < innerHeight - 2 && r.bottom > 0)
        .map(({ r, txt }) => `"${txt}" at ${r.top | 0}px`));
      stuck.push(...waiting);
    }
    const end = await p.evaluate(() => { const s = document.querySelector('[data-testid="stMain"]'); return s.scrollTop + s.clientHeight >= s.scrollHeight - 2; });
    if (end) break;
  }
  await p.waitForTimeout(1500);
  const first = await p.evaluate(() => Object.values(window.__first));
  const moving = first.filter(r => r.op < 0.9 || Math.abs(r.dy) > 4 || Math.abs(r.dx) > 4);
  const share = first.length ? moving.length / first.length : 0;
  ok(`${tag} seen: blocks are still arriving when they come on screen`, first.length >= 10 && share >= 0.8,
    `${moving.length} of ${first.length} (${Math.round(100 * share)}%)` +
    (share < 0.8 ? '; landed before seen: ' + first.filter(r => !moving.includes(r)).slice(0, 4).map(r => `"${r.txt}"`).join(', ') : ''));
  const dirs = await p.evaluate(() => window.__dirs);
  if (w >= 700) ok(`${tag} sides: blocks arrive from the left, the right and below`, dirs.l > 0 && dirs.r > 0 && dirs.up > 0, JSON.stringify(dirs));
  else ok(`${tag} sides: a phone only rises and slides from the left`, dirs.r === 0 && dirs.up > 0, JSON.stringify(dirs));
  ok(`${tag} rest: nothing on screen still waits 1.2s after the wheel stops`, !stuck.length,
    `${stops} stops${stuck.length ? '; ' + stuck.slice(0, 4).join('; ') : ''}`);
  ok(`${tag} no page errors`, !errors.length, errors.join(' | '));
  await p.context().close();
}
await b.close();
console.log(fails ? `${fails} FAILED` : 'all arrival checks passed');
process.exit(fails ? 1 : 0);
