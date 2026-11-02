// The whole page on every screen a visitor brings: phones, tablets, desktops.
//
//   node tools/device_check.mjs <url> [outdir]        DEVICES=ipad-air,iphone-se to pick
//
// site_check.mjs asks "does the page boot?" at one phone and one desktop size;
// deep_scan.mjs drives every control at 1440px. This opens every drawer and
// every tab at each device below and measures, in the real page:
//   - nothing scrolls sideways;
//   - no words sit on top of other words (two different blocks' text lines
//     intersecting), and no control sits on top of words that are not its own;
//   - every button, tab and link can be clicked where it is drawn (nothing
//     covers its centre);
//   - no text is cut off inside a box that hides overflow.
// John's ask (2026-10-17): the site must work on every phone and tablet, with
// no buttons over words. Screenshots of each device land in outdir. Exit code 1
// on any failure.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [,, url, outdir = '.'] = process.argv;

const DEVICES = {
  'android-360':      { width: 360,  height: 780,  scale: 3, mobile: true },
  'iphone-se':        { width: 375,  height: 667,  scale: 2, mobile: true },
  'iphone-15':        { width: 393,  height: 852,  scale: 3, mobile: true },
  'iphone-pro-max':   { width: 430,  height: 932,  scale: 3, mobile: true },
  'ipad-mini':        { width: 744,  height: 1133, scale: 2, mobile: true },
  'ipad-mini-land':   { width: 1133, height: 744,  scale: 2, mobile: true },
  'ipad-air':         { width: 820,  height: 1180, scale: 2, mobile: true },
  'ipad-air-land':    { width: 1180, height: 820,  scale: 2, mobile: true },
  'ipad-pro':         { width: 1024, height: 1366, scale: 2, mobile: true },
  'ipad-pro-land':    { width: 1366, height: 1024, scale: 2, mobile: true },
  'laptop':           { width: 1280, height: 800,  scale: 1, mobile: false },
  'desktop':          { width: 1440, height: 900,  scale: 1, mobile: false },
  'desktop-wide':     { width: 1920, height: 1080, scale: 1, mobile: false },
};
const pick = (process.env.DEVICES || '').split(',').filter(Boolean);
const names = pick.length ? pick : Object.keys(DEVICES);
const fails = [];
const ok = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) fails.push(msg); };

// Runs in the page. Everything is measured as the visitor sees it: each box is
// cut down to what its scrolling/clipping ancestors leave visible, fixed and
// sticky layers are left out (they float over whatever scrolls under), and so
// are Streamlit's hover toolbars (zero-size until a pointer rests on a table).
function measure() {
  const SKIP = '#boot-skel, [data-testid=stElementToolbar], [data-testid=stTooltipContent], [role=tooltip]';
  const vis = el => {
    if (el.closest(SKIP)) return false;
    for (let e = el; e && e !== document.body; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity === 0) return false;
      if (cs.position === 'fixed' || cs.position === 'sticky') return false;
      if (e.getAttribute('aria-hidden') === 'true') return false;
    }
    return true;
  };
  // the part of `r` its clipping boxes leave on screen (null when none). The
  // element's own overflow clips its own text too: Streamlit 1.66's "Skip to
  // main content" link is a 1px box that hides its words until it is focused
  // (then it is fixed, left out above); counted unclipped, its hidden words
  // read as lying over the page at every width (ME-16)
  const clipOf = (el, r) => {
    let b = { l: r.left, t: r.top, r: r.right, b: r.bottom };
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.overflowX === 'visible' && cs.overflowY === 'visible') continue;
      const c = e.getBoundingClientRect();
      b = { l: Math.max(b.l, c.left), t: Math.max(b.t, c.top), r: Math.min(b.r, c.right), b: Math.min(b.b, c.bottom) };
      if (b.r - b.l < 1 || b.b - b.t < 1) return null;
    }
    return b;
  };
  const hit = (a, b, m = 2) => a.l + m < b.r && b.l + m < a.r && a.t + m < b.b && b.t + m < a.b;
  const name = el => (el.innerText || el.getAttribute('aria-label') || el.tagName).trim().replace(/\s+/g, ' ').slice(0, 40);

  // every visible line of text; its box is the glyph band (font size around
  // the line's middle), not the line-height box - a 60px numeral's leading
  // reaching into the label above is not words touching words
  const lines = [], cut = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const el = n.parentElement;
    if (!n.textContent.trim() || !el || !vis(el) || el.closest('script,style,noscript,svg,canvas,iframe')) continue;
    const fs = parseFloat(getComputedStyle(el).fontSize) || 12;
    const rg = document.createRange(); rg.selectNodeContents(n);
    for (const r of rg.getClientRects()) {
      if (r.width < 2 || r.height < 2) continue;
      const mid = (r.top + r.bottom) / 2, half = Math.min(r.height, fs * 0.8) / 2;
      const glyph = { left: r.left, right: r.right, top: mid - half, bottom: mid + half };
      const seen = clipOf(el, glyph);
      if (!seen) continue;                                  // scrolled out of its box
      // cut text: the line runs past a box that hides overflow and cannot scroll
      if (seen.r < glyph.right - 2 || seen.l > glyph.left + 2) {
        let box = el, scrolls = false;
        for (; box; box = box.parentElement) { const o = getComputedStyle(box).overflowX; if (o !== 'visible') { scrolls = o === 'auto' || o === 'scroll'; break; } }
        if (!scrolls) cut.push(`"${n.textContent.trim().slice(0, 30)}" cut at ${Math.round(seen.r - seen.l)} of ${Math.round(glyph.right - glyph.left)}px`);
      }
      lines.push({ el, b: seen, text: n.textContent.trim().slice(0, 30) });
    }
  }
  const block = el => { for (let e = el; e; e = e.parentElement) {
    const d = getComputedStyle(e).display; if (!d.startsWith('inline') && d !== 'contents') return e; } return el; };
  for (const L of lines) L.blk = block(L.el);

  const words = [];
  for (let i = 0; i < lines.length; i++) for (let j = i + 1; j < lines.length; j++) {
    const a = lines[i], b = lines[j];
    if (a.blk === b.blk || a.blk.contains(b.blk) || b.blk.contains(a.blk)) continue;
    if (hit(a.b, b.b, 2)) words.push(`"${a.text}" / "${b.text}"`);
  }

  const ctrls = [...document.querySelectorAll('button, a[href], [role=button], [role=tab], input, select, textarea, [data-baseweb=select]')]
    .filter(e => vis(e)).map(e => ({ e, b: clipOf(e, e.getBoundingClientRect()) })).filter(c => c.b && c.b.r - c.b.l > 1 && c.b.b - c.b.t > 1);
  const overText = [];
  for (const c of ctrls) for (const L of lines) {
    if (c.e.contains(L.el) || L.el.contains(c.e)) continue;
    // a control's own label (Streamlit wraps widget + label in one container)
    const wrap = c.e.closest('[data-testid^="st"][data-testid$="Container"], [data-testid=stWidgetLabel], label, [data-testid=stTab]');
    if (wrap && wrap.contains(L.el)) continue;
    if (hit(c.b, L.b, 3)) { overText.push(`${name(c.e)} over "${L.text}"`); break; }
  }

  // a label cut short with an ellipsis is also text the visitor cannot read -
  // whether the ellipsis sits on the text's own element or on a box around it
  // (Streamlit 1.66 cuts a checkbox label on its wrapper, stMarkdownEllipsis,
  // and the leaf-only scan missed it at 390px; ME-16)
  for (const el of document.querySelectorAll('body *')) {
    if (!el.textContent.trim() || !vis(el)) continue;
    if (getComputedStyle(el).textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1)
      cut.push(`"${el.textContent.trim().slice(0, 40)}" ends in an ellipsis`);
  }

  return {
    sideways: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    words: [...new Set(words)], overText: [...new Set(overText)], clipped: [...new Set(cut)],
    nCtrls: ctrls.length, nLines: lines.length,
  };
}

// Every control is clickable where it is drawn: scroll it into view, then ask
// the page what sits at its centre.
async function covered(page) {
  const handles = await page.$$('button, a[href], [role=tab], [data-baseweb=select]');
  const bad = [];
  for (const h of handles) {
    const r = await h.evaluate(e => {
      const cs = getComputedStyle(e);
      for (let p = e; p; p = p.parentElement) {
        const s = getComputedStyle(p);
        if (s.display === 'none' || s.visibility === 'hidden' || s.position === 'fixed' || s.position === 'sticky' || p.getAttribute('aria-hidden') === 'true' || p.id === 'boot-skel' || p.dataset.testid === 'stElementToolbar') return null;
      }
      const b = e.getBoundingClientRect();
      if (b.width < 2 || b.height < 2 || cs.pointerEvents === 'none' || e.disabled) return null;
      e.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });   // the site scrolls smoothly; measure where it lands
      const c = e.getBoundingClientRect();
      const x = c.left + c.width / 2, y = c.top + c.height / 2;
      if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) return `${(e.innerText || e.getAttribute('aria-label') || e.tagName).trim().slice(0, 30)} off screen`;
      const top = document.elementFromPoint(x, y);
      if (!top || e === top || e.contains(top) || top.contains(e)) return '';
      return `${(e.innerText || e.getAttribute('aria-label') || e.tagName).trim().replace(/\s+/g, ' ').slice(0, 30)} covered by ${top.tagName.toLowerCase()}.${String(top.className).slice(0, 30)}`;
    }).catch(() => null);
    if (r) bad.push(r);
  }
  return bad;
}

const quiet = () => !document.querySelector('[data-testid="stStatusWidget"] *') && !document.querySelector('[data-stale="true"]');
// A page that never settles is itself a finding: record it and measure what is
// there, rather than losing every device after it.
let unsettled = [];
async function settle(page, where = '') {
  for (let i = 0; i < 2; i++) {
    const ok = await page.waitForFunction(quiet, null, { timeout: 180000 }).then(() => true, () => false);
    if (!ok) { unsettled.push(where || 'page'); return; }
    await page.waitForTimeout(600);
  }
}

const browser = await chromium.launch();
for (const dev of names) {
  const d = DEVICES[dev];
  const ctx = await browser.newContext({ viewport: { width: d.width, height: d.height },
    deviceScaleFactor: d.scale, isMobile: d.mobile, hasTouch: d.mobile });
  const page = await ctx.newPage();
  page.setDefaultTimeout(180000);
  const errors = [];
  page.on('pageerror', e => errors.push(String(e).slice(0, 160)));
  const t0 = Date.now(), step = m => console.log(`INFO  ${dev}: ${m} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  await page.goto(url);
  await page.waitForSelector('.verdict-number');
  step('headline drawn');
  await page.waitForFunction(() => /meleona\.support@gmail\.com/.test(document.body.textContent), null, { timeout: 180000 });
  await settle(page);
  await page.waitForTimeout(3000);             // hero fly-in settles

  // Drawers inside an inactive tab are not on screen, so each state opens the
  // ones a visitor could open right there (a hidden one would wait out the click).
  const openDrawers = async () => {
    for (const ex of await page.$$('[data-testid=stExpander] summary')) {
      const shut = await ex.evaluate(e => !e.parentElement.open && e.getClientRects().length > 0).catch(() => false);
      if (shut) await ex.click({ timeout: 5000 }).catch(() => {});
    }
  };
  await openDrawers();
  await settle(page);
  step('drawers open');
  const states = [['page', null]];
  const tabs = await page.$$('[role=tab]');
  for (let i = 0; i < tabs.length; i++) states.push([`tab ${i + 1}/${tabs.length}`, i]);

  const found = { words: new Set(), overText: new Set(), clipped: new Set(), covered: new Set() };
  let sideways = 0;
  for (const [, ti] of states) {
    if (ti !== null) {
      const t = (await page.$$('[role=tab]'))[ti];
      if (!t) continue;
      await t.click({ timeout: 10000 }).catch(() => {}); await settle(page, `tab ${ti + 1}`);
      await openDrawers(); await settle(page, `tab ${ti + 1} drawers`);
      const label = (await t.innerText().catch(() => '')).trim();
      if (ti === 0 || ti % 4 === 0) await page.screenshot({ path: `${outdir}/${dev}-tab-${ti + 1}.png`, fullPage: false }).catch(() => {});
      void label;
    }
    // Blocks reveal as they reach the screen (static/reveal.js): measure the
    // state each one lands in. A reveal moves only opacity and transform, never
    // layout, so this is the page the visitor sees once scrolled to each block -
    // and a block still waiting below the screen (opacity 0) is not skipped.
    await page.evaluate(() => window.__melionaReveal?.revealAll());
    const m = await page.evaluate(measure);
    sideways = Math.max(sideways, m.sideways);
    m.words.forEach(x => found.words.add(x));
    m.overText.forEach(x => found.overText.add(x));
    m.clipped.forEach(x => found.clipped.add(x));
    (await covered(page)).forEach(x => found.covered.add(x));
    step(`${ti === null ? 'page' : `tab ${ti + 1}`} measured`);
  }
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: `${outdir}/${dev}.png`, fullPage: true }).catch(() => {});
  const list = s => [...s].slice(0, 6).join('; ');
  const tag = `${dev} ${d.width}x${d.height}`;
  ok(!unsettled.length, `${tag}: the page settles after every click${unsettled.length ? ` (still busy after 3 min: ${unsettled.join(', ')})` : ''}`);
  unsettled = [];
  ok(sideways <= 1, `${tag}: nothing scrolls sideways (${sideways}px)`);
  ok(!found.words.size, `${tag}: no words on top of other words${found.words.size ? ` (${found.words.size}: ${list(found.words)})` : ''}`);
  ok(!found.overText.size, `${tag}: no control on top of words${found.overText.size ? ` (${found.overText.size}: ${list(found.overText)})` : ''}`);
  ok(!found.covered.size, `${tag}: every button, tab and link clickable where drawn${found.covered.size ? ` (${found.covered.size}: ${list(found.covered)})` : ''}`);
  ok(!found.clipped.size, `${tag}: no text cut off${found.clipped.size ? ` (${found.clipped.size}: ${list(found.clipped)})` : ''}`);
  ok(!errors.length, `${tag}: no script errors${errors.length ? ` (${errors.slice(0, 2)})` : ''}`);
  await ctx.close();
}
await browser.close();
console.log(`\n${fails.length ? fails.length + ' FAILED' : 'every device passes'}`);
process.exit(fails.length ? 1 : 0);
