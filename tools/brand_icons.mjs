// Renders every icon the site serves from the crest in assets/logo.svg, so
// none of them can drift from it (John, 2026-11-05: the tab still showed the
// old lion; every icon is now rendered here, at the size it is shown, and
// served under a content-hashed URL by tools/public_files.py):
//
//   public/favicon.svg         the tab icon for Chrome, Edge and Firefox: vector,
//                              sharp at any size. The crest cropped to its disc and
//                              gold bezel - three hairline rings 4 units apart are
//                              one grey smudge at 16px, so the tab leaves them out.
//   public/favicon.ico         16, 32 and 48px, each drawn at its own size (PNG
//                              entries): what Windows, Google's results and any
//                              browser asking /favicon.ico get.
//   public/favicon.png         96px, the PNG tab icon (Safari) and st.set_page_config's.
//   public/icon-192.png        Android home screen / the web app manifest.
//   public/icon-512.png        install splash / the manifest.
//   public/apple-touch-icon.png  180px on the page's beige, rings included (iOS
//                              needs it opaque and rounds the corners itself).
//   public/og-image.png        1200x627 link preview, from tools/og_image.html
//                              (its fonts load from Google Fonts).
//
//   node tools/brand_icons.mjs
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const logo = readFileSync(root + 'assets/logo.svg', 'utf8');

// The tab icon: the same file, rings removed, viewBox tightened to the bezel.
const tab = logo
  .replace(/<!--[\s\S]*?-->\s*/g, '')
  .replace(/\s*<circle class="crest-ring"[^>]*\/>/g, '')
  .replace('viewBox="-60 -60 120 120"', 'viewBox="-45 -45 90 90"')
  .replace(/aria-label="[^"]*"/, 'aria-label="Meliona"');
if (tab.includes('crest-ring') || !tab.includes('viewBox="-45 -45 90 90"')) {
  throw new Error('assets/logo.svg changed shape: update the tab-icon crop in brand_icons.mjs');
}
writeFileSync(root + 'public/favicon.svg', tab);

const browser = await chromium.launch();
async function render(svg, size, pad, ground) {
  const page = await browser.newPage({ viewport: { width: size, height: size } });
  await page.setContent(`<html><body style="margin:0;background:${ground}">
    <div style="width:${size}px;height:${size}px;padding:${pad}px;box-sizing:border-box">
    ${svg.replace('<svg ', '<svg width="100%" height="100%" ')}</div></body></html>`);
  const png = await page.screenshot({ omitBackground: ground === 'transparent' });
  await page.close();
  return png;
}

const ico = [];
for (const size of [16, 32, 48]) ico.push([size, await render(tab, size, 0, 'transparent')]);
for (const [file, svg, size, pad, ground] of [
  ['favicon.png', tab, 96, 0, 'transparent'],
  ['icon-192.png', tab, 192, 0, 'transparent'],
  ['icon-512.png', tab, 512, 0, 'transparent'],
  ['apple-touch-icon.png', logo, 180, 12, '#EDE9E3'],
]) writeFileSync(root + 'public/' + file, await render(svg, size, pad, ground));

// ICO with PNG entries (Vista and later, every current browser).
const head = Buffer.alloc(6 + 16 * ico.length);
head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(ico.length, 4);
let offset = head.length;
ico.forEach(([size, png], i) => {
  const e = 6 + 16 * i;
  head.writeUInt8(size, e); head.writeUInt8(size, e + 1);
  head.writeUInt16LE(1, e + 4); head.writeUInt16LE(32, e + 6);
  head.writeUInt32LE(png.length, e + 8); head.writeUInt32LE(offset, e + 12);
  offset += png.length;
});
writeFileSync(root + 'public/favicon.ico', Buffer.concat([head, ...ico.map(([, p]) => p)]));

const og = await browser.newPage({ viewport: { width: 1200, height: 627 } });
await og.setContent(readFileSync(root + 'tools/og_image.html', 'utf8').replace('<!--CREST-->', logo),
                    { waitUntil: 'networkidle' });
await og.evaluate(() => document.fonts.ready);
await og.screenshot({ path: root + 'public/og-image.png' });
await browser.close();
