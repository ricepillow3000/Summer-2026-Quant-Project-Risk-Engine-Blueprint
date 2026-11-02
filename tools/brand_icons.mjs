// Renders public/favicon.png (64px, transparent), public/apple-touch-icon.png
// (180px on the page's beige, which iOS needs opaque) and public/og-image.png
// (1200x627, the link preview, from tools/og_image.html) with the crest from
// assets/logo.svg, so none of them can drift from it. Run after editing the
// logo or the preview (the preview's fonts load from Google Fonts):
//   node tools/brand_icons.mjs
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const svg = readFileSync(root + 'assets/logo.svg', 'utf8');
const browser = await chromium.launch();
for (const [file, size, pad, ground] of [
  ['favicon.png', 64, 0, 'transparent'],
  ['apple-touch-icon.png', 180, 14, '#EDE9E3'],
]) {
  const page = await browser.newPage({ viewport: { width: size, height: size } });
  await page.setContent(`<html><body style="margin:0;background:${ground}">
    <div style="width:${size}px;height:${size}px;padding:${pad}px;box-sizing:border-box">
    ${svg.replace('<svg ', '<svg width="100%" height="100%" ')}</div></body></html>`);
  await page.screenshot({ path: root + 'public/' + file, omitBackground: ground === 'transparent' });
  await page.close();
}
const og = await browser.newPage({ viewport: { width: 1200, height: 627 } });
await og.setContent(readFileSync(root + 'tools/og_image.html', 'utf8').replace('<!--CREST-->', svg),
                    { waitUntil: 'networkidle' });
await og.evaluate(() => document.fonts.ready);
await og.screenshot({ path: root + 'public/og-image.png' });
await browser.close();
