// Renders public/favicon.png (64px, transparent) and public/apple-touch-icon.png
// (180px on the page's beige, which iOS needs opaque) from assets/logo.svg, so
// the icons never drift from the crest. Run after editing the logo:
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
await browser.close();
