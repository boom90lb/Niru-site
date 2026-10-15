#!/usr/bin/env node
// Renders the raster brand assets from the one woodcut drawing in
// src/lib/brand.ts: the multi-size favicon, the 32-pixel icon, the touch icon
// and the social card. The SVG favicon and header mark are built with the
// site, from the same drawing. Run `npm run brand` after changing the mark.
import { chromium } from "@playwright/test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import sharp from "sharp";
import { cutMark } from "../src/lib/brand.ts";

const root = resolve(import.meta.dirname, "..");
const publicDir = resolve(root, "public");
const INK = "#0b0b0a";
const GOLD = "#e0b66e";

const svg = (mark, size, pad = 0) => {
  const [, , w] = mark.viewBox.split(" ").map(Number);
  const box = w + pad * 2;
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="${-pad} ${-pad} ${box} ${box}">` +
      `<rect x="${-pad}" y="${-pad}" width="${box}" height="${box}" fill="${INK}"/>` +
      `<path d="${mark.ink}" fill="${GOLD}"/><path d="${mark.cut}" fill="${INK}"/></svg>`,
  );
};

// Small sizes carry more line weight, as a typeface does at small optical
// sizes; the drawing itself does not change. The arrow fills the square
// corner to corner.
const iconWeight = { 16: 3.6, 32: 2.2, 48: 1.8 };
const icon = (size) =>
  sharp(
    svg(
      cutMark({ weight: iconWeight[size], orientation: "diagonal", fit: "corners", spacing: 2.6 }),
      size,
    ),
  )
    .png({ compressionLevel: 9 })
    .toBuffer();

/** An ICO container holding PNG images, which every current browser reads. */
function ico(images) {
  const header = Buffer.alloc(6 + images.length * 16);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);
  let offset = header.length;
  images.forEach(({ size, data }, index) => {
    const entry = 6 + index * 16;
    header.writeUInt8(size >= 256 ? 0 : size, entry);
    header.writeUInt8(size >= 256 ? 0 : size, entry + 1);
    header.writeUInt8(0, entry + 2);
    header.writeUInt8(0, entry + 3);
    header.writeUInt16LE(1, entry + 4);
    header.writeUInt16LE(32, entry + 6);
    header.writeUInt32LE(data.length, entry + 8);
    header.writeUInt32LE(offset, entry + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...images.map(({ data }) => data)]);
}

const sizes = [16, 32, 48];
const pngs = await Promise.all(sizes.map(async (size) => ({ size, data: await icon(size) })));
await writeFile(resolve(publicDir, "NIRUML.ico"), ico(pngs));
await writeFile(resolve(publicDir, "icon-32.png"), pngs[1].data);

// iOS rounds the touch icon's corners, so it keeps a margin.
await sharp(svg(cutMark({ weight: 1.2, orientation: "diagonal", spacing: 1.2 }), 180, 26))
  .png({ compressionLevel: 9 })
  .toFile(resolve(publicDir, "apple-touch-icon.png"));

const font = (pkg, file) =>
  pathToFileURL(resolve(root, `node_modules/@fontsource-variable/${pkg}/files/${file}`)).href;
const badge = cutMark({ weight: 1.6, spacing: 1.4 });

// The social card is set like the site: one serif family, ink on paper, and
// the mark in the one accent color.
const PAPER = "#f7f5ef";
const TEXT = "#1c1b18";
const MUTED = "#69655c";
const ACCENT = "#7a561c";
const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  @font-face { font-family: "Source Serif 4 Variable"; font-weight: 200 900; src: url("${font("source-serif-4", "source-serif-4-latin-opsz-normal.woff2")}") format("woff2"); }
  body { margin: 0; width: 1200px; height: 630px; background: ${PAPER}; color: ${TEXT}; font-family: "Source Serif 4 Variable", serif; -webkit-font-smoothing: antialiased; }
  .card { box-sizing: border-box; width: 1200px; height: 630px; padding: 72px 96px 64px; display: grid; grid-template-rows: auto 1fr auto; }
  .brand { display: flex; align-items: center; gap: 18px; font-size: 30px; font-weight: 600; }
  .brand svg { width: 128px; height: auto; }
  h1 { align-self: center; margin: 0; max-width: 980px; font-size: 54px; font-weight: 500; line-height: 1.16; letter-spacing: -0.012em; }
  .nowrap { white-space: nowrap; }
  .foot { color: ${MUTED}; font-size: 24px; }
</style>
</head>
<body>
<div class="card">
  <div class="brand"><svg viewBox="${badge.viewBox}"><path d="${badge.ink}" fill="${ACCENT}"/><path d="${badge.cut}" fill="${PAPER}"/></svg>Niru ML</div>
  <h1>Niru ML is building AMLO, a system designed to do <span class="nowrap">machine-learning</span> research at a fraction of the cost, and to show its work at every step.</h1>
  <div class="foot">niru.ml</div>
</div>
</body>
</html>`;

const scratch = await mkdtemp(join(tmpdir(), "niru-brand-"));
try {
  const cardPath = join(scratch, "card.html");
  await writeFile(cardPath, html, "utf8");
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
    await page.goto(pathToFileURL(cardPath).href, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    const screenshot = await page.screenshot({ type: "png" });
    await sharp(screenshot)
      .png({ compressionLevel: 9, palette: true })
      .toFile(resolve(publicDir, "niru-social-card.png"));
  } finally {
    await browser.close();
  }
} finally {
  await rm(scratch, { recursive: true, force: true });
}

console.log(
  "Rendered public/NIRUML.ico, public/icon-32.png, public/apple-touch-icon.png, public/niru-social-card.png",
);
