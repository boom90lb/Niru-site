/**
 * Math in the site's text, written as TeX between dollar signs: `$x$` inline,
 * `$$x$$` on its own line.
 *
 * At build time KaTeX renders each formula as MathML, which needs no style
 * attributes (the site's CSP allows none) and reads as math without
 * JavaScript. The site ships no script to re-typeset it.
 */

import katex from "katex";
import { segments } from "./tex-text.ts";

export { plainTex, segments, type Segment } from "./tex-text.ts";

const escapeHtml = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/**
 * One formula as MathML. A display formula can scroll sideways on a narrow
 * screen, so it takes keyboard focus.
 */
export function tex(source: string, display = false): string {
  const mathml = katex.renderToString(source, {
    displayMode: display,
    output: "mathml",
    throwOnError: true,
  });
  return `<span class="tex${display ? " tex--display" : ""}"${display ? ' tabindex="0"' : ""}>${mathml}</span>`;
}

/** Text with `$…$` formulas, as HTML: the plain runs escaped, the formulas rendered. */
export function inlineTex(text: string): string {
  return segments(text)
    .map((s) => (s.math !== undefined ? tex(s.math, s.display) : escapeHtml(s.text ?? "")))
    .join("");
}

const decode = (s: string) =>
  s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");

/**
 * Rendered HTML (a Markdown entry) with its `$…$` formulas typeset. Only
 * text between tags is read, so attributes and code are left alone. Write
 * formulas in Markdown without `\,` or underscores, which Markdown reads
 * first.
 */
export function texHtml(html: string): string {
  return html
    .split(/(<[^>]+>)/)
    .map((part) =>
      part.startsWith("<") || !part.includes("$")
        ? part
        : segments(part)
            .map((s) => (s.math !== undefined ? tex(decode(s.math), s.display) : (s.text ?? "")))
            .join(""),
    )
    .join("");
}
