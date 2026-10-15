/**
 * The plain-text side of the site's math notation (`$…$` inline, `$$…$$`
 * display): splitting text into runs and formulas, and a Unicode rendering
 * for places that cannot hold markup, such as alt text. It has no dependency
 * on KaTeX.
 */

export interface Segment {
  text?: string;
  math?: string;
  display?: boolean;
}

/** Split text into plain runs and formulas. */
export function segments(text: string): Segment[] {
  const out: Segment[] = [];
  const pattern = /\$\$([^$]+)\$\$|\$([^$]+)\$/g;
  let last = 0;
  for (const m of text.matchAll(pattern)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index) });
    out.push(m[1] !== undefined ? { math: m[1], display: true } : { math: m[2]! });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

const SUPERSCRIPT: Record<string, string> = {
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
  "-": "⁻",
  "*": "*",
};

/**
 * The same text in plain Unicode, for places that cannot hold markup (alt
 * text) and for a canvas whose math fonts failed to load.
 */
export function plainTex(text: string): string {
  return segments(text)
    .map((s) => {
      if (s.math === undefined) return s.text ?? "";
      return s.math
        .replace(/\\(?:d|t)?frac\{([^}]*)\}\{([^}]*)\}/g, "$1/($2)")
        .replace(/\\(?:text|mathrm|operatorname)\{([^}]*)\}/g, "$1")
        .replace(/\^\{([^}]*)\}/g, (_, e: string) =>
          [...e].map((c) => SUPERSCRIPT[c] ?? c).join(""),
        )
        .replace(/\^(\S)/g, (_, c: string) => SUPERSCRIPT[c] ?? c)
        .replace(/_\{([^}]*)\}/g, "$1")
        .replace(/\\left|\\right/g, "")
        .replace(/\\Delta/g, "Δ")
        .replace(/\\eta/g, "η")
        .replace(/\\lambda/g, "λ")
        .replace(/\\lVert|\\rVert|\\Vert/g, "‖")
        .replace(/\\lfloor/g, "⌊")
        .replace(/\\rfloor/g, "⌋")
        .replace(/\\times/g, "×")
        .replace(/\\cdot/g, "·")
        .replace(/\\sqrt\{([^}]*)\}/g, "√$1")
        .replace(/\{,\}/g, ",")
        .replace(/\\[,;!]/g, " ")
        .replace(/[{}]/g, "")
        .replace(/-/g, "−");
    })
    .join("");
}
