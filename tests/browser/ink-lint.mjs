// Text lint for an ink drawing's SVG, run inside the page. Every label must:
//   keep `margin` px from the frame; not touch another label; not be crossed by a visible line,
//   box edge or dot (a line hidden under a box face does not count); not be hidden by a nearer
//   face; and use only characters the site's Source Serif subsets carry.
// Boxes are the glyphs' ink bounds measured with the page's own fonts, plus a little clearance.
// Ink printed in a box (`data-within`, naming a face's `data-box`) must stay inside that face,
// `inset` px from its edges. A frame caught mid-rewrite carries opacities: ink fainter than
// `minOpacity`, crossfading in or out, is left out of every check.
export async function lint({
  margin,
  selector = "svg.ink-svg",
  mark = true,
  inset = 2,
  minOpacity = 0,
}) {
  const svg = document.querySelector(selector);
  const W = svg.viewBox.baseVal.width;
  const H = svg.viewBox.baseVal.height;
  const CLEAR = 2;
  const CLEAR_TEXT = 1;
  const ranges = [
    [0x0000, 0x00ff],
    [0x0131, 0x0131],
    [0x0152, 0x0153],
    [0x02bb, 0x02bc],
    [0x02c6, 0x02c6],
    [0x02da, 0x02da],
    [0x02dc, 0x02dc],
    [0x2000, 0x206f],
    [0x20ac, 0x20ac],
    [0x2122, 0x2122],
    [0x2191, 0x2191],
    [0x2193, 0x2193],
    [0x2212, 0x2212],
    [0x2215, 0x2215],
    [0x0100, 0x02ba],
    [0x02bd, 0x02c5],
    [0x02c7, 0x02cc],
    [0x02ce, 0x02d7],
    [0x02dd, 0x02ff],
    [0x1d00, 0x1dbf],
    [0x1e00, 0x1e9f],
    [0x1ef2, 0x1eff],
    [0x2020, 0x2020],
    [0x20a0, 0x20ab],
    [0x20ad, 0x20c0],
    [0x2113, 0x2113],
    [0x2c60, 0x2c7f],
    [0xa720, 0xa7ff],
    [0x0370, 0x0377],
    [0x037a, 0x037f],
    [0x0384, 0x038a],
    [0x038c, 0x038c],
    [0x038e, 0x03a1],
    [0x03a3, 0x03ff],
  ];
  const covered = (cp) => ranges.some(([a, b]) => cp >= a && cp <= b);
  const ctx = document.createElement("canvas").getContext("2d");
  const fam = '"Source Serif 4 Variable"';
  const font = (el, size) => {
    const cs = getComputedStyle(el);
    return `${cs.fontStyle === "italic" ? "italic " : ""}${cs.fontWeight} ${size}px ${fam}`;
  };
  const problems = [];
  const faint = (el) => +(el.getAttribute("opacity") ?? 1) < minOpacity;
  const els = [...svg.querySelectorAll("path, circle, text")].filter((el) => !faint(el));
  const order = new Map(els.map((e, i) => [e, i]));
  const num = (s) => s.match(/[MLZ]|-?\d*\.?\d+(?:e-?\d+)?/g);
  const subpaths = (d) => {
    const out = [];
    let cur = null;
    const tok = num(d);
    for (let i = 0; i < tok.length;) {
      const t = tok[i++];
      if (t === "M") {
        cur = { pts: [[+tok[i++], +tok[i++]]], closed: false };
        out.push(cur);
      } else if (t === "L") cur.pts.push([+tok[i++], +tok[i++]]);
      else if (t === "Z") cur.closed = true;
    }
    return out;
  };
  const inPoly = (p, poly) => {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i];
      const [xj, yj] = poly[j];
      if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  // Occluders: paper faces and hollow dots, with their place in drawing order.
  const occluders = [];
  const segments = [];
  const dots = [];
  for (const el of els) {
    const i = order.get(el);
    const cls = el.getAttribute("class") || "";
    if (el.tagName === "circle") {
      const c = [+el.getAttribute("cx"), +el.getAttribute("cy"), +el.getAttribute("r")];
      if (cls.includes("ink-paper")) occluders.push({ i, disk: c });
      dots.push({ i, c });
    } else if (el.tagName === "path") {
      // A halo is paper; a wash is light laid over the drawing. Neither is a line.
      if (cls.includes("ink-halo") || cls.includes("ink-wash")) continue;
      const subs = subpaths(el.getAttribute("d"));
      if (cls.includes("ink-paper") && !cls.includes("ink-line")) {
        for (const s of subs) occluders.push({ i, poly: s.pts });
        continue;
      }
      const w = +(el.getAttribute("stroke-width") || 1);
      const ink = (cls.match(/ink-(ink|indigo|cinnabar|ochre|verdigris|faint)/) || [])[1];
      for (const s of subs) {
        const pts = s.closed ? [...s.pts, s.pts[0]] : s.pts;
        for (let k = 0; k < pts.length - 1; k++)
          segments.push({ i, a: pts[k], b: pts[k + 1], w, ink });
      }
    }
  }
  const hiddenAfter = (p, i) =>
    occluders.some(
      (o) =>
        o.i > i &&
        (o.poly
          ? inPoly(p, o.poly)
          : Math.hypot(p[0] - o.disk[0], p[1] - o.disk[1]) < o.disk[2] - 0.5),
    );
  // Liang–Barsky: the part of a segment inside a box, as parameters.
  const clip = (a, b, [x0, y0, x1, y1]) => {
    let t0 = 0;
    let t1 = 1;
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    for (const [p, q] of [
      [-dx, a[0] - x0],
      [dx, x1 - a[0]],
      [-dy, a[1] - y0],
      [dy, y1 - a[1]],
    ]) {
      if (p === 0) {
        if (q < 0) return null;
      } else {
        const r = q / p;
        if (p < 0) t0 = Math.max(t0, r);
        else t1 = Math.min(t1, r);
      }
    }
    return t0 <= t1 ? [t0, t1] : null;
  };
  const texts = [];
  for (const el of svg.querySelectorAll("text")) {
    if (faint(el)) continue;
    const size = +el.getAttribute("font-size");
    const x = +el.getAttribute("x");
    const y = +el.getAttribute("y");
    const anchor = el.getAttribute("text-anchor");
    const main = el.firstChild.nodeType === 3 ? el.firstChild.data : "";
    const span = el.querySelector("tspan");
    ctx.font = font(el, size);
    const m = ctx.measureText(main);
    let width = m.width;
    let asc = m.actualBoundingBoxAscent;
    let desc = m.actualBoundingBoxDescent;
    if (span) {
      const ss = +span.getAttribute("font-size");
      const dy = +span.getAttribute("dy");
      ctx.font = font(el, ss);
      const ms = ctx.measureText(span.textContent);
      width += ms.width;
      asc = Math.max(asc, ms.actualBoundingBoxAscent - dy);
      desc = Math.max(desc, ms.actualBoundingBoxDescent + dy);
    }
    const left = anchor === "middle" ? x - width / 2 : anchor === "end" ? x - width : x;
    const box = [left - CLEAR, y - asc - CLEAR, left + width + CLEAR, y + desc + CLEAR];
    const text = el.textContent;
    texts.push({ el, text, box, i: order.get(el) });
    for (const ch of text)
      if (!covered(ch.codePointAt(0)))
        problems.push(`"${text}": "${ch}" is not in the site's fonts`);
    // A glyph the font lacks falls back: the same text measures differently under two fallbacks.
    const f = font(el, size);
    await document.fonts.load(f, text);
    ctx.font = `${f}, monospace`;
    const a = ctx.measureText(text).width;
    ctx.font = `${f}, cursive`;
    if (Math.abs(a - ctx.measureText(text).width) > 0.01)
      problems.push(`"${text}": a glyph falls back`);
  }
  const name = (t) => `"${t.text}"`;
  const bad = new Set();
  const flag = (t, msg) => {
    bad.add(t);
    problems.push(msg);
  };
  // A label wholly under a nearer face is not seen and is left alone; one partly under it is cut.
  const seen = [];
  for (const t of texts) {
    const [x0, y0, x1, y1] = t.box;
    const corners = [
      [x0 + 2, y0 + 2],
      [x1 - 2, y0 + 2],
      [x1 - 2, y1 - 2],
      [x0 + 2, y1 - 2],
      [(x0 + x1) / 2, (y0 + y1) / 2],
    ];
    const hidden = corners.filter((p) => hiddenAfter(p, t.i)).length;
    if (hidden === corners.length) continue;
    seen.push(t);
    if (hidden) flag(t, `${name(t)} is cut by a nearer face`);
    if (x0 < margin || y0 < margin || x1 > W - margin || y1 > H - margin)
      flag(t, `${name(t)} is within ${Math.round(margin)}px of the frame`);
  }
  texts.splice(0, texts.length, ...seen);
  for (let a = 0; a < texts.length; a++)
    for (let b = a + 1; b < texts.length; b++) {
      const d = CLEAR - CLEAR_TEXT;
      const p = texts[a].box.map((v, k) => (k < 2 ? v + d : v - d));
      const q = texts[b].box.map((v, k) => (k < 2 ? v + d : v - d));
      if (p[0] < q[2] && q[0] < p[2] && p[1] < q[3] && q[1] < p[3]) {
        bad.add(texts[b]);
        flag(texts[a], `${name(texts[a])} overlaps ${name(texts[b])}`);
      }
    }
  // A label set inside a hollow dot, as the + of an add, belongs to the dot.
  const ownedByDot = (t) =>
    dots.some(
      (d) =>
        Math.hypot((t.box[0] + t.box[2]) / 2 - d.c[0], (t.box[1] + t.box[3]) / 2 - d.c[1]) <
        d.c[2] * 0.6,
    );
  for (const t of texts) {
    if (ownedByDot(t)) continue;
    let crossed = 0;
    const inks = new Set();
    for (const s of segments) {
      const pad = s.w / 2;
      const box = [t.box[0] - pad, t.box[1] - pad, t.box[2] + pad, t.box[3] + pad];
      const hit = clip(s.a, s.b, box);
      if (!hit) continue;
      for (let k = 0; k <= 4; k++) {
        const u = hit[0] + ((hit[1] - hit[0]) * k) / 4;
        const p = [s.a[0] + (s.b[0] - s.a[0]) * u, s.a[1] + (s.b[1] - s.a[1]) * u];
        if (!hiddenAfter(p, s.i)) {
          crossed++;
          inks.add(s.ink);
          break;
        }
      }
    }
    if (crossed)
      flag(
        t,
        `${name(t)} is touched by ${crossed} line segment${crossed > 1 ? "s" : ""} (${[...inks].join(", ")})`,
      );
    for (const d of dots) {
      const [cx, cy, r] = d.c;
      const nx = Math.max(t.box[0], Math.min(cx, t.box[2]));
      const ny = Math.max(t.box[1], Math.min(cy, t.box[3]));
      // A label set inside a hollow dot, as in an add, is that dot's own.
      const own =
        Math.hypot((t.box[0] + t.box[2]) / 2 - cx, (t.box[1] + t.box[3]) / 2 - cy) < r * 0.6;
      if (!own && Math.hypot(cx - nx, cy - ny) < r + 0.7 && !hiddenAfter([cx, cy], d.i))
        flag(t, `${name(t)} is touched by a dot`);
    }
  }
  // Containment: ink printed in a box stays inside the box's face.
  const boxes = new Map();
  for (const el of els) {
    const b = el.getAttribute("data-box");
    if (b && el.tagName === "path" && !boxes.has(b))
      boxes.set(b, subpaths(el.getAttribute("d"))[0].pts);
  }
  // The distance from a point inside a convex polygon to its nearest edge.
  const edgeDistance = (p, poly) => {
    let best = Infinity;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [ax, ay] = poly[j];
      const [bx, by] = poly[i];
      const len = Math.hypot(bx - ax, by - ay) || 1;
      best = Math.min(best, Math.abs((bx - ax) * (ay - p[1]) - (ax - p[0]) * (by - ay)) / len);
    }
    return best;
  };
  const textBox = new Map(texts.map((t) => [t.el, t.box]));
  for (const el of els) {
    const b = el.getAttribute("data-within");
    if (!b) continue;
    const poly = boxes.get(b);
    if (!poly) {
      problems.push(`"${b}": no box of that name`);
      continue;
    }
    let pts = [];
    let pad = 0;
    if (el.tagName === "text") {
      // A label wholly under a nearer face is not seen.
      if (!textBox.has(el)) continue;
      const [x0, y0, x1, y1] = textBox.get(el);
      pts = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
      ];
    } else if (el.tagName === "circle") {
      const [cx, cy, r] = ["cx", "cy", "r"].map((n) => +el.getAttribute(n));
      pts = [
        [cx - r, cy],
        [cx + r, cy],
        [cx, cy - r],
        [cx, cy + r],
      ];
    } else {
      if (/ink-halo|ink-wash/.test(el.getAttribute("class") || "")) continue;
      pts = subpaths(el.getAttribute("d")).flatMap((sp) => sp.pts);
      pad = +(el.getAttribute("stroke-width") || 0) / 2;
    }
    const out = pts.some((p) => !inPoly(p, poly) || edgeDistance(p, poly) < inset + pad);
    if (out) {
      const what =
        el.tagName === "text"
          ? `"${el.textContent}"`
          : `a ${el.tagName === "circle" ? "dot" : "mark"}`;
      problems.push(`${what} breaks out of its box ${b}`);
    }
  }
  for (const t of mark ? bad : []) {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", t.box[0]);
    r.setAttribute("y", t.box[1]);
    r.setAttribute("width", t.box[2] - t.box[0]);
    r.setAttribute("height", t.box[3] - t.box[1]);
    r.setAttribute("fill", "none");
    r.setAttribute("stroke", "red");
    r.setAttribute("stroke-width", "1.2");
    r.setAttribute("data-lint", "");
    svg.append(r);
  }
  return problems;
}
