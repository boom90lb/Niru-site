/**
 * Ink: a small renderer for diagrams drawn as ink on paper.
 *
 * A scene is built in 3D (x: composition, y: products stacked, z: tiled axes
 * receding from the viewer) and projected with an orthographic, axonometric
 * camera. Primitives are vector ink: strokes, faces with hatching, dots and
 * labels. Depth is carried by layers, the planes the diagram is drawn on.
 * Farther layers are drawn first and print fainter, on paper that stays opaque.
 * A nearer layer's strokes cut a paper-coloured gap through what lies behind
 * them, the way a technical illustration shows one line passing over another.
 * Within a layer, a group (a panel or a sheet and what is printed on it) is
 * drawn whole, face and strokes together, and groups are drawn back to front.
 *
 * One draw list goes to SVG (the still image, built with the site) and to
 * Canvas2D (the live window), so both show the same drawing.
 */

export type V3 = readonly [number, number, number];
export type Ink = "ink" | "indigo" | "cinnabar" | "ochre" | "verdigris" | "faint";

/** An orthographic camera: rotation, then a uniform scale and a screen offset. */
export interface Camera {
  yaw: number;
  pitch: number;
  scale: number;
  ox: number;
  oy: number;
}

interface Common {
  /**
   * A key names a primitive across two drawings of one layout, so a rewrite can move it from its
   * place in one to its place in the other. Drawings ignore it.
   */
  key?: string;
  /** A group drawn whole and in depth order with the other groups of its layer. */
  group?: string;
  /** The face (by its `box`) this is printed in: a check, not a drawing instruction. */
  within?: string;
  /** Opacity, for a primitive on its way into or out of a drawing. */
  alpha?: number;
  /**
   * A light around the ink, in pixels at the design size, while an action happens: the live
   * window draws it, the still image leaves it out.
   */
  glow?: number;
  /** The light's ink, if not the primitive's own. */
  glowInk?: Ink;
}

export interface Stroke extends Common {
  kind: "stroke";
  pts: V3[];
  w: number;
  ink: Ink;
  layer: number;
  dash?: number[];
  halo?: boolean;
}

export interface Face extends Common {
  kind: "face";
  pts: V3[];
  layer: number;
  /** Paper, nothing, or a wash of the edge's ink: light laid over the drawing by an action. */
  fill: "paper" | "none" | "wash";
  edge?: Ink;
  w?: number;
  dash?: number[];
  hatch?: { angle: number; gap: number; ink: Ink; w: number };
  /** Drawn beneath the layer's strokes: a sheet the layer is drawn on. */
  under?: boolean;
  /** A name for this face as a box that other primitives are printed in. */
  box?: string;
}

export interface Dot extends Common {
  kind: "dot";
  at: V3;
  r: number;
  ink: Ink;
  layer: number;
  hollow?: boolean;
}

export interface Label extends Common {
  kind: "label";
  at: V3;
  text: string;
  size: number;
  ink: Ink;
  layer: number;
  anchor?: "start" | "middle" | "end";
  italic?: boolean;
  weight?: number;
  dx?: number;
  dy?: number;
  /** A subscript or superscript set after the text, smaller. */
  sub?: string;
  sup?: string;
}

export type Prim = Stroke | Face | Dot | Label;
type P2 = [number, number];

/** Rotation only: a point's screen position at unit scale, and its depth. */
export function rotate(p: V3, yaw: number, pitch: number): [number, number, number] {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x1 = p[0] * cy + p[2] * sy;
  const z1 = -p[0] * sy + p[2] * cy;
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  return [x1, p[1] * cp - z1 * sp, p[1] * sp + z1 * cp];
}

export function project(p: V3, c: Camera): P2 {
  const [x, y] = rotate(p, c.yaw, c.pitch);
  return [c.ox + x * c.scale, c.oy - y * c.scale];
}

/** The size a script after a label is set at. */
export const scriptSize = (size: number) => Math.max(9, Math.round(size * 0.72));

/**
 * An estimate of a label's advance width in pixels, from Source Serif's
 * proportions. It only reserves room when fitting a camera; the browser
 * measures the real text.
 */
export function textWidth(text: string, size: number, italic = false, weight = 400): number {
  let em = 0;
  for (const ch of text) {
    if (" ".includes(ch)) em += 0.25;
    else if ("iljtfr.,:;'′|!·()[]".includes(ch)) em += 0.32;
    else if ("mwMW".includes(ch)) em += 0.8;
    else if (/[A-Z]/.test(ch)) em += 0.66;
    else if (/[0-9]/.test(ch)) em += 0.52;
    else em += 0.5;
  }
  return em * size * 1.06 * (weight >= 600 ? 1.06 : 1) + (italic ? 0 : 0);
}

/** A label's box in pixels around its anchor: left, right, top, bottom (screen y down). */
export function labelBox(p: Label): [number, number, number, number] {
  const script = p.sub ?? p.sup;
  const w =
    textWidth(p.text, p.size, p.italic, p.weight) +
    (script ? textWidth(script, scriptSize(p.size), p.italic, p.weight) : 0);
  const x0 = p.anchor === "middle" ? -w / 2 : p.anchor === "end" ? -w : 0;
  const dx = p.dx ?? 0;
  const dy = p.dy ?? 0;
  const below = p.sub ? 0.5 * p.size : 0.25 * p.size;
  return [x0 + dx, x0 + w + dx, dy - 0.8 * p.size, dy + below];
}

/** Fit a camera so the scene, its labels included, fills a width × height box with a margin. */
export function fit(
  prims: Prim[],
  yaw: number,
  pitch: number,
  width: number,
  height: number,
  pad = 24,
  maxScale = Infinity,
): Camera {
  const pts: Array<[number, number]> = [];
  const labels: Array<{ at: [number, number]; box: [number, number, number, number] }> = [];
  for (const p of prims) {
    if (p.kind === "label") {
      const [x, y] = rotate(p.at, yaw, pitch);
      labels.push({ at: [x, y], box: labelBox(p) });
    } else
      for (const q of p.kind === "dot" ? [p.at] : p.pts) {
        const [x, y] = rotate(q, yaw, pitch);
        pts.push([x, y]);
      }
  }
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [x, y] of pts) {
    x0 = Math.min(x0, x);
    x1 = Math.max(x1, x);
    y0 = Math.min(y0, y);
    y1 = Math.max(y1, y);
  }
  let scale = Math.min(
    maxScale,
    (width - 2 * pad) / (x1 - x0 || 1),
    (height - 2 * pad) / (y1 - y0 || 1),
  );
  let b = [x0, x1, y0, y1];
  // Labels are set in pixels, so the room they need in scene units depends on the scale. Where
  // lettering takes much of the width this converges slowly, so it runs to a fixed point.
  for (let i = 0; i < 16; i++) {
    b = [x0, x1, y0, y1];
    for (const { at, box } of labels) {
      b[0] = Math.min(b[0]!, at[0] + box[0] / scale);
      b[1] = Math.max(b[1]!, at[0] + box[1] / scale);
      b[2] = Math.min(b[2]!, at[1] - box[3] / scale);
      b[3] = Math.max(b[3]!, at[1] - box[2] / scale);
    }
    scale = Math.min(
      maxScale,
      (width - 2 * pad) / (b[1]! - b[0]! || 1),
      (height - 2 * pad) / (b[3]! - b[2]! || 1),
    );
  }
  return {
    yaw,
    pitch,
    scale,
    ox: width / 2 - ((b[0]! + b[1]!) / 2) * scale,
    oy: height / 2 + ((b[2]! + b[3]!) / 2) * scale,
  };
}

/**
 * Every view is designed at the plate's full width: lettering and line
 * weights are set in pixels at this size, and the drawing at any other size
 * is this picture scaled.
 */
export const REF = { width: 1248, height: 624, pad: 28 } as const;

/** The design camera scaled to a width × height box, and the factor `k` the ink scales by. */
export function frame(
  prims: Prim[],
  yaw: number,
  pitch: number,
  width: number,
  height: number,
  maxScale = Infinity,
): { cam: Camera; k: number } {
  const ref = fit(prims, yaw, pitch, REF.width, REF.height, REF.pad, maxScale);
  const k = Math.min(width / REF.width, height / REF.height);
  return {
    k,
    cam: {
      yaw,
      pitch,
      scale: ref.scale * k,
      ox: width / 2 + (ref.ox - REF.width / 2) * k,
      oy: height / 2 + (ref.oy - REF.height / 2) * k,
    },
  };
}

/** The inverse of `rotate`: a rotated point back to scene coordinates. */
export function unrotate(q: readonly [number, number, number], yaw: number, pitch: number): V3 {
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const y = q[1] * cp + q[2] * sp;
  const z1 = -q[1] * sp + q[2] * cp;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  return [q[0] * cy - z1 * sy, y, q[0] * sy + z1 * cy];
}

/** The scene point a camera puts at the centre of the view. */
function focus(c: Camera, width: number, height: number): V3 {
  return unrotate(
    [(width / 2 - c.ox) / c.scale, (c.oy - height / 2) / c.scale, 0],
    c.yaw,
    c.pitch,
  );
}

/**
 * Interpolate two cameras for a zoom. The scale moves geometrically, and the
 * focus moves in proportion to how far the view has narrowed, so the point
 * being zoomed into keeps its place on screen. Rotation is shared by both
 * ends at every frame, so a small turn between views cannot throw a deep
 * zoom off target.
 */
export function mix(a: Camera, b: Camera, t: number, width: number, height: number): Camera {
  const scale = a.scale * Math.pow(b.scale / a.scale, t);
  const u =
    Math.abs(1 / a.scale - 1 / b.scale) < 1e-12
      ? t
      : (1 / a.scale - 1 / scale) / (1 / a.scale - 1 / b.scale);
  const fa = focus(a, width, height);
  const fb = focus(b, width, height);
  const f: V3 = [
    fa[0] + (fb[0] - fa[0]) * u,
    fa[1] + (fb[1] - fa[1]) * u,
    fa[2] + (fb[2] - fa[2]) * u,
  ];
  const yaw = a.yaw + (b.yaw - a.yaw) * t;
  const pitch = a.pitch + (b.pitch - a.pitch) * t;
  const [fx, fy] = rotate(f, yaw, pitch);
  return { yaw, pitch, scale, ox: width / 2 - fx * scale, oy: height / 2 + fy * scale };
}

/** Ink weight of a layer: the front plane at full strength, deeper planes fainter. */
export const layerAlpha = (layer: number) => (layer >= 0 ? 1 : Math.max(0.24, 1 - 0.2 * -layer));

/** Hatch lines clipped to a convex screen polygon. */
export function hatchLines(poly: P2[], angle: number, gap: number): Array<[P2, P2]> {
  // A face pressed flat has no inside to hatch.
  let area = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    area += a[0] * b[1] - b[0] * a[1];
  }
  if (Math.abs(area) < 1) return [];
  const d: P2 = [Math.cos(angle), Math.sin(angle)];
  const n: P2 = [-d[1], d[0]];
  const ts = poly.map((p) => p[0] * n[0] + p[1] * n[1]);
  const lo = Math.min(...ts);
  const hi = Math.max(...ts);
  const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length;
  const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
  const out: Array<[P2, P2]> = [];
  for (let t = Math.ceil(lo / gap) * gap; t <= hi; t += gap) {
    const base: P2 = [n[0] * t, n[1] * t];
    let tMin = -1e9;
    let tMax = 1e9;
    let ok = true;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i]!;
      const b = poly[(i + 1) % poly.length]!;
      let nx = -(b[1] - a[1]);
      let ny = b[0] - a[0];
      if ((cx - a[0]) * nx + (cy - a[1]) * ny < 0) {
        nx = -nx;
        ny = -ny;
      }
      const denom = d[0] * nx + d[1] * ny;
      const num = (base[0] - a[0]) * nx + (base[1] - a[1]) * ny;
      if (Math.abs(denom) < 1e-9) {
        if (num < 0) ok = false;
        continue;
      }
      const s = -num / denom;
      if (denom > 0) tMin = Math.max(tMin, s);
      else tMax = Math.min(tMax, s);
    }
    if (ok && tMax > tMin)
      out.push([
        [base[0] + d[0] * tMin, base[1] + d[1] * tMin],
        [base[0] + d[0] * tMax, base[1] + d[1] * tMax],
      ]);
  }
  return out;
}

/** Every point of a primitive. */
const pointsOf = (p: Prim): readonly V3[] =>
  p.kind === "stroke" || p.kind === "face" ? p.pts : [p.at];

/**
 * Layers in drawing order, each with its primitives in the order they are drawn: the layer's
 * loose sheets, then its groups from back to front, then its loose strokes, solids, dots and
 * labels. Within a group, and among the loose primitives, a scene adds the faces of a solid
 * back to front and coplanar faces in the order they overlap, so that order is kept.
 */
function layered(
  prims: Prim[],
  yaw: number,
  pitch: number,
): Array<{ layer: number; prims: Prim[] }> {
  const rank = (p: Prim) =>
    p.kind === "face" && p.under ? 0 : { stroke: 1, face: 2, dot: 3, label: 4 }[p.kind];
  const byKind = (list: Prim[]) =>
    list
      .map((p, i) => ({ p, i }))
      .sort((a, b) => rank(a.p) - rank(b.p) || a.i - b.i)
      .map(({ p }) => p);
  const layers = new Map<number, Prim[]>();
  for (const p of prims) {
    const list = layers.get(p.layer);
    if (list) list.push(p);
    else layers.set(p.layer, [p]);
  }
  return [...layers.keys()]
    .sort((a, b) => a - b)
    .map((layer) => {
      const loose: Prim[] = [];
      const groups = new Map<string, Prim[]>();
      for (const p of layers.get(layer)!) {
        if (p.group === undefined) loose.push(p);
        else {
          const g = groups.get(p.group);
          if (g) g.push(p);
          else groups.set(p.group, [p]);
        }
      }
      // A group's depth is the mean depth of its points: nearer groups are drawn later.
      const depth = (list: Prim[]) => {
        let sum = 0;
        let n = 0;
        for (const p of list)
          for (const q of pointsOf(p)) {
            sum += rotate(q, yaw, pitch)[2];
            n++;
          }
        return n ? sum / n : 0;
      };
      const ordered = [...groups.values()]
        .map((list, i) => ({ list, i, d: depth(list) }))
        .sort((a, b) => a.d - b.d || a.i - b.i);
      const sheets = loose.filter((p) => rank(p) === 0);
      return {
        layer,
        prims: [
          ...sheets,
          ...ordered.flatMap(({ list }) => byKind(list)),
          ...byKind(loose.filter((p) => rank(p) !== 0)),
        ],
      };
    });
}

// ---------------------------------------------------------------------------------------------- SVG

const r1 = (v: number) => Math.round(v * 10) / 10;
const pathOf = (pts: P2[]) =>
  pts.map((p, i) => `${i ? "L" : "M"}${r1(p[0])} ${r1(p[1])}`).join("");
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * Draw a scene as SVG markup. Colours are CSS classes, so the page's palette,
 * light or dark, decides them; a deeper layer's class mixes its inks toward
 * the paper. Presentation attributes only: no style attribute. A face named
 * as a box, and what is printed in it, carry data attributes for the text lint.
 */
export function toSvg(
  prims: Prim[],
  cam: Camera,
  size: { width: number; height: number },
  meta: { id: string; title: string; desc: string },
  k = 1,
): string {
  const P = (q: V3) => project(q, cam);
  const parts: string[] = [];
  const all = layered(
    prims.filter((p) => (p.alpha ?? 1) > 0.001),
    cam.yaw,
    cam.pitch,
  );
  all.forEach(({ layer, prims: inLayer }, index) => {
    const g: string[] = [];
    if (index > 0)
      for (const s of inLayer)
        if (s.kind === "stroke" && s.halo !== false)
          g.push(
            `<path class="ink-halo" d="${pathOf(s.pts.map(P))}" stroke-width="${r1((s.w + 5) * k)}"${opacityOf(s)}/>`,
          );
    for (const p of inLayer) g.push(svgOf(p, P, k));
    const fade = Math.round(layerAlpha(layer) * 100);
    parts.push(
      fade < 100 ? `<g class="ink-fade-${fade}">${g.join("")}</g>` : `<g>${g.join("")}</g>`,
    );
  });
  return `<svg class="ink-svg" viewBox="0 0 ${size.width} ${size.height}" role="img" aria-labelledby="${meta.id}-t ${meta.id}-d" xmlns="http://www.w3.org/2000/svg"><title id="${meta.id}-t">${esc(meta.title)}</title><desc id="${meta.id}-d">${esc(meta.desc)}</desc>${parts.join("")}</svg>`;
}

const lineWidth = (w: number, k: number) => Math.max(0.4, w * k);
const opacityOf = (p: Prim) =>
  p.alpha !== undefined && p.alpha < 1 ? ` opacity="${p.alpha.toFixed(3)}"` : "";
const marksOf = (p: Prim) =>
  `${p.kind === "face" && p.box ? ` data-box="${esc(p.box)}"` : ""}${p.within ? ` data-within="${esc(p.within)}"` : ""}`;
const dashOf = (d: number[] | undefined, k: number) => d?.map((v) => r1(v * k));

function svgOf(p: Prim, P: (q: V3) => P2, k: number): string {
  const a = `${opacityOf(p)}${marksOf(p)}`;
  switch (p.kind) {
    case "stroke": {
      const d = dashOf(p.dash, k);
      const dash = d ? ` stroke-dasharray="${d.join(" ")}"` : "";
      return `<path class="ink-line ink-${p.ink}" d="${pathOf(p.pts.map(P))}" stroke-width="${r1(lineWidth(p.w, k))}"${dash}${a}/>`;
    }
    case "face": {
      const pts = p.pts.map(P);
      const out: string[] = [];
      if (p.fill === "paper") out.push(`<path class="ink-paper" d="${pathOf(pts)}Z"${a}/>`);
      if (p.fill === "wash")
        return `<path class="ink-wash ink-fill-${p.edge ?? "ink"}" d="${pathOf(pts)}Z"${a}/>`;
      if (p.hatch) {
        const lines = hatchLines(pts, p.hatch.angle, p.hatch.gap * k);
        if (lines.length)
          out.push(
            `<path class="ink-line ink-${p.hatch.ink}" stroke-width="${r1(lineWidth(p.hatch.w, k))}" d="${lines.map(([u, v]) => `M${r1(u[0])} ${r1(u[1])}L${r1(v[0])} ${r1(v[1])}`).join("")}"${opacityOf(p)}/>`,
          );
      }
      if (p.edge) {
        const d = dashOf(p.dash, k);
        const dash = d ? ` stroke-dasharray="${d.join(" ")}"` : "";
        out.push(
          `<path class="ink-line ink-${p.edge}" d="${pathOf(pts)}Z" stroke-width="${r1(lineWidth(p.w ?? 1.2, k))}"${dash}${a}/>`,
        );
      }
      return out.join("");
    }
    case "dot": {
      const [x, y] = P(p.at);
      return p.hollow
        ? `<circle class="ink-paper ink-line ink-${p.ink}" cx="${r1(x)}" cy="${r1(y)}" r="${r1(p.r * k)}" stroke-width="${r1(lineWidth(1.4, k))}"${a}/>`
        : `<circle class="ink-fill-${p.ink}" cx="${r1(x)}" cy="${r1(y)}" r="${r1(p.r * k)}"${a}/>`;
    }
    case "label": {
      const [x, y] = P(p.at);
      const style = `${p.italic ? ' font-style="italic"' : ""}${p.weight ? ` font-weight="${p.weight}"` : ""}`;
      const script = p.sub ?? p.sup;
      const shift = (p.sub ? 0.28 * p.size : -0.42 * p.size) * k;
      const tail = script
        ? `<tspan font-size="${r1(scriptSize(p.size) * k)}" dy="${r1(shift)}">${esc(script)}</tspan>`
        : "";
      return `<text class="ink-text ink-fill-${p.ink}" x="${r1(x + (p.dx ?? 0) * k)}" y="${r1(y + (p.dy ?? 0) * k)}" font-size="${r1(p.size * k)}" text-anchor="${p.anchor ?? "start"}"${style}${a}>${esc(p.text)}${tail}</text>`;
    }
  }
}

// ------------------------------------------------------------------------------------------- Canvas

export type Palette = Record<Ink | "paper", string>;

/** A hex colour's relative brightness, 0 to 1; mid-grey if it is not a hex colour. */
function luminance(c: string): number {
  const h = c.trim().replace(/^#/, "");
  const full = h.length === 3 ? [...h].map((x) => x + x).join("") : h;
  if (!/^[0-9a-f]{6}$/i.test(full)) return 0.5;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255);
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

const mixed = new Map<string, string>();

/** A colour mixed toward the paper by a fraction: how a deeper layer prints. */
function toward(color: string, paper: string, f: number): string {
  if (f >= 1) return color;
  const memo = `${color}|${paper}|${f}`;
  const hit = mixed.get(memo);
  if (hit) return hit;
  const rgb = (c: string) => {
    const h = c.trim().replace(/^#/, "");
    const full = h.length === 3 ? [...h].map((x) => x + x).join("") : h;
    return /^[0-9a-f]{6}$/i.test(full)
      ? [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16))
      : null;
  };
  const a = rgb(color);
  const b = rgb(paper);
  const out =
    a && b ? `rgb(${a.map((v, i) => Math.round(v * f + b[i]! * (1 - f))).join(", ")})` : color;
  mixed.set(memo, out);
  return out;
}

/** Draw a scene on a 2D canvas context, in CSS pixels, at a global opacity, its ink scaled by k. */
export function drawCanvas(
  ctx: CanvasRenderingContext2D,
  prims: Prim[],
  cam: Camera,
  palette: Palette,
  alpha = 1,
  k = 1,
) {
  if (alpha <= 0.001) return;
  const P = (q: V3) => project(q, cam);
  const line = (pts: P2[], close = false) => {
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (close) ctx.closePath();
  };
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowBlur = 0;
  // Light on paper: on light paper a glow is the ink's tint, on dark paper the ink itself.
  const dark = luminance(palette.paper) < 0.5;
  const light = (c: string) => (dark ? c : toward(c, palette.paper, 0.55));
  layered(prims, cam.yaw, cam.pitch).forEach(({ layer, prims: inLayer }, index) => {
    const fade = layerAlpha(layer);
    const ink = (i: Ink) => toward(palette[i], palette.paper, fade);
    if (index > 0) {
      ctx.strokeStyle = palette.paper;
      ctx.setLineDash([]);
      for (const s of inLayer)
        if (s.kind === "stroke" && s.halo !== false) {
          ctx.globalAlpha = alpha * (s.alpha ?? 1);
          ctx.lineWidth = (s.w + 5) * k;
          line(s.pts.map(P));
          ctx.stroke();
        }
    }
    for (const p of inLayer) {
      ctx.globalAlpha = alpha * (p.alpha ?? 1);
      if (ctx.globalAlpha <= 0.001) continue;
      if (p.glow) {
        const i = p.glowInk ?? (p.kind === "face" ? (p.edge ?? "ink") : p.ink);
        ctx.shadowColor = light(palette[i]);
        ctx.shadowBlur = p.glow * k;
      }
      if (p.kind === "face" && p.fill === "wash") {
        ctx.fillStyle = ink(p.edge ?? "ink");
        line(p.pts.map(P), true);
        ctx.fill();
        ctx.shadowBlur = 0;
        continue;
      }
      switch (p.kind) {
        case "stroke":
          ctx.strokeStyle = ink(p.ink);
          ctx.lineWidth = lineWidth(p.w, k);
          ctx.setLineDash(p.dash?.map((v) => v * k) ?? []);
          line(p.pts.map(P));
          ctx.stroke();
          break;
        case "face": {
          const pts = p.pts.map(P);
          if (p.fill === "paper") {
            ctx.fillStyle = palette.paper;
            line(pts, true);
            ctx.fill();
          }
          if (p.hatch) {
            ctx.strokeStyle = ink(p.hatch.ink);
            ctx.lineWidth = lineWidth(p.hatch.w, k);
            ctx.setLineDash([]);
            ctx.beginPath();
            for (const [a, b] of hatchLines(pts, p.hatch.angle, p.hatch.gap * k)) {
              ctx.moveTo(a[0], a[1]);
              ctx.lineTo(b[0], b[1]);
            }
            ctx.stroke();
          }
          if (p.edge) {
            ctx.strokeStyle = ink(p.edge);
            ctx.lineWidth = lineWidth(p.w ?? 1.2, k);
            ctx.setLineDash(p.dash?.map((v) => v * k) ?? []);
            line(pts, true);
            ctx.stroke();
          }
          break;
        }
        case "dot": {
          const [x, y] = P(p.at);
          ctx.beginPath();
          ctx.arc(x, y, p.r * k, 0, Math.PI * 2);
          if (p.hollow) {
            ctx.fillStyle = palette.paper;
            ctx.fill();
            ctx.strokeStyle = ink(p.ink);
            ctx.lineWidth = lineWidth(1.4, k);
            ctx.setLineDash([]);
            ctx.stroke();
          } else {
            ctx.fillStyle = ink(p.ink);
            ctx.fill();
          }
          break;
        }
        case "label": {
          const [x, y] = P(p.at);
          const font = (size: number) =>
            `${p.italic ? "italic " : ""}${p.weight ?? 400} ${size * k}px "Source Serif 4 Variable", "Source Serif 4", Georgia, serif`;
          ctx.fillStyle = ink(p.ink);
          ctx.textBaseline = "alphabetic";
          const script = p.sub ?? p.sup;
          const bx = x + (p.dx ?? 0) * k;
          const by = y + (p.dy ?? 0) * k;
          if (!script) {
            ctx.font = font(p.size);
            ctx.textAlign =
              p.anchor === "middle" ? "center" : p.anchor === "end" ? "right" : "left";
            ctx.fillText(p.text, bx, by);
            break;
          }
          // A script follows the text: measure both, then set them from the left.
          ctx.textAlign = "left";
          ctx.font = font(scriptSize(p.size));
          const ws = ctx.measureText(script).width;
          ctx.font = font(p.size);
          const wm = ctx.measureText(p.text).width;
          const left =
            p.anchor === "middle" ? bx - (wm + ws) / 2 : p.anchor === "end" ? bx - wm - ws : bx;
          ctx.fillText(p.text, left, by);
          ctx.font = font(scriptSize(p.size));
          ctx.fillText(script, left + wm, by + (p.sub ? 0.28 * p.size : -0.42 * p.size) * k);
          break;
        }
      }
      if (p.glow) ctx.shadowBlur = 0;
    }
  });
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
}

/** Map every point of a scene: used to place a child view inside its parent's box. */
export function mapPrims(prims: Prim[], f: (p: V3) => V3, layerShift = 0): Prim[] {
  return prims.map((p) => {
    if (p.kind === "stroke" || p.kind === "face")
      return { ...p, pts: p.pts.map(f), layer: p.layer + layerShift };
    return { ...p, at: f(p.at), layer: p.layer + layerShift };
  });
}

/** Even-odd point-in-polygon test in screen space. */
export function inside(pt: P2, poly: P2[]): boolean {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!;
    const [xj, yj] = poly[j]!;
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi)
      hit = !hit;
  }
  return hit;
}
