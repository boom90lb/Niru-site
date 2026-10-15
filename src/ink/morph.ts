/**
 * A rewrite drawn between two drawings of one layout. Primitives with the same key move from
 * their place in the first to their place in the second; an unkeyed primitive drawn identically
 * in both stays put; everything else fades out of the first and into the second. Ink printed in
 * a box that both drawings have (a panel that opens, say) moves with the box, in the box's own
 * frame, so it never leaves the box on the way. A frame is a pure function of the two drawings
 * and the fraction u.
 */
import type { Camera, Face, Prim, V3 } from "./engine.ts";

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const lerp3 = (a: V3, b: V3, u: number): V3 => [
  lerp(a[0], b[0], u),
  lerp(a[1], b[1], u),
  lerp(a[2], b[2], u),
];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** A box's frame: a corner, its two edges from that corner, and its unit normal. */
interface Basis {
  o: V3;
  e1: V3;
  e2: V3;
  n: V3;
  det: number;
}

function basis(pts: readonly V3[]): Basis | null {
  const o = pts[0]!;
  const e1 = sub(pts[1]!, o);
  const e2 = sub(pts[pts.length - 1]!, o);
  const c = cross(e1, e2);
  const len = Math.hypot(...c);
  if (len < 1e-9) return null;
  const n: V3 = [c[0] / len, c[1] / len, c[2] / len];
  return { o, e1, e2, n, det: dot(e1, cross(e2, n)) };
}

const toLocal =
  (b: Basis) =>
  (p: V3): V3 => {
    const d = sub(p, b.o);
    return [
      dot(d, cross(b.e2, b.n)) / b.det,
      dot(b.e1, cross(d, b.n)) / b.det,
      dot(b.e1, cross(b.e2, d)) / b.det,
    ];
  };

const fromLocal =
  (b: Basis) =>
  (l: V3): V3 => [
    b.o[0] + l[0] * b.e1[0] + l[1] * b.e2[0] + l[2] * b.n[0],
    b.o[1] + l[0] * b.e1[1] + l[1] * b.e2[1] + l[2] * b.n[1],
    b.o[2] + l[0] * b.e1[2] + l[1] * b.e2[2] + l[2] * b.n[2],
  ];

/** Every point of a primitive mapped. */
function carry(p: Prim, f: (q: V3) => V3): Prim {
  return p.kind === "stroke" || p.kind === "face"
    ? { ...p, pts: p.pts.map(f) }
    : { ...p, at: f(p.at) };
}

/** What a primitive looks like, apart from where it is. */
const styleOf = (p: Prim) => {
  const { key: _k, alpha: _a, ...rest } = p as Prim & Record<string, unknown>;
  const r = rest as Record<string, unknown>;
  delete r["pts"];
  delete r["at"];
  delete r["dx"];
  delete r["dy"];
  delete r["r"];
  return JSON.stringify(r);
};

/**
 * One primitive on its way from p to q: its place moves, and if its look changes too, it is
 * drawn in both looks, the second over the first (`then`). Null if the two are not the same
 * kind of thing. A label is never drawn twice: its look changes half way.
 */
function between(p: Prim, q: Prim, u: number): { now: Prim; then?: Prim } | null {
  const alpha = lerp(p.alpha ?? 1, q.alpha ?? 1, u);
  const a = alpha < 1 ? { alpha } : {};
  let place: Partial<Prim>;
  if ((p.kind === "stroke" && q.kind === "stroke") || (p.kind === "face" && q.kind === "face")) {
    if (p.pts.length !== q.pts.length) return null;
    place = { pts: p.pts.map((x, i) => lerp3(x, q.pts[i]!, u)) };
  } else if (p.kind === "dot" && q.kind === "dot")
    place = { at: lerp3(p.at, q.at, u), r: lerp(p.r, q.r, u) };
  else if (p.kind === "label" && q.kind === "label") {
    if (p.text !== q.text || p.sub !== q.sub || p.sup !== q.sup || p.size !== q.size) return null;
    // A label does not travel while it fades: one that appears or vanishes does so in place.
    if ((p.alpha ?? 1) !== (q.alpha ?? 1)) return null;
    place = {
      at: lerp3(p.at, q.at, u),
      dx: lerp(p.dx ?? 0, q.dx ?? 0, u),
      dy: lerp(p.dy ?? 0, q.dy ?? 0, u),
    };
  } else return null;
  if (styleOf(p) === styleOf(q) || p.kind === "label")
    return { now: { ...(u < 0.5 ? p : q), ...a, ...place } as Prim };
  return { now: { ...p, ...a, ...place } as Prim, then: { ...q, ...a, ...place } as Prim };
}

export interface Frame {
  /** Drawn at full strength: what stays, and what moves. */
  steady: Prim[];
  /** Fading out of the first drawing, and into the second. */
  gone: Prim[];
  come: Prim[];
  /** What changes its look as it stays or moves: each in its old look, then its new one. */
  restyled: Array<[Prim, Prim]>;
}

const boxes = (list: Prim[]) => {
  const out = new Map<string, Face>();
  for (const p of list) if (p.kind === "face" && p.box) out.set(p.box, p);
  return out;
};

export function morph(a: Prim[], b: Prim[], u: number): Frame {
  // The boxes both drawings have, and each one's frame in the first, the second and between.
  const inA = boxes(a);
  const inB = boxes(b);
  const shared = new Map<string, { a: Basis; b: Basis; at: Basis }>();
  for (const [name, fa] of inA) {
    const fb = inB.get(name);
    if (!fb || fb.pts.length !== fa.pts.length) continue;
    const ba = basis(fa.pts);
    const bb = basis(fb.pts);
    const at = basis(fa.pts.map((x, i) => lerp3(x, fb.pts[i]!, u)));
    if (ba && bb && at) shared.set(name, { a: ba, b: bb, at });
  }
  const box = (p: Prim) => (p.within === undefined ? undefined : shared.get(p.within));
  const gone = (p: Prim) => {
    const s = box(p);
    return s ? carry(carry(p, toLocal(s.a)), fromLocal(s.at)) : p;
  };
  const come = (q: Prim) => {
    const s = box(q);
    return s ? carry(carry(q, toLocal(s.b)), fromLocal(s.at)) : q;
  };
  const move = (p: Prim, q: Prim) => {
    const s = box(q);
    if (!s || p.within !== q.within) return between(p, q, u);
    const m = between(carry(p, toLocal(s.a)), carry(q, toLocal(s.b)), u);
    if (!m) return null;
    const back = fromLocal(s.at);
    return { now: carry(m.now, back), ...(m.then ? { then: carry(m.then, back) } : {}) };
  };
  const frame: Frame = { steady: [], gone: [], come: [], restyled: [] };
  const take = (m: { now: Prim; then?: Prim }) =>
    m.then ? frame.restyled.push([m.now, m.then]) : frame.steady.push(m.now);

  const sig = (p: Prim) => JSON.stringify(p);
  const keyed = new Map<string, Prim>();
  const plain = new Map<string, number>();
  for (const p of a)
    if (p.key) keyed.set(p.key, p);
    else plain.set(sig(p), (plain.get(sig(p)) ?? 0) + 1);
  const matched = new Set<string>();
  // An unkeyed primitive drawn in the same place in both, but in a different look.
  const where = (p: Prim) =>
    JSON.stringify([
      p.kind,
      p.kind === "stroke" || p.kind === "face" ? p.pts : p.at,
      p.kind === "label" ? [p.text, p.size, p.dx, p.dy] : null,
      p.group,
      p.layer,
    ]);
  const loose = new Map<string, Prim[]>();
  for (const q of b) {
    if (q.key) {
      const p = keyed.get(q.key);
      if (!p) {
        frame.come.push(come(q));
        continue;
      }
      matched.add(q.key);
      const m = move(p, q);
      if (m) take(m);
      else {
        frame.gone.push(gone(p));
        frame.come.push(come(q));
      }
      continue;
    }
    const n = plain.get(sig(q)) ?? 0;
    if (n > 0) {
      plain.set(sig(q), n - 1);
      frame.steady.push(q);
    } else {
      const list = loose.get(where(q));
      if (list) list.push(q);
      else loose.set(where(q), [q]);
    }
  }
  for (const p of a) {
    if (p.key) {
      if (!matched.has(p.key)) frame.gone.push(gone(p));
      continue;
    }
    const n = plain.get(sig(p)) ?? 0;
    if (n > 0) {
      plain.set(sig(p), n - 1);
      const q = loose.get(where(p))?.shift();
      const m = q && between(p, q, u);
      if (m) take(m);
      else {
        if (q) frame.come.push(come(q));
        frame.gone.push(gone(p));
      }
    }
  }
  for (const list of loose.values()) for (const q of list) frame.come.push(come(q));
  return frame;
}

export const band = (t: number, a: number, b: number) =>
  Math.min(1, Math.max(0, (t - a) / (b - a)));
const smooth = (t: number) => t * t * (3 - 2 * t);

/**
 * The frame of a rewrite a fraction u of the way, as one drawing, in three movements: what goes
 * fades out (u to 0.35), what moves travels (0.15 to 0.85), and what comes fades in (0.65 to
 * 1). What changes its look crossfades in place between 0.25 and 0.75. Ink printed in a box
 * leaves before the box moves and arrives after it settles. One list,
 * so the renderer draws every group, old or new, in depth order with the rest.
 */
export function rewriteFrame(a: Prim[], b: Prim[], u: number): Prim[] {
  const f = morph(a, b, smooth(band(u, 0.15, 0.85)));
  const fade = (p: Prim, alpha: number) =>
    alpha >= 1 ? p : { ...p, alpha: (p.alpha ?? 1) * alpha };
  const out = 1 - band(u, 0, 0.35);
  const outBoxed = 1 - band(u, 0, 0.2);
  const into = band(u, 0.65, 1);
  const intoBoxed = band(u, 0.8, 1);
  const look = smooth(band(u, 0.25, 0.75));
  return [
    ...f.steady,
    ...f.restyled.flatMap(([p, q]) => [fade(p, 1 - look), fade(q, look)]),
    ...f.gone.map((p) => fade(p, p.within ? outBoxed : out)),
    ...f.come.map((p) => fade(p, p.within ? intoBoxed : into)),
  ];
}

/** A camera between two: the scale moves geometrically, the rest in proportion. */
export const cameraBetween = (a: Camera, b: Camera, u: number): Camera => ({
  yaw: lerp(a.yaw, b.yaw, u),
  pitch: lerp(a.pitch, b.pitch, u),
  scale: a.scale * Math.pow(b.scale / a.scale, u),
  ox: lerp(a.ox, b.ox, u),
  oy: lerp(a.oy, b.oy, u),
});
