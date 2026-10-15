/**
 * The glyphs of the diagrams, after Abbott & Zardini (arXiv:2604.07242): an axis
 * is a wire, a copy a dot, a sum a cup, an add ⊕, a learned map a chipped box,
 * a norm a circle, a softmax a wedge, a mask a square with its lower half
 * hatched, an index map a hexagon. A sketch collects the primitives of one
 * view and the boxes in it that open into another view.
 */
import { scriptSize, textWidth, type Ink, type Prim, type V3 } from "./engine.ts";

/**
 * Pixels per scene unit at the design size. The registry builds each view
 * once to learn its scale, sets this, and builds it again, so every box that
 * holds text is wide enough for the text at the size it is drawn.
 */
export const design = { ppu: 20 };

/** The width, in scene units, of a label set at a given size. */
export function textUnits(
  text: string,
  size: number,
  o: { italic?: boolean; weight?: number; script?: string | undefined } = {},
) {
  return (
    (textWidth(text, size, o.italic, o.weight) +
      (o.script ? textWidth(o.script, scriptSize(size), o.italic) : 0)) /
    design.ppu
  );
}

export const LANE = 1.1;

/** The width a hexagon takes to hold its name. */
export const hexWidth = (name: string, sup = "") =>
  Math.max(2.6, name.length * 0.36 + sup.length * 0.34 + 1.5);

/**
 * One piece of a chain: a box, an index map, a small operation or a norm on the wire. A piece
 * with a key is drawn, with the stretch of wire that leads into it, as its own keyed group. A
 * ghost is the piece as it is about to be inserted or has just been deleted: drawn, invisible,
 * pressed to nothing where it would start, so a rewrite can grow it out of the wire or shrink
 * it back.
 */
export type Piece = { key?: string; ghost?: boolean } & (
  | {
      kind: "slab";
      w: number;
      h?: number;
      label?: string;
      sub?: string;
      note?: string;
      chip?: boolean;
      dashed?: boolean;
      size?: number;
      port?: string;
      ink?: Ink;
      open?: { id: string; label: string; child: string };
    }
  | {
      kind: "hex";
      name: string;
      sup?: string;
      w?: number;
      open?: { id: string; label: string; child: string };
    }
  | { kind: "op"; name: string; w?: number; sub?: string; sup?: string; ink?: Ink }
  | { kind: "norm"; port?: string; ink?: Ink; plain?: boolean }
  | { kind: "softmax" }
  | { kind: "lanes"; lanes: Lane[] }
  | { kind: "wire"; w: number }
  | { kind: "at"; then: (x: number) => void }
);

export interface Lane {
  name: string;
  ink: Ink;
  sub?: string;
}

/** A box that opens into another view, and the face it opens from. */
export interface Openable {
  id: string;
  label: string;
  child: string;
  face: V3[];
  /** The key its box is drawn under, so a reader who hides the box cannot open it. */
  key?: string;
}

type XY = [number, number];

interface At {
  z?: number;
  layer?: number;
}

export class Sketch {
  prims: Prim[] = [];
  openables: Openable[] = [];
  private keying: { key: string; n: number } | null = null;
  private unit: string | null = null;
  private inside: string | null = null;

  add(...p: Prim[]) {
    const g = this.keying;
    for (const x of p)
      this.prims.push({
        ...x,
        ...(g ? { key: `${g.key}:${g.n++}` } : {}),
        ...(this.unit !== null && x.group === undefined ? { group: this.unit } : {}),
        ...(this.inside !== null && x.within === undefined ? { within: this.inside } : {}),
      });
  }

  /**
   * Draw a group whose primitives are keyed in order, so two drawings of one layout can be
   * matched piece by piece when one is rewritten into the other.
   */
  keyed(key: string, draw: () => void) {
    const outer = this.keying;
    this.keying = { key: outer ? `${outer.key}/${key}` : key, n: 0 };
    try {
      draw();
    } finally {
      this.keying = outer;
    }
  }

  /**
   * Draw a panel or a sheet and what is printed on it as one group: the renderer draws it
   * whole, face and strokes together, in depth order with the other groups of its layer.
   */
  group(name: string, draw: () => void) {
    const outer = this.unit;
    this.unit = name;
    try {
      draw();
    } finally {
      this.unit = outer;
    }
  }

  /** Draw what is printed in a box (a face with `box: name`); the text lint holds it inside. */
  within(box: string, draw: () => void) {
    const outer = this.inside;
    this.inside = box;
    try {
      draw();
    } finally {
      this.inside = outer;
    }
  }

  /** Smooth circuit-diagram curve through waypoints, level at each one. */
  static curve(pts: XY[], z: number, steps = 16): V3[] {
    const out: V3[] = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i]!;
      const [x1, y1] = pts[i + 1]!;
      if (y0 === y1) {
        if (i === 0) out.push([x0, y0, z]);
        out.push([x1, y1, z]);
        continue;
      }
      const k = (x1 - x0) * 0.5;
      for (let s = i === 0 ? 0 : 1; s <= steps; s++) {
        const t = s / steps;
        const a = (1 - t) ** 3;
        const b = 3 * (1 - t) ** 2 * t;
        const c = 3 * (1 - t) * t * t;
        const d = t ** 3;
        out.push([a * x0 + b * (x0 + k) + c * (x1 - k) + d * x1, (a + b) * y0 + (c + d) * y1, z]);
      }
    }
    return out;
  }

  /**
   * An array: one wire per axis, offset vertically, following one path. Axis
   * names go left of the start ("start"), right of the end ("end"), or above
   * each wire just after the start ("over"), for a bundle that leaves a box.
   */
  bundle(
    path: XY[],
    lanes: Lane[],
    o: At & { label?: "start" | "end" | "over"; labelDx?: number; w?: number; gap?: number } = {},
  ) {
    const z = o.z ?? 0;
    const layer = o.layer ?? 0;
    // Lanes are offset vertically, which keeps a steep S-curve free of loops; on a stretch
    // that runs straight up or down they are offset sideways, so the array keeps its width.
    const centre = Sketch.curve(path, z);
    const normals = centre.map((_, k) => {
      const a = centre[Math.max(0, k - 1)]!;
      const b = centre[Math.min(centre.length - 1, k + 1)]!;
      const tx = b[0] - a[0];
      const ty = b[1] - a[1];
      if (Math.abs(tx) > 1e-6 || !ty) return [0, tx < 0 ? -1 : 1] as const;
      return [ty < 0 ? 1 : -1, 0] as const;
    });
    lanes.forEach((lane, i) => {
      const off = ((lanes.length - 1) / 2 - i) * (o.gap ?? LANE);
      const pts = centre.map(
        (q, k) => [q[0] + normals[k]![0] * off, q[1] + normals[k]![1] * off, z] as V3,
      );
      const p = [pts[0]!, pts[pts.length - 1]!].map((q) => [q[0], q[1]] as XY);
      this.add({ kind: "stroke", pts, w: o.w ?? 1.5, ink: lane.ink, layer });
      if (o.label) {
        const top = ((lanes.length - 1) / 2) * (o.gap ?? LANE);
        const [x, y] =
          o.label === "end"
            ? p[1]!
            : o.label === "over"
              ? ([path[0]![0], path[0]![1] + top] as XY)
              : p[0]!;
        this.add({
          kind: "label",
          at: [x, y, z],
          text: lane.name,
          ...(lane.sub ? { sub: lane.sub } : {}),
          size: 12,
          ink: lane.ink,
          layer,
          italic: true,
          anchor: o.label === "start" ? "end" : "start",
          dx: o.label === "start" ? -6 : o.label === "end" ? 6 : 3 + (o.labelDx ?? 0) + i * 13,
          dy: o.label === "over" ? -6 : 4,
        });
      }
    });
  }

  /** One wire, for a scalar or a parameter. */
  wire(path: XY[], ink: Ink, o: At & { w?: number; dash?: number[] } = {}) {
    this.add({
      kind: "stroke",
      pts: Sketch.curve(path, o.z ?? 0),
      w: o.w ?? 1.5,
      ink,
      layer: o.layer ?? 0,
      ...(o.dash ? { dash: o.dash } : {}),
    });
  }

  text(
    x: number,
    y: number,
    text: string,
    o: At & Partial<Omit<Prim & { kind: "label" }, "kind" | "at" | "text">> = {},
  ) {
    this.add({
      kind: "label",
      at: [x, y, o.z ?? 0],
      text,
      size: o.size ?? 12,
      ink: o.ink ?? "ink",
      layer: o.layer ?? 0,
      anchor: o.anchor ?? "start",
      ...(o.italic ? { italic: true } : {}),
      ...(o.weight ? { weight: o.weight } : {}),
      ...(o.dx ? { dx: o.dx } : {}),
      ...(o.dy ? { dy: o.dy } : {}),
      ...(o.sub ? { sub: o.sub } : {}),
      ...(o.sup ? { sup: o.sup } : {}),
      ...(o.alpha !== undefined ? { alpha: o.alpha } : {}),
    });
  }

  /**
   * A slab standing in the diagram plane: top and side faces hatched, the
   * front face paper. A learned map has its lower-left corner chipped; a
   * dashed edge marks a stand-in, or what a plan recomputes. Returns the front face.
   */
  slab(
    cx: number,
    cy: number,
    w: number,
    h: number,
    o: At & {
      chip?: boolean;
      dashed?: boolean;
      label?: string;
      sub?: string;
      note?: string;
      size?: number;
      ink?: Ink;
    } = {},
  ): V3[] {
    const z = o.z ?? 0;
    const layer = o.layer ?? 0;
    // Tall enough for its label, and its note under the label.
    h = Math.max(h, o.label ? ((o.size ?? 17) * (o.note ? 2.6 : 1.7)) / design.ppu + 0.5 : 0);
    const x0 = cx - w / 2;
    const x1 = cx + w / 2;
    const y0 = cy - h / 2;
    const y1 = cy + h / 2;
    const t = 0.9;
    const c = o.chip ? Math.min(0.9, w / 5, h / 5) : 0;
    const dash = o.dashed ? { dash: [5, 4] } : {};
    this.add({
      kind: "face",
      pts: [
        [x0, y1, z - t],
        [x1, y1, z - t],
        [x1, y1, z],
        [x0, y1, z],
      ],
      layer,
      fill: "paper",
      edge: "ink",
      w: 1.1,
      hatch: { angle: Math.PI / 4, gap: 3.2, ink: "ink", w: 0.6 },
      ...dash,
    });
    this.add({
      kind: "face",
      pts: [
        [x1, y0, z],
        [x1, y1, z],
        [x1, y1, z - t],
        [x1, y0, z - t],
      ],
      layer,
      fill: "paper",
      edge: "ink",
      w: 1.1,
      hatch: { angle: -Math.PI / 3, gap: 2.1, ink: "ink", w: 0.6 },
      ...dash,
    });
    const front: V3[] = c
      ? [
          [x0, y1, z],
          [x1, y1, z],
          [x1, y0, z],
          [x0 + c, y0, z],
          [x0, y0 + c, z],
        ]
      : [
          [x0, y1, z],
          [x1, y1, z],
          [x1, y0, z],
          [x0, y0, z],
        ];
    this.add({
      kind: "face",
      pts: front,
      layer,
      fill: "paper",
      edge: o.ink ?? "ink",
      w: 1.4,
      ...dash,
    });
    if (o.label) {
      const size = o.size ?? 17;
      const lift = o.note ? 5 : 0;
      this.text(cx, cy, o.label, {
        z,
        layer,
        size,
        ink: o.ink ?? "ink",
        anchor: "middle",
        weight: 600,
        dy: size / 3 - lift,
        ...(o.sub ? { sub: o.sub } : {}),
      });
      if (o.note)
        this.text(cx, cy, o.note, {
          z,
          layer,
          size: 11,
          italic: true,
          ink: "faint",
          anchor: "middle",
          dy: size / 3 + 12,
        });
    }
    return [
      [x0, y1, z],
      [x1, y1, z],
      [x1, y0, z],
      [x0, y0, z],
    ];
  }

  /** A sheet a layer of the diagram is drawn on. */
  sheet(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    o: At & { label?: string; box?: string } = {},
  ) {
    const z = o.z ?? 0;
    const layer = o.layer ?? 0;
    this.add({
      kind: "face",
      pts: [
        [x0, y1, z],
        [x1, y1, z],
        [x1, y0, z],
        [x0, y0, z],
      ],
      layer,
      fill: "paper",
      edge: "faint",
      w: 1,
      under: true,
      ...(o.box ? { box: o.box } : {}),
    });
    if (o.label)
      this.text(x1, y1, o.label, {
        z,
        layer,
        size: 11,
        ink: "faint",
        italic: true,
        anchor: "end",
        dx: -5,
        dy: 14,
      });
  }

  copy(x: number, ys: number[], inks: Ink[], o: At = {}) {
    ys.forEach((y, i) =>
      this.add({
        kind: "dot",
        at: [x, y, o.z ?? 0],
        r: 3.2,
        ink: inks[i] ?? "ink",
        layer: o.layer ?? 0,
      }),
    );
  }

  plus(x: number, y: number, o: At = {}) {
    this.add({
      kind: "dot",
      at: [x, y, o.z ?? 0],
      r: 8,
      ink: "ink",
      layer: o.layer ?? 0,
      hollow: true,
    });
    this.text(x, y, "+", { ...o, size: 15, anchor: "middle", weight: 600, dy: 5 });
  }

  /** A contraction: two wires joined by a cup, the summed index. */
  cup(x: number, yTop: number, yBot: number, ink: Ink, o: At = {}) {
    const r = (yTop - yBot) / 2;
    const cy = (yTop + yBot) / 2;
    const pts: V3[] = [];
    for (let i = 0; i <= 20; i++) {
      const a = Math.PI / 2 - (Math.PI * i) / 20;
      pts.push([x + Math.cos(a) * Math.min(r * 0.9, 1.6), cy + Math.sin(a) * r, o.z ?? 0]);
    }
    this.add({ kind: "stroke", pts, w: 1.5, ink, layer: o.layer ?? 0 });
  }

  softmax(x: number, y: number, o: At & { h?: number } = {}) {
    const z = o.z ?? 0;
    const h = o.h ?? 1.1;
    this.add({
      kind: "face",
      pts: [
        [x + 0.9, y + h, z],
        [x + 0.9, y - h, z],
        [x - 0.9, y, z],
      ],
      layer: o.layer ?? 0,
      fill: "paper",
      edge: "ink",
      w: 1.3,
    });
  }

  /** The causal mask: a square with its lower half hatched. */
  mask(x: number, y: number, o: At = {}) {
    const z = o.z ?? 0;
    const layer = o.layer ?? 0;
    const s = 1.1;
    this.add({
      kind: "face",
      pts: [
        [x - s, y + s, z],
        [x + s, y + s, z],
        [x + s, y - s, z],
        [x - s, y - s, z],
      ],
      layer,
      fill: "paper",
      edge: "ink",
      w: 1.3,
    });
    this.add({
      kind: "face",
      pts: [
        [x - s, y + s, z],
        [x + s, y - s, z],
        [x - s, y - s, z],
      ],
      layer,
      fill: "none",
      edge: "ink",
      w: 1,
      hatch: { angle: Math.PI / 4, gap: 2.2, ink: "ink", w: 0.7 },
    });
  }

  /** A small elementwise or fixed operation: a plain box with a name. */
  op(
    x: number,
    y: number,
    name: string,
    o: At & { w?: number; h?: number; sub?: string; sup?: string; ink?: Ink } = {},
  ) {
    const z = o.z ?? 0;
    const w = Math.max(
      o.w ?? 1.8,
      textUnits(name, 11, { italic: true, script: o.sub ?? o.sup }) + 0.9,
    );
    const h = Math.max(o.h ?? 2, (11 * (o.sub || o.sup ? 2.4 : 1.7)) / design.ppu + 0.5);
    this.add({
      kind: "face",
      pts: [
        [x - w / 2, y + h / 2, z],
        [x + w / 2, y + h / 2, z],
        [x + w / 2, y - h / 2, z],
        [x - w / 2, y - h / 2, z],
      ],
      layer: o.layer ?? 0,
      fill: "paper",
      edge: o.ink ?? "ink",
      w: 1.2,
    });
    this.text(x, y, name, {
      z,
      layer: o.layer ?? 0,
      size: 11,
      ink: o.ink ?? "ink",
      italic: true,
      anchor: "middle",
      dy: o.sub ? 2 : o.sup ? 5.5 : 4,
      ...(o.sub ? { sub: o.sub } : {}),
      ...(o.sup ? { sup: o.sup } : {}),
    });
  }

  /** An index map (a Read η node), drawn as a hexagon. */
  hexagon(
    x: number,
    y: number,
    name: string,
    o: At & { w?: number; sup?: string; r?: number } = {},
  ) {
    const z = o.z ?? 0;
    const r = o.r ?? 1.05;
    // A hexagon stretched to its name: flat top and bottom, pointed ends.
    const half =
      Math.max(
        o.w ?? hexWidth(name, o.sup),
        textUnits(name, 11, { italic: true, script: o.sup }) + 1.6 + r * 0.5,
      ) / 2;
    const pts: V3[] = [
      [x - half, y, z],
      [x - half + r * 0.62, y + r, z],
      [x + half - r * 0.62, y + r, z],
      [x + half, y, z],
      [x + half - r * 0.62, y - r, z],
      [x - half + r * 0.62, y - r, z],
    ];
    this.add({ kind: "face", pts, layer: o.layer ?? 0, fill: "paper", edge: "ink", w: 1.2 });
    this.text(x, y, name, {
      z,
      layer: o.layer ?? 0,
      size: 11,
      italic: true,
      anchor: "middle",
      dy: 4,
      ...(o.sup ? { sup: o.sup } : {}),
    });
  }

  /**
   * An index map that splits or merges an axis into its elements, drawn as a
   * tall hexagon: the array enters one flat side and its parts leave the other.
   */
  tallHex(x: number, y: number, h: number, name: string, o: At & { w?: number } = {}) {
    const z = o.z ?? 0;
    const w = Math.max(o.w ?? 2.2, textUnits(name, 11, { italic: true }) + 0.9) / 2;
    const tip = Math.min(0.9, h / 3);
    const pts: V3[] = [
      [x, y + h / 2, z],
      [x + w, y + h / 2 - tip, z],
      [x + w, y - h / 2 + tip, z],
      [x, y - h / 2, z],
      [x - w, y - h / 2 + tip, z],
      [x - w, y + h / 2 - tip, z],
    ];
    this.add({ kind: "face", pts, layer: o.layer ?? 0, fill: "paper", edge: "ink", w: 1.2 });
    this.text(x, y, name, {
      z,
      layer: o.layer ?? 0,
      size: 11,
      italic: true,
      anchor: "middle",
      dy: 4,
    });
    return { left: x - w, right: x + w };
  }

  /** RMSNorm: a circle with a V, the paper's learned norm; plain, a norm with no weight of its own. */
  norm(x: number, y: number, o: At & { r?: number; ink?: Ink; plain?: boolean } = {}) {
    const z = o.z ?? 0;
    const layer = o.layer ?? 0;
    const r = o.r ?? 0.95;
    const pts: V3[] = [];
    for (let i = 0; i < 28; i++)
      pts.push([
        x + Math.cos((i / 28) * 2 * Math.PI) * r,
        y + Math.sin((i / 28) * 2 * Math.PI) * r,
        z,
      ]);
    this.add({ kind: "face", pts, layer, fill: "paper", edge: o.ink ?? "ink", w: 1.8 });
    if (!o.plain)
      this.add({
        kind: "stroke",
        pts: [
          [x - r * 0.5, y + r * 0.45, z],
          [x, y - r * 0.55, z],
          [x + r * 0.5, y + r * 0.45, z],
        ],
        w: 1.2,
        ink: o.ink ?? "ink",
        layer,
        halo: false,
      });
  }

  /** A parameter port on a box's lower edge, with its leaf family's name. */
  port(x: number, y: number, name: string, o: At = {}) {
    this.add({
      kind: "dot",
      at: [x, y, o.z ?? 0],
      r: 2.6,
      ink: "faint",
      layer: o.layer ?? 0,
      hollow: true,
    });
    this.text(x, y, name, {
      ...o,
      size: 11,
      ink: "faint",
      italic: true,
      anchor: "middle",
      dy: 18,
    });
  }

  /** Small arrowheads along a wire mark an integer datatype, after the paper. */
  arrows(x0: number, x1: number, y: number, ink: Ink, o: At = {}) {
    const z = o.z ?? 0;
    for (let x = x0 + 0.8; x < x1 - 0.4; x += 1.6)
      this.add({
        kind: "stroke",
        pts: [
          [x - 0.3, y + 0.28, z],
          [x, y, z],
          [x - 0.3, y - 0.28, z],
        ],
        w: 1.3,
        ink,
        layer: o.layer ?? 0,
        halo: false,
      });
  }

  /**
   * A row of pieces along one wire, laid edge to edge with a short stretch of
   * the current array between each, so every wire meets what it connects.
   * Returns where the row ends.
   */
  chain(
    y: number,
    x: number,
    lanes: Lane[],
    pieces: Piece[],
    o: At & { gap?: number } = {},
  ): number {
    const gap = o.gap ?? 0.9;
    const at = { z: o.z ?? 0, layer: o.layer ?? 0 };
    let cur = lanes;
    // After a hexagon the next stretch starts from its centre, under its face, so the
    // outer wires meet its slanted edges.
    let from: number | null = null;
    const run = (x0: number, x1: number) => {
      const a = from ?? x0;
      from = null;
      if (x1 > a)
        this.bundle(
          [
            [a, y],
            [x1, y],
          ],
          cur,
          at,
        );
    };
    const span = () => ((cur.length - 1) / 2) * LANE;
    // A hexagon that changes the array's axes takes the wider of the two.
    const after = (i: number) => {
      for (let j = i + 1; j < pieces.length; j++) {
        const q = pieces[j]!;
        if (q.kind === "lanes") return q.lanes.length;
        if (q.kind !== "at") return 0;
      }
      return 0;
    };
    const piece = (i: number, p: Piece) => {
      switch (p.kind) {
        case "lanes":
          cur = p.lanes;
          break;
        case "wire":
          run(x, x + p.w);
          x += p.w;
          break;
        case "at":
          p.then(x);
          break;
        case "softmax":
          run(x, x + gap + 1.8);
          this.softmax(x + gap + 0.9, y, { ...at, h: Math.max(1.1, span() + 0.5) });
          from = x + gap + 0.9;
          x += gap + 1.8;
          break;
        case "norm":
          run(x, x + gap + 1.9);
          this.norm(x + gap + 0.95, y, {
            ...at,
            ...(p.ink ? { ink: p.ink } : {}),
            ...(p.plain ? { plain: true } : {}),
          });
          if (p.port) this.port(x + gap + 0.95, y - 0.95, p.port, at);
          x += gap + 1.9;
          break;
        case "hex": {
          const r0 = Math.max(1.05, span() + 0.45, ((after(i) - 1) / 2) * LANE + 0.45);
          const w = Math.max(
            p.w ?? hexWidth(p.name, p.sup),
            textUnits(p.name, 11, { italic: true, script: p.sup }) + 1.6 + r0 * 0.5,
          );
          run(x, x + gap + w / 2);
          // Tall enough to take every wire of the array it sits on.
          const r = r0;
          this.hexagon(x + gap + w / 2, y, p.name, {
            ...at,
            w,
            r,
            ...(p.sup ? { sup: p.sup } : {}),
          });
          if (p.open) {
            const c = x + gap + w / 2;
            this.open(p.open.id, p.open.label, p.open.child, [
              [c - w / 2, y + r, at.z],
              [c + w / 2, y + r, at.z],
              [c + w / 2, y - r, at.z],
              [c - w / 2, y - r, at.z],
            ]);
          }
          from = x + gap + w / 2;
          x += gap + w;
          break;
        }
        case "op": {
          const w = Math.max(
            p.w ?? 1.8,
            textUnits(p.name, 11, { italic: true, script: p.sub ?? p.sup }) + 0.9,
          );
          run(x, x + gap);
          this.op(x + gap + w / 2, y, p.name, {
            ...at,
            w,
            h: Math.max(2, 2 * span() + 0.9),
            ...(p.sub ? { sub: p.sub } : {}),
            ...(p.sup ? { sup: p.sup } : {}),
            ...(p.ink ? { ink: p.ink } : {}),
          });
          x += gap + w;
          break;
        }
        case "slab": {
          const h = p.h ?? Math.max(3.2, 2 * span() + 1.4);
          const pw = Math.max(
            p.w,
            p.label ? textUnits(p.label, p.size ?? 17, { weight: 600, script: p.sub }) + 1.2 : 0,
            p.note ? textUnits(p.note, 11, { italic: true }) + 1.2 : 0,
          );
          run(x, x + gap);
          const cx = x + gap + pw / 2;
          const face = this.slab(cx, y, pw, h, {
            ...at,
            ...(p.chip ? { chip: true } : {}),
            ...(p.dashed ? { dashed: true } : {}),
            ...(p.label ? { label: p.label } : {}),
            ...(p.sub ? { sub: p.sub } : {}),
            ...(p.note ? { note: p.note } : {}),
            ...(p.size ? { size: p.size } : {}),
            ...(p.ink ? { ink: p.ink } : {}),
          });
          if (p.port) this.port(cx, face[2]![1], p.port, at);
          if (p.open) this.open(p.open.id, p.open.label, p.open.child, face);
          x += gap + pw;
          break;
        }
      }
    };
    for (const [i, p] of pieces.entries()) {
      const first = this.prims.length;
      const x0 = x;
      if (p.key) this.keyed(p.key, () => piece(i, p));
      else piece(i, p);
      if (p.ghost) {
        const press = (q: V3): V3 => [x0, q[1], q[2]];
        for (let k = first; k < this.prims.length; k++) {
          const q = this.prims[k]!;
          this.prims[k] =
            q.kind === "stroke" || q.kind === "face"
              ? { ...q, pts: q.pts.map(press), alpha: 0 }
              : { ...q, at: press(q.at), ...(q.kind === "label" ? { dx: 0 } : {}), alpha: 0 };
        }
        x = x0;
      }
    }
    return x;
  }

  open(id: string, label: string, child: string, face: V3[], key?: string) {
    this.openables.push({ id, label, child, face, ...(key ? { key } : {}) });
  }
}
