/**
 * Figure B, typed rewriting: the moves of AMLO's 0.0.14 search acting on Monad's term, then on
 * the toy DeepSeek's. The views are steps: the population and its tournament, the parent's
 * legal sites, one move of each class, two moves that reach one architecture, admission, and a
 * move inside DeepSeek's nested Repeats.
 *
 * Every number on these drawings comes from moves.ts. A step that rewrites a term has a before
 * view in the same layout, keyed piece by piece, so the window can carry one into the other.
 */
import { type Ink, textWidth, unrotate, type V3 } from "./engine.ts";
import { DSV41, KIND, schedule } from "./dsv41.ts";
import { LANE, type Lane, Sketch, textUnits } from "./kit.ts";
import { block as blockView, MONAD, swiglu as swigluView } from "./monad.ts";
import {
  choose,
  type Class,
  CLASSES,
  commuting,
  counts,
  DEC,
  deepseek,
  GATES,
  moves,
  parameters,
  PICKS,
  POPULATION,
  type Proposal,
  SEED,
  show,
  type Site,
  sites,
  tournament,
  WEIGHTS,
  weights,
} from "./moves.ts";
import { fmt, type View } from "./view.ts";

/** The ink of what a move rewrites. */
export const NEW: Ink = "verdigris";
const T: Lane = { name: "t", ink: "ink" };
const D: Lane = { name: "d", ink: "ochre" };
const F: Lane = { name: "f", ink: "cinnabar" };

export const YAW = -0.2;
export const PITCH = 0.32;
export const ILLUSTRATION = "An illustration under the registry’s rules, not a recorded search.";

/**
 * The figure's stage: one rectangle at the figure's turn, in rotated scene units, that every
 * step is drawn inside, so the window keeps one camera from the first step to the last.
 */
export const SX = [-4.6, 55] as const;
export const SY = [4.4, -24] as const;
const STAGE: readonly V3[] = (
  [
    [SX[0], SY[0]],
    [SX[1], SY[0]],
    [SX[1], SY[1]],
    [SX[0], SY[1]],
  ] as const
).map(([x, y]) => unrotate([x, y, 0], YAW, PITCH));
/** A scene point at stage coordinates: where a flat piece of lettering hangs. */
export const stageAt = (rx: number, ry: number): V3 => unrotate([rx, ry, 0], YAW, PITCH);
const staged = { stage: STAGE, yaw: YAW, pitch: PITCH };
/** The stage and the turn a view of the figures shares, so the window keeps one camera. */
export const STAGE_VIEW = staged;

// ---------------------------------------------------------------------------------- the strip

/**
 * A glyph of the block's print on an open panel: a norm or a call. Only a glyph the move
 * changed carries an ink of its own; a slot that is `gone` keeps its room and draws nothing.
 */
type Glyph = {
  kind: "norm" | "box";
  name: string;
  ink?: Ink;
  wide?: boolean;
  plain?: boolean;
  gone?: boolean;
};

interface StripState {
  L: number;
  /** Panels drawn open; the rest fold. */
  open: ReadonlySet<number>;
  /** Panels with a body of their own: an override, marked by its edge and number. */
  own: ReadonlySet<number>;
  /** The print on an open panel: the block's norms and calls, left to right. */
  print: (i: number) => Glyph[];
  /** The base body itself is rewritten, so every layer, folded or not, reprints. */
  reprint?: boolean;
  /** A note under each panel with a body of its own. */
  note?: string;
  /** Draw #W2's leaves under the strip, stacked or shared. */
  deck?: "stacked" | "shared";
  W?: number;
  top?: number;
  /** A name for this strip, for a drawing that holds more than one. */
  id?: string;
}

const BASE: Glyph[] = [
  { kind: "norm", name: "n1" },
  { kind: "box", name: "attn" },
  { kind: "norm", name: "n2" },
  { kind: "box", name: "swiglu" },
];
const always = new Set([0, 1, 2, 3, MONAD.L - 1]);

/**
 * Where each glyph of a print sits along a panel of the given span: laid out by the glyphs'
 * widths, a wider gap between the block's two halves, centred, and scaled down if the row would
 * come nearer the panel's edges than a margin.
 */
function printRow(items: readonly Glyph[], span: number, g0: number) {
  const width = (g: Glyph) => (g.kind === "norm" ? 0.44 : g.wide ? 0.72 : 0.4) * g0;
  const gap = (g: Glyph, k: number) => (k === 0 ? 0 : (g.name === "n2" ? 0.4 : 0.16) * g0);
  const total = items.reduce((t, g, k) => t + width(g) + gap(g, k), 0);
  const f = Math.min(1, (span - 0.7 * g0) / total);
  let x = (span - total * f) / 2;
  return items.map((g, k) => {
    x += gap(g, k) * f;
    const w = width(g) * f;
    const c = x + w / 2;
    x += w;
    return { g, c, w, f };
  });
}

/** A bracket under a stretch of a strip, with its name beneath. */
function bracket(
  s: Sketch,
  key: string,
  xa: number,
  xb: number,
  y: number,
  z: number,
  text: string,
  ink: Ink = "faint",
) {
  s.keyed(key, () => {
    s.add({
      kind: "stroke",
      pts: [
        [xa + 0.12, y + 0.3, z],
        [xa + 0.12, y, z],
        [xb - 0.12, y, z],
        [xb - 0.12, y + 0.3, z],
      ],
      w: 0.9,
      ink,
      layer: 0,
      halo: false,
    });
    s.text((xa + xb) / 2, y, text, {
      z,
      size: 11.5,
      italic: true,
      ink: ink === "faint" ? "ink" : ink,
      anchor: "middle",
      dy: 17,
    });
  });
}

/**
 * Monad's Repeat as a concertina strip: a panel per layer, folded where nothing is shown. Each
 * panel and its print are one group, so a nearer panel covers a farther one whole.
 */
function strip(s: Sketch, x0: number, y0: number, o: StripState) {
  const panel = (i: number) => (o.id ? `${o.id}/panel${i}` : `panel${i}`);
  const W = o.W ?? 3.2;
  const C = 0.16;
  const top = o.top ?? 2.3;
  const depth = 1;
  const xs = [x0];
  for (let i = 0; i < o.L; i++) xs.push(xs[i]! + (o.open.has(i) ? W : C));
  const zAt = (e: number) => (e % 2 === 0 ? 0 : -depth);
  const at = (i: number, u: number, y: number): V3 => [
    xs[i]! + u * (xs[i + 1]! - xs[i]!),
    y0 + y,
    zAt(i) + u * (zAt(i + 1) - zAt(i)),
  ];
  for (let i = 0; i < o.L; i++) {
    const open = o.open.has(i);
    const own = o.own.has(i);
    s.group(panel(i), () => {
      s.keyed(`panel${i}`, () => {
        s.add({
          kind: "face",
          pts: [at(i, 0, top), at(i, 1, top), at(i, 1, -top), at(i, 0, -top)],
          layer: 0,
          fill: "paper",
          edge: own ? NEW : open ? "ink" : o.reprint ? NEW : "faint",
          w: open ? 1.2 : 0.5,
          under: true,
          box: panel(i),
        });
        for (const y of [top, -top])
          s.add({
            kind: "stroke",
            pts: [at(i, 0, y), at(i, 1, y)],
            w: open ? 1.2 : 0.8,
            ink: own ? NEW : "ink",
            layer: 0,
            halo: false,
          });
        for (const [y, ink] of [
          [LANE / 2, "ink"],
          [-LANE / 2, "ochre"],
        ] as const)
          s.add({
            kind: "stroke",
            pts: [at(i, 0, y), at(i, 1, y)],
            w: open ? 1.5 : 0.6,
            ink: open ? ink : "faint",
            layer: 0,
            halo: false,
          });
      });
      if (!open) return;
      const span = xs[i + 1]! - xs[i]!;
      const g0 = W / 3.2;
      const y = top - 0.85 * g0;
      s.within(panel(i), () =>
        s.keyed(`print${i}`, () => {
          for (const { g, c, w, f } of printRow(o.print(i), span, g0))
            s.keyed(g.name, () => {
              if (g.gone) return;
              const ink = g.ink ?? "ink";
              if (g.kind === "norm") {
                const ring: V3[] = [];
                const r = w / 2;
                for (let j = 0; j <= 20; j++) {
                  const a = (j / 20) * 2 * Math.PI;
                  ring.push(at(i, (c + Math.cos(a) * r) / span, y + Math.sin(a) * r));
                }
                s.add({ kind: "stroke", pts: ring, w: 0.9, ink, layer: 0, halo: false });
              } else {
                const h = 0.36 * g0 * f;
                s.add({
                  kind: "face",
                  pts: [
                    at(i, (c - w / 2) / span, y + h),
                    at(i, (c + w / 2) / span, y + h),
                    at(i, (c + w / 2) / span, y - h),
                    at(i, (c - w / 2) / span, y - h),
                  ],
                  layer: 0,
                  fill: "paper",
                  edge: ink,
                  w: 0.9,
                });
              }
            });
        }),
      );
    });
    if (!open) continue;
    const [cx, , cz] = at(i, 0.5, -top);
    s.keyed(`num${i}`, () => {
      s.text(cx, y0 - top, `${i}`, {
        z: cz,
        size: 11,
        ink: own ? NEW : "faint",
        anchor: "middle",
        dy: 20,
      });
      if (own && o.note)
        s.text(cx, y0 - top, o.note, {
          z: cz,
          size: 11,
          italic: true,
          ink: NEW,
          anchor: "middle",
          dy: 35,
        });
    });
  }
  // The folded runs, numbered.
  for (let a = 0; a < o.L; a++) {
    if (o.open.has(a) || (a > 0 && !o.open.has(a - 1))) continue;
    let b = a;
    while (b + 1 < o.L && !o.open.has(b + 1)) b++;
    const text = a === b ? `${a}` : b === a + 1 ? `${a}, ${b}` : `${a} … ${b}`;
    s.keyed(`run${a}`, () =>
      s.text((xs[a]! + xs[b + 1]!) / 2, y0 - top, text, {
        size: 11,
        ink: "faint",
        anchor: "middle",
        dy: 20,
      }),
    );
  }
  // A slot of the Repeat: stacked, one leaf per layer, or shared, one leaf for every layer.
  const deckY = -top - 2.5;
  if (o.deck) {
    const y = deckY;
    // Shared, every slice is drawn where the one leaf is, so a rewrite gathers them there.
    const mx = (xs[0]! + xs[o.L]!) / 2;
    const mz = -depth / 2;
    const leaf: V3[] = [
      [mx - 1.3, y0 + y + 0.7, mz],
      [mx + 1.3, y0 + y + 0.7, mz],
      [mx + 1.3, y0 + y - 0.7, mz],
      [mx - 1.3, y0 + y - 0.7, mz],
    ];
    for (let i = 0; i < o.L; i++)
      s.keyed(`slice${i}`, () =>
        s.add({
          kind: "face",
          pts:
            o.deck === "shared"
              ? leaf
              : [
                  at(i, 0.18, y + 0.6),
                  at(i, 0.82, y + 0.6),
                  at(i, 0.82, y - 0.6),
                  at(i, 0.18, y - 0.6),
                ],
          layer: 0,
          fill: "paper",
          edge: o.deck === "shared" ? NEW : o.open.has(i) ? "ink" : "faint",
          w: o.deck === "shared" ? 1.3 : 0.6,
        }),
      );
    if (o.deck === "stacked") {
      const [lx, , lz] = at(0, 0.18, y - 0.6);
      s.text(lx, y0 + y - 0.6, `#W2: ${o.L} leaves, one per layer`, {
        z: lz,
        size: 11.5,
        italic: true,
        dy: 24,
      });
    } else {
      s.slab(mx, y0 + y, 2.6, 1.4, { z: mz, label: "#W2", size: 12, ink: NEW });
      s.text(mx + 1.3, y0 + y, `one leaf, shared by all ${o.L} layers`, {
        z: mz,
        size: 11.5,
        italic: true,
        ink: NEW,
        dx: 14,
        dy: 4,
      });
    }
  }
  // The Repeat, named under its panels.
  bracket(
    s,
    "repeat",
    xs[0]!,
    xs[o.L]!,
    y0 + (o.deck ? deckY - 2.6 : -top - 2.4),
    -depth / 2,
    `Repeat(block, L = ${o.L})`,
  );
  return { xs, at, end: xs[o.L]!, top };
}

const monadStrip = (o: Partial<StripState> = {}): StripState => ({
  L: MONAD.L,
  W: 4.4,
  top: 2.6,
  open: new Set([...always, ...(o.own ?? [])]),
  own: new Set(),
  print: () => BASE,
  ...o,
});

/** The stream into and out of the strip. */
function ends(s: Sketch, x0: number, x1: number, y: number) {
  s.keyed("ends", () => {
    s.bundle(
      [
        [x0 - 3, y],
        [x0, y],
      ],
      [T, D],
      { label: "start" },
    );
    s.bundle(
      [
        [x1, y],
        [x1 + 2.4, y],
      ],
      [T, D],
    );
  });
}

// ---------------------------------------------------------------------------- the definitions

interface Half {
  box: "attn" | "swiglu";
  /** Whether the norm before the call is there; a delete takes it out, leaving a ghost. */
  norm?: boolean | "ghost";
  /** A new node after the call, the library's rms_norm, or its ghost before it is inserted. */
  added?: boolean | "ghost";
}

/** The block, as its own view draws it: two halves, each a branch off the stream and back. */
function block(s: Sketch, x0: number, y: number, halves: [Half, Half]) {
  let x = x0;
  s.bundle(
    [
      [x - 2.6, y],
      [x, y],
    ],
    [T, D],
    { label: "start" },
  );
  for (const [k, h] of halves.entries())
    s.keyed(`half${k}`, () => {
      s.copy(x, [y + LANE / 2, y - LANE / 2], ["ink", "ochre"]);
      s.bundle(
        [
          [x, y],
          [x + 2.4, y + 3.6],
        ],
        [T, D],
      );
      const ghost = (g: boolean | "ghost" | undefined) => (g === "ghost" ? { ghost: true } : {});
      const end = s.chain(
        y + 3.6,
        x + 2.4,
        [T, D],
        [
          ...(h.norm === false
            ? []
            : [{ key: "norm", kind: "norm", port: "#g", ...ghost(h.norm) } as const]),
          {
            key: "box",
            kind: "slab",
            w: 4.2,
            h: 3.2,
            chip: true,
            label: h.box,
            size: 15,
          },
          ...(h.added
            ? ([
                { key: "new", kind: "norm", ink: NEW, plain: true, ...ghost(h.added) },
                {
                  key: "new-name",
                  kind: "at",
                  ...ghost(h.added),
                  then: (at: number) =>
                    s.text(at - 0.95, y + 3.6, "rms_norm", {
                      size: 11,
                      italic: true,
                      ink: NEW,
                      anchor: "middle",
                      dy: 30,
                    }),
                },
              ] as const)
            : []),
          { key: "out", kind: "wire", w: 0.8 },
        ],
      );
      s.bundle(
        [
          [end, y + 3.6],
          [end + 2.6, y],
        ],
        [T, D],
      );
      s.bundle(
        [
          [x, y],
          [end + 2.6, y],
        ],
        [T, D],
      );
      s.plus(end + 2.6, y);
      x = end + 2.6;
    });
  s.keyed("block-end", () =>
    s.bundle(
      [
        [x, y],
        [x + 2.2, y],
      ],
      [T, D],
      { label: "end" },
    ),
  );
  return x + 2.2;
}

/** SwiGLU, as its own view draws it, with its activation and the lane f named. */
function swiglu(
  s: Sketch,
  x0: number,
  y: number,
  o: { act: string; actInk?: Ink; f?: Lane; w2?: string; w2Ink?: Ink },
) {
  const f = o.f ?? F;
  const up = 2.3;
  s.bundle(
    [
      [x0 - 2.6, y],
      [x0, y],
    ],
    [T, D],
    { label: "start" },
  );
  s.copy(x0, [y + LANE / 2, y - LANE / 2], ["ink", "ochre"]);
  for (const [dy, sub] of [
    [up, "1"],
    [-up, "3"],
  ] as const) {
    s.bundle(
      [
        [x0, y],
        [x0 + 2.6, y + dy],
      ],
      [T, D],
    );
    s.slab(x0 + 4, y + dy, 2.8, 2.8, { chip: true, label: "W", sub });
    s.keyed(`flane${sub}`, () =>
      s.bundle(
        [
          [x0 + 5.4, y + dy],
          [x0 + 12, y + dy],
        ],
        [T, f],
        sub === "3" ? { label: "over", labelDx: 10 } : {},
      ),
    );
  }
  s.keyed("act", () => s.op(x0 + 8.7, y + up, o.act, o.actInk ? { ink: o.actInk } : {}));
  s.add({
    kind: "stroke",
    pts: [
      [x0 + 12, y + up - LANE, 0],
      [x0 + 12, y - up + LANE, 0],
    ],
    w: 1.2,
    ink: "faint",
    layer: 0,
    dash: [4, 4],
    halo: false,
  });
  for (const dy of [up, -up])
    s.bundle(
      [
        [x0 + 12, y + dy],
        [x0 + 14.8, y],
      ],
      [T, f],
    );
  const w2 = s.slab(x0 + 16.2, y, 2.8, 2.8, {
    chip: true,
    label: "W",
    sub: "2",
    ...(o.w2Ink ? { ink: o.w2Ink } : {}),
  });
  s.keyed("w2", () => {
    s.port(x0 + 16.2, w2[2]![1], o.w2 ?? "#W2");
  });
  s.bundle(
    [
      [x0 + 17.6, y],
      [x0 + 20, y],
    ],
    [T, D],
    { label: "end" },
  );
  return x0 + 20;
}

// ------------------------------------------------------------------------------ admission lane

/** A member of the population, as a flat card standing in the drawing. */
export function card(s: Sketch, x: number, y: number, z: number, ink: Ink = "ink") {
  s.add({
    kind: "face",
    pts: [
      [x - 0.75, y + 1.05, z],
      [x + 0.75, y + 1.05, z],
      [x + 0.75, y - 1.05, z],
      [x - 0.75, y - 1.05, z],
    ],
    layer: 0,
    fill: "paper",
    edge: ink,
    w: 1.1,
  });
}

/** Admission, drawn small beside a move: its gates on one lane, the population at the end. */
export const LANE_AT = { x: 32.4, y: -10.6, pitch: 2.3 } as const;
export const gateX = (k: number) => LANE_AT.x + 1 + k * LANE_AT.pitch;
export const DECK_X = gateX(GATES.length - 1) + 4;

function admissionLane(s: Sketch) {
  const { x, y } = LANE_AT;
  s.keyed("lane", () => {
    s.wire(
      [
        [x - 0.6, y],
        [DECK_X - 1.4, y],
      ],
      "faint",
      { w: 1.1 },
    );
    for (const [k, g] of GATES.entries()) {
      s.keyed(`gate${k}`, () => s.slab(gateX(k), y, 0.36, 2.4, {}));
      const above = k % 2 === 0;
      s.text(gateX(k), y + (above ? 1.2 : -1.2), g.name, {
        size: 11,
        italic: true,
        anchor: "middle",
        dy: above ? -17 : 17,
      });
    }
    // The population: members are cards, the youngest in front; the oldest is at the back.
    for (let c = 4; c >= 0; c--) s.keyed(`card${c}`, () => card(s, DECK_X, y, -0.6 * c));
    s.text(DECK_X, y - 1.05, "population", { size: 11, italic: true, anchor: "middle", dy: 18 });
  });
}

// ------------------------------------------------------------------------------------ ledgers

interface Row {
  name: string;
  note: string;
  value: string;
}

interface Ledger {
  title: string;
  does?: string;
  rows: Row[];
  total?: Row;
  foot?: string[];
}

/**
 * Text set as a table at one point of the stage: a heading, rows of a name, a note and a value,
 * and a total. Every line hangs from the same point, so the rows stay level under any turn of
 * the drawing. Its key names the step it belongs to, so the next step's ledger replaces it
 * rather than sliding over it.
 */
function ledger(s: Sketch, key: string, place: { rx: number; ry: number }, o: Ledger) {
  const nameW =
    Math.max(
      ...o.rows.map((r) => textWidth(r.name, 11.5, true)),
      o.total ? textWidth(o.total.name, 12.5, false, 600) : 0,
    ) + 16;
  // The widths are estimates from the font's proportions, so the columns keep some slack.
  const noteW =
    1.08 *
      Math.max(...[...o.rows, ...(o.total ? [o.total] : [])].map((r) => textWidth(r.note, 12))) +
    24;
  const valueW = Math.max(
    ...o.rows.map((r) => textWidth(r.value, 12.5)),
    o.total ? textWidth(o.total.value, 13.5, false, 600) : 0,
  );
  const right = nameW + noteW + valueW;
  const [x, y, z] = stageAt(place.rx, place.ry);
  s.keyed(key, () => {
    let dy = 0;
    s.text(x, y, o.title, { z, size: 14, weight: 600, dy });
    if (o.does) {
      dy += 20;
      s.text(x, y, o.does, { z, size: 12, italic: true, dy });
    }
    dy += 12;
    for (const r of o.rows) {
      dy += 21;
      s.text(x, y, r.name, { z, size: 11.5, italic: true, ink: "faint", dy });
      s.text(x, y, r.note, { z, size: 12, dx: nameW, dy });
      s.text(x, y, r.value, { z, size: 12.5, anchor: "end", dx: right, dy });
    }
    if (o.total) {
      dy += 28;
      s.text(x, y, o.total.name, { z, size: 12.5, weight: 600, dy });
      s.text(x, y, o.total.note, { z, size: 12, dx: nameW, dy });
      s.text(x, y, o.total.value, { z, size: 13.5, weight: 600, anchor: "end", dx: right, dy });
    }
    for (const line of o.foot ?? []) {
      dy += 19;
      s.text(x, y, line, { z, size: 11, italic: true, ink: "faint", dy });
    }
  });
}

/** Words broken into lines no wider than `max` pixels at a size, by the font's proportions. */
export function wrap(text: string, size: number, max: number, italic = true): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && 1.08 * textWidth(next, size, italic) > max) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

/**
 * A move's ledger set as its product: the command and what it does, then the five factors in a
 * row, each named above and explained below, multiplied into the recorded probability. Every
 * line hangs from one point of the stage.
 */
export function product(
  s: Sketch,
  place: { rx: number; ry: number },
  p: Pick<Proposal, "id" | "command" | "does" | "factors" | "probability">,
  foot: string[],
) {
  const [x, y, z] = stageAt(place.rx, place.ry);
  const NOTE = 11;
  const cells = [
    ...p.factors.map((f) => ({ name: f.name, value: show(f.q), note: wrap(f.note, NOTE, 150) })),
    { name: "probability", value: show(p.probability), note: wrap("recorded exactly", NOTE, 150) },
  ];
  const sign = 30;
  const widths = cells.map((c, k) =>
    Math.max(
      1.06 *
        textWidth(
          c.value,
          k === cells.length - 1 ? 16 : 15,
          false,
          k === cells.length - 1 ? 600 : 400,
        ),
      ...c.note.map((l) => 1.08 * textWidth(l, NOTE, true)),
      textWidth(c.name, 11.5, true),
    ),
  );
  s.keyed(`product-${p.id}`, () => {
    s.text(x, y, p.command, { z, size: 14, weight: 600 });
    s.text(x, y, p.does, {
      z,
      size: 12,
      italic: true,
      dx: 1.06 * textWidth(p.command, 14, false, 600) + 12,
    });
    let dx = 0;
    cells.forEach((c, k) => {
      const mid = dx + widths[k]! / 2;
      const last = k === cells.length - 1;
      s.text(x, y, c.name, {
        z,
        size: 11.5,
        italic: true,
        ink: "faint",
        anchor: "middle",
        dx: mid,
        dy: 22,
        ...(last ? { weight: 600 } : {}),
      });
      s.text(x, y, c.value, {
        z,
        size: last ? 16 : 15,
        anchor: "middle",
        dx: mid,
        dy: 41,
        ...(last ? { weight: 600 } : {}),
      });
      c.note.forEach((line, j) =>
        s.text(x, y, line, {
          z,
          size: NOTE,
          italic: true,
          ink: "faint",
          anchor: "middle",
          dx: mid,
          dy: 57 + 14 * j,
        }),
      );
      dx += widths[k]!;
      if (k < cells.length - 1)
        s.text(x, y, k === cells.length - 2 ? "=" : "×", {
          z,
          size: 15,
          ink: "faint",
          anchor: "middle",
          dx: dx + sign / 2,
          dy: 41,
        });
      dx += sign;
    });
    const lines = Math.max(...cells.map((c) => c.note.length));
    foot.forEach((line, j) =>
      s.text(x, y, line, {
        z,
        size: 11,
        italic: true,
        ink: "faint",
        dy: 57 + 14 * lines + 7 + 14 * j,
      }),
    );
  });
}

// -------------------------------------------------------------------------------- the steps

const MOVES = moves();
const move = (id: string) => MOVES.find((m) => m.id === id)!;

/** The new node of an insert, in the block's print: a plain norm after attention. */
const INSERTED: Glyph[] = [
  ...BASE.slice(0, 2),
  { kind: "norm", name: "new", plain: true, ink: NEW },
  ...BASE.slice(2),
];
/** The body reprinted: swiglu's activation changed, at every layer. */
const REPRINTED: Glyph[] = [...BASE.slice(0, 3), { kind: "box", name: "swiglu", ink: NEW }];
const BOTH: Glyph[] = [...INSERTED.slice(0, 4), { kind: "box", name: "swiglu", ink: NEW }];
/** The norm before attention deleted: its slot keeps its room, empty. */
const DELETED: Glyph[] = [{ kind: "norm", name: "n1", gone: true }, ...BASE.slice(1)];
/** swiglu at f = 1,024: a wider call. */
const WIDENED: Glyph[] = [
  ...BASE.slice(0, 3),
  { kind: "box", name: "swiglu", wide: true, ink: NEW },
];

/** The tournament's draw: four members, the best-ranked of which is the parent. */
export const DRAWN = [1, 6, 13, 19] as const;

/** The population: 24 cards in rank order; then four drawn; then the best of them, the parent. */
function population(state: "row" | "drawn" | "parent"): View {
  const s = new Sketch();
  const { P, S } = POPULATION;
  const step = 2.3;
  const x0 = -1.6;
  const y0 = -6.4;
  const lift = 5.2;
  for (let r = 1; r <= P; r++) {
    const picked = state !== "row" && DRAWN.includes(r as (typeof DRAWN)[number]);
    const x = x0 + (r - 1) * step;
    const y = y0 + (picked ? lift : 0);
    s.keyed(`card${r}`, () => {
      s.slab(x, y, 1.8, 2.6, {
        label: `${r}`,
        size: 12,
        ...(picked ? { ink: r === 1 && state === "parent" ? NEW : "ink" } : {}),
      });
      if (r === 1) {
        s.text(x, y - 1.3, "the seed,", { size: 11, italic: true, anchor: "middle", dy: 18 });
        s.text(x, y - 1.3, "Monad", { size: 11, italic: true, anchor: "middle", dy: 32 });
      }
    });
  }
  if (state !== "row")
    s.keyed("tournament", () =>
      s.text(
        x0 + (DRAWN[1] - 1) * step - 0.9,
        y0 + lift + 1.3,
        `a tournament: ${S} members, drawn without replacement; the best-ranked is the parent`,
        { size: 11.5, italic: true, dy: -24 },
      ),
    );
  if (state === "parent")
    s.keyed("parent", () =>
      s.text(x0 - 0.9, y0 + lift + 1.3, "the parent", {
        size: 11.5,
        italic: true,
        ink: NEW,
        dy: -24,
      }),
    );
  s.text(
    x0 + (P - 1) * step + 0.9,
    y0 - 1.3,
    `the population: ${P} members, ranked by selection loss`,
    { size: 11.5, italic: true, anchor: "end", dy: 26 },
  );
  const id = {
    row: "rw/population/before",
    drawn: "rw/population/drawn",
    parent: "rw/population",
  }[state];
  return {
    id,
    title: "The search’s population",
    caption: `In AMLO’s 0.0.14 design, the search keeps a population of ${P} terms. For each proposal it draws ${S} members without replacement, and the best-ranked of them is the parent, so a parent of rank r is drawn with probability C(${P} − r, ${S - 1}) / C(${P}, ${S}). The ranks here are declared for the illustration. The design is not built, and no search has run.`,
    sketch: s,
    ...staged,
    ...(state === "parent" ? { before: "rw/population/before" } : {}),
  };
}

/** The tournament's ledger: the parent's factor at the first three ranks. */
export function populationLedger(s: Sketch, at: { rx: number; ry: number }) {
  const { P, S } = POPULATION;
  const rank = (r: number): Row => ({
    name: `rank ${r}`,
    note: `${fmt(Number(choose(P - r, S - 1)))}/${fmt(Number(choose(P, S)))}`,
    value: show(tournament(P, S, r)),
  });
  ledger(s, "ledger-population", at, {
    title: "The parent’s factor",
    does: `C(${P} − r, ${S - 1}) / C(${P}, ${S}) for the parent’s rank r`,
    rows: [rank(1), rank(2), rank(3)],
    foot: ["The ranks are declared for this illustration. No search has run."],
  });
}

/** The sites of one class, counted per definition: "gpt 6, block 6, attn 11, swiglu 5". */
function perDef(list: readonly Site[]): string {
  const by = new Map<string, number>();
  for (const x of list) by.set(x.def, (by.get(x.def) ?? 0) + 1);
  return [...by].map(([d, n]) => `${d} ${n}`).join(", ");
}

/** The keys the registry drawing gives the place a site names, so a class's sites can light. */
export function siteKeys(x: Site): string[] {
  const defs = SEED.defs;
  if (x.cls === "binding") {
    if (x.at.startsWith("#")) return [`slot/${x.def}/${x.at.slice(1)}`];
    // A Repeat slot, by its family address: x/<node of block>/<slot of what that node calls>.
    const [, node, slot] = x.at.split("/");
    const n = defs.block!.nodes.find((m) => m.name === node)!;
    return defs[n.call!]!.glyph ? [`slot/block/${node}/${slot}`] : [`slot/${n.call}/${slot}`];
  }
  if (x.cls === "shape") {
    if (x.def === "gpt") return [`control/${x.at}`];
    // A parameter of a definition: drawn at the node that binds it.
    for (const d of Object.values(defs))
      for (const n of d.nodes)
        if (n.call === x.def && n.axes?.[x.at] !== undefined) return [`node/${d.name}/${n.name}`];
    return [];
  }
  return [`node/${x.def}/${x.at}`];
}

/**
 * The parent's definitions as the registry reads them: one row of nodes per definition, in
 * stable order, each node a box named by its operation, each slot a port beneath the node that
 * reads it. Every legal site is one of these boxes, ports or the declared control F.
 */
function sitesView(): View {
  const s = new Sketch();
  const defs = SEED.defs;
  const rows = ["gpt", "block", "attn", "swiglu"] as const;
  const label = (n: (typeof defs)[string]["nodes"][number]) =>
    n.op === "repeat" ? `Repeat(${n.call})` : n.op === "call" ? n.call! : n.op.replace("_", " ");
  rows.forEach((d, i) => {
    const y = 1.2 - 4.8 * i;
    s.text(-0.7, y, d, { size: 12.5, weight: 600, anchor: "end", dy: 4 });
    let x = 0;
    for (const n of defs[d]!.nodes) {
      const name = label(n);
      const w = Math.max(1.8, textUnits(name, 11, { italic: true }) + 0.9);
      const cx = x + w / 2;
      s.keyed(`node/${d}/${n.name}`, () => s.op(cx, y, name, { w }));
      // Slots this node reads: its own definition's, or a glyph's it calls (a norm's #g).
      const slots = [
        ...n.args
          .filter((a) => a.startsWith("#"))
          .map((a) => [a.slice(1), `slot/${d}/${a.slice(1)}`] as const),
        ...(n.call && defs[n.call]!.glyph
          ? Object.keys(defs[n.call]!.slots).map((k) => [k, `slot/${d}/${n.name}/${k}`] as const)
          : []),
      ];
      for (const [slot, key] of slots)
        s.keyed(slot === "E" ? `${key}:${n.name}` : key, () => s.port(cx, y - 1, `#${slot}`));
      x += w + 0.7;
    }
    if (d === "gpt") {
      const [lo, hi] = SEED.ranges.F!;
      const name = `F, ${fmt(lo)} to ${fmt(hi)}`;
      const w = textUnits(name, 11, { italic: true }) + 0.9;
      s.keyed("control/F", () => s.op(x + 1.2 + w / 2, y, name, { w }));
      s.text(x + 1.2 + w / 2, y - 1, "a declared range", {
        size: 11,
        italic: true,
        ink: "faint",
        anchor: "middle",
        dy: 18,
      });
    }
  });
  const w = weights(counts(SEED));
  return {
    id: "rw/sites",
    title: "Legal sites for a move",
    caption: `For the parent, the registry lists every legal site of each class in a stable order. The weights of the classes with a site are scaled to sum to 1, with replace and insert ${show(w.replace)}, delete and binding ${show(w.delete)}, shape ${show(w.shape)}. Crossover is off in 0.0.14. The sites are counted on Monad’s definitions as this window draws them, with a norm as one node; AMLO’s builder can split nodes differently, and the counts would change with it.`,
    sketch: s,
    ...staged,
  };
}

/** The registry's ledger: each class's declared and renormalized weight, where and how many. */
export function sitesLedger(s: Sketch, at: { rx: number; ry: number }) {
  const all = sites(SEED);
  const c = counts(SEED);
  const w = weights(c);
  const stacked = all.binding.filter((x) => x.at.includes("/")).length;
  const [lo, hi] = SEED.ranges.F!;
  const where: Record<Class, string> = {
    replace: "silu in swiglu",
    insert: `a real-valued output: ${perDef(all.insert)}`,
    delete: `input and output types equal: ${perDef(all.delete)}`,
    binding: `the Repeat’s ${stacked} stacked slots; #E, read twice`,
    shape: `F, declared ${fmt(lo)} to ${fmt(hi)}; f in block and swiglu`,
  };
  ledger(s, "ledger-sites", at, {
    title: "The parent’s legal moves",
    rows: CLASSES.map((k) => ({
      name: k,
      note: `${show(WEIGHTS[k])}, renormalized ${show(w[k])}; ${where[k]}`,
      value: `${c[k]} ${c[k] === 1 ? "site" : "sites"}`,
    })),
    foot: ["Crossover is off in 0.0.14. A norm counts as one node."],
  });
}

/** Where a move's definition is drawn: under the strip's left end, its stream at this height. */
const DEF = { x: -0.8, block: -12.6, swiglu: -13.2 } as const;

/** The face that bounds what was drawn since `first`, for a drawing that opens. */
function bounds(s: Sketch, first: number): V3[] {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const p of s.prims.slice(first))
    for (const q of p.kind === "stroke" || p.kind === "face" ? p.pts : [p.at]) {
      x0 = Math.min(x0, q[0]);
      x1 = Math.max(x1, q[0]);
      y0 = Math.min(y0, q[1]);
      y1 = Math.max(y1, q[1]);
    }
  return [
    [x0, y1, 0],
    [x1, y1, 0],
    [x1, y0, 0],
    [x0, y0, 0],
  ];
}

/**
 * A move's view: the strip on top, the definition it edits below (which opens into its full
 * view, as edited), and admission beside it. The film sets the move's words and ledger under
 * the drawing.
 */
function moveView(o: {
  id: string;
  title: string;
  after: boolean;
  strip: StripState;
  detail: (s: Sketch, x: number, y: number) => number;
  def: "block" | "swiglu";
  p: Proposal;
  caption: string;
}): View {
  const s = new Sketch();
  const st = strip(s, 0, 0, o.strip);
  ends(s, 0, st.end, 0);
  const first = s.prims.length;
  o.detail(s, DEF.x, DEF[o.def]);
  const child = o.after ? `rw/open/${o.p.id}` : o.def;
  s.open(o.def, o.after ? `${o.def}, as rewritten` : o.def, child, bounds(s, first));
  admissionLane(s);
  return {
    id: o.after ? o.id : `${o.id}/before`,
    title: o.title,
    caption: o.caption,
    sketch: s,
    ...staged,
    ...(o.after ? { before: `${o.id}/before` } : {}),
  };
}

export const paramNote = (p: Pick<Proposal, "child">) => {
  const before = parameters(SEED);
  const after = parameters(p.child);
  return after === before
    ? `Parameters: ${fmt(after)}, unchanged.`
    : `Parameters: ${fmt(before)} before, ${fmt(after)} after.`;
};

function replaceView(after: boolean): View {
  const p = move("replace");
  return moveView({
    id: "rw/replace",
    title: "The replace move",
    after,
    p,
    strip: monadStrip({ ...(after ? { reprint: true, print: () => REPRINTED } : {}) }),
    def: "swiglu",
    detail: (s, x, y) =>
      swiglu(s, x, y, after ? { act: "gelu_tanh", actInk: NEW } : { act: "silu" }),
    caption: `SetOperator changes silu to gelu_tanh in swiglu. The reach is all ${p.n} layers, so the body itself changes, and every layer, folded or open, prints the change. No id is made or dropped. The recorded probability is the product of five exact factors, ${show(p.probability)}.`,
  });
}

function insertView(after: boolean): View {
  const p = move("insert");
  const [i] = PICKS.insert;
  return moveView({
    id: "rw/insert",
    title: "The insert move",
    after,
    p,
    strip: monadStrip({
      open: new Set(after ? [...always, i!] : always),
      own: new Set(after ? [i!] : []),
      note: "a copy",
      print: (j) => (after && j === i ? INSERTED : BASE),
    }),
    def: "block",
    detail: (s, x, y) =>
      block(s, x, y, [{ box: "attn", added: after ? true : "ghost" }, { box: "swiglu" }]),
    caption: `InsertNode puts the library’s rms_norm on attention’s output at layer ${i} only. That iteration of the Repeat gets an override, a copy of the block with the new node, drawn in its own ink. The new node’s id is made from the parent’s digest, the command’s digest and an ordinal; every other id is kept. Probability ${show(p.probability)}.`,
  });
}

function deleteView(after: boolean): View {
  const p = move("delete");
  const picks: readonly number[] = PICKS.delete;
  return moveView({
    id: "rw/delete",
    title: "The delete move",
    after,
    p,
    strip: monadStrip({
      open: new Set(after ? [...always, ...picks] : always),
      own: new Set(after ? picks : []),
      note: "a copy",
      print: (j) => (after && picks.includes(j) ? DELETED : BASE),
    }),
    def: "block",
    detail: (s, x, y) =>
      block(s, x, y, [{ box: "attn", ...(after ? { norm: "ghost" } : {}) }, { box: "swiglu" }]),
    caption: `DeleteNode removes the norm before attention at layers ${picks.join(", ")}, and attention reads the stream directly. The reach is some. Here k = ${picks.length} was drawn from 2 to ${p.n - 1}, then one of the ${fmt(Number(choose(p.n, picks.length)))} sets of ${picks.length} layers. The ${picks.length} layers share one copy of the block. Probability ${show(p.probability)}.`,
  });
}

function bindingView(after: boolean): View {
  const p = move("binding");
  return moveView({
    id: "rw/binding",
    title: "The binding move",
    after,
    p,
    strip: monadStrip({ deck: after ? "shared" : "stacked" }),
    def: "swiglu",
    detail: (s, x, y) =>
      swiglu(s, x, y, {
        act: "silu",
        w2: after ? "#W2, shared" : "#W2, stacked",
        ...(after ? { w2Ink: NEW } : {}),
      }),
    caption: `SetBind changes the Repeat’s slot #W2 from stacked, one leaf per layer, to shared, one leaf for all ${MONAD.L} layers. The Repeat occurs once, so the reach is all. The family keeps its id; its ${MONAD.L} slices become one leaf. Probability ${show(p.probability)}.`,
  });
}

function shapeView(after: boolean): View {
  const p = move("shape");
  const picks: readonly number[] = PICKS.shape;
  const [lo, hi] = SEED.ranges.F!;
  return moveView({
    id: "rw/shape",
    title: "The shape move",
    after,
    p,
    strip: monadStrip({
      open: new Set(after ? [...always, ...picks] : always),
      own: new Set(after ? picks : []),
      note: "f = 1,024",
      print: (j) => (after && picks.includes(j) ? WIDENED : BASE),
    }),
    def: "swiglu",
    detail: (s, x, y) => {
      const end = swiglu(s, x, y, {
        act: "silu",
        ...(after ? { f: { name: "f", ink: NEW } } : {}),
      });
      if (after)
        s.keyed("f-note", () =>
          s.text(x + 10.2, y + 3.3, `f = 1,024 at layers ${picks.join(" and ")}`, {
            size: 11,
            italic: true,
            ink: NEW,
            dy: -6,
          }),
        );
      return end;
    },
    caption: `SetAxisBinding sets f to 1,024 at layers ${picks.join(" and ")}. Those iterations get their own axis bindings; the body definition is not copied. Their slices of #W1, #W3 and #W2 keep their ids, change shape and are initialized again from their keys. F’s range, ${fmt(lo)} to ${fmt(hi)}, is declared for this illustration. Probability ${show(p.probability)}.`,
  });
}

function commuteView(after: boolean): View {
  const s = new Sketch();
  const { fromA, fromB } = commuting();
  const [i] = PICKS.insert;
  const small = { W: 2.1, top: 1.8 };
  const insertPrint = (j: number) => (j === i ? INSERTED : BASE);
  const ya = 1.4;
  const yb = -11.6;
  s.keyed("A", () => {
    const a = strip(
      s,
      0,
      ya,
      monadStrip({ ...small, id: "A", reprint: true, print: () => REPRINTED }),
    );
    s.text(
      0,
      ya + a.top,
      "A, ranked first as declared: the seed after SetOperator at every layer",
      {
        size: 11.5,
        italic: true,
        dy: -16,
      },
    );
  });
  s.keyed("B", () => {
    const b = strip(
      s,
      0,
      yb,
      monadStrip({
        ...small,
        id: "B",
        open: new Set([...always, i!]),
        own: new Set([i!]),
        print: insertPrint,
      }),
    );
    s.text(
      0,
      yb + b.top,
      `B, ranked second as declared: the seed after InsertNode at layer ${i}`,
      {
        size: 11.5,
        italic: true,
        dy: -16,
      },
    );
  });
  if (after)
    s.keyed("C", () => {
      const x0 = 31.4;
      const yc = -5.2;
      const c = strip(
        s,
        x0,
        yc,
        monadStrip({
          ...small,
          W: 2,
          id: "C",
          reprint: true,
          open: new Set([...always, i!]),
          own: new Set([i!]),
          print: (j) => (j === i ? BOTH : REPRINTED),
        }),
      );
      s.text(x0, yc + c.top, "A′ and B′: one architecture", { size: 12, weight: 600, dy: -16 });
      s.text(
        (x0 + c.end) / 2,
        yc - c.top - 2.4,
        "The minted ids differ; identity renames ids, so the two are equal.",
        { size: 11.5, italic: true, ink: "faint", anchor: "middle", dy: 44 },
      );
      for (const [y, p, side] of [
        [ya, fromA, 1],
        [yb, fromB, -1],
      ] as const) {
        s.wire(
          [
            [23.6, y],
            [x0 - 1.4, yc + side * 0.7],
          ],
          "faint",
          { w: 1.2 },
        );
        const what = p === fromA ? `InsertNode at layer ${i}` : "SetOperator at every layer";
        s.text(23.6, y, `${p.parent.name}′: ${what}`, {
          size: 11.5,
          dx: 10,
          dy: side > 0 ? -24 : 34,
        });
        s.text(23.6, y, `probability ${show(p.probability)}`, {
          size: 11.5,
          ink: "faint",
          dx: 10,
          dy: side > 0 ? -9 : 49,
        });
      }
    });
  return {
    id: after ? "rw/commute" : "rw/commute/before",
    title: "Two paths to one architecture",
    caption: `A and B are children of the seed that joined the population. Suppose that a generation later they rank first and second; these ranks are declared for the illustration. From A, InsertNode at layer ${i}, with probability ${show(fromA.probability)}. From B, SetOperator at every layer, with probability ${show(fromB.probability)}. The two children carry different minted ids, but architecture identity renames ids, so they are one architecture.`,
    sketch: s,
    ...staged,
    ...(after ? { before: "rw/commute/before" } : {}),
  };
}

/** Admission, drawn large: two proposals on two lanes through the seven gates. */
export const GATE_ROW = {
  y: -6.6,
  apart: 0.9,
  x: (k: number) => 3 + k * 7.1,
  get last() {
    return this.x(GATES.length - 1);
  },
  get out() {
    return this.last + 3.4;
  },
};

/** Admission: both proposals waiting; A′ through and admitted; then B′ refused at identity. */
function admissionView(state: "before" | "a" | "after"): View {
  const s = new Sketch();
  const { y, apart, last, out } = GATE_ROW;
  for (const [dy, name, ink, x1] of [
    [apart, "A′", "ink", out],
    [-apart, "B′", NEW, last],
  ] as const) {
    s.wire(
      [
        [-2, y + dy],
        [x1, y + dy],
      ],
      ink,
    );
    s.text(-2, y + dy, name, { size: 12, italic: true, ink, anchor: "end", dx: -8, dy: 4 });
  }
  for (const [k, g] of GATES.entries()) {
    const x = GATE_ROW.x(k);
    s.keyed(`gate${k}`, () => s.slab(x, y, 0.7, 4.6, {}));
    s.text(x, y + 2.3, g.name, { size: 12.5, weight: 600, anchor: "middle", dy: -24 });
    g.does.forEach((line, j) =>
      s.text(x, y - 2.3, line, {
        size: 11,
        italic: true,
        ink: "faint",
        anchor: "middle",
        dy: 20 + 14 * j,
      }),
    );
  }
  // The proposals, as two marks that the rewrite carries through the gates.
  s.keyed("a", () =>
    s.add({
      kind: "dot",
      at: [state === "before" ? -1.2 : out, y + apart, 0],
      r: 4.2,
      ink: "ink",
      layer: 0,
    }),
  );
  s.keyed("b", () =>
    s.add({
      kind: "dot",
      at: [state === "after" ? last - 1.1 : -1.2, y - apart, 0],
      r: 4.2,
      ink: NEW,
      layer: 0,
    }),
  );
  if (state !== "before")
    s.keyed("admitted", () =>
      s.text(out, y + apart, "Admitted", { size: 12.5, weight: 600, dx: 12, dy: 4 }),
    );
  if (state === "after")
    s.keyed("refused", () => {
      s.text(last, y - 2.3, "Refused: the same architecture as A′.", {
        size: 11.5,
        ink: NEW,
        anchor: "middle",
        dy: 66,
      });
      s.text(last, y - 2.3, "It resolves as duplicate.", {
        size: 11.5,
        ink: NEW,
        anchor: "middle",
        dy: 81,
      });
    });
  const id = { before: "rw/admission/before", a: "rw/admission/a", after: "rw/admission" }[state];
  return {
    id,
    title: "Admitting a proposal",
    caption: `Each proposal passes the gates in order, cheapest first, and gets exactly one verdict, Admitted or Refused with a reason. A′ comes first and is admitted. B′ comes later in the same generation. It passes the patch check, types, the ceilings and cost, and then the identity gate refuses it, because its architecture equals A′’s. The proposal resolves as duplicate.`,
    sketch: s,
    ...staged,
    ...(state === "after" ? { before: "rw/admission/before" } : {}),
  };
}

/** Two proposals in one generation, with their exact probabilities. */
export function admissionLedger(s: Sketch, at: { rx: number; ry: number }) {
  const { fromA, fromB } = commuting();
  ledger(s, "ledger-admission", at, {
    title: "Two proposals in one generation",
    rows: [
      {
        name: "A′",
        note: `from A: InsertNode at layer ${PICKS.insert[0]}`,
        value: show(fromA.probability),
      },
      { name: "B′", note: "from B: SetOperator at every layer", value: show(fromB.probability) },
    ],
    foot: ["Each gets exactly one verdict, Admitted or Refused with its reason."],
  });
}

function deepseekView(after: boolean): View {
  const s = new Sketch();
  const p = deepseek();
  const [layer] = p.selected;
  const { layers } = schedule();
  const L = layers.length;
  const W = 2.4;
  const C = 0.36;
  const top = 2.6;
  const depth = 1;
  const open = (i: number) => !KIND[layers[i]!].folds || (after && i === layer);
  const own = (i: number) => after && i === layer;
  const xs = [0];
  for (let i = 0; i < L; i++) xs.push(xs[i]! + (open(i) ? W : C));
  const zAt = (e: number) => (e % 2 === 0 ? 0 : -depth);
  const at = (i: number, u: number, yy: number): V3 => [
    xs[i]! + u * (xs[i + 1]! - xs[i]!),
    yy,
    zAt(i) + u * (zAt(i + 1) - zAt(i)),
  ];
  for (let i = 0; i < L; i++) {
    const k = KIND[layers[i]!];
    s.group(`panel${i}`, () => {
      s.keyed(`panel${i}`, () => {
        s.add({
          kind: "face",
          pts: [at(i, 0, top), at(i, 1, top), at(i, 1, -top), at(i, 0, -top)],
          layer: 0,
          fill: "paper",
          edge: own(i) ? NEW : open(i) ? k.ink : "faint",
          w: open(i) ? 1.3 : 0.5,
          under: true,
          box: `panel${i}`,
        });
        for (const yy of [top, -top])
          s.add({
            kind: "stroke",
            pts: [at(i, 0, yy), at(i, 1, yy)],
            w: open(i) ? 1.2 : 0.8,
            ink: own(i) ? NEW : "ink",
            layer: 0,
            halo: false,
          });
      });
      if (!open(i)) return;
      const [cx, , cz] = at(i, 0.5, top);
      s.within(`panel${i}`, () =>
        s.keyed(`name${i}`, () =>
          s.text(cx, top - 1.3, k.name, {
            z: cz,
            size: 11,
            italic: true,
            ink: k.ink === "faint" ? "ink" : k.ink,
            anchor: "middle",
          }),
        ),
      );
    });
    if (!open(i)) continue;
    const [bx, , bz] = at(i, 0.5, -top);
    s.keyed(`num${i}`, () => {
      s.text(bx, -top, `${i}`, {
        z: bz,
        size: 11,
        ink: own(i) ? NEW : "faint",
        anchor: "middle",
        dy: 20,
      });
      if (own(i))
        s.text(bx, -top, "a copy", {
          z: bz,
          size: 11,
          italic: true,
          ink: NEW,
          anchor: "middle",
          dy: 35,
        });
    });
  }
  // The folded runs, numbered.
  for (let a = 0; a < L; a++) {
    if (open(a) || (a > 0 && !open(a - 1))) continue;
    let b = a;
    while (b + 1 < L && !open(b + 1)) b++;
    s.keyed(`run${a}`, () =>
      s.text((xs[a]! + xs[b + 1]!) / 2, -top, a === b ? `${a}` : `${a}–${b}`, {
        size: 11,
        ink: "faint",
        anchor: "middle",
        dy: 20,
      }),
    );
  }
  // The decoder's nested Repeats, bracketed under the strip.
  const under = (a: number, b: number, yy: number, text: string, ink: Ink = "faint") =>
    bracket(s, `bracket${a}-${b}`, xs[a]!, xs[b + 1]!, yy, -depth / 2, text, ink);
  for (let g = 0; g < DEC.groups; g++) {
    const mine = after && g === p.group;
    under(
      DEC.first + g * DEC.size,
      DEC.first + (g + 1) * DEC.size - 1,
      -top - 2.6,
      "dec_group",
      mine ? NEW : "faint",
    );
    if (mine)
      s.text(
        (xs[DEC.first + g * DEC.size]! + xs[DEC.first + (g + 1) * DEC.size]!) / 2,
        -top - 2.6,
        "a copy",
        {
          z: -depth / 2,
          size: 11,
          italic: true,
          ink: NEW,
          anchor: "middle",
          dy: 30,
        },
      );
  }
  under(
    DEC.first,
    DEC.first + DEC.groups * DEC.size - 1,
    -top - 4.8,
    `Dec = Repeat(dec_group, ${DEC.groups})`,
  );
  under(2, 19, -top - 4.8, "encoder, ratio 2");
  // The routed experts at that layer: one sheet each, the activation on every sheet.
  const ex = xs[L]! + 5.2;
  const ey = -9;
  for (let e = DSV41.E - 1; e >= 0; e--) {
    const z = -e * 1.1;
    s.keyed(`expert${e}`, () => {
      s.sheet(ex - 3.4, ey - 2.6, ex + 10.6, ey + 2.6, { z, layer: -e, box: `expert${e}` });
      if (e === 0)
        s.within(`expert${e}`, () =>
          s.text(ex + 10.6, ey + 2.6, "expert 1", {
            z,
            size: 11,
            ink: "faint",
            italic: true,
            anchor: "end",
            dx: -5,
            dy: 14,
          }),
        );
      s.within(`expert${e}`, () =>
        s.chain(
          ey,
          ex - 2.6,
          [T, D],
          [
            { kind: "slab", w: 2.6, h: 2.6, chip: true, label: "W", sub: "1", size: 14 },
            { kind: "lanes", lanes: [T, F] },
            { kind: "op", name: "min 10", w: 2.4 },
            {
              key: "act",
              kind: "op",
              name: after ? "gelu_tanh" : "silu",
              ...(after ? { ink: NEW } : {}),
            },
            { kind: "wire", w: 1.4 },
          ],
          { z, layer: -e },
        ),
      );
    });
  }
  s.text(
    ex - 3.4,
    ey + 2.6,
    `the routed experts of layer ${layer}: e = ${DSV41.E}, one sheet each`,
    {
      z: -(DSV41.E - 1) * 1.1,
      size: 11.5,
      italic: true,
      dy: -10,
    },
  );
  return {
    id: after ? "rw/deepseek" : "rw/deepseek/before",
    title: "Moves on the toy DeepSeek",
    model: "dsv41",
    caption: `The same rules on the toy DeepSeek. SetOperator changes silu to gelu_tanh in the routed experts at layer ${layer} only. Layer ${layer} is iteration ${p.inner} of the Reuse run inside iteration ${p.group} of the decoder’s Repeat, so the copy reaches two Repeats deep, and ${p.copies.slice(0, -1).join(", ")} and ${p.copies.at(-1)} are copied for it. No control declares a range here, so shape has no site, and the weights renormalize over four classes. Probability ${show(p.probability)}.`,
    sketch: s,
    ...staged,
    ...(after ? { before: "rw/deepseek/before" } : {}),
  };
}

const builders: Record<string, () => View> = {
  "rw/population": () => population("parent"),
  "rw/population/drawn": () => population("drawn"),
  "rw/population/before": () => population("row"),
  "rw/sites": sitesView,
  "rw/replace": () => replaceView(true),
  "rw/replace/before": () => replaceView(false),
  "rw/insert": () => insertView(true),
  "rw/insert/before": () => insertView(false),
  "rw/delete": () => deleteView(true),
  "rw/delete/before": () => deleteView(false),
  "rw/binding": () => bindingView(true),
  "rw/binding/before": () => bindingView(false),
  "rw/shape": () => shapeView(true),
  "rw/shape/before": () => shapeView(false),
  "rw/commute": () => commuteView(true),
  "rw/commute/before": () => commuteView(false),
  "rw/admission": () => admissionView("after"),
  "rw/admission/a": () => admissionView("a"),
  "rw/admission/before": () => admissionView("before"),
  "rw/deepseek": () => deepseekView(true),
  "rw/deepseek/before": () => deepseekView(false),
};

/** Each move's definition as rewritten, which its inset opens into. */
function opened(id: string): View | undefined {
  const L = MONAD.L;
  const [i] = PICKS.insert;
  const del = list(PICKS.delete);
  const shp = list(PICKS.shape);
  switch (id) {
    case "replace":
      return swigluView({
        id: "rw/open/replace",
        title: "SwiGLU, as rewritten",
        act: "gelu_tanh",
        caption: `swiglu after SetOperator, with gelu_tanh in place of silu. The reach is all ${L} layers, so this is the body every layer runs. No id is made or dropped.`,
      });
    case "insert":
      return blockView({
        id: "rw/open/insert",
        title: `block.1, the copy for layer ${i}`,
        added: true,
        caption: `block.1, the override InsertNode made for layer ${i}, which puts the library’s rms_norm on attention’s output. The other ${L - 1} layers still run block. The new node’s id is minted; every other id is kept.`,
      });
    case "delete":
      return blockView({
        id: "rw/open/delete",
        title: `block.1, the copy for layers ${del}`,
        unnormed: true,
        caption: `block.1, the one copy DeleteNode made for layers ${del}. There attention reads the stream directly, with no norm before it. The other ${L - PICKS.delete.length} layers still run block.`,
      });
    case "binding":
      return swigluView({
        id: "rw/open/binding",
        title: "SwiGLU, as rewritten",
        w2: `#W2: one leaf, shared by all ${L} layers`,
        caption: `swiglu after SetBind. Its slot #W2 is one leaf shared by all ${L} layers, not ${L} stacked leaves. The family keeps its id.`,
      });
    case "shape":
      return swigluView({
        id: "rw/open/shape",
        title: `SwiGLU at layers ${shp}`,
        f: `f = 1,024 here; ${fmt(MONAD.F)} in the other layers`,
        caption: `swiglu as layers ${shp} run it after SetAxisBinding, with f = 1,024 there. The definition is not copied; those iterations bind f to 1,024, and their slices of #W1, #W3 and #W2 change shape.`,
      });
  }
  return undefined;
}

/** "7, 30 and 51". */
const list = (xs: readonly number[]) =>
  xs.length < 2 ? `${xs[0]}` : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`;

export function view(id: string): View | undefined {
  if (id.startsWith("rw/open/")) return opened(id.slice("rw/open/".length));
  return builders[id]?.();
}
