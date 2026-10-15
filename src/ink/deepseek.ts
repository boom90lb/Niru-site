/**
 * A toy DeepSeek-V4.1-Flash, drawn as AMLO's 0.0.14 design writes it, from
 * DeepSeek's reference implementation at revision 2cba9e42
 * (`inference/model.py`).
 *
 *   carry                 X:[t,s,d] (s = 4 residual streams), pre-mix:[t,s]
 *   layer (every kind)    for attention, then the MoE: (pre, post, comb) = mix(X);
 *                         X ← post ⊗ F(rmsnorm(pre′·X)) + comb·X, where pre′ is the
 *                         pre-mix the previous sublayer computed
 *   attention by kind     SWA: a window band and a sink; CSA adds selected compressed
 *                         entries. Full compresses, keys an index and selects; Reuse
 *                         reads both; Reindex re-selects from the decoder's keys
 *   moe                   sqrt-softplus scores; top-k on scores + bias, weighted by the
 *                         unbiased scores; eight routed experts and one shared
 *   engram                n-gram hash rows, gated into each stream before layers 1 and 14
 *
 * Main KV, index keys, top-k indices and the candidate pool are explicit wires
 * passed as Repeat closures, so the drawing can show them crossing layers.
 */
import { DSV41, KIND, type Kind, parameters, schedule } from "./dsv41.ts";
import type { Ink, V3 } from "./engine.ts";
import { PARTS } from "./deepseek-parts.ts";
import { LANE, type Lane, Sketch } from "./kit.ts";
import { trainingStep } from "./step.ts";
import { fmt, type View } from "./view.ts";

const T: Lane = { name: "t", ink: "ink" };
const Sl: Lane = { name: "s", ink: "verdigris" };
const Dl: Lane = { name: "d", ink: "ochre" };
const Vl: Lane = { name: "V", ink: "indigo" };

type P3 = [number, number, number];

function model(): View {
  const s = new Sketch();
  const { layers, engram } = schedule();
  const L = layers.length;
  const OPEN = 3.7;
  const FOLD = 0.56;
  const depth = 1.4;
  const top = 5.4;
  const X0 = 6;
  const xs: number[] = [X0];
  for (let i = 0; i < L; i++) xs.push(xs[i]! + (KIND[layers[i]!].folds ? FOLD : OPEN));
  const zAt = (edge: number) => (edge % 2 === 0 ? 0 : -depth);
  const at = (i: number, u: number, y: number): P3 => [
    xs[i]! + u * (xs[i + 1]! - xs[i]!),
    y,
    zAt(i) + u * (zAt(i + 1) - zAt(i)),
  ];
  const atX = (x: number, y: number): P3 => {
    let i = 0;
    while (i < L - 1 && xs[i + 1]! < x) i++;
    return at(i, (x - xs[i]!) / (xs[i + 1]! - xs[i]!), y);
  };

  // Units: an open panel, or a folded run of Reuse layers (one Repeat).
  const units: Array<{ a: number; b: number; kind: Kind }> = [];
  for (let i = 0; i < L; i++) {
    const k = layers[i]!;
    const last = units[units.length - 1];
    if (KIND[k].folds && last && last.kind === k && last.b === i - 1) last.b = i;
    else units.push({ a: i, b: i, kind: k });
  }
  const unit = (i: number) => units.find((u) => u.a <= i && i <= u.b)!;
  const ux = (u: { a: number; b: number }, f: number) => xs[u.a]! + f * (xs[u.b + 1]! - xs[u.a]!);

  // Panels and folded runs.
  for (const u of units) {
    if (KIND[u.kind].folds) {
      for (let e = u.a; e <= u.b + 1; e++)
        s.add({
          kind: "stroke",
          pts: [
            [xs[e]!, top, zAt(e)],
            [xs[e]!, -top, zAt(e)],
          ],
          w: 0.5,
          ink: "faint",
          layer: 0,
          halo: false,
        });
      for (const y of [top, -top])
        s.add({
          kind: "stroke",
          pts: xs.slice(u.a, u.b + 2).map((x, j) => [x, y, zAt(u.a + j)] as P3),
          w: 0.8,
          ink: "ink",
          layer: 0,
          halo: false,
        });
      const face = [
        [xs[u.a]!, top, zAt(u.a)],
        [xs[u.b + 1]!, top, zAt(u.b + 1)],
        [xs[u.b + 1]!, -top, zAt(u.b + 1)],
        [xs[u.a]!, -top, zAt(u.a)],
      ] as V3[];
      s.open(
        `layers-${u.a}-${u.b}`,
        u.kind === "reuse2" ? "Reuse layers, ratio 2" : "Reuse layers, ratio 1",
        `ds/layer/${u.kind}`,
        face,
      );
      continue;
    }
    const i = u.a;
    const face = [at(i, 0, top), at(i, 1, top), at(i, 1, -top), at(i, 0, -top)];
    s.add({
      kind: "face",
      pts: face,
      layer: 0,
      fill: "paper",
      edge: KIND[u.kind].ink,
      w: 1.3,
      under: true,
    });
    s.open(`layer-${i}`, openLabel[u.kind], `ds/layer/${u.kind}`, face);
    const [cx, , cz] = at(i, engram.includes(i) ? 0.58 : 0.5, top);
    s.text(cx, top - 1.4, KIND[u.kind].name, {
      z: cz,
      size: 11.5,
      italic: true,
      ink: u.kind === "swa" ? "ink" : KIND[u.kind].ink,
      anchor: "middle",
    });
  }

  // The carry through every layer: the residual streams X [t, s, d] and the pre-mix [t, s].
  const lanes: Array<[number, Ink]> = [
    [LANE, "ink"],
    [0, "verdigris"],
    [-LANE, "ochre"],
    [-2.2, "ink"],
    [-3.3, "verdigris"],
  ];
  for (const [y, ink] of lanes)
    for (const u of units) {
      const folds = KIND[u.kind].folds;
      s.add({
        kind: "stroke",
        pts: xs.slice(u.a, u.b + 2).map((x, j) => [x, y, zAt(u.a + j)] as P3),
        w: folds ? 0.6 : 1.5,
        ink: folds ? "faint" : ink,
        layer: 0,
        halo: false,
      });
    }

  // Wires between layers, above the strip: each passes a value from the layer that makes it
  // to the layers that read it. A dot is a copy; a folded run reads its closure once.
  const rows = { kv: top + 1.1, ik: top + 2.0, topk: top + 2.9, cand: top + 3.8 };
  const rise = { cand: 0.2, topk: 0.38, ik: 0.56, kv: 0.76 };
  const zBus = -depth / 2;
  const bus = (
    src: number,
    to: number[],
    key: keyof typeof rows,
    ink: Ink,
    o: { dash?: number[]; ints?: boolean } = {},
  ) => {
    const y = rows[key];
    const f = rise[key];
    const x0 = ux(unit(src), f);
    const targets = [...new Set(to.map(unit))];
    const x1 = ux(targets[targets.length - 1]!, f);
    const dash = o.dash ? { dash: o.dash } : {};
    s.add({ kind: "stroke", pts: [atX(x0, top), [x0, y, zBus]], w: 1.3, ink, layer: 0, ...dash });
    s.add({
      kind: "stroke",
      pts: [
        [x0, y, zBus],
        [x1, y, zBus],
      ],
      w: 1.3,
      ink,
      layer: 0,
      ...dash,
    });
    targets.forEach((t, j) => {
      const x = ux(t, f);
      s.add({ kind: "stroke", pts: [[x, y, zBus], atX(x, top)], w: 1.3, ink, layer: 0, ...dash });
      if (j < targets.length - 1) s.add({ kind: "dot", at: [x, y, zBus], r: 2.6, ink, layer: 0 });
    });
    if (o.ints) s.arrows(x0, x1, y, ink, { z: zBus });
  };
  const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, j) => a + j);
  for (const src of [2, 8, 14]) {
    bus(src, range(src + 1, src + 5), "kv", "cinnabar");
    bus(src, range(src + 1, src + 5), "topk", "ink", { ints: true });
  }
  bus(20, range(21, 39), "kv", "cinnabar");
  bus(20, [24, 28, 32, 36], "ik", "indigo");
  bus(20, [24, 28, 32, 36], "cand", "faint", { dash: [4, 3] });
  for (const src of [20, 24, 28, 32, 36])
    bus(src, range(src + 1, src + 3), "topk", "ink", { ints: true });

  // Token ids, an integer datatype, select rows of E and feed the two Engram lookups.
  const yE = top + 9;
  s.wire(
    [
      [-9, 0],
      [-1.4, 0],
    ],
    "ink",
  );
  s.arrows(-9, -1.4, 0, "ink");
  s.text(-9, 0, "inputs", { italic: true, anchor: "end", dx: -6, dy: 4 });
  s.copy(-6, [0], ["ink"]);
  const xLast = xs[engram[engram.length - 1]!]!;
  s.wire(
    [
      [-6, 0],
      [-3, yE],
      [xLast, yE],
    ],
    "ink",
  );
  s.arrows(-2, xLast - 0.6, yE, "ink");
  for (const e of engram) {
    const x = xs[e]!;
    const z = zAt(e);
    if (e !== engram[engram.length - 1]) s.copy(x, [yE], ["ink"]);
    s.add({
      kind: "stroke",
      pts: [
        [x, yE, 0],
        [x, yE - 1.5, z],
      ],
      w: 1.5,
      ink: "ink",
      layer: 0,
    });
    const face = s.slab(x, yE - 3, 6, 3, { z, chip: true, label: "Engram", size: 12 });
    s.open(`engram-${e}`, "Engram", "ds/engram", face);
    s.add({
      kind: "stroke",
      pts: [
        [x, yE - 4.5, z],
        [x, 0.9, z],
      ],
      w: 1.5,
      ink: "ochre",
      layer: 0,
    });
    s.plus(x, 0, { z });
  }

  // E, then a copy into four streams; the pre-mix starts one-hot on stream 0.
  s.slab(0, 0, 2.8, 3.2, { chip: true, label: "E", size: 16 });
  s.port(0, -1.6, "#E");
  s.bundle(
    [
      [1.4, 0],
      [3.4, 0],
    ],
    [T, Dl],
    {},
  );
  for (const [y0, y1, ink] of [
    [LANE / 2, LANE, "ink"],
    [-LANE / 2, -LANE, "ochre"],
  ] as const)
    s.wire(
      [
        [3.4, y0],
        [5.2, y1],
        [X0, y1],
      ],
      ink,
    );
  s.wire(
    [
      [3.4, 0],
      [X0, 0],
    ],
    "verdigris",
  );
  s.add({ kind: "dot", at: [3.4, 0, 0], r: 3.4, ink: "verdigris", layer: 0 });
  ["t", "s", "d"].forEach((name, j) =>
    s.text(4.1, LANE, name, {
      italic: true,
      size: 12,
      ink: (["ink", "verdigris", "ochre"] as const)[j]!,
      dx: j * 13,
      dy: -7,
    }),
  );
  s.op(3.4, -2.75, "e", { w: 2, sub: "0" });
  for (const [y, ink] of [
    [-2.2, "ink"],
    [-3.3, "verdigris"],
  ] as const)
    s.wire(
      [
        [4.3, y],
        [X0, y],
      ],
      ink,
    );
  s.text(3.4, -2.75, "pre-mix", {
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "middle",
    dy: 34,
  });

  // Layer numbers, counted from 0 as the reference counts them, and the parts of the stack.
  for (const u of units) {
    const [cx, , cz] = at(u.a, 0.5, -top);
    const x = (xs[u.a]! + xs[u.b + 1]!) / 2;
    const z = KIND[u.kind].folds ? (zAt(u.a) + zAt(u.b + 1)) / 2 : cz;
    s.text(KIND[u.kind].folds ? x : cx, -top, u.a === u.b ? `${u.a}` : `${u.a}–${u.b}`, {
      z,
      size: 11,
      ink: "faint",
      anchor: "middle",
      dy: 16,
    });
  }
  const bracket = (a: number, b: number, text: string) => {
    const y = -top - 2.6;
    s.add({
      kind: "stroke",
      pts: [
        [xs[a]! + 0.15, y + 0.35, -depth / 2],
        [xs[a]! + 0.15, y, -depth / 2],
        [xs[b + 1]! - 0.15, y, -depth / 2],
        [xs[b + 1]! - 0.15, y + 0.35, -depth / 2],
      ],
      w: 0.9,
      ink: "faint",
      layer: 0,
      halo: false,
    });
    s.text((xs[a]! + xs[b + 1]!) / 2, y, text, {
      z: -depth / 2,
      size: 11.5,
      italic: true,
      anchor: "middle",
      dy: 19,
    });
  };
  bracket(2, 19, "encoder, ratio 2");
  bracket(20, 39, "decoder, ratio 1");

  // After the last layer: the last pre-mix collapses the streams, then norm, head and loss.
  const xe = xs[L]!;
  for (const [y, ink] of lanes)
    s.wire(
      [
        [xe, y],
        [xe + 1.2, y],
      ],
      ink,
    );
  s.add({
    kind: "face",
    pts: [
      [xe + 1.2, 1.9, 0],
      [xe + 3.4, 1.9, 0],
      [xe + 3.4, -4.1, 0],
      [xe + 1.2, -4.1, 0],
    ],
    layer: 0,
    fill: "paper",
    edge: "ink",
    w: 1.2,
  });
  s.text(xe + 2.3, -1.1, "Σ", { size: 13, italic: true, anchor: "middle", dy: 4, sub: "s" });
  s.text(xe + 2.3, -4.1, "pre · X", { size: 11, italic: true, anchor: "middle", dy: 15 });
  s.bundle(
    [
      [xe + 3.4, 0],
      [xe + 7, 0],
    ],
    [T, Dl],
  );
  s.norm(xe + 5.2, 0, { r: 0.85 });
  const hx = xe + 8.9;
  s.slab(hx, 0, 3.8, 3.4, { chip: true, label: "head", size: 14 });
  s.port(hx, -1.7, "#H");
  const ce = hx + 10.6;
  s.bundle(
    [
      [hx + 1.9, 0],
      [ce - 4.3, 0],
    ],
    [T, Vl],
    { label: "over", labelDx: 14 },
  );
  s.slab(ce, 0, 8.6, 3.4, { label: "cross-entropy", size: 12 });
  for (const [dx, name] of [
    [-1.2, "labels"],
    [1.2, "loss mask"],
  ] as const) {
    s.add({
      kind: "stroke",
      pts: [
        [ce + dx, -5.6, 0],
        [ce + dx, -1.7, 0],
      ],
      w: 1.5,
      ink: "ink",
      layer: 0,
    });
    s.text(ce + dx, -5.6, name, {
      italic: true,
      size: 11,
      anchor: dx < 0 ? "end" : "start",
      dy: 14,
    });
  }
  s.wire(
    [
      [ce + 4.3, 0],
      [ce + 7.4, 0],
    ],
    "ink",
  );
  s.text(ce + 7.4, 0, "loss", { italic: true, dx: 6, dy: 4 });

  // A key to the wires between layers.
  const kx = xe + 2;
  const key: Array<[keyof typeof rows, Ink, string, number[] | undefined, boolean]> = [
    ["cand", "faint", "candidates [t, c′]: post-training only", [4, 3], false],
    ["topk", "ink", "top-k indices [t, K], integers", undefined, true],
    ["ik", "indigo", "index keys [c′, i]", undefined, false],
    ["kv", "cinnabar", "compressed keys and values [c, k]; c′ in the decoder", undefined, false],
  ];
  for (const [row, ink, text, dash, ints] of key) {
    const y = top + 5.6 - ["kv", "ik", "topk", "cand"].indexOf(row) * 1.5;
    s.add({
      kind: "stroke",
      pts: [
        [kx, y, zBus],
        [kx + 2.4, y, zBus],
      ],
      w: 1.3,
      ink,
      layer: 0,
      ...(dash ? { dash } : {}),
    });
    if (ints) s.arrows(kx, kx + 2.4, y, ink, { z: zBus });
    s.text(kx + 2.4, y, text, {
      z: zBus,
      size: 11,
      italic: true,
      ink: ink === "faint" ? "ink" : ink,
      dx: 6,
      dy: 4,
    });
  }

  return {
    id: "ds/model",
    title: "Toy DeepSeek with its loss",
    caption: `Forty layers of five kinds, numbered from 0 as in the reference. Full layers compress keys and values and choose the top-k entries that the Reuse layers folded after them read; the wires above the strip carry these between layers. Four residual streams run through every layer, and an Engram lookup of the token n-grams is added before layers 1 and 14. Width ${DSV41.D}, ${fmt(parameters())} parameters, counted from the definitions. Open any layer.`,
    sketch: s,
    yaw: -0.17,
    pitch: 0.4,
  };
}

interface Side {
  text: string;
  ink: Ink;
  dash?: number[];
  ints?: boolean;
}

const KV: Side = { text: "kv [c, k]", ink: "cinnabar" };
const KV1: Side = { text: "kv [c′, k]", ink: "cinnabar" };
const TOPK: Side = { text: "top-k [t, K]", ink: "ink", ints: true };
const IKEYS: Side = { text: "index keys [c′, i]", ink: "indigo" };
const CAND: Side = { text: "candidates [t, c′], post-training", ink: "faint", dash: [4, 3] };

/** What each kind of layer reads from earlier layers and passes on to later ones. */
const SIDE: Record<Kind, { ins: Side[]; outs: Side[] }> = {
  swa: { ins: [], outs: [] },
  full2: { ins: [], outs: [KV, TOPK] },
  reuse2: { ins: [KV, TOPK], outs: [] },
  full1: { ins: [], outs: [KV1, IKEYS, TOPK, CAND] },
  reindex: { ins: [KV1, IKEYS, CAND], outs: [TOPK] },
  reuse1: { ins: [KV1, TOPK], outs: [] },
};

const LAYER_TEXT: Record<Kind, string> = {
  swa: "An SWA layer (0 and 1) attends to a window of the last 64 positions and a learned sink.",
  full2:
    "A Full layer at ratio 2 (2, 8 and 14) pools every two positions into one compressed entry, keys an index on the entries and selects the top 32 for each query. It passes the entries and the selection to the five Reuse layers after it.",
  reuse2:
    "A Reuse layer at ratio 2 reads the compressed entries and the selection of the last Full layer instead of computing its own.",
  full1:
    "Layer 20, the Full layer at ratio 1, makes one entry per position from the last encoder layer’s output. Every decoder layer reads these entries; the Reindex layers also read its index keys and, in post-training, its candidate pool.",
  reindex:
    "A Reindex layer (24, 28, 32 and 36) scores layer 20’s index keys with its own queries and selects a new top 32 for the three Reuse layers after it.",
  reuse1:
    "A Reuse layer at ratio 1 reads layer 20’s entries and the selection of the Full or Reindex layer before it.",
};

const ATTN_NOTE: Record<Kind, string> = {
  swa: "window and sink",
  full2: "Full, ratio 2",
  reuse2: "Reuse, ratio 2",
  full1: "Full, ratio 1",
  reindex: "Reindex, ratio 1",
  reuse1: "Reuse, ratio 1",
};

const S2: Lane = { name: "s′", ink: "verdigris" };

function layer(kind: Kind): View {
  const s = new Sketch();
  const left = -3.4;
  const right = 45.4;
  const yS = 9.6;
  s.bundle(
    [
      [left, 0],
      [1, 0],
    ],
    [T, Sl, Dl],
    { label: "start" },
  );
  s.bundle(
    [
      [left, -3.3],
      [1, -3.3],
    ],
    [T, Sl],
    { label: "start" },
  );
  s.text(left, -3.3, "pre-mix", {
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -24,
    dy: 4,
  });
  const cell = (x0: number, name: string, child: string, ports: string, note: string) => {
    const pre = s.slab(x0 + 2.2, 0.9, 3.2, 11, { chip: true, label: "mhc", sub: "pre", size: 14 });
    s.open(`${child}-pre`, "a hyper-connection step", "ds/mhc", pre);
    s.port(x0 + 2.2, -4.6, "#fn #base #scale");
    const post = s.slab(x0 + 19, 2.3, 3.2, 8.2, { label: "mhc", sub: "post", size: 14 });
    s.open(`${child}-post`, "a hyper-connection step", "ds/mhc", post);
    s.bundle(
      [
        [x0 + 3.8, 0],
        [x0 + 17.4, 0],
      ],
      [T, Sl, Dl],
    );
    s.bundle(
      [
        [x0 + 3.8, 4.25],
        [x0 + 17.4, 4.25],
      ],
      [T, Sl],
      { gap: 0.42, w: 1.1 },
    );
    s.bundle(
      [
        [x0 + 3.8, 2.4],
        [x0 + 17.4, 2.4],
      ],
      [T, Sl, S2],
      { gap: 0.42, w: 1.1 },
    );
    s.text(x0 + 10.4, 4.46, "post", {
      size: 11,
      italic: true,
      ink: "verdigris",
      anchor: "middle",
      dy: -8,
    });
    s.text(x0 + 10.4, 2.82, "comb", {
      size: 11,
      italic: true,
      ink: "verdigris",
      anchor: "middle",
      dy: -5.5,
    });
    s.bundle(
      [
        [x0 + 3.8, 5.5],
        [x0 + 6, yS],
        [x0 + 8.6, yS],
      ],
      [T, Dl],
    );
    s.norm(x0 + 7.2, yS);
    s.port(x0 + 7.2, yS - 1, "#g");
    const f = s.slab(x0 + 11.6, yS, 6, 4.4, { chip: true, label: name, size: 15, note });
    s.open(child, name === "MoE" ? "the MoE" : "attention", child, f);
    s.port(x0 + 11.6, yS - 2.2, ports);
    s.bundle(
      [
        [x0 + 14.6, yS],
        [x0 + 17.4, 5.5],
      ],
      [T, Dl],
    );
  };
  cell(
    0,
    "attention",
    `ds/attn/${kind}`,
    SIDE[kind].outs.length || kind === "reindex" ? "#Wq #Wkv #Wo #sink …" : "#Wq #Wkv #Wo #sink",
    ATTN_NOTE[kind],
  );
  // The pre-mix this attention step computed is used by the MoE step: it passes under mhc post.
  s.bundle(
    [
      [3.8, -3.3],
      [22.6, -3.3],
    ],
    [T, Sl],
  );
  s.bundle(
    [
      [20.6, 0],
      [22.6, 0],
    ],
    [T, Sl, Dl],
  );
  cell(22, "MoE", "ds/moe", "#Wg #W1 #W2 #W3; state b", "8 routed, 1 shared");
  s.bundle(
    [
      [25.8, -3.3],
      [right, -3.3],
    ],
    [T, Sl],
    { label: "end" },
  );
  s.bundle(
    [
      [42.6, 0],
      [right, 0],
    ],
    [T, Sl, Dl],
    { label: "end" },
  );
  s.text(right, -3.3, "pre-mix", { size: 11, italic: true, ink: "faint", dx: 24, dy: 4 });

  // Values from earlier layers come in from the left; values for later layers leave to the right.
  const { ins, outs } = SIDE[kind];
  const rope = kind === "swa" ? "rope, base table" : "rope, CSA table with YaRN";
  const inputs: Side[] = [...ins, { text: rope, ink: "faint" }];
  const top = yS + 2.2;
  inputs.forEach((w, r) => {
    const y = top + 1.4 + r * 1.05;
    const x = 9.6 + r * 0.9;
    const dash = w.dash ? { dash: w.dash } : {};
    s.wire(
      [
        [left, y],
        [x, y],
      ],
      w.ink,
      dash,
    );
    s.wire(
      [
        [x, y],
        [x, top],
      ],
      w.ink,
      dash,
    );
    if (w.ints) s.arrows(left, x - 0.4, y, w.ink);
    s.text(left, y, w.text, { size: 11, italic: true, ink: w.ink, anchor: "end", dx: -6, dy: 4 });
  });
  outs.forEach((w, r) => {
    const y = top + 1.4 + r * 1.05;
    const x = 13.6 - r * 0.9;
    const dash = w.dash ? { dash: w.dash } : {};
    s.wire(
      [
        [x, top],
        [x, y],
      ],
      w.ink,
      dash,
    );
    s.wire(
      [
        [x, y],
        [right, y],
      ],
      w.ink,
      dash,
    );
    if (w.ints) s.arrows(x + 0.4, right, y, w.ink);
    s.text(right, y, w.text, { size: 11, italic: true, ink: w.ink, dx: 6, dy: 4 });
  });
  return {
    id: `ds/layer/${kind}`,
    title: `One layer (${ATTN_NOTE[kind]})`,
    caption: `${LAYER_TEXT[kind]} Each sublayer sits between two hyper-connection steps. mhc pre computes mixing coefficients from the four streams and collapses the streams with the pre-mix the previous sublayer computed, so each pre-mix is used one sublayer later. mhc post adds the output back into the four streams and mixes them. Open attention, the MoE or a step.`,
    sketch: s,
    yaw: -0.22,
    pitch: 0.34,
  };
}

const openLabel: Record<Kind, string> = {
  swa: "an SWA layer",
  full2: "a Full layer, ratio 2",
  reuse2: "Reuse layers, ratio 2",
  full1: "the Full layer, ratio 1",
  reindex: "a Reindex layer",
  reuse1: "Reuse layers, ratio 1",
};

function step(): View {
  return trainingStep({
    id: "ds/step",
    model: "ds/model",
    note: "toy DeepSeek-V4.1-Flash",
    theta: "θ: #E, #H, #Wq … #T",
    loss: "loss",
    update: "clip, then AdamW",
    state: { name: "gate bias b, state", rule: "b + 0.001 · sign(mean − c)" },
    caption:
      "In AMLO’s 0.0.14 design one training step is one term. The loss adds a balance term over the experts and the indexers’ objective to the cross-entropy; the gradient is derived from the model’s definitions; the update clips it and applies AdamW, with the Engram tables at five times the rate. The gate bias is state. No gradient reaches it, and a rule moves each expert’s bias by 0.001 toward even use, from the counts of tokens it took. Open the model.",
  });
}

const builders: Record<string, () => View> = {
  ...PARTS,
  "ds/step": step,
  "ds/model": model,
  ...Object.fromEntries(
    (Object.keys(KIND) as Kind[]).map((k) => [`ds/layer/${k}`, () => layer(k)]),
  ),
};

export function view(id: string): View | undefined {
  return builders[id]?.();
}
