/**
 * The toy DeepSeek's definitions below the layer: the hyper-connection step,
 * attention for each kind of layer and its parts, the MoE and Engram. Sizes
 * are the toy's; each drawing follows DeepSeek's reference implementation.
 */
import { DSV41, KIND, type Kind } from "./dsv41.ts";
import type { Ink } from "./engine.ts";
import { design, LANE, type Lane, Sketch, textUnits } from "./kit.ts";
import { C, C1, D, E, F, H, K, Kk, S, S2, T } from "./lanes.ts";
import type { View } from "./view.ts";

const J: Lane = { name: "j", ink: "ink" };
const QL: Lane = { name: "q", ink: "ochre" };
const N: Lane = { name: "4d", ink: "ochre" };
const M: Lane = { name: "24", ink: "verdigris" };
const G: Lane = { name: "g", ink: "indigo" };
const R: Lane = { name: "r", ink: "ochre" };
const A: Lane = { name: "2", ink: "indigo" };

const flat = (x0: number, x1: number, y: number): Array<[number, number]> => [
  [x0, y],
  [x1, y],
];

// ------------------------------------------------------------------------------- hyper-connections

function mhc(): View {
  const s = new Sketch();
  const ys = [4.2, 1.4, -1.4, -4.2];
  const g = 0.8;
  // X arrives bundled; it is read by the mixes and the collapse before its stream axis is split.
  s.bundle(flat(-16, -6.6, 0), [T, S, D], { label: "start" });
  s.copy(-14.6, [LANE, 0, -LANE], ["ink", "verdigris", "ochre"]);
  s.copy(-12.6, [LANE, 0, -LANE], ["ink", "verdigris", "ochre"]);
  const split = s.tallHex(-5.5, 0, 11.4, "s");
  const merge = s.tallHex(14.6, 0, 11.4, "s");
  ys.forEach((y, k) => {
    s.bundle(flat(split.right, -2, y), [T, D], { gap: g, w: 1.3 });
    s.text(split.right, y + g / 2, "x", {
      italic: true,
      ink: "verdigris",
      size: 13,
      dx: 6,
      dy: -10,
      sub: `${k}`,
    });
    s.bundle(flat(5, merge.left, y), [T, D], { gap: g, w: 1.3 });
    s.plus(9.5, y, { layer: 1 });
  });
  // comb mixes the streams: each output stream is a weighted sum of all four inputs.
  for (const yj of ys)
    for (const yk of ys)
      s.add({
        kind: "stroke",
        pts: [
          [-2, yj, 0],
          [5, yk, 0],
        ],
        w: 0.9,
        ink: "verdigris",
        layer: 0,
        halo: false,
      });
  s.bundle(flat(merge.right, 20, 0), [T, S, D], { label: "end" });
  s.text(-16, -LANE, "X", {
    italic: true,
    ink: "verdigris",
    size: 13,
    anchor: "end",
    dx: -6,
    dy: 22,
  });
  s.text(20, -LANE, "X′", { italic: true, ink: "verdigris", size: 13, dx: 6, dy: 22 });

  // The collapse: the previous sublayer's pre weights the streams into one input u.
  const yB = 8.4;
  s.bundle(
    [
      [-12.6, 0],
      [-10.6, yB - 0.6],
      [-9.4, yB - 0.6],
    ],
    [T, S, D],
  );
  s.bundle(flat(-11.2, -9.4, yB + 1.6), [T, S]);
  s.text(-11.2, yB + 1.6 + LANE / 2, "pre′", {
    italic: true,
    size: 12,
    ink: "verdigris",
    dx: 3,
    dy: -7,
  });
  s.slab(-8, yB + 0.6, 2.8, 5, { label: "Σ", sub: "s", size: 15 });
  s.bundle(flat(-6.6, -3.8, yB), [T, D]);
  s.text(-5.2, yB + LANE / 2, "u", { italic: true, size: 12, anchor: "middle", dy: -7 });
  s.slab(-0.8, yB, 6, 3, { dashed: true, label: "sublayer", size: 13 });
  s.bundle(flat(2.2, 8.5, yB), [T, D]);
  s.text(4.5, yB + LANE / 2, "y", { italic: true, size: 12, anchor: "middle", dy: -7 });
  s.op(9.5, yB, "×", { w: 2 });
  s.add({
    kind: "stroke",
    pts: [
      [9.5, yB - 1, 0],
      [9.5, ys[3]!, 0],
    ],
    w: 1.5,
    ink: "ink",
    layer: 1,
  });

  // The mixes: one projection of all four streams, normalized once per position.
  const yM = 14.8;
  s.bundle(
    [
      [-14.6, 0],
      [-12.6, yM],
      [-11.8, yM],
    ],
    [T, S, D],
  );
  const sd = s.tallHex(-10.6, yM, 3.4, "s·d", { w: 2.4 });
  s.bundle(flat(sd.right, -8.6, yM), [T, N]);
  s.op(-7.4, yM, "÷ rms", { w: 2.4 });
  s.bundle(flat(-6.2, -5.4, yM), [T, N]);
  s.slab(-3.8, yM, 3.2, 3.2, { chip: true, label: "W", sub: "fn" });
  s.port(-3.8, yM - 1.6, "#fn");
  s.bundle(flat(-2.2, -1, yM), [T, M]);
  const cut = s.tallHex(0.2, yM, 4.6, "cut", { w: 2.4 });
  s.port(-0.4, yM - 2.3, "#scale #base");
  // pre leaves for the next sublayer; post meets y; comb mixes the streams.
  s.bundle(
    [
      [cut.right, yM + 1.4],
      [2.4, yM + 2.6],
      [2.8, yM + 2.6],
    ],
    [T, S],
    { gap: 0.6 },
  );
  s.op(4, yM + 2.6, "σ + ε", { w: 2.4 });
  s.bundle(flat(5.2, 20, yM + 2.6), [T, S], { gap: 0.6 });
  s.text(20, yM + 2.6, "pre, for the next sublayer", {
    italic: true,
    size: 11,
    ink: "verdigris",
    dx: 6,
    dy: 4,
  });
  s.bundle(flat(cut.right, 2.8, yM), [T, S], { gap: 0.6 });
  s.op(4, yM, "2σ", { w: 2.4 });
  s.bundle(
    [
      [5.2, yM],
      [9.7, yM],
    ],
    [T, S],
    { gap: 0.6 },
  );
  s.bundle(
    [
      [9.7, yM],
      [10.2, yM],
      [10.2, yB + 1],
    ],
    [T, S],
    { gap: 0.6 },
  );
  s.text(7.4, yM + 0.3, "post", {
    italic: true,
    size: 11,
    ink: "verdigris",
    anchor: "middle",
    dy: -7,
  });
  s.bundle(
    [
      [cut.right, yM - 1.4],
      [2.2, yM - 2.6],
      [2.4, yM - 2.6],
    ],
    [T, S, S2],
    { gap: 0.5 },
  );
  const sink = s.slab(4.2, yM - 2.6, 3.6, 1.9, { label: "Sinkhorn", size: 12 });
  s.open("sinkhorn", "Sinkhorn", "lat/sinkhorn", sink);
  // comb comes down to the streams; only where it passes over y is it drawn in front.
  s.bundle(
    [
      [6, yM - 2.6],
      [7, yM - 2.6],
      [7, yB + 1.6],
    ],
    [T, S, S2],
    { gap: 0.5 },
  );
  s.bundle(
    [
      [7, yB + 1.6],
      [7, yB - 1.6],
    ],
    [T, S, S2],
    { gap: 0.5, layer: 1 },
  );
  s.bundle(
    [
      [7, yB - 1.6],
      [7, 5.4],
    ],
    [T, S, S2],
    { gap: 0.5 },
  );
  s.text(7, 5.4, "comb", { italic: true, size: 11, ink: "verdigris", dx: 22, dy: 4 });
  return {
    id: "ds/mhc",
    title: "A hyper-connection step, stream by stream",
    caption:
      "mhc pre and mhc post around one sublayer, with the four streams drawn one by one. One projection of all four streams gives 24 numbers per position. Of these, 4 become pre, 4 become post and 16 become comb, which Sinkhorn makes doubly stochastic, so every row and every column sums to 1. The sublayer reads the streams weighted by pre′, the pre the previous sublayer computed. Its output y, weighted by post, is added to each stream after comb mixes the streams. Open Sinkhorn.",
    sketch: s,
    yaw: -0.18,
    pitch: 0.28,
  };
}

// --------------------------------------------------------------------------------------- attention

const ownsC = (k: Kind) => k === "full2" || k === "full1";
const selects = (k: Kind) => ownsC(k) || k === "reindex";

function attention(kind: Kind): View {
  const s = new Sketch();
  const r = KIND[kind].ratio;
  const c = r === 2 ? C : C1;
  const hasC = r > 0;
  const yQ = 9;
  const yW = 3.5;
  const yI = -2.5;
  const yC = -8.5;
  const left = -7;
  s.bundle(flat(left, -3.4, 0), [T, D], { label: "start" });
  s.copy(-3.4, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
  const fan = [yQ, yW, ...(selects(kind) ? [yI + 0.4] : []), ...(ownsC(kind) ? [yC] : [])];
  for (const y of fan)
    s.bundle(
      [
        [-3.4, 0],
        [0.6, y],
      ],
      [T, D],
    );

  // Queries: a low-rank map and a norm, then the heads, rotated on their last 32 dimensions.
  let qr = 0;
  const endQ = s.chain(
    yQ,
    0.6,
    [T, D],
    [
      { kind: "slab", w: 3.6, chip: true, label: "W", sub: "qa" },
      { kind: "lanes", lanes: [T, QL] },
      { kind: "norm" },
      { kind: "at", then: (x) => (qr = x + 0.5) },
      { kind: "wire", w: 1.2 },
      { kind: "slab", w: 3.6, chip: true, label: "W", sub: "qb" },
      { kind: "lanes", lanes: [T, H, K] },
      { kind: "hex", name: "rope" },
    ],
  );
  // Window keys and values: one head, used as both key and value.
  const endW = s.chain(
    yW,
    0.6,
    [T, D],
    [
      { kind: "slab", w: 3.6, chip: true, label: "W", sub: "kv" },
      { kind: "lanes", lanes: [T, K] },
      { kind: "norm" },
      { kind: "wire", w: 4.8 },
      { kind: "hex", name: "rope" },
    ],
  );
  let endI = 0;
  let endC = 0;
  // The indexer sits under the queries' low-rank code qr, which drops straight into it.
  const ix = qr + 2.6;
  const drop = (x: number, y0: number, y1: number, ink: Ink, layer = 1) =>
    s.add({
      kind: "stroke",
      pts: [
        [x, y0, 0],
        [x, y1, 0],
      ],
      w: 1.5,
      ink,
      layer,
    });
  if (hasC && selects(kind)) {
    const box = s.slab(ix, yI, 6, 4.4, {
      chip: true,
      label: "indexer",
      size: 14,
      note: `top ${DSV41.TOPK} of ${c.name}`,
    });
    s.open("indexer", "the indexer", `ds/index/${kind}`, box);
    s.bundle(flat(0.6, ix - 3, yI + 0.4), [T, D]);
    s.copy(qr, [yQ + LANE / 2], ["ink"]);
    s.copy(qr + 0.6, [yQ - LANE / 2], ["ochre"]);
    drop(qr, yQ + LANE / 2, yI + 2.2, "ink");
    drop(qr + 0.6, yQ - LANE / 2, yI + 2.2, "ochre");
    if (kind === "reindex") {
      s.wire(flat(left, ix - 3, yI - 1.2), "indigo");
      s.text(left, yI - 1.2, "index keys, from layer 20", {
        italic: true,
        size: 11,
        ink: "indigo",
        anchor: "end",
        dx: -6,
        dy: 4,
      });
      s.wire(flat(left, ix - 3, yI - 1.8), "faint", { dash: [4, 3] });
      s.text(left, yI - 1.8, "candidates, post-training", {
        italic: true,
        size: 11,
        ink: "faint",
        anchor: "end",
        dx: -6,
        dy: 12,
      });
    }
    endI = ix + 3;
  }
  if (ownsC(kind)) {
    // The entries, before rotation, rise into the indexer to make its keys.
    endC = s.chain(
      yC,
      0.6,
      [T, D],
      [
        {
          kind: "slab",
          w: 5.6,
          chip: true,
          label: "compressor",
          size: 13,
          note: `ratio ${r}`,
          open: { id: "compressor", label: "the compressor", child: `ds/comp/${r}` },
        },
        { kind: "lanes", lanes: [c, K] },
        { kind: "wire", w: qr + 1.8 - 7.1 },
        {
          kind: "at",
          then: (x) => {
            s.copy(x, [yC + LANE / 2], ["ink"]);
            s.copy(x + 0.6, [yC - LANE / 2], ["cinnabar"]);
            drop(x, yC + LANE / 2, yI - 2.2, "ink", 0);
            drop(x + 0.6, yC - LANE / 2, yI - 2.2, "cinnabar", 1);
          },
        },
        { kind: "wire", w: 2.4 },
        { kind: "hex", name: "rope" },
      ],
    );
  }
  const ends = Math.max(endQ, endW, endI, endC);
  const coreL = ends + 8.6;
  // Rows into the core; values made by earlier layers come in from the left.
  const name = (y: number, text: string) =>
    s.text(ends + 0.3, y, text, { italic: true, size: 11, dy: -7 });
  s.bundle(flat(endQ, coreL, yQ), [T, H, K]);
  name(yQ + LANE, "q");
  s.bundle(flat(endW, coreL, yW), [T, K]);
  name(yW + LANE / 2, "window kv");
  if (hasC) {
    if (selects(kind)) {
      s.bundle(flat(endI, coreL, yI), [T, Kk]);
      s.arrows(endI + 0.2, ends, yI + LANE / 2, "ink");
    } else {
      s.bundle(flat(left, coreL, yI), [T, Kk]);
      s.arrows(left + 0.2, ends, yI + LANE / 2, "ink");
      s.text(
        left,
        yI,
        `top-k, from layer ${kind === "reuse2" ? "2, 8 or 14" : "20 or a Reindex layer"}`,
        {
          italic: true,
          size: 11,
          anchor: "end",
          dx: -6,
          dy: 4,
        },
      );
    }
    name(yI + LANE / 2, "top-k");
    if (ownsC(kind)) s.bundle(flat(endC, coreL, yC), [c, K]);
    else {
      s.bundle(flat(left, coreL, yC), [c, K]);
      s.text(left, yC, `kv entries, from layer ${r === 2 ? "2, 8 or 14" : "20"}`, {
        italic: true,
        size: 11,
        anchor: "end",
        dx: -6,
        dy: 4,
      });
    }
    name(yC + LANE / 2, "kv entries");
  }

  // The core: one softmax per head over the window, the selected entries and a sink.
  const coreTop = yQ + 2.2;
  const coreBot = hasC ? yC - 2.2 : yW - 2.2;
  const yO = (coreTop + coreBot) / 2;
  const coreW = Math.max(4.4, textUnits("one softmax", 11, { italic: true }) + 1.4);
  const core = s.slab(coreL + coreW / 2, yO, coreW, coreTop - coreBot, {
    chip: true,
    label: "core",
    size: 15,
    note: "one softmax",
  });
  s.open("core", "the core", `ds/core/${kind}`, core);
  s.port(coreL + coreW / 2, coreBot, "#sink");
  const right = s.chain(
    yO,
    coreL + coreW,
    [T, H, K],
    [
      { kind: "hex", name: "rope", sup: "−1" },
      { kind: "hex", name: "h = g·h′" },
      { kind: "lanes", lanes: [T, G, H, K] },
      { kind: "slab", w: 4.6, h: 5, chip: true, label: "W", sub: "oa", note: "per group" },
      { kind: "lanes", lanes: [T, G, R] },
      { kind: "slab", w: 3.4, chip: true, label: "W", sub: "ob" },
      { kind: "lanes", lanes: [T, D] },
      { kind: "wire", w: 2 },
    ],
  );
  const yEnd = yO;
  s.bundle(flat(right, right, yEnd), [T, D], { label: "end" });

  // What this layer passes on, taken just before the core.
  if (hasC) {
    const outs: Array<[number, Ink, string, number[]?]> = [];
    if (ownsC(kind)) outs.push([yC, "cinnabar", "kv entries, to later layers"]);
    if (selects(kind)) outs.push([yI, "ink", "top-k, to the next Reuse layers"]);
    outs.forEach(([y, ink, text], j) => {
      const x = coreL - 0.9 - j * 1.2;
      const yo = yC - 4.8 - j * 1.2;
      s.add({
        kind: "dot",
        at: [x, y + (ink === "ink" ? LANE / 2 : -LANE / 2), 0],
        r: 3,
        ink,
        layer: 0,
      });
      s.wire(
        [
          [x, y + (ink === "ink" ? LANE / 2 : -LANE / 2)],
          [x, yo],
          [right, yo],
        ],
        ink,
        { layer: 1 },
      );
      s.text(right, yo, text, { italic: true, size: 11, ink, dx: 6, dy: 4 });
    });
    if (kind === "full1")
      (
        [
          ["indigo", "index keys, to the Reindex layers"],
          ["faint", "candidates, to the Reindex layers in post-training"],
        ] as const
      ).forEach(([ink, text], j) => {
        const y = yI - 1.2 - j * 0.7;
        const x = coreL - 3.3 - j * 1.2;
        const yo = yC - 4.8 - (outs.length + j) * 1.2;
        s.wire(
          [
            [endI, y],
            [x, y],
            [x, yo],
            [right, yo],
          ],
          ink,
          { layer: 1, ...(ink === "faint" ? { dash: [4, 3] } : {}) },
        );
        s.text(right, yo, text, { italic: true, size: 11, ink, dx: 6, dy: 4 });
      });
  }

  const what: Record<Kind, string> = {
    swa: "In SWA attention, each query attends to the last 64 positions and to a learned sink.",
    full2:
      "In Full attention at ratio 2, the compressor pools every two positions into one entry, the indexer scores the entries and keeps the top 32 per query, and the core attends to the window, those entries and a sink.",
    reuse2:
      "In Reuse attention at ratio 2, the compressed entries and the selection come from the last Full layer; the queries and the window are this layer’s own.",
    full1:
      "Full attention at ratio 1 has one entry per position, made from the last encoder layer’s output. Its index keys and, in post-training, its candidate pool go to the Reindex layers.",
    reindex:
      "In Reindex attention, the indexer scores layer 20’s index keys with this layer’s own queries and keeps a new top 32; in post-training only candidates from layer 20’s pool are scored.",
    reuse1:
      "In Reuse attention at ratio 1, the entries come from layer 20 and the selection from the Full or Reindex layer before this one.",
  };
  return {
    id: `ds/attn/${kind}`,
    title:
      kind === "swa" ? "Sliding-window attention" : `${KIND[kind].name} attention, ratio ${r}`,
    caption: `${what[kind]} Queries go through a low-rank map; one window head serves as both key and value. After the core, the inverse rotation and a grouped output map bring the heads back to d. Open the parts.`,
    sketch: s,
    yaw: -0.24,
    pitch: 0.36,
  };
}

// ------------------------------------------------------------------------------------------- core

function core(kind: Kind): View {
  const s = new Sketch();
  const heads = DSV41.H;
  const dz = 1.9;
  const hasC = KIND[kind].ratio > 0;
  const k1: Lane = { name: "k", ink: "cinnabar" };
  const t1: Lane = { name: "t′", ink: "ink" };
  const K1: Lane = { name: "K", ink: "ink" };
  // Rows on a sheet, for one query position: values of the window on top, scores in the
  // middle, values of the selected entries below.
  const yV = 9.6;
  const yA = 5.2;
  const yQ = 1.6;
  const yB = -2;
  const yG = -5.6;
  const x0 = 1.2;
  const x1 = 31.6;
  const yTop = yV + 1.8;
  const yBot = hasC ? yG - 1.8 : yQ - 1.8;
  const cupAt = (x: number, ya: number, yb: number, ink: Ink, at: { z: number; layer: number }) =>
    s.cup(x, ya, yb, ink, at);
  for (let h = heads - 1; h >= 0; h--) {
    const z = -h * dz;
    const layer = -h;
    const at = { z, layer };
    s.sheet(x0, yBot, x1, yTop, { z, layer });
    if (h === 0 || h === heads - 1)
      s.text(x0, yTop, `head ${h + 1}`, {
        z,
        layer,
        size: 11,
        italic: true,
        ink: "faint",
        anchor: "end",
        dx: -6,
        dy: 4,
      });
    // The query, copied to each score path.
    s.wire(flat(x0 + 0.2, 3.4, yQ), "cinnabar", at);
    s.copy(3.4, [yQ], ["cinnabar"], at);
    s.wire(
      [
        [3.4, yQ],
        [5.2, yA - 1.6],
      ],
      "cinnabar",
      at,
    );
    if (hasC)
      s.wire(
        [
          [3.4, yQ],
          [5.2, yB + 1.6],
        ],
        "cinnabar",
        at,
      );
    // Window: scores against every window position, scaled and banded.
    s.bundle(flat(x0 + 0.2, 5.2, yA), [t1, k1], at);
    cupAt(5.2, yA - LANE / 2, yA - 1.6, "cinnabar", at);
    s.wire(
      [
        [5.2, yA + LANE / 2],
        [7.4, yA + 0.4],
        [16.6, yA + 0.4],
      ],
      "ink",
      at,
    );
    s.op(9.4, yA + 0.4, "÷ k", { ...at, sup: "1/2" });
    s.mask(12.4, yA + 0.4, at);
    if (h === 0) s.open("band", "the window band", "lat/band", maskFace(12.4, yA + 0.4, z));
    // Selected entries: scores against the 32 chosen, scaled and masked to complete ones.
    if (hasC) {
      s.bundle(flat(x0 + 0.2, 5.2, yB), [k1, K1], at);
      cupAt(5.2, yB + 1.6, yB + LANE / 2, "cinnabar", at);
      s.wire(
        [
          [5.2, yB - LANE / 2],
          [7.4, yB - 0.4],
          [16.6, yB - 0.4],
        ],
        "ink",
        at,
      );
      s.op(9.4, yB - 0.4, "÷ k", { ...at, sup: "1/2" });
      s.mask(12.4, yB - 0.4, at);
      if (h === 0)
        s.open("visible", "visible entries", "lat/visible", maskFace(12.4, yB - 0.4, z));
    }
    // The sink: one learned logit per head, joined to the scores before one softmax.
    s.add({ kind: "dot", at: [14.8, yQ, z], r: 3, ink: "ink", layer });
    s.text(14.8, yQ, "sink", { ...at, size: 11, italic: true, anchor: "end", dx: -7, dy: 4 });
    s.wire(flat(14.8, 16.6, yQ), "ink", at);
    const jy0 = hasC ? yB - 1.4 : yQ - 1.4;
    const jy1 = yA + 1.4;
    const join = s.tallHex(17.6, (jy0 + jy1) / 2, jy1 - jy0, "join", at);
    const ym = (jy0 + jy1) / 2;
    s.wire(flat(join.right, 19.6, ym), "ink", at);
    s.softmax(20.4, ym, at);
    s.wire(flat(21.2, 22.4, ym), "ink", at);
    const split = s.tallHex(23.4, ym, jy1 - jy0, "split", at);
    // Weights against values: the window entries above, the selected ones below.
    s.wire(flat(split.right, 26.2, yA + 0.4), "ink", at);
    s.bundle(flat(x0 + 0.2, 26.2, yV), [k1, t1], { ...at, w: 1.2 });
    cupAt(26.2, yV - LANE / 2, yA + 0.4, "ink", at);
    s.wire(
      [
        [26.2, yV + LANE / 2],
        [28.6, ym + 0.6],
        [29.4, ym + 0.6],
      ],
      "cinnabar",
      at,
    );
    if (hasC) {
      s.wire(flat(split.right, 26.2, yB - 0.4), "ink", at);
      s.bundle(flat(x0 + 0.2, 26.2, yG), [K1, k1], { ...at, w: 1.2 });
      cupAt(26.2, yB - 0.4, yG + LANE / 2, "ink", at);
      s.wire(
        [
          [26.2, yG - LANE / 2],
          [28.6, ym - 0.6],
          [29.4, ym - 0.6],
        ],
        "cinnabar",
        at,
      );
      s.plus(29.4, ym, at);
      s.wire(flat(30.2, x1 - 0.2, ym), "cinnabar", at);
    } else s.wire(flat(29.4, x1 - 0.2, ym + 0.6), "cinnabar", at);
  }
  // On the front: the inputs, shared by every head except the queries, which the heads split.
  const back = -(heads - 1) * dz;
  const front = (y: number, ink: Ink, x = x0) =>
    s.add({
      kind: "stroke",
      pts: [
        [x, y, 0],
        [x, y, back],
      ],
      w: 1.1,
      ink,
      layer: 0,
      dash: [2, 3],
    });
  s.bundle(flat(-6, x0, yQ), [{ name: "h", ink: "indigo" }, k1], { label: "start" });
  s.text(-6, yQ + LANE / 2, "q", { italic: true, size: 12, anchor: "end", dx: -22, dy: 4 });
  front(yQ + LANE / 2, "indigo");
  const swap = (x: number, y: number, x2: number, y2: number, lanes: Lane[]) =>
    lanes.forEach((lane, i) =>
      s.wire(
        [
          [x, y + (i ? -LANE / 2 : LANE / 2)],
          [x2, y2 + (i ? LANE / 2 : -LANE / 2)],
        ],
        lane.ink,
      ),
    );
  const yWin = yA + 2.2;
  s.bundle(flat(-6, -2.6, yWin), [t1, k1], { label: "start" });
  s.copy(-2.6, [yWin + LANE / 2, yWin - LANE / 2], ["ink", "cinnabar"]);
  s.bundle(
    [
      [-2.6, yWin],
      [x0, yA],
    ],
    [t1, k1],
  );
  swap(-2.6, yWin, x0, yV, [t1, k1]);
  s.text(-6, yWin, "window kv", { italic: true, size: 11, anchor: "end", dx: -24, dy: 4 });
  if (hasC) {
    const ySel = yB - 2.2;
    s.bundle(flat(-6, -2.6, ySel), [k1, K1], { label: "start" });
    s.copy(-2.6, [ySel + LANE / 2, ySel - LANE / 2], ["cinnabar", "ink"]);
    s.bundle(
      [
        [-2.6, ySel],
        [x0, yB],
      ],
      [k1, K1],
    );
    swap(-2.6, ySel, x0, yG, [k1, K1]);
    s.text(-6, ySel, "selected entries", {
      italic: true,
      size: 11,
      anchor: "end",
      dx: -24,
      dy: 4,
    });
  }
  s.text(x0, yTop, `h = ${heads}: one sheet per head`, {
    z: back,
    italic: true,
    ink: "indigo",
    dy: -10,
  });
  s.text(x0, yTop, "drawn for one query position; every position does the same", {
    z: back,
    italic: true,
    size: 11,
    ink: "faint",
    dy: -28,
  });
  const yOut = (hasC ? (yB - 1.4 + yA + 1.4) / 2 : (yQ - 1.4 + yA + 1.4) / 2) + (hasC ? 0 : 0.6);
  front(yOut + LANE / 2, "indigo", x1);
  s.bundle(flat(x1, x1 + 7, yOut), [{ name: "h", ink: "indigo" }, k1], { label: "end" });
  return {
    id: `ds/core/${kind}`,
    title: "The core, one sheet per head",
    caption: hasC
      ? `For one query and each head, the query is scored against the window entries, masked to the last ${DSV41.W} positions, and against its ${DSV41.TOPK} selected compressed entries, masked to those already complete. A learned sink joins the scores, and one softmax runs over all of them. The weights then sum the same entries, which serve as values too. Open the window band or the visible entries.`
      : `For one query and each head, the query is scored against the window entries, masked to the last ${DSV41.W} positions; a learned sink joins the scores, and one softmax runs over both. The weights then sum the same entries, which serve as values too. Open the window band.`,
    sketch: s,
    yaw: -0.34,
    pitch: 0.42,
  };
}

function maskFace(x: number, y: number, z: number) {
  const r = 1.3;
  return [
    [x - r, y + r, z],
    [x + r, y + r, z],
    [x + r, y - r, z],
    [x - r, y - r, z],
  ] as const as Array<[number, number, number]>;
}

// ------------------------------------------------------------------------------------- compressor

function compressor(r: 1 | 2): View {
  const s = new Sketch();
  if (r === 1) {
    s.bundle(flat(-4, 1.6, 0), [T, D], { label: "start" });
    s.slab(3.4, 0, 3.6, 3.4, { chip: true, label: "W", sub: "kv" });
    s.bundle(flat(5.2, 12, 0), [C1, K], { label: "end" });
    s.norm(8.2, 0);
    s.port(8.2, -1, "#g");
    return {
      id: "ds/comp/1",
      title: "The compressor at ratio 1",
      caption:
        "At ratio 1 every position is its own entry, c′ = t, with a map to 64 dimensions and a norm. There is no gate, because a softmax over a single position would give it a weight with no gradient.",
      sketch: s,
      yaw: -0.2,
      pitch: 0.3,
    };
  }
  s.bundle(flat(-5, -1.5, 0), [T, D], { label: "start" });
  s.copy(-1.5, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
  const stride = { id: "stride", label: "the stride", child: "lat/stride" };
  const ends: number[] = [];
  for (const [y, name] of [
    [3.6, "kv"],
    [-3.6, "gate"],
  ] as const) {
    s.bundle(
      [
        [-1.5, 0],
        [1.4, y],
      ],
      [T, D],
    );
    ends.push(
      s.chain(
        y,
        1.4,
        [T, D],
        [
          { kind: "slab", w: 3.8, chip: true, label: "W", sub: name },
          { kind: "lanes", lanes: [T, K] },
          { kind: "hex", name: "t = 2c + j", ...(name === "kv" ? { open: stride } : {}) },
          { kind: "lanes", lanes: [C, J, K] },
          ...(name === "gate" ? ([{ kind: "softmax" }] as const) : []),
        ],
      ),
    );
  }
  const m = Math.max(...ends) + 1.2;
  s.bundle(flat(ends[0]!, m, 3.6), [C, J, K]);
  s.bundle(flat(ends[1]!, m, -3.6), [C, J, K]);
  s.text(ends[1]! - 0.9, -3.6 - LANE, "over j", {
    italic: true,
    size: 11,
    ink: "faint",
    anchor: "middle",
    dy: 18,
  });
  // The product, element by element, then the pair summed.
  s.add({
    kind: "stroke",
    pts: [
      [m, 3.6 - LANE - 0.4, 0],
      [m, -3.6 + LANE + 0.4, 0],
    ],
    w: 1.2,
    ink: "faint",
    layer: 0,
    dash: [4, 4],
    halo: false,
  });
  s.text(m, 0, "× elementwise", {
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -8,
    dy: 4,
  });
  for (const y of [3.6, -3.6])
    s.bundle(
      [
        [m, y],
        [m + 3, 0],
      ],
      [C, J, K],
    );
  s.chain(
    0,
    m + 3,
    [C, J, K],
    [
      { kind: "op", name: "Σ", sub: "j", w: 2 },
      { kind: "lanes", lanes: [C, K] },
      { kind: "norm", port: "#g" },
      { kind: "wire", w: 2.4 },
    ],
  );
  const end = m + 3 + 0.9 + 2 + 0.9 + 1.9 + 2.4;
  s.bundle(flat(end, end, 0), [C, K], { label: "end" });
  return {
    id: "ds/comp/2",
    title: "The compressor at ratio 2",
    caption:
      "Positions 2c and 2c + 1 make entry c. Two maps give values and gate scores; the index map t = 2c + j splits the positions into pairs, a softmax over each pair weights the values channel by channel, and the pair is summed and normalized. The entry is returned before rotation, because the indexer keys are made from it. Open the stride.",
    sketch: s,
    yaw: -0.22,
    pitch: 0.34,
  };
}

// --------------------------------------------------------------------------------------- indexer

function indexer(kind: Kind): View {
  const s = new Sketch();
  const heads = DSV41.IH;
  const dz = 2.2;
  const own = ownsC(kind);
  const c = kind === "full2" ? C : C1;
  const i1: Lane = { name: "i", ink: "cinnabar" };
  const h1: Lane = { name: "h", sub: "I", ink: "indigo" };
  const yW = 8.2;
  const yQ = 3.6;
  const yK = -1.8;
  const x0 = 9;
  const x1 = 22;
  const back = -(heads - 1) * dz;
  // The head weights: one per index head, from the layer's input.
  s.wire(flat(-6, -1.6, yW), "ochre");
  s.text(-6, yW, "u", { italic: true, size: 12, anchor: "end", dx: -6, dy: 4 });
  const endW = s.chain(
    yW,
    -1.6,
    [D],
    [
      { kind: "slab", w: 3.4, chip: true, label: "W", sub: "w" },
      { kind: "lanes", lanes: [h1] },
      { kind: "op", name: `× 1/${Math.sqrt(DSV41.IK * heads)}`, w: 2.8 },
    ],
  );
  s.wire(flat(endW, x0, yW), "indigo");
  // The index query, from the queries' low-rank code qr.
  s.bundle(flat(-6, -1.6, yQ), [QL], { label: "start" });
  s.text(-6, yQ, "qr", { italic: true, size: 12, anchor: "end", dx: -22, dy: 4 });
  const endQ = s.chain(
    yQ,
    -1.6,
    [QL],
    [
      { kind: "slab", w: 3.4, chip: true, label: "W", sub: "qI" },
      { kind: "lanes", lanes: [h1, i1] },
      { kind: "hex", name: "rope" },
    ],
  );
  s.bundle(flat(endQ, x0, yQ), [h1, i1]);
  // Index keys: made here from the entries, or read from layer 20.
  if (own) {
    s.bundle(flat(-6, -1.6, yK), [c, K], { label: "start" });
    s.text(-6, yK, "entries", {
      italic: true,
      size: 11,
      ink: "faint",
      anchor: "end",
      dx: -22,
      dy: 4,
    });
    const endK = s.chain(
      yK,
      -1.6,
      [c, K],
      [
        { kind: "slab", w: 3.4, chip: true, label: "W", sub: "k" },
        { kind: "lanes", lanes: [i1, c] },
        { kind: "norm" },
        { kind: "hex", name: "rope" },
      ],
    );
    s.bundle(flat(endK, x0, yK), [i1, c]);
  } else {
    s.bundle(flat(-6, x0, yK), [i1, c], { label: "start" });
    s.text(-6, yK, "index keys, from layer 20", {
      italic: true,
      size: 11,
      ink: "indigo",
      anchor: "end",
      dx: -22,
      dy: 4,
    });
  }
  const z = (h: number) => -h * dz;
  for (let h = heads - 1; h >= 0; h--) {
    const at = { z: z(h), layer: -h };
    s.sheet(x0, yK - 2.2, x1, yW + 1.4, at);
    s.wire(flat(x0 + 0.2, 10.6, yQ - LANE / 2), "cinnabar", at);
    s.bundle(flat(x0 + 0.2, 10.6, yK), [i1, c], at);
    s.cup(10.6, yQ - LANE / 2, yK + LANE / 2, "cinnabar", at);
    s.wire(
      [
        [10.6, yK - LANE / 2],
        [12.6, 0.8],
        [13, 0.8],
      ],
      "ink",
      at,
    );
    s.op(14.2, 0.8, "relu", { ...at, w: 2.4 });
    s.wire(flat(15.4, 16.2, 0.8), "ink", at);
    s.op(17.4, 0.8, "× w", { ...at, w: 2.4 });
    s.wire(
      [
        [x0 + 0.2, yW],
        [17.4, yW],
        [17.4, 1.8],
      ],
      "indigo",
      at,
    );
    s.wire(flat(18.6, x1 - 0.2, 0.8), "ink", at);
  }
  s.text(x1, yW + 1.4, "index head 1", {
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -6,
    dy: 16,
  });
  s.text(x0, yW + 1.4, `index head ${heads}`, {
    z: back,
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -6,
    dy: 4,
  });
  for (const [x, y] of [
    [x0, yW],
    [x0, yQ + LANE / 2],
    [x1, 0.8 + LANE / 2],
  ] as const)
    s.add({
      kind: "stroke",
      pts: [
        [x, y, 0],
        [x, y, back],
      ],
      w: 1.1,
      ink: "indigo",
      layer: 0,
      dash: [2, 3],
    });
  s.text(x0, yW + 1.4, `h_I = ${heads}: one sheet per index head`.replace("h_I", "index heads"), {
    z: back,
    italic: true,
    ink: "indigo",
    dy: -10,
  });
  s.text(x0, yW + 1.4, "drawn for one query position", {
    z: back,
    italic: true,
    size: 11,
    ink: "faint",
    dy: -28,
  });
  // The heads summed, entries not yet complete masked, then the top 32 kept.
  s.bundle(flat(x1, x1 + 1.2, 0.8), [h1, c]);
  let mx = 0;
  let maskX = 0;
  const end = s.chain(
    0.8,
    x1 + 1.2,
    [h1, c],
    [
      { kind: "op", name: "Σ", sub: "h", w: 2 },
      { kind: "lanes", lanes: [c] },
      { kind: "at", then: (x) => (maskX = x + 0.9 + 1.1) },
      { kind: "wire", w: 0.9 + 2.2 },
      ...(kind === "reindex" ? ([{ kind: "wire", w: 3.1 }] as const) : []),
      { kind: "at", then: (x) => (mx = x + 0.45) },
      { kind: "op", name: `top ${DSV41.TOPK}`, w: 3 },
      { kind: "lanes", lanes: [Kk] },
      { kind: "wire", w: 3.6 },
    ],
  );
  s.mask(maskX, 0.8);
  s.open("visible", "visible entries", "lat/visible", maskFace(maskX, 0.8, 0));
  s.text(maskX, 0.8, "complete", {
    italic: true,
    size: 11,
    ink: "faint",
    anchor: "middle",
    dy: -38,
  });
  if (kind === "reindex") {
    s.mask(maskX + 3.1, 0.8);
    s.wire(flat(maskX + 3.1, maskX + 3.1, -4), "faint", { dash: [4, 3] });
    s.wire(
      [
        [maskX + 3.1, -4.6],
        [maskX + 3.1, -0.4],
      ],
      "faint",
      { dash: [4, 3] },
    );
    s.text(maskX + 3.1, -4.6, "candidates, post-training", {
      italic: true,
      size: 11,
      ink: "faint",
      anchor: "middle",
      dy: 16,
    });
  }
  s.arrows(end - 3.4, end, 0.8, "ink");
  s.text(end, 0.8, "top-k indices, with validity", { italic: true, size: 11, dx: 6, dy: 4 });
  if (kind === "full1") {
    s.copy(mx, [0.8], ["ink"]);
    s.wire(
      [
        [mx, 0.8],
        [mx, -4.4],
        [mx + 0.9, -4.4],
      ],
      "ink",
    );
    const box = s.slab(mx + 4, -4.4, 6.2, 3, { label: "candidate pool", size: 12 });
    s.open("pool", "the candidate pool", "lat/blocks", box);
    const e2 = Math.max(end, mx + 9.6);
    s.wire(flat(mx + 7.1, e2, -4.4), "faint", { dash: [4, 3] });
    s.text(e2, -4.4, "candidates, post-training", {
      italic: true,
      size: 11,
      ink: "faint",
      dx: 6,
      dy: 4,
    });
  }
  return {
    id: `ds/index/${kind}`,
    title: "The indexer, one sheet per index head",
    caption: `For one query and each of ${heads} index heads, the index query is scored against every ${c.name === "c" ? "" : "decoder "}index key, rectified and weighted by that head’s weight; the heads are summed. Entries not yet complete are masked${kind === "reindex" ? ", and in post-training so is every entry outside layer 20’s candidate pool" : ""}, and the top ${DSV41.TOPK} are kept with a flag for each that is valid.${kind === "full1" ? ` Layer 20 also builds the candidate pool used in post-training: the best ${DSV41.NB} blocks of ${DSV41.BS} entries.` : ""} Open the visible entries${kind === "full1" ? " or the candidate pool" : ""}.`,
    sketch: s,
    yaw: -0.3,
    pitch: 0.4,
  };
}

// -------------------------------------------------------------------------------------------- MoE

function moe(): View {
  const s = new Sketch();
  const experts = DSV41.E;
  const dz = 1.8;
  let x0 = 18;
  let x1 = 0;
  const yTop = 7.8;
  const yBot = -4.4;
  const yWt = 6.6;
  const yIx = 5;
  const back = -(experts - 1) * dz;
  const body = (yc: number, xs: number, at: { z: number; layer: number }) => {
    // A SwiGLU with clamps: W1 capped above, W3 clamped both ways, multiplied element by element.
    for (const [y, name] of [
      [yc + 2.4, "1"],
      [yc - 2.4, "3"],
    ] as const)
      s.chain(
        y,
        xs,
        [T, D],
        [
          { kind: "slab", w: 3, h: 3, chip: true, label: "W", sub: name },
          { kind: "lanes", lanes: [T, F] },
          { kind: "op", name: name === "1" ? "min 10" : "clamp ±10", w: name === "1" ? 2.4 : 3 },
          ...(name === "1"
            ? ([{ kind: "op", name: "silu", w: 2 }] as const)
            : ([{ kind: "wire", w: 2.6 }] as const)),
          { kind: "wire", w: 0.6 },
        ],
        { ...at, gap: 0.6 },
      );
    const xm = xs + 10.9;
    s.add({
      kind: "stroke",
      pts: [
        [xm, yc + 2.4 - LANE, at.z],
        [xm, yc - 2.4 + LANE, at.z],
      ],
      w: 1.1,
      ink: "faint",
      layer: at.layer,
      dash: [4, 4],
      halo: false,
    });
    for (const y of [yc + 2.4, yc - 2.4])
      s.bundle(
        [
          [xm, y],
          [xm + 1.6, yc],
        ],
        [T, F],
        at,
      );
    return xm + 1.6;
  };
  s.bundle(flat(-12, -10.4, 0), [T, D], { label: "start" });
  s.text(-12, 0, "u", { italic: true, size: 12, anchor: "end", dx: -22, dy: 4 });
  s.copy(-10.4, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
  // The gate: scores from a map to the experts. Each token's top 2 are chosen on score plus
  // the bias and weighted by the scores alone.
  const yU = 10.2;
  const yL = 7;
  const yG = (yU + yL) / 2;
  s.bundle(
    [
      [-10.4, 0],
      [-8.6, yG],
      [-8.4, yG],
    ],
    [T, D],
  );
  const endG = s.chain(
    yG,
    -8.4,
    [T, D],
    [
      { kind: "slab", w: 3.2, chip: true, label: "W", sub: "g" },
      { kind: "lanes", lanes: [T, E] },
      { kind: "op", name: "softplus", sup: "1/2", w: 3.4 },
    ],
  );
  const cg = endG + 0.5;
  s.copy(cg, [yG + LANE / 2, yG - LANE / 2], ["ink", "indigo"]);
  s.bundle(
    [
      [cg, yG],
      [cg + 1.4, yU],
      [cg + 1.6, yU],
    ],
    [T, E],
  );
  s.bundle(
    [
      [cg, yG],
      [cg + 1.4, yL],
      [cg + 1.6, yL],
    ],
    [T, E],
  );
  let pick = 0;
  const endI = s.chain(
    yL,
    cg + 1.6,
    [T, E],
    [
      { kind: "op", name: "+ b", w: 1.8 },
      {
        kind: "slab",
        w: 3.2,
        h: 2.6,
        label: `top ${DSV41.A}`,
        size: 12,
        open: { id: "route", label: "the routing", child: "lat/route" },
      },
      { kind: "lanes", lanes: [T, A] },
      { kind: "at", then: (x) => (pick = x + 0.5) },
      { kind: "wire", w: 2 },
    ],
  );
  const bx = cg + 1.6 + 0.9 + 0.9;
  s.wire(
    [
      [bx, yL - 2.6],
      [bx, yL - 1],
    ],
    "faint",
    { dash: [4, 3] },
  );
  s.text(bx, yL - 2.6, "bias b: state, no gradient", {
    italic: true,
    size: 11,
    ink: "faint",
    anchor: "middle",
    dy: 15,
  });
  // The weights read the scores at the chosen experts.
  s.copy(pick, [yL + LANE / 2], ["ink"]);
  s.copy(pick + 0.6, [yL - LANE / 2], ["indigo"]);
  const endW = s.chain(
    yU,
    cg + 1.6,
    [T, E],
    [
      { kind: "wire", w: pick - cg - 1.6 - 1.4 },
      { kind: "hex", name: "at idx", w: 3.6 },
      { kind: "lanes", lanes: [T, A] },
      { kind: "op", name: "÷ Σ, × 1.5", w: 3.8 },
    ],
  );
  for (const [x, y0, ink] of [
    [pick, yL + LANE / 2, "ink"],
    [pick + 0.6, yL - LANE / 2, "indigo"],
  ] as const)
    s.add({
      kind: "stroke",
      pts: [
        [x, y0, 0],
        [x, yU - 1.6, 0],
      ],
      w: 1.5,
      ink,
      layer: 1,
    });
  x0 = Math.max(endW, endI) + 4.2;
  s.text(endW, yU + LANE / 2, "weights", { italic: true, size: 11, dx: 6, dy: -6 });
  s.text(endI, yL - LANE / 2, "indices", { italic: true, size: 11, dx: 6, dy: 20 });
  s.bundle(
    [
      [endW, yU],
      [x0 - 1.6, yU],
      [x0 - 0.2, yWt],
      [x0, yWt],
    ],
    [T, A],
  );
  s.bundle(
    [
      [endI, yL],
      [x0 - 1.6, yL],
      [x0 - 0.2, yIx],
      [x0, yIx],
    ],
    [T, A],
  );
  s.arrows(endI + 0.3, x0 - 2, yL + LANE / 2, "ink");
  // Routed experts: each token goes to the two it chose, as the reference routes it.
  s.bundle(flat(-10.4, x0, 0), [T, D]);
  const ends: number[] = [];
  for (let e = experts - 1; e >= 0; e--) {
    const at = { z: -e * dz, layer: -e };
    s.bundle(flat(x0 + 0.2, x0 + 1.6, yIx), [T, A], at);
    s.bundle(flat(x0 + 0.2, x0 + 17.2, yWt), [T, A], at);
    s.bundle(flat(x0 + 0.2, x0 + 1.4, 0), [T, D], at);
    s.tallHex(x0 + 2.8, 0.4, 3.4, "dispatch", { ...at, w: 2.8 });
    s.wire(
      [
        [x0 + 1.6, yIx + LANE / 2],
        [x0 + 2.2, yIx + LANE / 2],
        [x0 + 2.2, 2.1],
      ],
      "ink",
      at,
    );
    s.wire(
      [
        [x0 + 1.6, yIx - LANE / 2],
        [x0 + 2.8, yIx - LANE / 2],
        [x0 + 2.8, 2.1],
      ],
      "indigo",
      at,
    );
    for (const y of [2.4, -2.4])
      s.bundle(
        [
          [x0 + 4.2, 0.4],
          [x0 + 5.4, y],
          [x0 + 5.6, y],
        ],
        [T, D],
        at,
      );
    const xm = body(0, x0 + 5.6, at);
    const end = s.chain(
      0,
      xm,
      [T, F],
      [
        { kind: "op", name: "× w", w: 2 },
        { kind: "slab", w: 3, h: 3, chip: true, label: "W", sub: "2" },
        { kind: "lanes", lanes: [T, D] },
        { kind: "hex", name: "combine", w: 3.2 },
        { kind: "wire", w: 1 },
      ],
      { ...at, gap: 0.5 },
    );
    s.bundle(
      [
        [x0 + 17.2, yWt],
        [xm + 1.5, yWt],
        [xm + 1.5, 1.2],
      ],
      [T, A],
      { ...at, gap: 0.5 },
    );
    ends.push(end);
  }
  // Each sheet a little wider than what is drawn on it.
  x1 = Math.max(...ends) + 2.4;
  for (let e = experts - 1; e >= 0; e--) {
    const at = { z: -e * dz, layer: -e };
    s.sheet(x0, yBot, x1, yTop, at);
    s.bundle(flat(ends[0]!, x1 - 0.2, 0), [T, D], at);
  }
  s.text(x1, yTop, "expert 1", {
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -6,
    dy: 16,
  });
  s.text(x0, yTop, `expert ${experts}`, {
    z: back,
    size: 11,
    italic: true,
    ink: "faint",
    anchor: "end",
    dx: -6,
    dy: 4,
  });
  s.text(x0, yTop, `e = ${experts}: one sheet per routed expert`, {
    z: back,
    italic: true,
    ink: "indigo",
    dy: -10,
  });
  for (const [x, y] of [
    [x0, 0],
    [x0, yWt],
    [x0, yIx],
    [x1, LANE / 2],
  ] as const)
    s.add({
      kind: "stroke",
      pts: [
        [x, y, 0],
        [x, y, back],
      ],
      w: 1.1,
      ink: "indigo",
      layer: 0,
      dash: [2, 3],
    });
  // The experts' outputs summed, past the stack, then the shared expert added.
  const xr = x1 + 4.4;
  s.bundle(flat(x1, xr, 0), [E, T, D]);
  const endR = s.chain(
    0,
    xr,
    [E, T, D],
    [
      { kind: "op", name: "Σ", sub: "e", w: 2 },
      { kind: "lanes", lanes: [T, D] },
      { kind: "wire", w: 1.4 },
    ],
  );
  // The shared expert: every token, unweighted.
  const yS = -10;
  s.bundle(
    [
      [-10.4, 0],
      [-8.6, yS],
      [x0 + 4, yS],
    ],
    [T, D],
  );
  for (const y of [yS + 2.4, yS - 2.4])
    s.bundle(
      [
        [x0 + 4, yS],
        [x0 + 5.4, y],
        [x0 + 5.6, y],
      ],
      [T, D],
    );
  s.text(x0 - 4, yS - LANE / 2, "shared expert", { italic: true, size: 12, dy: 20 });
  const xm = body(yS, x0 + 5.6, { z: 0, layer: 0 });
  const endS = s.chain(
    yS,
    xm,
    [T, F],
    [
      { kind: "slab", w: 3, h: 3, chip: true, label: "W", sub: "2" },
      { kind: "lanes", lanes: [T, D] },
    ],
    { gap: 0.5 },
  );
  const xp = endR + 1.2;
  s.bundle(
    [
      [endS, yS],
      [xp - 1.4, yS],
      [xp, -1.2],
    ],
    [T, D],
  );
  s.plus(xp, 0);
  s.bundle(flat(endR, xp - 0.6, 0), [T, D]);
  s.bundle(flat(xp + 0.6, xp + 4, 0), [T, D], { label: "end" });
  return {
    id: "ds/moe",
    title: "The MoE, eight routed experts and one shared",
    caption: `The gate scores the ${experts} experts with the square root of a softplus. Each token picks its top ${DSV41.A} by score plus a bias, and weights them by the scores alone, normalized and scaled by 1.5; the bias is state that a rule outside the gradient adjusts. Each routed expert is a SwiGLU with clamps, its weight applied before the last map. The shared expert sees every token. Drawn as the reference routes; AMLO’s toy term runs every expert on every token and zeroes the rest, which gives the same values and gradients at four times the work. Open the routing.`,
    sketch: s,
    yaw: -0.22,
    pitch: 0.4,
  };
}

// ----------------------------------------------------------------------------------------- Engram

function engram(): View {
  const s = new Sketch();
  const yH = 7.4;
  const yG = 1.2;
  const yX = -6;
  const N: Lane = { name: "n", ink: "ink" };
  const Rw: Lane = { name: "6", ink: "ink" };
  const Wd: Lane = { name: "6·64", ink: "ochre" };
  const D5: Lane = { name: "5d", ink: "ochre" };
  const note = (x: number, text: string) =>
    s.text(x, yH, text, {
      italic: true,
      size: 11,
      ink: "faint",
      anchor: "middle",
      dy: design.ppu * 1.25 + 14,
    });
  // Integer steps first: arrowheads mark the integer wires.
  const ints = (x0: number, x1: number, lanes: Lane[]) => {
    s.bundle(flat(x0, x1, yH), lanes);
    lanes.forEach((_, i) =>
      s.arrows(x0 + 0.2, x1, yH + ((lanes.length - 1) / 2 - i) * LANE, "ink"),
    );
  };
  let x = -6;
  ints(-9, x, [T]);
  s.text(-9, yH, "token ids", { italic: true, size: 12, anchor: "end", dx: -6, dy: 4 });
  const wMap = textUnits("token map", 11, { italic: true }) + 1.2;
  s.op(x + 0.6 + wMap / 2, yH, "token map", { w: wMap });
  note(x + 0.6 + wMap / 2, "50,257 ids to 32,962");
  x += 0.6 + wMap;
  ints(x, x + 2.6, [T]);
  x += 2.6;
  const wLook = textUnits("t − n", 11, { italic: true }) + 2.4;
  note(x + wLook / 2, "n = 0 … 3");
  x += wLook;
  ints(x - wLook / 2, x + 2.6, [T, N]);
  x += 2.6;
  const wHash = textUnits("hash", 11, { italic: true }) + 1.6;
  s.op(x + 0.6 + wHash / 2, yH, "hash", { w: wHash, h: 2.6 });
  note(x + 0.6 + wHash / 2, "× m, xor, mod p");
  s.text(x + 0.6 + wHash / 2, yH, "3 orders × 2 heads", {
    italic: true,
    size: 11,
    ink: "faint",
    anchor: "middle",
    dy: design.ppu * 1.25 + 30,
  });
  x += 0.6 + wHash;
  ints(x, x + 3.2, [T, Rw]);
  x += 3.2;
  // Rows of the table, then one map to four keys and a value.
  let xSplit = 0;
  const endW = s.chain(
    yH,
    x,
    [T, Rw],
    [
      { kind: "slab", w: 4, label: "table", size: 13, chip: true, port: "#T" },
      { kind: "lanes", lanes: [T, Wd] },
      { kind: "slab", w: 3.4, chip: true, label: "W", sub: "kv" },
      { kind: "lanes", lanes: [T, D5] },
      { kind: "hex", name: "split" },
      { kind: "at", then: (v) => (xSplit = v) },
    ],
    { gap: 0.9 },
  );
  void endW;
  // The value runs on; the keys come down to the gate.
  s.bundle(flat(xSplit, xSplit + 13, yH), [T, D]);
  s.hexagon(
    -6 +
      0.6 +
      textUnits("token map", 11, { italic: true }) +
      1.2 +
      2.6 +
      (textUnits("t − n", 11, { italic: true }) + 2.4) / 2,
    yH,
    "t − n",
    {
      w: textUnits("t − n", 11, { italic: true }) + 2.4,
      r: 1.25,
    },
  );
  s.text(xSplit + 1.2, yH + LANE / 2, "value", { italic: true, size: 11, dy: -7 });
  s.bundle(
    [
      [xSplit - 0.2, yH],
      [xSplit + 2.4, yG + 1.6],
      [xSplit + 5.4, yG + 1.6],
    ],
    [T, S, D],
  );
  s.text(xSplit + 3.6, yG + 1.6 + LANE, "keys", { italic: true, size: 11, dy: -7 });
  // The streams, and a gate per stream from a normalized dot product with its key.
  const xg = xSplit + 8.4;
  s.bundle(flat(-9, xg - 5, yX), [T, S, D], { label: "start" });
  s.text(-9, yX - LANE, "X", {
    italic: true,
    ink: "verdigris",
    size: 13,
    anchor: "end",
    dx: -22,
    dy: 18,
  });
  const xc = xg - 5;
  s.copy(xc, [yX + LANE, yX, yX - LANE], ["ink", "verdigris", "ochre"]);
  s.bundle(
    [
      [xc, yX],
      [xc + 2.4, yG - 1.6],
      [xg - 2.6, yG - 1.6],
    ],
    [T, S, D],
  );
  s.slab(xg, yG, 5.2, 7.6, { label: "gate", size: 13, note: "per stream" });
  s.port(xg, yG - 3.8, "#q·#k");
  const xm = xg + 2.6 + textUnits("σ of a signed root", 11, { italic: true }) + 2;
  s.bundle(flat(xg + 2.6, xm, yG), [T, S]);
  s.text(xg + 2.6, yG + LANE / 2, "σ of a signed root", {
    italic: true,
    size: 11,
    ink: "faint",
    dx: 12,
    dy: -7,
  });
  const wx = textUnits("×", 11, { italic: true }) + 1.6;
  s.op(xm + wx / 2, yG, "×", { w: wx, h: 2.6 });
  s.bundle(
    [
      [xSplit + 13, yH],
      [xm + wx / 2 + 0.3, yH],
      [xm + wx / 2 + 0.3, yG + 1.3],
    ],
    [T, D],
  );
  const xo = xm + wx;
  s.bundle(flat(xc, xo + 2, yX), [T, S, D]);
  s.bundle(
    [
      [xo, yG],
      [xo + 1.4, yG],
      [xo + 2.6, yX + 1.1],
    ],
    [T, S, D],
  );
  s.plus(xo + 2.6, yX);
  s.bundle(flat(xo + 3.4, xo + 7, yX), [T, S, D], { label: "end" });
  s.text(xo + 7, yX - LANE, "X′", { italic: true, ink: "verdigris", size: 13, dx: 22, dy: 18 });
  return {
    id: "ds/engram",
    title: "Engram, an n-gram lookup added to the streams",
    caption: `Before layers 1 and 14, each token’s last two, three and four compressed ids are hashed by two heads into a table of ${DSV41.engramDim}-wide rows (${DSV41.engramRows[0]!.toLocaleString("en-US")} rows at layer 1, ${DSV41.engramRows[1]!.toLocaleString("en-US")} at layer 14). One map turns the six rows into a key for each stream and one shared value; each stream adds the value, gated by a sigmoid of the signed square root of its normalized dot product with its key. The token map, look-back and hash are integer arithmetic in the program.`,
    sketch: s,
    yaw: -0.22,
    pitch: 0.34,
  };
}

const kinds = Object.keys(KIND) as Kind[];

export const PARTS: Record<string, () => View> = {
  "ds/mhc": mhc,
  "ds/moe": moe,
  "ds/engram": engram,
  "ds/comp/1": () => compressor(1),
  "ds/comp/2": () => compressor(2),
  ...Object.fromEntries(kinds.map((k) => [`ds/attn/${k}`, () => attention(k)])),
  ...Object.fromEntries(kinds.map((k) => [`ds/core/${k}`, () => core(k)])),
  ...Object.fromEntries(kinds.filter(selects).map((k) => [`ds/index/${k}`, () => indexer(k)])),
};
