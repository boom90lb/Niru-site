/**
 * Figure D's views: a family of Monads. The reader moves five controls over ranges declared for
 * this illustration (family.ts); the window redraws the model at each setting and carries the
 * drawing from one setting to the next.
 *
 * Each weight is a block whose edges follow its axes: an edge grows by one step each time its axis
 * doubles. An axis is one lane of colour wherever it appears, so a control resizes every edge on
 * its lane together: d, set by h and k, runs through every weight. Beside the weights, three lanes
 * recount the member from the definitions: its parameters, and its work per token for one
 * sequence of 1,024 tokens, forward and with the derived backward.
 *
 * The measured check opens from here: this site's spectral study, a toy model trained with NumPy,
 * redrawn from its data file. It does not use AMLO's model or its training path.
 */
import type { Ink, V3 } from "./engine.ts";
import { inRange, MONAD_SETTING, member, plain, RANGES, type Setting, T } from "./family.ts";
import { Sketch, textUnits } from "./kit.ts";
import { STAGE_VIEW, stageAt } from "./rewrite.ts";
import { fmt, type View } from "./view.ts";

/** The lanes: one ink per axis, wherever the axis appears. */
const LANES: Record<"v" | "d" | "h" | "k" | "f", Ink> = {
  v: "indigo",
  d: "ochre",
  h: "cinnabar",
  k: "verdigris",
  f: "ink",
};

/** An edge's length: one step for an axis of 1, and one more each time the axis doubles. */
const STEP = 0.42;
const edge = (n: number) => STEP * (1 + Math.log2(n));

/** A thin edge, for an axis the weight does not have. */
const THIN = 0.4;

const KEYS = ["V", "L", "H", "K", "F"] as const;

/** A setting's view id: `fam/model` at Monad's setting, else with the setting spelled out. */
export function familyId(s: Setting): string {
  return KEYS.every((k) => s[k] === MONAD_SETTING[k])
    ? "fam/model"
    : `fam/model/${KEYS.map((k) => `${k}${s[k]}`).join("-")}`;
}

/** The setting a view id names, or null if the id names none in the declared ranges. */
export function settingOf(id: string): Setting | null {
  if (id === "fam/model") return { ...MONAD_SETTING };
  const m = /^fam\/model\/V(\d+)-L(\d+)-H(\d+)-K(\d+)-F(\d+)$/.exec(id);
  if (!m) return null;
  const s = { V: +m[1]!, L: +m[2]!, H: +m[3]!, K: +m[4]!, F: +m[5]! };
  return inRange(s) ? s : null;
}

/**
 * A block standing on the drawing's plane: its front is `w` wide and `h` tall, and it reaches `depth`
 * back. Each named edge is lined with its lane's ink.
 */
function block(
  sk: Sketch,
  name: string,
  x0: number,
  y0: number,
  w: number,
  h: number,
  depth: number,
  lanes: { w?: keyof typeof LANES; h?: keyof typeof LANES; depth?: keyof typeof LANES },
) {
  const x1 = x0 + w;
  const y1 = y0 + h;
  const z = 0;
  sk.keyed(`w/${name}`, () => {
    sk.add({
      kind: "face",
      pts: [
        [x0, y1, z - depth],
        [x1, y1, z - depth],
        [x1, y1, z],
        [x0, y1, z],
      ],
      layer: 0,
      fill: "paper",
      edge: "ink",
      w: 1.1,
      hatch: { angle: Math.PI / 4, gap: 3.2, ink: "ink", w: 0.6 },
    });
    sk.add({
      kind: "face",
      pts: [
        [x1, y0, z],
        [x1, y1, z],
        [x1, y1, z - depth],
        [x1, y0, z - depth],
      ],
      layer: 0,
      fill: "paper",
      edge: "ink",
      w: 1.1,
      hatch: { angle: -Math.PI / 3, gap: 2.1, ink: "ink", w: 0.6 },
    });
    sk.add({
      kind: "face",
      pts: [
        [x0, y1, z],
        [x1, y1, z],
        [x1, y0, z],
        [x0, y0, z],
      ],
      layer: 0,
      fill: "paper",
      edge: "ink",
      w: 1.4,
    });
    sk.text(x0 + w / 2, y0 + h / 2, name, {
      size: 13,
      weight: 600,
      anchor: "middle",
      dy: 5,
    });
  });
  // The lanes: strokes of the axis's ink just outside the edges they size.
  const lane = (axis: keyof typeof LANES, which: string, pts: V3[]) =>
    sk.keyed(`lane/${axis}/${name}/${which}`, () =>
      sk.add({ kind: "stroke", pts, w: 3, ink: LANES[axis], layer: 0, halo: false }),
    );
  if (lanes.w)
    lane(lanes.w, "w", [
      [x0, y0 - 0.3, z],
      [x1, y0 - 0.3, z],
    ]);
  if (lanes.h)
    lane(lanes.h, "h", [
      [x0 - 0.3, y0, z],
      [x0 - 0.3, y1, z],
    ]);
  if (lanes.depth)
    lane(lanes.depth, "depth", [
      [x1 + 0.25, y0, z],
      [x1 + 0.25, y0, z - depth],
    ]);
}

/** A norm's gain: a rod as tall as d. */
function rod(sk: Sketch, name: string, x0: number, y0: number, h: number) {
  sk.keyed(`w/${name}`, () =>
    sk.add({
      kind: "face",
      pts: [
        [x0, y0 + h, 0],
        [x0 + 0.34, y0 + h, 0],
        [x0 + 0.34, y0, 0],
        [x0, y0, 0],
      ],
      layer: 0,
      fill: "paper",
      edge: "ink",
      w: 1.2,
    }),
  );
  sk.keyed(`lane/d/${name}/h`, () =>
    sk.add({
      kind: "stroke",
      pts: [
        [x0 - 0.3, y0, 0],
        [x0 - 0.3, y0 + h, 0],
      ],
      w: 3,
      ink: LANES.d,
      layer: 0,
      halo: false,
    }),
  );
}

/** Flat lettering at stage coordinates. */
const say = (
  sk: Sketch,
  rx: number,
  ry: number,
  text: string,
  o: Parameters<Sketch["text"]>[3] = {},
) => {
  const [x, y, z] = stageAt(rx, ry);
  sk.text(x, y, text, { z, ...o });
};

/** The recount: three lanes on one scale of powers of ten. */
const LO = 4;
const HI = 11;
const RX0 = 37;
const RX1 = 53.4;
const at = (n: number) => RX0 + ((Math.log10(n) - LO) / (HI - LO)) * (RX1 - RX0);

/**
 * A member's three counts, as its lanes print them: in plain words, and exactly. The window also
 * sets them as page text beside the sliders on a stage too narrow to show the lanes.
 */
export function counts(s: Setting) {
  const m = member(s);
  const monad = member(MONAD_SETTING);
  // Work per token is a sequence's work over its tokens; the exact figure is the sequence's.
  return [
    {
      id: "params",
      name: "parameters",
      value: m.parameters,
      ref: monad.parameters,
      exact: fmt(m.parameters),
    },
    {
      id: "forward",
      name: "forward, operations per token",
      value: m.forward,
      ref: monad.forward,
      exact: `${fmt(m.sequence.forward)} for ${fmt(T)} tokens`,
    },
    {
      id: "training",
      name: "training, operations per token",
      value: m.training,
      ref: monad.training,
      exact: `${fmt(m.sequence.training)} for ${fmt(T)} tokens`,
    },
  ].map((r) => ({ ...r, words: plain(r.value) }));
}

function recount(sk: Sketch, s: Setting) {
  counts(s).forEach((r, i) => {
    const ry = 2.2 - i * 4.6;
    sk.keyed(`count/${r.id}/name`, () => say(sk, RX0, ry, r.name, { size: 12, weight: 600 }));
    sk.keyed(`count/${r.id}/value`, () => say(sk, RX0, ry - 1.1, r.words, { size: 15, dy: 4 }));
    sk.keyed(`count/${r.id}/exact`, () =>
      say(sk, RX1, ry - 1.1, r.exact, { size: 11, ink: "faint", anchor: "end", dy: 4 }),
    );
    const [a, b] = [stageAt(RX0, ry - 2.5), stageAt(RX1, ry - 2.5)];
    sk.keyed(`count/${r.id}/axis`, () =>
      sk.add({ kind: "stroke", pts: [a, b], w: 1, ink: "faint", layer: 0, halo: false }),
    );
    sk.keyed(`count/${r.id}/bar`, () =>
      sk.add({
        kind: "stroke",
        pts: [stageAt(RX0, ry - 2.5), stageAt(at(r.value), ry - 2.5)],
        w: 6,
        ink: "ink",
        layer: 0,
        halo: false,
      }),
    );
    // Monad's place on the lane, so a member reads against the model the family moves from.
    sk.keyed(`count/${r.id}/monad`, () =>
      sk.add({
        kind: "stroke",
        pts: [stageAt(at(r.ref), ry - 1.9), stageAt(at(r.ref), ry - 3.1)],
        w: 1.4,
        ink: "ochre",
        layer: 0,
        halo: false,
      }),
    );
  });
  const ry = 2.2 - 2 * 4.6 - 2.5;
  sk.keyed("count/ticks", () => {
    for (const [p, words] of [
      [4, "10 thousand"],
      [6, "1 million"],
      [8, "100 million"],
      [10, "10 billion"],
    ] as const) {
      sk.add({
        kind: "stroke",
        pts: [stageAt(at(10 ** p), ry + 0.3), stageAt(at(10 ** p), ry - 0.3)],
        w: 1,
        ink: "faint",
        layer: 0,
        halo: false,
      });
      say(sk, at(10 ** p), ry - 0.4, words, {
        size: 10.5,
        ink: "faint",
        anchor: "middle",
        dy: 14,
      });
    }
  });
  sk.keyed("count/key", () =>
    say(sk, RX0, ry - 2.2, "The short ochre mark is Monad. Each step of the scale is ten times.", {
      size: 11,
      italic: true,
      ink: "faint",
      dy: 4,
    }),
  );
  shares(sk, s);
}

/** A share in plain words, to two figures: "67%", "3.7%". */
const percent = (f: number) => `${Number((f * 100).toPrecision(2))}%`;

/**
 * The member's parameters by part, on a linear scale, as the log lanes cannot show them: one bar
 * cut into E, attention and SwiGLU, each named with its share under its part. The norms' gains,
 * under a quarter of a percent at any setting, are too small to draw, and the caption names them.
 */
function shares(sk: Sketch, s: Setting) {
  const { parameters, parts } = member(s);
  const ry = -13.3;
  const [top, bottom] = [ry - 0.7, ry - 1.35];
  const labelY = bottom - 0.95;
  sk.keyed("count/share/name", () =>
    say(sk, RX0, ry, "parameters, by part", { size: 12, weight: 600 }),
  );
  const x = (n: number) => RX0 + (n / parameters) * (RX1 - RX0);
  let from = 0;
  const segments = [
    { id: "E", name: "E", n: parts.E },
    { id: "attention", name: "attention", n: parts.attention },
    { id: "swiglu", name: "SwiGLU", n: parts.swiglu },
  ].map((p) => {
    const [x0, x1] = [x(from), x(from + p.n)];
    from += p.n;
    const text = `${p.name} ${percent(p.n / parameters)}`;
    const w = textUnits(text, 11);
    return { ...p, x0, x1, text, w, at: (x0 + x1) / 2 - w / 2 };
  });
  // Neighbouring parts are hatched across each other, so even a sliver reads as its own part.
  segments.forEach((p, i) =>
    sk.keyed(`count/share/bar/${p.id}`, () =>
      sk.add({
        kind: "face",
        pts: [
          stageAt(p.x0, top),
          stageAt(p.x1, top),
          stageAt(p.x1, bottom),
          stageAt(p.x0, bottom),
        ],
        layer: 0,
        fill: "paper",
        edge: "ink",
        w: 1.1,
        hatch: { angle: ((i % 2 ? -1 : 1) * Math.PI) / 4, gap: 2.6, ink: "ink", w: 0.6 },
      }),
    ),
  );
  // Each name sits under the middle of its part, moved aside only as far as keeps the names apart
  // and inside the lane.
  const gap = 1.3;
  let left = RX0;
  for (const p of segments) {
    p.at = Math.max(p.at, left);
    left = p.at + p.w + gap;
  }
  let right = RX1;
  for (const p of [...segments].reverse()) {
    p.at = Math.min(p.at, right - p.w);
    right = p.at - gap;
  }
  for (const p of segments)
    sk.keyed(`count/share/name/${p.id}`, () => say(sk, p.at, labelY, p.text, { size: 11, dy: 4 }));
  const norms = parts.norms / parameters;
  sk.keyed("count/share/key", () =>
    say(
      sk,
      RX0,
      labelY - 1.15,
      `On a linear scale. The norms, ${norms < 0.001 ? "under 0.1%" : percent(norms)}, are too small to draw.`,
      { size: 11, italic: true, ink: "faint", dy: 4 },
    ),
  );
}

function familyView(id: string, s: Setting): View {
  const sk = new Sketch();
  const { D, P } = member(s);
  const ev = edge(s.V);
  const ed = edge(D);
  const eh = edge(s.H);
  const ek = edge(s.K);
  const ef = edge(s.F);
  // The embedding, and the head that reads it.
  const yE = -9;
  block(sk, "E", 0, yE, ev, ed, THIN, { w: "v", h: "d" });
  sk.keyed("note/E", () =>
    sk.text(0, yE - 1.1, "the head reads E too", {
      size: 11,
      italic: true,
      ink: "faint",
      dy: 12,
    }),
  );
  // One block: attention above, the SwiGLU below, inside a sheet repeated L times.
  const bx = Math.max(ev + 2.2, 9.2);
  const y1 = -4.4;
  const y2 = y1 - 2 - ed;
  let x = bx + 0.9;
  rod(sk, "g1", x, y1, ed);
  x += 1.1;
  for (const w of ["Wq", "Wk", "Wv", "Wo"]) {
    block(sk, w, x, y1, ek, ed, eh, { w: "k", h: "d", depth: "h" });
    x += ek + eh * 0.55 + 1.2;
  }
  const attnEnd = x;
  x = bx + 0.9;
  rod(sk, "g2", x, y2, ed);
  x += 1.1;
  for (const w of ["W1", "W3", "W2"]) {
    block(sk, w, x, y2, ef, ed, THIN, { w: "f", h: "d" });
    x += ef + 1.3;
  }
  const bx1 = Math.max(attnEnd, x) + 0.2;
  const top = y1 + ed + eh * 0.6 + 1.6;
  const bottom = y2 - 1.6;
  // The sheet, and the layers behind it: as many as L, up to twelve drawn.
  const shown = Math.min(s.L, 12);
  for (let i = shown - 1; i >= 1; i--)
    sk.keyed(`layer/${i}`, () =>
      sk.add({
        kind: "face",
        pts: [
          [bx, top, -0.5 * i],
          [bx1, top, -0.5 * i],
          [bx1, bottom, -0.5 * i],
          [bx, bottom, -0.5 * i],
        ],
        layer: 0,
        fill: "paper",
        edge: "faint",
        w: 0.8,
        under: true,
      }),
    );
  sk.keyed("layer/0", () =>
    sk.add({
      kind: "face",
      pts: [
        [bx, top, 0],
        [bx1, top, 0],
        [bx1, bottom, 0],
        [bx, bottom, 0],
      ],
      layer: 0,
      fill: "paper",
      edge: "ink",
      w: 1,
      under: true,
    }),
  );
  sk.keyed("note/block", () =>
    sk.text(bx, bottom, `one block, repeated ${fmt(s.L)} times`, {
      size: 12,
      italic: true,
      dy: 38,
    }),
  );
  // The lane key: each axis and its size.
  const key: Array<[keyof typeof LANES, string]> = [
    ["v", `v ${fmt(s.V)}`],
    ["d", `d ${fmt(D)}`],
    ["h", `h ${fmt(s.H)}`],
    ["k", `k ${fmt(s.K)}`],
    ["f", `f ${fmt(s.F)}`],
  ];
  key.forEach(([axis, text], i) => {
    const rx = 0.6 + i * 5.2;
    sk.keyed(`key/${axis}`, () => {
      sk.add({
        kind: "stroke",
        pts: [stageAt(rx, -20.4), stageAt(rx + 1.2, -20.4)],
        w: 3,
        ink: LANES[axis],
        layer: 0,
        halo: false,
      });
      say(sk, rx + 1.6, -20.4, text, { size: 12, dy: 4 });
    });
  });
  sk.keyed("key/rule", () =>
    say(
      sk,
      0.6,
      -22.2,
      `An edge grows one step each time its axis doubles. Derived, never set: d = h·k = ${fmt(D)}, p = k/2 = ${fmt(P)}.`,
      {
        size: 11,
        italic: true,
        ink: "faint",
        dy: 4,
      },
    ),
  );
  recount(sk, s);
  // The measured check opens from here.
  const cx = 45.2;
  const cy = -19.6;
  sk.keyed("check", () =>
    sk.slab(cx, cy, 10.4, 2.6, { chip: false, label: "the measured check", size: 13 }),
  );
  sk.open("check", "the measured check", "fam/spectral", [
    [cx - 5.2, cy + 1.3, 0],
    [cx + 5.2, cy + 1.3, 0],
    [cx + 5.2, cy - 1.3, 0],
    [cx - 5.2, cy - 1.3, 0],
  ]);
  return {
    id,
    title: id === "fam/model" ? "Monad’s family" : "One member of the family",
    caption: `Monad’s weights at the setting chosen below, inside ranges declared for this figure. Each edge follows an axis, and each axis is one colour wherever it appears, so a control resizes all its edges together. The counts come from the definitions; work is per token for one sequence of ${fmt(T)} tokens, without the update. Work counts each arithmetic, transcendental and compare operation as one unit, AMLO’s default weights. The result is an exact count of the program’s operations, not hardware time. Softmax and the loss are priced as AMLO’s builder composes them.`,
    sketch: sk,
    evidence: "count",
    ...STAGE_VIEW,
  };
}

// ------------------------------------------------------------------------------- the measured check

/**
 * The study's certificate rows, as src/data/spectral-certificate.json records them: each ratio to
 * its value at width 64. Copied so the window does not ship the whole data file; a unit test
 * checks every value against the file.
 */
export const CERTIFICATE = {
  band: [0.8, 1.25],
  rows: {
    sp: [
      { width: 64, act_rms_ratio: 1.0, W_op_norm_ratio: 1.0, dW_op_norm_ratio: 1.0 },
      {
        width: 128,
        act_rms_ratio: 1.0309200441206956,
        W_op_norm_ratio: 2.009839091470231,
        dW_op_norm_ratio: 2.1123112869435534,
      },
      {
        width: 256,
        act_rms_ratio: 1.0794966717034846,
        W_op_norm_ratio: 3.711322090078057,
        dW_op_norm_ratio: 3.9437056747121297,
      },
      {
        width: 512,
        act_rms_ratio: 1.061043978349872,
        W_op_norm_ratio: 6.114942687658708,
        dW_op_norm_ratio: 6.532352516665121,
      },
      {
        width: 1024,
        act_rms_ratio: 1.5702458182877472,
        W_op_norm_ratio: 11.111783111156118,
        dW_op_norm_ratio: 11.907424867343494,
      },
    ],
    umup: [
      { width: 64, act_rms_ratio: 1.0, W_op_norm_ratio: 1.0, dW_op_norm_ratio: 1.0 },
      {
        width: 128,
        act_rms_ratio: 0.9522425565764878,
        W_op_norm_ratio: 1.0288574118179104,
        dW_op_norm_ratio: 1.0335253081454867,
      },
      {
        width: 256,
        act_rms_ratio: 0.898317691747236,
        W_op_norm_ratio: 1.0292969145459823,
        dW_op_norm_ratio: 0.9961100275812994,
      },
      {
        width: 512,
        act_rms_ratio: 0.8533110866517343,
        W_op_norm_ratio: 1.0506326254314682,
        dW_op_norm_ratio: 0.9536085544772401,
      },
      {
        width: 1024,
        act_rms_ratio: 0.7975093533430805,
        W_op_norm_ratio: 1.0775369313257714,
        dW_op_norm_ratio: 0.9238116895686438,
      },
    ],
    mup: [
      { width: 64, act_rms_ratio: 1.0, W_op_norm_ratio: 1.0, dW_op_norm_ratio: 1.0 },
      {
        width: 128,
        act_rms_ratio: 1.0861846865621336,
        W_op_norm_ratio: 1.0518711553970743,
        dW_op_norm_ratio: 1.0185108027097216,
      },
      {
        width: 256,
        act_rms_ratio: 1.1465589918228027,
        W_op_norm_ratio: 1.0779827699842688,
        dW_op_norm_ratio: 1.0644420244967012,
      },
      {
        width: 512,
        act_rms_ratio: 1.1711027645207626,
        W_op_norm_ratio: 1.0828882001824967,
        dW_op_norm_ratio: 1.0801058949089668,
      },
      {
        width: 1024,
        act_rms_ratio: 1.1948573942254703,
        W_op_norm_ratio: 1.0946924178109252,
        dW_op_norm_ratio: 1.0946520964620916,
      },
    ],
  },
} as const;

interface Row {
  width: number;
  act_rms_ratio?: number;
  W_op_norm_ratio?: number;
  dW_op_norm_ratio?: number;
}

const SCHEMES = [
  { key: "sp", name: "SP" },
  { key: "umup", name: "u-μP in this toy" },
  { key: "mup", name: "Input-rate variant" },
] as const;

const SERIES = [
  { key: "act_rms_ratio", name: "activation RMS", shape: "circle" },
  { key: "W_op_norm_ratio", name: "norm of W", shape: "square" },
  { key: "dW_op_norm_ratio", name: "norm of ΔW", shape: "triangle" },
] as const;

const [BAND_LO, BAND_HI] = CERTIFICATE.band;

/** A mark at stage coordinates: a dot, a square or a triangle. */
function markAt(sk: Sketch, rx: number, ry: number, shape: string, ink: Ink) {
  if (shape === "circle") {
    sk.add({ kind: "dot", at: stageAt(rx, ry), r: 3.4, ink, layer: 0 });
    return;
  }
  const r = 0.2;
  const pts =
    shape === "square"
      ? [
          stageAt(rx - r, ry + r),
          stageAt(rx + r, ry + r),
          stageAt(rx + r, ry - r),
          stageAt(rx - r, ry - r),
        ]
      : [stageAt(rx, ry + r * 1.3), stageAt(rx + r * 1.2, ry - r), stageAt(rx - r * 1.2, ry - r)];
  sk.add({ kind: "face", pts, layer: 0, fill: "paper", edge: ink, w: 1.4 });
}

function spectralView(): View {
  const sk = new Sketch();
  const W = 13.2;
  const H = 12.6;
  const y0 = -14.6;
  // Width 64 to 1,024 across, ratios 1/2 to 16 up, both on log scales.
  const px = (x0: number, width: number) => x0 + ((Math.log2(width) - 5.6) / 4.8) * W;
  const py = (ratio: number) => y0 + ((Math.log2(ratio) + 1) / 5) * H;
  const outside = (v: number) => v < BAND_LO || v > BAND_HI;
  SCHEMES.forEach((scheme, i) => {
    const x0 = 2 + i * 17.4;
    const rows: readonly Row[] = CERTIFICATE.rows[scheme.key];
    sk.keyed(`panel/${scheme.key}`, () => {
      say(sk, x0, py(16) + 1.2, scheme.name, { size: 13, weight: 600, dy: 4 });
      // The band, washed.
      sk.add({
        kind: "face",
        pts: [
          stageAt(x0, py(BAND_HI)),
          stageAt(x0 + W, py(BAND_HI)),
          stageAt(x0 + W, py(BAND_LO)),
          stageAt(x0, py(BAND_LO)),
        ],
        layer: 0,
        fill: "wash",
        edge: "ochre",
        alpha: 0.22,
        w: 0,
      });
      for (const v of [1, 2, 4, 8, 16]) {
        sk.add({
          kind: "stroke",
          pts: [stageAt(x0, py(v)), stageAt(x0 + W, py(v))],
          w: v === 1 ? 1 : 0.6,
          ink: "faint",
          layer: 0,
          halo: false,
          ...(v === 1 ? {} : { dash: [2, 4] }),
        });
        say(sk, x0, py(v), String(v), { size: 10.5, ink: "faint", anchor: "end", dx: -6, dy: 4 });
      }
      sk.add({
        kind: "stroke",
        pts: [stageAt(x0, py(0.5)), stageAt(x0 + W, py(0.5))],
        w: 1,
        ink: "ink",
        layer: 0,
        halo: false,
      });
      for (const r of rows)
        say(sk, px(x0, r.width), py(0.5), fmt(r.width), {
          size: 10.5,
          ink: "faint",
          anchor: "middle",
          dy: 16,
        });
      say(sk, x0 + W / 2, py(0.5) - 1.6, "width", {
        size: 10.5,
        italic: true,
        ink: "faint",
        anchor: "middle",
        dy: 4,
      });
      SERIES.forEach((s, j) => {
        for (const r of rows) {
          const v = r[s.key];
          if (v === undefined) continue;
          markAt(
            sk,
            px(x0, r.width) + (j - 1) * 0.3,
            py(v),
            s.shape,
            outside(v) ? "cinnabar" : "ink",
          );
        }
      });
      // The widest model's values outside the band, as the study's figure labels them.
      const widest = rows.at(-1)!;
      const notes: Array<[number, string]> = [];
      const act = widest.act_rms_ratio;
      if (act !== undefined && outside(act)) notes.push([act, act.toFixed(act < 1 ? 3 : 2)]);
      const spectral = [widest.W_op_norm_ratio, widest.dW_op_norm_ratio].filter(
        (v): v is number => v !== undefined && outside(v),
      );
      if (spectral.length) {
        const v = Math.max(...spectral);
        notes.push([v, v.toFixed(1)]);
      }
      for (const [v, text] of notes)
        say(sk, px(x0, widest.width) - 1, py(v), text, {
          size: 11,
          ink: "cinnabar",
          anchor: "end",
          dy: v < 1 ? 16 : 4,
        });
      if (!notes.length)
        say(sk, x0 + W, py(BAND_HI) + 0.4, "plotted means inside", {
          size: 11,
          italic: true,
          ink: "faint",
          anchor: "end",
          dy: -4,
        });
    });
  });
  // The key, above the panels.
  sk.keyed("legend", () => {
    let rx = 2;
    SERIES.forEach((s) => {
      markAt(sk, rx, 3.4, s.shape, "ink");
      say(sk, rx + 0.6, 3.4, s.name, { size: 11.5, dy: 4 });
      rx += s.name.length * 0.33 + 2.4;
    });
    sk.add({
      kind: "face",
      pts: [stageAt(rx, 3.9), stageAt(rx + 1, 3.9), stageAt(rx + 1, 2.9), stageAt(rx, 2.9)],
      layer: 0,
      fill: "wash",
      edge: "ochre",
      alpha: 0.3,
      w: 0,
    });
    say(sk, rx + 1.5, 3.4, `the band, ${BAND_LO.toFixed(2)} to ${BAND_HI.toFixed(2)}`, {
      size: 11.5,
      dy: 4,
    });
  });
  sk.keyed("note", () =>
    [
      "A setting tuned on a small member may not hold on a wide one. AMLO’s system contract asks for a",
      "transfer certificate, with coordinate and spectral checks, before a setting moves to a wider model.",
      "This site’s spectral study measured such diagnostics in a toy model trained with NumPy. Each ratio is to",
      "its value at width 64, and ΔW is the change of W over training. Each point is the mean of the seeds that",
      "trained, of three, with no interval. Plotted means cover the two hidden layers, excluding input and",
      "output layers. The study does not use AMLO’s training path or qualify transfer for AMLO’s models.",
    ].forEach((line, i) => say(sk, 2, -18.6 - i * 1.15, line, { size: 12, dy: 4 })),
  );
  return {
    id: "fam/spectral",
    title: "The measured check",
    caption:
      "From this site’s spectral study, a measured diagnostic panel. At the rate tuned at width 64, it shows each value as a ratio to its value at width 64, on log scales. Marks in red fall outside the band, 0.80 to 1.25. Plotted means cover the two hidden layers and exclude input and output layers. Each point averages the seeds that trained, of three; at width 1,024 one SP seed failed. Raw per-seed values were not retained, so no uncertainty interval can be computed. A toy study, not a test of AMLO’s training path.",
    sketch: sk,
    evidence: "measured",
    kicker: "This site’s spectral study",
    ...STAGE_VIEW,
  };
}

/** The figure's views by id. */
export function figureView(id: string): View | undefined {
  if (id === "fam/spectral") return spectralView();
  const s = settingOf(id);
  return s ? familyView(id, s) : undefined;
}

export { RANGES };
