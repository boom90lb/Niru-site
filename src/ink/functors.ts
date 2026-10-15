/**
 * Figure C, one term with many functors: Monad's block, and the programs and figures AMLO's 0.0.14
 * design derives from it, each drawn as a band aligned under the term, piece by piece:
 *
 * - the derived backward, run from the output back to the input, with residual wires to the values
 *   it reads;
 * - the cost fold: every node priced exactly, forward and backward, by AMLO's work rules as
 *   derive.ts models them, one unit per scalar operation;
 * - the plan: precision as casts, and the residual policy, StoreAll or Remat, which moves memory
 *   and plan work, never mathematics;
 * - the identities, and which edit moves which; under Remat, the reader's plan edit, execution is
 *   marked as moved.
 *
 * Composing two pieces composes the backward, the cost and the plan: the block is two halves, and
 * each of those bands of the block is its halves' bands joined, the backward's in reverse order.
 * The identities hash the whole block.
 *
 * Work is per layer for one declared sequence of T = 1,024 tokens. The identity column names each
 * identity, what it hashes and which edit moves it, and shows no hash values: AMLO does not compute
 * these identities yet, and the site makes none of its own. The design is not built.
 */
import { type Prim, unrotate, type V3 } from "./engine.ts";
import { blockWork, type Controls, units, ZERO, plus, type Step, type Work } from "./derive.ts";
import { T as SEQ } from "./family.ts";
import { LANE, type Lane, Sketch } from "./kit.ts";
import { MONAD } from "./monad.ts";
import { ampere, type Component, COMPONENTS, EDITS, residuals, work } from "./plan.ts";
import { type Beat, type Chapter, type Fx, light, makeFilm, mark, swell } from "./reel.ts";
import { NEW, PITCH, stageAt, SX, SY, YAW } from "./rewrite.ts";
import { fmt, type View } from "./view.ts";

const T: Lane = { name: "t", ink: "ink" };
const D: Lane = { name: "d", ink: "ochre" };
const C: Controls = { ...MONAD, T: SEQ };

export const HONESTY = "AMLO’s 0.0.14 rules as this window models them, not AMLO’s output.";

// ------------------------------------------------------------------------------------ the pieces

/** The block's pieces, left to right, and the half each belongs to. */
const PIECES = [
  { id: "n1", kind: "norm", half: 0 },
  { id: "attn", kind: "box", half: 0, label: "attn" },
  { id: "x1", kind: "add", half: 0 },
  { id: "n2", kind: "norm", half: 1 },
  { id: "swiglu", kind: "box", half: 1, label: "swiglu" },
  { id: "x2", kind: "add", half: 1 },
] as const;
type Piece = (typeof PIECES)[number]["id"];

/** Each piece's place along the term; the second half moves right when the halves part. */
const X: Record<Piece, number> = {
  n1: 4.8,
  attn: 10,
  x1: 15,
  n2: 19,
  swiglu: 24.2,
  x2: 29.2,
};
const GAP = 3.6;
const END = 32.4;
const Y = { term: 0, back: -6.2, cost: -12, plan: -15.2 } as const;

/** The piece a backward step belongs to. */
function pieceOf(s: Step): Piece | "in" {
  if (s.kind === "fanout") {
    if (s.of === "x") return "in";
    if (s.of === "x1") return "x1";
    if (s.of === "n1") return "n1";
    if (s.of === "n2") return "n2";
    return "attn";
  }
  if (s.def === "block") return s.of as Piece;
  return s.def as Piece;
}

/** Work per piece, forward and backward, for the block at Monad's sizes. */
export function costs(c: Controls = C) {
  const w = blockWork(c);
  const e = { t: c.T, "t′": c.T, d: c.D, h: c.H, k: c.K, p: c.K / 2, f: c.F, v: c.V };
  const forward = new Map<string, Work>();
  const backward = new Map<string, Work>();
  for (const n of w.program.nodes) {
    const k = n.def === "block" ? n.result : n.def;
    forward.set(k, plus(forward.get(k) ?? ZERO, w.program.forward(n, e)));
  }
  for (const s of w.steps) {
    const k = pieceOf(s);
    backward.set(k, plus(backward.get(k) ?? ZERO, s.work));
  }
  return { forward, backward, total: { forward: units(w.forward), backward: units(w.backward) } };
}

/** A count in plain words, to two figures. */
function plain(n: number): string {
  const r = Number(n.toPrecision(2));
  if (r >= 1e9) return `${Number((r / 1e9).toPrecision(2))} billion`;
  if (r >= 1e6) return `${Number((r / 1e6).toPrecision(2))} million`;
  if (r >= 1e3) return `${Number((r / 1e3).toPrecision(2))} thousand`;
  return String(r);
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

// ------------------------------------------------------------------------------------ drawing

interface State {
  backward: boolean;
  cost: boolean;
  plan: boolean;
  identity: boolean;
  policy: "StoreAll" | "Remat";
  /** The halves parted, to show each band as its pieces' bands joined. */
  parted: boolean;
  /** Bars at full height, or pressed flat before the cost fold grows them. */
  grown: boolean;
}

const x = (p: Piece, s: State) =>
  X[p] + (s.parted && PIECES.find((q) => q.id === p)!.half === 1 ? GAP : 0);
const end = (s: State) => END + (s.parted ? GAP : 0);

/**
 * The figure's stage: the strip's, with room on the left for the column of band names. Every view
 * of the figure, the derived views too, keeps this one scale.
 */
const STAGE_VIEW = {
  stage: (
    [
      [-8.4, SY[0]],
      [SX[1], SY[0]],
      [SX[1], SY[1]],
      [-8.4, SY[1]],
    ] as const
  ).map(([sx, sy]) => unrotate([sx, sy, 0], YAW, PITCH)),
  yaw: YAW,
  pitch: PITCH,
};

/** Where every band starts, and the column of band names left of it. */
const START = -0.4;
const bandName = (sk: Sketch, y: number, text: string, o: { alpha?: number } = {}) =>
  sk.text(START, y, text, { size: 12, weight: 600, anchor: "end", dx: -24, dy: 4, ...o });

/**
 * A norm glyph: a circle with the paper's V. A dashed circle is recomputed, as the design draws
 * recomputed nodes; a pullback is drawn solid, in the backward band.
 */
function normGlyph(sk: Sketch, cx: number, y: number, o: { dashed?: boolean } = {}) {
  const pts: V3[] = [];
  for (let i = 0; i < 28; i++)
    pts.push([
      cx + Math.cos((i / 28) * 2 * Math.PI) * 0.95,
      y + Math.sin((i / 28) * 2 * Math.PI) * 0.95,
      0,
    ]);
  sk.add({
    kind: "face",
    pts,
    layer: 0,
    fill: "paper",
    edge: "ink",
    w: 1.6,
    ...(o.dashed ? { dash: [4, 3] } : {}),
  });
  sk.add({
    kind: "stroke",
    pts: [
      [cx - 0.47, y + 0.43, 0],
      [cx, y - 0.52, 0],
      [cx + 0.47, y + 0.43, 0],
    ],
    w: 1.1,
    ink: "ink",
    layer: 0,
    halo: false,
  });
}

/** One half of the term: a branch off the stream through a norm and a call, back to an add. */
function termHalf(sk: Sketch, s: State, half: 0 | 1, y: number) {
  const [norm, call, add] =
    half === 0 ? (["n1", "attn", "x1"] as const) : (["n2", "swiglu", "x2"] as const);
  const xa = x(norm, s) - 2;
  const top = y + 2.6;
  sk.keyed(`term/${half}`, () => {
    sk.copy(xa, [y + LANE / 2, y - LANE / 2], ["ink", "ochre"]);
    sk.bundle(
      [
        [xa, y],
        [xa + 1.2, top],
        [x(call, s) - 2.3, top],
      ],
      [T, D],
    );
    sk.bundle(
      [
        [x(call, s) + 2.3, top],
        [x(add, s) - 1.3, top],
        [x(add, s), y],
      ],
      [T, D],
    );
    sk.bundle(
      [
        [xa, y],
        [x(add, s), y],
      ],
      [T, D],
    );
    sk.keyed(norm, () => normGlyph(sk, x(norm, s), top));
    sk.keyed(call, () =>
      sk.slab(x(call, s), top, 4.6, 3, {
        chip: true,
        label: call,
        size: 15,
      }),
    );
    sk.keyed(add, () => sk.plus(x(add, s), y));
  });
}

/** The term: Monad's block, two halves on one stream. */
function term(sk: Sketch, s: State) {
  const y = Y.term;
  sk.keyed("term/in", () =>
    sk.bundle(
      [
        [START, y],
        [x("n1", s) - 2, y],
      ],
      [T, D],
      { label: "start" },
    ),
  );
  termHalf(sk, s, 0, y);
  // Parted, the halves stand apart: each band shows its two pieces with nothing joining them.
  if (!s.parted)
    sk.keyed("term/mid", () =>
      sk.bundle(
        [
          [x("x1", s), y],
          [x("n2", s) - 2, y],
        ],
        [T, D],
      ),
    );
  termHalf(sk, s, 1, y);
  sk.keyed("term/out", () =>
    sk.bundle(
      [
        [x("x2", s), y],
        [end(s), y],
      ],
      [T, D],
      { label: "end" },
    ),
  );
  sk.keyed("term/name", () => bandName(sk, y, "the term"));
}

/** Small arrowheads along a wire, pointing left: the cotangent's direction. */
function leftward(sk: Sketch, x0: number, x1: number, y: number) {
  for (let xx = x1 - 0.9; xx > x0 + 0.4; xx -= 2.4)
    sk.add({
      kind: "stroke",
      pts: [
        [xx + 0.28, y + 0.26, 0],
        [xx, y, 0],
        [xx + 0.28, y - 0.26, 0],
      ],
      w: 1.2,
      ink: "ink",
      layer: 0,
      halo: false,
    });
}

/** The derived backward: the same pieces, pulled back from right to left. */
function backward(sk: Sketch, s: State) {
  const y = Y.back;
  const top = y + 2.4;
  const pull = (half: 0 | 1) => {
    const [norm, call, add] =
      half === 0 ? (["n1", "attn", "x1"] as const) : (["n2", "swiglu", "x2"] as const);
    const xa = x(norm, s) - 2;
    sk.keyed(`back/${half}`, () => {
      // The add's transpose copies the cotangent into the branch and the skip.
      sk.copy(x(add, s), [y + 0.4, y - 0.4], ["ink", "ink"]);
      sk.wire(
        [
          [x(add, s), y],
          [x(add, s) - 1.3, top],
          [x(call, s) + 2.3, top],
        ],
        "ink",
        { w: 1.3 },
      );
      sk.wire(
        [
          [x(call, s) - 2.3, top],
          [xa + 1.2, top],
          [xa, y],
        ],
        "ink",
        { w: 1.3 },
      );
      sk.wire(
        [
          [xa, y],
          [x(add, s), y],
        ],
        "ink",
        { w: 1.3 },
      );
      leftward(sk, xa, x(add, s), y);
      sk.keyed(`${call}T`, () =>
        sk.slab(x(call, s), top, 4.6, 3, {
          chip: true,
          label: call,
          sub: "T",
          size: 14,
        }),
      );
      sk.keyed(`${norm}T`, () => normGlyph(sk, x(norm, s), top));
      // Where the branch's cotangent meets the skip's, they add: an accumulation.
      sk.keyed(`${add}acc`, () => sk.plus(xa, y));
    });
  };
  sk.keyed("back/in", () => {
    sk.wire(
      [
        [START, y],
        [x("n1", s) - 2, y],
      ],
      "ink",
      { w: 1.3 },
    );
    leftward(sk, START, x("n1", s) - 2, y);
  });
  pull(0);
  if (!s.parted)
    sk.keyed("back/mid", () => {
      sk.wire(
        [
          [x("x1", s), y],
          [x("n2", s) - 2, y],
        ],
        "ink",
        { w: 1.3 },
      );
    });
  pull(1);
  sk.keyed("back/out", () =>
    sk.wire(
      [
        [x("x2", s), y],
        [end(s), y],
      ],
      "ink",
      { w: 1.3 },
    ),
  );
  sk.keyed("back/name", () => bandName(sk, y, "the backward"));
  // Residual wires: each pullback reads values its piece computed in the forward. Under StoreAll
  // those values are kept, and the wires are solid and faint. Under Remat they are not kept: the
  // backward recomputes them, and the wires are dashed, as the design draws what is recomputed.
  const remat = s.policy === "Remat";
  for (const p of ["n1", "attn", "n2", "swiglu"] as const) {
    const dx = p.startsWith("n") ? 0 : 1.6;
    const inset = p.startsWith("n") ? 0.95 : 1.5;
    sk.keyed(`${remat ? "remat" : "residuals"}/${p}`, () =>
      sk.add({
        kind: "stroke",
        pts: [
          [x(p, s) + dx, Y.term + 2.6 - inset, 0],
          [x(p, s) + dx, Y.back + 2.4 + inset, 0],
        ],
        w: remat ? 1.6 : 1,
        ink: remat ? NEW : "faint",
        layer: 0,
        ...(remat ? { dash: [6, 4] } : {}),
        halo: false,
      }),
    );
  }
}

/** The cost fold: under each piece, its forward and its backward, to one scale. */
function cost(sk: Sketch, s: State) {
  const { forward, backward: back } = costs();
  const max = Math.max(...[...back.values(), ...forward.values()].map(units));
  const y0 = Y.cost;
  const H = 2.6;
  // Before the fold, the band is pressed flat and unseen, so the bars grow in from the axis.
  const hidden = s.grown ? {} : { alpha: 0 };
  const bar = (key: string, cx: number, value: number, hatch: boolean) =>
    sk.keyed(key, () =>
      sk.add({
        kind: "face",
        pts: [
          [cx - 0.36, y0 + (s.grown ? (H * value) / max : 0), 0],
          [cx + 0.36, y0 + (s.grown ? (H * value) / max : 0), 0],
          [cx + 0.36, y0, 0],
          [cx - 0.36, y0, 0],
        ],
        layer: 0,
        fill: "paper",
        edge: "ink",
        w: 1.1,
        ...(hatch ? { hatch: { angle: Math.PI / 4, gap: 3, ink: "ink", w: 0.6 } } : {}),
        ...(s.grown ? {} : { alpha: 0 }),
      }),
    );
  for (const p of PIECES) {
    if (p.kind === "add") continue; // an add's few operations would not show at this scale
    const f = units(forward.get(p.id) ?? ZERO);
    const b = units(back.get(p.id) ?? ZERO) + (p.id === "n1" ? units(back.get("in") ?? ZERO) : 0);
    bar(`cost/${p.id}/f`, x(p.id, s) - 0.42, f, false);
    bar(`cost/${p.id}/b`, x(p.id, s) + 0.42, b, true);
    // Only the calls' figures are set; a norm's few million would not fit between them.
    if (p.kind === "box")
      sk.keyed(`cost/${p.id}/n`, () => {
        const o = { size: 11, anchor: "middle", ...(s.grown ? {} : { alpha: 0 }) } as const;
        sk.text(x(p.id, s), y0, `forward ${plain(f)}`, { ...o, dy: 17 });
        sk.text(x(p.id, s), y0, `backward ${plain(b)}`, { ...o, dy: 31 });
      });
  }
  const axis = (key: string, x0: number, x1: number) =>
    sk.keyed(key, () =>
      sk.add({
        kind: "stroke",
        pts: [
          [x0, y0, 0],
          [x1, y0, 0],
        ],
        w: 1,
        ink: "faint",
        layer: 0,
        halo: false,
        ...hidden,
      }),
    );
  axis("cost/axis/0", START, x("x1", s));
  if (!s.parted) axis("cost/axis/mid", x("x1", s), x("n2", s) - 2);
  axis("cost/axis/1", x("n2", s) - 2, end(s));
  sk.keyed("cost/name", () => bandName(sk, y0 + 1.3, "the cost", hidden));
}

/** The plan: casts per piece, the stream in f32, and the residual policy's memory and work. */
function plan(sk: Sketch, s: State) {
  const a = ampere();
  const casts = (ids: readonly string[]) =>
    a.casts.filter((c) => ids.some((id) => c.value === id || c.value.startsWith(`${id}.`)));
  const byPiece: Record<Piece, string> = {
    n1: `${casts(["n1"]).length} cast`,
    attn: `bf16 in, ${casts(["Wq", "Wk", "Wv", "Wo", "q", "kr", "pr"]).length} casts`,
    x1: "f32",
    n2: `${casts(["n2"]).length} cast`,
    swiglu: `bf16 in, ${casts(["W1", "W3", "W2", "m"]).length} casts`,
    x2: "f32",
  };
  const y = Y.plan;
  for (const p of PIECES)
    sk.keyed(`plan/${p.id}`, () =>
      sk.text(x(p.id, s), y, byPiece[p.id], {
        size: 11,
        italic: p.kind !== "box",
        anchor: "middle",
        dy: 4,
      }),
    );
  const r = residuals(C, s.policy);
  const w = work(C, s.policy);
  sk.keyed("plan/name", () => bandName(sk, y, "the plan"));
  sk.keyed(`plan/policy/${s.policy}`, () =>
    sk.text(
      end(s),
      y,
      s.policy === "StoreAll"
        ? `ampere-bf16, StoreAll: ${mb(r.bytes)} kept per layer`
        : `ampere-bf16, Remat: ${mb(r.bytes)} kept, ${plain(w.plan)} recomputed`,
      { size: 11.5, dx: 10, dy: 4, ...(s.policy === "Remat" ? { ink: NEW } : {}) },
    ),
  );
}

/**
 * What each identity hashes, in a phrase; the identity chapter's ledger gives it in full. The
 * column shows no hash values: AMLO does not compute these identities yet, and the site makes none.
 */
const HASHES: Record<Component, string> = {
  structure: "the canonical form",
  configuration: "the root controls",
  numerics: "the initializers",
  randomness: "the leaf and site ids",
  execution: "the above and the plan",
};

/** The identity column's left edge, each identity's row, and each moved-by mark's column. */
const RX = 38.4;
const rowY = (i: number) => 1.4 - 1.25 * i;
const markX = (j: number) => RX + 11.8 + j * 0.9;

/** Execution identity marked as moved: a verdigris dot after its row, in the moved-by column. */
const executionMoved = (o: { alpha?: number; glow?: number } = {}): Prim => ({
  kind: "dot",
  at: stageAt(markX(COMPONENTS.length - 1), rowY(COMPONENTS.length - 1) + 0.25),
  r: 3.4,
  ink: NEW,
  layer: 0,
  ...o,
});

/** In the film, a plan edit marks execution as moved while its beat lights the row. */
const markExecution = (): Fx =>
  Object.assign(
    (_drawing: Prim[], p: number) => {
      const a = swell(p, 0.1);
      return a > 0.01
        ? { add: [{ ...executionMoved({ alpha: a, glow: 10 * a }), key: "id/execution/moved:0" }] }
        : {};
    },
    { keys: ["id/execution"] },
  );

/** The identity column: each identity and what it hashes, then which edit moves which. */
function identities(sk: Sketch, s: State) {
  sk.keyed("id/name", () =>
    sk.text(...xy(RX, 3.3), "the identities", { size: 12, weight: 600, z: xyz(RX, 3.3) }),
  );
  COMPONENTS.forEach((c, i) => {
    const ry = rowY(i);
    sk.keyed(`id/${c.id}`, () => {
      sk.text(...xy(RX, ry), c.id, { size: 11.5, italic: true, z: xyz(RX, ry) });
      sk.text(...xy(RX, ry), HASHES[c.id], { size: 11.5, ink: "faint", dx: 112, z: xyz(RX, ry) });
      // The reader's plan edit, Remat on the Repeat, has moved execution identity.
      if (c.id === "execution" && s.policy === "Remat")
        sk.keyed("moved", () => sk.add(executionMoved()));
    });
  });
  EDITS.forEach((e, i) => {
    const ry = -5.6 - 1.15 * i;
    sk.keyed(`edit/${i}`, () => {
      sk.text(...xy(RX, ry), e.edit, { size: 11, z: xyz(RX, ry) });
      COMPONENTS.forEach((c, j) => {
        const moved = e.moves.includes(c.id);
        const [px, py, pz] = stageAt(markX(j), ry + 0.25);
        sk.add({
          kind: "dot",
          at: [px, py, pz],
          r: moved ? 3.4 : 2.2,
          ink: moved ? "ink" : "faint",
          layer: 0,
          hollow: !moved,
        });
      });
    });
  });
  sk.keyed("edit/head", () => {
    COMPONENTS.forEach((c, j) => {
      const [px, py, pz] = stageAt(markX(j), -4.5);
      sk.text(px, py, c.id[0]!.toUpperCase(), {
        z: pz,
        size: 11,
        italic: true,
        ink: "faint",
        anchor: "middle",
        dy: 4,
      });
    });
  });
}

/** A flat label's anchor at stage coordinates, and its depth. */
const xy = (rx: number, ry: number): [number, number] => {
  const p = stageAt(rx, ry);
  return [p[0], p[1]];
};
const xyz = (rx: number, ry: number) => stageAt(rx, ry)[2];

function draw(id: string, title: string, caption: string, s: State, opens = true): View {
  const sk = new Sketch();
  term(sk, s);
  if (s.backward) backward(sk, s);
  if (s.cost) cost(sk, s);
  if (s.plan) plan(sk, s);
  if (s.identity) identities(sk, s);
  if (opens && s.backward)
    for (const call of ["attn", "swiglu"] as const) {
      const cx = x(call, s);
      const yb = Y.back + 2.4;
      sk.open(
        `${call}T`,
        `${call}’s derived backward`,
        `fn/${call}`,
        [
          [cx - 2.3, yb + 1.5, 0],
          [cx + 2.3, yb + 1.5, 0],
          [cx + 2.3, yb - 1.5, 0],
          [cx - 2.3, yb - 1.5, 0],
        ],
        `back/${call === "attn" ? 0 : 1}/${call}T`,
      );
    }
  return { id, title, caption, sketch: sk, evidence: "model", ...STAGE_VIEW };
}

const BASE: State = {
  backward: false,
  cost: false,
  plan: false,
  identity: false,
  policy: "StoreAll",
  parted: false,
  grown: false,
};

const CAPTION = `Monad’s block, x + attn(rmsnorm x), then + swiglu(rmsnorm x), and what AMLO’s 0.0.14 design derives from it. That is the backward (the transpose of its linearization), its cost, an execution plan, and its identities. The cost is AMLO’s work rules as this window ports them, with each arithmetic, transcendental and compare operation counts as one unit, AMLO’s default weights, so it is an exact count of the program’s operations, not hardware time. Softmax is priced as AMLO’s builder composes it, from a row maximum, subtract, exp, sum and divide. Work is per layer for one sequence of ${fmt(SEQ)} tokens, declared for this figure. The column names each identity and marks which edit moves it. It shows no hash values: AMLO does not compute these identities yet, and the site makes none of its own.`;

const KEYFRAMES: Record<string, () => View> = {
  "fn/term": () => draw("fn/term", "The term", CAPTION, BASE),
  "fn/backward": () =>
    draw("fn/backward", "The backward functor", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
    }),
  "fn/cost": () =>
    draw("fn/cost", "The cost functor", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
    }),
  "fn/plan": () =>
    draw("fn/plan", "The plan functor", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
    }),
  "fn/remat": () =>
    draw("fn/remat", "The plan functor", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
      policy: "Remat",
    }),
  "fn/identity": () =>
    draw("fn/identity", "The identities", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
      identity: true,
    }),
  "fn/halves": () =>
    draw("fn/halves", "Functors compose", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
      identity: true,
      parted: true,
    }),
  "fn/block": () =>
    draw("fn/block", "One term, many shadows", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
      identity: true,
    }),
  "fn/block/remat": () =>
    draw("fn/block/remat", "One term, many shadows", CAPTION, {
      ...BASE,
      backward: true,
      cost: true,
      grown: true,
      plan: true,
      identity: true,
      policy: "Remat",
    }),
};

// ----------------------------------------------------------------------- derived views

/** A definition's operations as the derived view draws them: rope's two rotations are one. */
const OPS: Record<"attn" | "swiglu", ReadonlyArray<{ name: string; nodes: readonly string[] }>> = {
  attn: [
    { name: "·Wq", nodes: ["q0"] },
    { name: "rope", nodes: ["q.0", "q.1"] },
    { name: "·Wk", nodes: ["k0"] },
    { name: "rope", nodes: ["kr.0", "kr.1"] },
    { name: "·Wv", nodes: ["v"] },
    { name: "q·k", nodes: ["s"] },
    { name: "scale", nodes: ["s1"] },
    { name: "mask", nodes: ["s2"] },
    { name: "softmax", nodes: ["sm.max", "sm.shifted", "sm.exp", "sm.sum", "pr"] },
    { name: "p·v", nodes: ["o"] },
    { name: "·Wo", nodes: ["a"] },
  ],
  swiglu: [
    { name: "·W1", nodes: ["m.a"] },
    { name: "silu", nodes: ["m.g"] },
    { name: "·W3", nodes: ["m.b"] },
    { name: "times", nodes: ["m.h"] },
    { name: "·W2", nodes: ["m"] },
  ],
};

/**
 * A definition's derived view: its forward operations left to right, the
 * pullback of each beneath with its work, and a residual wire from each value a pullback reads.
 */
function derivedView(def: "attn" | "swiglu"): View {
  const w = blockWork(C);
  const g = w.program;
  const sk = new Sketch();
  const ops = OPS[def];
  // Both views keep one scale: attn's eleven operations span the stage, swiglu's five sit centred.
  const step = 4.4;
  const x0 = 4 + ((OPS.attn.length - ops.length) * step) / 2;
  const yf = -3;
  const yb = -10.5;
  const h = 2.2;
  const groupOf = new Map<string, number>();
  ops.forEach((op, i) => op.nodes.forEach((n) => groupOf.set(n, i)));
  // A layout (rope's pairs, the keys' relabel) is read through to the value it rearranges.
  const through = (v: string): string => {
    const n = g.nodes.find((m) => m.result === v);
    return n && n.prim.kind === "relabel" && n.operands.length ? through(n.operands.at(-1)!) : v;
  };
  ops.forEach((op, i) => {
    const cx = x0 + i * step;
    const pulls = w.steps.filter((s) => s.kind === "pull" && op.nodes.includes(s.of));
    const work = pulls.reduce((acc, s) => plus(acc, s.work), ZERO);
    sk.keyed(`f/${i}`, () => sk.op(cx, yf, op.name, { w: 3, h }));
    sk.keyed(`b/${i}`, () => {
      sk.op(cx, yb, op.name, { w: 3, h, sup: "T" });
      sk.text(cx, yb, plain(units(work)), { size: 11, ink: "faint", anchor: "middle", dy: 40 });
    });
    const reads = new Set(
      pulls.flatMap((s) => s.reads.map(through)).filter((r) => groupOf.has(r)),
    );
    for (const r of reads) {
      const rx = x0 + groupOf.get(r)! * step;
      sk.add({
        kind: "stroke",
        pts: [
          [rx, yf - h / 2 - 0.3, 0],
          [cx, yb + h / 2 + 0.3, 0],
        ],
        w: 0.9,
        ink: "faint",
        layer: 0,
        halo: false,
      });
    }
  });
  const left = x0 - 1.5;
  sk.text(left, yf, "forward", { size: 12, weight: 600, anchor: "end", dx: -12, dy: 4 });
  sk.text(left, yb, "backward", { size: 12, weight: 600, anchor: "end", dx: -12, dy: 4 });
  const rules =
    def === "attn" ? " Rope and softmax are priced as AMLO’s primitives compose them." : "";
  sk.text(left, yf, `Under each pullback, its work, one unit per scalar operation.${rules}`, {
    size: 11.5,
    italic: true,
    ink: "faint",
    dy: -40,
  });
  return {
    id: `fn/${def}`,
    title: `${def}’s derived backward`,
    caption: `${def}’s operations, left to right, and the pullback of each beneath, with its work in scalar operations at Monad’s sizes for one sequence of ${fmt(SEQ)} tokens. A faint wire runs from each value computed in this definition that a pullback reads; reads of the definition’s inputs and weights are not drawn. The work under a pullback leaves out the adds that sum the cotangents of a value read more than once; the band’s totals include them.`,
    sketch: sk,
    evidence: "model",
    ...STAGE_VIEW,
  };
}

/** The figure's views by id. */
export function figureView(id: string): View | undefined {
  if (id === "fn/attn") return derivedView("attn");
  if (id === "fn/swiglu") return derivedView("swiglu");
  return KEYFRAMES[id]?.();
}

// ------------------------------------------------------------------------------------- the film

function chapters(): Chapter[] {
  const cs = costs();
  const kinds = blockWork(C).forward;
  const st = residuals(C, "StoreAll");
  const rm = residuals(C, "Remat");
  const wr = work(C, "Remat");
  const casts = ampere().casts.length;
  const ledgerText = (lines: string[]) => (sk: Sketch) =>
    lines.forEach((line, i) => {
      const [px, py, pz] = stageAt(START, -17.6);
      sk.text(px, py, line, {
        z: pz,
        size: i ? 12 : 14,
        ...(i ? {} : { weight: 600 }),
        dy: i * 20,
      });
    });
  const beat = (o: Beat) => o;
  return [
    {
      id: "term",
      title: "The term",
      caption: CAPTION,
      ledger: ledgerText([
        "block[t,d,h,k,p,f]",
        "x becomes x + attn(rmsnorm x), then x + swiglu(rmsnorm x); attn rotates q and k by rope.",
        "Monad: d 256, h 4, k 64, f 768",
      ]),
      beats: [
        beat({
          ms: 3200,
          to: "fn/term",
          say: "One term holds Monad’s block, an attention half and a SwiGLU half on one stream.",
        }),
      ],
    },
    {
      id: "backward",
      title: "The backward functor",
      caption: CAPTION,
      ledger: ledgerText([
        "The derived backward",
        "vjp is the transpose of linearize. The backward is a program in the same form.",
        "Each of the 9 contractions is pulled back by 2; 9 values read twice accumulate their cotangents.",
      ]),
      beats: [
        beat({
          ms: 3600,
          to: "fn/backward",
          move: 0.7,
          say: "Its derivative, the backward, is another program in the same form.",
          fx: [
            light(["back/1", "back/0", "back/mid", "back/out", "back/in"], {
              from: 0.5,
              glow: 12,
            }),
          ],
        }),
        beat({
          ms: 2800,
          to: "fn/backward",
          say: "It runs from the output back to the input, through every piece in reverse.",
          fx: [light(["back/1"], { from: 0, to: 0.55 }), light(["back/0"], { from: 0.45 })],
        }),
        beat({
          ms: 2800,
          to: "fn/backward",
          say: "Each pullback reads values the forward computed, its residuals.",
          fx: [light(["residuals"], { glow: 10 })],
        }),
      ],
    },
    {
      id: "cost",
      title: "The cost functor",
      caption: CAPTION,
      ledger: ledgerText([
        "The cost fold, per layer, one sequence of 1,024 tokens",
        `attn ${fmt(units(cs.forward.get("attn")!))} forward, ${fmt(units(cs.backward.get("attn")!))} backward`,
        `swiglu ${fmt(units(cs.forward.get("swiglu")!))} forward, ${fmt(units(cs.backward.get("swiglu")!))} backward`,
        `block ${fmt(cs.total.forward)} forward, ${fmt(cs.total.backward)} backward`,
        `the block’s forward by kind: ${fmt(kinds.arith)} arithmetic, ${fmt(kinds.transcendental)} transcendental, ${fmt(kinds.compare_select)} compare or select`,
      ]),
      beats: [
        beat({
          ms: 3400,
          to: "fn/cost",
          move: 0.75,
          say: "AMLO’s work rules price every node exactly, one unit per scalar operation.",
        }),
        beat({
          ms: 3000,
          to: "fn/cost",
          say: "Each contraction becomes two in the backward, about twice the forward’s work.",
          fx: [
            mark(["cost/attn/b", "cost/swiglu/b"], { pad: 0.2 }),
            light(["back/0/attnT", "back/1/swigluT"], { ink: NEW }),
          ],
        }),
      ],
    },
    {
      id: "plan",
      title: "The plan functor",
      caption: CAPTION,
      ledger: ledgerText([
        "The plan: ampere-bf16",
        `bf16 contractions accumulating in f32; f32 norms, softmax, SiLU and stream; ${casts} casts as nodes, applied before the backward is derived.`,
        `StoreAll keeps ${mb(st.bytes)} per layer; Remat keeps ${mb(rm.bytes)} and recomputes ${fmt(wr.plan)} operations.`,
        "Casts are plan work. AMLO’s work rules give them no price, so this plan counts none.",
        "A plan edit moves execution identity, never structure. Mathematical work does not move.",
      ]),
      beats: [
        beat({
          ms: 3200,
          to: "fn/plan",
          move: 0.7,
          say: "A plan says how to run the term, not what it computes.",
        }),
        beat({
          ms: 3000,
          to: "fn/plan",
          say: `Contractions run in bf16 and add in f32; norms, softmax and the stream stay f32, with ${casts} casts.`,
          fx: [mark(["plan/attn", "plan/swiglu", "plan/n1", "plan/n2"], { pad: 0.25 })],
        }),
        beat({
          ms: 4200,
          to: "fn/remat",
          move: 0.6,
          say: `Remat keeps only the stream, ${mb(rm.bytes)} instead of ${mb(st.bytes)} per layer, and recomputes what the backward reads.`,
          fx: [
            // The forward runs again inside the backward, and feeds each pullback.
            light(["term/0/n1", "term/0/attn", "term/1/n2", "term/1/swiglu"], {
              ink: NEW,
              from: 0.25,
              to: 0.7,
            }),
            light(["remat"], { ink: NEW, from: 0.5, glow: 10 }),
          ],
        }),
        beat({
          ms: 2800,
          to: "fn/remat",
          say: "Recomputing is plan work, and the budget charges it. The mathematics does not move.",
          fx: [light(["cost"], { from: 0.1, glow: 10 })],
        }),
      ],
    },
    {
      id: "identity",
      title: "The identities",
      caption: CAPTION,
      ledger: ledgerText([
        "The identities",
        // A line too long for the band goes on in a second line, cut after a semicolon.
        ...COMPONENTS.flatMap((c) => {
          const line = `${c.id}: ${c.hashes}`;
          const cut = line.length > 96 ? line.lastIndexOf("; ") : -1;
          return cut > 0 ? [line.slice(0, cut + 1), line.slice(cut + 2)] : [line];
        }),
      ]),
      beats: [
        beat({
          ms: 3400,
          to: "fn/identity",
          move: 0.7,
          say: "Four component identities, and execution, which adds the plan. Search deduplicates on the first three.",
        }),
        ...EDITS.filter((_, i) => [0, 1, 4, 7].includes(i)).map((e) =>
          beat({
            ms: 2200,
            to: "fn/identity",
            say: e.say,
            fx: [
              mark([`edit/${EDITS.indexOf(e)}`], { pad: 0.2 }),
              light(
                e.moves.map((m) => `id/${m}`),
                { ink: NEW },
              ),
              ...(e.edit.startsWith("Plan:") ? [markExecution()] : []),
            ],
          }),
        ),
      ],
    },
    {
      id: "composition",
      title: "Functors compose",
      caption: CAPTION,
      ledger: ledgerText([
        "Composition",
        "The block is two pieces composed; its backward, cost and plan are its pieces’ joined.",
        "The backward runs the second piece’s pullback first. Costs add. Plans join. The identities hash the whole block.",
      ]),
      beats: [
        beat({
          ms: 3400,
          to: "fn/halves",
          move: 0.6,
          say: "The block is two pieces composed, an attention half and a SwiGLU half.",
        }),
        beat({
          ms: 3600,
          to: "fn/block",
          move: 0.6,
          say: "Compose them, and the backward, cost and plan compose too, the backward in reverse order and the costs added.",
          // The light falls before the film ends, so the drawing rests in ink.
          fx: [light(["back", "cost", "plan"], { from: 0.5, to: 0.9, glow: 8 })],
        }),
      ],
    },
  ];
}

/** Figure C's film, from Monad's block view. At rest, the reader sets which bands show. */
export const { film, chapterAt, chapterEnd, frameAt, OPENS_FROM } = makeFilm({
  chapters: () => {
    const list = chapters();
    // The film's last view follows the reader's policy, so the switch works at rest.
    const lastChapter = list.at(-1)!;
    lastChapter.beats.at(-1)!.to = (o) => (o.policy === "Remat" ? "fn/block/remat" : "fn/block");
    return list;
  },
  opensFrom: "block",
  honesty: HONESTY,
  sayX: START,
  sayY: -19.4,
});

/** The bands the reader can hide at rest, by the keys they are drawn under. */
export const WALLS = [
  { id: "backward", label: "Backward", keys: ["back", "residuals", "remat"] },
  { id: "cost", label: "Cost", keys: ["cost"] },
  { id: "plan", label: "Plan", keys: ["plan"] },
  { id: "identity", label: "Identity", keys: ["id", "edit"] },
] as const;
