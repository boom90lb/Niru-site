/**
 * Monad, drawn as AMLO's 0.0.14 design writes it:
 * the `gpt` family with Monad's sizes and the head tied to the embedding.
 *
 *   rmsnorm[n]            param #g:[n]
 *   attn[t,d,h,k,p]       d := h·k, k := 2·p; params #Wq,#Wk,#Wv:[d,h,k], #Wo:[h,k,d]
 *                         q,k = rope(x·Wq), rope(x·Wk); v = x·Wv; s = q·k/√k, masked where key > query
 *                         y = (softmax(s)·v)·Wo
 *   swiglu[d,f]           params #W1,#W3:[d,f], #W2:[f,d]; (silu(x·W1) ⊙ x·W3)·W2
 *   block[t,d,h,k,p,f]    x ← x + attn(rmsnorm x, rope); x ← x + swiglu(rmsnorm x)
 *   gpt                   param #E:[V,D]; x = gather(E, inputs);
 *                         x = Repeat(block, L; stacked; closure rope)(x)
 *                         loss = cross_entropy(contract_D(rmsnorm x, E), labels, loss_mask)
 *
 * Each view is one definition. A box that is a call opens into the view of its definition.
 */
import type { Ink } from "./engine.ts";
import { LANE, type Lane, Sketch } from "./kit.ts";
import { trainingStep } from "./step.ts";
import { fmt, type View } from "./view.ts";

export const MONAD = { V: 8192, D: 256, L: 64, H: 4, K: 64, F: 768 } as const;

/** Parameters of the gpt family, counted from the definitions above. */
export function parameters(
  c: { V: number; D: number; L: number; H: number; K: number; F: number } = MONAD,
): number {
  const rmsnorm = (n: number) => n;
  const attn = 3 * c.D * c.H * c.K + c.H * c.K * c.D;
  const swiglu = 2 * c.D * c.F + c.F * c.D;
  const block = 2 * rmsnorm(c.D) + attn + swiglu;
  return c.V * c.D + c.L * block + rmsnorm(c.D);
}

const T: Lane = { name: "t", ink: "ink" };
const Tk: Lane = { name: "t′", ink: "ink" };
const Dl: Lane = { name: "d", ink: "ochre" };
const Hl: Lane = { name: "h", ink: "indigo" };
const Kl: Lane = { name: "k", ink: "cinnabar" };
const Fl: Lane = { name: "f", ink: "cinnabar" };
const Vl: Lane = { name: "V", ink: "indigo" };

function step(): View {
  return trainingStep({
    id: "step",
    model: "model",
    note: "gpt, Monad’s sizes",
    theta: "θ: #E, #g, #Wq … #W2",
    loss: "loss",
    update: "clip, then AdamW",
    caption:
      "In AMLO’s 0.0.14 design one training step is one term. It holds the model and its loss, a gradient derived from them, and an update that clips the gradient and applies AdamW. The gradient is computed from the model’s own definitions and is never stored. Open the model.",
  });
}

function model(): View {
  const s = new Sketch();
  const { V, D, L, H, K } = MONAD;
  // Token ids, an integer datatype, select rows of E.
  s.wire(
    [
      [-9, 0],
      [-1.4, 0],
    ],
    "ink",
  );
  s.arrows(-9, -1.4, 0, "ink");
  s.text(-9, 0, "inputs", { italic: true, anchor: "end", dx: -6, dy: 4 });
  s.slab(0, 0, 2.8, 3.2, { chip: true, label: "E", size: 16 });
  s.bundle(
    [
      [1.4, 0],
      [6, 0],
    ],
    [T, Dl],
    { label: "over", labelDx: 14 },
  );

  // The Repeat: a strip of L panels, one per layer, folded like a concertina.
  const full = new Set([0, 1, 2, 3, L - 1]);
  const W = 4.4;
  const C = 0.19;
  const depth = 1.4;
  const top = 3.2;
  const xs: number[] = [6];
  for (let i = 0; i < L; i++) xs.push(xs[i]! + (full.has(i) ? W : C));
  const zAt = (edge: number) => (edge % 2 === 0 ? 0 : -depth);
  const at = (i: number, u: number, y: number): [number, number, number] => [
    xs[i]! + u * (xs[i + 1]! - xs[i]!),
    y,
    zAt(i) + u * (zAt(i + 1) - zAt(i)),
  ];
  const outline = (y: number) => xs.map((x, e) => [x, y, zAt(e)] as [number, number, number]);
  for (let i = 0; i < L; i++) {
    if (!full.has(i)) continue;
    const face = [at(i, 0, top), at(i, 1, top), at(i, 1, -top), at(i, 0, -top)];
    s.add({ kind: "face", pts: face, layer: 0, fill: "paper", edge: "ink", w: 1.2, under: true });
    s.open(`layer-${i}`, `layer ${i}`, "block", face);
  }
  // The folded run between layer 4 and layer 64: one fold per layer.
  for (let e = 4; e <= L - 1; e++)
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
  s.add({
    kind: "stroke",
    pts: outline(top).slice(4, L),
    w: 0.8,
    ink: "ink",
    layer: 0,
    halo: false,
  });
  s.add({
    kind: "stroke",
    pts: outline(-top).slice(4, L),
    w: 0.8,
    ink: "ink",
    layer: 0,
    halo: false,
  });
  // The residual stream through every layer, drawn finer where the run is folded.
  for (const [y, ink] of [
    [LANE / 2, "ink"],
    [-LANE / 2, "ochre"],
  ] as const) {
    const pts = xs.map((x, e) => [x, y, zAt(e)] as [number, number, number]);
    s.add({ kind: "stroke", pts: pts.slice(0, 5), w: 1.5, ink, layer: 0, halo: false });
    s.add({ kind: "stroke", pts: pts.slice(4, L), w: 0.6, ink: "faint", layer: 0, halo: false });
    s.add({ kind: "stroke", pts: pts.slice(L - 1), w: 1.5, ink, layer: 0, halo: false });
  }
  // A print of the block on each open panel: two norms and two calls on the stream.
  for (const i of full) {
    for (const [u, kind] of [
      [0.16, "norm"],
      [0.36, "box"],
      [0.6, "norm"],
      [0.8, "box"],
    ] as const) {
      const [x, y, z] = at(i, u, 1.95);
      if (kind === "norm") {
        const ring: Array<[number, number, number]> = [];
        for (let j = 0; j <= 20; j++) {
          const a = (j / 20) * 2 * Math.PI;
          ring.push(at(i, u + Math.cos(a) * 0.05, 1.95 + Math.sin(a) * 0.32));
        }
        s.add({ kind: "stroke", pts: ring, w: 0.9, ink: "ink", layer: 0, halo: false });
        void x;
        void y;
        void z;
      } else {
        const a = at(i, u - 0.065, 2.4);
        const b = at(i, u + 0.065, 1.5);
        s.add({
          kind: "face",
          pts: [
            [a[0], a[1], a[2]],
            [b[0], a[1], b[2]],
            [b[0], b[1], b[2]],
            [a[0], b[1], a[2]],
          ],
          layer: 0,
          fill: "paper",
          edge: "ink",
          w: 0.9,
        });
      }
    }
    s.add({
      kind: "stroke",
      pts: [at(i, 0.06, 0.55), at(i, 0.16, 1.5)],
      w: 1,
      ink: "faint",
      layer: 0,
      halo: false,
    });
    s.add({
      kind: "stroke",
      pts: [at(i, 0.44, 1.95), at(i, 0.5, 0.55)],
      w: 1,
      ink: "faint",
      layer: 0,
      halo: false,
    });
    s.add({
      kind: "stroke",
      pts: [at(i, 0.52, 0.55), at(i, 0.6, 1.5)],
      w: 1,
      ink: "faint",
      layer: 0,
      halo: false,
    });
    s.add({
      kind: "stroke",
      pts: [at(i, 0.88, 1.95), at(i, 0.95, 0.55)],
      w: 1,
      ink: "faint",
      layer: 0,
      halo: false,
    });
    const [cx, , cz] = at(i, 0.5, -top);
    s.text(cx, -top, `${i}`, { z: cz, size: 11, ink: "faint", anchor: "middle", dy: 16 });
  }
  const mid = (xs[4]! + xs[L - 1]!) / 2;
  s.text(mid, -top, `4 … ${L - 2}`, { size: 11, ink: "faint", anchor: "middle", dy: 16 });
  s.text(
    (xs[0]! + xs[L]!) / 2,
    -top - 2.6,
    `Repeat(block, L = ${L}): one body, its parameters stacked`,
    {
      size: 12,
      italic: true,
      anchor: "middle",
      dy: 22,
    },
  );
  // The rope table is computed once and closed over by every layer.
  s.slab(9, 7.6, 8, 2.4, { label: "rope table", size: 12 });
  s.add({
    kind: "stroke",
    pts: [[9, 6.4, 0], at(0, (9 - xs[0]!) / (xs[1]! - xs[0]!), top)],
    w: 1.3,
    ink: "faint",
    layer: 0,
  });
  s.text(9.4, 5, "rope [t p 2]", { italic: true, ink: "faint", size: 11, dx: 4 });

  const end = xs[L]!;
  s.bundle(
    [
      [end, 0],
      [end + 4.4, 0],
    ],
    [T, Dl],
  );
  s.norm(end + 2.4, 0, { r: 0.85 });
  s.bundle(
    [
      [end + 4.4, 0],
      [end + 6.7, 0],
    ],
    [T, Dl],
  );
  const hx = end + 8.6;
  s.slab(hx, 0, 3.8, 3.4, { chip: true, label: "head", size: 14 });
  // Tied: the head reads the embedding's own parameter.
  s.add({ kind: "dot", at: [0, 1.6, 0], r: 2.6, ink: "faint", layer: 0, hollow: true });
  s.add({ kind: "dot", at: [hx, 1.7, 0], r: 2.6, ink: "faint", layer: 0, hollow: true });
  s.add({
    kind: "stroke",
    pts: [
      [0, 1.75, 0],
      [0, 10.4, 0],
      [hx, 10.4, 0],
      [hx, 1.85, 0],
    ],
    w: 1.3,
    ink: "indigo",
    layer: 0,
  });
  s.text(0.6, 10.4, `#E [V, d] is read twice: the embedding is also the head (tied)`, {
    size: 12,
    italic: true,
    ink: "indigo",
    dy: -8,
  });
  const ce = hx + 10.4;
  s.bundle(
    [
      [hx + 1.9, 0],
      [ce - 3.8, 0],
    ],
    [T, Vl],
    { label: "over", labelDx: 14 },
  );
  s.slab(ce, 0, 7.6, 3.4, { label: "cross-entropy", size: 12.5 });
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
      [ce + 3.8, 0],
      [ce + 7.6, 0],
    ],
    "ink",
  );
  s.text(ce + 7.6, 0, "loss", { italic: true, dx: 6, dy: 4 });
  return {
    id: "model",
    title: "Monad with its loss",
    caption: `Monad has ${L} layers of one block, with width ${D}, ${H} heads of ${K}, a vocabulary of ${fmt(V)} and ${fmt(parameters())} parameters, counted from the definitions. The embedding E is also the output head (tied). Open any layer.`,
    sketch: s,
    yaw: -0.3,
    pitch: 0.42,
  };
}

/**
 * An edit drawn into a definition's full view, for a step of the search that opens it: what the
 * move changed is printed in verdigris, the ink of a rewrite.
 */
export interface Edit {
  id: string;
  title: string;
  caption: string;
}

const NEW: Ink = "verdigris";

export function block(
  o: Partial<Edit> & {
    /** The library's rms_norm after attention, inserted. */
    added?: boolean;
    /** The norm before attention, deleted. */
    unnormed?: boolean;
  } = {},
): View {
  const s = new Sketch();
  s.bundle(
    [
      [-5, 0],
      [0, 0],
    ],
    [T, Dl],
    { label: "start" },
  );
  const half = (
    x0: number,
    name: string,
    child: string,
    ports: string,
    w: number,
    edit: { added?: boolean; unnormed?: boolean } = {},
  ) => {
    s.copy(x0, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
    s.bundle(
      [
        [x0, 0],
        [x0 + 3.4, 4],
      ],
      [T, Dl],
    );
    s.bundle(
      [
        [x0 + 3.4, 4],
        [x0 + 8 - w / 2, 4],
      ],
      [T, Dl],
    );
    if (!edit.unnormed) {
      s.norm(x0 + 4.4, 4);
      s.port(x0 + 4.4, 3.0, "#g");
    }
    const bx = x0 + 7.4;
    const face = s.slab(bx + 0.6, 4, w, 3.8, { chip: true, label: name, size: 16 });
    s.open(child, name, child, face);
    s.port(bx + 0.6, 2.1, ports);
    // An inserted norm sits on the wire after the call, and the branch returns after it.
    const extra = edit.added ? 2.6 : 0;
    if (edit.added) {
      s.bundle(
        [
          [bx + 0.6 + w / 2, 4],
          [bx + 0.6 + w / 2 + extra, 4],
        ],
        [T, Dl],
      );
      s.norm(bx + 0.6 + w / 2 + 1.3, 4, { ink: NEW, plain: true });
      s.text(bx + 0.6 + w / 2 + 1.3, 3.0, "rms_norm", {
        size: 11,
        italic: true,
        ink: NEW,
        anchor: "middle",
        dy: 24,
      });
    }
    const out = bx + 0.6 + w / 2 + extra;
    s.bundle(
      [
        [out, 4],
        [out + 2.4, 4],
        [out + 5, 0],
      ],
      [T, Dl],
    );
    s.bundle(
      [
        [x0, 0],
        [out + 5, 0],
      ],
      [T, Dl],
    );
    s.plus(out + 5, 0);
    return out + 5;
  };
  const a = half(0, "attn", "attn", "#Wq #Wk #Wv #Wo", 5.6, o);
  s.add({
    kind: "stroke",
    pts: [
      [8, 9.2, 0],
      [8, 5.9, 0],
    ],
    w: 1.3,
    ink: "faint",
    layer: 0,
  });
  s.text(8, 9.2, "rope", { italic: true, ink: "faint", size: 11, anchor: "middle", dy: -6 });
  s.bundle(
    [
      [a, 0],
      [a + 2.6, 0],
    ],
    [T, Dl],
  );
  const b = half(a + 2.6, "swiglu", "swiglu", "#W1 #W3 #W2", 5.2);
  s.bundle(
    [
      [b, 0],
      [b + 4.4, 0],
    ],
    [T, Dl],
    { label: "end" },
  );
  return {
    id: o.id ?? "block",
    title: o.title ?? "One block, the same for every layer",
    caption:
      o.caption ??
      `block[t,d,h,k,p,f]: x becomes x + attn(rmsnorm x, rope), then x + swiglu(rmsnorm x). All ${MONAD.L} layers run this one definition, each with its own parameters. Open attention or SwiGLU.`,
    sketch: s,
    yaw: -0.24,
    pitch: 0.34,
  };
}

function attn(): View {
  const s = new Sketch();
  const heads = MONAD.H;
  const dz = 3.6;
  s.bundle(
    [
      [-3, 0],
      [2, 0],
    ],
    [T, Dl],
    { label: "start" },
  );
  s.copy(2, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
  const rows = [
    [6, "q", T],
    [1, "k", Tk],
    [-4, "v", Tk],
  ] as const;
  for (const [y] of rows)
    s.bundle(
      [
        [2, 0],
        [5.7, y],
      ],
      [T, Dl],
    );
  for (const [y, sub, tok] of rows) {
    // The axes are named once, on the values, which no rotation covers.
    s.bundle(
      [
        [9.1, y],
        [14.6, y],
      ],
      [tok, Hl, Kl],
      sub === "v" ? { label: "over", labelDx: 12 } : {},
    );
    s.slab(7.4, y, 3.4, 3.8, { chip: true, label: "W", sub });
  }
  // Queries and keys are rotated by the rope table before the heads split.
  const rx = 11.85;
  const rr = LANE + 0.45;
  s.hexagon(rx, 6, "rope", { r: rr });
  s.hexagon(rx, 1, "rope", { r: rr });
  s.open("rope", "the rope pairs", "lat/rope", hexFace(rx, 6, 0, rr));
  s.add({
    kind: "stroke",
    pts: [
      [rx, 10.6, 0],
      [rx, 6 + rr, 0],
    ],
    w: 1.2,
    ink: "faint",
    layer: 0,
  });
  s.add({
    kind: "stroke",
    pts: [
      [rx, 6 - rr, 0],
      [rx, 1 + rr, 0],
    ],
    w: 1.2,
    ink: "faint",
    layer: 0,
  });
  s.text(rx, 10.6, "rope [t p 2]", {
    italic: true,
    ink: "faint",
    size: 11,
    anchor: "middle",
    dy: -6,
  });
  // One sheet per head: h is tiled.
  for (let k = heads - 1; k >= 0; k--) {
    const z = -k * dz;
    const layer = -k;
    s.sheet(14.1, -6.4, 37.7, 8.6, { z, layer, label: `head ${k + 1}` });
    s.bundle(
      [
        [14.6, 6],
        [18.6, 6],
      ],
      [T, Kl],
      { z, layer },
    );
    s.bundle(
      [
        [14.6, 1],
        [18.6, 1],
      ],
      [Tk, Kl],
      { z, layer },
    );
    s.cup(18.6, 6 - LANE / 2, 1 - LANE / 2, "cinnabar", { z, layer });
    s.bundle(
      [
        [18.6, 6 + LANE / 2],
        [21, 3.5 + LANE / 2],
        [30.5, 3.5 + LANE / 2],
      ],
      [T],
      { z, layer },
    );
    s.bundle(
      [
        [18.6, 1 + LANE / 2],
        [21, 3.5 - LANE / 2],
        [30.5, 3.5 - LANE / 2],
      ],
      [Tk],
      { z, layer },
    );
    s.op(22.5, 3.5, "÷ k", { z, layer, sup: "1/2" });
    s.mask(25.6, 3.5, { z, layer });
    if (k === 0) s.open("causal", "the causal mask", "lat/causal", hexFace(25.6, 3.5, z, 1.3));
    s.softmax(28.4, 3.5, { z, layer });
    s.bundle(
      [
        [14.6, -4],
        [30.5, -4],
      ],
      [Tk, Kl],
      { z, layer },
    );
    s.cup(30.5, 3.5 - LANE / 2, -4 + LANE / 2, "ink", { z, layer });
    s.bundle(
      [
        [30.5, 3.5 + LANE / 2],
        [33.5, 0.6 + LANE / 2],
        [37.4, 0.6 + LANE / 2],
      ],
      [T],
      { z, layer },
    );
    s.bundle(
      [
        [30.5, -4 - LANE / 2],
        [33.5, 0.6 - LANE / 2],
        [37.4, 0.6 - LANE / 2],
      ],
      [Kl],
      { z, layer },
    );
  }
  const back = -(heads - 1) * dz;
  for (const [x, y] of [
    [14.6, 6],
    [14.6, 1],
    [14.6, -4],
    [37.4, 0.6],
  ] as const)
    s.add({
      kind: "stroke",
      pts: [
        [x, y, 0],
        [x, y, back],
      ],
      w: 1.2,
      ink: "indigo",
      layer: 0,
      dash: [2, 3],
    });
  s.text(14.1, 8.6, `h = ${heads}: one sheet per head`, {
    z: back,
    italic: true,
    ink: "indigo",
    dy: -8,
  });
  s.bundle(
    [
      [37.4, 0.6],
      [39.1, 0.6],
    ],
    [T, Hl, Kl],
  );
  s.slab(40.8, 0.6, 3.4, 3.8, { chip: true, label: "W", sub: "o" });
  s.bundle(
    [
      [42.5, 0.6],
      [47, 0.6],
    ],
    [T, Dl],
    { label: "end" },
  );
  return {
    id: "attn",
    title: "Attention, one sheet per head",
    caption:
      "attn[t,d,h,k,p]: queries and keys are rotated by the rope table; the scores q·k, divided by the square root of k, are masked where the key comes after the query, put through a softmax over the keys t′ and summed against the values. The head axis h is tiled, so this runs once per head, each on its own sheet. The output map contracts h and k back to d.",
    sketch: s,
    yaw: -0.38,
    pitch: 0.5,
  };
}

export function swiglu(
  o: Partial<Edit> & {
    /** The activation, if a move replaced SiLU. */
    act?: string;
    /** A note on f, if a move set its width. */
    f?: string;
    /** A note on #W2, if a move changed its binding. */
    w2?: string;
  } = {},
): View {
  const s = new Sketch();
  const Fx: Lane = o.f ? { ...Fl, ink: NEW } : Fl;
  s.bundle(
    [
      [-3, 0],
      [1.5, 0],
    ],
    [T, Dl],
    { label: "start" },
  );
  s.copy(1.5, [LANE / 2, -LANE / 2], ["ink", "ochre"]);
  for (const [y, sub] of [
    [3.2, "1"],
    [-3.2, "3"],
  ] as const) {
    s.bundle(
      [
        [1.5, 0],
        [4.3, y],
      ],
      [T, Dl],
    );
    s.slab(6, y, 3.4, 3.4, { chip: true, label: "W", sub });
  }
  s.bundle(
    [
      [7.7, 3.2],
      [14, 3.2],
    ],
    [T, Fx],
    { label: "over", labelDx: 14 },
  );
  s.op(10.8, 3.2, o.act ?? "silu", o.act ? { ink: NEW } : {});
  s.bundle(
    [
      [7.7, -3.2],
      [14, -3.2],
    ],
    [T, Fx],
    { label: "over", labelDx: 14 },
  );
  if (o.f) s.text(10.8, -4.1, o.f, { size: 11, italic: true, ink: NEW, anchor: "middle", dy: 18 });
  // Elementwise product: the paper's dashed wire between the two arrays.
  s.add({
    kind: "stroke",
    pts: [
      [14, 3.2 - LANE, 0],
      [14, -3.2 + LANE, 0],
    ],
    w: 1.2,
    ink: "faint",
    layer: 0,
    dash: [4, 4],
    halo: false,
  });
  s.bundle(
    [
      [14, 3.2],
      [16.8, 0],
    ],
    [T, Fx],
  );
  s.bundle(
    [
      [14, -3.2],
      [16.8, 0],
    ],
    [T, Fx],
  );
  s.text(14, 0, "× elementwise", {
    size: 11,
    italic: true,
    ink: "faint",
    dx: -8,
    dy: 4,
    anchor: "end",
  });
  s.slab(18.5, 0, 3.4, 3.4, { chip: true, label: "W", sub: "2", ...(o.w2 ? { ink: NEW } : {}) });
  if (o.w2) {
    s.add({ kind: "dot", at: [18.5, -1.7, 0], r: 2.6, ink: NEW, layer: 0, hollow: true });
    s.text(18.5, -1.7, o.w2, { size: 11, italic: true, ink: NEW, anchor: "middle", dy: 20 });
  }
  s.bundle(
    [
      [20.2, 0],
      [24, 0],
    ],
    [T, Dl],
    { label: "end" },
  );
  return {
    id: o.id ?? "swiglu",
    title: o.title ?? "SwiGLU",
    caption:
      o.caption ??
      `swiglu[d,f]: two maps from d to f, one passed through SiLU, multiplied element by element, then a map from f back to d. For Monad, f = ${MONAD.F}.`,
    sketch: s,
    yaw: -0.24,
    pitch: 0.34,
  };
}

/** The face a small glyph opens from, for hit testing and the zoom. */
function hexFace(x: number, y: number, z = 0, r = 1.4): Array<[number, number, number]> {
  return [
    [x - r, y + r, z],
    [x + r, y + r, z],
    [x + r, y - r, z],
    [x - r, y - r, z],
  ];
}

const builders: Record<string, () => View> = {
  step,
  model,
  block: () => block(),
  attn,
  swiglu: () => swiglu(),
};

export function view(id: string): View | undefined {
  return builders[id]?.();
}
