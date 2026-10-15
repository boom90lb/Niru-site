/**
 * Figure B as a film: the search's moves on Monad's term, told in ten chapters that play by
 * themselves. A chapter is a row of beats; each beat carries the drawing into one view of the
 * figure, says one plain sentence, and lights what it acts on. The whole film is one pure
 * function from time to a frame, so pausing leaves a still image and every frame can be
 * checked like a drawing.
 *
 * It is an illustration under the registry's rules, not a recorded search, and says so in
 * every frame. Each number comes from moves.ts.
 */
import { type Ink, type Prim, textWidth, type V3 } from "./engine.ts";
import { view } from "./models.ts";
import {
  chance,
  type Class,
  CLASSES,
  commuting,
  counts,
  deepseek,
  GATES,
  moves,
  odds,
  POPULATION,
  type Proposal,
  SEED,
  sites,
  tournament,
  weights,
} from "./moves.ts";
import {
  at,
  BAND,
  band,
  type Beat,
  type Chapter,
  face,
  type Fx,
  light,
  makeFilm,
  mark,
  particles,
  quad,
  seal,
  smooth,
  swell,
  wash,
} from "./reel.ts";
import {
  admissionLedger,
  DECK_X,
  DRAWN,
  GATE_ROW,
  gateX,
  ILLUSTRATION,
  LANE_AT,
  NEW,
  paramNote,
  populationLedger,
  product,
  siteKeys,
  sitesLedger,
  stageAt,
} from "./rewrite.ts";

/** The figure's own view builder, which the registry in models.ts calls for its "rw/" ids. */
export { view as figureView } from "./rewrite.ts";

/**
 * Light across the strip's panels: a sweep from the first layer to the last when the change
 * reaches all of them, a spotlight on each selected layer otherwise.
 */
const sweep =
  (L: number, selected: "all" | readonly number[], o: { from?: number; to?: number } = {}): Fx =>
  (drawing, p) => {
    const add: Prim[] = [];
    const q = band(p, o.from ?? 0.1, o.to ?? 0.85);
    for (let i = 0; i < L; i++) {
      const pts = face(drawing, `panel${i}:0`);
      if (!pts) continue;
      let a: number;
      if (selected === "all") {
        const d = i / (L - 1) - (q * 1.3 - 0.15);
        a = 0.2 * Math.exp(-(d * d) / 0.004);
      } else if (selected.includes(i)) {
        // A selected layer may be folded to a sliver: a beam of light marks it, and a marker
        // over the strip points it out.
        const s = swell(q, 0, 1, 0.3, 0);
        const [a0, a1, b1, b0] = pts as [V3, V3, V3, V3];
        const cx = (a0[0] + a1[0]) / 2;
        const cz = (a0[2] + a1[2]) / 2;
        const top = a0[1];
        const bottom = b0[1];
        void b1;
        add.push(
          wash(
            [
              [cx - 0.5, top + 1.2, cz],
              [cx + 0.5, top + 1.2, cz],
              [cx + 0.5, bottom, cz],
              [cx - 0.5, bottom, cz],
            ],
            0.22 * s,
          ),
          {
            kind: "face",
            pts: [
              [cx - 0.42, top + 2.1, cz],
              [cx + 0.42, top + 2.1, cz],
              [cx, top + 1.4, cz],
            ],
            layer: 1,
            fill: "paper",
            edge: NEW,
            w: 1.4,
            alpha: s,
            glow: 12 * s,
          },
        );
        continue;
      } else a = 0;
      if (a > 0.01) add.push(wash(pts, a));
    }
    return { add };
  };

/**
 * Sparks of light gathering into each place a node arrives, or scattering from each place one
 * leaves. They rise from above it and fade before they reach the glyph, so they never cross
 * its lettering.
 */
/**
 * A gate's light as a mark goes by, d units past it: it rises from nothing as the mark comes
 * near, peaks at `at`, and fades slowly behind the mark, so a passed gate stays lit a moment.
 */
const passed = (d: number, at: number, width: number) =>
  d < at
    ? Math.exp(-((d - at) ** 2) / width) * smooth(band(d, at - 2, at - 0.6))
    : Math.exp(-(d - at) / 9);

/** A proposal on the admission lane: a mark that runs through the gates, which light as it passes. */
const token =
  (o: { from?: number; to?: number } = {}): Fx =>
  (drawing, p) => {
    const q = smooth(band(p, o.from ?? 0.05, o.to ?? 0.9));
    const x0 = LANE_AT.x - 0.4;
    const x1 = gateX(GATES.length - 1) + 1.2;
    const x = x0 + (x1 - x0) * q;
    // Each gate's light rises from nothing as the mark comes near it, and fades behind it.
    const lit = GATES.map((_, k) => passed(x - gateX(k), 0.3, 0.9));
    const shown = smooth(band(p, 0, 0.06)) * (1 - smooth(band(p, 0.9, 0.98)));
    return {
      drawing: drawing.map((g) => {
        const m = g.key?.match(/^lane\/gate(\d)/);
        const a = m ? lit[+m[1]!]! : 0;
        return a > 0.02 ? { ...g, glow: 16 * a, glowInk: NEW } : g;
      }),
      add:
        shown > 0.001
          ? [
              {
                kind: "dot",
                at: [x, LANE_AT.y, 0],
                r: 4.6,
                ink: NEW,
                layer: 1,
                alpha: shown,
                glow: 12 * shown,
              },
            ]
          : [],
    };
  };

/** Gates light as a proposal's mark, moved by the rewrite, passes them. */
const passing =
  (mark: string, ink: Ink = NEW): Fx =>
  (drawing) => {
    const dot = drawing.find((q) => q.key === `${mark}:0` && q.kind === "dot");
    if (!dot || dot.kind !== "dot") return {};
    return {
      drawing: drawing.map((g) => {
        const m = g.key?.match(/^gate(\d)/);
        if (!m || g.kind !== "face") return g;
        const a = passed(dot.at[0] - GATE_ROW.x(+m[1]!), 0.4, 2.2);
        return a > 0.02 ? { ...g, glow: 18 * a, glowInk: ink } : g;
      }),
    };
  };

/** The child joins the population as its youngest member, and the oldest retires. */
const join: Fx = (drawing, p) => {
  const q = smooth(band(p, 0.1, 0.8));
  const { y } = LANE_AT;
  const x = gateX(GATES.length - 1) + 1.2 + (DECK_X - gateX(GATES.length - 1) - 1.2) * q;
  const z = 0.6 * q;
  return {
    drawing: drawing.map((g) => {
      if (!g.key?.startsWith("lane/card4:")) return g;
      const lift = (v: V3): V3 => [v[0] + 0.8 * q, v[1] + 1.6 * q, v[2]];
      return g.kind === "face" ? { ...g, pts: g.pts.map(lift), alpha: 1 - q } : g;
    }),
    add: [
      {
        kind: "face",
        pts: [
          [x - 0.75, y + 1.05, z],
          [x + 0.75, y + 1.05, z],
          [x + 0.75, y - 1.05, z],
          [x - 0.75, y - 1.05, z],
        ],
        layer: 1,
        fill: "paper",
        edge: NEW,
        w: 1.6,
        alpha: smooth(band(p, 0.05, 0.25)),
        glow: 14,
      },
    ],
  };
};

// ---------------------------------------------------------------------------------- instruments

/** Where the draw instruments stand: over the admission lane, right of the definition. */
const INST = { x0: 32.4, x1: 53.6, y: -6.6 } as const;

/**
 * The class drawn: the five classes as one bar, each as wide as its weight. A marker runs along
 * it, slowing, and stops on the class drawn.
 */
const classBar =
  (w: Record<Class, { n: bigint; d: bigint }>, drawn: Class): Fx =>
  (_drawing, p) => {
    const vis = swell(p, 0, 1, 0.12, 0);
    if (vis <= 0.01) return {};
    const { x0, x1, y } = INST;
    const W = x1 - x0;
    const add: Prim[] = [];
    let x: number = x0;
    let target: number = x0;
    for (const c of CLASSES) {
      const share = Number(w[c].n) / Number(w[c].d);
      const xa = x;
      const xb = x + W * share;
      const landed = c === drawn ? band(p, 0.72, 0.82) : 0;
      add.push({
        kind: "face",
        pts: quad(xa, y, xb, y - 1.4),
        layer: 2,
        fill: "paper",
        edge: c === drawn && landed > 0 ? NEW : "ink",
        w: 1.1,
        alpha: vis,
        ...(landed > 0 ? { glow: 14 * landed, glowInk: NEW } : {}),
      });
      if (landed > 0) add.push({ ...wash(quad(xa, y, xb, y - 1.4), 0.22 * landed), layer: 2 });
      add.push(
        at((xa + xb) / 2, y - 0.7, c, {
          size: 11,
          italic: true,
          anchor: "middle",
          dy: 4,
          alpha: vis,
          ...(c === drawn && landed > 0 ? { ink: NEW } : {}),
        }),
      );
      if (c === drawn) target = (xa + xb) / 2;
      x = xb;
    }
    // The marker: two laps and a landing, slowing down.
    const travel = 2 * W + (target - x0);
    const m = x0 + ((travel * (1 - (1 - band(p, 0.08, 0.75)) ** 3)) % W);
    add.push({
      kind: "face",
      pts: [stageAt(m - 0.35, y + 0.75), stageAt(m + 0.35, y + 0.75), stageAt(m, y + 0.15)],
      layer: 2,
      fill: "paper",
      edge: NEW,
      w: 1.2,
      alpha: vis,
      glow: 8,
    });
    return { add };
  };

/** The choice drawn: the candidates in a row, a light that runs over them and stops on one. */
const choices =
  (items: readonly string[], chosen: number, o: { range?: [string, string] } = {}): Fx =>
  (_drawing, p) => {
    const vis = swell(p, 0, 1, 0.12, 0);
    if (vis <= 0.01) return {};
    const { x0, x1, y } = INST;
    const add: Prim[] = [];
    if (o.range) {
      // A number line over a declared range, the marker landing at the chosen end.
      add.push({
        kind: "stroke",
        pts: [stageAt(x0 + 1, y - 0.7), stageAt(x1 - 1, y - 0.7)],
        w: 1.3,
        ink: "ink",
        layer: 2,
        alpha: vis,
      });
      add.push(
        at(x0 + 1, y - 0.7, o.range[0], { size: 11, anchor: "middle", dy: 18, alpha: vis }),
      );
      add.push(
        at(x1 - 1, y - 0.7, o.range[1], { size: 11, anchor: "middle", dy: 18, alpha: vis }),
      );
      const q = 1 - (1 - band(p, 0.08, 0.75)) ** 3;
      const m = x0 + 1 + (x1 - x0 - 2) * (0.5 + 0.5 * Math.sin(q * Math.PI * 2.5 - Math.PI / 2));
      add.push({
        kind: "dot",
        at: stageAt(m, y - 0.7),
        r: 5,
        ink: NEW,
        layer: 2,
        alpha: vis,
        glow: 10,
      });
      return { add };
    }
    const sizes = items.map((t) => textWidth(t, 11, true) / 20 + 0.6);
    const rows = items.length > 6 ? 2 : 1;
    const per = Math.ceil(items.length / rows);
    const scan = Math.floor(band(p, 0.05, 0.7) * (items.length * 2 + chosen + 0.999));
    items.forEach((t, i) => {
      const row = Math.floor(i / per);
      const col = i % per;
      const inRow = items.slice(row * per, row * per + per);
      const total = inRow.reduce((a, _, j) => a + sizes[row * per + j]!, 0);
      const gap = (x1 - x0 - total) / Math.max(1, inRow.length - 1);
      let x = x0;
      for (let j = 0; j < col; j++) x += sizes[row * per + j]! + gap;
      const lit = p > 0.72 ? i === chosen : scan % items.length === i;
      add.push(
        at(x, y - 0.6 - 1.2 * row, t, {
          size: 11,
          italic: true,
          dy: 4,
          alpha: vis * (lit ? 1 : 0.7),
          ink: lit ? NEW : "ink",
        }),
      );
    });
    return { add };
  };

// ------------------------------------------------------------------------------------- the band

/** The draws of a move as chips: "parent 1 in 6 × class 5 in 18 × … = about 1 in N". */
function chips(
  p: Pick<Proposal, "factors" | "probability">,
  shown: number,
  total: number,
): Prim[] {
  const out: Prim[] = [];
  let dx = 0;
  const [x, y, z] = stageAt(BAND.x, BAND.draws);
  const put = (text: string, o: Partial<Extract<Prim, { kind: "label" }>>) =>
    out.push({ kind: "label", at: [x, y, z], text, size: 13, ink: "ink", layer: 2, dx, ...o });
  p.factors.forEach((f, i) => {
    if (i >= shown) return;
    if (i > 0) {
      put("×", { ink: "faint", size: 13 });
      dx += 22;
    }
    put(f.name, { size: 11.5, italic: true, ink: "faint" });
    dx += textWidth(f.name, 11.5, true) * 1.08 + 7;
    put(chance(f.q), {});
    dx += textWidth(chance(f.q), 13) * 1.08 + 14;
  });
  if (total > 0) {
    put("=", { ink: "faint", alpha: total });
    dx += 22;
    put(odds(p.probability), {
      size: 15,
      weight: 600,
      alpha: total,
      ...(total < 1 ? { glow: 64 * total * (1 - total), glowInk: NEW } : {}),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------- the chapters

const MOVES = moves();
const byId = (id: string) => MOVES.find((m) => m.id === id)!;
const at0 = { rx: BAND.x, ry: BAND.ledger };

/** The keys of every legal site of a class, on the registry drawing. */
const classKeys = (c: Class) => sites(SEED)[c].flatMap(siteKeys);

/** Where the move's site is, on the definition drawn below the strip. */
const SITE_KEYS: Record<string, string[]> = {
  replace: ["act"],
  insert: ["half0/out"],
  delete: ["half0/norm"],
  binding: ["w2"],
  shape: ["flane1", "flane3"],
};
/** Where the new or removed ink is, for the sparks. */
const SPARK: Record<string, { places: string[][]; mode: "gather" | "scatter" }> = {
  replace: { places: [["act"]], mode: "gather" },
  insert: { places: [["half0/new"]], mode: "gather" },
  delete: { places: [["half0/norm"]], mode: "scatter" },
  // The 64 leaves gathering into one are the motion; they are lit instead.
  binding: { places: [], mode: "gather" },
  shape: { places: [["print20/swiglu"], ["print41/swiglu"]], mode: "gather" },
};
const CHOICES: Record<string, Fx | null> = {
  replace: choices(["relu", "gelu_tanh", "gelu_erf", "tanh", "sigmoid", "square"], 1),
  insert: choices(
    [
      "relu",
      "gelu_tanh",
      "gelu_erf",
      "silu",
      "tanh",
      "sigmoid",
      "square",
      "rms_norm",
      "learned_scale",
      "causal_dwconv3",
    ],
    7,
  ),
  delete: null,
  binding: null,
  shape: choices([], 0, { range: ["512", "1,024"] }),
};

/** A move, told in nine beats: the five draws, the rewrite, admission, the verdict, joining. */
function moveChapter(id: string): Chapter {
  const m = byId(id);
  const w = weights(counts(SEED));
  const L = m.n === 1 ? 64 : m.n;
  const reach: "all" | readonly number[] = m.reach === "all" ? "all" : m.selected;
  const before = `rw/${id}/before`;
  const after = `rw/${id}`;
  // The draws appear one by one; the odds appear over the end of the reach beat, then stay.
  const draws =
    (shown: number, odds: "none" | "reveal" | "shown" = "none") =>
    (p: number) =>
      chips(m, shown, odds === "reveal" ? band(p, 0.55, 0.85) : odds === "shown" ? 1 : 0);
  const choice = CHOICES[id];
  return {
    id,
    title: view(after).title,
    caption: view(after).caption,
    ledger: (s) => product(s, at0, m, [paramNote(m)]),
    beats: [
      {
        ms: 1500,
        to: before,
        say: m.factors[0]!.say,
        fx: [light(["ends"], { glow: 8 })],
        band: draws(1),
      },
      { ms: 1800, to: before, say: m.factors[1]!.say, fx: [classBar(w, m.cls)], band: draws(2) },
      {
        ms: 1500,
        to: before,
        say: m.factors[2]!.say,
        fx: [mark(SITE_KEYS[id]!), light(SITE_KEYS[id]!, { ink: NEW })],
        band: draws(3),
      },
      {
        ms: 1500,
        to: before,
        say: m.factors[3]!.say,
        fx: [...(choice ? [choice] : []), light(SITE_KEYS[id]!, { ink: NEW, to: 0.9 })],
        band: draws(4),
      },
      {
        ms: 1900,
        to: before,
        say: m.factors[4]!.say,
        fx: [sweep(L, reach)],
        band: draws(5, "reveal"),
      },
      {
        ms: 2200,
        to: after,
        move: 0.8,
        say: m.say,
        fx: [
          particles(SPARK[id]!.places, SPARK[id]!.mode),
          ...(id === "binding"
            ? [
                light(
                  Array.from({ length: 64 }, (_, i) => `slice${i}`),
                  { ink: NEW, from: 0.2, glow: 10 },
                ),
              ]
            : []),
          ...(reach === "all" ? [sweep(L, "all", { from: 0.3, to: 0.95 })] : []),
        ],
        band: draws(5, "shown"),
      },
      {
        ms: 1900,
        to: after,
        say: "Admission checks it, the cheapest test first.",
        fx: [token()],
        band: draws(5, "shown"),
      },
      {
        ms: 1100,
        to: after,
        say: "It passes every check, so it is admitted.",
        fx: [seal(() => [DECK_X - 0.4, LANE_AT.y + 4.2, 0], "Admitted", NEW)],
        band: draws(5, "shown"),
      },
      {
        ms: 1700,
        to: after,
        say: "If its first training run succeeds, it joins the population as the youngest member; the oldest retires.",
        fx: [join],
        band: draws(5, "shown"),
      },
    ],
  };
}

/** Where admission's verdict for A′ is sealed: over the end of its lane. */
const admittedAt = (): V3 => [GATE_ROW.out, GATE_ROW.y + 3.8, 0];

function chapters(): Chapter[] {
  const { P, S } = POPULATION;
  const c = counts(SEED);
  const total = Object.values(c).reduce((a, b) => a + b, 0);
  const plural = (k: number, one: string, many: string) => `${k} ${k === 1 ? one : many}`;
  const classSay: Record<Class, string> = {
    replace: `Replace an operation in ${plural(c.replace, "place", "places")}.`,
    insert: `Insert a node in ${plural(c.insert, "place", "places")}.`,
    delete: `Delete a node in ${plural(c.delete, "place", "places")}.`,
    binding: `Share or untie a weight in ${plural(c.binding, "place", "places")}.`,
    shape: `Change a width in its declared range, in ${plural(c.shape, "place", "places")}.`,
  };
  const parentOdds = tournament(P, S, 1);
  const { fromA, fromB } = commuting();
  const ds = deepseek();
  const dsAll = [...ds.factors];
  const dsChips =
    (shown: number, total = 0) =>
    () =>
      chips({ factors: dsAll, probability: ds.probability }, shown, total);
  return [
    {
      id: "population",
      title: view("rw/population").title,
      caption: view("rw/population").caption,
      ledger: (s) => populationLedger(s, at0),
      beats: [
        {
          ms: 2800,
          to: "rw/population/before",
          say: `In AMLO’s 0.0.14 design, a search keeps a population of ${P} models, ranked by loss.`,
        },
        {
          ms: 2600,
          to: "rw/population/drawn",
          say: `It draws ${S} of them at random.`,
          fx: [
            light(
              DRAWN.map((r) => `card${r}`),
              { from: 0.35, ink: "ochre", glow: 22 },
            ),
          ],
        },
        {
          ms: 3200,
          to: "rw/population",
          say: "The best-ranked of the four is the parent, here the seed, Monad.",
          fx: [
            light(["card1"], { ink: NEW, from: 0.3, glow: 22 }),
            mark(["card1"], { from: 0.3, pad: 0.4 }),
          ],
          band: () =>
            chips({ factors: [{ ...MOVES[0]!.factors[0]! }], probability: parentOdds }, 1, 0),
        },
      ],
    },
    {
      id: "sites",
      title: view("rw/sites").title,
      caption: view("rw/sites").caption,
      ledger: (s) => sitesLedger(s, { rx: BAND.x, ry: -16.2 }),
      beats: [
        {
          ms: 2600,
          to: "rw/sites",
          say: `The parent’s definitions have ${total} places where one move can change them.`,
        },
        ...CLASSES.map((k, i): Beat => ({
          ms: 1900,
          to: "rw/sites",
          say: classSay[k],
          // The last class stays lit, so the chapter's last frame still shows its places.
          fx: [mark(classKeys(k), { from: 0.04, to: i === CLASSES.length - 1 ? 1 : 0.96 })],
        })),
      ],
    },
    moveChapter("replace"),
    moveChapter("insert"),
    moveChapter("delete"),
    moveChapter("binding"),
    moveChapter("shape"),
    {
      id: "commute",
      title: view("rw/commute").title,
      caption: view("rw/commute").caption,
      ledger: (s) => admissionLedger(s, at0),
      beats: [
        {
          ms: 2800,
          to: "rw/commute/before",
          say: "A and B are two children of the seed. Suppose that a generation later they rank first and second.",
          fx: [light(["A", "B"], { from: 0.4, glow: 8 })],
        },
        {
          ms: 4000,
          to: "rw/commute",
          move: 0.7,
          say: `From A, insert a norm (${odds(fromA.probability)}). From B, replace silu (${odds(fromB.probability)}).`,
        },
        {
          ms: 3400,
          to: "rw/commute",
          say: "The two results are one architecture. Identity looks past their different ids.",
          fx: [light(["C"], { ink: NEW, glow: 10 })],
        },
      ],
    },
    {
      id: "admission",
      title: view("rw/admission").title,
      caption: view("rw/admission").caption,
      ledger: (s) => admissionLedger(s, at0),
      beats: [
        {
          ms: 2400,
          to: "rw/admission/before",
          say: "Both come to admission in the same generation.",
        },
        {
          ms: 3600,
          to: "rw/admission/a",
          move: 0.75,
          say: "A′ passes every gate, the cheapest first, and is admitted.",
          fx: [passing("a", "ink"), seal(admittedAt, "Admitted", "ink", { from: 0.72 })],
        },
        {
          ms: 4600,
          to: "rw/admission",
          move: 0.6,
          say: "B′ passes six gates. The seventh, identity, refuses it as a duplicate.",
          fx: [
            passing("b"),
            light(["gate6"], { ink: "cinnabar", from: 0.5, glow: 18 }),
            seal(admittedAt, "Admitted", "ink", { from: -1 }),
            seal(() => [GATE_ROW.last, GATE_ROW.y + 6.8, 0], "Refused", NEW, { from: 0.6 }),
          ],
        },
      ],
    },
    {
      id: "deepseek",
      title: view("rw/deepseek").title,
      caption: view("rw/deepseek").caption,
      model: "dsv41",
      ledger: (s) =>
        product(s, at0, ds, [
          "No control declares a range, so shape has no site and 4 classes remain.",
        ]),
      beats: [
        {
          ms: 3000,
          to: "rw/deepseek/before",
          say: "The same rules work on the toy DeepSeek.",
          band: dsChips(1),
        },
        {
          ms: 2400,
          to: "rw/deepseek/before",
          say: `${ds.factors[1]!.say} ${ds.factors[2]!.say}`,
          fx: [light(["expert0/act"], { ink: NEW })],
          band: dsChips(4),
        },
        {
          ms: 2400,
          to: "rw/deepseek/before",
          say: ds.factors[4]!.say,
          fx: [sweep(40, ds.selected)],
          band: dsChips(5, 1),
        },
        {
          ms: 3000,
          to: "rw/deepseek",
          move: 0.75,
          say: `Layer ${ds.selected[0]} gets its own copy, two Repeats deep.`,
          fx: [light([`bracket${24}-${27}`, `num${ds.selected[0]}`], { ink: NEW, from: 0.6 })],
          band: dsChips(5, 1),
        },
        { ms: 2600, to: "rw/deepseek", say: ds.say, band: dsChips(5, 1) },
      ],
    },
  ];
}

/** Figure B's film: the search's moves, from Monad's model view. */
export const { film, chapterAt, chapterEnd, frameAt, OPENS_FROM } = makeFilm({
  chapters,
  opensFrom: "model",
  honesty: ILLUSTRATION,
});

export type { Chapter, FilmFrame } from "./reel.ts";
