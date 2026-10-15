/**
 * A film of the window: chapters of beats, each beat carrying the drawing into one view, saying
 * one plain sentence and lighting what it acts on. A film is one pure function from time to a
 * frame, so a pause leaves a still image and every frame can be checked like a drawing. Figures B
 * and C are films; this module holds what they share: the effects, the band under the drawing,
 * and the clock.
 *
 * Effects accompany actions only. At rest the drawing stays ink on paper.
 */
import { type Ink, type Prim, type V3 } from "./engine.ts";
import { design, Sketch } from "./kit.ts";
import { view } from "./models.ts";
import { rewriteFrame } from "./morph.ts";
import { NEW, SX, stageAt } from "./rewrite.ts";
import type { View } from "./view.ts";

// ------------------------------------------------------------------------------------- effects

/** What a beat lights, as a function of its progress p from 0 to 1 and the drawing it lights. */
/** An effect on a beat's drawing; `keys` names what it acts on, so a test can check it is drawn. */
export type Fx = ((drawing: Prim[], p: number) => { drawing?: Prim[]; add?: Prim[] }) & {
  keys?: readonly string[];
};

const acting = (keys: readonly string[], fx: Fx): Fx => Object.assign(fx, { keys });

export const clamp = (t: number) => Math.min(1, Math.max(0, t));
export const band = (t: number, a: number, b: number) => clamp((t - a) / (b - a));
export const smooth = (t: number) => t * t * (3 - 2 * t);
/** Rises over [a, a + rise], holds, falls over [b - fall, b]. */
export const swell = (p: number, a = 0.1, b = 1, rise = 0.2, fall = 0.15) =>
  Math.min(smooth(band(p, a, a + rise)), b >= 1 ? 1 : 1 - smooth(band(p, b - fall, b)));

export const matches = (p: Prim, prefixes: readonly string[]) =>
  p.key !== undefined &&
  prefixes.some((k) => p.key === k || p.key!.startsWith(`${k}:`) || p.key!.startsWith(`${k}/`));

/** Light the primitives under some keys: a glow around their ink. */
export const light = (
  keys: readonly string[],
  o: { ink?: Ink; from?: number; to?: number; glow?: number } = {},
): Fx =>
  acting(keys, (drawing, p) => {
    const s = swell(p, o.from ?? 0.08, o.to ?? 1);
    if (s <= 0.01) return {};
    return {
      drawing: drawing.map((q) =>
        matches(q, keys) && q.kind !== "label"
          ? { ...q, glow: (o.glow ?? 14) * s, ...(o.ink ? { glowInk: o.ink } : {}) }
          : q,
      ),
    };
  });

/** Light washed over a face: a translucent sheet of one ink. */
export const wash = (pts: readonly V3[], alpha: number, ink: Ink = NEW): Prim => ({
  kind: "face",
  pts: [...pts],
  layer: 0,
  fill: "wash",
  edge: ink,
  alpha,
  glow: 10 * alpha,
});

/**
 * Mark places: light washed over each thing drawn under a key. Each key is one place, however
 * many primitives draw it.
 */
export const mark = (
  keys: readonly string[],
  o: { ink?: Ink; from?: number; to?: number; pad?: number } = {},
): Fx =>
  acting(keys, (drawing, p) => {
    const s = swell(p, o.from ?? 0.05, o.to ?? 1, 0.25, 0.15);
    if (s <= 0.01) return {};
    const pad = o.pad ?? 0.32;
    const add: Prim[] = [];
    for (const k of keys) {
      let x0 = Infinity;
      let x1 = -Infinity;
      let y0 = Infinity;
      let y1 = -Infinity;
      let z = 0;
      for (const q of drawing)
        if (matches(q, [k]) && q.kind !== "label")
          for (const v of q.kind === "stroke" || q.kind === "face" ? q.pts : [q.at]) {
            x0 = Math.min(x0, v[0]);
            x1 = Math.max(x1, v[0]);
            y0 = Math.min(y0, v[1]);
            y1 = Math.max(y1, v[1]);
            z = v[2];
          }
      if (x0 === Infinity) continue;
      const pts: V3[] = [
        [x0 - pad, y1 + pad, z],
        [x1 + pad, y1 + pad, z],
        [x1 + pad, y0 - pad, z],
        [x0 - pad, y0 - pad, z],
      ];
      // A wash, like a highlighter's: it marks the place without a line across its lettering.
      add.push({ ...wash(pts, 0.26 * s, o.ink ?? NEW), glow: 12 * s });
    }
    return { add };
  });

/** The centre of everything drawn under some keys. */
export function centre(drawing: Prim[], keys: readonly string[]): V3 | null {
  let n = 0;
  const c = [0, 0, 0];
  for (const q of drawing)
    if (matches(q, keys))
      for (const v of q.kind === "stroke" || q.kind === "face" ? q.pts : [q.at]) {
        c[0]! += v[0];
        c[1]! += v[1];
        c[2]! += v[2];
        n++;
      }
  return n ? [c[0]! / n, c[1]! / n, c[2]! / n] : null;
}

/** A face's points, by its key. */
export const face = (drawing: Prim[], key: string) => {
  const f = drawing.find((q) => q.key === key && q.kind === "face");
  return f && f.kind === "face" ? f.pts : null;
};

/**
 * Sparks of light gathering into each place a node arrives, or scattering from each place one
 * leaves. They rise from above it and fade before they reach the glyph, so they never cross
 * its lettering.
 */
export const particles = (
  places: readonly (readonly string[])[],
  mode: "gather" | "scatter",
  o: { from?: number; to?: number } = {},
): Fx =>
  acting(places.flat(), (drawing, p) => {
    const q = band(p, o.from ?? 0.15, o.to ?? 0.85);
    if (q <= 0 || q >= 1) return {};
    const add: Prim[] = [];
    for (const keys of places) {
      const at = centre(drawing, keys);
      if (!at) continue;
      const n = 14;
      for (let i = 0; i < n; i++) {
        const a = Math.PI / 2 + ((i / (n - 1)) * 2 - 1) * 1.05 + ((i * 7) % 3) * 0.05;
        const far = 2.4 + ((i * 7) % 5) * 0.3;
        const r = 1.3 + (far - 1.3) * (mode === "gather" ? 1 - smooth(q) : smooth(q));
        const fade =
          mode === "gather"
            ? smooth(band(q, 0, 0.3)) * (1 - band(q, 0.7, 0.95))
            : smooth(band(q, 0, 0.12)) * (1 - q);
        add.push({
          kind: "dot",
          at: [at[0] + Math.cos(a) * r, at[1] + Math.sin(a) * r, at[2]],
          r: 1.6 + (i % 3) * 0.6,
          ink: NEW,
          layer: 1,
          alpha: 0.85 * fade,
          glow: 8,
        });
      }
    }
    return { add };
  });

/** A seal pressed at a point: a ring with the verdict inside. */
export const seal =
  (at: () => V3, text: string, ink: Ink, o: { from?: number } = {}): Fx =>
  (_drawing, p) => {
    const q = band(p, o.from ?? 0.1, (o.from ?? 0.1) + 0.35);
    if (q <= 0) return {};
    const k = smooth(q);
    const r = (1.18 - 0.18 * k) * 38;
    const c = at();
    return {
      add: [
        { kind: "dot", at: c, r, ink, layer: 2, hollow: true, alpha: k, glow: 16 * (1 - 0.6 * k) },
        {
          kind: "label",
          at: c,
          text,
          size: 13,
          weight: 600,
          ink,
          layer: 2,
          anchor: "middle",
          dy: 4.5,
          alpha: k,
        },
      ],
    };
  };

// ------------------------------------------------------------------------------- the stage band

/** A label hung at a stage point. */
export const at = (
  rx: number,
  ry: number,
  text: string,
  o: Partial<Extract<Prim, { kind: "label" }>> = {},
): Prim => ({
  kind: "label",
  at: stageAt(rx, ry),
  text,
  size: 12,
  ink: "ink",
  layer: 2,
  ...o,
});

/** A flat box at the stage, by its corners in stage units. */
export const quad = (x0: number, y0: number, x1: number, y1: number): V3[] => [
  stageAt(x0, y0),
  stageAt(x1, y0),
  stageAt(x1, y1),
  stageAt(x0, y1),
];

/** The band under the drawing: the beat's sentence, then what the beat adds, and the ledger. */
export const BAND = { x: -2.6, say: -18.2, draws: -20.6, ledger: -17.6 } as const;

const say = (text: string, alpha = 1, x: number = BAND.x, y: number = BAND.say): Prim => ({
  ...at(x, y, text, { size: 17, alpha }),
});

/** What a beat leaves lit or added dims over this long as the next begins, in milliseconds. */
const RELEASE = 450;
/** The band between beats: what goes fades out over the first span, then what comes fades in. */
const LETTERS_OUT = [0, 220] as const;
const LETTERS_IN = [180, 560] as const;
/** With details on, a chapter's ledger fades in over its last moments, in milliseconds. */
const LEDGER_IN = 700;

/** A primitive's light at a fraction: its glow, and the whole of a wash, which is all light. */
const unlit = (q: Prim, k: number): Prim =>
  k >= 1
    ? q
    : q.kind === "face" && q.fill === "wash"
      ? dim(q, k)
      : q.glow
        ? { ...q, glow: q.glow * k }
        : q;

/** A primitive at a fraction of its opacity, its light dimmed with it. */
const dim = (q: Prim, k: number): Prim =>
  k >= 1 ? q : { ...q, alpha: (q.alpha ?? 1) * k, ...(q.glow ? { glow: q.glow * k } : {}) };

// ----------------------------------------------------------------------------------- the clock

/**
 * What the reader sets: details, which stop the film at each chapter's end with its ledger; the
 * ledger shown now, from 0 to 1, when the reader asks for it mid-chapter; and for a figure with
 * walls, which are hidden and a policy.
 */
export interface Options {
  details?: boolean;
  ledger?: number;
  hide?: readonly string[];
  policy?: "StoreAll" | "Remat";
}

export interface Beat {
  /** How long the beat lasts, in milliseconds. */
  ms: number;
  /** The view the beat carries the drawing into; it may depend on the reader's options. */
  to: string | ((o: Options) => string);
  /** The share of the beat spent carrying the drawing there. */
  move?: number;
  say: string;
  fx?: Fx[];
  /** The band under the drawing during this beat, as a function of its progress. */
  band?: (p: number) => Prim[];
}

export interface Chapter {
  id: string;
  title: string;
  /** The full caption, under the window when details are shown. */
  caption: string;
  /** The model the chapter draws, when it is not Monad. */
  model?: string;
  beats: Beat[];
  /** The chapter's ledger, set in the band when details are shown. */
  ledger: (s: Sketch) => void;
}

export interface FilmFrame {
  /** The views the drawing is carried between, and how far: the window's camera follows. */
  from: View;
  to: View;
  u: number;
  prims: Prim[];
  chapter: number;
  beat: number;
  say: string;
}

/** A film: its chapters, its clock and its frames. */
export interface Film {
  film(): { chapters: Chapter[]; starts: number[]; total: number };
  chapterAt(t: number): number;
  chapterEnd(i: number): number;
  frameAt(t: number, o?: Options): FilmFrame;
  /** The view the film opens from. */
  OPENS_FROM: string;
}

/**
 * Make a film from its chapters: where each starts, and the frame at any time. `honesty` is said
 * in every frame, in the stage's lower right corner.
 */
export function makeFilm(o: {
  chapters: () => Chapter[];
  opensFrom: string;
  honesty: string;
  /** Where the beat's sentence starts, in stage units, when not at the band's usual place. */
  sayX?: number;
  sayY?: number;
}): Film {
  let cache: { chapters: Chapter[]; starts: number[]; total: number } | null = null;
  // A view is a pure function of its id, and a film draws the same few many times a second: each is
  // built once, with the design scale it was built at, which the band's glyphs are set at too.
  const views = new Map<string, { view: View; ppu: number }>();
  const viewOf = (id: string) => {
    let v = views.get(id);
    if (!v) {
      const built = view(id);
      views.set(id, (v = { view: built, ppu: design.ppu }));
    }
    return v;
  };
  const film = () => {
    if (!cache) {
      const list = o.chapters();
      const starts: number[] = [];
      let t = 0;
      for (const c of list) {
        starts.push(t);
        t += c.beats.reduce((a, b) => a + b.ms, 0);
      }
      cache = { chapters: list, starts, total: t };
    }
    return cache;
  };
  // A chapter's end belongs to it, so a pause there shows its last frame.
  const chapterAt = (t: number) => {
    const { starts, total } = film();
    const clamped = Math.min(Math.max(t, 0), total);
    let i = 0;
    while (i + 1 < starts.length && clamped > starts[i + 1]!) i++;
    return i;
  };
  const chapterEnd = (i: number) => {
    const { starts, total } = film();
    return i + 1 < starts.length ? starts[i + 1]! : total;
  };
  const target = (b: Beat, opts: Options) => (typeof b.to === "string" ? b.to : b.to(opts));
  const honesty = (): Prim =>
    at(SX[1] - 0.9, -23.1, o.honesty, { size: 11, italic: true, ink: "faint", anchor: "end" });
  // A chapter's ledger, set once, on the band's layer.
  const ledgers = new Map<number, Prim[]>();
  const ledgerOf = (ci: number, c: Chapter) => {
    let l = ledgers.get(ci);
    if (!l) {
      const s = new Sketch();
      c.ledger(s);
      ledgers.set(ci, (l = s.prims.map((q) => ({ ...q, layer: 2 }))));
    }
    return l;
  };
  /**
   * One beat's frame at time t, on its own: the drawing with the beat's effects, what they add,
   * and the band. The band is the beat's sentence and what the beat adds to it, giving way to the
   * chapter's ledger: at once as far as `ledger` asks, and with `details` over the chapter's last
   * moments. `hide` leaves out what is drawn under some keys.
   */
  const compose = (t: number, opts: Options) => {
    const { chapters: list, starts, total } = film();
    const clamped = Math.min(Math.max(t, 0), total);
    const ci = chapterAt(clamped);
    const c = list[ci]!;
    let local = clamped - starts[ci]!;
    let bi = 0;
    while (bi + 1 < c.beats.length && local > c.beats[bi]!.ms) {
      local -= c.beats[bi]!.ms;
      bi++;
    }
    const b = c.beats[bi]!;
    const p = clamp(local / b.ms);
    // The view this beat starts from: the previous beat's, the previous chapter's last, or
    // where the figure opens.
    const prev =
      bi > 0
        ? target(c.beats[bi - 1]!, opts)
        : ci > 0
          ? target(list[ci - 1]!.beats.at(-1)!, opts)
          : o.opensFrom;
    const from = viewOf(prev).view;
    const arriving = viewOf(target(b, opts));
    const to = arriving.view;
    design.ppu = arriving.ppu;
    const u = prev === to.id ? 1 : smooth(band(p, 0, b.move ?? 0.55));
    const hidden = (list: Prim[]) =>
      opts.hide?.length ? list.filter((q) => !matches(q, opts.hide!)) : list;
    let drawing =
      u >= 1
        ? hidden(to.sketch.prims)
        : rewriteFrame(hidden(from.sketch.prims), hidden(to.sketch.prims), u);
    let add: Prim[] = [];
    for (const fx of b.fx ?? []) {
      const r = fx(drawing, p);
      if (r.drawing) drawing = r.drawing;
      if (r.add) add.push(...r.add);
    }
    // The sentence leaves over the first half of the way to the ledger, and the ledger comes
    // over the second, so the two are never set over each other.
    const end = chapterEnd(ci);
    const r = Math.max(
      opts.ledger ?? 0,
      opts.details ? band(clamped, end - LEDGER_IN, end - 0.5) : 0,
    );
    const leaving = smooth(band(r, 0, 0.5));
    const coming = smooth(band(r, 0.5, 1));
    // Stopped for the ledger, the drawing rests in ink: its lights go out with the sentence.
    if (leaving > 0) {
      drawing = drawing.map((q) => (q.glow ? unlit(q, 1 - leaving) : q));
      add = add.map((q) => unlit(q, 1 - leaving));
    }
    const said = leaving < 1 ? [say(b.say, 1, o.sayX, o.sayY), ...(b.band?.(p) ?? [])] : [];
    const bandPrims = [
      ...(leaving > 0 ? said.map((q) => dim(q, 1 - leaving)) : said),
      ...(coming > 0 ? ledgerOf(ci, c).map((q) => dim(q, coming)) : []),
    ];
    return { ci, bi, b, from, to, u, drawing, add, band: bandPrims, start: clamped - local };
  };

  /**
   * The frame at time t: the beat's own, carried on from the beat before. What that beat lit or
   * added dims over its first moments, and the band's lettering crossfades: what stays stays,
   * what goes fades out, and then what comes fades in. So nothing in a film starts or stops in
   * one frame, whatever an effect does at its beat's ends.
   */
  const frameAt = (t: number, opts: Options = {}): FilmFrame => {
    const cur = compose(t, opts);
    const { ci, bi, b, from, to, u } = cur;
    let { drawing, add, band: said } = cur;
    const into = Math.min(Math.max(t, 0), film().total) - cur.start;
    const entering = smooth(band(into, LETTERS_IN[0], LETTERS_IN[1]));
    // The same primitive, whatever its opacity and light.
    const sig = (q: Prim) => JSON.stringify({ ...q, alpha: 0, glow: 0 });
    if (cur.start <= 0) said = said.map((q) => dim(q, entering));
    else if (into < Math.max(RELEASE, LETTERS_IN[1])) {
      const prev = compose(cur.start - 0.01, opts);
      const k = 1 - smooth(band(into, 0, RELEASE));
      if (k > 0.001) {
        // A light the beat before left on dims, unless this beat lights it brighter.
        const lit = new Map(
          prev.drawing.flatMap((q) =>
            q.key && (q.glow ?? 0) > 0.01 ? [[q.key, q] as const] : [],
          ),
        );
        if (lit.size)
          drawing = drawing.map((q) => {
            const was = q.key ? lit.get(q.key) : undefined;
            const g = (was?.glow ?? 0) * k;
            return was && g > (q.glow ?? 0)
              ? { ...q, glow: g, ...(was.glowInk ? { glowInk: was.glowInk } : {}) }
              : q;
          });
        // What it added fades out, unless this beat draws the same thing in the same place.
        const kept = new Set(add.map(sig));
        add = [...prev.add.filter((q) => !kept.has(sig(q))).map((q) => dim(q, k)), ...add];
      }
      const before = new Set(prev.band.map(sig));
      const after = new Set(said.map(sig));
      const leaving = 1 - smooth(band(into, LETTERS_OUT[0], LETTERS_OUT[1]));
      said = [
        ...(leaving > 0.001
          ? prev.band.filter((q) => !after.has(sig(q))).map((q) => dim(q, leaving))
          : []),
        ...said.map((q) => (before.has(sig(q)) ? q : dim(q, entering))),
      ];
    }
    // Coming in from a drawing with its own camera, the band waits for the stage to arrive.
    const arrive = from.stage ? 1 : smooth(band(u, 0.7, 1));
    const fadeIn = (list: Prim[]) => (arrive >= 1 ? list : list.map((q) => dim(q, arrive)));
    const bandPrims = said;
    return {
      from,
      to,
      u,
      prims: [...drawing, ...add, ...fadeIn([...bandPrims, honesty()])],
      chapter: ci,
      beat: bi,
      say: b.say,
    };
  };
  return { film, chapterAt, chapterEnd, frameAt, OPENS_FROM: o.opensFrom };
}
