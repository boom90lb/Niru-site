/**
 * The algebra window: a canvas that draws one view at a time and opens a box
 * in place, the way AMLO's 0.0.14 workspace zooms: the child view is drawn
 * inside the box it opens from while the camera moves into it. The window
 * remembers the box each view was opened from, so a definition used in many
 * places closes back into the one it came from.
 *
 * Every view is one picture designed at the plate's full width; at any other
 * size, and through a zoom, its lettering and line weights scale with it.
 * Every frame is a function of the state and the transition time. Nothing
 * animates at rest.
 *
 * Figures B and C are films, each with its own clock. B, the search's moves on
 * Monad's term, starts by itself when the window is in view; a film pauses when
 * the window leaves the view or the tab hides, and never starts by itself when
 * the reader prefers reduced motion: then it is a row of still chapters. A
 * film's frame is a pure function of its clock, so a pause leaves a still
 * image. With details on, it stops at the end of each chapter and sets the
 * chapter's ledger under the drawing. At C's end, the reader shows or hides
 * each of its bands and chooses its residual policy.
 *
 * Figure D is views: the reader moves Monad's controls, and the window carries
 * the drawing from one member of the family to the next.
 */
import {
  drawCanvas,
  fit,
  frame,
  inside,
  mapPrims,
  mix,
  project,
  REF,
  rotate,
  type Camera,
  type Palette,
  type Prim,
  type V3,
} from "./engine.ts";
import type { Setting } from "./family.ts";
import type { Openable } from "./kit.ts";
import {
  FIGURES,
  load,
  loadFigure,
  MODELS,
  view,
  type Figure,
  type FigureFilm,
  type Model,
} from "./models.ts";
import { cameraBetween, rewriteFrame } from "./morph.ts";
import type { FilmFrame, Options } from "./reel.ts";
import { EVIDENCE, evidenceOf, framing, type View } from "./view.ts";

type Family = typeof import("./fam.ts");

const DURATION = 760;
/** A jump between chapters, or into or out of the film: a short blend of one frame into another. */
const BLEND = 520;

interface Placement {
  /** Child point → parent point: p ↦ offset + s·p. */
  offset: V3;
  s: number;
}

/** A view on the way down from the training step, and the box it was opened from. */
interface Entry {
  view: View;
  via?: Openable;
  /** The entry is a figure's film, by the figure's id: its view is the one the beat draws. */
  film?: string;
}

/** A film's clock. */
interface Clock {
  t: number;
  playing: boolean;
  /** Paused by the reader, so leaving and returning to view does not resume it. */
  held: boolean;
  /** When the last animation frame advanced the clock. */
  last: number;
  /** The chapter and beat last shown, to update the controls when either changes. */
  chapter: number;
  beat: number;
}

/** A figure's film once loaded: its module, its clock, and what the reader set for it. */
interface Reel {
  figure: Figure;
  mod: FigureFilm;
  clock: Clock;
  /** The bands the reader hid, by id, and the residual policy: for a film with walls. */
  hide: Set<string>;
  policy: "StoreAll" | "Remat";
}

/** Whether a key is one of some prefixes or lies under one, as the films match keys. */
const under = (key: string, prefixes: readonly string[]) =>
  prefixes.some((k) => key === k || key.startsWith(`${k}/`) || key.startsWith(`${k}:`));

/** Sentence case for a figure's label. */
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** A drawing and the camera it was drawn with: a frame to blend from. */
interface Still {
  prims: Prim[];
  cam: Camera;
  k: number;
}

type Transition = {
  start: number;
  duration: number;
  then?: () => void;
} & (
  | { kind: "zoom"; parent: View; child: View; place: Placement; inward: boolean }
  | { kind: "fade"; from: View; to: View }
  | {
      kind: "blend";
      from: Still;
      to: () => Still;
      /** For a family blend: the drawing whose words show from the first frame. */
      words?: () => Prim[];
    }
);

/** A family member's words that change with its setting: the lanes, the key and the block's note. */
const WORDS = ["count", "key", "note/block"] as const;

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const band = (t: number, a: number, b: number) => Math.min(1, Math.max(0, (t - a) / (b - a)));

export function mount() {
  for (const root of document.querySelectorAll<HTMLElement>("[data-ink-window]")) {
    const narrow = matchMedia("(max-width: 640px)");
    const button = root.querySelector<HTMLButtonElement>(".ink-mobile-open");
    const activate = () => {
      if (root.hasAttribute("data-ink-interactive")) return;
      root.setAttribute("data-ink-interactive", "");
      if (button) {
        button.setAttribute("aria-expanded", "true");
        button.hidden = true;
      }
      const stage = root.querySelector<HTMLElement>(".ink-stage");
      stage?.setAttribute(
        "aria-label",
        "The interactive drawing; on a narrow screen it scrolls sideways",
      );
      stage?.setAttribute("tabindex", "0");
      new InkWindow(root);
    };
    if (!narrow.matches) activate();
    else {
      if (button) button.hidden = false;
      button?.addEventListener("click", () => {
        activate();
        root.querySelector<HTMLElement>(".ink-stage")?.focus({ preventScroll: true });
      });
      narrow.addEventListener("change", () => {
        if (!narrow.matches) activate();
      });
    }
  }
}

class InkWindow {
  private views = new Map<string, View>();
  private designs = new Map<string, number>();
  private model: Model;
  private stack: Entry[];
  private turn = { yaw: 0, pitch: 0 };
  private transition: Transition | null = null;
  private hover: Openable | null = null;
  private palette: Palette;
  private width = 0;
  private height = 0;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly reduce = matchMedia("(prefers-reduced-motion: reduce)");
  private loading = false;

  /** Each film figure, once loaded, with its clock. */
  private reels = new Map<string, Reel>();
  /** See details: the reader's choice for every film. */
  private details = false;
  /** The chapter's ledger, called up by See details mid-chapter, shown until the reader plays on. */
  private ledger = false;
  /** The family's module, once loaded, and a setting waiting for the drawing to arrive. */
  private family: Family | null = null;
  private pending: Setting | null = null;
  /** Bands and a policy the reader chose while the drawing moved, with the film they were for. */
  private pendingWalls: Array<{ reel: Reel; change: (r: Reel) => void }> = [];
  private ticking = false;
  /** The film starts by itself once per page, the first time the window is in view. */
  private autoplayed = false;
  private inView = false;
  /** The reader scrolled the stage sideways during the beat on show: it stays where they put it. */
  private readerScrolled = false;

  constructor(private readonly root: HTMLElement) {
    this.model = MODELS[0]!;
    this.stack = this.path(this.model);
    this.canvas = root.querySelector("canvas")!;
    this.ctx = this.canvas.getContext("2d")!;
    this.palette = this.readPalette();
    this.wire();
    void document.fonts.ready.then(() => {
      root.querySelector(".ink-still")?.setAttribute("hidden", "");
      this.canvas.hidden = false;
      for (const el of root.querySelectorAll(".ink-open, .ink-models"))
        el.removeAttribute("hidden");
      this.resize();
      this.chrome();
      this.watch();
    });
  }

  private get top(): Entry {
    return this.stack[this.stack.length - 1]!;
  }

  /** The view on show: for a film, the view its current beat draws. */
  private get current(): View {
    return this.top.film ? this.filmFrame().to : this.top.view;
  }

  private get inFilm(): boolean {
    return this.top.film !== undefined;
  }

  /** The film on show, if a film is. */
  private get reel(): Reel | null {
    return this.reelOf(this.top);
  }

  private reelOf(e: Entry): Reel | null {
    return e.film ? (this.reels.get(e.film) ?? null) : null;
  }

  /** The bands a film's reader hid, as the keys they are drawn under. */
  private hidden(r: Reel): string[] {
    if (!r.hide.size || !("WALLS" in r.mod)) return [];
    return r.mod.WALLS.filter((w) => r.hide.has(w.id)).flatMap((w) => [...w.keys]);
  }

  private options(r: Reel): Options {
    return {
      details: this.details,
      ledger: this.ledger ? 1 : 0,
      hide: this.hidden(r),
      policy: r.policy,
    };
  }

  /** The boxes that open from what is on show; a box in a hidden band does not. */
  private get openables(): Openable[] {
    const all = this.current.sketch.openables;
    const r = this.reel;
    const hide = r ? this.hidden(r) : [];
    return hide.length ? all.filter((o) => !o.key || !under(o.key, hide)) : all;
  }

  /** The model the drawing shows: the current view's own, or else the one chosen. */
  private get shown(): Model {
    return MODELS.find((m) => m.id === this.current.model) ?? this.model;
  }

  private get(id: string): View {
    let v = this.views.get(id);
    if (!v) {
      v = view(id);
      // The family's members are as many as the reader's settings; only its start is kept.
      if (!id.startsWith("fam/model/")) this.views.set(id, v);
    }
    return v;
  }

  /** The scale a view was designed at: its camera at the plate's full width. */
  private design(v: View): number {
    let d = this.designs.get(v.id);
    if (d === undefined) {
      d = fit(framing(v), v.yaw, v.pitch, REF.width, REF.height, REF.pad, v.maxScale).scale;
      this.designs.set(v.id, d);
    }
    return d;
  }

  /** A model's views from its training step down to the one it starts on. */
  private path(m: Model): Entry[] {
    const out: Entry[] = [];
    for (const id of m.path) {
      const v = this.get(id);
      const via = out.length
        ? out[out.length - 1]!.view.sketch.openables.find((o) => o.child === id)
        : undefined;
      out.push(via ? { view: v, via } : { view: v });
    }
    return out;
  }

  private readPalette(): Palette {
    const css = getComputedStyle(this.root);
    const v = (name: string) => css.getPropertyValue(name).trim();
    return {
      paper: v("--paper"),
      ink: v("--ink"),
      indigo: v("--ink-indigo"),
      cinnabar: v("--ink-cinnabar"),
      ochre: v("--accent"),
      verdigris: v("--ink-verdigris"),
      faint: v("--ink-faint"),
    };
  }

  private camera(v: View): Camera {
    return frame(
      framing(v),
      v.yaw + this.turn.yaw,
      v.pitch + this.turn.pitch,
      this.width,
      this.height,
      v.maxScale,
    ).cam;
  }

  /** Where a child view sits inside the face of the box it opens from. */
  private placement(child: View, face: V3[]): Placement {
    let cx0 = Infinity;
    let cx1 = -Infinity;
    let cy0 = Infinity;
    let cy1 = -Infinity;
    for (const p of child.sketch.prims)
      for (const q of p.kind === "stroke" || p.kind === "face" ? p.pts : [p.at]) {
        cx0 = Math.min(cx0, q[0]);
        cx1 = Math.max(cx1, q[0]);
        cy0 = Math.min(cy0, q[1]);
        cy1 = Math.max(cy1, q[1]);
      }
    const fx0 = Math.min(...face.map((q) => q[0]));
    const fx1 = Math.max(...face.map((q) => q[0]));
    const fy0 = Math.min(...face.map((q) => q[1]));
    const fy1 = Math.max(...face.map((q) => q[1]));
    const fz = Math.max(...face.map((q) => q[2]));
    const s = 0.86 * Math.min((fx1 - fx0) / (cx1 - cx0), (fy1 - fy0) / (cy1 - cy0));
    return {
      s,
      offset: [
        (fx0 + fx1) / 2 - s * ((cx0 + cx1) / 2),
        (fy0 + fy1) / 2 - s * ((cy0 + cy1) / 2),
        fz + 0.05,
      ],
    };
  }

  /** The child's own camera, expressed in the parent's coordinates. */
  private childCameraInParent(child: View, place: Placement): Camera {
    const c = this.camera(child);
    const k = c.scale / place.s;
    const [ax, ay] = rotate(place.offset, c.yaw, c.pitch);
    return { yaw: c.yaw, pitch: c.pitch, scale: k, ox: c.ox - k * ax, oy: c.oy + k * ay };
  }

  private resize() {
    const box = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    this.width = box.width;
    this.height = box.height;
    this.canvas.width = Math.round(box.width * dpr);
    this.canvas.height = Math.round(box.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  // ------------------------------------------------------------------------------ the film

  /** A film's frame now: the film on show, or the one an entry below it holds. */
  private filmFrame(e: Entry = this.top): FilmFrame {
    const r = this.reelOf(e)!;
    return r.mod.frameAt(r.clock.t, this.options(r));
  }

  /** The film's frame as a drawing with its camera. */
  private filmStill(): Still {
    const f = this.filmFrame();
    const cam = cameraBetween(this.camera(f.from), this.camera(f.to), f.u);
    const design = this.design(f.from) * Math.pow(this.design(f.to) / this.design(f.from), f.u);
    return { prims: f.prims, cam, k: cam.scale / design };
  }

  /** What is on show now, as a still to blend from. */
  private still(): Still {
    if (this.inFilm) return this.filmStill();
    const cam = this.camera(this.current);
    return { prims: this.current.sketch.prims, cam, k: cam.scale / this.design(this.current) };
  }

  /** Run the film's clock: one animation frame at a time, while it plays. */
  private play() {
    const r = this.reel;
    if (!r) return;
    const { total } = r.mod.film();
    if (r.clock.t >= total) r.clock.t = 0;
    r.clock.playing = true;
    r.clock.held = false;
    r.clock.last = performance.now();
    if (!this.ticking) {
      this.ticking = true;
      requestAnimationFrame((n) => this.tick(n));
    }
    this.controls();
  }

  private pause(byReader: boolean) {
    const r = this.reel;
    if (!r) return;
    r.clock.playing = false;
    if (byReader) r.clock.held = true;
    this.controls();
    this.draw();
  }

  private tick(now: number) {
    const r = this.reel;
    if (!r || !r.clock.playing) {
      this.ticking = false;
      return;
    }
    const { mod: f, clock } = r;
    if (this.transition) {
      // A blend or zoom has the canvas; the clock waits for it.
      clock.last = now;
      requestAnimationFrame((n) => this.tick(n));
      return;
    }
    const dt = Math.min(100, Math.max(0, now - clock.last));
    clock.last = now;
    const before = clock.t;
    let t = before + dt;
    const end = f.chapterEnd(f.chapterAt(before));
    const { total } = f.film();
    // With details on, the film stops at the end of each chapter for its ledger.
    if (this.details && before < end && t >= end) {
      t = end;
      clock.playing = false;
      clock.held = true;
      this.toBand();
    }
    if (t >= total) {
      t = total;
      clock.playing = false;
    }
    clock.t = t;
    this.draw();
    this.follow();
    this.rail(r);
    if (clock.playing) requestAnimationFrame((n) => this.tick(n));
    else {
      this.ticking = false;
      this.controls();
    }
  }

  /** Keep the controls, the trail and the caption in step with the film's chapter and beat. */
  private follow() {
    const r = this.reel;
    if (!r) return;
    const f = this.filmFrame();
    if (f.chapter !== r.clock.chapter) {
      r.clock.chapter = f.chapter;
      r.clock.beat = f.beat;
      this.chrome();
      this.track(f);
    } else if (f.beat !== r.clock.beat) {
      r.clock.beat = f.beat;
      this.canvas.setAttribute("aria-label", `${f.to.title}. ${f.say}`);
      this.note(f.to);
      this.track(f);
      this.sayLine();
    }
  }

  /**
   * On a stage too narrow for the drawing, the beat's sentence as page text under it, where the
   * drawing's own may be scrolled out of view; not while the band holds the chapter's ledger.
   */
  private sayLine() {
    const line = this.root.querySelector<HTMLElement>(".ink-say");
    if (!line) return;
    const r = this.reel;
    if (!r) {
      line.hidden = true;
      return;
    }
    const f = this.filmFrame();
    const ledger = this.ledger || (this.details && r.clock.t >= r.mod.chapterEnd(f.chapter) - 0.5);
    line.hidden = ledger;
    line.textContent = f.say;
  }

  /**
   * On a stage narrower than the plate, which scrolls sideways, bring into view what a new beat
   * lights: the box around the primitives under its effects' keys, as the camera projects them.
   * If the reader scrolled the stage during the beat before, it stays where they put it.
   */
  private track(f: FilmFrame) {
    const stage = this.root.querySelector<HTMLElement>(".ink-stage");
    const r = this.reel;
    if (!stage || !r || stage.scrollWidth <= stage.clientWidth + 1) return;
    if (this.readerScrolled) {
      this.readerScrolled = false;
      return;
    }
    const beat = r.mod.film().chapters[f.chapter]!.beats[f.beat]!;
    const keys = (beat.fx ?? []).flatMap((fx) => fx.keys ?? []);
    if (!keys.length) return;
    const cam = this.camera(f.to);
    let x0 = Infinity;
    let x1 = -Infinity;
    for (const p of f.to.sketch.prims)
      if (p.key !== undefined && under(p.key, keys))
        for (const q of p.kind === "stroke" || p.kind === "face" ? p.pts : [p.at]) {
          const [x] = project(q, cam);
          x0 = Math.min(x0, x);
          x1 = Math.max(x1, x);
        }
    if (x0 === Infinity) return;
    const [v0, v1] = [stage.scrollLeft, stage.scrollLeft + stage.clientWidth];
    // In view already, or wider than the stage and filling it: the stage stays.
    if ((x0 >= v0 && x1 <= v1) || (x0 <= v0 && x1 >= v1)) return;
    const left = Math.max(
      0,
      Math.min((x0 + x1) / 2 - stage.clientWidth / 2, stage.scrollWidth - stage.clientWidth),
    );
    stage.scrollTo({ left, behavior: this.reduce.matches ? "auto" : "smooth" });
  }

  /** The kicker over the drawing and the note under it: what the view on show rests on. */
  private note(v: View) {
    const kicker = this.root.querySelector(".ink-kicker");
    if (kicker) kicker.textContent = v.kicker ?? this.shown.kicker;
    const evidence = this.root.querySelector(".ink-evidence");
    if (evidence) evidence.textContent = EVIDENCE[evidenceOf(v)];
  }

  /** A film figure's reel: loaded once, then kept with its clock. */
  private async reelFor(f: Figure): Promise<Reel> {
    const had = this.reels.get(f.id);
    if (had) return had;
    const mod = (await loadFigure(f)) as FigureFilm;
    const r = this.reels.get(f.id) ?? {
      figure: f,
      mod,
      clock: { t: 0, playing: false, held: false, last: 0, chapter: -1, beat: -1 },
      hide: new Set<string>(),
      policy: "StoreAll" as const,
    };
    this.reels.set(f.id, r);
    return r;
  }

  /** Begin a film from the view that offers it, or resume it where it was left. */
  private async enterFilm(f: Figure, o: { auto?: boolean } = {}) {
    if (this.transition || this.loading) return;
    this.clearLoadError();
    this.loading = true;
    let r: Reel;
    try {
      r = await this.reelFor(f);
    } catch {
      this.loadError();
      return;
    } finally {
      this.loading = false;
    }
    if (this.transition || this.inFilm) return;
    const { mod: film, clock } = r;
    const reduce = this.reduce.matches;
    if (reduce) {
      // Reduced motion: still chapters, each at its end, stepped with Back and Next.
      if (clock.t <= 0 || clock.t >= film.film().total) clock.t = film.chapterEnd(0);
      clock.held = true;
    } else if (clock.t >= film.film().total) clock.t = 0;
    const from = this.still();
    this.stack.push({ view: this.top.view, film: f.id });
    // A film that plays by itself does not speak: its caption is silent until the reader acts.
    this.voice(!o.auto || reduce);
    clock.chapter = -1;
    this.follow();
    if (clock.t > 0 && !reduce) this.blendTo(from, BLEND, () => this.play());
    else if (!reduce) this.play();
    else this.draw();
    if (!o.auto) this.focusControls();
  }

  /**
   * Play on, from a pause, from the end (from the start again, blended), or from a details stop.
   * A ledger the reader called up mid-chapter gives the sentence back first.
   */
  private resume() {
    const r = this.reel;
    if (!r || this.reduce.matches || this.transition) return;
    if (r.clock.t >= r.mod.film().total) {
      const from = this.still();
      this.ledger = false;
      r.clock.t = 0;
      this.follow();
      this.blendTo(from, BLEND, () => this.play());
      return;
    }
    if (this.ledger) {
      const from = this.still();
      this.ledger = false;
      this.blendTo(from, BLEND * 0.85, () => this.play());
      this.controls();
      return;
    }
    this.play();
  }

  /**
   * See details: the film stops at once and its sentence gives way to the chapter's ledger, and
   * while details are on it also stops at each chapter's end with that chapter's ledger. Hide
   * details gives the sentence back and plays on.
   */
  private toggleDetails() {
    const r = this.reel;
    if (r && this.transition) return;
    this.details = !this.details;
    if (!r) {
      this.controls();
      return;
    }
    const { mod: f, clock } = r;
    const total = f.film().total;
    const end = f.chapterEnd(f.chapterAt(clock.t));
    // At a chapter's end the ledger comes with details by itself; elsewhere it is called up.
    const atEnd = clock.t >= end - 0.5;
    const from = this.still();
    if (this.details) {
      clock.playing = false;
      clock.held = true;
      this.ledger = !atEnd;
      this.toBand();
    } else this.ledger = false;
    const playOn = !this.details && !this.reduce.matches && clock.t < total;
    this.blendTo(from, this.reduce.matches ? 0 : BLEND * 0.85, () =>
      playOn ? this.play() : this.controls(),
    );
    this.controls();
  }

  /** On a stage too narrow for the drawing, show the band under it, where the ledger is set. */
  private toBand() {
    const stage = this.root.querySelector<HTMLElement>(".ink-stage");
    if (stage && stage.scrollWidth > stage.clientWidth + 1)
      stage.scrollTo({ left: 0, behavior: this.reduce.matches ? "auto" : "smooth" });
  }

  /** Whether the caption is announced as it changes: silent while a film plays by itself. */
  private voice(on: boolean) {
    this.root.querySelector(".ink-caption")?.setAttribute("aria-live", on ? "polite" : "off");
  }

  /** Leave the film for the view it was entered from; the clock keeps its place. */
  private leaveFilm(then?: () => void) {
    const r = this.reel;
    if (!r) return;
    const from = this.still();
    r.clock.playing = false;
    this.stack.pop();
    this.blendTo(from, this.reduce.matches ? 0 : BLEND * 1.2, then);
  }

  /** Jump to a time in the film, blending from what is on show. */
  private jump(t: number, o: { thenPlay?: boolean } = {}) {
    const r = this.reel;
    if (!r || this.transition) return;
    const from = this.still();
    this.ledger = false;
    r.clock.playing = false;
    r.clock.t = Math.max(0, Math.min(t, r.mod.film().total));
    this.follow();
    this.blendTo(from, this.reduce.matches ? 0 : BLEND, () => {
      if (o.thenPlay) this.play();
      else this.controls();
    });
  }

  /** Go one chapter forward or back. */
  private chapterStep(delta: 1 | -1) {
    const r = this.reel;
    if (!r || this.transition) return;
    // Stepping is the reader's own move: the stage follows the chapter they step to.
    this.readerScrolled = false;
    const { mod: f, clock } = r;
    const { starts, chapters } = f.film();
    const t = clock.t;
    const ci = f.chapterAt(t);
    if (this.reduce.matches) {
      // Still chapters: each shown at its end.
      const to = Math.max(0, Math.min(chapters.length - 1, ci + delta));
      if (to !== ci) this.jump(f.chapterEnd(to));
      return;
    }
    const going = clock.playing || !clock.held;
    if (delta > 0) {
      if (ci + 1 >= chapters.length) return;
      // Held at a chapter's end, Next goes on into the next chapter as the film would.
      if (t >= f.chapterEnd(ci) - 0.5) {
        clock.t = f.chapterEnd(ci) + 0.5;
        this.play();
      } else this.jump(starts[ci + 1]! + 0.5, { thenPlay: true });
      return;
    }
    // Back: to this chapter's start, or the previous one's if this one has just begun.
    const to = t - starts[ci]! > 1200 || ci === 0 ? ci : ci - 1;
    this.jump(starts[to]! + 0.5, { thenPlay: going });
  }

  /**
   * The search's film starts when the window is half in view, or fills half the screen; a film
   * pauses when the window leaves the view or the tab hides, and goes on when it returns unless
   * the reader paused it.
   */
  private watch() {
    const figure = FIGURES.find((f) => f.autoplay);
    const update = () => {
      const visible = this.inView && document.visibilityState === "visible";
      const r = this.reel;
      if (!visible) {
        if (r?.clock.playing) this.pause(false);
        return;
      }
      if (this.reduce.matches) return;
      // Small screens begin with an overview; their films are always an explicit choice.
      const canAutoplay = !matchMedia("(max-width: 640px)").matches;
      if (
        canAutoplay &&
        figure &&
        !this.autoplayed &&
        this.top.view.id === figure.from &&
        !this.transition
      ) {
        this.autoplayed = true;
        void this.enterFilm(figure, { auto: true });
      } else if (r && !r.clock.playing && !r.clock.held && !this.transition) this.play();
    };
    // In view: half the window, or half the screen where the window is taller than the screen,
    // as on a phone, where half the window may never fit.
    new IntersectionObserver(
      (entries) => {
        for (const e of entries)
          this.inView =
            e.isIntersecting &&
            e.intersectionRect.height >=
              0.5 * Math.min(e.boundingClientRect.height, innerHeight) - 1;
        update();
      },
      { threshold: Array.from({ length: 21 }, (_, i) => i / 20) },
    ).observe(this.root);
    document.addEventListener("visibilitychange", update);
  }

  // ---------------------------------------------------------------------------- drawing

  /** Draw a view with a camera; its ink scales with the camera against the view's design. */
  private paint(v: View, cam: Camera, alpha: number) {
    drawCanvas(this.ctx, v.sketch.prims, cam, this.palette, alpha, cam.scale / this.design(v));
  }

  private draw(now = performance.now()) {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.palette.paper;
    ctx.fillRect(0, 0, this.width, this.height);
    const tr = this.transition;
    if (!tr) {
      if (this.inFilm) {
        const s = this.filmStill();
        drawCanvas(ctx, s.prims, s.cam, this.palette, 1, s.k);
        if (this.hover) this.outline(this.hover.face, s.cam);
        return;
      }
      const cam = this.camera(this.current);
      this.paint(this.current, cam, 1);
      if (this.hover) this.outline(this.hover.face, cam);
      return;
    }
    const raw = tr.duration ? Math.min(1, (now - tr.start) / tr.duration) : 1;
    if (tr.kind === "fade") {
      const t = ease(raw);
      this.paint(tr.from, this.camera(tr.from), 1 - band(t, 0, 0.6));
      this.paint(tr.to, this.camera(tr.to), band(t, 0.4, 1));
    } else if (tr.kind === "blend") {
      const u = ease(raw);
      const to = tr.to();
      const cam = cameraBetween(tr.from.cam, to.cam, u);
      const k = tr.from.k * Math.pow(to.k / tr.from.k, u);
      if (tr.words) {
        // A member's numbers are the setting the sliders show, from the first frame; only the
        // drawing moves, so no in-between member's count is ever printed.
        const word = (p: Prim) => p.kind === "label" && p.key !== undefined && under(p.key, WORDS);
        const moving = rewriteFrame(
          tr.from.prims.filter((p) => !word(p)),
          to.prims.filter((p) => !word(p)),
          u,
        );
        drawCanvas(ctx, [...moving, ...tr.words().filter(word)], cam, this.palette, 1, k);
      } else drawCanvas(ctx, rewriteFrame(tr.from.prims, to.prims, u), cam, this.palette, 1, k);
    } else {
      const t = ease(tr.inward ? raw : 1 - raw);
      const a = this.camera(tr.parent);
      const b = this.childCameraInParent(tr.child, tr.place);
      const cam = mix(a, b, t, this.width, this.height);
      const placed = mapPrims(tr.child.sketch.prims, (p) => [
        tr.place.offset[0] + tr.place.s * p[0],
        tr.place.offset[1] + tr.place.s * p[1],
        tr.place.offset[2] + tr.place.s * p[2],
      ]);
      this.paint(tr.parent, cam, 1 - band(t, 0.35, 0.92));
      drawCanvas(
        ctx,
        placed,
        cam,
        this.palette,
        band(t, 0.2, 0.8),
        (cam.scale * tr.place.s) / this.design(tr.child),
      );
    }
    if (raw < 1) {
      requestAnimationFrame((n) => this.draw(n));
    } else {
      this.transition = null;
      tr.then?.();
      this.chrome();
      this.draw();
      // A setting the reader chose while the drawing moved: carry it there now.
      const next = this.pending;
      this.pending = null;
      if (next) queueMicrotask(() => this.familyTo(next));
      // Bands and a policy chosen while the drawing moved, for the film still on show.
      const walls = this.pendingWalls.splice(0).filter((w) => w.reel === this.reel);
      if (walls.length)
        queueMicrotask(() => this.setWalls((r) => walls.forEach((w) => w.change(r))));
    }
  }

  private outline(face: V3[], cam: Camera) {
    const { ctx } = this;
    ctx.strokeStyle = this.palette.ochre;
    ctx.lineWidth = 2.4;
    ctx.setLineDash([]);
    ctx.beginPath();
    face
      .map((q) => project(q, cam))
      .forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
  }

  private begin(t: Transition) {
    this.transition = t;
    this.hover = null;
    this.canvas.style.cursor = "";
    requestAnimationFrame((n) => this.draw(n));
  }

  /** Blend from a still into whatever is on show once the state has changed. */
  private blendTo(from: Still, duration: number, then?: () => void, words?: () => Prim[]) {
    this.begin({
      kind: "blend",
      from,
      to: () => this.still(),
      start: performance.now(),
      duration,
      ...(then ? { then } : {}),
      ...(words ? { words } : {}),
    });
  }

  private open(o: Openable) {
    if (this.transition) return;
    if (this.reel?.clock.playing) this.pause(true);
    const child = this.get(o.child);
    const parent = this.current;
    this.begin({
      kind: "zoom",
      parent,
      child,
      place: this.placement(child, o.face),
      inward: true,
      start: performance.now(),
      duration: this.reduce.matches ? 0 : DURATION,
      then: () => this.stack.push({ view: child, via: o }),
    });
  }

  /** Close the current view into the box it was opened from; repeat until `depth` is shown. */
  private up(depth = this.stack.length - 2) {
    if (this.transition || this.stack.length < 2 || depth >= this.stack.length - 1) return;
    const top = this.top;
    const more = depth < this.stack.length - 2;
    const next = () => {
      if (more) queueMicrotask(() => this.up(depth));
    };
    if (top.film) {
      this.leaveFilm(next);
      return;
    }
    const below = this.stack[this.stack.length - 2]!;
    const parent = below.film ? this.filmFrame(below).to : below.view;
    if (!top.via) {
      this.begin({
        kind: "fade",
        from: top.view,
        to: parent,
        start: performance.now(),
        duration: this.reduce.matches ? 0 : DURATION * 0.7,
        then: () => {
          this.stack.pop();
          next();
        },
      });
      return;
    }
    this.begin({
      kind: "zoom",
      parent,
      child: top.view,
      place: this.placement(top.view, top.via.face),
      inward: false,
      start: performance.now(),
      duration: this.reduce.matches ? 0 : more ? DURATION * 0.6 : DURATION,
      then: () => {
        this.stack.pop();
        next();
      },
    });
  }

  private async switchTo(m: Model) {
    if (this.transition || this.loading || m === this.shown) return;
    this.clearLoadError();
    this.loading = true;
    try {
      await load(m);
    } catch {
      this.loadError();
      return;
    } finally {
      this.loading = false;
    }
    if (this.transition) return;
    const from = this.still();
    if (this.reel?.clock.playing) this.pause(true);
    this.model = m;
    this.stack = this.path(m);
    this.turn = { yaw: 0, pitch: 0 };
    this.blendTo(from, this.reduce.matches ? 0 : DURATION * 0.7);
  }

  // ---------------------------------------------------------------------------- figures

  private clearLoadError() {
    this.root.querySelector<HTMLElement>(".ink-load-error")?.setAttribute("hidden", "");
  }

  /** Keep the current drawing usable when an optional model or figure chunk fails. */
  private loadError() {
    const message = this.root.querySelector<HTMLElement>(".ink-load-error");
    if (!message) return;
    message.hidden = false;
    const button = message.querySelector<HTMLButtonElement>("button");
    // Browsers may cache a rejected module import; a page reload also recovers stale deploy chunks.
    if (button) button.onclick = () => location.reload();
  }

  /** Open a figure of views at its first step, as a new level of the drawing. */
  private async openFigure(f: Figure) {
    if (this.transition || this.loading) return;
    this.clearLoadError();
    this.loading = true;
    try {
      const mod = await loadFigure(f);
      if (f.id === "family") this.family = mod as Family;
    } catch {
      this.loadError();
      return;
    } finally {
      this.loading = false;
    }
    if (this.transition) return;
    if (this.reel?.clock.playing) this.pause(true);
    const from = this.current;
    const to = this.get(f.steps[0]!);
    this.begin({
      kind: "fade",
      from,
      to,
      start: performance.now(),
      duration: this.reduce.matches ? 0 : DURATION * 0.7,
      then: () => {
        this.stack.push({ view: to });
        queueMicrotask(() =>
          this.root
            .querySelector<HTMLElement>(".ink-family input")
            ?.focus({ preventScroll: true }),
        );
      },
    });
  }

  /** Carry the family's drawing to another member; settings that come mid-way wait their turn. */
  private familyTo(s: Setting) {
    const fam = this.family;
    if (!fam || !this.top.view.id.startsWith("fam/model")) return;
    if (this.transition) {
      this.pending = s;
      return;
    }
    const id = fam.familyId(s);
    if (id === this.top.view.id) return;
    const from = this.still();
    this.stack[this.stack.length - 1] = { view: this.get(id) };
    // The words follow the sliders: a setting chosen while the drawing moves shows at once, and
    // the drawing follows it when this blend ends.
    let words: { id: string; prims: Prim[] } | null = null;
    const shown = () => {
      const want = fam.familyId(this.pending ?? s);
      if (words?.id !== want) words = { id: want, prims: this.get(want).sketch.prims };
      return words.prims;
    };
    this.blendTo(from, this.reduce.matches ? 0 : 340, undefined, shown);
  }

  /** The setting the family's sliders show. */
  private sliders(): Setting {
    const s = { V: 0, L: 0, H: 0, K: 0, F: 0 };
    for (const input of this.root.querySelectorAll<HTMLInputElement>(".ink-family input")) {
      const k = input.dataset.control as keyof Setting;
      s[k] = Number(input.value);
    }
    return s;
  }

  /** Set the sliders and their read-outs to a setting, and the member's counts beside them. */
  private showSetting(s: Setting) {
    for (const input of this.root.querySelectorAll<HTMLInputElement>(".ink-family input")) {
      const k = input.dataset.control as keyof Setting;
      input.value = String(s[k]);
      const out = input.parentElement?.querySelector("output");
      if (out) out.textContent = s[k].toLocaleString("en-US");
    }
    const list = this.root.querySelector(".ink-counts");
    if (list && this.family)
      list.replaceChildren(
        ...this.family.counts(s).map((c) => {
          const row = document.createElement("div");
          const exact = Object.assign(document.createElement("span"), {
            className: "ink-exact",
            textContent: c.exact,
          });
          const value = document.createElement("dd");
          value.append(c.words, " ", exact);
          row.append(Object.assign(document.createElement("dt"), { textContent: c.name }), value);
          return row;
        }),
      );
  }

  /**
   * Show or hide a band of C, or choose its policy, carrying the drawing to the new state. A choice
   * made while the drawing moves waits for it to arrive, then applies, so no click is lost.
   */
  private setWalls(change: (r: Reel) => void) {
    const r = this.reel;
    if (!r) return;
    if (this.transition) {
      this.pendingWalls.push({ reel: r, change });
      return;
    }
    const from = this.still();
    change(r);
    this.blendTo(from, this.reduce.matches ? 0 : 420);
    this.controls();
  }

  // ---------------------------------------------------------------------------- controls

  /** Move focus to the film's first control, when the reader started it. */
  private focusControls() {
    queueMicrotask(() =>
      this.root
        .querySelector<HTMLElement>(".ink-play:not([hidden]), .ink-step-next")
        ?.focus({ preventScroll: true }),
    );
  }

  /** A film's own controls: play or pause, details, the chapter, Back and Next, and C's bands. */
  private controls() {
    const steps = this.root.querySelector<HTMLElement>(".ink-steps");
    if (!steps) return;
    const r = this.reel;
    steps.toggleAttribute("hidden", !r);
    this.walls(r);
    if (!r) {
      this.sayLine();
      return;
    }
    const { mod: f, clock } = r;
    // A button the reader is on may be hidden or disabled below: then focus moves to another.
    const had = document.activeElement;
    steps.setAttribute("aria-label", `${sentence(r.figure.label)}, chapter by chapter`);
    const { chapters, total, starts } = f.film();
    const ci = f.chapterAt(clock.t);
    const reduce = this.reduce.matches;
    const play = steps.querySelector<HTMLButtonElement>(".ink-play")!;
    play.toggleAttribute("hidden", reduce);
    const ended = clock.t >= total;
    // Stopped for details, by See details or at a chapter's end, the film waits to go on.
    const waiting =
      !clock.playing &&
      !ended &&
      (this.ledger || (this.details && clock.t >= f.chapterEnd(ci) - 0.5));
    play.textContent = clock.playing ? "Pause" : ended ? "Replay" : waiting ? "Continue" : "Play";
    const details = steps.querySelector<HTMLButtonElement>(".ink-details")!;
    details.textContent = this.details ? "Hide details" : "See details";
    this.rail(r);
    this.sayLine();
    const count = steps.querySelector(".ink-step-count");
    if (count) count.textContent = `Chapter ${ci + 1} of ${chapters.length}`;
    steps.querySelector<HTMLButtonElement>(".ink-step-back")!.disabled =
      ci === 0 && (reduce || clock.t - starts[0]! < 1200);

    const back = steps.querySelector<HTMLButtonElement>(".ink-step-back")!;
    const next = steps.querySelector<HTMLButtonElement>(".ink-step-next")!;
    next.disabled = ci === chapters.length - 1;
    if (had instanceof HTMLButtonElement && steps.contains(had) && (had.disabled || had.hidden))
      [play, next, back].find((b) => !b.hidden && !b.disabled)?.focus({ preventScroll: true });
    const caption = this.root.querySelector(".ink-caption");
    if (caption) {
      const c = chapters[ci]!;
      caption.textContent = this.details ? c.caption : `${c.title}. ${c.beats[0]!.say}`;
    }
  }

  /**
   * The film's rail: one segment per chapter, as long as the chapter, filled to the film's time.
   * With details on, each segment's end is marked, where the film will stop.
   */
  private rail(r: Reel) {
    const rail = this.root.querySelector<HTMLElement>(".ink-rail");
    if (!rail) return;
    const { mod: f, clock } = r;
    const { chapters, starts } = f.film();
    if (rail.dataset.figure !== r.figure.id) {
      rail.dataset.figure = r.figure.id;
      rail.replaceChildren(
        ...chapters.map((_, i) =>
          Object.assign(document.createElement("span"), {
            style: `flex-grow: ${f.chapterEnd(i) - starts[i]!}`,
          }),
        ),
      );
    }
    rail.classList.toggle("ink-rail-stops", this.details);
    [...rail.children].forEach((s, i) => {
      const fill = Math.min(
        1,
        Math.max(0, (clock.t - starts[i]!) / (f.chapterEnd(i) - starts[i]!)),
      );
      const v = fill.toFixed(4);
      const el = s as HTMLElement;
      if (el.style.getPropertyValue("--fill") !== v) el.style.setProperty("--fill", v);
    });
  }

  /** At the end of a film with walls, a toggle for each band and a switch for the policy. */
  private walls(r: Reel | null) {
    const box = this.root.querySelector<HTMLElement>(".ink-walls");
    if (!box) return;
    const ended = r !== null && r.clock.t >= r.mod.film().total;
    const walls = r && "WALLS" in r.mod ? r.mod.WALLS : null;
    box.toggleAttribute("hidden", !walls || !ended);
    if (!r || !walls || !ended) return;
    if (box.dataset.figure !== r.figure.id) {
      box.dataset.figure = r.figure.id;
      const bands = document.createElement("span");
      bands.className = "ink-walls-bands";
      bands.append(
        Object.assign(document.createElement("span"), { textContent: "Show" }),
        ...walls.map((w) => {
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.wall = w.id;
          b.textContent = w.label;
          b.addEventListener("click", () =>
            this.setWalls((x) => (x.hide.has(w.id) ? x.hide.delete(w.id) : x.hide.add(w.id))),
          );
          return b;
        }),
      );
      const policy = document.createElement("span");
      policy.className = "ink-walls-policy";
      policy.setAttribute("role", "group");
      policy.setAttribute("aria-label", "Residual policy");
      policy.append(
        Object.assign(document.createElement("span"), { textContent: "Residuals" }),
        ...(["StoreAll", "Remat"] as const).map((p) => {
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.policy = p;
          b.textContent = p;
          b.addEventListener("click", () =>
            this.setWalls((x) => {
              x.policy = p;
            }),
          );
          return b;
        }),
      );
      box.replaceChildren(bands, policy);
    }
    for (const b of box.querySelectorAll<HTMLButtonElement>("[data-wall]"))
      b.setAttribute("aria-pressed", String(!r.hide.has(b.dataset.wall!)));
    for (const b of box.querySelectorAll<HTMLButtonElement>("[data-policy]"))
      b.setAttribute("aria-pressed", String(r.policy === b.dataset.policy));
  }

  /** The model switch, the trail of views, the caption and one button per box that opens. */
  private chrome() {
    const v = this.current;
    // The buttons are rebuilt below; focus inside the window stays inside it.
    const active = document.activeElement;
    const keep = active !== null && this.root.contains(active) && !active.closest(".ink-models");
    for (const b of this.root.querySelectorAll<HTMLButtonElement>(".ink-models button"))
      b.setAttribute("aria-pressed", String(b.dataset.model === this.shown.id));
    this.note(v);
    const list = this.root.querySelector(".ink-crumbs ol");
    if (list) {
      list.replaceChildren(
        ...this.stack.map((entry, i, all) => {
          const li = document.createElement("li");
          // A film is named by its chapter, which holds while its beats move the drawing.
          const r = this.reelOf(entry);
          const title = r
            ? r.mod.film().chapters[r.mod.chapterAt(r.clock.t)]!.title
            : entry.view.title;
          if (i === all.length - 1) {
            const span = document.createElement("span");
            span.setAttribute("aria-current", "location");
            span.textContent = title;
            li.append(span);
          } else {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = title;
            button.addEventListener("click", () => this.up(i));
            li.append(button);
          }
          return li;
        }),
      );
      // The trail keeps to one line; where it is longer than the plate, show its end.
      list.scrollLeft = list.scrollWidth;
    }
    const caption = this.root.querySelector(".ink-caption");
    if (caption) caption.textContent = v.caption;
    this.canvas.setAttribute(
      "aria-label",
      this.inFilm ? `${v.title}. ${this.filmFrame().say}` : `${v.title}. ${v.caption}`,
    );
    const bar = this.root.querySelector(".ink-open-buttons");
    if (bar) {
      const seen = new Set<string>();
      const buttons = this.openables
        .filter((o) => !seen.has(o.child) && seen.add(o.child))
        .map((o) => {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = o.label;
          b.addEventListener("click", () => this.open(o));
          return b;
        });
      bar.replaceChildren(...buttons);
      // A band the reader hid takes its boxes with it. Then the row keeps its place, empty, so the
      // controls under it do not move from under the pointer.
      const film = this.reel;
      const reserved =
        buttons.length === 0 &&
        film !== null &&
        this.hidden(film).length > 0 &&
        this.current.sketch.openables.length > 0;
      const row = this.root.querySelector(".ink-open");
      row?.toggleAttribute("hidden", buttons.length === 0 && !reserved);
      row?.classList.toggle("ink-open-reserved", reserved);
    }
    // The figures this view offers: a film plays, a figure of views opens.
    const offers = this.inFilm ? [] : FIGURES.filter((f) => f.from === v.id);
    const offer = this.root.querySelector<HTMLElement>(".ink-figure");
    if (offer) {
      offer.toggleAttribute("hidden", offers.length === 0);
      offer.replaceChildren(
        ...offers.map((f) => {
          const item = document.createElement("span");
          item.className = "ink-figure-item";
          const lead = document.createElement("span");
          const r = this.reels.get(f.id);
          const total = r?.mod.film().total ?? 0;
          lead.textContent =
            f.kind === "views"
              ? "Open"
              : this.reduce.matches
                ? "Step through"
                : r && r.clock.t > 0 && r.clock.t < total
                  ? "Resume"
                  : "Play";
          const b = document.createElement("button");
          b.type = "button";
          b.dataset.figure = f.id;
          b.textContent = f.label;
          b.addEventListener("click", () =>
            f.kind === "film" ? void this.enterFilm(f) : void this.openFigure(f),
          );
          item.append(lead, b);
          return item;
        }),
      );
    }
    // The family's sliders, while one of its members is on show.
    const fam = this.root.querySelector<HTMLElement>(".ink-family");
    if (fam) {
      const member = this.family?.settingOf(this.top.view.id) ?? null;
      fam.toggleAttribute("hidden", member === null);
      this.root.querySelector(".ink-counts")?.toggleAttribute("hidden", member === null);
      if (member && !this.pending) this.showSetting(member);
    }
    this.controls();
    // A control that was hidden or disabled above still holds focus until the browser's next
    // update, so test for that too.
    const now = document.activeElement;
    const lost =
      !now ||
      !this.root.contains(now) ||
      now.closest("[hidden]") !== null ||
      (now instanceof HTMLButtonElement && now.disabled);
    if (keep && lost) {
      const to = this.inFilm
        ? ".ink-play:not([hidden]), .ink-step-next:not(:disabled), .ink-step-back:not(:disabled)"
        : ".ink-stage";
      this.root.querySelector<HTMLElement>(to)?.focus({ preventScroll: true });
    }
  }

  private hit(x: number, y: number): Openable | null {
    if (this.transition) return null;
    const cam = this.inFilm ? this.filmStill().cam : this.camera(this.current);
    for (const o of this.openables)
      if (
        inside(
          [x, y],
          o.face.map((q) => project(q, cam)),
        )
      )
        return o;
    return null;
  }

  private wire() {
    // The reader's first action in the window, a press, a key other than moving focus, or a
    // slider, gives the caption its voice again after a film began by itself.
    const acted = (e: Event) => {
      if (e instanceof KeyboardEvent && ["Tab", "Shift", "Control", "Alt", "Meta"].includes(e.key))
        return;
      this.voice(true);
    };
    for (const type of ["click", "keydown", "input"])
      this.root.addEventListener(type, acted, { capture: true });
    // A mouse or pen drag turns the drawing; on touch the stage scrolls
    // natively. A press that does not move opens the box under it.
    let press: {
      x: number;
      y: number;
      yaw: number;
      pitch: number;
      moved: boolean;
      turns: boolean;
    } | null = null;
    const local = (e: PointerEvent): [number, number] => {
      const r = this.canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    this.canvas.addEventListener("pointerdown", (e) => {
      const turns = e.pointerType !== "touch";
      press = {
        x: e.clientX,
        y: e.clientY,
        yaw: this.turn.yaw,
        pitch: this.turn.pitch,
        moved: false,
        turns,
      };
      if (turns) this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (press) {
        const dx = e.clientX - press.x;
        const dy = e.clientY - press.y;
        if (Math.abs(dx) + Math.abs(dy) > 6) press.moved = true;
        if (press.moved && press.turns && !this.transition) {
          this.turn.yaw = Math.max(-0.3, Math.min(0.3, press.yaw + dx * 0.004));
          this.turn.pitch = Math.max(-0.22, Math.min(0.22, press.pitch + dy * 0.003));
          this.draw();
        }
        if (press.turns) return;
      }
      if (e.pointerType === "touch") return;
      const o = this.hit(...local(e));
      if (o !== this.hover) {
        this.hover = o;
        this.canvas.style.cursor = o ? "pointer" : "";
        this.draw();
      }
    });
    this.canvas.addEventListener("pointerup", (e) => {
      const was = press;
      press = null;
      if (was && !was.moved) {
        const o = this.hit(...local(e));
        if (o) this.open(o);
      }
    });
    this.canvas.addEventListener("pointercancel", () => {
      press = null;
    });
    this.canvas.addEventListener("pointerleave", () => {
      if (this.hover) {
        this.hover = null;
        this.draw();
      }
    });
    this.root.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.stack.length > 1) {
        e.preventDefault();
        this.up();
        return;
      }
      const stage = this.root.querySelector<HTMLElement>(".ink-stage");
      // In a film, Space on the stage plays and pauses, and the arrow keys step chapters,
      // unless they would scroll a stage too narrow to show the drawing.
      if (this.inFilm && e.key === " " && e.target === stage) {
        e.preventDefault();
        if (this.reel?.clock.playing) this.pause(true);
        else this.resume();
        return;
      }
      if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && this.inFilm) {
        if (stage && e.target === stage && stage.scrollWidth > stage.clientWidth + 1) return;
        e.preventDefault();
        this.chapterStep(e.key === "ArrowRight" ? 1 : -1);
      }
    });
    this.root.querySelector(".ink-play")?.addEventListener("click", () => {
      if (this.reel?.clock.playing) this.pause(true);
      else this.resume();
    });
    this.root.querySelector(".ink-details")?.addEventListener("click", () => this.toggleDetails());
    for (const input of this.root.querySelectorAll<HTMLInputElement>(".ink-family input"))
      input.addEventListener("input", () => {
        const s = this.sliders();
        this.showSetting(s);
        this.familyTo(s);
      });
    this.root
      .querySelector(".ink-step-back")
      ?.addEventListener("click", () => this.chapterStep(-1));
    this.root
      .querySelector(".ink-step-next")
      ?.addEventListener("click", () => this.chapterStep(1));
    for (const b of this.root.querySelectorAll<HTMLButtonElement>(".ink-models button"))
      b.addEventListener("click", () => {
        const m = MODELS.find((x) => x.id === b.dataset.model);
        if (m) void this.switchTo(m);
      });
    new ResizeObserver(() => this.resize()).observe(this.canvas);
    // A stage narrower than the plate scrolls sideways: the page then repeats D's counts as text,
    // and a film's stage follows each beat until the reader scrolls it.
    const stage = this.root.querySelector<HTMLElement>(".ink-stage");
    if (stage) {
      new ResizeObserver(() =>
        this.root.classList.toggle("ink-narrow", stage.scrollWidth > stage.clientWidth + 1),
      ).observe(stage);
      const scrolled = () => {
        this.readerScrolled = true;
      };
      let touch: { x: number; y: number } | null = null;
      stage.addEventListener(
        "touchstart",
        (e) => {
          const t = e.touches[0];
          touch = t ? { x: t.clientX, y: t.clientY } : null;
        },
        { passive: true },
      );
      stage.addEventListener(
        "touchmove",
        (e) => {
          const t = e.touches[0];
          if (
            touch &&
            t &&
            Math.abs(t.clientX - touch.x) > Math.max(8, Math.abs(t.clientY - touch.y))
          )
            scrolled();
        },
        { passive: true },
      );
      stage.addEventListener(
        "wheel",
        (e) => {
          if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) scrolled();
        },
        { passive: true },
      );
      // The stage's own scrollbar, or the keys that scroll it.
      stage.addEventListener("pointerdown", (e) => {
        if (e.target === stage) scrolled();
      });
      stage.addEventListener("keydown", (e) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) scrolled();
      });
    }
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.palette = this.readPalette();
      this.draw();
    });
  }
}
