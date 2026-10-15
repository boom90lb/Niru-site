/**
 * The models the window draws, and the views of all of them by id. Each model
 * starts at its model view, one level below its training step. Monad's views
 * ship with the page; DeepSeek's load when it is first chosen, and the steps of
 * a figure when it is first opened.
 */
import { fit, REF } from "./engine.ts";
import { design } from "./kit.ts";
import * as lattice from "./lattice.ts";
import * as monad from "./monad.ts";
import { framing, type View } from "./view.ts";

export interface Model {
  id: string;
  name: string;
  kicker: string;
  /** Views from the training step down to the one the window starts on. */
  path: readonly string[];
}

export const MODELS: readonly Model[] = [
  {
    id: "monad",
    name: "Monad",
    kicker: "Monad, a small language model, written as one term",
    path: ["step", "model"],
  },
  {
    id: "dsv41",
    name: "DeepSeek-V4.1-Flash, toy",
    kicker: "A toy DeepSeek-V4.1-Flash, written as one term",
    path: ["ds/step", "ds/model"],
  },
];

/**
 * A figure opened from one view. B, the search's moves on the term (film.ts), and C, one term with
 * many functors (functors.ts), play as films; D, a family (fam.ts), is views the reader moves
 * through with its controls. Its steps are the views the checks visit: for a film, the view each
 * chapter ends on.
 */
export interface Figure {
  id: string;
  /** The view that offers it, and the words of its button there. */
  from: string;
  label: string;
  steps: readonly string[];
  /** The prefix of the figure's view ids. */
  prefix: string;
  /** A film plays from the view that offers it; a figure of views opens at its first step. */
  kind: "film" | "views";
  /** Whether it starts by itself when the window comes into view. */
  autoplay?: boolean;
}

export const FIGURES: readonly Figure[] = [
  {
    id: "rewrite",
    from: "model",
    label: "the search’s moves on this term",
    prefix: "rw/",
    kind: "film",
    autoplay: true,
    steps: [
      "rw/population",
      "rw/sites",
      "rw/replace",
      "rw/insert",
      "rw/delete",
      "rw/binding",
      "rw/shape",
      "rw/commute",
      "rw/admission",
      "rw/deepseek",
    ],
  },
  {
    id: "functors",
    from: "block",
    label: "its derivative, cost, plan and identities",
    prefix: "fn/",
    kind: "film",
    steps: [
      "fn/term",
      "fn/backward",
      "fn/cost",
      "fn/plan",
      "fn/remat",
      "fn/identity",
      "fn/halves",
      "fn/block",
      "fn/block/remat",
    ],
  },
  {
    id: "family",
    from: "model",
    label: "Monad with its controls moved",
    prefix: "fam/",
    kind: "views",
    steps: [
      "fam/model",
      "fam/model/V65536-L96-H16-K128-F4096",
      "fam/model/V4096-L2-H1-K16-F256",
      "fam/spectral",
    ],
  },
];

type Builder = (id: string) => View | undefined;

/** Builders by the prefix of the ids they draw; the empty prefix last. */
const sources: Array<[string, Builder]> = [
  ["lat/", lattice.view],
  ["", monad.view],
];

/** Make a model's views available; DeepSeek's are a separate chunk. */
export async function load(m: Model): Promise<void> {
  if (m.id === "dsv41" && !sources.some(([prefix]) => prefix === "ds/")) {
    const deepseek = await import("./deepseek.ts");
    if (!sources.some(([prefix]) => prefix === "ds/")) sources.unshift(["ds/", deepseek.view]);
  }
}

/** A figure's film: its module, which also builds the figure's views. */
export type FigureFilm = typeof import("./film.ts") | typeof import("./functors.ts");

/** A figure's module: a film's, or the views of a figure the reader moves through. */
export type FigureModule = FigureFilm | typeof import("./fam.ts");

const IMPORTS: Record<string, () => Promise<FigureModule>> = {
  rewrite: () => import("./film.ts"),
  functors: () => import("./functors.ts"),
  family: () => import("./fam.ts"),
};

/**
 * Make a figure available: its views, and its film if it has one, a separate chunk loaded when
 * the figure is first played or opened.
 */
export async function loadFigure(f: Figure): Promise<FigureModule> {
  const mod = await IMPORTS[f.id]!();
  if (!sources.some(([prefix]) => prefix === f.prefix))
    sources.unshift([f.prefix, mod.figureView]);
  return mod;
}

export const loadAll = () =>
  Promise.all([...MODELS.map(load), ...FIGURES.map(loadFigure)]).then(() => undefined);

function build(id: string): View {
  const v = sources.find(([prefix]) => id.startsWith(prefix))?.[1](id);
  if (!v) throw new Error(`no view ${id}`);
  return v;
}

/**
 * A view, built until the boxes that hold text fit it at the scale the view is drawn. A before
 * view is built at the scale of the view it becomes, so what a rewrite leaves alone is drawn
 * identically in both.
 */
export function view(id: string): View {
  const after = id.endsWith("/before") ? id.slice(0, -"/before".length) : id;
  design.ppu = 20;
  let v = build(after);
  for (let i = 0; i < 3; i++) {
    design.ppu = fit(framing(v), v.yaw, v.pitch, REF.width, REF.height, REF.pad, v.maxScale).scale;
    v = build(after);
  }
  return after === id ? v : build(id);
}
