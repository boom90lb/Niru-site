import type { Prim, V3 } from "./engine.ts";
import type { Sketch } from "./kit.ts";

/**
 * What a view shows rests on one kind of evidence, and the note under the window says which:
 * AMLO's 0.0.14 design, the default; figure C's model of AMLO's built work rules and its design;
 * figure D's counts from Monad's definitions; or the site's measured spectral study.
 */
export type Evidence = "design" | "model" | "count" | "measured";

export const EVIDENCE: Record<Evidence, string> = {
  design: "Drawn from AMLO’s 0.0.14 design, which is written and under review and not yet built.",
  model:
    "AMLO’s built work rules and its 0.0.14 design, as this window models them. The plan and identities are not built.",
  count: "Counted by this window from Monad’s definitions.",
  measured: "Measured by this site’s spectral study, in a toy model trained with NumPy.",
};

/**
 * One drawing in the window: a definition, or the elements an index map
 * touches. A view does not know its parent. The window remembers the box a
 * view was opened from, so one definition can open from many places, as
 * every Reuse layer of the DeepSeek strip opens the same layer.
 */
export interface View {
  id: string;
  title: string;
  caption: string;
  sketch: Sketch;
  yaw: number;
  pitch: number;
  /** A cap on the design scale, in pixels per unit, for a view with little in it. */
  maxScale?: number;
  /**
   * For a step that rewrites a term: the view of the term before the step, drawn in the same
   * layout and keyed piece by piece, which the window carries into this one.
   */
  before?: string;
  /**
   * The stage of a figure told in steps: points the camera fits instead of the view's own
   * drawing, so every step of the figure is seen through one camera and nothing jumps between
   * them. A step is drawn to fit inside it.
   */
  stage?: readonly V3[];
  /** The model a figure's step draws, when it is not the model the figure was opened from. */
  model?: string;
  /** What the view rests on, for the note under the window: AMLO's 0.0.14 design if unset. */
  evidence?: Evidence;
  /** The line above the drawing, when it is not the model's. */
  kicker?: string;
}

export const evidenceOf = (v: View): Evidence => v.evidence ?? "design";

/** What a view's camera is fitted to: its stage, or else its drawing. */
export const framing = (v: View): Prim[] =>
  v.stage ? [{ kind: "stroke", pts: [...v.stage], w: 0, ink: "faint", layer: 0 }] : v.sketch.prims;

export const fmt = (n: number) => n.toLocaleString("en-US");
