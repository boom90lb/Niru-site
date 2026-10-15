/**
 * How a trace divides between the generations played in brief and the
 * featured window that every figure shows in full. The trace starts at
 * generation 1; the hero plays the earlier generations as a short prelude,
 * and the last two are the ones the figures, the walkthrough and the notes
 * describe.
 */

import type { SearchTrace } from "./schema.ts";

/** Generations shown in full at the end of the trace. */
export const FEATURED = 2;

const generations = (trace: SearchTrace): number[] => trace.snapshots.map((s) => s.generation);

/** The featured generations, oldest first. */
export function featured(trace: SearchTrace): number[] {
  return generations(trace).slice(-FEATURED);
}

/** The generations before the featured window, played in brief. */
export function brief(trace: SearchTrace): number[] {
  return generations(trace).slice(0, -FEATURED);
}

/** The first featured generation: the one the single-generation figures replay. */
export function featuredGeneration(trace: SearchTrace): number {
  const g = featured(trace)[0];
  if (g === undefined) throw new Error("trace has no generations");
  return g;
}
