/**
 * The projection of a search trace that the figures import. It keeps every
 * event and field a figure draws and drops the rest: the terminal
 * `proposal.resolved` events, the operator parameters and source of each
 * proposal, the reason code of each admission check, and the candidate hash
 * on admission, screen and promotion events, which the figures look up by
 * proposal instead, and the per-example differences of the sealed
 * comparison, which only a rerun of the estimator needs. The full trace stays
 * the record; this is only what a page has to carry. A unit test pins the
 * committed projection to the committed trace.
 */

import type { SearchTrace, TraceEvent } from "./schema.ts";

type Proposed = Extract<TraceEvent, { event_type: "proposal.proposed" }>;
type Validated = Extract<TraceEvent, { event_type: "proposal.validated" }>;
type ByProposal = Extract<
  TraceEvent,
  { event_type: "proposal.screened" | "proposal.promotion_decision" }
>;

/** An event as the figures carry it. */
export type FigureEvent =
  | Exclude<
      TraceEvent,
      {
        event_type:
          | "proposal.resolved"
          | "proposal.proposed"
          | "proposal.validated"
          | "proposal.screened"
          | "proposal.promotion_decision";
      }
    >
  | Omit<ByProposal, "candidate_arch_hash">
  | (Omit<Proposed, "payload"> & {
      payload: { actions: Array<Pick<Proposed["payload"]["actions"][number], "operator">> };
    })
  | (Omit<Validated, "payload" | "candidate_arch_hash"> & {
      payload: Omit<Validated["payload"], "reason_code">;
    });

/** A trace as the figures carry it. Every field a figure reads is present. */
export interface FigureTrace extends Omit<SearchTrace, "events"> {
  events: FigureEvent[];
}

export function forFigures(trace: SearchTrace): FigureTrace {
  const events: FigureEvent[] = [];
  for (const e of trace.events) {
    if (e.event_type === "proposal.resolved") continue;
    if (e.event_type === "proposal.proposed") {
      events.push({
        ...e,
        payload: { actions: e.payload.actions.map((a) => ({ operator: a.operator })) },
      });
      continue;
    }
    if (e.event_type === "proposal.validated") {
      const { candidate_arch_hash: _hash, ...rest } = e;
      const { reason_code: _code, ...payload } = e.payload;
      events.push({ ...rest, payload });
      continue;
    }
    if (e.event_type === "proposal.screened" || e.event_type === "proposal.promotion_decision") {
      const { candidate_arch_hash: _hash, ...rest } = e;
      events.push(rest);
      continue;
    }
    if (e.event_type === "confirmation.decided") {
      const { diffs: _diffs, ...payload } = e.payload;
      events.push({ ...e, payload });
      continue;
    }
    events.push(e);
  }
  return { ...trace, events };
}
