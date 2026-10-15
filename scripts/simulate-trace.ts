/**
 * Writes the search traces the figures replay, the projections of them that
 * the pages import, and the simulator's spread of outcomes:
 * - `simulated.json`: the home stage. Its first featured generation shows
 *   every mechanism the figures describe, and its full-scale comparison
 *   supports the claim.
 * - `simulated-inconclusive.json`: the System walkthrough. Its featured
 *   generations show every mechanism between them, and its full-scale
 *   comparison is inconclusive, so the walkthrough shows a run that ends
 *   without a claim.
 * - `outcomes.json`: how the full-scale comparison ends over the first
 *   seeds, so the pages can say how often each ending occurs.
 *
 * Each run starts at generation 1 and has three generations and the
 * full-scale comparison. Generation 1 plays in brief; 2 and 3 are the
 * featured window. The mechanisms are a Tier-0 rejection; a screen audit; a
 * close call, a positive improvement whose interval still crosses the sheet
 * at Tier 1; a Tier-1 pass that does not hold at Tier 2; and a new current
 * best through the Tier-3 gate. Seeds are not hand-picked: among those that
 * qualify, the script takes the one whose figure data is smallest. A seed
 * whose search already fails is not run to the end. Run with `npm run trace`.
 */

import { writeFile } from "node:fs/promises";
import { format, resolveConfig } from "prettier";
import { forFigures } from "../src/lib/trace/figures.ts";
import { result, simulate, Simulation } from "../src/lib/trace/simulate.ts";
import type { SearchTrace, TraceEvent } from "../src/lib/trace/schema.ts";

const GENERATIONS = 3;
const START = 1;
/** The featured generations: one played in brief, then two in full. */
const FEATURED = [2, 3];
const SEEDS = 6000;
/** Seeds run to the end to count how the full-scale comparison ends. */
const OUTCOME_SEEDS = 500;

type Status = "supported" | "inconclusive" | "refuted";

/** Which mechanisms the events of some generations show. */
function mechanisms(events: TraceEvent[]): Set<string> {
  const shown = new Set<string>();
  const decision = (id: string, tier: 1 | 2 | 3) =>
    events.find(
      (e): e is Extract<TraceEvent, { event_type: "proposal.promotion_decision" }> =>
        e.event_type === "proposal.promotion_decision" &&
        e.proposal_id === id &&
        e.payload.from_tier === tier,
    );
  for (const e of events) {
    if (
      e.event_type === "proposal.validated" &&
      e.payload.gate === "tier0" &&
      e.payload.status === "failed"
    )
      shown.add("tier0");
    if (e.event_type === "audit.recorded") shown.add("audit");
    if (e.event_type === "incumbent.updated") shown.add("best");
    if (e.event_type !== "proposal.promotion_decision" || e.payload.from_tier !== 1) continue;
    if (e.payload.reason.split("|").includes("CI_CROSSES_ZERO") && e.payload.observed_diff > 0)
      shown.add("close");
    if (e.payload.promote && decision(e.proposal_id, 2)?.payload.promote === false)
      shown.add("lapse");
  }
  return shown;
}
const EVERY = ["tier0", "audit", "best", "close", "lapse"];
const showsAll = (events: TraceEvent[]) => {
  const shown = mechanisms(events);
  return EVERY.every((m) => shown.has(m));
};
const statusOf = (trace: SearchTrace): Status | undefined => {
  const c = trace.events.find((e) => e.event_type === "confirmation.decided");
  return c?.event_type === "confirmation.decided" ? c.payload.status : undefined;
};

const outcomes: Record<Status, number> = { supported: 0, inconclusive: 0, refuted: 0 };
for (let seed = 1; seed <= OUTCOME_SEEDS; seed++) {
  const status = statusOf(simulate(seed, GENERATIONS, START).trace);
  if (status) outcomes[status]++;
}

const picks: Record<"home" | "system", { trace: SearchTrace; bytes: number } | null> = {
  home: null,
  system: null,
};
const consider = (key: "home" | "system", trace: SearchTrace) => {
  const bytes = JSON.stringify(forFigures(trace)).length;
  if (!picks[key] || bytes < picks[key].bytes) picks[key] = { trace, bytes };
};
for (let seed = 1; seed <= SEEDS; seed++) {
  const sim = new Simulation(seed, START);
  for (let g = START; g < FEATURED[0]!; g++) sim.step();
  for (const _ of FEATURED) sim.step();
  const of = (gens: number[]) => sim.events.filter((e) => gens.includes(e.generation));
  const home = showsAll(of([FEATURED[0]!]));
  const system = showsAll(of(FEATURED));
  if (!home && !system) continue;
  while (sim.generation < START + GENERATIONS) sim.step();
  sim.finale();
  const { trace } = result(sim);
  const status = statusOf(trace);
  if (home && status === "supported") consider("home", trace);
  if (system && status === "inconclusive") consider("system", trace);
}
if (!picks.home) throw new Error("no seed qualifies for the home stage");
if (!picks.system) throw new Error("no seed qualifies for the System walkthrough");

const write = async (name: string, data: unknown) => {
  const out = new URL(`../src/data/traces/${name}`, import.meta.url);
  const style = (await resolveConfig(out, { editorconfig: true })) ?? {};
  await writeFile(out, await format(JSON.stringify(data), { ...style, parser: "json" }));
  return out;
};
for (const [name, pick] of [
  ["simulated", picks.home],
  ["simulated-inconclusive", picks.system],
] as const) {
  const out = await write(`${name}.json`, pick.trace);
  await write(`${name}.figures.json`, forFigures(pick.trace));
  const seed = pick.trace.source.kind === "simulated" ? pick.trace.source.seed : "n/a";
  console.log(`${name}: seed ${seed}, ${pick.trace.events.length} events -> ${out.pathname}`);
}
await write("outcomes.json", {
  note: "How the simulated full-scale comparison ends over the first seeds of src/lib/trace/simulate.ts.",
  seeds: OUTCOME_SEEDS,
  ...outcomes,
});
console.log("outcomes", outcomes);
