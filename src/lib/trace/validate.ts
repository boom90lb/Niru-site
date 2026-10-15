/**
 * Checks that a search trace obeys the rules it claims to follow. The
 * simulator's output must pass, and so must any future export from AMLO:
 * this is the gate a trace passes before a figure replays it.
 *
 * Each check names AMLO's rule. A failure returns a readable message; an
 * empty list means the trace is consistent.
 */

import {
  mean,
  pairedBca,
  tailBounds,
  TAIL_MONTE_CARLO_ERROR,
  TAIL_SCHEDULE_VERSIONS,
  TAIL_STAGE_MULTIPLIERS,
} from "../math/stats.ts";
import { keyFromSeed, label, Stream } from "../math/threefry.ts";
import { TRACE_SCHEMA, type SearchTrace, type TailEvidence, type TraceEvent } from "./schema.ts";

type Of<T extends TraceEvent["event_type"]> = Extract<TraceEvent, { event_type: T }>;

/** The checks the site's replay of a simulated claim runs, in order. */
export const SITE_REPLAY_CHECKS = [
  "selection and analysis froze before the seal opened",
  "every compute receipt is within its tolerance",
  "the estimator, rerun, gives the same interval",
  "the tail schedule, rerun, stops at the same count and outcome",
];

const finite = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 + 1e-9 * Math.abs(b);

export function validateTrace(trace: SearchTrace): string[] {
  const problems: string[] = [];
  const fail = (message: string) => problems.push(message);

  /**
   * Replays a bootstrap tail certificate from its counts, as AMLO's
   * `validate_bootstrap_tail` does, and returns the gate's reasons for it:
   * none when the interval clears zero and the tail's exact upper bound is
   * within 1 − the required confidence. Undefined when the evidence cannot
   * be replayed, which never passes a gate.
   */
  const tailReasons = (
    where: string,
    e: Partial<TailEvidence> & { p_value?: number; ci_lower?: number; ci_upper?: number },
    requiredConfidence: number,
  ): string[] | undefined => {
    const n = e.n_resamples;
    const count = e.tail_count;
    const stage = e.tail_stage;
    if (
      !Number.isInteger(n) ||
      !Number.isInteger(count) ||
      !Number.isInteger(stage) ||
      !finite(e.tail_lower_bound) ||
      !finite(e.tail_upper_bound) ||
      !finite(e.ci_lower) ||
      !finite(e.ci_upper)
    ) {
      fail(`${where}: the bootstrap carries no complete tail certificate`);
      return undefined;
    }
    let ok = true;
    const bad = (message: string) => {
      fail(`${where}: ${message}`);
      ok = false;
    };
    if (!(TAIL_SCHEDULE_VERSIONS as readonly string[]).includes(e.tail_schedule_version ?? ""))
      bad(`unknown tail schedule ${String(e.tail_schedule_version)}`);
    if (e.monte_carlo_error !== TAIL_MONTE_CARLO_ERROR)
      bad("the Monte Carlo error budget is not the schedule's");
    if (e.tail_initial_resamples !== trace.policy.bootstrap_resamples)
      bad("the first look is not the policy's resample count");
    if (stage! < 1 || stage! > TAIL_STAGE_MULTIPLIERS.length)
      bad("the tail stage is off the schedule");
    else if (n !== e.tail_initial_resamples! * TAIL_STAGE_MULTIPLIERS[stage! - 1]!)
      bad("the resample count does not match the tail stage");
    if (count! < 0 || count! > n!) bad("the tail count is outside [0, resamples]");
    if (!ok) return undefined;
    if (!finite(e.p_value) || Math.abs(e.p_value - count! / n!) > 1e-15)
      bad("the p-value is not the tail count over the resamples");
    if (!(e.ci_lower! <= e.ci_upper!)) bad("the interval's bounds are out of order");
    if (!(requiredConfidence > 0 && requiredConfidence < 1))
      bad("the required confidence is outside (0, 1)");
    const [lower, upper] = tailBounds(count!, n!);
    if (!near(e.tail_lower_bound!, lower) || !near(e.tail_upper_bound!, upper))
      bad("the tail bounds do not replay from the counts");
    if (!ok) return undefined;
    const threshold = 1 - requiredConfidence;
    const unresolved = upper > threshold && lower <= threshold;
    if (unresolved && stage! < TAIL_STAGE_MULTIPLIERS.length)
      bad("the schedule stopped while the tail was unresolved");
    const reasons: string[] = [];
    if (!(e.ci_lower! > 0)) reasons.push("CI_CROSSES_ZERO");
    if (upper > threshold)
      reasons.push(unresolved ? "BOOTSTRAP_TAIL_UNRESOLVED" : "CONFIDENCE_BELOW_THRESHOLD");
    return ok ? reasons : undefined;
  };

  if (trace.schema !== TRACE_SCHEMA)
    fail(`schema is ${String(trace.schema)}, not ${TRACE_SCHEMA}`);
  if (trace.snapshots.length === 0) fail("trace has no snapshots");
  const tiers = new Map(trace.policy.tiers.map((t) => [t.tier, t]));
  const compute = trace.policy.tiers.map((t) => t.target_compute_flops);
  if (compute.some((c, i) => i > 0 && c <= compute[i - 1]!)) {
    fail("tier compute targets must strictly increase");
  }

  const known = (h: string) => h in trace.architectures;
  for (const s of trace.snapshots) {
    for (const h of [s.incumbent, ...s.population, ...s.elites]) {
      if (!known(h)) fail(`snapshot g${s.generation} names unknown architecture ${h}`);
    }
  }

  const ofType = <T extends TraceEvent["event_type"]>(type: T) =>
    trace.events.filter((e): e is Of<T> => e.event_type === type);

  for (const e of trace.events) {
    if (!known(e.candidate_arch_hash)) fail(`${e.proposal_id}: unknown architecture`);
  }

  // Matched compute: steps = floor(C / (M·B·L)), slack within the limit.
  for (const e of ofType("proposal.evaluated")) {
    const p = e.payload.compute_budget_plan;
    const spec = tiers.get(e.payload.tier);
    const where = `${e.proposal_id} tier ${e.payload.tier}`;
    if (!spec) {
      fail(`${where}: no tier spec`);
      continue;
    }
    if (p.target_compute_flops !== spec.target_compute_flops)
      fail(`${where}: C differs from the tier spec`);
    if (p.global_batch !== spec.global_batch || p.sequence_length !== spec.sequence_length) {
      fail(`${where}: batch geometry differs from the tier spec`);
    }
    const quantum = p.global_batch * p.sequence_length * p.selected_flops_per_token;
    const expected = Math.floor(p.target_compute_flops / quantum + 1e-9);
    // M is recorded to six significant figures, so allow its rounding.
    if (Math.abs(expected - p.steps) > Math.max(1, p.steps * 1e-5))
      fail(`${where}: steps ${p.steps} ≠ floor(C/(M·B·L)) = ${expected}`);
    if (e.payload.sound) {
      if (p.steps < 1) fail(`${where}: a sound evaluation needs at least one step`);
      if (p.relative_compute_slack > trace.policy.max_relative_compute_slack + 1e-9) {
        fail(`${where}: slack ${p.relative_compute_slack} exceeds the limit`);
      }
      if (e.payload.loss === null) fail(`${where}: a sound evaluation needs a loss`);
      if (e.payload.n_examples < trace.policy.min_examples_for_bootstrap) {
        fail(`${where}: fewer examples than the bootstrap minimum`);
      }
    } else if (!e.payload.failure_code) {
      fail(`${where}: an unsound evaluation needs a failure code`);
    }
  }

  // Every run records what it used, within the receipt tolerance.
  const tolerance = trace.policy.max_relative_compute_error;
  for (const e of [...ofType("proposal.evaluated"), ...ofType("target.evaluated")]) {
    const r = e.payload.receipt;
    if (!r) continue;
    const c = e.payload.compute_budget_plan.target_compute_flops;
    if (Math.abs((r.realized_compute_flops - c) / c - r.relative_compute_error) > 1e-3)
      fail(`${e.proposal_id}: the receipt's compute error does not match its realized compute`);
    if (tolerance !== undefined && Math.abs(r.relative_compute_error) > tolerance)
      fail(`${e.proposal_id}: realized compute misses C by ${r.relative_compute_error}`);
  }

  // Each tier asks its own question; Tier 3 trains at its stated width.
  for (const e of ofType("proposal.evaluated")) {
    const spec = tiers.get(e.payload.tier);
    const wide = spec?.change?.width_multiplier ?? 1;
    if ((e.payload.width_multiplier ?? 1) !== wide) {
      fail(
        `${e.proposal_id} tier ${e.payload.tier}: trained at width ×${e.payload.width_multiplier ?? 1}, not ×${wide}`,
      );
    }
  }

  // The promotion gate, and Benjamini–Hochberg over every candidate tested at
  // a tier in a generation. A candidate that did not improve enters with p = 1.
  const decisions = ofType("proposal.promotion_decision");
  const families = new Map<string, Of<"proposal.promotion_decision">[]>();
  for (const e of decisions) {
    const p = e.payload;
    const where = `${e.proposal_id} ${p.from_tier}→${p.to_tier}`;
    const expectedNext = p.from_tier === 3 ? "best" : p.from_tier + 1;
    if (p.to_tier !== expectedNext)
      fail(`${where}: tier ${p.from_tier} promotes to ${expectedNext}`);
    // The thresholds are the policy's, not whatever the event says.
    const spec = tiers.get(p.from_tier);
    if (!spec || p.required_confidence !== spec.required_confidence)
      fail(`${where}: the required confidence is not the tier's`);
    if (p.fdr_q !== trace.policy.fdr_q) fail(`${where}: the FDR level is not the policy's`);
    const hasInterval =
      p.ci_lower !== undefined || p.ci_upper !== undefined || p.p_value !== undefined;
    if (p.reason === "LOSS_NOT_IMPROVED") {
      if (p.observed_diff > 0) fail(`${where}: LOSS_NOT_IMPROVED with a positive improvement`);
      if (hasInterval)
        fail(`${where}: the gate stops before the bootstrap when the loss does not improve`);
    } else if (!(p.observed_diff > 0)) {
      fail(`${where}: a loss that did not improve must stop at LOSS_NOT_IMPROVED`);
    }
    const reasons = hasInterval ? tailReasons(where, p, p.required_confidence) : undefined;
    if (hasInterval && !(p.ci_lower! <= p.observed_diff && p.observed_diff <= p.ci_upper!)) {
      fail(`${where}: the interval does not contain the observed improvement`);
    }
    if (reasons && (!finite(p.confidence) || !near(p.confidence, 1 - p.tail_upper_bound!)))
      fail(`${where}: the confidence is not 1 − the tail's upper bound`);
    const gatePassed = reasons !== undefined && reasons.length === 0 && p.observed_diff > 0;
    if (p.promote && !gatePassed) fail(`${where}: promoted without passing the gate`);
    if (p.promote && p.fdr_verdict !== "accepted")
      fail(`${where}: promoted without BH acceptance`);
    if ((p.reason === "LOSS_IMPROVED" || p.reason === "OUTRANKED_AT_TIER") && !gatePassed)
      fail(`${where}: ${p.reason} but the gate fails`);
    if (p.reason === "OUTRANKED_AT_TIER" && (p.to_tier !== "best" || p.promote))
      fail(`${where}: only the Tier-3 gate can outrank a passing candidate`);
    if (reasons && reasons.length > 0 && p.reason !== reasons.join("|"))
      fail(`${where}: the reason ${p.reason} is not the gate's ${reasons.join("|")}`);
    const key = `${e.generation}:${p.from_tier}`;
    const list = families.get(key) ?? [];
    list.push(e);
    families.set(key, list);
  }
  for (const [key, family] of families) {
    const m = family.length;
    // As in AMLO, BH reads the certified upper bound on each tail.
    const pOf = (e: Of<"proposal.promotion_decision">) => e.payload.tail_upper_bound ?? 1;
    const sorted = family.slice().sort((a, b) => pOf(a) - pOf(b));
    let cutoff = -1;
    sorted.forEach((e, rank) => {
      if (pOf(e) <= ((rank + 1) / m) * e.payload.fdr_q) cutoff = rank;
    });
    sorted.forEach((e, rank) => {
      const accepted = rank <= cutoff;
      if ((e.payload.fdr_verdict === "accepted") !== accepted) {
        fail(
          `${e.proposal_id} (${key}): BH verdict disagrees with the step-up rule over the tier`,
        );
      }
    });
    // Out of Tier 3, one candidate at most takes the current best's place,
    // and it is the passing candidate with the largest improvement.
    if (key.endsWith(":3")) {
      const crowned = family.filter((e) => e.payload.promote);
      if (crowned.length > 1) fail(`g${key}: more than one candidate replaced the current best`);
      const passing = family.filter(
        (e) => e.payload.promote || e.payload.reason === "OUTRANKED_AT_TIER",
      );
      const top = passing.reduce<Of<"proposal.promotion_decision"> | undefined>(
        (best, e) => (!best || e.payload.observed_diff > best.payload.observed_diff ? e : best),
        undefined,
      );
      if (top && !top.payload.promote)
        fail(`g${key}: the Tier-3 gate crowned the wrong candidate`);
    }
  }

  // The current best changes only through the Tier-3 gate.
  const crowned = new Set(
    decisions
      .filter((e) => e.payload.to_tier === "best" && e.payload.promote)
      .map((e) => `${e.generation}:${e.proposal_id}`),
  );
  const updates = ofType("incumbent.updated");
  for (const e of updates) {
    if (!crowned.has(`${e.generation}:${e.proposal_id}`))
      fail(`${e.proposal_id}: the current best changed without passing the Tier-3 gate`);
  }
  if (updates.length !== crowned.size) fail("a Tier-3 promotion did not update the current best");

  // The surrogate screen keeps max(1, floor(n · rate)), ranked by the bound,
  // and its audit lottery draws among the rest with a probability logged
  // before the draw.
  const audits = ofType("audit.recorded");
  const evaluatedAt1 = new Set(
    ofType("proposal.evaluated")
      .filter((e) => e.payload.tier === 1)
      .map((e) => e.proposal_id),
  );
  const screens = new Map<number, Of<"proposal.screened">[]>();
  for (const e of ofType("proposal.screened")) {
    const list = screens.get(e.generation) ?? [];
    list.push(e);
    screens.set(e.generation, list);
  }
  for (const [gen, list] of screens) {
    const keep = Math.max(1, Math.floor(list.length * trace.policy.surrogate_screening_rate));
    const kept = list.filter((e) => e.payload.selected).length;
    if (kept !== keep) fail(`g${gen}: the screen kept ${kept}, not ${keep}`);
    const out = list.filter((e) => !e.payload.selected);
    const draws = Math.min(trace.policy.screen_audit_draws ?? 0, out.length);
    const drawn = out.filter((e) => e.payload.audit?.drawn);
    if (drawn.length !== draws)
      fail(`g${gen}: the audit lottery drew ${drawn.length}, not ${draws}`);
    for (const e of out) {
      if (draws > 0 && Math.abs((e.payload.audit?.probability ?? -1) - draws / out.length) > 1e-4)
        fail(`${e.proposal_id}: audit probability is not ${draws}/${out.length}`);
      const trained = evaluatedAt1.has(e.proposal_id);
      if (trained !== Boolean(e.payload.audit?.drawn))
        fail(`${e.proposal_id}: a screened-out candidate trains only when the lottery draws it`);
    }
    // The Auditor's result stays sealed: an audited candidate gets an audit
    // record and never a gate decision, so the search cannot act on it.
    for (const e of drawn) {
      const record = audits.find((a) => a.proposal_id === e.proposal_id && a.generation === gen);
      const ran = ofType("proposal.evaluated").find(
        (x) => x.proposal_id === e.proposal_id && x.generation === gen && x.payload.tier === 1,
      );
      if (!record && ran?.payload.sound)
        fail(`${e.proposal_id}: drawn for audit but the Auditor recorded nothing`);
      if (!record) continue;
      {
        const a = record.payload;
        if (Math.abs(a.probability - (e.payload.audit?.probability ?? -1)) > 1e-9)
          fail(`${e.proposal_id}: the audit record's probability differs from the draw's`);
        if (a.released || a.audience !== "auditor")
          fail(`${e.proposal_id}: the audit result left the Auditor`);
        if (
          a.ci_lower !== undefined &&
          a.ci_upper !== undefined &&
          !(a.ci_lower <= a.observed_diff && a.observed_diff <= a.ci_upper)
        )
          fail(`${e.proposal_id}: the audit interval does not contain its improvement`);
        // The Auditor compares as the Tier-1 gate would.
        const t1 = tiers.get(1);
        const auditReasons =
          a.ci_lower !== undefined && t1
            ? tailReasons(`${e.proposal_id} audit`, a, t1.required_confidence)
            : undefined;
        const wouldPass = auditReasons !== undefined && auditReasons.length === 0;
        if (a.screen_missed !== wouldPass)
          fail(`${e.proposal_id}: the audit's screen miss disagrees with the Tier-1 gate`);
      }
      if (decisions.some((d) => d.proposal_id === e.proposal_id && d.generation === gen))
        fail(`${e.proposal_id}: the search acted on a sealed audit`);
    }
    const ranked = list.filter((e) => e.payload.prediction);
    for (const e of ranked) {
      const p = e.payload.prediction!;
      if (e.payload.selected !== p.rank <= keep)
        fail(`${e.proposal_id}: selection disagrees with rank`);
      if (Math.abs(p.bound - (p.q50 - 1.96 * ((p.q84 - p.q16) / 2))) > 2e-4) {
        fail(`${e.proposal_id}: bound ≠ median − 1.96 σ`);
      }
    }
  }

  // Each account pays only for its own purpose and stays within its reserve.
  const accounts = trace.policy.budget_accounts;
  if (accounts) {
    const drawnIds = new Set(
      ofType("proposal.screened")
        .filter((e) => e.payload.audit?.drawn)
        .map((e) => e.proposal_id),
    );
    const spent = { search: 0, replication: 0, transfer: 0, audit: 0, confirmation: 0 };
    for (const e of ofType("proposal.evaluated")) {
      const expected =
        e.payload.tier === 1
          ? drawnIds.has(e.proposal_id)
            ? "audit"
            : "search"
          : e.payload.tier === 2
            ? "replication"
            : "transfer";
      if (e.payload.account !== expected)
        fail(
          `${e.proposal_id} tier ${e.payload.tier}: charged to ${e.payload.account}, not ${expected}`,
        );
      if (e.payload.sound)
        spent[expected] +=
          e.payload.receipt?.realized_compute_flops ??
          e.payload.compute_budget_plan.planned_compute_flops;
    }
    for (const e of ofType("target.evaluated"))
      spent.confirmation += e.payload.receipt.realized_compute_flops;
    for (const k of Object.keys(spent) as Array<keyof typeof spent>) {
      if (spent[k] > accounts[k] * (1 + 1e-9))
        fail(
          `account ${k}: spent ${spent[k].toExponential(2)} of ${accounts[k].toExponential(2)}`,
        );
    }
  }

  // Only promoted candidates are evaluated at the next tier.
  const promotedTo = new Set(
    decisions.filter((e) => e.payload.promote).map((e) => `${e.proposal_id}:${e.payload.to_tier}`),
  );
  for (const e of ofType("proposal.evaluated")) {
    if (e.payload.tier > 1 && !promotedTo.has(`${e.proposal_id}:${e.payload.tier}`)) {
      fail(`${e.proposal_id}: evaluated at tier ${e.payload.tier} without a promotion`);
    }
  }

  // The fixed reference is the current best when the window opens.
  if (trace.reference) {
    const first = trace.snapshots[0];
    if (first && trace.reference.arch_hash !== first.incumbent)
      fail("the fixed reference is not the current best of the first generation");
    if (
      first &&
      JSON.stringify(trace.reference.tier_losses) !== JSON.stringify(first.incumbent_tier_losses)
    )
      fail("the fixed reference's losses differ from the first snapshot");
  }

  // After the search: the choice is frozen before anything sealed opens; the
  // champion and the reference train under one compute target; the claim
  // follows the sealed comparison.
  const order = (type: TraceEvent["event_type"]) =>
    trace.events.findIndex((e) => e.event_type === type);
  const frozen = ofType("selection.frozen")[0];
  const targets = ofType("target.evaluated");
  const confirmation = ofType("confirmation.decided")[0];
  const claim = ofType("claim.recorded")[0];
  const validation = ofType("outcome.validated")[0];
  const verified = ofType("claim.verified")[0];
  if (frozen || targets.length > 0 || confirmation || claim) {
    if (!frozen || !confirmation || !claim || !validation || !verified || targets.length !== 2) {
      fail(
        "the full-scale record needs a frozen selection, two target runs, an outcome-validation receipt, a confirmation, a claim and a verification",
      );
    } else {
      // The order the design requires: freeze, train, the Auditor's receipt,
      // the seal opens, the claim, then the replay.
      const steps: TraceEvent["event_type"][] = [
        "selection.frozen",
        "target.planned",
        "target.evaluated",
        "outcome.validated",
        "confirmation.decided",
        "claim.recorded",
        "claim.verified",
      ];
      steps.forEach((type, i) => {
        if (i > 0 && order(type) < order(steps[i - 1]!))
          fail(`${type} comes before ${steps[i - 1]}`);
      });
      if (validation.payload.status !== "valid")
        fail("the seal opened without a valid outcome-validation receipt");
      if (!["metric", "formal", "judge"].includes(validation.payload.path))
        fail("the outcome-validation receipt names no validation path");
      const lastSearch = trace.events.reduce(
        (k, e, i) =>
          e.event_type.startsWith("proposal.") || e.event_type === "incumbent.updated" ? i : k,
        -1,
      );
      if (order("selection.frozen") < lastSearch)
        fail("the selection froze before the search ended");
      if (order("confirmation.decided") < order("selection.frozen"))
        fail("the sealed set opened before the selection froze");
      const lastBest =
        updates.length > 0
          ? updates[updates.length - 1]!.candidate_arch_hash
          : trace.snapshots[0]?.incumbent;
      if (frozen.payload.champion !== lastBest)
        fail("the frozen champion is not the current best");
      if (trace.reference && frozen.payload.reference !== trace.reference.arch_hash)
        fail("the frozen reference is not the fixed reference");
      const target = trace.policy.target;
      const roles = new Map(targets.map((e) => [e.payload.role, e]));
      for (const [role, source] of [
        ["champion", frozen.payload.champion],
        ["reference", frozen.payload.reference],
      ] as const) {
        const e = roles.get(role);
        if (!e) {
          fail(`no full-scale run for the ${role}`);
          continue;
        }
        if (e.payload.source_arch_hash !== source)
          fail(`the ${role}'s full-scale run scales the wrong design`);
        const p = e.payload.compute_budget_plan;
        if (target && p.target_compute_flops !== target.target_compute_flops)
          fail(`the ${role}'s full-scale run has its own compute target`);
        const expected = Math.floor(
          p.target_compute_flops /
            (p.global_batch * p.sequence_length * p.selected_flops_per_token) +
            1e-9,
        );
        if (Math.abs(expected - p.steps) > Math.max(1, p.steps * 1e-5))
          fail(`the ${role}'s full-scale steps ≠ floor(C/(M·B·L))`);
      }
      const c = confirmation.payload;
      const a = frozen.payload.analysis;
      // The analysis is frozen as the policy declares it.
      if (
        a.estimator !== "bca_paired" ||
        a.confidence_level !== trace.policy.confidence_level ||
        (target !== undefined &&
          (a.required_confidence !== target.required_confidence ||
            a.n_examples !== target.n_examples))
      )
        fail("the frozen analysis is not the policy's");
      if (a.n_examples !== c.n_examples || a.required_confidence !== c.required_confidence)
        fail("the sealed comparison differs from the frozen analysis");
      if (!(c.ci_lower <= c.observed_diff && c.observed_diff <= c.ci_upper))
        fail("the sealed interval does not contain the observed improvement");
      const reasons = tailReasons("the sealed comparison", c, a.required_confidence);
      if (reasons) {
        const status =
          reasons.length === 0 ? "supported" : c.ci_upper < 0 ? "refuted" : "inconclusive";
        if (c.status !== status)
          fail(`the confirmation says ${c.status}, its evidence says ${status}`);
        if (c.reason !== (reasons.join("|") || "LOSS_IMPROVED"))
          fail(`the confirmation's reason ${c.reason} is not its evidence's`);
      }
      if (claim.payload.status !== c.status)
        fail("the claim's status differs from the confirmation");
      if (claim.payload.baseline !== frozen.payload.reference)
        fail("the claim's baseline is not the reference");
      if ((c.status === "supported") === Boolean(claim.payload.next))
        fail("only a claim that is not supported lists what may follow");
      // The claim covers the one configuration that ran, at its own size.
      const ran = roles.get("champion");
      const arch = ran ? trace.architectures[ran.candidate_arch_hash] : undefined;
      const scope = claim.payload.scope;
      if (
        !arch ||
        scope.width !== arch.width ||
        scope.depth !== arch.body.length ||
        scope.params !== arch.params
      )
        fail("the claim's scope is not the configuration that ran");
      if (ran) {
        const plan = ran.payload.compute_budget_plan;
        if (scope.compute_flops !== plan.target_compute_flops)
          fail("the claim's compute is not the run's");
        if (
          scope.sequence_length !== plan.sequence_length ||
          scope.tokens !== plan.steps * plan.global_batch * plan.sequence_length
        )
          fail("the claim's tokens are not the run's");
      }
      // The planned step names the size each design actually trained at.
      const planned = ofType("target.planned")[0]?.payload;
      for (const [role, size] of [
        ["champion", planned?.target],
        ["reference", planned?.reference],
      ] as const) {
        const run = roles.get(role);
        const trained = run ? trace.architectures[run.candidate_arch_hash] : undefined;
        if (
          !size ||
          !trained ||
          size.params !== trained.params ||
          size.width !== trained.width ||
          size.depth !== trained.body.length
        )
          fail(`the planned size for the ${role} is not the size it trained at`);
      }
      // The replay: its checks are the ones it declares, and every check holds.
      const checks = verified.payload.checks;
      if (checks.length === 0) fail("the verification ran no checks");
      if (
        trace.source.kind === "simulated" &&
        JSON.stringify(checks.map((x) => x.check)) !== JSON.stringify(SITE_REPLAY_CHECKS)
      )
        fail("the verification's checks are not the site's replay");
      if (!checks.every((x) => x.passed) || !verified.payload.passed)
        fail("the verification did not pass every check");
      // The sealed differences are the evidence; a full trace must carry them.
      if (!c.diffs) fail("the sealed differences are missing, so the estimator cannot rerun");
      else {
        if (c.diffs.length !== c.n_examples)
          fail("the sealed differences do not cover the sealed set");
        if (Math.abs(mean(c.diffs) - c.observed_diff) > 1e-4)
          fail("the sealed differences do not average to the observed improvement");
        if (trace.source.kind === "simulated") {
          const key = keyFromSeed(trace.source.seed);
          const rerun = pairedBca(
            c.diffs,
            new Stream(key, label(`sealed-bootstrap|${c.diffs.join(",")}`)),
            {
              resamples: trace.policy.bootstrap_resamples,
              confidence: trace.policy.confidence_level,
              tailThreshold: 1 - a.required_confidence,
            },
          );
          if (Math.abs(rerun.lo - c.ci_lower) > 1e-5 || Math.abs(rerun.hi - c.ci_upper) > 1e-5)
            fail("the estimator, rerun on the sealed differences, gives another interval");
          if (
            rerun.tail.count !== c.tail_count ||
            rerun.tail.resamples !== c.n_resamples ||
            rerun.p !== c.p_value
          )
            fail("the tail schedule, rerun on the sealed differences, gives another count");
        }
      }
    }
  }

  return problems;
}
