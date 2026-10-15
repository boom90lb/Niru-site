/**
 * A simulated window of AMLO's evolutionary search, and the full-scale run
 * that follows it, written as a `niru.search-trace/1` trace.
 *
 * Most rules follow AMLO's search as built for 0.0.14: tournament
 * breeding with crossover and the live mutation operators on strict LM
 * spines; admission gates (genotype dedup, Tier 0, u-μP compliance,
 * historical dedup) with Tier 0's budget screen; a surrogate screen that keeps
 * the better half by an optimistic bound; the matched-compute planner
 * steps = floor(C / (M·B·L)) with its 1% slack limit; and a promotion gate
 * with a paired BCa bootstrap and a confidence requirement.
 *
 * Where AMLO does not build them yet, the rules follow its design:
 * Benjamini–Hochberg runs over every candidate tested at a tier in a
 * generation, not only the survivors; Tier 2 trains again from a new seed,
 * on new examples and with more compute; Tier 3 trains the candidate and the
 * current best wider, with the same u-μP settings, and its gate (not a
 * critic) decides the current best; the screen sends one screened-out
 * candidate, drawn by lottery with a logged probability, to the Auditor, who
 * trains it and keeps the result sealed from the search; compute comes from
 * separate accounts reserved in advance, and every run records a receipt of
 * what it used; effects are kept against a fixed reference; and after the
 * search the choice and the analysis are frozen, the champion and the
 * reference train at full scale, the Auditor issues a receipt for the
 * outcome's validation path and opens a sealed set, and a replay of the
 * record checks the claim's arithmetic.
 *
 * The losses are not AMLO's. They come from a scaling-law loss model with
 * per-example and per-seed noise, chosen only so the mechanics have
 * something to act on. At full scale only part of a design's proxy-scale
 * advantage carries over, so the final comparison can fail.
 */

import {
  benjaminiHochberg,
  mean,
  pairedBca,
  TAIL_MONTE_CARLO_ERROR,
  TAIL_SCHEDULE_VERSION,
  type BcaResult,
} from "../math/stats.ts";
import { normalizedFiedlerValue } from "../math/spectral.ts";
import { keyFromSeed, label, Stream } from "../math/threefry.ts";
import {
  TRACE_SCHEMA,
  type Architecture,
  type BudgetAccounts,
  type ComputeBudgetPlan,
  type ComputeReceipt,
  type Operator,
  type SearchPolicy,
  type SearchTrace,
  type Snapshot,
  type TailEvidence,
  type TargetSpec,
  type TierSpec,
  type TraceEvent,
} from "./schema.ts";

/** A bootstrap's tail certificate, in AMLO's field names. */
function tailEvidence(boot: BcaResult): TailEvidence {
  return {
    n_resamples: boot.tail.resamples,
    tail_initial_resamples: boot.tail.initial,
    tail_count: boot.tail.count,
    tail_lower_bound: boot.tail.lower,
    tail_upper_bound: boot.tail.upper,
    monte_carlo_error: TAIL_MONTE_CARLO_ERROR,
    tail_stage: boot.tail.stage,
    tail_schedule_version: TAIL_SCHEDULE_VERSION,
  };
}

/**
 * The reasons a bootstrap-backed gate refuses, as AMLO's `should_promote`
 * collects them (d09a43d36): the interval must clear zero, and the exact upper
 * bound on the tail must be within 1 − the required confidence. An empty list
 * passes.
 */
export function gateReasons(boot: BcaResult, requiredConfidence: number): string[] {
  const threshold = 1 - requiredConfidence;
  const reasons: string[] = [];
  if (!(boot.lo > 0)) reasons.push("CI_CROSSES_ZERO");
  if (boot.tail.upper > threshold)
    reasons.push(
      boot.tail.lower > threshold ? "CONFIDENCE_BELOW_THRESHOLD" : "BOOTSTRAP_TAIL_UNRESOLVED",
    );
  return reasons;
}

const VOCAB = 4096;
const WIDTHS = [64, 128, 256, 512, 1024] as const;
const SEED_LAYERS = 4;
const MAX_LAYERS = SEED_LAYERS * 2;
const BLOCK = "gpt_block";
/** Tier 0 plans at the reference geometry B = 4·8, L = 2048. */
const TIER0_QUANTUM = 32 * 2048;

/**
 * Tier specs in the e14_demo pattern: one batch geometry (B = 8, L = 128) and
 * compute targets in the ratio 1 : 5 : 20, scaled to the small models here.
 * Each tier after the first asks a new question: Tier 2 a new seed and new
 * examples, Tier 3 twice the width with the same u-μP settings.
 */
export const TIERS: TierSpec[] = [1, 5, 20].map((ratio, i) => ({
  tier: (i + 1) as 1 | 2 | 3,
  target_compute_flops: 2e13 * ratio,
  global_batch: 8,
  sequence_length: 128,
  step_ceiling: 250000,
  n_examples: 32,
  required_confidence: i === 0 ? 0.95 : 0.999,
  change: {
    new_seed: i > 0,
    new_examples: i > 0,
    width_multiplier: i === 2 ? 2 : 1,
  },
}));

/**
 * The full-scale runs after the search. Both designs grow to one deployment
 * size: three times the depth, and the width that brings each to the same
 * parameter count. Both train under one compute target, so the comparison at
 * full scale is matched in the same way as every tier and only the designs
 * differ. The sequence length stays at the proxy's, so context length is not
 * one of the changes.
 */
export const TARGET: TargetSpec = {
  target_compute_flops: 1e21,
  global_batch: 1024,
  sequence_length: 128,
  target_params: 1.3e9,
  depth_multiplier: 3,
  n_examples: 256,
  required_confidence: 0.999,
};

/** The largest |realized − C| / C a compute receipt may show. */
const RECEIPT_TOLERANCE = 0.02;

/**
 * Compute reserved per account before the search starts, in FLOPs. The
 * confirmation account holds the two full-scale runs, with room for their
 * receipts to land above plan within the tolerance.
 */
export const ACCOUNTS: BudgetAccounts = {
  search: 8e14,
  replication: 1.6e15,
  transfer: 2.4e15,
  audit: 1e14,
  confirmation: 2 * TARGET.target_compute_flops * (1 + RECEIPT_TOLERANCE),
};

export const POLICY: SearchPolicy = {
  population_size: 8,
  elite_size: 2,
  tournament_size: 5,
  crossover_rate: 0.5,
  mutation_rate: 0.8,
  surrogate_screening_rate: 0.5,
  promotion_mode: "enforced",
  fdr_q: 0.05,
  bootstrap_resamples: 2000,
  confidence_level: 0.95,
  min_examples_for_bootstrap: 20,
  max_relative_compute_slack: 0.01,
  tiers: TIERS,
  screen_audit_draws: 1,
  max_relative_compute_error: RECEIPT_TOLERANCE,
  budget_accounts: ACCOUNTS,
  target: TARGET,
};

/** Loss-model constants. Illustrative, not fitted to AMLO runs. */
const LOSS = {
  floor: 2.05,
  paramCoef: 1.05,
  paramExp: 0.3,
  dataCoef: 1.4,
  dataExp: 0.29,
  /** Penalty per squared octave away from width = 64 · depth. */
  shape: 0.03,
  quirk: 0.012,
  difficulty: 0.45,
  /** Model-by-example noise at each tier. */
  noise: [0.05, 0.034, 0.024] as const,
  /**
   * Training-seed noise at each tier: a shift in a candidate's whole loss
   * that one seed gives and another does not. Pairing on one seed cannot see
   * it; a new seed can.
   */
  seed: [0.025, 0.018, 0.012] as const,
  /** Noise per example on the sealed set at full scale. */
  targetNoise: 0.02,
  /**
   * Training-seed noise at full scale. The paired interval over sealed
   * examples cannot see it: it describes this trained pair only.
   */
  targetSeed: 0.004,
  /**
   * How much of a design's proxy-scale character survives at full scale: a
   * tenth of its shape effect, and its quirk with correlation 0.6 to a new
   * one. A proxy winner can therefore lose at full scale.
   */
  carry: { shape: 0.1, quirk: 0.6 },
  /** A receipt's overhead over the plan: padding, restarts and the like. */
  overhead: 0.003,
};

export interface Genome {
  layers: number;
  width: number;
}

/** The search state that opens the window. */
export const START: { incumbent: Genome; neighbours: Genome[] } = {
  incumbent: { layers: 2, width: 512 },
  neighbours: [
    { layers: 2, width: 256 },
    { layers: 1, width: 512 },
    { layers: 3, width: 256 },
    { layers: 2, width: 1024 },
    { layers: 1, width: 256 },
    { layers: 4, width: 128 },
    { layers: 3, width: 128 },
  ],
};

const r5 = (x: number) => Math.round(x * 1e5) / 1e5;
const sig = (x: number, digits = 4) => Number(x.toPrecision(digits));

function heads(width: number): number {
  return Math.max(1, Math.min(16, width / 32));
}

function hashOf(g: Genome): string {
  const a = label(`spine|${BLOCK}x${g.layers}|w${g.width}|h${heads(g.width)}`);
  const b = label(`${a}|${g.layers}|${g.width}`);
  return (a.toString(16).padStart(8, "0") + b.toString(16).padStart(8, "0")).slice(0, 12);
}

function bodyParams(g: Genome): number {
  return 12 * g.layers * g.width * g.width;
}

function params(g: Genome): number {
  return bodyParams(g) + 2 * VOCAB * g.width + (2 * g.layers + 1) * 2 * g.width;
}

/** M at sequence length L: three times the forward FLOPs per token position. */
function flopsPerToken(g: Genome, sequenceLength: number): number {
  return 3 * (2 * bodyParams(g) + 2 * VOCAB * g.width + 2 * g.layers * g.width * sequenceLength);
}

/** The genome a tier trains: Tier 3 trains a wider copy. */
export function atTier(g: Genome, tier: TierSpec): Genome {
  const k = tier.change?.width_multiplier ?? 1;
  return k === 1 ? g : { ...g, width: g.width * k };
}

/** The genome trained at full scale: deeper, and as wide as the target size needs. */
export function atTarget(g: Genome, target: TargetSpec = TARGET): Genome {
  const layers = g.layers * target.depth_multiplier;
  // params = 12·L·w² + (2V + 4L + 2)·w; solve for w and round to 256.
  const a = 12 * layers;
  const b = 2 * VOCAB + 4 * layers + 2;
  const w = (-b + Math.sqrt(b * b + 4 * a * target.target_params)) / (2 * a);
  return { layers, width: Math.max(256, Math.round(w / 256) * 256) };
}

export function plan(
  g: Genome,
  tier: Pick<TierSpec, "target_compute_flops" | "global_batch" | "sequence_length">,
): ComputeBudgetPlan {
  const m = flopsPerToken(g, tier.sequence_length);
  const quantum = tier.global_batch * tier.sequence_length;
  const steps = Math.floor(tier.target_compute_flops / (m * quantum) + 1e-9);
  const planned = steps * quantum * m;
  return {
    target_compute_flops: tier.target_compute_flops,
    global_batch: tier.global_batch,
    sequence_length: tier.sequence_length,
    steps,
    selected_flops_per_token: sig(m, 6),
    planned_compute_flops: sig(planned, 6),
    relative_compute_slack: sig(
      (tier.target_compute_flops - planned) / tier.target_compute_flops,
      4,
    ),
  };
}

function architecture(g: Genome): Architecture {
  const n = g.layers + 4;
  const edges = Array.from({ length: n - 1 }, (_, i) => [i, i + 1] as const);
  return {
    arch_hash: hashOf(g),
    body: Array.from({ length: g.layers }, () => BLOCK),
    width: g.width,
    num_heads: heads(g.width),
    params: params(g),
    flops_per_token: sig(flopsPerToken(g, 128), 6),
    fiedler_unweighted: r5(normalizedFiedlerValue({ nodes: n, edges })),
  };
}

export class Simulation {
  readonly key: [number, number];
  readonly events: TraceEvent[] = [];
  readonly architectures: Record<string, Architecture> = {};
  readonly genomes = new Map<string, Genome>();
  /** Every arch hash that has entered the search. */
  readonly history = new Set<string>();
  /** Per-example losses by arch hash and tier. */
  readonly evidence = new Map<string, number[]>();
  readonly fitness = new Map<string, number>();
  readonly tierOf = new Map<string, number>();
  incumbent: string;
  incumbentLosses: Record<"1" | "2" | "3", number>;
  /** The fixed reference: the current best when the window opens. */
  readonly reference: string;
  readonly referenceLosses: Record<"1" | "2" | "3", number>;
  /** Compute spent per account. */
  readonly spent: BudgetAccounts = {
    search: 0,
    replication: 0,
    transfer: 0,
    audit: 0,
    confirmation: 0,
  };
  population: string[];
  generation: number;
  private attempt = 0;
  private readonly breed: Stream;

  readonly seed: number;

  constructor(
    seed: number,
    startGeneration: number,
    start: Genome = START.incumbent,
    neighbours: Genome[] = START.neighbours,
  ) {
    this.seed = seed;
    this.key = keyFromSeed(seed);
    this.breed = Stream.of(this.key, "breed");
    this.generation = startGeneration;

    this.incumbent = this.register(start);
    this.incumbentLosses = { "1": 0, "2": 0, "3": 0 };
    for (const tier of TIERS) {
      const losses = this.examples(this.incumbent, tier);
      this.incumbentLosses[String(tier.tier) as "1" | "2" | "3"] = r5(mean(losses));
    }
    this.fitness.set(this.incumbent, -this.incumbentLosses["3"]);
    this.tierOf.set(this.incumbent, 3);
    this.reference = this.incumbent;
    this.referenceLosses = { ...this.incumbentLosses };

    // The population carried into the window: the incumbent and its
    // neighbours, already evaluated at Tier 1.
    this.population = [this.incumbent];
    for (const g of neighbours) {
      const h = this.register(g);
      this.fitness.set(h, -r5(mean(this.examples(h, TIERS[0]!))));
      this.tierOf.set(h, 1);
      this.population.push(h);
    }
  }

  private register(g: Genome): string {
    const a = architecture(g);
    this.architectures[a.arch_hash] = a;
    this.genomes.set(a.arch_hash, g);
    this.history.add(a.arch_hash);
    return a.arch_hash;
  }

  private shape(g: Genome): number {
    const octaves = Math.log2((64 * g.layers) / g.width);
    return LOSS.shape * octaves * octaves;
  }

  private quirk(hash: string, salt = "quirk"): number {
    return Stream.of(this.key, `${salt}|${hash}`).normal() * LOSS.quirk;
  }

  /** The deterministic part of a genome's loss after a planned run. */
  lossOf(
    hash: string,
    g: Genome,
    spec: Pick<TierSpec, "target_compute_flops" | "global_batch" | "sequence_length">,
  ): number {
    const p = plan(g, spec);
    const tokens = (p.steps * spec.global_batch * spec.sequence_length) / 1e6;
    return this.scaling(g, tokens) + this.shape(g) + this.quirk(hash);
  }

  private scaling(g: Genome, tokensM: number): number {
    const n = bodyParams(g) / 1e6;
    return (
      LOSS.floor +
      LOSS.paramCoef * n ** -LOSS.paramExp +
      LOSS.dataCoef * Math.max(tokensM, 1e-3) ** -LOSS.dataExp
    );
  }

  /**
   * The loss of a design trained at full scale, before example noise. Size
   * and data follow the same scaling law. Of the design's proxy-scale
   * character, a tenth of its shape effect survives, and its quirk carries
   * with correlation 0.6. One training seed adds its own shift.
   */
  targetLoss(hash: string, proxy: Genome, g: Genome, role: string): number {
    const p = plan(g, TARGET);
    const tokens = (p.steps * TARGET.global_batch * TARGET.sequence_length) / 1e6;
    const rho = LOSS.carry.quirk;
    return (
      this.scaling(g, tokens) +
      LOSS.carry.shape * this.shape(proxy) +
      rho * this.quirk(hash) +
      Math.sqrt(1 - rho * rho) * this.quirk(hash, "quirk-target") +
      Stream.of(this.key, `target-seed|${role}`).normal() * LOSS.targetSeed
    );
  }

  /** What a run used: its plan plus a small overhead. */
  private receipt(p: ComputeBudgetPlan, id: string): ComputeReceipt {
    const over = Math.abs(Stream.of(this.key, `receipt|${id}`).normal()) * LOSS.overhead;
    const realized = p.planned_compute_flops * (1 + over);
    return {
      realized_compute_flops: sig(realized, 6),
      relative_compute_error: sig((realized - p.target_compute_flops) / p.target_compute_flops, 4),
    };
  }

  /** The deterministic part of an architecture's loss at a tier. */
  trueLoss(hash: string, tier: TierSpec): number {
    return this.lossOf(hash, atTier(this.genomes.get(hash)!, tier), tier);
  }

  /**
   * Per-example losses on the tier's shared, ordered evaluation set. Example
   * difficulty is shared by every candidate; only the interaction noise is
   * the candidate's own. Pairing cancels the shared part. Every candidate at
   * a tier trains from that tier's seed, and the seed's effect on one
   * architecture is its own: pairing cannot cancel it, but a new seed at the
   * next tier draws it again.
   */
  examples(hash: string, tier: TierSpec): number[] {
    const id = `${hash}|${tier.tier}`;
    const known = this.evidence.get(id);
    if (known) return known;
    const seed =
      Stream.of(this.key, `seed|${tier.tier}|${hash}`).normal() * LOSS.seed[tier.tier - 1]!;
    const base = this.trueLoss(hash, tier) + seed;
    const shared = Stream.of(this.key, `examples|${tier.tier}`);
    const own = Stream.of(this.key, `noise|${id}`);
    const sigma = LOSS.noise[tier.tier - 1]!;
    const losses = Array.from({ length: tier.n_examples }, () =>
      r5(base + shared.normal() * LOSS.difficulty + own.normal() * sigma),
    );
    this.evidence.set(id, losses);
    return losses;
  }

  private emit(event: TraceEvent): void {
    this.events.push(event);
  }

  private tournament(): string {
    const pool = this.population.slice();
    const picked: string[] = [];
    const k = Math.min(POLICY.tournament_size, pool.length);
    for (let i = 0; i < k; i++) picked.push(pool.splice(this.breed.int(pool.length), 1)[0]!);
    let best = picked[0]!;
    for (const h of picked) if (this.fitness.get(h)! > this.fitness.get(best)!) best = h;
    return best;
  }

  private mutate(g: Genome): {
    genome: Genome;
    operator: Operator;
    params?: Record<string, number>;
  } {
    const ops: Operator[] = ["add_node", "remove_node", "modify_param"];
    const op = this.breed.pick(ops);
    if (op === "add_node") {
      return {
        genome: { ...g, layers: Math.min(g.layers + 1, MAX_LAYERS) },
        operator: op,
        params: { position: this.breed.int(g.layers + 1) },
      };
    }
    if (op === "remove_node" && g.layers >= 2) {
      return {
        genome: { ...g, layers: g.layers - 1 },
        operator: op,
        params: { position: this.breed.int(g.layers) },
      };
    }
    const choices = WIDTHS.filter((w) => w !== g.width);
    const width = this.breed.pick(choices);
    return { genome: { ...g, width }, operator: "modify_param", params: { width } };
  }

  private proposalId(): string {
    return `sim:g${this.generation}:a${this.attempt++}`;
  }

  /** Breed offspring until the population is full, recording every attempt. */
  private propose(elites: string[]): string[] {
    const admitted: string[] = [];
    const current = new Set(elites);
    const target = POLICY.population_size - elites.length;
    let tries = 0;
    while (admitted.length < target && tries < POLICY.population_size * 20) {
      tries++;
      const id = this.proposalId();
      const p1 = this.tournament();
      const g1 = this.genomes.get(p1)!;
      let child: Genome = { ...g1 };
      const parents = [p1];
      const actions: Array<{ operator: Operator; params?: Record<string, number | string> }> = [];
      if (this.breed.uniform() < POLICY.crossover_rate && this.population.length >= 2) {
        const p2 = this.tournament();
        const g2 = this.genomes.get(p2)!;
        const s1 = 1 + this.breed.int(g1.layers);
        const s2 = this.breed.int(g2.layers);
        const layers = Math.min(Math.max(s1 + (g2.layers - s2), 1), MAX_LAYERS);
        child = { width: g1.width, layers };
        parents.push(p2);
        actions.push({ operator: "crossover", params: { cut_a: s1, cut_b: s2 } });
      } else {
        actions.push({ operator: "clone" });
      }
      if (this.breed.uniform() < POLICY.mutation_rate) {
        const m = this.mutate(child);
        child = m.genome;
        actions.push(
          m.params ? { operator: m.operator, params: m.params } : { operator: m.operator },
        );
      }

      const known = this.genomes.has(hashOf(child));
      const hash = known ? hashOf(child) : this.register(child);
      const envelope = { proposal_id: id, generation: this.generation, candidate_arch_hash: hash };
      this.emit({
        event_type: "proposal.proposed",
        ...envelope,
        origin_parent_arch_hashes: parents,
        payload: { source: "evolve", actions },
      });

      const reject = (
        gate: "dedup_intra" | "tier0" | "dedup_historical",
        reason: string,
        state: "duplicate" | "invalid",
        detail?: string,
      ) => {
        this.emit({
          event_type: "proposal.validated",
          ...envelope,
          payload: { gate, status: "failed", reason_code: reason, ...(detail ? { detail } : {}) },
        });
        this.emit({
          event_type: "proposal.resolved",
          ...envelope,
          payload: { terminal_state: state, reason_code: reason },
        });
      };
      // AMLO's admission order: genotype dedup, Tier 0, compliance, historical dedup.
      if (current.has(hash) || this.population.includes(hash)) {
        reject("dedup_intra", "duplicate_architecture", "duplicate");
        continue;
      }
      const tier0 = this.tier0(child);
      if (tier0) {
        reject("tier0", "tier0_rejected", "invalid", tier0);
        continue;
      }
      // Every spine block declares its u-μP rule, so compliance passes here.
      if (known) {
        reject("dedup_historical", "earlier_duplicate", "duplicate");
        continue;
      }
      this.emit({
        event_type: "proposal.validated",
        ...envelope,
        payload: { gate: "tier0", status: "passed", reason_code: "admitted" },
      });
      current.add(hash);
      admitted.push(hash);
      this.tierOf.set(hash, 0);
      this.proposalOf.set(hash, id);
    }
    return admitted;
  }

  readonly proposalOf = new Map<string, string>();
  readonly snapshots: Snapshot[] = [];

  /** Tier 0's budget screen at the smallest tier's compute. Null when it passes. */
  private tier0(g: Genome): string | null {
    const m = flopsPerToken(g, 128);
    const c = TIERS[0]!.target_compute_flops;
    if (c < m) return "budget_below_one_token";
    if (Math.floor(c / (m * TIER0_QUANTUM)) <= 0) return "no_token_budget";
    return null;
  }

  private envelope(hash: string) {
    return {
      proposal_id: this.proposalOf.get(hash) ?? `sim:carried:${hash}`,
      generation: this.generation,
      candidate_arch_hash: hash,
    };
  }

  /** The surrogate screen: who trains at Tier 1, and who goes to the Auditor. */
  private screen(candidates: string[]): {
    selected: string[];
    audited: string[];
    probability: number;
  } {
    const s = Stream.of(this.key, `surrogate|${this.generation}`);
    const tier = TIERS[0]!;
    const scored = candidates.map((h) => {
      const median = this.trueLoss(h, tier) + s.normal() * 0.05;
      const spread = Math.max(0.012, 0.035 + s.normal() * 0.012);
      return { h, median, spread, bound: median - 1.96 * spread };
    });
    const ranked = scored.slice().sort((a, b) => a.bound - b.bound);
    const keep = Math.max(1, Math.floor(candidates.length * POLICY.surrogate_screening_rate));
    const selected = new Set(ranked.slice(0, keep).map((x) => x.h));
    // The audit lottery: draws among the screened-out, each equally likely,
    // with the probability fixed before the draw.
    const out = ranked.slice(keep).map((x) => x.h);
    const draws = Math.min(POLICY.screen_audit_draws ?? 0, out.length);
    const probability = out.length > 0 ? r5(draws / out.length) : 0;
    const lottery = Stream.of(this.key, `audit|${this.generation}`);
    const pool = out.slice();
    const drawn = new Set<string>();
    for (let k = 0; k < draws; k++) drawn.add(pool.splice(lottery.int(pool.length), 1)[0]!);
    for (const x of scored) {
      const rank = ranked.indexOf(x) + 1;
      const chosen = selected.has(x.h);
      this.emit({
        event_type: "proposal.screened",
        ...this.envelope(x.h),
        payload: {
          tier: 1,
          selected: chosen,
          reason_code: chosen ? "screen_selected" : "screened_out",
          prediction: {
            q16: r5(x.median - x.spread),
            q50: r5(x.median),
            q84: r5(x.median + x.spread),
            bound: r5(x.bound),
            rank,
          },
          ...(chosen ? {} : { audit: { drawn: drawn.has(x.h), probability } }),
        },
      });
      if (!chosen && !drawn.has(x.h)) {
        this.fitness.set(x.h, -9999);
        this.tierOf.set(x.h, 1);
        this.emit({
          event_type: "proposal.resolved",
          ...this.envelope(x.h),
          payload: { terminal_state: "rejected", reason_code: "screened_out" },
        });
      }
    }
    return {
      selected: candidates.filter((h) => selected.has(h)),
      audited: candidates.filter((h) => drawn.has(h)),
      probability,
    };
  }

  /** The tier's budget admission for a plan: null when it may run. */
  private admission(p: ComputeBudgetPlan, tier: TierSpec): string | null {
    return p.steps <= 0
      ? "zero_complete_steps"
      : p.steps > tier.step_ceiling
        ? "step_ceiling_exceeded"
        : p.relative_compute_slack > POLICY.max_relative_compute_slack
          ? "compute_slack_exceeded"
          : null;
  }

  /**
   * Train each candidate at the tier. An audit run trains the same way from
   * the audit account, and its loss stays out of the search's fitness.
   */
  private evaluate(hashes: string[], tier: TierSpec, audit = false): string[] {
    const sound: string[] = [];
    for (const h of hashes) {
      const p = plan(atTier(this.genomes.get(h)!, tier), tier);
      const account: keyof BudgetAccounts = audit
        ? "audit"
        : tier.tier === 1
          ? "search"
          : tier.tier === 2
            ? "replication"
            : "transfer";
      const wide = tier.change?.width_multiplier ?? 1;
      const fails = this.admission(p, tier);
      if (fails) {
        if (!audit) this.fitness.set(h, -Infinity);
        this.tierOf.set(h, tier.tier);
        this.emit({
          event_type: "proposal.evaluated",
          ...this.envelope(h),
          payload: {
            tier: tier.tier,
            sound: false,
            loss: null,
            fitness: null,
            failure_code: "FAIRNESS",
            failure_stage: `tier${tier.tier}_budget_admission`,
            failure_reason: fails,
            compute_budget_plan: p,
            n_examples: 0,
            ...(wide !== 1 ? { width_multiplier: wide } : {}),
            account,
          },
        });
        this.emit({
          event_type: "proposal.resolved",
          ...this.envelope(h),
          payload: { terminal_state: "rejected", reason_code: `tier${tier.tier}_${fails}` },
        });
        continue;
      }
      const loss = r5(mean(this.examples(h, tier)));
      if (!audit) this.fitness.set(h, -loss);
      this.tierOf.set(h, tier.tier);
      const receipt = this.receipt(p, `${h}|${audit ? "audit" : tier.tier}`);
      this.spent[account] += receipt.realized_compute_flops;
      this.emit({
        event_type: "proposal.evaluated",
        ...this.envelope(h),
        payload: {
          tier: tier.tier,
          sound: true,
          loss,
          fitness: -loss,
          compute_budget_plan: p,
          n_examples: tier.n_examples,
          ...(wide !== 1 ? { width_multiplier: wide } : {}),
          account,
          receipt,
        },
      });
      sound.push(h);
    }
    return sound;
  }

  /**
   * The Auditor's screen audit. A candidate the screen rejected trains at
   * Tier 1 from the audit account and is compared with the current best as
   * the Tier-1 gate would compare it. The result is sealed: it never enters
   * the search's gates, fitness or breeding, and the screen's rejection
   * stands.
   */
  private audit(hashes: string[], probability: number): void {
    const tier = TIERS[0]!;
    const incumbent = this.examples(this.incumbent, tier);
    for (const h of hashes) this.fitness.set(h, -9999);
    // A run that fails budget admission records its failure and nothing else.
    for (const h of this.evaluate(hashes, tier, true)) {
      const losses = this.examples(h, tier);
      const loss = r5(mean(losses));
      const observed = r5(this.incumbentLosses["1"] - loss);
      const diffs = incumbent.map((x, i) => x - losses[i]!);
      const boot =
        observed > 0
          ? pairedBca(diffs, new Stream(this.key, label(`audit-bootstrap|${losses.join(",")}`)), {
              resamples: POLICY.bootstrap_resamples,
              confidence: POLICY.confidence_level,
              tailThreshold: 1 - tier.required_confidence,
            })
          : undefined;
      const missed = Boolean(boot && gateReasons(boot, tier.required_confidence).length === 0);
      this.emit({
        event_type: "audit.recorded",
        ...this.envelope(h),
        payload: {
          tier: 1,
          probability,
          observed_diff: observed,
          ...(boot
            ? {
                ci_lower: r5(boot.lo),
                ci_upper: r5(boot.hi),
                bootstrap_quantiles: boot.quantiles.map(r5),
                ...tailEvidence(boot),
              }
            : {}),
          screen_missed: missed,
          audience: "auditor",
          released: false,
        },
      });
      this.fitness.set(h, -9999);
      this.tierOf.set(h, 1);
      this.emit({
        event_type: "proposal.resolved",
        ...this.envelope(h),
        payload: { terminal_state: "rejected", reason_code: "screened_out" },
      });
    }
  }

  /**
   * The statistical gate out of `tier`, with Benjamini–Hochberg over every
   * candidate tested at the tier, on the certified upper bound of each tail.
   * A candidate whose loss did not improve stops before the bootstrap and
   * enters the family at 1. Out of
   * Tier 3 the gate decides the current best: of the candidates that pass,
   * the one with the largest improvement takes its place.
   */
  private promote(hashes: string[], tier: TierSpec): string[] {
    const next = tier.tier === 3 ? ("best" as const) : ((tier.tier + 1) as 2 | 3);
    const key = String(tier.tier) as "1" | "2" | "3";
    const incumbent = this.examples(this.incumbent, tier);
    const order = hashes.slice().sort((a, b) => this.fitness.get(b)! - this.fitness.get(a)!);
    type Decision = {
      h: string;
      observed: number;
      reason: string;
      passed: boolean;
      boot?: ReturnType<typeof pairedBca>;
    };
    const decisions: Decision[] = order.map((h) => {
      const losses = this.examples(h, tier);
      const observed = r5(this.incumbentLosses[key] - mean(losses));
      if (observed <= 0) return { h, observed, reason: "LOSS_NOT_IMPROVED", passed: false };
      const diffs = incumbent.map((x, i) => x - losses[i]!);
      const seed = label(`bootstrap|${losses.join(",")}|${incumbent.join(",")}`);
      const boot = pairedBca(diffs, new Stream(this.key, seed), {
        resamples: POLICY.bootstrap_resamples,
        confidence: POLICY.confidence_level,
        tailThreshold: 1 - tier.required_confidence,
      });
      const reasons = gateReasons(boot, tier.required_confidence);
      return reasons.length === 0
        ? { h, observed, reason: "LOSS_IMPROVED", passed: true, boot }
        : { h, observed, reason: reasons.join("|"), passed: false, boot };
    });

    // As in AMLO, Benjamini–Hochberg reads the certified upper bound on each
    // tail, not its Monte Carlo fraction.
    const accepted = benjaminiHochberg(
      decisions.map((d) => d.boot?.tail.upper ?? 1),
      POLICY.fdr_q,
    );
    const verdict = new Map(
      decisions.map((d, i) => [d.h, accepted[i] ? "accepted" : "rejected"] as const),
    );
    // Out of Tier 3 only one candidate can take the current best's place.
    const passing = decisions.filter((d) => d.passed && verdict.get(d.h) === "accepted");
    const crowned =
      next === "best" ? passing.slice().sort((a, b) => b.observed - a.observed)[0]?.h : undefined;

    const promoted: string[] = [];
    for (const d of decisions) {
      const fdr = verdict.get(d.h)!;
      const outranked = next === "best" && d.passed && fdr === "accepted" && d.h !== crowned;
      const promote = d.passed && fdr === "accepted" && !outranked;
      const payload = {
        from_tier: tier.tier,
        to_tier: next,
        promote,
        reason: outranked ? "OUTRANKED_AT_TIER" : d.reason,
        observed_diff: d.observed,
        required_confidence: tier.required_confidence,
        fdr_q: POLICY.fdr_q,
        ...(d.boot
          ? {
              ci_lower: r5(d.boot.lo),
              ci_upper: r5(d.boot.hi),
              p_value: d.boot.p,
              confidence: 1 - d.boot.tail.upper,
              bootstrap_method: "bca_paired" as const,
              bootstrap_quantiles: d.boot.quantiles.map(r5),
              ...tailEvidence(d.boot),
            }
          : {}),
        fdr_verdict: fdr,
      };
      this.emit({ event_type: "proposal.promotion_decision", ...this.envelope(d.h), payload });
      if (promote) promoted.push(d.h);
      else {
        this.emit({
          event_type: "proposal.resolved",
          ...this.envelope(d.h),
          payload: {
            terminal_state: "rejected",
            reason_code:
              d.passed && fdr === "rejected"
                ? `tier${tier.tier}_fdr_rejected`
                : `tier${tier.tier}_${(outranked ? "outranked" : d.reason).toLowerCase()}`,
          },
        });
      }
    }
    return promoted;
  }

  /** The candidate that passed the Tier-3 gate becomes the current best. */
  private crown(hashes: string[]): void {
    for (const h of hashes) {
      const losses = {
        "1": r5(mean(this.examples(h, TIERS[0]!))),
        "2": r5(mean(this.examples(h, TIERS[1]!))),
        "3": r5(mean(this.examples(h, TIERS[2]!))),
      };
      this.emit({
        event_type: "incumbent.updated",
        ...this.envelope(h),
        payload: {
          previous: this.incumbent,
          reason: "tier3_gate",
          replaced: true,
          tier_losses: losses,
        },
      });
      this.incumbent = h;
      this.incumbentLosses = losses;
    }
  }

  /** One generation: breed, admit, screen, then Tiers 1–3 with their gates. */
  step(): void {
    this.attempt = 0;
    const ranked = this.population
      .slice()
      .sort((a, b) => this.fitness.get(b)! - this.fitness.get(a)!);
    const elites = ranked.slice(0, POLICY.elite_size);
    this.snapshots.push({
      generation: this.generation,
      incumbent: this.incumbent,
      incumbent_tier_losses: { ...this.incumbentLosses },
      population: this.population.slice(),
      elites,
    });
    const offspring = this.propose(elites);
    const { selected, audited, probability } = this.screen(offspring);
    this.audit(audited, probability);
    const t1 = this.evaluate(selected, TIERS[0]!);
    const toT2 = this.promote(t1, TIERS[0]!);
    const t2 = this.evaluate(toT2, TIERS[1]!);
    const toT3 = this.promote(t2, TIERS[1]!);
    const t3 = this.evaluate(toT3, TIERS[2]!);
    this.crown(this.promote(t3, TIERS[2]!));
    this.population = [...elites, ...offspring];
    this.generation++;
  }

  /**
   * After the search: freeze the choice and the analysis, plan one step to
   * full scale, train the champion and the fixed reference there under one
   * compute target, and let the Auditor validate the outcome and open the
   * sealed set. The claim bundle records the result, whatever it is, and a
   * replay of the record checks its arithmetic.
   */
  finale(): void {
    const generation = this.generation;
    const champion = this.incumbent;
    const reference = this.reference;
    const at = (hash: string, id: string) => ({
      proposal_id: id,
      generation,
      candidate_arch_hash: hash,
    });
    const estimand =
      "mean loss per token on the sealed set, reference minus chosen design, at the target configuration";
    const sealedSet = label(`sealed-set|${this.seed}`).toString(16).padStart(8, "0");
    this.emit({
      event_type: "selection.frozen",
      ...at(champion, "target:selection"),
      payload: {
        champion,
        reference,
        policy: "BaselineAllocationPolicy@v0",
        state_hash: label(`state|${this.seed}|${generation}|${champion}`)
          .toString(16)
          .padStart(8, "0"),
        analysis: {
          estimand,
          estimator: "bca_paired",
          confidence_level: POLICY.confidence_level,
          required_confidence: TARGET.required_confidence,
          sealed_set: sealedSet,
          n_examples: TARGET.n_examples,
        },
      },
    });

    const proxy = this.genomes.get(champion)!;
    const target = atTarget(proxy);
    const tier3 = TIERS[2]!;
    const proxyTokens =
      plan(atTier(proxy, tier3), tier3).steps * tier3.global_batch * tier3.sequence_length;
    const targetPlan = plan(target, TARGET);
    const targetTokens = targetPlan.steps * TARGET.global_batch * TARGET.sequence_length;
    this.emit({
      event_type: "target.planned",
      ...at(champion, "target:plan"),
      payload: {
        axes: [
          { axis: "width", from: proxy.width, to: target.width, rule: "u-μP" },
          { axis: "depth", from: proxy.layers, to: target.layers, rule: "u-μP residual scaling" },
          { axis: "tokens", from: proxyTokens, to: targetTokens, rule: "schedule in tokens" },
        ],
        tested: TIERS.filter((tier) => tier.tier !== 2).map((tier) => {
          const g = atTier(proxy, tier);
          return { tier: tier.tier, width: g.width, depth: g.layers, params: params(g) };
        }),
        target: { width: target.width, depth: target.layers, params: params(target) },
        reference: (() => {
          const g = atTarget(this.genomes.get(reference)!);
          return { width: g.width, depth: g.layers, params: params(g) };
        })(),
        evidence: "untested",
      },
    });

    // Both runs see the same sealed examples in the same order.
    const sealed = Stream.of(this.key, "sealed");
    const difficulty = Array.from(
      { length: TARGET.n_examples },
      () => sealed.normal() * LOSS.difficulty,
    );
    const runs: Record<"champion" | "reference", number[]> = { champion: [], reference: [] };
    const receipts: ComputeReceipt[] = [];
    for (const [role, hash] of [
      ["reference", reference],
      ["champion", champion],
    ] as const) {
      const source = this.genomes.get(hash)!;
      const g = atTarget(source);
      const a = this.register(g);
      const p = plan(g, TARGET);
      const base = this.targetLoss(hash, source, g, role);
      const own = Stream.of(this.key, `sealed-noise|${role}`);
      const losses = difficulty.map((d) => r5(base + d + own.normal() * LOSS.targetNoise));
      runs[role] = losses;
      const receipt = this.receipt(p, `target|${role}`);
      receipts.push(receipt);
      this.spent.confirmation += receipt.realized_compute_flops;
      const drift = Stream.of(this.key, `diagnostics|${role}`);
      const series = () => Array.from({ length: 8 }, () => r5(Math.exp(drift.normal() * 0.06)));
      this.emit({
        event_type: "target.evaluated",
        ...at(a, `target:${role}`),
        payload: {
          role,
          source_arch_hash: hash,
          compute_budget_plan: p,
          receipt,
          loss: r5(mean(losses)),
          n_examples: TARGET.n_examples,
          account: "confirmation",
          diagnostics: { band: [0.8, 1.25], coordinate: series(), spectral: series() },
        },
      });
    }

    // The Auditor's receipt for the outcome's validation path. Only then does
    // the seal open.
    this.emit({
      event_type: "outcome.validated",
      ...at(champion, "target:validation"),
      payload: {
        status: "valid",
        // Held-out loss is a metric outcome, fixed with the analysis.
        path: "metric",
        checks: [
          "the sealed set stayed out of the search",
          "it stayed out of training",
          "the metric was fixed with the analysis",
          "no evaluator that shaped training scores this comparison",
        ],
      },
    });

    const diffs = runs.reference.map((x, i) => r5(x - runs.champion[i]!));
    const boot = this.sealedBootstrap(diffs);
    const reasons = gateReasons(boot, TARGET.required_confidence);
    const status = reasons.length === 0 ? "supported" : boot.hi < 0 ? "refuted" : "inconclusive";
    this.emit({
      event_type: "confirmation.decided",
      ...at(champion, "target:confirmation"),
      payload: {
        observed_diff: r5(mean(diffs)),
        ci_lower: r5(boot.lo),
        ci_upper: r5(boot.hi),
        p_value: boot.p,
        required_confidence: TARGET.required_confidence,
        reason: reasons.join("|") || "LOSS_IMPROVED",
        bootstrap_method: "bca_paired",
        bootstrap_quantiles: boot.quantiles.map(r5),
        ...tailEvidence(boot),
        n_examples: TARGET.n_examples,
        status,
        diffs,
      },
    });
    this.emit({
      event_type: "claim.recorded",
      ...at(champion, "target:claim"),
      payload: {
        status,
        estimand,
        baseline: reference,
        scope: {
          width: target.width,
          depth: target.layers,
          params: params(target),
          tokens: targetTokens,
          compute_flops: TARGET.target_compute_flops,
          sequence_length: TARGET.sequence_length,
        },
        uncertainty: "sealed examples, for this trained pair",
        contents: [
          "claim and estimand",
          "chosen design and baseline",
          "selection history and policy",
          "compute receipts",
          "planned scale step",
          "measurement protocol and sealed set",
          "uncertainty model",
          "revocation triggers",
        ],
        revoked_by: [
          "change to an operator contract",
          "change to the optimizer",
          "change to the data",
          "a failed target anchor",
        ],
        ...(status === "supported"
          ? {}
          : {
              next: [
                "extend with a new sealed set",
                "let the comparison expire",
                "audit the search",
              ],
            }),
      },
    });

    // A replay of the claim's arithmetic from the record. It reruns the
    // site's own estimator, so it checks this record, not AMLO.
    const frozenAt = this.events.findIndex((e) => e.event_type === "selection.frozen");
    const openedAt = this.events.findIndex((e) => e.event_type === "confirmation.decided");
    const tolerance = POLICY.max_relative_compute_error ?? 0.02;
    const rerun = this.sealedBootstrap(diffs);
    const checks = [
      {
        check: "selection and analysis froze before the seal opened",
        passed: frozenAt >= 0 && frozenAt < openedAt,
      },
      {
        check: "every compute receipt is within its tolerance",
        passed: this.events.every(
          (e) =>
            (e.event_type !== "proposal.evaluated" && e.event_type !== "target.evaluated") ||
            !e.payload.receipt ||
            Math.abs(e.payload.receipt.relative_compute_error) <= tolerance,
        ),
      },
      {
        check: "the estimator, rerun, gives the same interval",
        passed: r5(rerun.lo) === r5(boot.lo) && r5(rerun.hi) === r5(boot.hi),
      },
      {
        check: "the tail schedule, rerun, stops at the same count and outcome",
        passed:
          rerun.tail.count === boot.tail.count &&
          rerun.tail.resamples === boot.tail.resamples &&
          gateReasons(rerun, TARGET.required_confidence).join("|") === reasons.join("|"),
      },
    ];
    this.emit({
      event_type: "claim.verified",
      ...at(champion, "target:verified"),
      payload: {
        verifier: "niru-site trace replay",
        checks,
        passed: checks.every((c) => c.passed),
      },
    });
  }

  private sealedBootstrap(diffs: number[]) {
    return pairedBca(diffs, new Stream(this.key, label(`sealed-bootstrap|${diffs.join(",")}`)), {
      resamples: POLICY.bootstrap_resamples,
      confidence: POLICY.confidence_level,
      tailThreshold: 1 - TARGET.required_confidence,
    });
  }
}

export interface SimulationResult {
  trace: SearchTrace;
  replacements: number;
  /** Compute spent per account. */
  spent: BudgetAccounts;
}

export function simulate(seed: number, generations = 3, startGeneration = 7): SimulationResult {
  const sim = new Simulation(seed, startGeneration);
  for (let i = 0; i < generations; i++) sim.step();
  sim.finale();
  return result(sim);
}

/** The trace of a finished simulation. */
export function result(sim: Simulation): SimulationResult {
  const seed = sim.seed;
  const used = new Set<string>([
    ...sim.snapshots.flatMap((s) => [...s.population, s.incumbent]),
    ...sim.events.map((e) => e.candidate_arch_hash),
  ]);
  for (const e of sim.events) {
    if (e.event_type === "proposal.proposed")
      e.origin_parent_arch_hashes.forEach((h) => used.add(h));
  }
  const architectures = Object.fromEntries(
    Object.entries(sim.architectures).filter(([h]) => used.has(h)),
  );
  const replacements = sim.events.filter((e) => e.event_type === "incumbent.updated").length;
  return {
    replacements,
    spent: { ...sim.spent },
    trace: {
      schema: TRACE_SCHEMA,
      source: {
        kind: "simulated",
        generator: "niru-site src/lib/trace/simulate.ts",
        seed,
        note: "Simulated run. The search rules are derived from AMLO's code; the losses come from a synthetic loss model.",
      },
      objective: { metric: "loss", unit: "nats per token", direction: "minimize" },
      policy: POLICY,
      reference: { arch_hash: sim.reference, tier_losses: sim.referenceLosses },
      architectures,
      snapshots: sim.snapshots,
      events: sim.events,
    },
  };
}
