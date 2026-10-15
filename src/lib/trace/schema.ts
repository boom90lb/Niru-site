/**
 * `niru.search-trace/1`: the record of a window of evolutionary search, and
 * of what follows it at full scale, that the site's figures replay.
 *
 * The site owns this format. It follows the shape of AMLO's proposal event
 * log (`proposal_events.jsonl`): the same event types, and AMLO's field names
 * wherever AMLO records the same quantity, so that a future exporter is a
 * projection rather than a translation. Fields that AMLO does not record today
 * are marked "enrichment"; a figure must still render when they are absent.
 *
 * Today every trace on the site comes from `simulate.ts`, which applies
 * AMLO's rules to a synthetic loss model. `source.kind` says so, and the
 * figures print it.
 */

export const TRACE_SCHEMA = "niru.search-trace/1";

export type TraceSource =
  | {
      kind: "simulated";
      generator: string;
      seed: number;
      /** Plain statement of what is simulated, shown with the figure. */
      note: string;
    }
  | {
      kind: "amlo";
      /** AMLO release and commit that produced the run. */
      release: string;
      commit: string;
      search_id: string;
      exported_at: string;
      note: string;
    };

export interface TierSpec {
  tier: 1 | 2 | 3;
  /** C: the compute target in floating-point operations. */
  target_compute_flops: number;
  /** B and L. */
  global_batch: number;
  sequence_length: number;
  step_ceiling: number;
  /** Evaluation examples (sequences) per candidate. */
  n_examples: number;
  /** Confidence that promotion out of this tier requires, as 1 − p. */
  required_confidence: number;
  /**
   * Enrichment: what this tier changes from the one before it, so that each
   * tier asks its own question. Tier 1 is the proxy; Tier 2 trains again
   * from a new seed, on new examples and with more compute (does the lead
   * hold when all three change?); Tier 3 trains a wider copy of the
   * candidate and of the current best with the same u-μP settings (does the
   * advantage survive a change of width?).
   */
  change?: { new_seed: boolean; new_examples: boolean; width_multiplier: number };
}

/**
 * Enrichment: compute reserved for each purpose before the search starts.
 * Accounts are separate; spending never moves between them.
 */
export interface BudgetAccounts {
  search: number;
  replication: number;
  transfer: number;
  audit: number;
  confirmation: number;
}

/**
 * Enrichment: the full-scale configuration that follows the search. Both
 * designs grow to one deployment size and train under one compute target.
 */
export interface TargetSpec {
  target_compute_flops: number;
  global_batch: number;
  sequence_length: number;
  /** Parameters at full size; the width is set to reach it. */
  target_params: number;
  depth_multiplier: number;
  /** Size of the sealed evaluation set, opened only after the choice is frozen. */
  n_examples: number;
  required_confidence: number;
}

export interface SearchPolicy {
  population_size: number;
  elite_size: number;
  tournament_size: number;
  crossover_rate: number;
  mutation_rate: number;
  surrogate_screening_rate: number;
  promotion_mode: "enforced" | "shadow";
  fdr_q: number;
  bootstrap_resamples: number;
  confidence_level: number;
  min_examples_for_bootstrap: number;
  max_relative_compute_slack: number;
  tiers: TierSpec[];
  /**
   * Enrichment: screened-out candidates drawn at random for the Auditor to
   * train anyway, so the screen's misses are measured.
   */
  screen_audit_draws?: number;
  /** Enrichment: the largest |realized − C| / C a compute receipt may show. */
  max_relative_compute_error?: number;
  budget_accounts?: BudgetAccounts;
  target?: TargetSpec;
}

/** A strict language-model spine: input → embedding → body × n → head → output. */
export interface Architecture {
  arch_hash: string;
  /** Component name of each body block, bottom to top. */
  body: string[];
  width: number;
  num_heads: number;
  /** Parameter count. */
  params: number;
  /** M: training FLOPs per token position. */
  flops_per_token: number;
  /** λ₂ of the normalized Laplacian of the contracted graph. */
  fiedler_unweighted: number;
}

export type Operator = "add_node" | "remove_node" | "modify_param" | "crossover" | "clone";

export interface ProposedPayload {
  source: "initial" | "evolve" | "deterministic_variant";
  actions: Array<{ operator: Operator; params?: Record<string, number | string> }>;
}

export interface ValidatedPayload {
  gate: "dedup_intra" | "tier0" | "compliance" | "dedup_historical";
  status: "passed" | "failed";
  reason_code: string;
  /** Enrichment: the Tier-0 gate message, e.g. `budget_below_one_token`. */
  detail?: string;
}

export interface ScreenedPayload {
  tier: 1;
  selected: boolean;
  reason_code: string;
  /** Enrichment: the surrogate's loss quantiles and its optimistic bound. */
  prediction?: { q16: number; q50: number; q84: number; bound: number; rank: number };
  /**
   * Enrichment: for a screened-out candidate, the lottery that may still send
   * it to the Auditor, who trains it at Tier 1 to audit the screen. The
   * probability is logged before the draw.
   */
  audit?: { drawn: boolean; probability: number };
}

/** Enrichment: what a run actually spent, recorded after it ran. */
export interface ComputeReceipt {
  realized_compute_flops: number;
  /** (realized − C) / C: the run's relative compute error against its target. */
  relative_compute_error: number;
}

export interface ComputeBudgetPlan {
  target_compute_flops: number;
  global_batch: number;
  sequence_length: number;
  steps: number;
  selected_flops_per_token: number;
  planned_compute_flops: number;
  relative_compute_slack: number;
}

export interface EvaluatedPayload {
  tier: 1 | 2 | 3;
  sound: boolean;
  /** Mean loss over the evaluation examples; null when the evaluation failed. */
  loss: number | null;
  fitness: number | null;
  compute_budget_plan: ComputeBudgetPlan;
  n_examples: number;
  /** Set when the evaluation failed, e.g. FAIRNESS at `tier1_budget_admission`. */
  failure_code?: string;
  failure_stage?: string;
  /** Enrichment: the admission issue, e.g. `compute_slack_exceeded`. */
  failure_reason?: string;
  /** Enrichment: the width the candidate trained at, as a multiple of its own (Tier 3). */
  width_multiplier?: number;
  /** Enrichment: the budget account this evaluation spent. */
  account?: keyof BudgetAccounts;
  /** Enrichment: the compute the run actually used. */
  receipt?: ComputeReceipt;
}

/**
 * Enrichment: the Auditor's sealed record of a screen audit. The candidate
 * the screen rejected trains at Tier 1 from the audit account and is
 * compared with the current best as the Tier-1 gate would compare it. The
 * result stays with the Auditor: the search never sees it, and the
 * screen's rejection stands. It counts toward the screen's measured miss
 * rate, which reaches the allocation policy only through a declared release.
 */
export interface AuditRecordedPayload extends Partial<TailEvidence> {
  tier: 1;
  probability: number;
  observed_diff: number;
  ci_lower?: number;
  ci_upper?: number;
  bootstrap_quantiles?: number[];
  /** Whether the Tier-1 gate would have passed it: a miss by the screen. */
  screen_missed: boolean;
  audience: "auditor";
  released: false;
}

/**
 * The bootstrap tail certificate, in AMLO's field names (d09a43d36,
 * `promotion_decision_payload`). `p_value` beside it is the descriptive
 * fraction `tail_count / n_resamples`; the exact binomial bounds decide. A
 * replay recomputes the bounds from the counts.
 */
export interface TailEvidence {
  n_resamples: number;
  tail_initial_resamples: number;
  tail_count: number;
  tail_lower_bound: number;
  tail_upper_bound: number;
  /** Total Monte Carlo error over both bounds at every scheduled look. */
  monte_carlo_error: number;
  /** The look the schedule stopped at, 1 to 4. */
  tail_stage: number;
  tail_schedule_version: string;
}

export interface PromotionDecisionPayload extends Partial<TailEvidence> {
  from_tier: 1 | 2 | 3;
  /** `best`: the Tier-3 gate, which decides whether the candidate becomes the current best. */
  to_tier: 2 | 3 | "best";
  promote: boolean;
  reason: string;
  observed_diff: number;
  /** Absent when the gate stops before the bootstrap (LOSS_NOT_IMPROVED). */
  ci_lower?: number;
  ci_upper?: number;
  p_value?: number;
  /** 1 − the certified upper bound on the tail, as AMLO records it. */
  confidence?: number;
  required_confidence: number;
  bootstrap_method?: "bca_paired";
  fdr_verdict?: "accepted" | "rejected";
  fdr_q: number;
  /** Enrichment: quantiles of the bootstrap distribution of the mean improvement. */
  bootstrap_quantiles?: number[];
}

export interface ResolvedPayload {
  terminal_state: "promoted" | "rejected" | "duplicate" | "invalid";
  reason_code: string;
}

export interface IncumbentPayload {
  previous: string;
  /** The rule that replaced it: the candidate passed the Tier-3 gate. */
  reason: "tier3_gate";
  replaced: true;
  tier_losses: Record<"1" | "2" | "3", number>;
}

/**
 * Enrichment: the search's choice and the analysis are frozen before any
 * sealed evidence opens: what is compared, the estimand, the estimator and
 * the sealed set's commitment.
 */
export interface SelectionFrozenPayload {
  champion: string;
  reference: string;
  policy: string;
  state_hash: string;
  analysis: {
    estimand: string;
    estimator: "bca_paired";
    confidence_level: number;
    required_confidence: number;
    sealed_set: string;
    n_examples: number;
  };
}

export type ScaleAxis = "width" | "depth" | "tokens";

/**
 * Enrichment: the one planned step from the proxy to the full-scale
 * configuration. Width, depth and tokens change together, so the full-scale
 * run tests the step as a whole and not each axis apart. Before that run,
 * the target configuration has no measurement.
 */
export interface TargetPlannedPayload {
  axes: Array<{ axis: ScaleAxis; from: number; to: number; rule: string }>;
  /** Configurations of the chosen design measured during the search. */
  tested: Array<{ tier: 1 | 2 | 3; width: number; depth: number; params: number }>;
  target: { width: number; depth: number; params: number };
  /**
   * The reference at full size. Each design takes the width that brings it
   * nearest the target size, so the two sizes differ; compute is matched,
   * size is not.
   */
  reference: { width: number; depth: number; params: number };
  evidence: "untested";
}

/** Enrichment: one full-scale training run. */
export interface TargetEvaluatedPayload {
  role: "champion" | "reference";
  /** The architecture this run scales up. */
  source_arch_hash: string;
  compute_budget_plan: ComputeBudgetPlan;
  receipt: ComputeReceipt;
  loss: number;
  n_examples: number;
  account: "confirmation";
  /** Transfer diagnostics recorded during training, as multiples of their target value. */
  diagnostics: { band: [number, number]; coordinate: number[]; spectral: number[] };
}

/**
 * Enrichment: the Auditor's receipt, before the seal opens, for the outcome's
 * validation path. The Auditor keeps custody of the sealed set on every
 * path; what it checks depends on the outcome: a held-out metric fixed with
 * the analysis ("metric"), a trusted formal check ("formal"), or an
 * independent judge for a broad measure of capability ("judge").
 */
export interface OutcomeValidationPayload {
  status: "valid" | "unavailable";
  path: "metric" | "formal" | "judge";
  checks: string[];
}

/** Enrichment: the paired comparison at full scale, on the sealed set. */
export interface ConfirmationPayload extends TailEvidence {
  observed_diff: number;
  ci_lower: number;
  ci_upper: number;
  /** Descriptive: `tail_count / n_resamples`. */
  p_value: number;
  required_confidence: number;
  /**
   * The gate's reasons, as AMLO joins them: `LOSS_IMPROVED`, or any of
   * `CI_CROSSES_ZERO`, `CONFIDENCE_BELOW_THRESHOLD` (the tail is certified
   * above 1 − required_confidence) and `BOOTSTRAP_TAIL_UNRESOLVED` (the
   * schedule ended with the threshold between the bounds).
   */
  reason: string;
  bootstrap_method: "bca_paired";
  bootstrap_quantiles: number[];
  n_examples: number;
  status: "supported" | "inconclusive" | "refuted";
  /**
   * Per-example loss differences (reference − champion) on the sealed set,
   * so the estimator can be rerun. Only the full trace carries them.
   */
  diffs?: number[];
}

/**
 * Enrichment: the claim bundle. Its status is the Auditor's. Its scope is
 * the one configuration the comparison ran at; its interval covers the
 * sealed examples for this trained pair, not other training seeds or other
 * sizes.
 */
export interface ClaimPayload {
  status: ConfirmationPayload["status"];
  estimand: string;
  baseline: string;
  scope: {
    width: number;
    depth: number;
    params: number;
    tokens: number;
    compute_flops: number;
    sequence_length: number;
  };
  uncertainty: string;
  /** What the bundle carries, by part. */
  contents: string[];
  revoked_by: string[];
  /** For a status other than supported: what may follow. */
  next?: string[];
}

/**
 * Enrichment: a replay of the claim's arithmetic from the record. `verifier`
 * names what ran; the site's simulated traces name the site's own check.
 */
export interface VerifiedPayload {
  verifier: string;
  checks: Array<{ check: string; passed: boolean }>;
  passed: boolean;
}

interface Envelope<T extends string, P> {
  event_type: T;
  proposal_id: string;
  generation: number;
  candidate_arch_hash: string;
  payload: P;
}

export type TraceEvent =
  | (Envelope<"proposal.proposed", ProposedPayload> & { origin_parent_arch_hashes: string[] })
  | Envelope<"proposal.validated", ValidatedPayload>
  | Envelope<"proposal.screened", ScreenedPayload>
  | Envelope<"proposal.evaluated", EvaluatedPayload>
  | Envelope<"proposal.promotion_decision", PromotionDecisionPayload>
  | Envelope<"proposal.resolved", ResolvedPayload>
  /** Site event: AMLO logs incumbent changes but does not emit an event for them. */
  | Envelope<"incumbent.updated", IncumbentPayload>
  /** Site event: the Auditor's sealed screen audit; AMLO does not record it today. */
  | Envelope<"audit.recorded", AuditRecordedPayload>
  /** Site events for what follows the search; AMLO does not record these today. */
  | Envelope<"selection.frozen", SelectionFrozenPayload>
  | Envelope<"target.planned", TargetPlannedPayload>
  | Envelope<"target.evaluated", TargetEvaluatedPayload>
  | Envelope<"outcome.validated", OutcomeValidationPayload>
  | Envelope<"confirmation.decided", ConfirmationPayload>
  | Envelope<"claim.recorded", ClaimPayload>
  | Envelope<"claim.verified", VerifiedPayload>;

export interface Snapshot {
  generation: number;
  incumbent: string;
  incumbent_tier_losses: Record<"1" | "2" | "3", number>;
  /** Population members by arch hash. */
  population: string[];
  /** The members carried unchanged as elites, by arch hash. */
  elites: string[];
}

export interface SearchTrace {
  schema: typeof TRACE_SCHEMA;
  source: TraceSource;
  objective: { metric: "loss"; unit: "nats per token"; direction: "minimize" };
  policy: SearchPolicy;
  /**
   * Enrichment: the fixed reference, the current best when the window opens.
   * Effects are reported against it as well as against the moving current
   * best, so gains are never chained from one best to the next.
   */
  reference?: { arch_hash: string; tier_losses: Record<"1" | "2" | "3", number> };
  architectures: Record<string, Architecture>;
  /**
   * Search state at the start of each generation in the window. AMLO keeps
   * this in its checkpoints (population, incumbent) rather than its event log.
   */
  snapshots: Snapshot[];
  events: TraceEvent[];
}
