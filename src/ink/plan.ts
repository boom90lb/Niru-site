/**
 * Figure C's plan and identity walls, computed as AMLO's 0.0.14 design describes them.
 *
 * A plan says how a program is executed, never what it computes. Under `ampere-bf16`, contractions
 * take bf16 operands and accumulate in f32; reductions and transcendentals (norms, softmax, SiLU's
 * logistic) and the residual stream are f32; masters are f32; an elementwise node computes in its
 * widest operand's storage. Casts are nodes, one per value and dtype, wherever a producer's storage
 * differs from a consumer's compute dtype. A contraction stores bf16, or f32 when every consumer
 * computes in f32. Where that rule refers back to itself (the scores, whose consumer is an
 * elementwise scale), the window takes f32, which needs no cast; the design does not settle it.
 *
 * The residual policy is a plan entry on the Repeat: StoreAll keeps every value the backward
 * reads; Remat keeps the stream and recomputes, in the backward, what the backward reads. The
 * backward is derived after the casts are applied, as the design orders the passes, so a
 * contraction keeps the bf16 operand it reads. Recomputation and casts are plan work;
 * AMLO's built work rules give a convert no price, so the plan counts the casts as none.
 * Mathematical work does not move. A plan edit moves execution identity, never structure.
 */
import {
  block,
  type Controls,
  extentsOf,
  NORMS,
  plus,
  Program,
  units,
  type Work,
  ZERO,
} from "./derive.ts";

export type Dtype = "f32" | "bf16";

/** The precision class of each node of the block, as the design names them. */
export type Class = "contraction" | "f32 class" | "elementwise" | "carry" | "layout";

/**
 * Nodes the table puts in the f32 class: reductions, transcendentals and norms. Softmax is its
 * row max and sum (reductions) and its exp and divide (transcendentals).
 */
const F32_RECIPES = new Set(["rsqrt", "exp", "log", "silu", "gelu_tanh", "div"]);

export interface Planned {
  /** Storage dtype of each value, and compute dtype of each node. */
  storage: Map<string, Dtype>;
  compute: Map<string, Dtype>;
  classOf: Map<string, Class>;
  /** Casts: a value converted for its consumers, once per dtype. */
  casts: Array<{ value: string; to: Dtype }>;
}

const CARRY = new Set(["x", "x1", "x2"]);

/** Resolve `ampere-bf16` over the block: each node's class, dtypes, and the casts it needs. */
export function ampere(g: Program = block()): Planned {
  const storage = new Map<string, Dtype>();
  const compute = new Map<string, Dtype>();
  const classOf = new Map<string, Class>();
  // Masters, inputs and the stream are f32; constants (the rope table, the mask, a literal) are
  // not real values the plan casts, and never widen an elementwise node.
  for (const v of g.values.values())
    if (v.kind === "parameter" || v.kind === "input") storage.set(v.name, "f32");
  for (const n of g.nodes) {
    const p = n.prim;
    let cls: Class;
    if (CARRY.has(n.result)) cls = "carry";
    else if (p.kind === "contract") cls = "contraction";
    else if (p.kind === "relabel") cls = "layout";
    else if (p.kind === "sum" || p.kind === "max") cls = "f32 class";
    else if (p.kind === "ew" && F32_RECIPES.has(p.recipe)) cls = "f32 class";
    else if (NORMS.has(n.def)) cls = "f32 class";
    else cls = "elementwise";
    classOf.set(n.result, cls);
  }
  // The consumers of a value, looking through layouts, which move no dtype.
  const consumers = (name: string): typeof g.nodes =>
    g.nodes
      .filter((n) => n.operands.includes(name))
      .flatMap((n) => (classOf.get(n.result) === "layout" ? consumers(n.result) : [n]));
  const widest = (ops: readonly string[]) =>
    ops.some((o) => storage.get(sourceOf(g, o)) === "f32") ? "f32" : "bf16";
  // A contraction stores f32 when every consumer computes in f32. Starting every contraction at
  // f32 and walking to a fixed point gives the plan with the fewest casts that keeps that rule.
  for (const n of g.nodes)
    if (classOf.get(n.result) === "contraction") storage.set(n.result, "f32");
  for (let changed = true; changed;) {
    changed = false;
    for (const n of g.nodes) {
      const cls = classOf.get(n.result)!;
      let c: Dtype;
      let st: Dtype;
      if (cls === "contraction") {
        c = "bf16";
        st = consumers(n.result).every((u) => compute.get(u.result) !== "bf16") ? "f32" : "bf16";
      } else if (cls === "elementwise" || cls === "layout") c = st = widest(n.operands);
      else c = st = "f32";
      if (compute.get(n.result) !== c || storage.get(n.result) !== st) changed = true;
      compute.set(n.result, c);
      storage.set(n.result, st);
    }
  }
  // Casts: once per (value, dtype) where a consumer computes in another dtype than it is stored.
  const casts: Planned["casts"] = [];
  const seen = new Set<string>();
  for (const n of g.nodes) {
    if (classOf.get(n.result) === "layout") continue;
    const want = compute.get(n.result)!;
    for (const o of n.operands) {
      const src = sourceOf(g, o);
      const have = storage.get(src);
      if (!have || have === want) continue;
      const key = `${src}→${want}`;
      if (seen.has(key)) continue;
      seen.add(key);
      casts.push({ value: src, to: want });
    }
  }
  return { storage, compute, classOf, casts };
}

/** Follow layouts back to the value they rearrange: a relabel moves no dtype. */
function sourceOf(g: Program, name: string): string {
  const n = g.nodes.find((m) => m.result === name);
  return n && n.prim.kind === "relabel" && n.operands.length === 1
    ? sourceOf(g, n.operands[0]!)
    : name;
}

const BYTES: Record<Dtype, number> = { f32: 4, bf16: 2 };

/** The nodes that consume a value, looking through layouts, which move no dtype. */
function consumersOf(g: Program, plan: Planned, name: string): Program["nodes"] {
  return g.nodes
    .filter((n) => n.operands.includes(name))
    .flatMap((n) =>
      plan.classOf.get(n.result) === "layout" ? consumersOf(g, plan, n.result) : [n],
    );
}

/**
 * The block with its casts applied, as the design orders the passes: precision first, then the
 * derivative. Each cast is a node, after the value it converts, and each consumer that computes in
 * the cast's dtype reads it, directly or through a layout. Derived from this program, a
 * contraction keeps the cast operand it reads, since a contraction's other operand is a residual,
 * and the casts of the weights, made inside the body, are stacked per iteration with the body's
 * other values.
 */
export function precise(g: Program = block(), plan: Planned = ampere(g)) {
  const p = new Program();
  const storage = new Map(plan.storage);
  const to = new Map(plan.casts.map((c) => [c.value, c.to]));
  const cast = new Map<string, string>();
  // A cast belongs to the definition of the first node that reads it.
  const defOf = (v: string) => consumersOf(g, plan, v)[0]?.def ?? "block";
  const emit = (v: string) => {
    const d = to.get(v);
    if (!d) return;
    const c = `${v}:${d}`;
    p.convert(defOf(v), c, v, d);
    storage.set(c, d);
    cast.set(v, c);
  };
  for (const v of g.values.values()) if (v.kind !== "node") p.value(v.name, v.axes, v.kind);
  for (const v of [...p.values.keys()]) emit(v);
  for (const n of g.nodes) {
    const layout = plan.classOf.get(n.result) === "layout";
    const operands = n.operands.map((o) => {
      const c = cast.get(o);
      if (!c) return o;
      const d = to.get(o)!;
      const reads = layout
        ? consumersOf(g, plan, n.result).every((u) => plan.compute.get(u.result) === d)
        : plan.compute.get(n.result) === d;
      return reads ? c : o;
    });
    p.value(n.result, g.values.get(n.result)!.axes, "node");
    p.nodes.push({ ...n, operands });
    emit(n.result);
  }
  return { program: p, storage };
}

/** The nodes that compute some values, with every node those depend on, in program order. */
function ancestors(p: Program, of: Iterable<string>): Program["nodes"] {
  const producer = new Map(p.nodes.map((n) => [n.result, n]));
  const seen = new Set<string>();
  const visit = (v: string) => {
    const n = producer.get(v);
    if (!n || seen.has(v)) return;
    seen.add(v);
    n.operands.forEach(visit);
  };
  for (const v of of) visit(v);
  return p.nodes.filter((n) => seen.has(n.result));
}

/**
 * Memory and work under a residual policy, per iteration of the block's Repeat. The backward is derived from the block with its casts (`precise`). StoreAll stores every
 * forward value that backward reads, at its storage dtype; Remat stores the stream into the
 * iteration and recomputes, as plan work, the values the backward reads and the nodes they depend
 * on. A derived program has no dead nodes, so the recompute leaves out what the backward never
 * reads: the block's last contraction, m, and its output, x2.
 */
export function residuals(c: Controls, policy: "StoreAll" | "Remat") {
  const e = extentsOf(c);
  const { program: p, storage } = precise();
  const steps = p.backward(["x2"], ["x"], e);
  const reads = steps.flatMap((s) => s.reads);
  // The forward values the backward reads, each counted once by the value a layout rearranges.
  const read = new Set(reads.map((v) => sourceOf(p, v)));
  const values = [...read].filter((v) => p.values.get(v)?.kind === "node" || v === "x");
  const elements = (v: string) => p.values.get(v)!.axes.reduce((n, a) => n * e[a]!, 1);
  const kept = policy === "StoreAll" ? values : ["x"];
  const bytes = kept.reduce((b, v) => b + elements(v) * BYTES[storage.get(v) ?? "f32"], 0);
  const recomputed = policy === "Remat" ? ancestors(p, reads) : [];
  return {
    kept,
    bytes,
    recompute: recomputed.reduce((w, n) => plus(w, p.forward(n, e)), ZERO),
    recomputed: recomputed.map((n) => n.result),
  };
}

/**
 * Mathematical work, derived from the block with its casts, and plan work, per iteration, under a
 * policy. A cast does no arithmetic, so applying precision moves no mathematical work.
 */
export function work(c: Controls, policy: "StoreAll" | "Remat") {
  const { program: p } = precise();
  const e = extentsOf(c);
  const steps = p.backward(["x2"], ["x"], e);
  const math: Work = steps.reduce((w, s) => plus(w, s.work), p.forwardWork(e));
  return { math: units(math), plan: units(residuals(c, policy).recompute) };
}

// --------------------------------------------------------------------------------- identities

/**
 * The identities the column shows, each with what it hashes: the four components, and execution,
 * which hashes them with the resolved plan. The design has more identities; the column draws these
 * five.
 */
export const COMPONENTS = [
  {
    id: "structure",
    hashes: "the canonical form: ops, wiring, bindings, bind kinds, Repeat base and overrides",
  },
  { id: "configuration", hashes: "the root controls" },
  { id: "numerics", hashes: "initializer declarations per canonical leaf" },
  { id: "randomness", hashes: "pairs of canonical leaf or site position and stable id" },
  {
    id: "execution",
    hashes:
      "architecture, randomness and the resolved plan, less three operational fields; derivation, lowering and operator-table versions",
  },
] as const;

export type Component = (typeof COMPONENTS)[number]["id"];

/**
 * Edits, and which of the five each moves, as the design gives them. Architecture identity is
 * structure, configuration and numerics, so it moves when any of those does; execution identity
 * moves with architecture, randomness or the plan, less donation and two other operational
 * fields. Labels and layout enter none of the five.
 *
 * One row is the window's reading, not the column's: untying #E also moves numerics. The untied
 * head is a new leaf with its own initializer declaration, and numerics hashes the initializer
 * declared for each leaf; the design names only Init edits as moving numerics.
 */
export const EDITS: ReadonlyArray<{
  edit: string;
  moves: readonly Component[];
  why: string;
  /** The edit in one plain sentence, as the film says it. */
  say: string;
}> = [
  {
    edit: "SetOperator: silu to gelu_tanh",
    moves: ["structure", "execution"],
    why: "a computation edit",
    say: "SetOperator, silu to gelu_tanh, moves structure and execution.",
  },
  {
    edit: "SetControl: F 768 to 1,024",
    moves: ["configuration", "execution"],
    why: "a root control",
    say: "SetControl, F from 768 to 1,024, moves configuration and execution.",
  },
  {
    edit: "Init: a new scale for #W2",
    moves: ["numerics", "execution"],
    why: "an initializer declaration",
    say: "A new initial scale for #W2 moves numerics and execution.",
  },
  {
    edit: "Untie #E from the head",
    moves: ["structure", "numerics", "randomness", "execution"],
    why: "a tie edit that adds a leaf family, with its own initializer declaration",
    say: "Untying #E from the head moves structure, numerics, randomness and execution.",
  },
  {
    edit: "Plan: Remat on the Repeat",
    moves: ["execution"],
    why: "a resolved plan field",
    say: "A plan edit, Remat on the Repeat, moves execution only.",
  },
  {
    edit: "Plan: f32 contractions",
    moves: ["execution"],
    why: "a resolved plan field",
    say: "A plan edit, f32 contractions, moves execution only.",
  },
  {
    edit: "Plan: donate off",
    moves: [],
    why: "excluded from execution identity",
    say: "A plan edit, donate off, moves none of the five.",
  },
  {
    edit: "Group the halves, rename a definition",
    moves: [],
    why: "labels and layout enter none of the five",
    say: "Grouping the halves and renaming a definition move none of the five.",
  },
];

/** Whether architecture identity, the search's dedup key, moves under an edit. */
export const movesArchitecture = (moves: readonly Component[]) =>
  moves.some((m) => m === "structure" || m === "configuration" || m === "numerics");
