/**
 * The derived backward of a program, and its exact work, for figure C: the window's own model of
 * how AMLO prices a program, by these rules.
 *
 * - Work is a vector of exact counts, scalarized at one unit per arithmetic, transcendental and
 *   compare/select operation;
 * - each elementwise opcode has a price per application;
 * - an elementwise recipe is a straight-line scalar program, whose forward and transpose work are
 *   derived from the program, with one add per extra cotangent arrival;
 * - a contraction costs `terms·(operands − 1) + (terms − cells)`, and its transpose toward each
 *   operand is again a contraction; a sum folds `size(in) − size(out)` cells and its transpose is a
 *   free broadcast;
 * - backward work is reverse mode with respect to the parameters (and here also the block's input,
 *   which inside the model depends on earlier layers); two costs exist only at graph level:
 *   reducing a cotangent back onto a broadcast operand, and accumulating the cotangents of a value
 *   used more than once.
 *
 * Softmax is composed as a row max that is a stop-gradient value, then subtract, exp, sum and
 * divide, and the loss's log-softmax as log and subtract after the same steps. Their backward is
 * derived from those nodes like any other. Rope's rotation of each pair is written as the two
 * rotations by the same rules. AMLO's own pricing of Monad's block, final norm, head and loss gives
 * every count this module gives. AMLO also counts bytes moved, which the window does not draw.
 */

/** Exact counts of elementary scalar operations. */
export interface Work {
  arith: number;
  transcendental: number;
  compare_select: number;
}

export const ZERO: Work = { arith: 0, transcendental: 0, compare_select: 0 };
export const plus = (a: Work, b: Work): Work => ({
  arith: a.arith + b.arith,
  transcendental: a.transcendental + b.transcendental,
  compare_select: a.compare_select + b.compare_select,
});
export const times = (a: Work, n: number): Work => ({
  arith: a.arith * n,
  transcendental: a.transcendental * n,
  compare_select: a.compare_select * n,
});
/** The default scalarization: one unit per operation of every kind. */
export const units = (w: Work) => w.arith + w.transcendental + w.compare_select;

const A: Work = { arith: 1, transcendental: 0, compare_select: 0 };
const T: Work = { arith: 0, transcendental: 1, compare_select: 0 };
const C: Work = { arith: 0, transcendental: 0, compare_select: 1 };

/** The price of one application of each elementwise opcode used here. */
const OPCODE: Record<string, Work> = {
  add: A,
  subtract: A,
  multiply: A,
  divide: A,
  negate: A,
  maximum: C,
  select: C,
  exponential: T,
  log: T,
  rsqrt: T,
  tanh: T,
};

const count = (codes: readonly string[]): Work =>
  codes.reduce((w, c) => plus(w, OPCODE[c]!), ZERO);

// ------------------------------------------------------------------------------- scalar recipes

/** One scalar operation: its forward opcode, and its transpose's opcodes per operand. */
interface Elementary {
  arity: number;
  opcode: string;
  transpose: readonly (readonly string[])[];
  passive?: readonly number[];
  saving?: readonly string[];
}

const ELEMENTARY: Record<string, Elementary> = {
  add: { arity: 2, opcode: "add", transpose: [[], []] },
  sub: { arity: 2, opcode: "subtract", transpose: [[], ["negate"]] },
  mul: { arity: 2, opcode: "multiply", transpose: [["multiply"], ["multiply"]] },
  div: {
    arity: 2,
    opcode: "divide",
    transpose: [["divide"], ["divide", "multiply", "negate"]],
    saving: ["divide"],
  },
  add_const: { arity: 1, opcode: "add", transpose: [[]] },
  mul_const: { arity: 1, opcode: "multiply", transpose: [["multiply"]] },
  neg: { arity: 1, opcode: "negate", transpose: [["negate"]] },
  exp: { arity: 1, opcode: "exponential", transpose: [["multiply"]] },
  log: { arity: 1, opcode: "log", transpose: [["divide"]] },
  max: { arity: 2, opcode: "maximum", transpose: [["select"], ["select"]] },
  rsqrt: {
    arity: 1,
    opcode: "rsqrt",
    transpose: [["multiply", "multiply", "multiply", "divide"]],
  },
  reciprocal: { arity: 1, opcode: "divide", transpose: [["multiply", "multiply", "negate"]] },
  tanh: { arity: 1, opcode: "tanh", transpose: [["multiply", "subtract", "multiply"]] },
  where: { arity: 3, opcode: "select", transpose: [[], ["select"], ["select"]], passive: [0] },
};

/** A straight-line scalar program: inputs are registers 0..n−1, each instruction appends one. */
interface Recipe {
  inputs: number;
  program: ReadonlyArray<readonly [string, readonly number[]]>;
}

const r1 = (op: string): Recipe => ({ inputs: 1, program: [[op, [0]]] });
const r2 = (op: string): Recipe => ({ inputs: 2, program: [[op, [0, 1]]] });

/** Recipes by name: those used here, and the two rotations of a rope pair. */
export const RECIPES: Record<string, Recipe> = {
  add: r2("add"),
  sub: r2("sub"),
  mul: r2("mul"),
  div: r2("div"),
  max: r2("max"),
  neg: r1("neg"),
  exp: r1("exp"),
  log: r1("log"),
  rsqrt: r1("rsqrt"),
  scale: r1("mul_const"),
  shift: r1("add_const"),
  square: { inputs: 1, program: [["mul", [0, 0]]] },
  where: { inputs: 3, program: [["where", [0, 1, 2]]] },
  silu: {
    inputs: 1,
    program: [
      ["neg", [0]],
      ["exp", [1]],
      ["add_const", [2]],
      ["reciprocal", [3]],
      ["mul", [0, 4]],
    ],
  },
  gelu_tanh: {
    inputs: 1,
    program: [
      ["mul", [0, 0]],
      ["mul", [1, 0]],
      ["mul_const", [2]],
      ["add", [3, 0]],
      ["mul_const", [4]],
      ["tanh", [5]],
      ["add_const", [6]],
      ["mul_const", [0]],
      ["mul", [8, 7]],
    ],
  },
  // The two rotations of a rope pair (x0, x1) by the table's (cos, sin): the window's own recipes.
  rot0: {
    inputs: 4,
    program: [
      ["mul", [0, 2]],
      ["mul", [1, 3]],
      ["sub", [4, 5]],
    ],
  },
  rot1: {
    inputs: 4,
    program: [
      ["mul", [0, 3]],
      ["mul", [1, 2]],
      ["add", [4, 5]],
    ],
  },
};

const forwardOf = (r: Recipe) => count(r.program.map(([op]) => ELEMENTARY[op]!.opcode));

function activeRegisters(r: Recipe, wrt: readonly number[]): Set<number> {
  const active = new Set(wrt);
  r.program.forEach(([op, args], i) => {
    const e = ELEMENTARY[op]!;
    if (args.some((a, j) => !e.passive?.includes(j) && active.has(a))) active.add(r.inputs + i);
  });
  return active;
}

/** Reverse mode through a recipe toward some inputs: the opcodes run, and which inputs receive. */
function pullRecipe(r: Recipe, wrt: readonly number[]) {
  const active = activeRegisters(r, wrt);
  const out = r.inputs + r.program.length - 1;
  const arrivals = new Map<number, number>(active.has(out) ? [[out, 1]] : []);
  const codes: string[] = [];
  for (let i = r.program.length - 1; i >= 0; i--) {
    if (!arrivals.has(r.inputs + i)) continue;
    const [op, args] = r.program[i]!;
    const e = ELEMENTARY[op]!;
    const toward = args.flatMap((a, j) => (active.has(a) && !e.passive?.includes(j) ? [j] : []));
    const run = toward.flatMap((j) => e.transpose[j]!);
    // Differentiated toward every operand at once, a shared quotient is computed once.
    if (e.saving && toward.length === e.arity - (e.passive?.length ?? 0) && toward.length > 1)
      for (const s of e.saving) run.splice(run.indexOf(s), 1);
    codes.push(...run);
    for (const j of toward) arrivals.set(args[j]!, (arrivals.get(args[j]!) ?? 0) + 1);
  }
  for (const n of arrivals.values()) for (let k = 1; k < n; k++) codes.push("add");
  return {
    work: count(codes),
    receiving: [...arrivals.keys()].filter((x) => x < r.inputs && wrt.includes(x)),
  };
}

// -------------------------------------------------------------------------------- the program

export type Extents = Readonly<Record<string, number>>;
const size = (axes: readonly string[], e: Extents) =>
  axes.reduce((n, a) => {
    const v = e[a];
    if (v === undefined) throw new Error(`no extent for ${a}`);
    return n * v;
  }, 1);

export interface Value {
  name: string;
  axes: readonly string[];
  kind: "input" | "parameter" | "constant" | "node";
}

export type Primitive =
  | { kind: "ew"; recipe: string; axes: readonly string[] }
  | { kind: "sum" | "max"; axes: readonly string[]; result: readonly string[] }
  | { kind: "contract"; result: readonly string[] }
  | { kind: "relabel" }
  /** Rows looked up by indices: moves values, and adds them back. */
  | { kind: "gather"; result: readonly string[] }
  /** A cast to another dtype: linear, its transpose the cast back. */
  | { kind: "convert"; to: string };

export interface Node {
  result: string;
  prim: Primitive;
  operands: readonly string[];
  /** A part of its operand (index, count of equal parts): a split's parts share one cotangent. */
  part?: readonly [number, number];
  /** A stop-gradient value: no cotangent passes through it. */
  detached?: boolean;
  /** The definition the node belongs to, for drawing and grouping. */
  def: string;
}

/** One step of the derived backward: a pulled node, or the accumulation of a value used twice. */
export interface Step {
  of: string;
  kind: "pull" | "fanout";
  work: Work;
  /** Forward values this step reads. */
  reads: readonly string[];
  def: string;
}

export class Program {
  values = new Map<string, Value>();
  nodes: Node[] = [];

  value(name: string, axes: readonly string[], kind: Value["kind"]) {
    this.values.set(name, { name, axes, kind });
    return name;
  }

  node(def: string, result: string, axes: readonly string[], prim: Primitive, operands: string[]) {
    for (const o of operands) if (!this.values.has(o)) throw new Error(`no value ${o}`);
    this.value(result, axes, "node");
    this.nodes.push({ def, result, prim, operands });
    return result;
  }

  ew(def: string, result: string, recipe: string, operands: string[], axes?: readonly string[]) {
    const out = axes ?? this.values.get(operands[0]!)!.axes;
    return this.node(def, result, out, { kind: "ew", recipe, axes: out }, operands);
  }

  contract(def: string, result: string, operands: string[], axes: readonly string[]) {
    return this.node(def, result, axes, { kind: "contract", result: axes }, operands);
  }

  reduce(
    def: string,
    kind: "sum" | "max",
    result: string,
    operand: string,
    keep: readonly string[],
    o: { detached?: boolean } = {},
  ) {
    const axes = this.values.get(operand)!.axes;
    this.node(def, result, keep, { kind, axes, result: keep }, [operand]);
    if (o.detached) this.nodes.at(-1)!.detached = true;
    return result;
  }

  /** Cast a value to another dtype: the plan inserts these, one per value and dtype. */
  convert(def: string, result: string, operand: string, to: string) {
    const axes = this.values.get(operand)!.axes;
    return this.node(def, result, axes, { kind: "convert", to }, [operand]);
  }

  /** Look up a table's entries at some indices: the result has the indices' axes. */
  gather(def: string, result: string, table: string, indices: string) {
    const axes = this.values.get(indices)!.axes;
    return this.node(def, result, axes, { kind: "gather", result: axes }, [table, indices]);
  }

  relabel(def: string, result: string, operand: string, axes: readonly string[]) {
    return this.node(def, result, axes, { kind: "relabel" }, [operand]);
  }

  forward(n: Node, e: Extents): Work {
    const p = n.prim;
    switch (p.kind) {
      case "ew":
        return times(forwardOf(RECIPES[p.recipe]!), size(p.axes, e));
      case "sum":
        return { ...ZERO, arith: size(p.axes, e) - size(p.result, e) };
      case "max":
        return { ...ZERO, compare_select: size(p.axes, e) - size(p.result, e) };
      case "contract": {
        const all = [...new Set(n.operands.flatMap((o) => this.values.get(o)!.axes))];
        const terms = size(all, e);
        return { ...ZERO, arith: terms * (n.operands.length - 1) + (terms - size(p.result, e)) };
      }
      case "relabel":
      case "gather":
        // A layout or a lookup moves values and computes none; AMLO counts its bytes, which the
        // window does not draw.
        return ZERO;
      case "convert":
        // A cast is plan work, which AMLO's built work rules give no price: it does
        // no arithmetic, transcendental or compare operation.
        return ZERO;
    }
  }

  forwardWork(e: Extents): Work {
    return this.nodes.reduce((w, n) => plus(w, this.forward(n, e)), ZERO);
  }

  /** Values depending differentiably on a parameter or an active input. */
  active(inputs: readonly string[]): Set<string> {
    const on = new Set<string>([
      ...[...this.values.values()].filter((v) => v.kind === "parameter").map((v) => v.name),
      ...inputs,
    ]);
    for (const n of this.nodes) {
      if (n.detached) continue;
      const toward = n.operands.flatMap((o, i) => (on.has(o) ? [i] : []));
      if (!toward.length) continue;
      if (n.prim.kind === "ew") {
        const r = RECIPES[n.prim.recipe]!;
        if (!activeRegisters(r, toward).has(r.inputs + r.program.length - 1)) continue;
      }
      on.add(n.result);
    }
    return on;
  }

  /**
   * The derived backward, step by step, from the outputs to the parameters and the active inputs:
   * each pulled node with its work and the forward values it reads, then one accumulation per
   * value whose cotangent arrives more than once.
   */
  backward(outputs: readonly string[], inputs: readonly string[], e: Extents): Step[] {
    const on = this.active(inputs);
    // Cotangent arrivals per value: whole reads, and the parts of each split read.
    const arrivals = new Map<string, number>(outputs.map((o) => [o, 1]));
    const splits = new Map<string, Map<number, number>>();
    const steps: Step[] = [];
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i]!;
      if (!arrivals.has(n.result) && !splits.has(n.result)) continue;
      const toward = n.operands.flatMap((o, j) => (on.has(o) ? [j] : []));
      if (!toward.length) continue;
      const { work, receiving, reads } = this.pull(n, toward, e);
      steps.push({ of: n.result, kind: "pull", work, reads, def: n.def });
      for (const j of receiving) {
        const o = n.operands[j]!;
        if (n.part) {
          const reads = splits.get(o) ?? new Map<number, number>();
          reads.set(n.part[0], (reads.get(n.part[0]) ?? 0) + 1);
          splits.set(o, reads);
        } else arrivals.set(o, (arrivals.get(o) ?? 0) + 1);
      }
    }
    // A value's cotangents accumulate: one add per extra whole read, a split counting as one
    // read whose parts are each read once.
    for (const name of new Set([...arrivals.keys(), ...splits.keys()])) {
      const v = this.values.get(name)!;
      const whole = (arrivals.get(name) ?? 0) + (splits.has(name) ? 1 : 0);
      let adds = Math.max(whole - 1, 0) * size(v.axes, e);
      for (const k of splits.get(name)?.values() ?? [])
        adds += Math.max(k - 1, 0) * (size(v.axes, e) / 2);
      if (adds)
        steps.push({
          of: name,
          kind: "fanout",
          work: { ...ZERO, arith: adds },
          reads: [],
          def: this.nodes.find((n) => n.result === name)?.def ?? "input",
        });
    }
    return steps;
  }

  private pull(n: Node, toward: number[], e: Extents) {
    const p = n.prim;
    const others = (j: number) => n.operands.filter((_, i) => i !== j);
    switch (p.kind) {
      case "contract": {
        let work = ZERO;
        const reads = new Set<string>();
        for (const j of toward) {
          // The transpose toward operand j contracts the result's cotangent with the others.
          const target = this.values.get(n.operands[j]!)!.axes;
          const rest = others(j).map((o) => this.values.get(o)!.axes);
          const reach = new Set([...p.result, ...rest.flat()]);
          const kept = target.filter((a) => reach.has(a));
          const all = [...new Set([...p.result, ...rest.flat()])];
          const terms = size(all, e);
          const k = rest.length + 1;
          work = plus(work, { ...ZERO, arith: terms * (k - 1) + (terms - size(kept, e)) });
          for (const o of others(j)) reads.add(o);
        }
        return { work, receiving: toward, reads: [...reads] };
      }
      case "ew": {
        const r = RECIPES[p.recipe]!;
        const { work: per, receiving } = pullRecipe(r, toward);
        let work = times(per, size(p.axes, e));
        const full = size(p.axes, e);
        for (const j of receiving) {
          // A cotangent reduced back onto a broadcast operand.
          const axes = this.values.get(n.operands[j]!)!.axes;
          if (size(axes, e) < full) work = plus(work, { ...ZERO, arith: full - size(axes, e) });
        }
        return { work, receiving, reads: readsOf(p.recipe, n) };
      }
      case "sum":
        return { work: ZERO, receiving: toward, reads: [] };
      case "max": {
        const c = size(p.axes, e);
        return {
          work: { ...ZERO, arith: c, compare_select: c },
          receiving: toward,
          reads: [n.operands[0]!, n.result],
        };
      }
      case "relabel":
      case "convert":
        // A layout's transpose is its inverse; a cast's is the cast back. Neither reads a value.
        return { work: ZERO, receiving: toward, reads: [] };
      case "gather":
        // Its transpose adds each looked-up cotangent back into the table, one add per element
        // moved; the indices take no cotangent.
        return {
          work: { ...ZERO, arith: size(p.result, e) },
          receiving: toward.filter((j) => j === 0),
          reads: [],
        };
    }
  }
}

/** The forward values an elementwise recipe's transpose reads. */
function readsOf(recipe: string, n: Node): string[] {
  const [a, b, c, d] = n.operands;
  switch (recipe) {
    case "mul":
      return [a!, b!];
    case "div":
      return [b!, n.result];
    case "exp":
      return [n.result];
    case "log":
      return [a!];
    case "max":
      return [a!, b!];
    case "rsqrt":
      return [a!, n.result];
    case "square":
    case "silu":
    case "gelu_tanh":
      return [a!];
    case "where":
      return [a!];
    case "rot0":
    case "rot1":
      return [c!, d!];
    default:
      return [];
  }
}

// ---------------------------------------------------------------------------- Monad's block

/** Monad's controls, and the declared sequence length the figures price work at (not Monad's). */
export interface Controls {
  V: number;
  D: number;
  L: number;
  H: number;
  K: number;
  F: number;
  T: number;
}

export const extentsOf = (c: Controls): Extents => ({
  t: c.T,
  "t′": c.T,
  d: c.D,
  h: c.H,
  k: c.K,
  p: c.K / 2,
  f: c.F,
  v: c.V,
});

/** The norms of the model, by the definition name their nodes carry. */
export const NORMS = new Set(["n1", "n2", "nf"]);

/** RMSNorm over the last axis, as Monad's definition writes it: g · x · rsqrt(mean(x²) + ε). */
function rmsnorm(g: Program, def: string, x: string, gain: string, rows: readonly string[]) {
  const sq = g.ew(def, `${def}.sq`, "square", [x]);
  const ss = g.reduce(def, "sum", `${def}.ss`, sq, rows);
  const ms = g.ew(def, `${def}.ms`, "scale", [ss]);
  const ve = g.ew(def, `${def}.ve`, "shift", [ms]);
  const r = g.ew(def, `${def}.r`, "rsqrt", [ve]);
  const xr = g.ew(def, `${def}.xr`, "mul", [x, r], g.values.get(x)!.axes);
  return g.ew(def, def, "mul", [xr, gain], g.values.get(x)!.axes);
}

/** Rotate each pair of a [·, h, k] array by the rope table: the two rotations of every pair. */
function rope(g: Program, name: string, x: string, pos: string) {
  const axes = [pos, "h", "p"];
  const x0 = g.relabel("attn", `${name}.even`, x, axes);
  const x1 = g.relabel("attn", `${name}.odd`, x, axes);
  // The halves are a split of x, not two whole reads of it.
  g.nodes.at(-2)!.part = [0, 2];
  g.nodes.at(-1)!.part = [1, 2];
  const cos = pos === "t" ? "cos" : "cos′";
  const sin = pos === "t" ? "sin" : "sin′";
  const o0 = g.ew("attn", `${name}.0`, "rot0", [x0, x1, cos, sin], axes);
  const o1 = g.ew("attn", `${name}.1`, "rot1", [x0, x1, cos, sin], axes);
  // The two halves interleave back into [·, h, k]: a layout, with no arithmetic.
  g.value(`${name}.pairs`, [pos, "h", "k"], "node");
  g.nodes.push({
    def: "attn",
    result: `${name}.pairs`,
    prim: { kind: "relabel" },
    operands: [o0, o1],
  });
  return `${name}.pairs`;
}

/**
 * Monad's block as a program over arrays (the definitions moves.ts draws):
 * x ← x + attn(rmsnorm x), then x ← x + swiglu(rmsnorm x). Its input x is the residual stream.
 */
export function block(): Program {
  const g = new Program();
  g.value("x", ["t", "d"], "input");
  g.value("g1", ["d"], "parameter");
  g.value("g2", ["d"], "parameter");
  for (const w of ["Wq", "Wk", "Wv"]) g.value(w, ["d", "h", "k"], "parameter");
  g.value("Wo", ["h", "k", "d"], "parameter");
  g.value("W1", ["d", "f"], "parameter");
  g.value("W3", ["d", "f"], "parameter");
  g.value("W2", ["f", "d"], "parameter");
  for (const [n, pos] of [
    ["cos", "t"],
    ["sin", "t"],
    ["cos′", "t′"],
    ["sin′", "t′"],
  ] as const)
    g.value(n, [pos, "p"], "constant");
  g.value("mask", ["t", "t′"], "constant");
  g.value("−∞", [], "constant");

  const n1 = rmsnorm(g, "n1", "x", "g1", ["t"]);
  // Attention: one sheet per head; keys and values read the same rows at their own positions.
  const q0 = g.contract("attn", "q0", [n1, "Wq"], ["t", "h", "k"]);
  const q = rope(g, "q", q0, "t");
  const nk = g.relabel("attn", "n1′", n1, ["t′", "d"]);
  const k0 = g.contract("attn", "k0", [nk, "Wk"], ["t′", "h", "k"]);
  const kr = rope(g, "kr", k0, "t′");
  const v = g.contract("attn", "v", [nk, "Wv"], ["t′", "h", "k"]);
  const s = g.contract("attn", "s", [q, kr], ["t", "t′", "h"]);
  const s1 = g.ew("attn", "s1", "scale", [s]);
  const s2 = g.ew("attn", "s2", "where", ["mask", s1, "−∞"], ["t", "t′", "h"]);
  // Softmax over the keys, composed from a row max, subtract, exp, sum and divide: the row max is a
  // stop-gradient value, and the backward is derived from these nodes.
  const SC = ["t", "t′", "h"];
  const top = g.reduce("attn", "max", "sm.max", s2, ["t", "h"], { detached: true });
  const sh = g.ew("attn", "sm.shifted", "sub", [s2, top], SC);
  const ex = g.ew("attn", "sm.exp", "exp", [sh]);
  const sum = g.reduce("attn", "sum", "sm.sum", ex, ["t", "h"]);
  const pr = g.ew("attn", "pr", "div", [ex, sum], SC);
  const o = g.contract("attn", "o", [pr, v], ["t", "h", "k"]);
  const y = g.contract("attn", "a", [o, "Wo"], ["t", "d"]);
  const x1 = g.ew("block", "x1", "add", ["x", y]);
  const n2 = rmsnorm(g, "n2", x1, "g2", ["t"]);
  const a = g.contract("swiglu", "m.a", [n2, "W1"], ["t", "f"]);
  const sg = g.ew("swiglu", "m.g", "silu", [a]);
  const b = g.contract("swiglu", "m.b", [n2, "W3"], ["t", "f"]);
  const hh = g.ew("swiglu", "m.h", "mul", [sg, b]);
  const m = g.contract("swiglu", "m", [hh, "W2"], ["t", "d"]);
  g.ew("block", "x2", "add", [x1, m]);
  return g;
}

/** The block's work at some controls: forward, and the derived backward, by definition. */
export function blockWork(c: Controls) {
  const g = block();
  const e = extentsOf(c);
  const steps = g.backward(["x2"], ["x"], e);
  return {
    forward: g.forwardWork(e),
    backward: steps.reduce((w, s) => plus(w, s.work), ZERO),
    steps,
    program: g,
  };
}

// ------------------------------------------------------------------------------ the whole model

/**
 * Monad's model (gpt) priced per token, for one sequence of `T` tokens.
 * Monad declares no sequence length or batch ("construction only"); the figures declare T and
 * price a single sequence, and say so. Training work here is the forward and the derived backward:
 * the update runs once per step, so its share per token depends on a batch, and it is left out.
 *
 * - The embedding is a gather: rows are moved, not computed (no arithmetic); its transpose adds
 *   each row's cotangent back into #E, one add per element moved.
 * - L iterations of the block: L times the block's work (a Repeat means its unrolling).
 * - The final norm and the tied head, `contract_d(rmsnorm x, E)`.
 * - The loss, cross-entropy composed as a stable log-softmax over v whose row max is a
 *   stop-gradient value, the label's entry gathered, negated and masked, summed, and divided by
 *   the mask's count behind a guard against an empty mask. Its backward is derived from those
 *   nodes. The window scores every position, as Monad's drawn loss reads a label for each; a loss
 *   that leaves out the last position has one row fewer.
 * - #E is read twice, by the gather and by the head, so its cotangents add once (V·D adds).
 */
export function model(c: Controls) {
  const e = extentsOf(c);
  const b = blockWork(c);
  const tail = new Program();
  tail.value("xL", ["t", "d"], "input");
  tail.value("gf", ["d"], "parameter");
  tail.value("E", ["v", "d"], "parameter");
  tail.value("labels", ["t"], "constant");
  tail.value("loss mask", ["t"], "constant");
  tail.value("zero", [], "constant");
  const nf = rmsnorm(tail, "nf", "xL", "gf", ["t"]);
  const logits = tail.contract("gpt", "logits", [nf, "E"], ["t", "v"]);
  const top = tail.reduce("loss", "max", "ls.max", logits, ["t"], { detached: true });
  const sh = tail.ew("loss", "ls.shifted", "sub", [logits, top], ["t", "v"]);
  const ex = tail.ew("loss", "ls.exp", "exp", [sh]);
  const lse = tail.ew("loss", "ls.log", "log", [tail.reduce("loss", "sum", "ls.sum", ex, ["t"])]);
  const lp = tail.ew("loss", "ls.logp", "sub", [sh, lse], ["t", "v"]);
  const picked = tail.gather("loss", "ls.picked", lp, "labels");
  const masked = tail.ew("loss", "ls.masked", "mul", [
    tail.ew("loss", "ls.neg", "neg", [picked]),
    "loss mask",
  ]);
  const total = tail.reduce("loss", "sum", "ls.total", masked, []);
  const count = tail.reduce("loss", "sum", "ls.count", "loss mask", []);
  const any = tail.ew("loss", "ls.any", "max", [count, "zero"], []);
  const mean = tail.ew("loss", "ls.mean", "div", [total, count], []);
  tail.ew("loss", "loss", "where", [any, mean, "zero"], []);
  const tailSteps = tail.backward(["loss"], ["xL"], e);
  const of = (inLoss: boolean) => ({
    forward: tail.nodes
      .filter((n) => (n.def === "loss") === inLoss)
      .reduce((w, n) => plus(w, tail.forward(n, e)), ZERO),
    backward: tailSteps
      .filter((s) => (s.def === "loss") === inLoss)
      .reduce((w, s) => plus(w, s.work), ZERO),
  });
  const parts = {
    embedding: { forward: ZERO, backward: { ...ZERO, arith: c.T * c.D } },
    blocks: { forward: times(b.forward, c.L), backward: times(b.backward, c.L) },
    head: of(false),
    loss: of(true),
    tied: { forward: ZERO, backward: { ...ZERO, arith: c.V * c.D } },
  };
  const forward = Object.values(parts).reduce((w, p) => plus(w, p.forward), ZERO);
  const backward = Object.values(parts).reduce((w, p) => plus(w, p.backward), ZERO);
  return {
    parts,
    forward,
    backward,
    /** Scalar operations per token: the forward, and the forward with the derived backward. */
    perToken: {
      forward: units(forward) / c.T,
      training: (units(forward) + units(backward)) / c.T,
    },
  };
}
