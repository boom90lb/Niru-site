/**
 * The moves of AMLO's 0.0.14 search, applied to Monad's term for figure B:
 *
 * - the classes and their weights, the reach of a selection, and the recorded probability, an
 *   exact fraction of five factors;
 * - admission, one pipeline, cheapest first; kept and minted ids; the population and its
 *   tournament;
 * - one, some or all occurrences, and the overrides of a Repeat; architecture identity;
 * - replace is `SetOperator` only, and crossover is off.
 *
 * Nothing here is a recorded search. The population's ranks, the range of F and the occurrences
 * each `one` or `some` draw selects are declared for the illustration, and the drawings say so.
 * Sites are counted on the definitions as the window draws them, with a norm as one node: AMLO may
 * split nodes differently, and the counts would change with it.
 */
import { MONAD } from "./monad.ts";
import { schedule } from "./dsv41.ts";

// ------------------------------------------------------------------------------------- fractions

/** An exact rational in lowest terms, with a positive denominator. */
export interface Q {
  n: bigint;
  d: bigint;
}

const gcd = (a: bigint, b: bigint): bigint => {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b) [a, b] = [b, a % b];
  return a;
};

export function q(n: bigint | number, d: bigint | number = 1n): Q {
  let N = BigInt(n);
  let D = BigInt(d);
  if (D === 0n) throw new Error("zero denominator");
  if (D < 0n) [N, D] = [-N, -D];
  const g = gcd(N, D) || 1n;
  return { n: N / g, d: D / g };
}

export const mul = (...xs: Q[]): Q => xs.reduce((a, b) => q(a.n * b.n, a.d * b.d), q(1));
export const add = (a: Q, b: Q): Q => q(a.n * b.d + b.n * a.d, a.d * b.d);
export const eq = (a: Q, b: Q) => a.n === b.n && a.d === b.d;

const group = (n: bigint) => n.toLocaleString("en-US");
/** A fraction as the drawings set it: "1/967,680", or a whole number. */
export const show = (x: Q) => (x.d === 1n ? group(x.n) : `${group(x.n)}/${group(x.d)}`);

/**
 * A probability in plain words, for a reader who is not a researcher: "1 in 540" when it is
 * one in a whole number under a thousand, otherwise "about 1 in N" with N to two significant
 * figures, in millions or billions where it is that large.
 */
export function odds(x: Q): string {
  if (x.n <= 0n) throw new Error("no chance");
  if (x.n === x.d) return "certain";
  if (x.d % x.n === 0n && x.d / x.n < 1000n) return `1 in ${group(x.d / x.n)}`;
  const N = Number(x.d) / Number(x.n);
  const digits = Math.floor(Math.log10(N));
  const step = 10 ** (digits - 1);
  const r = Math.round(N / step) * step;
  const words =
    r >= 1e9
      ? `${trim(r / 1e9)} billion`
      : r >= 1e6
        ? `${trim(r / 1e6)} million`
        : r >= 1000
          ? Math.round(r).toLocaleString("en-US")
          : trim(r);
  return `about 1 in ${words}`;
}
const trim = (v: number) => String(Number(v.toPrecision(2)));

/** One factor in plain words: "5 in 18", or "always" for a factor of 1. */
export const chance = (x: Q) => (x.n === x.d ? "always" : `${group(x.n)} in ${group(x.d)}`);

export function choose(n: number, k: number): bigint {
  if (k < 0 || k > n) return 0n;
  let r = 1n;
  for (let i = 1; i <= k; i++) r = (r * BigInt(n - k + i)) / BigInt(i);
  return r;
}

// ---------------------------------------------------------------------------------- the registry

export type Class = "replace" | "insert" | "delete" | "binding" | "shape";
export const CLASSES: readonly Class[] = ["replace", "insert", "delete", "binding", "shape"];

/** Declared class weights. Crossover's 1/10 is off: 0.0.14 leaves it out. */
export const WEIGHTS: Record<Class, Q> = {
  replace: q(1, 4),
  insert: q(1, 4),
  delete: q(1, 10),
  binding: q(1, 10),
  shape: q(1, 5),
};

/** The activation family that `SetOperator` moves within. */
export const FAMILY = ["relu", "gelu_tanh", "gelu_erf", "silu", "tanh", "sigmoid", "square"];

/**
 * What `InsertNode` may place at a real-valued output: a family op, or a call to a T→T library
 * block: `rms_norm`, `learned_scale` or `causal_dwconv3`, an identity-initialized convolution.
 */
export const INSERTS = [...FAMILY, "rms_norm", "learned_scale", "causal_dwconv3"];

/** The weights renormalized over the classes that have a legal site. */
export function weights(has: Record<Class, number>): Record<Class, Q> {
  const live = CLASSES.filter((c) => has[c] > 0);
  const total = live.reduce((s, c) => add(s, WEIGHTS[c]), q(0));
  return Object.fromEntries(
    CLASSES.map((c) => [c, live.includes(c) ? mul(WEIGHTS[c], q(total.d, total.n)) : q(0)]),
  ) as Record<Class, Q>;
}

/** The chance that a size-S tournament, drawn from P members without replacement, picks rank r. */
export const tournament = (P: number, S: number, r: number): Q =>
  q(choose(P - r, S - 1), choose(P, S));

export type ReachKind = "all" | "one" | "some";

/**
 * The reach of a command over n occurrences: for n ≥ 3, all 2/5, one 2/5 and
 * some 1/5; for n = 2, all or one at 1/2; for n = 1, all. One is uniform over the occurrences;
 * some draws k uniformly in 2..n−1, then a k-subset uniformly.
 */
export function reach(n: number, kind: ReachKind, k = 0): Q {
  if (n < 1) throw new Error("no occurrence");
  if (kind === "all") return n >= 3 ? q(2, 5) : n === 2 ? q(1, 2) : q(1);
  if (kind === "one") {
    if (n === 1) throw new Error("one occurrence: the reach is all");
    return mul(n >= 3 ? q(2, 5) : q(1, 2), q(1, n));
  }
  if (n < 3 || k < 2 || k > n - 1) throw new Error(`no some of size ${k} in ${n}`);
  return mul(q(1, 5), q(1, n - 2), q(1n, choose(n, k)));
}

// -------------------------------------------------------------------------------------- the term

export interface Ty {
  kind: "real" | "fin" | "bool";
  axes: readonly string[];
}

export interface Node {
  name: string;
  /** An operator tag, a library block, "call" or "repeat". */
  op: string;
  /** For a call or a Repeat: the definition called, and its parameters' bindings. */
  call?: string;
  axes?: Readonly<Record<string, string | number>>;
  /** Inputs of the definition, its slots (`#W`) or earlier nodes, by name. */
  args: readonly string[];
  out: Ty;
}

export interface Def {
  name: string;
  params: readonly string[];
  inputs: Readonly<Record<string, Ty>>;
  /** Leaf families this definition creates, by slot name, with their shapes. */
  slots: Readonly<Record<string, readonly string[]>>;
  nodes: readonly Node[];
  /** Drawn as one glyph; its body is not counted for sites. */
  glyph?: boolean;
}

export interface Body {
  def: string;
  axes: Readonly<Record<string, string | number>>;
}

/** A program with one root definition and, in it, one Repeat (Monad's shape). */
export interface Term {
  defs: Readonly<Record<string, Def>>;
  controls: Readonly<Record<string, number>>;
  /** Declared ranges [lo, hi] that the shape class may use. */
  ranges: Readonly<Record<string, readonly [number, number]>>;
  /** Per-iteration override bodies of the Repeat. */
  overrides: ReadonlyMap<number, Body>;
  /** Repeat slots bound `shared` instead of `stacked`, by family address. */
  shared: ReadonlySet<string>;
}

const R = (...axes: string[]): Ty => ({ kind: "real", axes });
const ROOT = "gpt";
const REPEAT = "x";

/**
 * Monad's definitions, node for node as the window draws them:
 * the model view, the block, attention and SwiGLU. Keys and values use the drawing's t′.
 */
const MONAD_DEFS: Record<string, Def> = {
  gpt: {
    name: "gpt",
    params: [],
    inputs: {
      inputs: { kind: "fin", axes: ["T"] },
      labels: { kind: "fin", axes: ["T"] },
      loss_mask: { kind: "bool", axes: ["T"] },
    },
    slots: { E: ["V", "D"] },
    nodes: [
      { name: "rope", op: "rope_table", args: [], out: R("T", "P", "2") },
      { name: "x0", op: "gather", args: ["#E", "inputs"], out: R("T", "D") },
      {
        name: REPEAT,
        op: "repeat",
        call: "block",
        axes: { t: "T", d: "D", h: "H", k: "K", p: "P", f: "F" },
        args: ["x0", "rope"],
        out: R("T", "D"),
      },
      {
        name: "n",
        op: "call",
        call: "rmsnorm",
        axes: { n: "D" },
        args: [REPEAT],
        out: R("T", "D"),
      },
      { name: "logits", op: "contract", args: ["n", "#E"], out: R("T", "V") },
      {
        name: "loss",
        op: "cross_entropy",
        args: ["logits", "labels", "loss_mask"],
        out: R(),
      },
    ],
  },
  block: {
    name: "block",
    params: ["t", "d", "h", "k", "p", "f"],
    inputs: { x: R("t", "d"), rope: R("t", "p", "2") },
    slots: {},
    nodes: [
      { name: "n1", op: "call", call: "rmsnorm", axes: { n: "d" }, args: ["x"], out: R("t", "d") },
      {
        name: "a",
        op: "call",
        call: "attn",
        axes: { t: "t", d: "d", h: "h", k: "k", p: "p" },
        args: ["n1", "rope"],
        out: R("t", "d"),
      },
      { name: "x1", op: "add", args: ["x", "a"], out: R("t", "d") },
      {
        name: "n2",
        op: "call",
        call: "rmsnorm",
        axes: { n: "d" },
        args: ["x1"],
        out: R("t", "d"),
      },
      {
        name: "m",
        op: "call",
        call: "swiglu",
        axes: { d: "d", f: "f" },
        args: ["n2"],
        out: R("t", "d"),
      },
      { name: "x2", op: "add", args: ["x1", "m"], out: R("t", "d") },
    ],
  },
  attn: {
    name: "attn",
    params: ["t", "d", "h", "k", "p"],
    inputs: { x: R("t", "d"), rope: R("t", "p", "2") },
    slots: { Wq: ["d", "h", "k"], Wk: ["d", "h", "k"], Wv: ["d", "h", "k"], Wo: ["h", "k", "d"] },
    nodes: [
      { name: "q0", op: "contract", args: ["x", "#Wq"], out: R("t", "h", "k") },
      { name: "q", op: "rope", args: ["q0", "rope"], out: R("t", "h", "k") },
      { name: "k0", op: "contract", args: ["x", "#Wk"], out: R("t′", "h", "k") },
      { name: "kr", op: "rope", args: ["k0", "rope"], out: R("t′", "h", "k") },
      { name: "v", op: "contract", args: ["x", "#Wv"], out: R("t′", "h", "k") },
      { name: "s", op: "contract", args: ["q", "kr"], out: R("t", "t′", "h") },
      { name: "s1", op: "scale", args: ["s"], out: R("t", "t′", "h") },
      { name: "s2", op: "where", args: ["s1"], out: R("t", "t′", "h") },
      { name: "pr", op: "softmax", args: ["s2"], out: R("t", "t′", "h") },
      { name: "o", op: "contract", args: ["pr", "v"], out: R("t", "h", "k") },
      { name: "y", op: "contract", args: ["o", "#Wo"], out: R("t", "d") },
    ],
  },
  swiglu: {
    name: "swiglu",
    params: ["d", "f"],
    inputs: { x: R("t", "d") },
    slots: { W1: ["d", "f"], W3: ["d", "f"], W2: ["f", "d"] },
    nodes: [
      { name: "a", op: "contract", args: ["x", "#W1"], out: R("t", "f") },
      { name: "g", op: "silu", args: ["a"], out: R("t", "f") },
      { name: "b", op: "contract", args: ["x", "#W3"], out: R("t", "f") },
      { name: "h", op: "mul", args: ["g", "b"], out: R("t", "f") },
      { name: "y", op: "contract", args: ["h", "#W2"], out: R("t", "d") },
    ],
  },
  rmsnorm: {
    name: "rmsnorm",
    params: ["n"],
    inputs: { x: R("t", "n") },
    slots: { g: ["n"] },
    nodes: [],
    glyph: true,
  },
};

/** Monad as 0.0.14 constructs it, with F's range declared for this illustration. */
export const SEED: Term = {
  defs: MONAD_DEFS,
  controls: { ...MONAD, P: MONAD.K / 2 },
  ranges: { F: [512, 1024] },
  overrides: new Map(),
  shared: new Set(),
};

const repeatNode = (t: Term) => t.defs[ROOT]!.nodes.find((n) => n.name === REPEAT)!;
const base = (t: Term): Body => ({ def: repeatNode(t).call!, axes: repeatNode(t).axes! });

/** One use of a definition: its path, its axis values, and the Repeat iteration it sits in. */
export interface Instance {
  def: string;
  path: string;
  /** The family address: the path with iteration markers stripped. */
  address: string;
  axes: Record<string, number>;
  iteration?: number;
}

export function instances(t: Term): Instance[] {
  const out: Instance[] = [];
  const visit = (
    def: string,
    path: string,
    address: string,
    axes: Record<string, number>,
    iteration?: number,
  ) => {
    out.push({ def, path, address, axes, ...(iteration === undefined ? {} : { iteration }) });
    for (const n of t.defs[def]!.nodes) {
      if (!n.call) continue;
      const bind = (b: Readonly<Record<string, string | number>>) =>
        Object.fromEntries(
          Object.entries(b).map(([k, v]) => [k, typeof v === "number" ? v : axes[v]!]),
        );
      if (n.op === "repeat") {
        for (let i = 0; i < t.controls.L!; i++) {
          const body = t.overrides.get(i) ?? base(t);
          visit(body.def, `${path}${n.name}[${i}]/`, `${address}${n.name}/`, bind(body.axes), i);
        }
      } else visit(n.call, `${path}${n.name}/`, `${address}${n.name}/`, bind(n.axes!), iteration);
    }
  };
  visit(ROOT, "", "", { ...t.controls });
  return out;
}

export const occurrences = (t: Term, def: string) =>
  instances(t).filter((i) => i.def === def).length;

/** Parameters, counted per use: a stacked slot once per iteration, a shared one once. */
export function parameters(t: Term): number {
  let n = 0;
  const seen = new Set<string>();
  for (const inst of instances(t))
    for (const [slot, shape] of Object.entries(t.defs[inst.def]!.slots)) {
      const family = `${inst.address}${slot}`;
      if (inst.iteration !== undefined && t.shared.has(family)) {
        if (seen.has(family)) continue;
        seen.add(family);
      }
      n += shape.reduce((s, a) => s * (a === "2" ? 2 : inst.axes[a]!), 1);
    }
  return n;
}

// ------------------------------------------------------------------------------------- the sites

export interface Site {
  cls: Class;
  def: string;
  /** A node, a family address or a control. */
  at: string;
}

const sameAxes = (a: Ty, b: Ty) =>
  a.kind === b.kind && a.axes.length === b.axes.length && a.axes.every((x, i) => x === b.axes[i]);

/** The type of an argument: an input, a slot (as a real array) or an earlier node. */
function argType(def: Def, name: string): Ty | undefined {
  if (name.startsWith("#")) {
    const shape = def.slots[name.slice(1)];
    return shape ? R(...shape) : undefined;
  }
  return def.inputs[name] ?? def.nodes.find((n) => n.name === name)?.out;
}

/** The legal sites of each class, in a stable order. */
export function sites(t: Term): Record<Class, Site[]> {
  const used = new Set(instances(t).map((i) => i.def));
  const defs = Object.values(t.defs).filter((d) => used.has(d.name) && !d.glyph);
  const out: Record<Class, Site[]> = {
    replace: [],
    insert: [],
    delete: [],
    binding: [],
    shape: [],
  };
  for (const d of defs)
    for (const n of d.nodes) {
      if (FAMILY.includes(n.op)) out.replace.push({ cls: "replace", def: d.name, at: n.name });
      if (n.out.kind === "real") out.insert.push({ cls: "insert", def: d.name, at: n.name });
      if (
        n.args.some((a) => {
          const ty = argType(d, a);
          return ty !== undefined && sameAxes(ty, n.out);
        })
      )
        out.delete.push({ cls: "delete", def: d.name, at: n.name });
    }
  // Binding: each Repeat slot, stacked to shared or back; each slot read twice, untied.
  const families = new Set<string>();
  for (const inst of instances(t))
    if (inst.iteration !== undefined)
      for (const slot of Object.keys(t.defs[inst.def]!.slots))
        families.add(`${inst.address}${slot}`);
  for (const f of [...families].sort()) out.binding.push({ cls: "binding", def: ROOT, at: f });
  for (const d of defs)
    for (const slot of Object.keys(d.slots))
      if (d.nodes.filter((n) => n.args.includes(`#${slot}`)).length > 1)
        out.binding.push({ cls: "binding", def: d.name, at: `#${slot}` });
  // Shape: a control with a declared range, and each binding that resolves to one.
  for (const c of Object.keys(t.ranges)) out.shape.push({ cls: "shape", def: ROOT, at: c });
  const ranged = (def: string, param: string): boolean => {
    for (const d of Object.values(t.defs))
      for (const n of d.nodes)
        if (n.call === def && n.axes) {
          const v = n.axes[param];
          if (typeof v !== "string") continue;
          if (d.name === ROOT ? v in t.ranges : ranged(d.name, v)) return true;
        }
    return false;
  };
  for (const d of Object.values(t.defs))
    for (const n of d.nodes)
      if (n.call && n.axes && used.has(n.call))
        for (const [param, v] of Object.entries(n.axes))
          if (typeof v === "string" && (d.name === ROOT ? v in t.ranges : ranged(d.name, v)))
            out.shape.push({ cls: "shape", def: n.call, at: param });
  return out;
}

export const counts = (t: Term) =>
  Object.fromEntries(Object.entries(sites(t)).map(([c, s]) => [c, s.length])) as Record<
    Class,
    number
  >;

// ---------------------------------------------------------------------------------- the commands

/** The iterations a selection names; `all` is every one. */
type Selection = "all" | readonly number[];

/** A fresh name for a copy of a definition, as `block.1`. */
function fresh(t: Term, name: string): string {
  for (let i = 1; ; i++) if (!t.defs[`${name}.${i}`]) return `${name}.${i}`;
}

/**
 * Apply a change to the Repeat body at the selected iterations. Selecting every use folds the
 * change into the body itself; a subset gets an override body, a copy of the body definition
 * (only the callers and ancestors that need isolating are copied).
 */
function atIterations(t: Term, sel: Selection, change: (d: Def) => Def): Term {
  if (sel === "all" || sel.length === t.controls.L) {
    const name = base(t).def;
    return { ...t, defs: { ...t.defs, [name]: change(t.defs[name]!) } };
  }
  const name = fresh(t, base(t).def);
  const copy = { ...change(t.defs[base(t).def]!), name };
  const overrides = new Map(t.overrides);
  for (const i of sel) overrides.set(i, { def: name, axes: (t.overrides.get(i) ?? base(t)).axes });
  return { ...t, defs: { ...t.defs, [name]: copy }, overrides };
}

/** `SetOperator` on a node of a definition the Repeat body calls, at every use. */
export function setOperator(t: Term, def: string, node: string, op: string): Term {
  const d = t.defs[def]!;
  const nodes = d.nodes.map((n) => (n.name === node ? { ...n, op } : n));
  return { ...t, defs: { ...t.defs, [def]: { ...d, nodes } } };
}

/** `InsertNode` of a library block at a node's output in the Repeat body. */
export function insertNode(t: Term, after: string, op: string, sel: Selection): Term {
  return atIterations(t, sel, (d) => {
    const i = d.nodes.findIndex((n) => n.name === after);
    const name = `${after}.${op}`;
    const added: Node = { name, op, args: [after], out: d.nodes[i]!.out };
    const nodes = d.nodes.map((n) => ({
      ...n,
      args: n.args.map((a) => (a === after ? name : a)),
    }));
    return { ...d, nodes: [...nodes.slice(0, i + 1), added, ...nodes.slice(i + 1)] };
  });
}

/** `DeleteNode(redirect)` in the Repeat body: consumers read the input of equal type instead. */
export function deleteNode(t: Term, node: string, sel: Selection): Term {
  return atIterations(t, sel, (d) => {
    const gone = d.nodes.find((n) => n.name === node)!;
    const to = gone.args.find((a) => {
      const ty = argType(d, a);
      return ty !== undefined && sameAxes(ty, gone.out);
    })!;
    const nodes = d.nodes
      .filter((n) => n.name !== node)
      .map((n) => ({ ...n, args: n.args.map((a) => (a === node ? to : a)) }));
    return { ...d, nodes };
  });
}

/** `SetBind` on a Repeat slot: stacked to shared. */
export const share = (t: Term, family: string): Term => ({
  ...t,
  shared: new Set([...t.shared, family]),
});

/**
 * `SetAxisBinding` of a body parameter at selected iterations: each gets an override body with
 * its own axes, and the body definition is not copied.
 */
export function setAxisBinding(
  t: Term,
  param: string,
  value: number,
  sel: readonly number[],
): Term {
  const overrides = new Map(t.overrides);
  for (const i of sel) {
    const body = t.overrides.get(i) ?? base(t);
    overrides.set(i, { def: body.def, axes: { ...body.axes, [param]: value } });
  }
  return { ...t, overrides };
}

// -------------------------------------------------------------------------------------- identity

/**
 * Architecture identity, in miniature: structure and configuration with
 * names alpha-renamed. Definitions are written out by content, nodes and slots by position, so a
 * copy and its original, or two copies made along different paths, compare equal. Initializers
 * are equal in every term here, so numerics are left out; a hash in AMLO, a string here.
 */
export function identity(t: Term): string {
  const memo = new Map<string, string>();
  const canon = (name: string): string => {
    const hit = memo.get(name);
    if (hit) return hit;
    const d = t.defs[name]!;
    const slots = Object.keys(d.slots);
    const inputs = Object.keys(d.inputs);
    const ref = (a: string) =>
      a.startsWith("#")
        ? `s${slots.indexOf(a.slice(1))}`
        : inputs.includes(a)
          ? `i${inputs.indexOf(a)}`
          : `n${d.nodes.findIndex((n) => n.name === a)}`;
    const pos = (v: string | number) =>
      typeof v === "number" ? v : d.params.includes(v) ? `p${d.params.indexOf(v)}` : v;
    const body = d.nodes.map((n) => [
      n.op,
      n.op === "repeat" ? repeat() : n.call ? canon(n.call) : null,
      n.axes ? Object.values(n.axes).map(pos) : null,
      n.args.map(ref),
      n.out.kind,
      n.out.axes.map(pos),
    ]);
    const s = JSON.stringify([
      d.params.length,
      Object.values(d.inputs).map((ty) => [ty.kind, ty.axes.map(pos)]),
      Object.values(d.slots).map((shape) => shape.map(pos)),
      d.glyph ? name : body,
    ]);
    memo.set(name, s);
    return s;
  };
  // Family addresses use node names, which copies keep, so the shared set compares as it is.
  const repeat = () =>
    JSON.stringify([
      canon(base(t).def),
      [...t.overrides.entries()]
        .sort(([a], [b]) => a - b)
        .map(([i, b]) => [i, canon(b.def), Object.values(b.axes)]),
      [...t.shared].sort(),
    ]);
  return JSON.stringify([canon(ROOT), Object.entries(t.controls).sort()]);
}

// ------------------------------------------------------------------------------------- proposals

/** The declared population: P members, tournaments of S (the design's defaults). */
export const POPULATION = { P: 24, S: 4 } as const;

export interface Factor {
  name: string;
  q: Q;
  note: string;
  /** The draw in one plain sentence. */
  say: string;
}

export interface Move {
  id: string;
  cls: Class;
  command: string;
  /** What the move does, in words. */
  does: string;
  /** What the move does, in one plain sentence for a reader who is not a researcher. */
  say: string;
  parent: { name: string; rank: number };
  /** The occurrences of the edited definition, and the ones the reach selects. */
  n: number;
  reach: ReachKind;
  selected: readonly number[];
  factors: Factor[];
  probability: Q;
}

/** A move on Monad's term, with the child it makes. */
export interface Proposal extends Move {
  child: Term;
}

const ordinal = (r: number) => ["first", "second", "third"][r - 1] ?? `${r}th`;
/** "7, 30 and 51". */
const list2 = (xs: readonly number[]) =>
  xs.length < 2 ? `${xs[0]}` : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`;

function propose(o: {
  id: string;
  cls: Class;
  command: string;
  does: string;
  say: string;
  /** The site and the choice in plain words, and the reach where the default would mislead. */
  where: string;
  what: string;
  reachSay?: string;
  parent: { name: string; rank: number; term: Term };
  site: (s: Site) => boolean;
  siteNote: string;
  choice: Q;
  choiceNote: string;
  def: string;
  reach: ReachKind;
  selected: readonly number[];
  child: Term;
}): Proposal {
  const { P, S } = POPULATION;
  const all = sites(o.parent.term);
  const list = all[o.cls];
  const k = list.findIndex(o.site);
  if (k < 0) throw new Error(`${o.id}: no such site`);
  const w = weights(counts(o.parent.term));
  const n = occurrences(o.parent.term, o.def);
  const r = reach(n, o.reach, o.selected.length);
  const live = CLASSES.filter((c) => all[c].length > 0).length;
  const factors: Factor[] = [
    {
      name: "parent",
      q: tournament(P, S, o.parent.rank),
      note: `${o.parent.name}, ranked ${ordinal(o.parent.rank)} of ${P}, in a tournament of ${S}`,
      say: `The parent is ${o.parent.name}, ranked ${ordinal(o.parent.rank)}.`,
    },
    {
      name: "class",
      q: w[o.cls],
      note: `${o.cls}: ${show(WEIGHTS[o.cls])}, renormalized over ${live} classes`,
      say: `The class drawn is ${o.cls}.`,
    },
    {
      name: "site",
      q: q(1, list.length),
      note: o.siteNote,
      say: `The place is ${o.where}, ${list.length === 1 ? "the only one" : `1 of ${list.length}`}.`,
    },
    { name: "choice", q: o.choice, note: o.choiceNote, say: o.what },
    {
      name: "reach",
      q: r,
      note:
        o.reach === "all"
          ? n === 1
            ? "one use, so all"
            : `all ${n} layers`
          : o.reach === "one"
            ? `one of ${n} layers`
            : `some: ${o.selected.length} of ${n} layers`,
      say:
        o.reachSay ??
        (o.reach === "all"
          ? n === 1
            ? "It is used once, so the change reaches all of it."
            : `The change reaches all ${n} layers.`
          : o.reach === "one"
            ? `The change reaches one layer of ${n}: layer ${o.selected[0]}.`
            : `The change reaches ${o.selected.length} layers of ${n}: ${list2(o.selected)}.`),
    },
  ];
  return {
    id: o.id,
    cls: o.cls,
    command: o.command,
    does: o.does,
    say: o.say,
    parent: { name: o.parent.name, rank: o.parent.rank },
    n,
    reach: o.reach,
    selected: o.selected,
    factors,
    probability: mul(...factors.map((f) => f.q)),
    child: o.child,
  };
}

/** The occurrences each drawn `one` or `some` selects, declared for the illustration. */
export const PICKS = { insert: [12], delete: [7, 30, 51], shape: [20, 41] } as const;

const seed = { name: "the seed", rank: 1, term: SEED };
/** The widths F's declared range allows, the current one among them. */
const WIDTHS = SEED.ranges.F![1] - SEED.ranges.F![0] + 1;
const W2 = `${REPEAT}/m/W2`;

/** One move of each class, each from the seed. */
export function moves(): Proposal[] {
  const L = SEED.controls.L!;
  return [
    propose({
      id: "replace",
      cls: "replace",
      command: "SetOperator",
      does: "silu becomes gelu_tanh in swiglu",
      say: "Replace silu with gelu_tanh, in every layer.",
      where: "silu in SwiGLU",
      what: "The new operation is gelu_tanh, 1 of 6 others.",
      parent: seed,
      site: (s) => s.def === "swiglu" && s.at === "g",
      siteNote: `silu in swiglu, the only family op drawn`,
      choice: q(1, FAMILY.length - 1),
      choiceNote: `gelu_tanh, one of ${FAMILY.length - 1} other members`,
      def: "swiglu",
      reach: "all",
      selected: [],
      child: setOperator(SEED, "swiglu", "g", "gelu_tanh"),
    }),
    propose({
      id: "insert",
      cls: "insert",
      command: "InsertNode",
      does: "the library’s rms_norm on attention’s output",
      say: "Insert a norm after attention, in one layer.",
      where: "attention’s output",
      what: "The new node is rms_norm, 1 of 10 that fit there.",
      parent: seed,
      site: (s) => s.def === "block" && s.at === "a",
      siteNote: "attention’s output, one of the real-valued outputs",
      choice: q(1, INSERTS.length),
      choiceNote: `rms_norm, one of ${INSERTS.length} that fit there`,
      def: "block",
      reach: "one",
      selected: PICKS.insert,
      child: insertNode(SEED, "a", "rms_norm", PICKS.insert),
    }),
    propose({
      id: "delete",
      cls: "delete",
      command: "DeleteNode",
      does: "the norm before attention",
      say: "Delete the norm before attention, in three layers.",
      where: "the norm before attention",
      what: "Its input takes its place, the only choice.",
      parent: seed,
      site: (s) => s.def === "block" && s.at === "n1",
      siteNote: "the norm before attention: input and output types equal",
      choice: q(1),
      choiceNote: "one input to redirect to",
      def: "block",
      reach: "some",
      selected: PICKS.delete,
      child: deleteNode(SEED, "n1", PICKS.delete),
    }),
    propose({
      id: "binding",
      cls: "binding",
      command: "SetBind",
      does: `#W2 of swiglu, stacked over ${L} layers, becomes one shared leaf`,
      say: `Share one #W2 across all ${L} layers.`,
      where: "#W2 of SwiGLU",
      what: "Stacked becomes shared, the only choice.",
      reachSay: `The Repeat is used once, so all ${L} layers change.`,
      parent: seed,
      site: (s) => s.at === W2,
      siteNote: "#W2, one of the Repeat’s slots or the tied #E",
      choice: q(1),
      choiceNote: "stacked to shared",
      def: ROOT,
      reach: "all",
      selected: [],
      child: share(SEED, W2),
    }),
    propose({
      id: "shape",
      cls: "shape",
      command: "SetAxisBinding",
      does: "f becomes 1,024 in the selected layers",
      say: "Widen f to 1,024, in two layers.",
      where: "the width f of the block",
      what: `The new width is 1,024, 1 of ${fmtN(WIDTHS - 1)} others in the declared range.`,
      parent: seed,
      site: (s) => s.def === "block" && s.at === "f",
      siteNote: "f where the Repeat binds block, one of F’s three",
      choice: q(1, WIDTHS - 1),
      choiceNote: `1,024, one of ${fmtN(WIDTHS - 1)} other widths`,
      def: "block",
      reach: "some",
      selected: PICKS.shape,
      child: setAxisBinding(SEED, "f", 1024, PICKS.shape),
    }),
  ];
}

const fmtN = (n: number) => n.toLocaleString("en-US");

/**
 * Two moves, one architecture. A and B joined the population as children of the seed (the
 * replace and insert moves above); suppose that a generation later they rank first and second,
 * ranks declared for the illustration. From A,
 * insert at layer 12; from B, replace at every layer. The children differ in their minted ids,
 * which identity renames, so they are one architecture, and the later is refused.
 */
export function commuting(): { a: Term; b: Term; fromA: Proposal; fromB: Proposal } {
  const [replace, insert] = moves();
  const a = replace!.child;
  const b = insert!.child;
  const fromA = propose({
    id: "from-a",
    cls: "insert",
    command: "InsertNode",
    does: "rms_norm on attention’s output",
    say: "From A, insert a norm after attention, in one layer.",
    where: "attention’s output",
    what: "The new node is rms_norm, 1 of 10.",
    parent: { name: "A", rank: 1, term: a },
    site: (s) => s.def === "block" && s.at === "a",
    siteNote: "attention’s output, one of the real-valued outputs",
    choice: q(1, INSERTS.length),
    choiceNote: `rms_norm, one of ${INSERTS.length}`,
    def: "block",
    reach: "one",
    selected: PICKS.insert,
    child: insertNode(a, "a", "rms_norm", PICKS.insert),
  });
  const fromB = propose({
    id: "from-b",
    cls: "replace",
    command: "SetOperator",
    does: "silu becomes gelu_tanh in swiglu",
    say: "From B, replace silu with gelu_tanh, in every layer.",
    where: "silu in SwiGLU",
    what: "The new operation is gelu_tanh, 1 of 6 others.",
    parent: { name: "B", rank: 2, term: b },
    site: (s) => s.def === "swiglu" && s.at === "g",
    siteNote: "silu in swiglu",
    choice: q(1, FAMILY.length - 1),
    choiceNote: `gelu_tanh, one of ${FAMILY.length - 1}`,
    def: "swiglu",
    reach: "all",
    selected: [],
    child: setOperator(b, "swiglu", "g", "gelu_tanh"),
  });
  return { a, b, fromA, fromB };
}

/**
 * Admission, cheapest first: each proposal gets exactly one Admitted or
 * Refused(reason). What each gate checks, in two short lines for the drawing.
 */
export const GATES = [
  { name: "decode", does: ["the patch is", "well formed"] },
  { name: "base", does: ["the parent’s", "digest resolves"] },
  { name: "patch", does: ["dedup on (parent", "digest, patch_id)"] },
  { name: "transact", does: ["types, apply,", "check, lowering"] },
  { name: "ceilings", does: ["nodes, definitions,", "work, steps, memory"] },
  { name: "cost", does: ["modeled", "work"] },
  { name: "identity", does: ["dedup on", "architecture identity"] },
] as const;

// ------------------------------------------------------------------------------------- DeepSeek

/**
 * The toy DeepSeek's activation-family ops, as the window draws them: the hyper-connection's
 * two sigmoids, the indexer's relu, the SiLU of the routed and of the shared experts, and the
 * sigmoid of Engram's gate. Its other classes have sites too (an attention output to insert at,
 * a norm to delete, the stacked slots of every run); no control declares a range, so shape has
 * none.
 */
export const DS_REPLACE = [
  { def: "mhc_mixes", op: "sigmoid", where: "the pre-mix" },
  { def: "mhc_mixes", op: "sigmoid", where: "the post-mix" },
  { def: "index_score", op: "relu", where: "the index scores" },
  { def: "experts_dense", op: "silu", where: "the routed experts" },
  { def: "expert", op: "silu", where: "the shared expert" },
  { def: "engram", op: "sigmoid", where: "Engram’s gate" },
] as const;

/** The decoder's nested Repeats: Dec = Repeat(dec_group, 4) from layer 24. */
export const DEC = { first: 24, groups: 4, size: 4 } as const;

export function deepseek(layer = 26): Move & {
  group: number;
  inner: number;
  copies: string[];
} {
  const { layers } = schedule();
  const has: Record<Class, number> = {
    replace: DS_REPLACE.length,
    insert: 1,
    delete: 1,
    binding: 1,
    shape: 0,
  };
  const w = weights(has);
  const n = layers.length;
  const group = Math.floor((layer - DEC.first) / DEC.size);
  const inner = ((layer - DEC.first) % DEC.size) - 1;
  if (layers[layer] !== "reuse1" || inner < 0) throw new Error(`layer ${layer} is not a Reuse`);
  const { P, S } = POPULATION;
  const factors: Factor[] = [
    {
      name: "parent",
      q: tournament(P, S, 1),
      note: `the seed, ranked first of ${P}, in a tournament of ${S}`,
      say: "The parent is the seed, ranked first.",
    },
    {
      name: "class",
      q: w.replace,
      note: "replace: 1/4, renormalized over 4 classes",
      say: "The class drawn is replace.",
    },
    {
      name: "site",
      q: q(1, DS_REPLACE.length),
      note: `silu in the routed experts, one of ${DS_REPLACE.length} family ops`,
      say: `The place is silu in the routed experts, 1 of ${DS_REPLACE.length}.`,
    },
    {
      name: "choice",
      q: q(1, FAMILY.length - 1),
      note: "gelu_tanh, one of 6 other members",
      say: "The new operation is gelu_tanh, 1 of 6 others.",
    },
    {
      name: "reach",
      q: reach(n, "one"),
      note: `one of ${n} layers`,
      say: `The change reaches one layer of ${n}, layer ${layer}.`,
    },
  ];
  return {
    id: "deepseek",
    cls: "replace",
    command: "SetOperator",
    does: "silu becomes gelu_tanh in the routed experts",
    say: `Replace silu with gelu_tanh in the routed experts, in layer ${layer} only.`,
    parent: { name: "the seed", rank: 1 },
    n,
    reach: "one",
    selected: [layer],
    factors,
    probability: mul(...factors.map((f) => f.q)),
    group,
    inner,
    copies: ["dec_group", "layer_reuse", "experts_dense"],
  };
}
