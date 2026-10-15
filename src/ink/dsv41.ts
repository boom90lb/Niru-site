/**
 * The toy DeepSeek-V4.1-Flash's configuration, as AMLO's 0.0.14 design
 * declares it, its layer schedule and its parameter
 * count, from the reference implementation's module shapes.
 */
import type { Ink } from "./engine.ts";

export const DSV41 = {
  V: 50_257,
  D: 256,
  H: 8,
  K: 64,
  P: 32,
  QL: 64,
  G: 4,
  OL: 64,
  W: 64,
  T: 512,
  IH: 4,
  IK: 64,
  TOPK: 32,
  NB: 8,
  BS: 8,
  E: 8,
  A: 2,
  F: 256,
  S: 4,
  engramHeads: 2,
  engramDim: 64,
  engramOrders: 3,
  engramRows: [49_304, 49_572],
} as const;

export type Kind = "swa" | "full2" | "reuse2" | "full1" | "reindex" | "reuse1";

export const KIND: Record<Kind, { ratio: 0 | 1 | 2; name: string; ink: Ink; folds: boolean }> = {
  swa: { ratio: 0, name: "SWA", ink: "ink", folds: false },
  full2: { ratio: 2, name: "Full", ink: "cinnabar", folds: false },
  reuse2: { ratio: 2, name: "Reuse", ink: "faint", folds: true },
  full1: { ratio: 1, name: "Full", ink: "cinnabar", folds: false },
  reindex: { ratio: 1, name: "Reindex", ink: "indigo", folds: false },
  reuse1: { ratio: 1, name: "Reuse", ink: "faint", folds: true },
};

/**
 * The root's runs as 0.0.14 writes them, expanded to one kind per layer:
 *   L0 = swa; E1 = engram; L1 = swa
 *   Enc = Repeat[(enc_group, 2), (enc_group_e, 1)]   enc_group = full; Repeat(reuse, 5)
 *   D20 = full; D21 = Repeat(reuse, 3); Dec = Repeat(dec_group, 4)   dec_group = reindex; Repeat(reuse, 3)
 */
export function schedule(): { layers: Kind[]; engram: number[] } {
  const layers: Kind[] = [];
  const engram: number[] = [];
  const group = (head: Kind, reuse: Kind, n: number) =>
    layers.push(head, ...Array<Kind>(n).fill(reuse));
  layers.push("swa");
  engram.push(layers.length);
  layers.push("swa");
  group("full2", "reuse2", 5);
  group("full2", "reuse2", 5);
  engram.push(layers.length);
  group("full2", "reuse2", 5);
  group("full1", "reuse1", 3);
  for (let i = 0; i < 4; i++) group("reindex", "reuse1", 3);
  return { layers, engram };
}

/** Parameters, counted from the reference's module shapes. The gate bias is state, not a parameter. */
export function parameters(c: typeof DSV41 = DSV41): number {
  const norm = (n: number) => n;
  const attention =
    c.H + // sink logits
    c.D * c.QL +
    norm(c.QL) +
    c.QL * c.H * c.K +
    c.D * c.K +
    norm(c.K) +
    ((c.H * c.K) / c.G) * (c.G * c.OL) +
    c.G * c.OL * c.D;
  const moe = c.E * c.D + (c.E + 1) * 3 * c.D * c.F;
  const mix = (2 + c.S) * c.S * (c.S * c.D) + (2 + c.S) * c.S + 3;
  const layer = attention + moe + 2 * norm(c.D) + 2 * mix;
  const compressor = (r: number) => c.D * c.K * (r > 1 ? 2 : 1) + norm(c.K);
  const indexer = (ownsKeys: boolean) =>
    c.QL * c.IH * c.IK + c.D * c.IH + (ownsKeys ? c.K * c.IK + norm(c.IK) : 0);
  const extra: Record<Kind, number> = {
    swa: 0,
    reuse2: 0,
    reuse1: 0,
    full2: compressor(2) + indexer(true),
    full1: compressor(1) + indexer(true),
    reindex: indexer(false),
  };
  const engram = (rows: number) =>
    rows * c.engramDim +
    c.engramOrders * c.engramHeads * c.engramDim * c.D * (c.S + 1) +
    2 * c.S * c.D;
  return (
    2 * c.V * c.D +
    norm(c.D) +
    schedule().layers.reduce((n, k) => n + layer + extra[k], 0) +
    c.engramRows.reduce((n, r) => n + engram(r), 0)
  );
}
