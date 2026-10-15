/**
 * Threefry-2x32 with 20 rounds (Salmon et al., "Parallel random numbers: as
 * easy as 1, 2, 3", SC 2011): the counter-based generator family that JAX
 * uses. A value depends only on its key and counter, so any draw can be
 * recomputed in isolation and in any order.
 */

const PARITY = 0x1bd11bda;

/**
 * Twenty rounds in five groups of four, with the key injected after each
 * group. The rotation constants repeat every eight rounds: 13, 15, 26, 6,
 * then 17, 29, 16, 24. Written out in full: the bootstrap draws millions of
 * values, and a loop with table lookups costs several times as much.
 */
export function threefry2x32(
  key: readonly [number, number],
  counter: readonly [number, number],
): [number, number] {
  const k0 = key[0] >>> 0;
  const k1 = key[1] >>> 0;
  const k2 = (PARITY ^ k0 ^ k1) >>> 0;
  let x0 = (counter[0] + k0) >>> 0;
  let x1 = (counter[1] + k1) >>> 0;
  // Rounds 0–3, then inject s = 1.
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 13) | (x1 >>> 19)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 15) | (x1 >>> 17)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 26) | (x1 >>> 6)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 6) | (x1 >>> 26)) ^ x0) >>> 0;
  x0 = (x0 + k1) >>> 0;
  x1 = (x1 + k2 + 1) >>> 0;
  // Rounds 4–7, then s = 2.
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 17) | (x1 >>> 15)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 29) | (x1 >>> 3)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 16) | (x1 >>> 16)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 24) | (x1 >>> 8)) ^ x0) >>> 0;
  x0 = (x0 + k2) >>> 0;
  x1 = (x1 + k0 + 2) >>> 0;
  // Rounds 8–11, then s = 3.
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 13) | (x1 >>> 19)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 15) | (x1 >>> 17)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 26) | (x1 >>> 6)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 6) | (x1 >>> 26)) ^ x0) >>> 0;
  x0 = (x0 + k0) >>> 0;
  x1 = (x1 + k1 + 3) >>> 0;
  // Rounds 12–15, then s = 4.
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 17) | (x1 >>> 15)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 29) | (x1 >>> 3)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 16) | (x1 >>> 16)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 24) | (x1 >>> 8)) ^ x0) >>> 0;
  x0 = (x0 + k1) >>> 0;
  x1 = (x1 + k2 + 4) >>> 0;
  // Rounds 16–19, then s = 5.
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 13) | (x1 >>> 19)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 15) | (x1 >>> 17)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 26) | (x1 >>> 6)) ^ x0) >>> 0;
  x0 = (x0 + x1) >>> 0;
  x1 = (((x1 << 6) | (x1 >>> 26)) ^ x0) >>> 0;
  x0 = (x0 + k2) >>> 0;
  x1 = (x1 + k0 + 5) >>> 0;
  return [x0, x1];
}

/** Derive a child key from a parent key and a label, as `jax.random.fold_in` does. */
export function foldIn(key: readonly [number, number], data: number): [number, number] {
  return threefry2x32(key, [0, data >>> 0]);
}

/** A key from a small integer seed, as `jax.random.PRNGKey(seed)` builds it. */
export function keyFromSeed(seed: number): [number, number] {
  return [Math.floor(seed / 2 ** 32) >>> 0, seed >>> 0];
}

/** Stable 32-bit label for a string, so streams can be named. */
export function label(name: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * A named stream of uniform and normal draws. Draw `i` of a stream is
 * Threefry(key, [stream, i]), so streams never overlap and reorderings of
 * unrelated code do not change any value.
 */
export class Stream {
  private i = 0;
  private spare: number | null = null;
  private readonly key: readonly [number, number];
  private readonly id: number;

  constructor(key: readonly [number, number], id: number) {
    this.key = key;
    this.id = id;
  }

  static of(key: readonly [number, number], name: string): Stream {
    return new Stream(key, label(name));
  }

  /** Uniform on [0, 1) with 32 bits of resolution. */
  uniform(): number {
    const [a] = threefry2x32(this.key, [this.id, this.i++]);
    return a / 2 ** 32;
  }

  int(n: number): number {
    return Math.floor(this.uniform() * n);
  }

  normal(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    while (u <= 1e-12) u = this.uniform();
    const v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)]!;
  }
}
