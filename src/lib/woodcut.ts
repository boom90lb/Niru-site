/**
 * A small woodcut renderer for the brand mark and plates.
 *
 * A woodcut line is the wood left standing between two knife cuts, so every
 * line here is a filled outline, not a stroke. Its width follows a profile
 * that tapers where the knife entered and left, and each edge wavers with its
 * own smooth noise, the way two separate cuts never quite agree. Tone is made
 * with hatching: parallel cuts whose width follows the form. Every value comes
 * from a seeded generator, so each build cuts the same block.
 */

export type Point = readonly [x: number, y: number];

/** Deterministic PRNG (mulberry32). */
export function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let x = Math.imul(state ^ (state >>> 15), 1 | state);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** Smooth 1D value noise in [-1, 1], with two octaves for a hand-cut edge. */
export function noise(seed: number): (x: number) => number {
  const rand = random(seed);
  const lattice = Array.from({ length: 512 }, () => rand() * 2 - 1);
  const octave = (x: number) => {
    const i = Math.floor(x);
    const f = x - i;
    const a = lattice[i & 511]!;
    const b = lattice[(i + 1) & 511]!;
    const u = f * f * (3 - 2 * f);
    return a + (b - a) * u;
  };
  return (x) => (octave(x) * 2 + octave(x * 2.7 + 17.3)) / 3;
}

const smoothstep = (edge0: number, edge1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - edge0) / Math.max(1e-9, edge1 - edge0)));
  return t * t * (3 - 2 * t);
};

/** Width profile: knife entry and exit tapers, with an optional swell at the middle. */
export function profile(base: number, entry = 0.18, exit = 0.18, swell = 0) {
  return (t: number) =>
    base *
    (entry > 0 ? smoothstep(0, entry, t) : 1) *
    (exit > 0 ? smoothstep(0, exit, 1 - t) : 1) *
    (1 + swell * Math.sin(Math.PI * t));
}

/** Points along a quadratic or cubic Bezier, or a straight line. */
export function bezier(points: readonly Point[], samples = 24): Point[] {
  const at = (t: number): Point => {
    if (points.length === 2) {
      const [a, b] = points as [Point, Point];
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    }
    if (points.length === 3) {
      const [a, b, c] = points as [Point, Point, Point];
      const u = 1 - t;
      return [
        u * u * a[0] + 2 * u * t * b[0] + t * t * c[0],
        u * u * a[1] + 2 * u * t * b[1] + t * t * c[1],
      ];
    }
    const [a, b, c, d] = points as [Point, Point, Point, Point];
    const u = 1 - t;
    return [
      u * u * u * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t * t * t * d[0],
      u * u * u * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t * t * t * d[1],
    ];
  };
  return Array.from({ length: samples + 1 }, (_, k) => at(k / samples));
}

function resample(line: readonly Point[], spacing: number): { points: Point[]; length: number } {
  const lengths = [0];
  for (let k = 1; k < line.length; k += 1) {
    const [x0, y0] = line[k - 1]!;
    const [x1, y1] = line[k]!;
    lengths.push(lengths[k - 1]! + Math.hypot(x1 - x0, y1 - y0));
  }
  const length = lengths[lengths.length - 1]!;
  const count = Math.max(2, Math.ceil(length / spacing) + 1);
  const points: Point[] = [];
  let segment = 1;
  for (let k = 0; k < count; k += 1) {
    const s = (k / (count - 1)) * length;
    while (segment < line.length - 1 && lengths[segment]! < s) segment += 1;
    const s0 = lengths[segment - 1]!;
    const s1 = lengths[segment]!;
    const f = s1 > s0 ? (s - s0) / (s1 - s0) : 0;
    const [x0, y0] = line[segment - 1]!;
    const [x1, y1] = line[segment]!;
    points.push([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]);
  }
  return { points, length };
}

export interface CarveOptions {
  /** Width at parameter t in [0, 1] along the line. */
  width: (t: number) => number;
  seed: number;
  /** Edge waver as a fraction of the local half-width. */
  rough?: number;
  /** Waver frequency, in cycles per unit of length. */
  grain?: number;
  /** Sample spacing along the line. */
  spacing?: number;
}

/** The outline of one carved line, as closed polygon points. */
export function carve(line: readonly Point[], options: CarveOptions): Point[] {
  const { width, seed, rough = 0.22, grain = 0.35, spacing = 0.5 } = options;
  const { points, length } = resample(line, spacing);
  const left = noise(seed);
  const right = noise(seed + 101);
  const leftEdge: Point[] = [];
  const rightEdge: Point[] = [];
  points.forEach((p, k) => {
    const prev = points[Math.max(0, k - 1)]!;
    const next = points[Math.min(points.length - 1, k + 1)]!;
    let tx = next[0] - prev[0];
    let ty = next[1] - prev[1];
    const norm = Math.hypot(tx, ty) || 1;
    tx /= norm;
    ty /= norm;
    const t = k / (points.length - 1);
    const s = t * length * grain;
    const half = width(t) / 2;
    const l = half * (1 + rough * left(s));
    const r = half * (1 + rough * right(s));
    leftEdge.push([p[0] - ty * l, p[1] + tx * l]);
    rightEdge.push([p[0] + ty * r, p[1] - tx * r]);
  });
  return [...leftEdge, ...rightEdge.reverse()];
}

/** Hatching: parallel cuts across a region, each one carved. */
export interface HatchOptions {
  /** Start and end of the hatch band, measured along `across`. */
  from: number;
  to: number;
  count: number;
  /** Maps a position in [0, 1] across the band to the cut's centerline. */
  line: (u: number) => Point[];
  /** Cut width at position u across the band and t along the cut. */
  width: (u: number, t: number) => number;
  seed: number;
  rough?: number;
}

export function hatch(options: HatchOptions): Point[][] {
  const { count, line, width, seed, rough = 0.3 } = options;
  const jitter = random(seed);
  return Array.from({ length: count }, (_, k) => {
    const u = (k + 0.5 + (jitter() - 0.5) * 0.35) / count;
    return carve(line(u), {
      width: (t) => width(u, t),
      seed: seed + k * 7,
      rough,
      grain: 0.6,
    });
  });
}

/**
 * Path data for polygons, compact: each point is rounded to `precision`
 * decimals in absolute terms, then written as a relative step from the last
 * rounded point, so rounding never accumulates. Leading zeros are dropped
 * and a minus sign doubles as a separator.
 */
export function pathData(polygons: readonly (readonly Point[])[], precision = 1): string {
  const f = 10 ** precision;
  const q = (v: number) => Math.round(v * f);
  const num = (k: number) => {
    const v = k / f;
    if (v === 0) return "0";
    return String(v).replace(/^(-?)0\./, "$1.");
  };
  const join = (values: number[]) =>
    values.map((k, i) => (i > 0 && k >= 0 ? " " : "") + num(k)).join("");
  return polygons
    .filter((polygon) => polygon.length > 2)
    .map((polygon) => {
      let x = q(polygon[0]![0]);
      let y = q(polygon[0]![1]);
      const steps: number[] = [];
      for (let k = 1; k < polygon.length; k += 1) {
        const nx = q(polygon[k]![0]);
        const ny = q(polygon[k]![1]);
        if (nx === x && ny === y) continue;
        steps.push(nx - x, ny - y);
        x = nx;
        y = ny;
      }
      const start = `M${join([q(polygon[0]![0]), q(polygon[0]![1])])}`;
      return steps.length > 0 ? `${start}l${join(steps)}z` : `${start}z`;
    })
    .join("");
}

/** Transform helpers for composing plates. */
export const rotate = (p: Point, radians: number, origin: Point = [0, 0]): Point => {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const x = p[0] - origin[0];
  const y = p[1] - origin[1];
  return [origin[0] + x * c - y * s, origin[1] + x * s + y * c];
};
