/**
 * Spectral graph quantities for small graphs. The cyclic Jacobi method is
 * exact enough and simple for the 4–40 node graphs that architecture cells
 * produce.
 */

export interface Graph {
  nodes: number;
  /** Undirected edges as index pairs. Duplicates and self-loops are ignored. */
  edges: ReadonlyArray<readonly [number, number]>;
}

export type Matrix = number[][];

export function adjacency(g: Graph): Matrix {
  const a: Matrix = Array.from({ length: g.nodes }, () => new Array<number>(g.nodes).fill(0));
  for (const [i, j] of g.edges) {
    if (i === j) continue;
    a[i]![j] = 1;
    a[j]![i] = 1;
  }
  return a;
}

/** L = D − A. */
export function laplacian(g: Graph): Matrix {
  const a = adjacency(g);
  return a.map((row, i) => row.map((v, j) => (i === j ? row.reduce((s, x) => s + x, 0) : -v)));
}

/** L_sym = I − D^{-1/2} A D^{-1/2}. Isolated nodes keep a zero row. */
export function normalizedLaplacian(g: Graph): Matrix {
  const a = adjacency(g);
  const d = a.map((row) => row.reduce((s, x) => s + x, 0));
  return a.map((row, i) =>
    row.map((v, j) => {
      if (i === j) return d[i]! > 0 ? 1 : 0;
      return d[i]! > 0 && d[j]! > 0 ? -v / Math.sqrt(d[i]! * d[j]!) : 0;
    }),
  );
}

export interface Eigen {
  /** Ascending eigenvalues. */
  values: number[];
  /** vectors[k] is the unit eigenvector for values[k]. */
  vectors: number[][];
}

/** Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations. */
export function symmetricEigen(m: Matrix, sweeps = 64): Eigen {
  const n = m.length;
  const a = m.map((row) => row.slice());
  const v: Matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p]![q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]![k]!;
          const aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k]![p]!;
          const vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i]![i]! - a[j]![j]!);
  return {
    values: order.map((i) => a[i]![i]!),
    vectors: order.map((i) => {
      const vec = v.map((row) => row[i]!);
      // Fix the sign so layouts do not flip between runs.
      const pivot = vec.reduce((best, x) => (Math.abs(x) > Math.abs(best) ? x : best), 0);
      return pivot < 0 ? vec.map((x) => -x) : vec;
    }),
  };
}

/** λ₂ of the combinatorial Laplacian: the algebraic connectivity. */
export function fiedlerValue(g: Graph): number {
  if (g.nodes < 2) return 0;
  return Math.max(0, symmetricEigen(laplacian(g)).values[1]!);
}

/**
 * Spectral coordinates: node i sits at (v₂[i], v₃[i], v₄[i]) of the Laplacian,
 * scaled to unit radius. Connected nodes land near each other, and the sign of
 * v₂ splits the graph along its sparsest cut.
 */
export function spectralLayout(g: Graph, dims = 3): { points: number[][]; eigen: Eigen } {
  const eigen = symmetricEigen(laplacian(g));
  const cols = Array.from({ length: dims }, (_, k) => eigen.vectors[k + 1] ?? []);
  const points = Array.from({ length: g.nodes }, (_, i) => cols.map((c) => c[i] ?? 0));
  const r = Math.max(1e-9, ...points.map((p) => Math.hypot(...p)));
  return { points: points.map((p) => p.map((x) => x / r)), eigen };
}

/**
 * λ₂ of the normalized Laplacian L = diag(deg > 0) − D^{-1/2} A D^{-1/2},
 * the form AMLO's topology features use. It lies in [0, 2].
 */
export function normalizedFiedlerValue(g: Graph): number {
  if (g.nodes < 2) return 0;
  return Math.max(0, symmetricEigen(normalizedLaplacian(g)).values[1]!);
}
