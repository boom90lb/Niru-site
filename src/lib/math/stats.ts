/**
 * The statistics behind a promotion decision: a paired bias-corrected and
 * accelerated (BCa) bootstrap interval for the mean improvement, and the
 * Benjamini–Hochberg step-up procedure across a tier.
 */

import type { Stream } from "./threefry.ts";

/** Standard normal CDF through the complementary error function below. */
export function normalCdf(x: number): number {
  return 0.5 * erfc(-x / Math.SQRT2);
}

function erfc(x: number): number {
  // Numerical Recipes erfc with Chebyshev fit; relative error < 1.2e-7.
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/** Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9 after one Newton step). */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  let x: number;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else if (p <= 1 - lo) {
    const q = p - 0.5;
    const r = q * q;
    x =
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  const e = normalCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

export const mean = (xs: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i]!;
  return s / xs.length;
};

/** Linear-interpolated quantile of an ascending-sorted sample. */
export function quantileSorted(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const h = (n - 1) * Math.min(Math.max(q, 0), 1);
  const i = Math.floor(h);
  const f = h - i;
  return sorted[i]! + (i + 1 < n ? (sorted[i + 1]! - sorted[i]!) * f : 0);
}

/** ln Γ(x) for x > 0 (Lanczos, g = 7, nine terms; relative error near 1e-15). */
function logGamma(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0]!;
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i]! / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Continued fraction for the incomplete beta function (modified Lentz). */
function betaFraction(x: number, a: number, b: number): number {
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 100_000; m++) {
    const m2 = 2 * m;
    for (const aa of [
      (m * (b - m) * x) / ((a + m2 - 1) * (a + m2)),
      (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1)),
    ]) {
      d = 1 + aa * d;
      if (Math.abs(d) < tiny) d = tiny;
      c = 1 + aa / c;
      if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      h *= d * c;
    }
    if (Math.abs(d * c - 1) <= 4 * Number.EPSILON) return h;
  }
  throw new Error("incomplete beta did not converge");
}

/** The regularized incomplete beta function I_x(a, b). */
function betaCdf(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log1p(-x),
  );
  return x < (a + 1) / (a + b + 2)
    ? (front * betaFraction(x, a, b)) / a
    : 1 - (front * betaFraction(1 - x, b, a)) / b;
}

/** The q-quantile of Beta(a, b), by bisection to the last representable bit. */
function betaQuantile(q: number, a: number, b: number): number {
  let lo = 0;
  let hi = 1;
  for (;;) {
    const mid = lo + (hi - lo) / 2;
    if (mid === lo || mid === hi) return mid;
    if (betaCdf(mid, a, b) < q) lo = mid;
    else hi = mid;
  }
}

/**
 * The bootstrap tail schedule, as AMLO certifies it (d09a43d36,
 * `src/automation/evolution/bootstrap.py`): cumulative resamples B, 4B, 16B
 * and 64B, two one-sided exact binomial bounds per look, and a total Monte
 * Carlo error of 0.01 over all eight. The tail estimate decides nothing; its
 * bounds do.
 */
export const TAIL_MONTE_CARLO_ERROR = 0.01;
export const TAIL_STAGE_MULTIPLIERS = [1, 4, 16, 64] as const;
/** The site draws its resamples from Threefry, AMLO from PCG64; the schedule is the same. */
export const TAIL_SCHEDULE_VERSION = "b-4b-16b-64b-threefry-v1";
export const TAIL_SCHEDULE_VERSIONS = [TAIL_SCHEDULE_VERSION, "b-4b-16b-64b-pcg64-v1"] as const;

/**
 * Exact binomial (Clopper–Pearson) bounds on the bootstrap tail from `count`
 * of `resamples` means at or below zero, each side spending 0.01/8.
 */
export function tailBounds(count: number, resamples: number): [number, number] {
  if (!Number.isInteger(resamples) || resamples <= 0)
    throw new Error("resamples must be a positive integer");
  if (!Number.isInteger(count) || count < 0 || count > resamples)
    throw new Error("count must be an integer in [0, resamples]");
  const error = TAIL_MONTE_CARLO_ERROR / (2 * TAIL_STAGE_MULTIPLIERS.length);
  const lower = count === 0 ? 0 : betaQuantile(error, count, resamples - count + 1);
  const upper = count === resamples ? 1 : betaQuantile(1 - error, count + 1, resamples - count);
  return [lower, upper];
}

/** What the tail schedule measured, in AMLO's terms. */
export interface TailCertificate {
  /** B, the first look. */
  initial: number;
  /** The look the schedule stopped at, 1 to 4. */
  stage: number;
  /** Resamples drawn in all, B times the stage's multiplier. */
  resamples: number;
  /** Bootstrap means at or below zero. */
  count: number;
  lower: number;
  upper: number;
}

export interface BcaResult {
  /** Mean improvement over the paired examples. */
  point: number;
  lo: number;
  hi: number;
  /** Bias correction z₀ and acceleration a. */
  z0: number;
  accel: number;
  /**
   * Fraction of bootstrap means at or below zero. It is a Monte Carlo
   * estimate and only describes; the certificate's bounds decide.
   */
  p: number;
  tail: TailCertificate;
  /** Quantiles of the bootstrap distribution, for drawing it. */
  quantiles: number[];
}

/**
 * Paired BCa bootstrap for the mean of per-example improvements
 * d_i = loss_incumbent_i − loss_candidate_i (positive means the candidate is
 * better). Resampling draws example indices, so pairs stay together.
 *
 * With a `tailThreshold` (1 − the required confidence), resampling continues
 * on the tail schedule while the exact bounds leave the threshold between
 * them, as AMLO's gate does; the interval then uses every resample drawn.
 * Without one, it stops at B.
 */
export function pairedBca(
  improvements: ArrayLike<number>,
  stream: Stream,
  {
    resamples = 2000,
    confidence = 0.95,
    quantileCount = 33,
    tailThreshold = undefined as number | undefined,
  } = {},
): BcaResult {
  const n = improvements.length;
  if (n < 2) throw new Error("paired bootstrap needs at least two examples");
  if (tailThreshold !== undefined && !(tailThreshold >= 0 && tailThreshold <= 1))
    throw new Error("tailThreshold must be in [0, 1]");
  const point = mean(improvements);

  const initial = resamples;
  let boot = new Float64Array(0);
  let tail!: TailCertificate;
  for (const [i, multiplier] of TAIL_STAGE_MULTIPLIERS.entries()) {
    const total = initial * multiplier;
    const grown = new Float64Array(total);
    grown.set(boot);
    for (let b = boot.length; b < total; b++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += improvements[stream.int(n)]!;
      grown[b] = s / n;
    }
    boot = grown;
    let count = 0;
    for (let b = 0; b < total; b++) if (boot[b]! <= 0) count++;
    const [lower, upper] = tailBounds(count, total);
    tail = { initial, stage: i + 1, resamples: total, count, lower, upper };
    if (tailThreshold === undefined || upper <= tailThreshold || lower > tailThreshold) break;
  }
  resamples = boot.length;
  boot.sort();

  let below = 0;
  let equal = 0;
  for (let b = 0; b < resamples; b++) {
    if (boot[b]! < point) below++;
    else if (boot[b]! === point) equal++;
  }
  const frac = Math.min(
    Math.max((below + equal / 2) / resamples, 1 / (2 * resamples)),
    1 - 1 / (2 * resamples),
  );
  const z0 = normalQuantile(frac);

  // Jackknife acceleration. Leave-one-out means of a mean have a closed form.
  let sum = 0;
  for (let i = 0; i < n; i++) sum += improvements[i]!;
  const jack = new Float64Array(n);
  for (let i = 0; i < n; i++) jack[i] = (sum - improvements[i]!) / (n - 1);
  const jackMean = mean(jack);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const u = jackMean - jack[i]!;
    num += u * u * u;
    den += u * u;
  }
  const accel = den > 0 ? num / (6 * den ** 1.5) : 0;

  const alpha = (1 - confidence) / 2;
  const adjust = (zq: number) => normalCdf(z0 + (z0 + zq) / (1 - accel * (z0 + zq)));
  const lo = quantileSorted(boot, adjust(normalQuantile(alpha)));
  const hi = quantileSorted(boot, adjust(normalQuantile(1 - alpha)));

  const p = tail.count / tail.resamples;

  const quantiles = Array.from({ length: quantileCount }, (_, k) =>
    quantileSorted(boot, k / (quantileCount - 1)),
  );
  return { point, lo, hi, z0, accel, p, tail, quantiles };
}

/**
 * Benjamini–Hochberg step-up at false-discovery rate q. Returns, for each
 * input p-value in input order, whether its null hypothesis is rejected.
 */
export function benjaminiHochberg(pValues: readonly number[], q = 0.05): boolean[] {
  const m = pValues.length;
  const order = pValues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p);
  let cutoff = -1;
  order.forEach(({ p }, rank) => {
    if (p <= ((rank + 1) / m) * q) cutoff = rank;
  });
  const reject = new Array<boolean>(m).fill(false);
  for (let rank = 0; rank <= cutoff; rank++) reject[order[rank]!.i] = true;
  return reject;
}
