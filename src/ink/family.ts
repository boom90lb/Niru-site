/**
 * Figure D, a family: Monad's controls moved over ranges declared for this illustration, with the
 * parameter count and the work per token recounted from the definitions at every setting.
 *
 * Five controls move: V, L, H, K and F. Two are derived and never set: D := H·K (attention's
 * width) and P := K/2 (rope's pairs). Work is per token for one declared sequence of T tokens, the
 * forward and the forward with its derived backward; Monad's acceptance row declares no sequence
 * length or batch, and the update, whose share per token depends on a batch, is left out.
 */
import { model, units } from "./derive.ts";
import { MONAD, parameters } from "./monad.ts";

export const T = 1024;

export interface Range {
  lo: number;
  hi: number;
  step: number;
}

/** The ranges declared for the illustration, around Monad's row. */
export const RANGES: Readonly<Record<"V" | "L" | "H" | "K" | "F", Range>> = {
  V: { lo: 4096, hi: 65536, step: 1024 },
  L: { lo: 2, hi: 96, step: 1 },
  H: { lo: 1, hi: 16, step: 1 },
  K: { lo: 16, hi: 128, step: 2 },
  F: { lo: 256, hi: 4096, step: 256 },
};

export interface Setting {
  V: number;
  L: number;
  H: number;
  K: number;
  F: number;
}

export const MONAD_SETTING: Setting = {
  V: MONAD.V,
  L: MONAD.L,
  H: MONAD.H,
  K: MONAD.K,
  F: MONAD.F,
};

/** Whether a setting lies in the declared ranges and on their steps. */
export function inRange(s: Setting): boolean {
  return (Object.keys(RANGES) as Array<keyof Setting>).every((k) => {
    const r = RANGES[k];
    return s[k] >= r.lo && s[k] <= r.hi && (s[k] - r.lo) % r.step === 0;
  });
}

/** A member of the family, recounted: its derived axes, parameters and work per token. */
export function member(s: Setting) {
  if (!inRange(s)) throw new Error("outside the declared ranges");
  const D = s.H * s.K;
  const P = s.K / 2;
  const c = { ...s, D, T };
  const w = model(c);
  return {
    D,
    P,
    parameters: parameters({ V: s.V, D, L: s.L, H: s.H, K: s.K, F: s.F }),
    /** The parameters by part: the embedding, attention's and SwiGLU's weights, the norms' gains. */
    parts: {
      E: s.V * D,
      attention: s.L * 4 * D * s.H * s.K,
      swiglu: s.L * 3 * D * s.F,
      norms: (2 * s.L + 1) * D,
    },
    forward: units(w.forward) / T,
    training: (units(w.forward) + units(w.backward)) / T,
    /** The same work for the whole sequence: whole numbers of operations. */
    sequence: {
      forward: units(w.forward),
      training: units(w.forward) + units(w.backward),
    },
  };
}

/** A count in plain words, to two significant figures: "180 million", "1.2 billion". */
export function plain(n: number): string {
  const r = Number(n.toPrecision(2));
  if (r >= 1e9) return `${Number((r / 1e9).toPrecision(2))} billion`;
  if (r >= 1e6) return `${Number((r / 1e6).toPrecision(2))} million`;
  if (r >= 1e3) return `${Number((r / 1e3).toPrecision(2))} thousand`;
  return String(r);
}
