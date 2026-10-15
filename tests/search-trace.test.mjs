import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizedFiedlerValue, laplacian, symmetricEigen } from "../src/lib/math/spectral.ts";
import * as stats from "../src/lib/math/stats.ts";
import { benjaminiHochberg, normalQuantile, pairedBca } from "../src/lib/math/stats.ts";
import { keyFromSeed, Stream, threefry2x32 } from "../src/lib/math/threefry.ts";
import { forFigures } from "../src/lib/trace/figures.ts";
import { simulate } from "../src/lib/trace/simulate.ts";
import { validateTrace } from "../src/lib/trace/validate.ts";

const committed = JSON.parse(
  await readFile(new URL("../src/data/traces/simulated.json", import.meta.url), "utf8"),
);
const figures = JSON.parse(
  await readFile(new URL("../src/data/traces/simulated.figures.json", import.meta.url), "utf8"),
);
const systemTrace = JSON.parse(
  await readFile(
    new URL("../src/data/traces/simulated-inconclusive.json", import.meta.url),
    "utf8",
  ),
);
const systemFigures = JSON.parse(
  await readFile(
    new URL("../src/data/traces/simulated-inconclusive.figures.json", import.meta.url),
    "utf8",
  ),
);
const outcomes = JSON.parse(
  await readFile(new URL("../src/data/traces/outcomes.json", import.meta.url), "utf8"),
);
const cuts = { home: figures, system: systemFigures };
const status = (t) =>
  t.events.find((e) => e.event_type === "confirmation.decided")?.payload.status;

test("Threefry-2x32 matches the Random123 known-answer vectors", () => {
  const hex = (pair) => pair.map((x) => x.toString(16).padStart(8, "0"));
  assert.deepEqual(hex(threefry2x32([0, 0], [0, 0])), ["6b200159", "99ba4efe"]);
  assert.deepEqual(hex(threefry2x32([0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff])), [
    "1cb996fc",
    "bb002be7",
  ]);
  assert.deepEqual(hex(threefry2x32([0x13198a2e, 0x03707344], [0x243f6a88, 0x85a308d3])), [
    "c4923a9c",
    "483df7a0",
  ]);
});

test("Laplacian spectra match closed forms", () => {
  const path = (n) => ({ nodes: n, edges: Array.from({ length: n - 1 }, (_, i) => [i, i + 1]) });
  const values = symmetricEigen(laplacian(path(5))).values;
  values.forEach((v, k) => assert.ok(Math.abs(v - (2 - 2 * Math.cos((k * Math.PI) / 5))) < 1e-9));
  for (const n of [4, 8, 12]) {
    const expected = 1 - Math.cos(Math.PI / (n - 1));
    assert.ok(Math.abs(normalizedFiedlerValue(path(n)) - expected) < 1e-9);
  }
});

test("the paired BCa interval brackets a known improvement and tightens with evidence", () => {
  const draw = (n, seed) => {
    const s = Stream.of(keyFromSeed(seed), "data");
    return Array.from({ length: n }, () => 0.02 + 0.05 * s.normal());
  };
  const small = pairedBca(draw(32, 1), Stream.of(keyFromSeed(1), "boot"));
  const large = pairedBca(draw(512, 1), Stream.of(keyFromSeed(1), "boot"));
  for (const r of [small, large]) assert.ok(r.lo < r.point && r.point < r.hi);
  assert.ok(large.lo < 0.02 && 0.02 < large.hi);
  assert.ok(large.hi - large.lo < (small.hi - small.lo) / 2);
  assert.ok(Math.abs(normalQuantile(0.975) - 1.959964) < 1e-6);
});

// Exact binomial bounds at 0.01/8 per side, from SciPy through AMLO's
// `bootstrap_tail_bounds` (d09a43d36), not from the site's own formula.
const SCIPY_TAIL_BOUNDS = [
  [0, 2000, 0.0, 0.003336726577215291],
  [1, 2000, 6.253907552694347e-7, 0.004483927667919362],
  [2, 2000, 2.543250688750257e-5, 0.005468674929652861],
  [5, 8000, 9.727149846507321e-5, 0.0020164513908972987],
  [0, 8000, 0.0, 0.0008352274691545352],
  [13, 32000, 0.00014781729939698573, 0.0008763893058252386],
  [0, 128000, 0.0, 5.222216549764672e-5],
  [1000, 2000, 0.4659916180563903, 0.5340083819436098],
  [1999, 2000, 0.9955160723320806, 0.9999993746092447],
  [2000, 2000, 0.9966632734227847, 1.0],
  [37, 128000, 0.0001662811490437333, 0.00046379494648559136],
  [250, 2000, 0.10362529485350853, 0.1488184292040592],
];

test("the tail bounds are AMLO's exact binomial bounds", () => {
  for (const [k, n, lower, upper] of SCIPY_TAIL_BOUNDS) {
    const [lo, hi] = stats.tailBounds(k, n);
    assert.ok(Math.abs(lo - lower) <= 1e-12 + 1e-9 * lower, `lower ${k}/${n}: ${lo} vs ${lower}`);
    assert.ok(Math.abs(hi - upper) <= 1e-12 + 1e-9 * upper, `upper ${k}/${n}: ${hi} vs ${upper}`);
  }
});

// 256 paired improvements alternating −0.8125 and +1.1875: a resample mean is
// ≤ 0 exactly when at most 104 draws are positive, so the true bootstrap tail
// is P(Binomial(256, ½) ≤ 104) = 0.0016154, above the 0.001 that 99.9% allows.
const alternating = Array.from({ length: 256 }, (_, i) => (i % 2 === 0 ? -0.8125 : 1.1875));

test("a bootstrap tail above the threshold never certifies at 99.9%", () => {
  const empirical = [];
  // Seeds 1–12 hold five of the six that passed the old rule in seeds 1–20
  // (audit-20261013-independent); each unresolved seed costs 128,000 resamples.
  for (let seed = 1; seed <= 12; seed++) {
    const fixed = pairedBca(alternating, Stream.of(keyFromSeed(seed), "boot"));
    if (fixed.lo > 0 && 1 - fixed.p >= 0.999) empirical.push(seed);
    const r = pairedBca(alternating, Stream.of(keyFromSeed(seed), "boot"), {
      tailThreshold: 0.001,
    });
    assert.equal(typeof r.tail?.upper, "number", "the result carries its tail certificate");
    assert.ok(!(r.lo > 0 && r.tail.upper <= 0.001), `seed ${seed} certified a 0.0016 tail`);
  }
  // The empirical fraction at B = 2000, as AMLO used it before d09a43d36, passes some seeds.
  assert.deepEqual(empirical, [1, 2, 6, 9, 12]);
});

test("the tail schedule extends only an unresolved tail", () => {
  const clear = Array.from({ length: 64 }, (_, i) => 0.05 + 0.01 * Math.sin(i));
  const pass = pairedBca(clear, Stream.of(keyFromSeed(3), "boot"), { tailThreshold: 0.05 });
  assert.deepEqual([pass.tail.stage, pass.tail.resamples, pass.tail.count], [1, 2000, 0]);
  // Zero tails in 2000 cannot certify 0.001; the next look, at 8000, can.
  const strict = pairedBca(clear, Stream.of(keyFromSeed(3), "boot"), { tailThreshold: 0.001 });
  assert.deepEqual([strict.tail.stage, strict.tail.resamples], [2, 8000]);
  assert.ok(strict.tail.upper <= 0.001);
  const worse = clear.map((x) => -x);
  const fail = pairedBca(worse, Stream.of(keyFromSeed(3), "boot"), { tailThreshold: 0.001 });
  assert.deepEqual([fail.tail.stage, fail.tail.resamples], [1, 2000]);
  assert.ok(fail.tail.lower > 0.001);
  assert.equal(fail.p, fail.tail.count / fail.tail.resamples);
});

test("Benjamini–Hochberg rejects by the step-up rule", () => {
  assert.deepEqual(benjaminiHochberg([0.001, 0.04, 0.03, 0.2, 0.012], 0.05), [
    true,
    true,
    true,
    false,
    true,
  ]);
  assert.deepEqual(benjaminiHochberg([0.04, 0.05], 0.05), [true, true]);
  assert.deepEqual(benjaminiHochberg([0.03, 0.2], 0.05), [false, false]);
});

test("the committed traces are reproducible from their seeds and obey their rules", () => {
  for (const t of [committed, systemTrace]) {
    const { trace } = simulate(t.source.seed, t.snapshots.length, t.snapshots[0].generation);
    assert.deepEqual(JSON.parse(JSON.stringify(trace)), t);
    assert.deepEqual(validateTrace(t), []);
  }
  // The home stage ends with a claim; the System walkthrough ends without one.
  assert.equal(status(committed), "supported");
  assert.equal(status(systemTrace), "inconclusive");
  assert.equal(outcomes.supported + outcomes.inconclusive + outcomes.refuted, outcomes.seeds);
});

test("the figures carry exactly the projection of the committed traces", () => {
  assert.deepEqual(forFigures(committed), figures);
  assert.deepEqual(forFigures(systemTrace), systemFigures);
});

test("the validator catches rule violations", () => {
  const broken = (mutate) => {
    const copy = structuredClone(committed);
    mutate(copy);
    return validateTrace(copy);
  };
  const decision = (t, test) =>
    t.events.find((e) => e.event_type === "proposal.promotion_decision" && test(e.payload));

  assert.ok(
    broken((t) => {
      const d = decision(t, (p) => p.reason.split("|").includes("CI_CROSSES_ZERO"));
      d.payload.promote = true;
      d.payload.fdr_verdict = "accepted";
    }).some((m) => m.includes("without passing the gate")),
  );
  assert.ok(
    broken((t) => {
      const e = t.events.find((x) => x.event_type === "proposal.evaluated" && x.payload.sound);
      e.payload.compute_budget_plan.steps += 7;
    }).some((m) => m.includes("floor(C/(M·B·L))")),
  );
  assert.ok(
    broken((t) => {
      const e = t.events.find((x) => x.event_type === "proposal.evaluated" && x.payload.sound);
      e.payload.compute_budget_plan.relative_compute_slack = 0.02;
    }).some((m) => m.includes("slack")),
  );
  assert.ok(
    broken((t) => {
      const s = t.events.find((x) => x.event_type === "proposal.screened" && !x.payload.selected);
      s.payload.selected = true;
    }).some((m) => m.includes("the screen kept")),
  );
  // Benjamini–Hochberg runs over every candidate tested at a tier.
  assert.ok(
    broken((t) => {
      decision(t, (p) => p.reason === "LOSS_NOT_IMPROVED").payload.fdr_verdict = "accepted";
    }).some((m) => m.includes("step-up rule")),
  );
  // The Auditor's result stays sealed: the search never acts on an audit.
  assert.ok(
    broken((t) => {
      const a = t.events.find((x) => x.event_type === "audit.recorded");
      const d = t.events.find((x) => x.event_type === "proposal.promotion_decision");
      t.events.push({
        ...structuredClone(d),
        proposal_id: a.proposal_id,
        generation: a.generation,
      });
    }).some((m) => m.includes("acted on a sealed audit")),
  );
  assert.ok(
    broken((t) => {
      t.events.find((x) => x.event_type === "audit.recorded").payload.released = true;
    }).some((m) => m.includes("left the Auditor")),
  );
  // Every receipt stays within its tolerance of C.
  assert.ok(
    broken((t) => {
      const e = t.events.find((x) => x.event_type === "proposal.evaluated" && x.payload.receipt);
      e.payload.receipt.relative_compute_error = 0.05;
      e.payload.receipt.realized_compute_flops =
        e.payload.compute_budget_plan.target_compute_flops * 1.05;
    }).some((m) => m.includes("realized compute misses C")),
  );
  // The seal opens only after the Auditor's outcome-validation receipt.
  assert.ok(
    broken((t) => {
      t.events.find((x) => x.event_type === "outcome.validated").payload.status = "unavailable";
    }).some((m) => m.includes("outcome-validation receipt")),
  );
  assert.ok(
    broken((t) => {
      delete t.events.find((x) => x.event_type === "outcome.validated").payload.path;
    }).some((m) => m.includes("no validation path")),
  );
  // The replay reruns the estimator on the sealed differences.
  assert.ok(
    broken((t) => {
      const c = t.events.find((x) => x.event_type === "confirmation.decided");
      c.payload.ci_lower -= 0.001;
    }).some((m) => m.includes("rerun on the sealed differences")),
  );
  // The claim covers only the configuration that ran.
  assert.ok(
    broken((t) => {
      t.events.find((x) => x.event_type === "claim.recorded").payload.scope.width *= 2;
    }).some((m) => m.includes("claim's scope")),
  );
  // A screened-out candidate trains only when the audit lottery draws it.
  assert.ok(
    broken((t) => {
      const s = t.events.find(
        (x) => x.event_type === "proposal.screened" && x.payload.audit && !x.payload.audit.drawn,
      );
      s.payload.audit.drawn = true;
    }).some((m) => m.includes("audit lottery")),
  );
  // The current best changes only through the Tier-3 gate.
  assert.ok(
    broken((t) => {
      decision(t, (p) => p.to_tier === "best" && p.promote).payload.promote = false;
    }).some((m) => m.includes("without passing the Tier-3 gate")),
  );
  // Each account pays only for its own purpose.
  assert.ok(
    broken((t) => {
      const e = t.events.find(
        (x) => x.event_type === "proposal.evaluated" && x.payload.tier === 2,
      );
      e.payload.account = "search";
    }).some((m) => m.includes("charged to search")),
  );
  // The sealed set opens only after the selection is frozen.
  assert.ok(
    broken((t) => {
      const i = t.events.findIndex((x) => x.event_type === "confirmation.decided");
      const [c] = t.events.splice(i, 1);
      const f = t.events.findIndex((x) => x.event_type === "selection.frozen");
      t.events.splice(f, 0, c);
    }).some((m) => m.includes("sealed set opened before")),
  );
  // The champion and the reference train under one compute target.
  assert.ok(
    broken((t) => {
      const e = t.events.find((x) => x.event_type === "target.evaluated");
      e.payload.compute_budget_plan.target_compute_flops *= 2;
    }).some((m) => m.includes("its own compute target")),
  );
});

test("the validator rejects records that do not bind their evidence", () => {
  const broken = (mutate) => {
    const copy = structuredClone(committed);
    mutate(copy);
    return validateTrace(copy);
  };
  const one = (t, type) => t.events.find((e) => e.event_type === type);
  const gated = (t) =>
    t.events.find(
      (e) => e.event_type === "proposal.promotion_decision" && e.payload.tail_count !== undefined,
    );
  // Each of these passed the validator before 2026-10-14 (audit-20261013-independent).
  const cases = {
    "frozen interval level": (t) =>
      (one(t, "selection.frozen").payload.analysis.confidence_level = 0.99),
    "p-value out of range": (t) => (one(t, "confirmation.decided").payload.p_value = -1),
    "claimed parameters": (t) => (one(t, "claim.recorded").payload.scope.params *= 100),
    "empty replay": (t) => (one(t, "claim.verified").payload.checks = []),
    "missing sealed differences": (t) => delete one(t, "confirmation.decided").payload.diffs,
    "confirmation tail count": (t) => (one(t, "confirmation.decided").payload.tail_count += 3),
    "promotion tail bound": (t) => (gated(t).payload.tail_upper_bound /= 2),
    "promotion threshold": (t) => (gated(t).payload.required_confidence = 0.5),
    "promotion FDR q": (t) => (gated(t).payload.fdr_q = 0.5),
    "claimed tokens": (t) => (one(t, "claim.recorded").payload.scope.tokens += 1),
    "reference size": (t) => (one(t, "target.planned").payload.reference.params += 1),
    // A consistent certificate whose schedule stopped with 0.001 between its bounds.
    "early stop": (t) => {
      const c = one(t, "confirmation.decided").payload;
      const [lower, upper] = stats.tailBounds(1, 2000);
      Object.assign(c, {
        tail_count: 1,
        n_resamples: 2000,
        tail_stage: 1,
        p_value: 1 / 2000,
        tail_lower_bound: lower,
        tail_upper_bound: upper,
      });
    },
  };
  const accepted = Object.entries(cases).flatMap(([name, mutate]) => {
    try {
      return broken(mutate).length > 0 ? [] : [name];
    } catch (error) {
      return [`${name} (${error.message})`];
    }
  });
  assert.deepEqual(accepted, []);
});

test("the full-scale record names both trained sizes", () => {
  const runs = committed.events.filter((e) => e.event_type === "target.evaluated");
  const sizes = Object.fromEntries(
    runs.map((r) => [r.payload.role, committed.architectures[r.candidate_arch_hash].params]),
  );
  const planned = committed.events.find((e) => e.event_type === "target.planned").payload;
  assert.equal(planned.target.params, sizes.champion);
  assert.equal(planned.reference?.params, sizes.reference);
});
