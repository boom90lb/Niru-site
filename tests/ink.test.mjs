import assert from "node:assert/strict";
import test from "node:test";

import { KIND, parameters as dsParameters, schedule } from "../src/ink/dsv41.ts";
import { sinkhorn } from "../src/ink/lattice.ts";
import { FIGURES, loadAll, MODELS, view } from "../src/ink/models.ts";
import { parameters } from "../src/ink/monad.ts";
import * as mv from "../src/ink/moves.ts";
import { ILLUSTRATION } from "../src/ink/rewrite.ts";

await loadAll();

// AMLO's 0.0.14 design gives Monad's parameters as counted from its definitions.
test("the drawn definitions count the parameters AMLO's acceptance table gives", () => {
  assert.equal(parameters(), 56_656_128); // monad
  assert.equal(parameters({ V: 257, D: 64, L: 2, H: 4, K: 16, F: 192 }), 123_264); // gpt-cpu
  assert.equal(parameters({ V: 50_257, D: 512, L: 8, H: 8, K: 64, F: 1536 }), 53_003_264); // gpt-gpu
});

// The toy DeepSeek has 112,726,448 parameters, counted from the reference's module shapes.
test("the toy DeepSeek counts 112,726,448 parameters", () => {
  assert.equal(dsParameters(), 112_726_448);
});

// The reference's config arrays, read from config.json at revision 2cba9e42.
test("the schedule drawn from the 0.0.14 root equals the reference's layer arrays", () => {
  const { layers, engram } = schedule();
  const at = (pred) => layers.flatMap((k, i) => (pred(k) ? [i] : []));
  assert.equal(layers.length, 40);
  assert.deepEqual(
    layers.map((k) => KIND[k].ratio),
    [0, 0, ...Array(18).fill(2), ...Array(20).fill(1)],
  );
  assert.deepEqual(
    at((k) => k === "full2" || k === "full1"),
    [2, 8, 14, 20],
  );
  assert.deepEqual(
    at((k) => k === "full2" || k === "full1" || k === "reindex"),
    [2, 8, 14, 20, 24, 28, 32, 36],
  );
  assert.deepEqual(
    at((k) => k === "full1"),
    [20],
  );
  assert.deepEqual(engram, [1, 14]);
});

/** Every view reachable from either model's training step, and every step of every figure. */
function reachable() {
  const seen = new Map();
  const queue = [...MODELS.flatMap((m) => m.path), ...FIGURES.flatMap((f) => f.steps)];
  while (queue.length) {
    const id = queue.shift();
    if (seen.has(id)) continue;
    const v = view(id);
    seen.set(id, v);
    for (const o of v.sketch.openables) queue.push(o.child);
    if (v.before) queue.push(v.before);
  }
  return seen;
}

test("every box that opens leads to a view, and each model's path is a chain of boxes", () => {
  const views = reachable();
  assert.ok(views.size >= 40, `${views.size} views`);
  for (const [id, v] of views) {
    assert.equal(v.id, id);
    for (const o of v.sketch.openables) assert.ok(o.face.length >= 3, `${id}: ${o.id}`);
  }
  for (const m of MODELS)
    m.path.slice(1).forEach((id, i) =>
      assert.ok(
        view(m.path[i]).sketch.openables.some((o) => o.child === id),
        `${m.id}: ${m.path[i]} does not open ${id}`,
      ),
    );
});

test("a view draws the same picture every time it is built", () => {
  for (const id of [
    "model",
    "attn",
    "ds/model",
    "ds/attn/full1",
    "ds/moe",
    "lat/sinkhorn",
    "rw/delete",
    "rw/delete/before",
    "rw/deepseek",
  ])
    assert.deepEqual(view(id).sketch.prims, view(id).sketch.prims);
});

test("Sinkhorn, run as the reference runs it, makes rows and columns sum to 1", () => {
  const m = sinkhorn([
    [2.0, 0.4, -0.6, 0.1],
    [0.3, 1.6, 0.2, -0.8],
    [-0.4, 0.5, 1.2, 0.9],
    [0.6, -0.2, 0.3, 1.8],
  ]);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(m[i].reduce((a, b) => a + b, 0) - 1) < 1e-4);
    assert.ok(Math.abs(m.reduce((a, row) => a + row[i], 0) - 1) < 1e-4);
  }
});

// The site loads Source Serif's Latin, Latin Extended and Greek subsets only; a character
// outside them would be set in a fallback font.
test("every label and caption uses characters the site's fonts carry", () => {
  const ranges = [
    [0x0000, 0x00ff],
    [0x0131, 0x0131],
    [0x0152, 0x0153],
    [0x02bb, 0x02bc],
    [0x02c6, 0x02c6],
    [0x02da, 0x02da],
    [0x02dc, 0x02dc],
    [0x2000, 0x206f],
    [0x20ac, 0x20ac],
    [0x2122, 0x2122],
    [0x2191, 0x2191],
    [0x2193, 0x2193],
    [0x2212, 0x2212],
    [0x2215, 0x2215],
    [0x0100, 0x02ba],
    [0x1e00, 0x1e9f],
    [0x0370, 0x03ff],
  ];
  const covered = (ch) =>
    ranges.some(([a, b]) => ch.codePointAt(0) >= a && ch.codePointAt(0) <= b);
  for (const [id, v] of reachable()) {
    const texts = [v.title, v.caption];
    for (const p of v.sketch.prims)
      if (p.kind === "label") texts.push(p.text, p.sub ?? "", p.sup ?? "");
    for (const t of texts) for (const ch of t) assert.ok(covered(ch), `${id}: "${ch}" in "${t}"`);
  }
});

// ------------------------------------------------------------------------------------- figure B
// The moves of AMLO's 0.0.14 search on Monad's term. Each fraction is derived by hand in the
// comment beside it.

const show = (x) => mv.show(x);

test("a figure's steps open from a view that offers them, and each rewrite has a before", () => {
  for (const f of FIGURES) {
    assert.ok(reachable().has(f.from), `${f.id} opens from ${f.from}`);
    for (const id of f.steps) {
      const v = view(id);
      if (!v.before) continue;
      const b = view(v.before);
      assert.equal(b.yaw, v.yaw);
      assert.equal(b.pitch, v.pitch);
      const keys = new Set(b.sketch.prims.flatMap((p) => (p.key ? [p.key] : [])));
      assert.ok(
        v.sketch.prims.some((p) => p.key && keys.has(p.key)),
        `${id}: nothing to carry from ${v.before}`,
      );
    }
  }
});

test("reach: all, one and some, as the search draws them", () => {
  assert.equal(show(mv.reach(64, "all")), "2/5");
  assert.equal(show(mv.reach(64, "one")), "1/160"); // 2/5 · 1/64
  assert.equal(show(mv.reach(64, "some", 2)), "1/624,960"); // 1/5 · 1/62 · 1/C(64, 2) = 1/2,016
  assert.equal(show(mv.reach(64, "some", 3)), "1/12,915,840"); // 1/5 · 1/62 · 1/41,664
  assert.equal(show(mv.reach(2, "all")), "1/2");
  assert.equal(show(mv.reach(2, "one")), "1/4"); // 1/2 · 1/2
  assert.equal(show(mv.reach(1, "all")), "1");
  // Over every reach and every occurrence or subset it can select, the draws sum to 1.
  for (const n of [1, 2, 3, 5, 8, 64]) {
    let total = mv.reach(n, "all");
    if (n >= 2) total = mv.add(total, mv.mul(mv.reach(n, "one"), mv.q(n)));
    for (let k = 2; k <= n - 1; k++)
      total = mv.add(total, mv.mul(mv.reach(n, "some", k), mv.q(mv.choose(n, k))));
    assert.equal(show(total), "1", `n = ${n}`);
  }
});

test("the tournament factor C(P − r, S − 1) / C(P, S) at P 24, S 4", () => {
  assert.equal(show(mv.tournament(24, 4, 1)), "1/6"); // C(23, 3) / C(24, 4) = 1,771/10,626
  assert.equal(show(mv.tournament(24, 4, 2)), "10/69"); // 1,540/10,626
  assert.equal(show(mv.tournament(24, 4, 3)), "95/759"); // 1,330/10,626
  let total = mv.q(0);
  for (let r = 1; r <= 24; r++) total = mv.add(total, mv.tournament(24, 4, r));
  assert.equal(show(total), "1");
});

test("class weights renormalize over the classes with a legal site", () => {
  // Declared 1/4, 1/4, 1/10, 1/10, 1/5 sum to 9/10 (crossover off); each is divided by 9/10.
  const five = mv.weights({ replace: 1, insert: 1, delete: 1, binding: 1, shape: 1 });
  assert.deepEqual(Object.values(five).map(show), ["5/18", "5/18", "1/9", "1/9", "2/9"]);
  // Without shape they sum to 7/10.
  const four = mv.weights({ replace: 1, insert: 1, delete: 1, binding: 1, shape: 0 });
  assert.deepEqual(Object.values(four).map(show), ["5/14", "5/14", "1/7", "1/7", "0"]);
});

test("Monad's legal sites, uses and parameters, counted on the drawn definitions", () => {
  // insert, a real-valued output: gpt 6 (rope table, gather, Repeat, norm, head, loss),
  // block 6, attn 11, swiglu 5. delete, input and output types equal: gpt 2 (Repeat, norm),
  // block 6, attn 5 (two ropes, scale, mask, softmax), swiglu 2 (silu, product). binding: the
  // Repeat's 9 stacked slots and the tied #E. shape: F, and f where block and swiglu bind it.
  assert.deepEqual(mv.counts(mv.SEED), {
    replace: 1,
    insert: 28,
    delete: 15,
    binding: 10,
    shape: 3,
  });
  const uses = ["gpt", "block", "attn", "swiglu", "rmsnorm"].map((d) =>
    mv.occurrences(mv.SEED, d),
  );
  assert.deepEqual(uses, [1, 64, 64, 64, 129]); // rmsnorm: 2 per layer and the final norm
  assert.equal(mv.parameters(mv.SEED), parameters());
});

test("one move of each class: its factors, its fraction and its child", () => {
  const got = Object.fromEntries(mv.moves().map((m) => [m.id, m]));
  const fr = (id) => got[id].factors.map((f) => show(f.q));
  // replace: 1/6 · 5/18 · 1/1 · 1/6 · 2/5 = 1/324
  assert.deepEqual(fr("replace"), ["1/6", "5/18", "1", "1/6", "2/5"]);
  assert.equal(show(got.replace.probability), "1/324");
  // insert at one layer: 1/6 · 5/18 · 1/28 · 1/10 · (2/5 · 1/64) = 1/967,680
  assert.deepEqual(fr("insert"), ["1/6", "5/18", "1/28", "1/10", "1/160"]);
  assert.equal(show(got.insert.probability), "1/967,680");
  // delete at some 3 layers: 1/6 · 1/9 · 1/15 · 1 · 1/12,915,840 = 1/10,461,830,400
  assert.deepEqual(fr("delete"), ["1/6", "1/9", "1/15", "1", "1/12,915,840"]);
  assert.equal(show(got.delete.probability), "1/10,461,830,400");
  // binding, the Repeat occurring once: 1/6 · 1/9 · 1/10 · 1 · 1 = 1/540
  assert.deepEqual(fr("binding"), ["1/6", "1/9", "1/10", "1", "1"]);
  assert.equal(show(got.binding.probability), "1/540");
  // shape at some 2 layers: 1/6 · 2/9 · 1/3 · 1/512 · 1/624,960 = 1/25,918,341,120
  assert.deepEqual(fr("shape"), ["1/6", "2/9", "1/3", "1/512", "1/624,960"]);
  assert.equal(show(got.shape.probability), "1/25,918,341,120");
  for (const m of Object.values(got))
    assert.ok(mv.eq(m.probability, mv.mul(...m.factors.map((f) => f.q))));
  // Parameters: replace leaves them; delete drops 3 norms of 256; sharing #W2 [768, 256] keeps
  // one of 64 slices; f = 1,024 at 2 layers adds 2 · 3 · 256 · 256.
  assert.equal(mv.parameters(got.replace.child), 56_656_128);
  assert.equal(mv.parameters(got.delete.child), 56_656_128 - 3 * 256);
  assert.equal(mv.parameters(got.binding.child), 56_656_128 - 63 * 768 * 256);
  assert.equal(mv.parameters(got.shape.child), 56_656_128 + 2 * 3 * 256 * 256);
  // A subset edit gives its layers one copy of the block; selecting every layer copies nothing.
  assert.equal(Object.keys(got.delete.child.defs).length, Object.keys(mv.SEED.defs).length + 1);
  assert.equal(Object.keys(got.replace.child.defs).length, Object.keys(mv.SEED.defs).length);
});

test("two moves that commute reach one architecture, which identity dedup refuses", () => {
  const { a, b, fromA, fromB } = mv.commuting();
  // From A, ranked first: 1/6 · 5/18 · 1/28 · 1/10 · 1/160 = 1/967,680.
  assert.equal(show(fromA.probability), "1/967,680");
  // From B, ranked second: 10/69 · 5/18 · 1/1 · 1/6 · 2/5 = 10/3,726 = 5/1,863.
  assert.equal(show(fromB.probability), "5/1,863");
  const id = mv.identity;
  assert.equal(id(fromA.child), id(fromB.child));
  for (const t of [mv.SEED, a, b]) assert.notEqual(id(fromA.child), id(t));
  assert.notEqual(id(a), id(b));
});

test("the DeepSeek move: one layer of a Reuse run inside the decoder's nested Repeat", () => {
  const { layers } = schedule();
  const { first, groups, size } = mv.DEC;
  assert.deepEqual(
    layers.slice(first),
    Array.from({ length: groups }).flatMap(() => ["reindex", ...Array(size - 1).fill("reuse1")]),
  );
  const d = mv.deepseek(26);
  assert.equal(d.group, 0);
  assert.equal(d.inner, 1);
  // 1/6 · 5/14 (no range, so four classes) · 1/6 · 1/6 · (2/5 · 1/40) = 1/60,480
  assert.deepEqual(
    d.factors.map((f) => show(f.q)),
    ["1/6", "5/14", "1/6", "1/6", "1/100"],
  );
  assert.equal(show(d.probability), "1/60,480");
});

test("each move in plain words: one sentence, and its odds to two figures", () => {
  // The film says each probability as a reader who is not a researcher would: "1 in N" when N
  // is a whole number under 1,000, otherwise "about 1 in N" to two significant figures.
  assert.equal(mv.odds(mv.q(1, 324)), "1 in 324");
  assert.equal(mv.odds(mv.q(1, 967_680)), "about 1 in 970,000");
  assert.equal(mv.odds(mv.q(1, 10_461_830_400)), "about 1 in 10 billion");
  assert.equal(mv.odds(mv.q(1, 25_918_341_120)), "about 1 in 26 billion");
  assert.equal(mv.odds(mv.q(1, 12_915_840)), "about 1 in 13 million");
  assert.equal(mv.odds(mv.q(1, 60_480)), "about 1 in 60,000");
  assert.equal(mv.odds(mv.q(5, 1_863)), "about 1 in 370"); // 372.6
  assert.equal(mv.odds(mv.q(5, 18)), "about 1 in 3.6");
  assert.equal(mv.odds(mv.q(1)), "certain");
  assert.equal(mv.chance(mv.q(5, 18)), "5 in 18");
  assert.equal(mv.chance(mv.q(1)), "always");
  const got = Object.fromEntries(mv.moves().map((m) => [m.id, m]));
  const plain = (m) => [m.say, mv.odds(m.probability), m.factors.map((f) => mv.chance(f.q))];
  assert.deepEqual(plain(got.replace), [
    "Replace silu with gelu_tanh, in every layer.",
    "1 in 324",
    ["1 in 6", "5 in 18", "always", "1 in 6", "2 in 5"],
  ]);
  assert.deepEqual(plain(got.insert), [
    "Insert a norm after attention, in one layer.",
    "about 1 in 970,000",
    ["1 in 6", "5 in 18", "1 in 28", "1 in 10", "1 in 160"],
  ]);
  assert.deepEqual(plain(got.delete), [
    "Delete the norm before attention, in three layers.",
    "about 1 in 10 billion",
    ["1 in 6", "1 in 9", "1 in 15", "always", "1 in 12,915,840"],
  ]);
  assert.deepEqual(plain(got.binding), [
    "Share one #W2 across all 64 layers.",
    "1 in 540",
    ["1 in 6", "1 in 9", "1 in 10", "always", "always"],
  ]);
  assert.deepEqual(plain(got.shape), [
    "Widen f to 1,024, in two layers.",
    "about 1 in 26 billion",
    ["1 in 6", "2 in 9", "1 in 3", "1 in 512", "1 in 624,960"],
  ]);
  assert.deepEqual(
    got.insert.factors.map((f) => f.say),
    [
      "The parent is the seed, ranked first.",
      "The class drawn is insert.",
      "The place is attention’s output, 1 of 28.",
      "The new node is rms_norm, 1 of 10 that fit there.",
      "The change reaches one layer of 64: layer 12.",
    ],
  );
  assert.equal(got.delete.factors[4].say, "The change reaches 3 layers of 64: 7, 30 and 51.");
  assert.equal(got.binding.factors[4].say, "The Repeat is used once, so all 64 layers change.");
  const { fromA, fromB } = mv.commuting();
  assert.equal(mv.odds(fromA.probability), "about 1 in 970,000");
  assert.equal(mv.odds(fromB.probability), "about 1 in 370");
  const d = mv.deepseek(26);
  assert.equal(d.say, "Replace silu with gelu_tanh in the routed experts, in layer 26 only.");
  assert.equal(mv.odds(d.probability), "about 1 in 60,000");
});

// ------------------------------------------------------------------------------------- the film
// Figure B plays as a film: one pure function from time to a frame.

test("the film: ten chapters, about two minutes, each beat drawing a view that exists", async () => {
  await loadAll();
  const { film, chapterAt, chapterEnd } = await import("../src/ink/film.ts");
  const { chapters, starts, total } = film();
  assert.deepEqual(
    chapters.map((c) => c.id),
    [
      "population",
      "sites",
      "replace",
      "insert",
      "delete",
      "binding",
      "shape",
      "commute",
      "admission",
      "deepseek",
    ],
  );
  assert.ok(total > 100_000 && total < 150_000, `${total} ms`);
  for (const c of chapters) for (const b of c.beats) assert.ok(view(b.to), b.to);
  // A chapter's end belongs to it, so a pause there shows its last frame.
  chapters.forEach((_, i) => {
    assert.equal(chapterAt(chapterEnd(i)), i);
    if (i + 1 < chapters.length) assert.equal(chapterAt(chapterEnd(i) + 1), i + 1);
    assert.equal(chapterAt(starts[i] + 1), i);
  });
  assert.equal(chapters[9].model, "dsv41");
});

test("a film frame is a pure function of its time, and every frame says what it is", async () => {
  await loadAll();
  const { film, frameAt } = await import("../src/ink/film.ts");
  const { total } = film();
  for (let t = 0; t <= total; t += 997) {
    const a = frameAt(t);
    assert.equal(JSON.stringify(frameAt(t).prims), JSON.stringify(a.prims), `t = ${t}`);
    assert.ok(
      a.prims.some((p) => p.kind === "label" && p.text === ILLUSTRATION),
      `t = ${t}: no illustration line`,
    );
  }
});

test("each move chapter says its move plainly, with its odds, and its ledger exactly", async () => {
  await loadAll();
  const { film, frameAt, chapterEnd } = await import("../src/ink/film.ts");
  const { chapters, starts } = film();
  const texts = (prims) => prims.filter((p) => p.kind === "label").map((p) => p.text);
  for (const m of mv.moves()) {
    const ci = chapters.findIndex((c) => c.id === m.id);
    const c = chapters[ci];
    // The rewrite beat says the move; the five draws say their factors, in order.
    assert.ok(
      c.beats.some((b) => b.say === m.say),
      m.id,
    );
    assert.deepEqual(
      c.beats.slice(0, 5).map((b) => b.say),
      m.factors.map((f) => f.say),
    );
    // By the chapter's end the band shows the draws as chances and the odds in plain words.
    const end = texts(frameAt(chapterEnd(ci)).prims);
    for (const f of m.factors)
      assert.ok(end.includes(mv.chance(f.q)), `${m.id}: ${mv.chance(f.q)}`);
    assert.ok(end.includes(mv.odds(m.probability)), `${m.id}: ${mv.odds(m.probability)}`);
    assert.ok(
      !end.includes(mv.show(m.probability)) || mv.show(m.probability) === mv.odds(m.probability),
    );
    // Once the reach is drawn, the odds stay through every later beat of the chapter.
    let t = starts[ci];
    for (const [i, b] of c.beats.entries()) {
      if (i > 4)
        for (const f of [0.02, 0.5])
          assert.ok(
            frameAt(t + b.ms * f).prims.some(
              (p) =>
                p.kind === "label" && p.text === mv.odds(m.probability) && (p.alpha ?? 1) > 0.99,
            ),
            `${m.id}, beat ${i + 1}`,
          );
      t += b.ms;
    }
    // With details, its ledger gives the exact fraction.
    const ledger = texts(frameAt(chapterEnd(ci), { details: true }).prims);
    assert.ok(ledger.includes(mv.show(m.probability)), `${m.id}: ${mv.show(m.probability)}`);
  }
});

// Every effect names what it acts on; a key that matches nothing would light nothing, silently.
test("every keyed effect in each film acts on something its beat draws", async () => {
  await loadAll();
  const { matches } = await import("../src/ink/reel.ts");
  for (const mod of ["film.ts", "functors.ts"]) {
    const { film, frameAt } = await import(`../src/ink/${mod}`);
    const { chapters, starts } = film();
    chapters.forEach((c, ci) => {
      let t = starts[ci];
      c.beats.forEach((b, bi) => {
        const f = frameAt(t + b.ms / 2);
        const drawn = [...f.from.sketch.prims, ...f.to.sketch.prims];
        for (const fx of b.fx ?? [])
          for (const k of fx.keys ?? [])
            assert.ok(
              drawn.some((p) => matches(p, [k])),
              `${mod} ${c.id} beat ${bi + 1}: nothing drawn under ${k}`,
            );
        t += b.ms;
      });
    });
  }
});

// ----------------------------------------------------------------------------------- figure C
// One term, many functors: the derived backward, its cost fold, the plan and the identities of
// Monad's block. Work follows AMLO's work rules as src/ink/derive.ts models them; each figure is
// derived by hand beside it.

const dv = await import("../src/ink/derive.ts");
const pl = await import("../src/ink/plan.ts");
const fam = await import("../src/ink/family.ts");

test("a contraction costs terms·(k − 1) + (terms − cells), and its backward is two contractions", () => {
  const g = new dv.Program();
  g.value("x", ["t", "d"], "input");
  g.value("W", ["d", "n"], "parameter");
  g.contract("f", "y", ["x", "W"], ["t", "n"]);
  const e = { t: 3, d: 4, n: 5 };
  // Each of the 15 outputs: 4 products and 3 sums.
  assert.equal(dv.units(g.forwardWork(e)), 15 * 7);
  // Toward x, contract(ȳ[t,n], W[d,n]) → [t,d]: 12 cells of 5 products and 4 sums (108);
  // toward W, contract(x[t,d], ȳ[t,n]) → [d,n]: 20 cells of 3 products and 2 sums (100).
  const steps = g.backward(["y"], ["x"], e);
  assert.deepEqual(
    steps.map((s) => [s.of, dv.units(s.work)]),
    [["y", 108 + 100]],
  );
  assert.deepEqual(steps[0].reads, ["W", "x"]); // toward x it reads W; toward W, x
});

test("elementwise recipes: SiLU's forward and its transpose, counted from its scalar program", () => {
  // silu(x) = x · 1/(1 + exp(−x)): negate, exp, add, divide, multiply.
  const g = new dv.Program();
  g.value("a", ["n"], "input");
  g.ew("f", "y", "silu", ["a"]);
  assert.deepEqual(g.forwardWork({ n: 1 }), { arith: 4, transcendental: 1, compare_select: 0 });
  // Reverse: the product's two transposes (2), the reciprocal's (3: two products and a negation),
  // exp's product (1), the negation's (1), and one add where x's two cotangents meet: 8.
  const [s] = g.backward(["y"], ["a"], { n: 1 });
  assert.deepEqual(s.work, { arith: 8, transcendental: 0, compare_select: 0 });
});

test("RMSNorm over [t, n]: its forward and backward by hand", () => {
  const g = new dv.Program();
  g.value("x", ["t", "n"], "input");
  g.value("g", ["n"], "parameter");
  // sq = x², ss = Σn sq, ms = ss/n, ve = ms + ε, r = rsqrt ve, xr = x·r, y = xr·g
  const sq = g.ew("n", "sq", "square", ["x"]);
  const ss = g.reduce("n", "sum", "ss", sq, ["t"]);
  const ms = g.ew("n", "ms", "scale", [ss]);
  const ve = g.ew("n", "ve", "shift", [ms]);
  const r = g.ew("n", "r", "rsqrt", [ve]);
  const xr = g.ew("n", "xr", "mul", ["x", r], ["t", "n"]);
  g.ew("n", "y", "mul", [xr, "g"], ["t", "n"]);
  const t = 2;
  const n = 3;
  const e = { t, n };
  // Forward: tn + t(n − 1) + t + t + tn + tn arithmetic, t rsqrt.
  assert.deepEqual(g.forwardWork(e), {
    arith: 3 * t * n + t * (n - 1) + 2 * t,
    transcendental: t,
    compare_select: 0,
  });
  // Backward: y toward xr and g, 2tn, plus g's cotangent summed onto [n], tn − n; xr toward x and
  // r, 2tn, plus r's summed onto [t], tn − t; rsqrt 4 per row; the scale 1 per row; the square
  // 3 per element (two products, one add where they meet); x read twice, tn adds.
  const back = g.backward(["y"], ["x"], e).reduce((s, x) => s + dv.units(x.work), 0);
  assert.equal(back, 3 * t * n - n + (3 * t * n - t) + 4 * t + t + 3 * t * n + t * n);
});

test("Monad's block at T = 1,024: contractions by formula, totals pinned", () => {
  const c = { V: 8192, D: 256, L: 64, H: 4, K: 64, F: 768, T: 1024 };
  const { program: g, forward, backward, steps } = dv.blockWork(c);
  const e = dv.extentsOf(c);
  const { T: t, D: d, H: h, K: k, F: f } = c;
  // The nine contractions, each cells·(2·summed − 1).
  const contractions =
    3 * t * h * k * (2 * d - 1) + // q0, k0, v
    t * t * h * (2 * k - 1) + // s = q·k over k
    t * h * k * (2 * t - 1) + // o = p·v over t′
    t * d * (2 * h * k - 1) + // a = o·Wo over h, k
    2 * t * f * (2 * d - 1) + // m.a, m.b
    t * d * (2 * f - 1); // m = h·W2 over f
  const counted = g.nodes
    .filter((n) => n.prim.kind === "contract")
    .reduce((s, n) => s + dv.units(g.forward(n, e)), 0);
  assert.equal(counted, contractions);
  assert.equal(dv.units(forward), 2_849_501_184);
  // Softmax as AMLO's builder composes it: the divide toward both operands, 3 per element with the
  // shared quotient, and the row sum's cotangent reduced, all − rows; exp's product, all; the row
  // max is a stop-gradient value; exp is read twice, so its cotangents add, all: 6·all − rows.
  const all = t * t * h;
  const softmax = steps
    .filter((s) => ["sm.max", "sm.shifted", "sm.exp", "sm.sum", "pr"].includes(s.of))
    .reduce((n, s) => n + dv.units(s.work), 0);
  assert.equal(softmax, 6 * all - t * h);
  assert.equal(dv.units(backward), 5_678_763_520);
  // Every forward contraction is pulled back by exactly two (one per operand).
  const pulled = steps.filter(
    (s) => g.nodes.find((n) => n.result === s.of)?.prim.kind === "contract",
  );
  assert.equal(pulled.length, 9);
  // Cotangents accumulate where a value is read twice or more: the stream into each half and the
  // norm outputs (three reads each: two in the norm, one by the add; or queries, keys and values),
  // each half of a rope pair (read by both rotations), and softmax's exps (read by the sum and the
  // divide). A split is one read.
  assert.deepEqual(
    steps
      .filter((s) => s.kind === "fanout")
      .map((s) => s.of)
      .sort(),
    ["kr.even", "kr.odd", "n1", "n1′", "n2", "q.even", "q.odd", "sm.exp", "x", "x1"],
  );
});

test("the whole model per token, for one sequence of 1,024 tokens", () => {
  const c = { V: 8192, D: 256, L: 64, H: 4, K: 64, F: 768, T: 1024 };
  const m = dv.model(c);
  // L blocks, the final norm and the tied head (cells T·V of 2D − 1), and the loss.
  assert.equal(dv.units(m.parts.blocks.forward), 64 * 2_849_501_184);
  assert.equal(
    dv.units(m.parts.head.forward) - 1024 * 8192 * (2 * 256 - 1),
    3 * 1024 * 256 + 1024 * 255 + 2 * 1024 + 1024, // the final norm, as RMSNorm above
  );
  // The loss, composed as a stable log-softmax, over every position. Forward:
  // the row max (TV − T compares), subtract (TV), exp (TV), sum (TV − T), log (T), subtract (TV);
  // the label's lookup moves values only; negate and mask (2T), the loss's and the mask's sums
  // (2T − 2), and the guard's max, the divide and the select (3): 5TV + 3T + 1. Backward: the
  // select and the divide (2); the mask's product, the negation and the label's scatter (3T); the
  // second subtract toward the rows' log (TV negations, TV − T to reduce onto the rows); log (T);
  // exp (TV); and the shifted logits, read twice (TV): 4TV + 3T + 2.
  const [T, TV] = [1024, 1024 * 8192];
  assert.equal(dv.units(m.parts.loss.forward), 5 * TV + 3 * T + 1);
  assert.equal(dv.units(m.parts.loss.backward), 4 * TV + 3 * T + 2);
  assert.equal(dv.units(m.forward), 186_697_651_201);
  assert.equal(dv.units(m.forward) + dv.units(m.backward), 558_764_633_859);
  assert.equal(m.perToken.training, 558_764_633_859 / 1024);
  assert.equal(dv.units(m.parts.tied.backward), 8192 * 256); // #E's two cotangents meet once
});

test("the plan: ampere-bf16's casts, StoreAll against Remat, and what each moves", () => {
  const a = pl.ampere();
  // Seven masters to bf16, and the f32 values entering bf16 contractions: both norm outputs, both
  // rope outputs, the attention probabilities and SwiGLU's product.
  assert.deepEqual(
    a.casts.map((x) => `${x.value}→${x.to}`),
    [
      "n1→bf16",
      "Wq→bf16",
      "Wk→bf16",
      "Wv→bf16",
      "q.pairs→bf16",
      "kr.pairs→bf16",
      "pr→bf16",
      "Wo→bf16",
      "n2→bf16",
      "W1→bf16",
      "W3→bf16",
      "m.h→bf16",
      "W2→bf16",
    ],
  );
  // Only contractions feeding contractions store bf16.
  assert.deepEqual(
    [...a.storage].filter(([, d]) => d === "bf16").map(([v]) => v),
    ["v", "o"],
  );
  const c = { V: 8192, D: 256, L: 64, H: 4, K: 64, F: 768, T: 1024 };
  // Remat keeps the stream into the iteration: T·D f32 values.
  assert.equal(pl.residuals(c, "Remat").bytes, 1024 * 256 * 4);
  // StoreAll keeps each value the backward reads, at its storage. The backward is derived after
  // the casts, as the design orders the passes, so each contraction keeps the bf16 operand it reads: in bf16,
  // n1 and n2, the rotated q and k, v and o, the probabilities, m.h and the seven weights; in f32,
  // the stream and the norms' rows (x, n1.xr, x1, n2.xr; n1.ve, n1.r, n2.ve, n2.r), softmax's
  // probabilities, exps and row sums, and m.a, m.g and m.b.
  const [T, D, H, K, F, f32, bf16] = [1024, 256, 4, 64, 768, 4, 2];
  const weights = 4 * D * H * K + 3 * D * F;
  const storeAll =
    (2 * T * D + 4 * T * H * K + T * T * H + T * F + weights) * bf16 +
    (4 * T * D + 4 * T + 2 * T * T * H + T * H + 3 * T * F) * f32;
  assert.equal(storeAll, 62_029_824);
  assert.equal(pl.residuals(c, "StoreAll").bytes, storeAll);
  // Recomputation is plan work; mathematical work does not move with the plan. Remat recomputes
  // what the backward reads and what that depends on: a derived program passes check, which
  // refuses dead nodes. The backward reads neither the last contraction, m = m.h·W2 (T·F·D
  // products and T·F·D − T·D adds), nor the block's output x2 (T·D adds). Casts are priced at none.
  const s = pl.work(c, "StoreAll");
  const r = pl.work(c, "Remat");
  const forward = 2_849_501_184;
  const m = 2 * T * F * D - T * D;
  assert.equal(m, 402_391_040);
  assert.equal(s.math, r.math);
  assert.equal(s.plan, 0);
  assert.equal(r.plan, forward - m - T * D);
  assert.equal(r.plan, 2_446_848_000);
  assert.equal(s.math, forward + 5_678_763_520);
  // The casts are nodes, applied before the backward is derived: each contraction reads bf16,
  // and a cast does no arithmetic, so the mathematical work is the block's own.
  const { program: p, storage } = pl.precise();
  const recomputed = pl.residuals(c, "Remat").recomputed;
  assert.deepEqual(
    p.nodes.map((n) => n.result).filter((v) => !recomputed.includes(v)),
    ["m", "x2"],
  );
  assert.deepEqual(pl.residuals(c, "StoreAll").recomputed, []);
  const through = (v) => {
    const n = p.nodes.find((m) => m.result === v);
    return n && n.prim.kind === "relabel" && n.operands.length === 1 ? through(n.operands[0]) : v;
  };
  assert.deepEqual(
    p.nodes
      .filter((n) => n.prim.kind === "convert")
      .map((n) => `${n.operands[0]}→${n.prim.to}`)
      .sort(),
    a.casts.map((x) => `${x.value}→${x.to}`).sort(),
  );
  for (const n of p.nodes.filter((m) => m.prim.kind === "contract"))
    for (const o of n.operands) assert.equal(storage.get(through(o)), "bf16", `${n.result}: ${o}`);
  const own = dv.blockWork(c);
  assert.equal(s.math, dv.units(own.forward) + dv.units(own.backward));
  // Plan edits move execution identity only; architecture is structure, configuration, numerics.
  for (const x of pl.EDITS) {
    const plan = x.edit.startsWith("Plan:");
    assert.equal(pl.movesArchitecture(x.moves), !plan && x.moves.length > 0, x.edit);
  }
});

// ----------------------------------------------------------------------------------- figure D

test("the family: declared ranges, derived axes, recounted parameters and work", () => {
  const m = fam.member(fam.MONAD_SETTING);
  assert.equal(m.D, 256);
  assert.equal(m.P, 32);
  assert.equal(m.parameters, 56_656_128);
  assert.equal(m.forward, 186_697_651_201 / 1024);
  assert.equal(fam.plain(m.forward), "180 million");
  assert.equal(fam.plain(m.parameters), "57 million");
  // The smallest member: V 4,096, L 2, H 1, K 16, F 256 (D 16): 4,096·16 + 2·(2·16 + 4·16·16 +
  // 3·16·256) + 16 = 65,536 + 2·25,632 + 16.
  const small = fam.member({ V: 4096, L: 2, H: 1, K: 16, F: 256 });
  assert.equal(small.parameters, 65_536 + 2 * (32 + 1024 + 12_288) + 16);
  assert.ok(!fam.inRange({ ...fam.MONAD_SETTING, K: 63 }));
  assert.throws(() => fam.member({ ...fam.MONAD_SETTING, L: 97 }));
});

// The log lanes keep every count legible; the share bar shows the parameters on a linear scale.
test("a member's parameters by part sum to its count, at Monad and at both corners", () => {
  const settings = [
    fam.MONAD_SETTING,
    { V: 4096, L: 2, H: 1, K: 16, F: 256 },
    { V: 65536, L: 96, H: 16, K: 128, F: 4096 },
  ];
  for (const s of settings) {
    const { parts, parameters } = fam.member(s);
    assert.equal(
      parts.E + parts.attention + parts.swiglu + parts.norms,
      parameters,
      JSON.stringify(s),
    );
  }
  // Monad by hand: E is 8,192·256; attention 64·4·256·256; SwiGLU 64·3·256·768; the norms 129·256.
  assert.deepEqual(fam.member(fam.MONAD_SETTING).parts, {
    E: 2_097_152,
    attention: 16_777_216,
    swiglu: 37_748_736,
    norms: 33_024,
  });
  const texts = view("fam/model")
    .sketch.prims.filter((p) => p.kind === "label" && p.key?.startsWith("count/share/"))
    .map((p) => p.text);
  for (const t of ["E 3.7%", "attention 30%", "SwiGLU 67%"]) assert.ok(texts.includes(t), t);
  assert.ok(texts.some((t) => t.includes("The norms, under 0.1%")));
});

test("a member's work for its whole sequence is a whole number of operations", () => {
  const m = fam.member(fam.MONAD_SETTING);
  assert.equal(m.sequence.forward, 186_697_651_201);
  assert.equal(m.sequence.training, 558_764_633_859);
  assert.equal(m.sequence.training / 1024, m.training);
});

// D draws each weight with edges that follow its axes: one step per doubling, one ink per axis.
test("the family's drawing: a control resizes every edge on its lane together", async () => {
  await loadAll();
  const d = await import("../src/ink/fam.ts");
  assert.equal(d.familyId(fam.MONAD_SETTING), "fam/model");
  const wide = { ...fam.MONAD_SETTING, K: 128 };
  assert.deepEqual(d.settingOf(d.familyId(wide)), wide);
  assert.equal(d.settingOf("fam/model/V1-L1-H1-K1-F1"), null);
  const lanes = (s, axis) =>
    view(d.familyId(s))
      .sketch.prims.filter((p) => p.key?.startsWith(`lane/${axis}/`) && p.kind === "stroke")
      .map((p) => Math.hypot(...[0, 1, 2].map((i) => p.pts[1][i] - p.pts[0][i])));
  // d = h·k: doubling k adds one step to every d edge, in every weight that has one.
  const [before, after] = [lanes(fam.MONAD_SETTING, "d"), lanes(wide, "d")];
  assert.equal(before.length, after.length);
  assert.ok(before.length >= 10, `${before.length} d edges`);
  for (const [i, b] of before.entries()) {
    assert.ok(Math.abs(b - 0.42 * 9) < 1e-9, `d edge ${i}: ${b}`);
    assert.ok(Math.abs(after[i] - 0.42 * 10) < 1e-9, `d edge ${i}: ${after[i]}`);
  }
  // The recount is drawn from the definitions.
  const texts = view("fam/model")
    .sketch.prims.filter((p) => p.kind === "label")
    .map((p) => p.text);
  for (const t of ["57 million", "56,656,128", "180 million", "186,697,651,201 for 1,024 tokens"])
    assert.ok(texts.includes(t), t);
});

test("the measured check draws the spectral study's rows, as its data file records them", async () => {
  const { readFile } = await import("node:fs/promises");
  const data = JSON.parse(
    await readFile(new URL("../src/data/spectral-certificate.json", import.meta.url), "utf8"),
  );
  const { CERTIFICATE } = await import("../src/ink/fam.ts");
  assert.deepEqual(CERTIFICATE.band, data.settings.band);
  const keys = ["width", "act_rms_ratio", "W_op_norm_ratio", "dW_op_norm_ratio"];
  for (const scheme of ["sp", "umup", "mup"])
    assert.deepEqual(
      CERTIFICATE.rows[scheme],
      data.certificate[scheme].rows.map((r) => Object.fromEntries(keys.map((k) => [k, r[k]]))),
      scheme,
    );
});

// Effects accompany actions, and none starts or stops in one frame: stepped at 60 Hz, a film's
// light, its effects' ink and its lettering change over several frames, at beat changes and as a
// chapter's ledger comes with details on. A cut is one frame's change far above its neighbours'.
test("a film never cuts: its light and lettering change over several frames", async () => {
  await loadAll();
  const light = (prims) =>
    prims.reduce(
      (e, p) =>
        e +
        (p.glow ?? 0) * (p.alpha ?? 1) +
        (p.kind === "face" && p.fill === "wash" ? 40 * (p.alpha ?? 1) : 0),
      0,
    );
  const lettering = (prims) =>
    prims
      .filter((p) => p.kind === "label" && p.key === undefined)
      .reduce((e, p) => e + (p.alpha ?? 1), 0);
  const marks = (prims) =>
    prims
      .filter(
        (p) =>
          p.key === undefined && p.kind !== "label" && !(p.kind === "face" && p.fill === "wash"),
      )
      .reduce((e, p) => e + (p.alpha ?? 1), 0);
  for (const mod of ["film.ts", "functors.ts"]) {
    const f = await import(`../src/ink/${mod}`);
    const { total } = f.film();
    for (const opts of [{}, { details: true }]) {
      const frames = [];
      for (let t = 0; t <= total + 8; t += 1000 / 60) {
        const { prims } = f.frameAt(Math.min(t, total), opts);
        frames.push({ t, l: light(prims), x: lettering(prims), m: marks(prims) });
      }
      const d = (k, i) => (i > 0 && i < frames.length ? frames[i][k] - frames[i - 1][k] : 0);
      const cuts = [];
      for (let i = 1; i < frames.length; i++)
        for (const [k, floor] of [
          ["l", Math.max(40, 0.2 * Math.max(frames[i].l, frames[i - 1].l))],
          ["x", 0.4],
          ["m", 1.2],
        ])
          if (
            Math.abs(d(k, i)) > floor &&
            Math.abs(d(k, i)) > 3 * Math.max(Math.abs(d(k, i - 1)), Math.abs(d(k, i + 1)))
          )
            cuts.push(`${k} at ${Math.round(frames[i].t)} ms`);
      assert.deepEqual(cuts, [], `${mod} ${JSON.stringify(opts)}`);
    }
  }
});

// C plays once, then the reader sets which bands show and the residual policy.
test("figure C's film: six chapters; at rest its bands hide and its policy switches", async () => {
  await loadAll();
  const c = await import("../src/ink/functors.ts");
  const { chapters, total } = c.film();
  assert.deepEqual(
    chapters.map((x) => x.id),
    ["term", "backward", "cost", "plan", "identity", "composition"],
  );
  assert.equal(c.frameAt(total).to.id, "fn/block");
  assert.equal(c.frameAt(total, { policy: "Remat" }).to.id, "fn/block/remat");
  for (const w of c.WALLS) {
    const prims = c.frameAt(total, { hide: [...w.keys] }).prims;
    const left = prims.filter(
      (p) => p.key && w.keys.some((k) => p.key === k || p.key.startsWith(`${k}/`)),
    );
    assert.equal(left.length, 0, w.id);
  }
  // Every frame says it is the window's model of the rules, not AMLO's output.
  for (let t = 0; t <= total; t += 1999)
    assert.ok(
      c.frameAt(t).prims.some((p) => p.kind === "label" && p.text === c.HONESTY),
      `t = ${t}`,
    );
});

// Effects accompany actions only: when a film ends, its drawing rests in ink, whatever the reader
// then sets.
test("each film's last frame rests in ink: no light, no wash", async () => {
  await loadAll();
  const lit = (prims) =>
    prims.filter((p) => (p.glow ?? 0) > 0.01 || (p.kind === "face" && p.fill === "wash"));
  const b = await import("../src/ink/film.ts");
  assert.deepEqual(lit(b.frameAt(b.film().total).prims), [], "B");
  const c = await import("../src/ink/functors.ts");
  const end = c.film().total;
  for (const o of [{}, { policy: "Remat" }, { hide: ["cost"] }, { details: true }])
    assert.deepEqual(lit(c.frameAt(end, o).prims), [], `C ${JSON.stringify(o)}`);
});

// The design draws recomputed nodes dashed. In C a dash means recompute and nothing
// else: pullbacks and the values StoreAll keeps are solid, and only Remat's wires are dashed.
test("figure C dashes only what Remat recomputes", async () => {
  await loadAll();
  const dashed = (id) =>
    view(id)
      .sketch.prims.filter((p) => p.dash?.length)
      .map((p) => p.key ?? "(no key)");
  for (const id of ["fn/backward", "fn/cost", "fn/plan", "fn/identity", "fn/block", "fn/attn"])
    assert.deepEqual(dashed(id), [], id);
  const remat = dashed("fn/block/remat");
  assert.ok(remat.length >= 4, `${remat.length} dashed`);
  assert.ok(
    remat.every((k) => k.startsWith("remat/")),
    remat.join(", "),
  );
});

// What the design says moves each identity: structure by computation, binding and tie edits;
// configuration by SetControl; numerics by Init; randomness by whatever adds or moves leaf
// families; execution by all of those and by plan edits, except donation and two other
// operational fields. Labels and layout move none of the five. One row departs from the design
// on purpose: untying #E adds a leaf with its own initializer declaration, and numerics hashes the
// initializer declared for each leaf, so the window has Untie move numerics too.
test("each identity edit moves what the design's table says, and says so", () => {
  const expected = {
    "SetOperator: silu to gelu_tanh": ["structure", "execution"],
    "SetControl: F 768 to 1,024": ["configuration", "execution"],
    "Init: a new scale for #W2": ["numerics", "execution"],
    "Untie #E from the head": ["structure", "numerics", "randomness", "execution"],
    "Plan: Remat on the Repeat": ["execution"],
    "Plan: f32 contractions": ["execution"],
    "Plan: donate off": [],
    "Group the halves, rename a definition": [],
  };
  assert.deepEqual(Object.fromEntries(pl.EDITS.map((e) => [e.edit, [...e.moves]])), expected);
  for (const e of pl.EDITS) {
    const named = pl.COMPONENTS.map((c) => c.id).filter((c) => e.say.includes(c));
    if (e.moves.length) assert.deepEqual(named, [...e.moves], e.say);
    else assert.match(e.say, /moves? none of the five\.$/, e.say);
  }
});

// The note under the window says what each view rests on: AMLO's design by default, C's model of
// AMLO's built work rules, D's counts, or the site's measured study.
test("every view says what it rests on: design, the window's model, its count, or a measurement", async () => {
  const { EVIDENCE, evidenceOf } = await import("../src/ink/view.ts");
  const seen = new Set();
  const queue = [...MODELS.flatMap((m) => m.path), ...FIGURES.flatMap((f) => f.steps)];
  while (queue.length) {
    const id = queue.shift();
    if (seen.has(id)) continue;
    seen.add(id);
    const v = view(id);
    queue.push(...v.sketch.openables.map((o) => o.child), ...(v.before ? [v.before] : []));
  }
  const want = (id) =>
    id.startsWith("fn/")
      ? "model"
      : id === "fam/spectral"
        ? "measured"
        : id.startsWith("fam/model")
          ? "count"
          : "design";
  for (const id of seen) {
    assert.ok(evidenceOf(view(id)) in EVIDENCE, id);
    assert.equal(evidenceOf(view(id)), want(id), id);
  }
  for (const id of ["fn/attn", "fn/swiglu", "fam/model", "fam/spectral", "rw/population", "model"])
    assert.ok(seen.has(id), id);
  assert.equal(view("fam/spectral").kicker, "This site’s spectral study");
});

test("C's cost ledger: the block's forward by kind sums to its total", async () => {
  const c = await import("../src/ink/functors.ts");
  const kinds = dv.blockWork({ V: 8192, D: 256, L: 64, H: 4, K: 64, F: 768, T: 1024 }).forward;
  assert.deepEqual(kinds, {
    arith: 2_836_133_888,
    transcendental: 4_982_784,
    compare_select: 8_384_512,
  });
  assert.equal(kinds.arith + kinds.transcendental + kinds.compare_select, c.costs().total.forward);
});
