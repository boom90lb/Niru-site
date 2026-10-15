import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

import { symmetricEigen } from "../src/lib/math/spectral.ts";
import { keyFromSeed, Stream } from "../src/lib/math/threefry.ts";

const data = JSON.parse(
  await readFile(new URL("../src/data/spectral-certificate.json", import.meta.url), "utf8"),
);
const generator = await readFile(new URL("../scripts/spectral-certificate.py", import.meta.url));

/** Largest singular value of a row-major matrix, from the eigenvalues of MᵀM. */
function opNorm(m) {
  const cols = m[0].length;
  const gram = Array.from({ length: cols }, (_, i) =>
    Array.from({ length: cols }, (_, j) => m.reduce((s, row) => s + row[i] * row[j], 0)),
  );
  const values = symmetricEigen(gram).values;
  return Math.sqrt(Math.max(...values));
}

const rms = (m) => Math.sqrt(m.flat().reduce((s, x) => s + x * x, 0) / m.flat().length);

test("stacking r copies keeps the entrywise RMS and multiplies the operator norm by √r", () => {
  const n = 24;
  const rng = Stream.of(keyFromSeed(5), "spectral-certificate");
  for (const r of [1, 2, 3, 4, 6, 8]) {
    const block = Array.from({ length: n / r }, () =>
      Array.from({ length: n }, () => rng.normal() / Math.sqrt(n)),
    );
    const stacked = Array.from({ length: r }, () => block).flat();
    assert.ok(Math.abs(rms(stacked) - rms(block)) < 1e-12);
    const ratio = opNorm(stacked) / opNorm(block);
    assert.ok(Math.abs(ratio - Math.sqrt(r)) < 1e-9, `r = ${r}: ratio ${ratio}`);
  }
});

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

/**
 * The data names the exact source that wrote it: the generator itself, or a
 * byte-exact copy kept in scripts/spectral-history/ after a change that does
 * not alter the computation. tests/python/test_spectral_provenance.py checks
 * that such a copy and the current generator are the same program.
 */
test("the data file was written by source kept in this tree", async () => {
  const history = new URL("../scripts/spectral-history/", import.meta.url);
  const kept = await readdir(history).catch(() => []);
  const digests = [sha256(generator)];
  for (const name of kept.filter((n) => n.endsWith(".py")))
    digests.push(sha256(await readFile(new URL(name, history))));
  assert.ok(
    digests.includes(data.generator.sha256),
    `no source in the tree has the recorded digest ${data.generator.sha256}`,
  );
  assert.equal(data.schema, "niru.spectral-certificate/1");
  for (const key of [
    "loss",
    "W",
    "dW",
    "op_norm",
    "spectral_normalisation",
    "act_rms",
    "band",
    "act3",
  ])
    assert.ok(typeof data.definitions[key] === "string", `definition of ${key}`);
});

test("act 3 rows obey the stacking identity and keep the coordinate scale", () => {
  const [lo, hi] = data.settings.band;
  for (const row of data.act3) {
    assert.ok(row.identity_error < 1e-9, `r = ${row.r}`);
    assert.ok(Math.abs(row.stacked_rms - row.block_rms) < 1e-12);
    assert.ok(Math.abs(row.stacked_op - Math.sqrt(row.r) * row.block_op) < 1e-9 * row.stacked_op);
    assert.equal(row.coord_inside, row.coord_ratio >= lo && row.coord_ratio <= hi);
    assert.equal(row.spectral_inside, row.spectral_ratio >= lo && row.spectral_ratio <= hi);
    // The stacked matrix has the rank of one block.
    assert.equal(row.stacked_rank, row.block_shape[0]);
  }
  assert.equal(data.act3[0].spectral_ratio, 1);
});

test("the sweep spans at least four octaves of width and resolves every minimum", () => {
  const widths = data.settings.widths;
  assert.ok(widths.every((w, i) => i === 0 || w > widths[i - 1]));
  assert.ok(widths.at(-1) / widths[0] >= 16);
  for (const [p, curves] of Object.entries(data.curves)) {
    assert.deepEqual(
      curves.map((c) => c.width),
      widths,
    );
    for (const c of curves) {
      const grid = c.points.map((q) => q.log2_lr);
      assert.ok(!c.minimum.edge, `${p} width ${c.width}: minimum at the grid edge`);
      assert.ok(c.minimum.log2_lr > grid[0] && c.minimum.log2_lr < grid.at(-1));
      // The fitted minimum lies within one grid step of the best grid point.
      const best = c.points.reduce((a, b) => (b.failed === 0 && b.loss < a.loss ? b : a));
      assert.ok(Math.abs(c.minimum.log2_lr - best.log2_lr) <= 0.5 + 1e-9);
    }
  }
});

test("certificate ratios are relative to the proxy and their flags follow the band", () => {
  const [lo, hi] = data.settings.band;
  for (const cert of Object.values(data.certificate)) {
    const proxy = cert.rows[0];
    for (const key of ["dW_op_norm", "W_op_norm", "act_rms"]) {
      assert.ok(Math.abs(proxy[`${key}_ratio`] - 1) < 1e-12);
      for (const row of cert.rows) {
        if (row[`${key}_ratio`] === undefined) continue;
        const v = row[`${key}_ratio`];
        assert.equal(row[`${key}_inside`], v >= lo && v <= hi);
      }
    }
  }
});

test("the data holds every scheme, width, grid rate and seed, failures included", () => {
  const { widths, seeds, lr_grid: grid } = data.settings;
  assert.deepEqual(Object.keys(data.curves).sort(), ["mup", "sp", "umup"]);
  assert.deepEqual(Object.keys(data.certificate).sort(), ["mup", "sp", "umup"]);
  for (const [p, curves] of Object.entries(data.curves)) {
    for (const c of curves) {
      assert.deepEqual(
        c.points.map((q) => q.log2_lr),
        grid[p],
        `${p} width ${c.width}: grid`,
      );
      for (const q of c.points) {
        assert.ok(Number.isInteger(q.failed) && q.failed >= 0 && q.failed <= seeds.length);
        assert.ok(Number.isFinite(q.loss), `${p} width ${c.width}: loss at ${q.log2_lr}`);
      }
    }
    assert.deepEqual(
      data.certificate[p].rows.map((row) => row.width),
      widths,
    );
  }
});

test("each certificate ratio is its value over the value at the proxy width", () => {
  for (const [p, cert] of Object.entries(data.certificate)) {
    const proxy = cert.rows[0];
    for (const row of cert.rows) {
      assert.ok(row.failed >= 0 && row.failed <= data.settings.seeds.length);
      for (const key of ["dW_op_norm", "W_op_norm", "act_rms"]) {
        if (row[key] === undefined) {
          assert.equal(row[`${key}_ratio`], undefined, `${p} width ${row.width}: ${key}`);
          continue;
        }
        const expected = row[key] / proxy[key];
        assert.ok(
          Math.abs(row[`${key}_ratio`] - expected) <= 1e-12 * Math.abs(expected),
          `${p} width ${row.width}: ${key} ratio`,
        );
      }
    }
  }
});
