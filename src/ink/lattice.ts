/**
 * The element level: an index map drawn on the elements it touches. Each view
 * is a small instance of the real map, drawn exactly; the caption gives the
 * sizes the model uses. A filled dot is an element the map keeps, a small
 * open dot one it leaves out.
 */
import { DSV41 } from "./dsv41.ts";
import type { Ink } from "./engine.ts";
import { design, Sketch } from "./kit.ts";
import type { View } from "./view.ts";

const STEP = 1.6;
/** Dot sizes follow the grid spacing as drawn, so a lattice reads at any scale. */
const big = () => Math.max(4, 0.2 * STEP * design.ppu);
const small = () => Math.max(1.6, 0.05 * STEP * design.ppu);

interface Grid {
  rows: number;
  cols: number;
  on: (r: number, c: number) => Ink | null;
  rowName: string;
  colName: string;
  /** Radius of a kept element, by its value when the map carries one. */
  size?: (r: number, c: number) => number;
}

/** A lattice with row 0 at the top: rows are queries or tokens, columns what they read. */
function grid(s: Sketch, g: Grid, x0 = 0, y0 = 0) {
  for (let r = 0; r < g.rows; r++)
    for (let c = 0; c < g.cols; c++) {
      const ink = g.on(r, c);
      const at: [number, number, number] = [x0 + c * STEP, y0 - r * STEP, 0];
      s.add(
        ink
          ? { kind: "dot", at, r: g.size ? g.size(r, c) : big(), ink, layer: 0 }
          : { kind: "dot", at, r: small(), ink: "faint", layer: 0 },
      );
    }
  for (let r = 0; r < g.rows; r++)
    s.text(x0 - 1.2, y0 - r * STEP, `${r}`, { size: 11, ink: "faint", anchor: "end", dy: 4 });
  for (let c = 0; c < g.cols; c++)
    s.text(x0 + c * STEP, y0 + 1.1, `${c}`, { size: 11, ink: "faint", anchor: "middle" });
  s.text(x0 - 1.2, y0 + 1.1, g.rowName, { size: 12, italic: true, anchor: "end" });
  s.text(x0 + (g.cols - 1) * STEP + 1.4, y0 + 1.1, g.colName, { size: 12, italic: true });
}

function lattice(id: string, title: string, caption: string, draw: (s: Sketch) => void): View {
  const s = new Sketch();
  draw(s);
  return { id, title, caption, sketch: s, yaw: -0.12, pitch: 0.22, maxScale: 26 };
}

const note = (s: Sketch, x: number, y: number, text: string) =>
  s.text(x, y, text, { size: 12, italic: true });

function causal() {
  return lattice(
    "lat/causal",
    "The causal mask, element by element",
    "Query t reads key t′ only when t′ is at most t, so no position sees the future. Drawn for 8 positions; the mask is the same at any length.",
    (s) => {
      grid(s, {
        rows: 8,
        cols: 8,
        rowName: "t",
        colName: "t′",
        on: (t, k) => (k <= t ? "ink" : null),
      });
      note(s, 14.6, -1, "query t reads key t′");
      note(s, 14.6, -2.6, "when t′ is at most t");
    },
  );
}

function rope() {
  return lattice(
    "lat/rope",
    "Rotary positions, channels in pairs",
    "Rotary positions pair adjacent channels. Channel c = 2p + j is element j of pair p, and pair p turns by an angle of t times its own frequency. Drawn with 12 channels; Monad’s heads have 64, in 32 pairs.",
    (s) => {
      const n = 12;
      for (let c = 0; c < n; c++) {
        const x = c * STEP;
        const p = Math.floor(c / 2);
        const j = c % 2;
        const px = p * 2 * STEP + STEP / 2;
        const py = -5 - j * STEP;
        s.add({ kind: "dot", at: [x, 0, 0], r: big(), ink: "cinnabar", layer: 0 });
        s.text(x, 0, `${c}`, { size: 11, ink: "faint", anchor: "middle", dy: -big() - 7 });
        s.add({
          kind: "stroke",
          pts: [
            [x, -0.5, 0],
            [px + (j ? 0.3 : -0.3), py + 0.4, 0],
          ],
          w: 1,
          ink: "faint",
          layer: 0,
        });
        s.add({
          kind: "dot",
          at: [px + (j ? 0.3 : -0.3), py, 0],
          r: big(),
          ink: "cinnabar",
          layer: 0,
        });
      }
      for (let p = 0; p < n / 2; p++)
        s.text(p * 2 * STEP + STEP / 2, -5 - 2 * STEP, `${p}`, {
          size: 11,
          ink: "faint",
          anchor: "middle",
          dy: 12,
        });
      s.text(-1.2, 0, "c", { size: 12, italic: true, anchor: "end", dy: 4 });
      s.text(-1.2, -5, "j = 0", { size: 12, italic: true, anchor: "end", dy: 4 });
      s.text(-1.2, -5 - STEP, "j = 1", { size: 12, italic: true, anchor: "end", dy: 4 });
      s.text(-1.2, -5 - 2 * STEP, "p", { size: 12, italic: true, anchor: "end", dy: 12 });
      note(s, n * STEP + 1, -1, "c = 2p + j");
      note(s, n * STEP + 1, -2.6, "pair p turns by t times its frequency");
    },
  );
}

function band() {
  const W = 4;
  return lattice(
    "lat/band",
    "The window band, element by element",
    `Query t reads the window entries t′ from t − W + 1 to t, its own position included. Drawn for 12 positions and W = ${W}; the toy uses W = ${DSV41.W} at ${DSV41.T} positions. AMLO computes the band as a dense mask.`,
    (s) => {
      grid(s, {
        rows: 12,
        cols: 12,
        rowName: "t",
        colName: "t′",
        on: (t, k) => (t - W < k && k <= t ? "ink" : null),
      });
      note(s, 20.6, -1, `query t reads t′ in (t − ${W}, t]`);
    },
  );
}

function visible() {
  const r = 2;
  return lattice(
    "lat/visible",
    "Visible compressed entries",
    `At ratio 2, entry c holds positions 2c and 2c + 1, so query t may read it only once both are past, which is when 2(c + 1) is at most t + 1. At ratio 1 entry c is position c. Drawn for 12 queries; the indexer masks the rest before it selects.`,
    (s) => {
      grid(s, {
        rows: 12,
        cols: 6,
        rowName: "t",
        colName: "c",
        on: (t, c) => (r * (c + 1) <= t + 1 ? "cinnabar" : null),
      });
      note(s, 11, -1, "entry c is complete for query t");
      note(s, 11, -2.6, "when 2(c + 1) is at most t + 1");
    },
  );
}

function stride() {
  return lattice(
    "lat/stride",
    "The compression stride",
    "The index map t = 2c + j sends position t to element j of entry c, so positions 2c and 2c + 1 become entry c. Drawn for 12 positions. Entry c is rotated as position 2c.",
    (s) => {
      for (let t = 0; t < 12; t++) {
        const c = Math.floor(t / 2);
        const x = t * STEP;
        const cx = c * 2 * STEP + STEP / 2;
        s.add({ kind: "dot", at: [x, 0, 0], r: big(), ink: "ink", layer: 0 });
        s.text(x, 0, `${t}`, { size: 11, ink: "faint", anchor: "middle", dy: -big() - 7 });
        s.add({
          kind: "stroke",
          pts: [
            [x, -0.5, 0],
            [cx, -4.6, 0],
          ],
          w: 1,
          ink: t % 2 ? "faint" : "ink",
          layer: 0,
        });
      }
      for (let c = 0; c < 6; c++) {
        const cx = c * 2 * STEP + STEP / 2;
        s.add({ kind: "dot", at: [cx, -5, 0], r: big() * 1.2, ink: "cinnabar", layer: 0 });
        s.text(cx, -5, `${c}`, { size: 11, ink: "faint", anchor: "middle", dy: big() * 1.2 + 16 });
      }
      s.text(-1.2, 0, "t", { size: 12, italic: true, anchor: "end", dy: 4 });
      s.text(-1.2, -5, "c", { size: 12, italic: true, anchor: "end", dy: 4 });
      note(s, 12 * STEP + 0.6, -1.4, "t = 2c + j");
      note(s, 12 * STEP + 0.6, -3, "j = 0 in ink, j = 1 faint");
    },
  );
}

function route() {
  // An illustrative choice of two experts per token, fixed so the drawing is deterministic.
  const pick = [
    [1, 6],
    [0, 3],
    [3, 4],
    [2, 6],
    [5, 7],
    [1, 2],
    [4, 6],
    [0, 7],
    [3, 5],
    [1, 4],
  ];
  return lattice(
    "lat/route",
    "Routing two experts per token",
    `Each token uses ${DSV41.A} of the ${DSV41.E} routed experts. They are chosen on the gate score plus the bias and weighted by the score alone. The choices drawn for these ten tokens are an example, not a measurement.`,
    (s) => {
      grid(s, {
        rows: pick.length,
        cols: DSV41.E,
        rowName: "t",
        colName: "e",
        on: (t, e) => (pick[t]!.includes(e) ? "indigo" : null),
      });
      note(s, 14.6, -1, "each row: the two chosen experts");
      note(s, 14.6, -2.6, "every column: one expert’s tokens");
    },
  );
}

/** Sinkhorn as the reference runs it, on a fixed example: a softmax along each row, then columns, then rows and columns again. */
export function sinkhorn(logits: number[][], rounds = 20, eps = 1e-6): number[][] {
  let m = logits.map((row) => {
    const mx = Math.max(...row);
    const e = row.map((v) => Math.exp(v - mx));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z + eps);
  });
  const cols = () => {
    const sums = m[0]!.map((_, k) => m.reduce((a, row) => a + row[k]!, 0));
    m = m.map((row) => row.map((v, k) => v / (sums[k]! + eps)));
  };
  cols();
  for (let i = 1; i < rounds; i++) {
    m = m.map((row) => {
      const z = row.reduce((a, b) => a + b, 0);
      return row.map((v) => v / (z + eps));
    });
    cols();
  }
  return m;
}

function sinkhornView() {
  const logits = [
    [2.0, 0.4, -0.6, 0.1],
    [0.3, 1.6, 0.2, -0.8],
    [-0.4, 0.5, 1.2, 0.9],
    [0.6, -0.2, 0.3, 1.8],
  ];
  const before = logits.map((row) => {
    const mx = Math.max(...row);
    const e = row.map((v) => Math.exp(v - mx));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  });
  const after = sinkhorn(logits);
  return lattice(
    "lat/sinkhorn",
    "Sinkhorn makes comb doubly stochastic",
    "comb starts as 16 numbers per position. A softmax along each row makes the rows sum to 1; then the columns are divided by their sums, and rows and columns are normalized in turn, 20 times in all. Every row and every column of the result sums to 1, so each output stream is a weighted average of the input streams. The example numbers are fixed for the drawing; each dot’s area is its value.",
    (s) => {
      const size = (m: number[][]) => (r: number, c: number) =>
        big() * (0.25 + 1.6 * Math.sqrt(m[r]![c]!));
      const draw = (m: number[][], x0: number, title: string) => {
        grid(
          s,
          { rows: 4, cols: 4, rowName: "j", colName: "k", on: () => "verdigris", size: size(m) },
          x0,
          0,
        );
        s.text(x0 + 1.5 * STEP, -4 * STEP - 0.6, title, {
          size: 12,
          italic: true,
          anchor: "middle",
        });
        m.forEach((row, r) =>
          s.text(x0 + 3 * STEP + 1.5, -r * STEP, row.reduce((a, b) => a + b, 0).toFixed(2), {
            size: 11,
            ink: "faint",
            dy: 4,
          }),
        );
        for (let c = 0; c < 4; c++)
          s.text(x0 + c * STEP, -3 * STEP - 1.5, m.reduce((a, row) => a + row[c]!, 0).toFixed(2), {
            size: 11,
            ink: "faint",
            anchor: "middle",
            dy: 4,
          });
      };
      draw(before, 0, "a softmax along each row");
      draw(after, 13, "after 20 rounds");
      s.text(11.2, -2.4, "rows, then columns", {
        size: 11,
        italic: true,
        ink: "faint",
        anchor: "end",
        dx: 4,
      });
    },
  );
}

function blocks() {
  const bs = 4;
  // Example index scores for 16 entries, fixed for the drawing.
  const score = [0.2, 0.9, 0.4, 0.1, 0.3, 0.2, 0.1, 0.5, 1.4, 0.6, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3];
  const newest = 3;
  const blockMax = [0, 1, 2, 3].map((b) => Math.max(...score.slice(b * bs, b * bs + bs)));
  const kept = new Set([newest, blockMax.indexOf(Math.max(...blockMax.slice(0, newest)))]);
  return lattice(
    "lat/blocks",
    "The candidate pool",
    `Layer 20 scores its entries in blocks, and a block’s score is the best score in it. The block holding the query’s newest entry is always kept, and the best other blocks fill the pool. In post-training the Reindex layers score only entries in the pool. Drawn with blocks of ${bs} and a pool of 2 blocks; the toy keeps ${DSV41.NB} blocks of ${DSV41.BS}. The scores are an example.`,
    (s) => {
      score.forEach((v, c) => {
        const b = Math.floor(c / bs);
        const x = c * STEP + b * 0.8;
        const ink: Ink = kept.has(b) ? "indigo" : "faint";
        s.add({
          kind: "dot",
          at: [x, 0, 0],
          r: big() * (0.3 + 0.8 * Math.sqrt(v)),
          ink,
          layer: 0,
        });
        s.text(x, 0, `${c}`, { size: 11, ink: "faint", anchor: "middle", dy: -big() * 1.1 - 7 });
      });
      for (let b = 0; b < 4; b++) {
        const x0 = b * bs * STEP + b * 0.8 - 0.5;
        const x1 = x0 + (bs - 1) * STEP + 1;
        s.add({
          kind: "stroke",
          pts: [
            [x0, -1.2, 0],
            [x0, -1.6, 0],
            [x1, -1.6, 0],
            [x1, -1.2, 0],
          ],
          w: 1,
          ink: kept.has(b) ? "indigo" : "faint",
          layer: 0,
          halo: false,
        });
        s.text(
          (x0 + x1) / 2,
          -1.6,
          b === newest
            ? "newest: kept"
            : `best ${blockMax[b]!.toFixed(1)}${kept.has(b) ? ": kept" : ""}`,
          {
            size: 11,
            italic: true,
            ink: kept.has(b) ? "indigo" : "faint",
            anchor: "middle",
            dy: 16,
          },
        );
      }
      s.text(-1.2, 0, "c′", { size: 12, italic: true, anchor: "end", dy: 4 });
    },
  );
}

const builders: Record<string, () => View> = {
  "lat/causal": causal,
  "lat/rope": rope,
  "lat/band": band,
  "lat/visible": visible,
  "lat/stride": stride,
  "lat/route": route,
  "lat/sinkhorn": sinkhornView,
  "lat/blocks": blocks,
};

export function view(id: string): View | undefined {
  return builders[id]?.();
}
