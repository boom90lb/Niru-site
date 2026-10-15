/**
 * The Niru mark: a broadhead arrow (Manchu niru), cut as a woodcut.
 *
 * There is one drawing. Every size uses the same proportions, the same barbs
 * and the same seeded knife; only the line weight changes, the way a typeface
 * carries more weight at small optical sizes. The ink layer is the standing
 * wood; the cut layer is carved back out in the ground color.
 */
import { bezier, carve, hatch, pathData, profile, random, rotate, type Point } from "./woodcut.ts";

export interface Cut {
  viewBox: string;
  width: number;
  height: number;
  ink: string;
  cut: string;
}

export interface CutOptions {
  /** Line weight. 1 is the plate; small sizes need more. */
  weight?: number;
  /** "flat" points right; "diagonal" points to the top-right corner. */
  orientation?: "flat" | "diagonal";
  /**
   * For the diagonal: "bounds" fits the whole drawing in its square, which
   * already puts the tip on the top-right corner because the drawing is
   * symmetric about its axis. "corners" then scales about the tip, so the
   * feathers also run into the bottom-left corner and just past the edges.
   */
  fit?: "bounds" | "corners";
  /** Outline sample spacing, in block units; larger is lighter in bytes. */
  spacing?: number;
}

const Y = 32;
const NOCK = 16;
const TIP = 233;
const SOCKET = 205;
const SHOULDER = 214;
const HALF = 6.8;
const FRONT = 76;
const BACK = 22;
const BARBS = 22;
const REACH = 23;

/** The arrow, pointing right along y = 32, in a 240 x 64 block. */
function arrow(weight: number, spacing: number): { ink: Point[][]; cut: Point[][] } {
  const w = weight;
  const rand = random(0x6e697275); // "niru"
  const ink: Point[][] = [];
  const cut: Point[][] = [];
  const rough = 0.22 / Math.sqrt(w);
  const line = (points: Point[], width: (t: number) => number, edge = rough) =>
    carve(points, { width, seed: Math.floor(rand() * 1e6), rough: edge, spacing });

  // Proportions follow the print: the head is about an eighth of the arrow,
  // the fletching about a quarter, and the vanes stand wider than the head.
  const tail: Point = [SOCKET, Y];
  const upper: Point = [SHOULDER, Y - HALF];
  const tip: Point = [TIP, Y];
  const lower: Point = [SHOULDER, Y + HALF];

  // Shaft, with a grain cut along its upper side.
  ink.push(
    line(
      bezier(
        [
          [NOCK, Y],
          [SOCKET + 2, Y],
        ],
        2,
      ),
      profile(3 * w, 0.012, 0.01),
      rough * 0.6,
    ),
  );
  cut.push(
    line(
      bezier(
        [
          [FRONT + 10, Y - 0.5 * w],
          [SOCKET - 11, Y - 0.5 * w],
        ],
        2,
      ),
      profile(0.6 * w, 0.3, 0.3),
      0.4,
    ),
  );

  // Broadhead: a narrow lozenge with a raised center ridge.
  const edge = (a: Point, b: Point, entry: number, exit: number) =>
    ink.push(line(bezier([a, b], 2), profile(1.8 * w, entry, exit)));
  edge(tail, upper, 0.03, 0.05);
  edge(upper, tip, 0.05, 0.3);
  edge(tip, lower, 0.3, 0.05);
  edge(lower, tail, 0.05, 0.03);
  ink.push(
    line(
      bezier(
        [
          [SOCKET + 2, Y],
          [TIP - 3, Y],
        ],
        2,
      ),
      profile(1.4 * w, 0.06, 0.4),
    ),
  );

  // Light falls from above: the lower facet carries denser, heavier cuts.
  const facet = (sign: 1 | -1, count: number, base: number, extra: number) =>
    hatch({
      from: 0,
      to: 1,
      count,
      seed: Math.floor(rand() * 1e6),
      rough: rough + 0.1,
      line: (u) => {
        const y = Y + sign * (HALF * 0.25 + u * HALF * 0.53);
        const k = Math.abs(y - Y) / HALF;
        const left = tail[0] + k * (upper[0] - tail[0]) + 1.6;
        const right = tip[0] - k * (tip[0] - upper[0]) - 2;
        return bezier(
          [
            [left, y],
            [right, y],
          ],
          2,
        );
      },
      width: (u, t) => (base + extra * u) * w * (0.5 + 0.5 * Math.sin(Math.PI * t)),
    });
  ink.push(...facet(1, 3, 0.7, 0.4));
  ink.push(...facet(-1, 1, 0.4, 0));

  // Sinew bindings where the head and the fletching meet the shaft.
  // Weight changes line width only; every length and offset below is fixed.
  const binding = (x0: number, turns: number) => {
    for (let k = 0; k < turns; k += 1) {
      const x = x0 + k * 2.6;
      ink.push(
        line(
          bezier(
            [
              [x - 0.5, Y - 4.4],
              [x + 0.5, Y + 4.4],
            ],
            2,
          ),
          profile(1.05 * w, 0.2, 0.2),
        ),
      );
    }
  };
  binding(SOCKET - 9, 4);
  binding(FRONT + 4, 3);

  // Fletching: each vane is a row of barbs swept back from the shaft. They
  // vary a little in length and angle, the way feathers do.
  for (const sign of [-1, 1] as const) {
    for (let k = 0; k < BARBS; k += 1) {
      const f = k / (BARBS - 1);
      const x = FRONT - f * (FRONT - BACK);
      // Vane outline: a quick rise at the front, a long full body, a square cut at the back.
      const body = Math.min(1, 0.2 + ((FRONT - x) / 14) * 0.8);
      const reach = REACH * body * (0.92 + rand() * 0.16);
      const angle = ((36 + (rand() - 0.5) * 6) * Math.PI) / 180;
      const end: Point = [x - Math.cos(angle) * reach, Y + sign * Math.sin(angle) * reach];
      const bend: Point = [
        x - Math.cos(angle) * reach * 0.42,
        Y + sign * Math.sin(angle) * reach * 0.64,
      ];
      ink.push(
        line(bezier([[x, Y + sign * 1.2], bend, end], 10), profile(1.15 * w, 0.06, 0.55, 0.12)),
      );
    }
  }

  // Nock: the shaft ends in a short split.
  for (const sign of [-1, 1] as const) {
    ink.push(
      line(
        bezier(
          [
            [NOCK + 3, Y + sign * 0.5],
            [NOCK - 5, Y + sign * 4],
          ],
          2,
        ),
        profile(1.3 * w, 0.05, 0.4),
      ),
    );
  }
  return { ink, cut };
}

const transform = (polygons: Point[][], map: (p: Point) => Point) =>
  polygons.map((polygon) => polygon.map(map));

export function cutMark(options: CutOptions = {}): Cut {
  const { weight = 1, orientation = "flat", fit = "bounds", spacing = 0.7 } = options;
  const { ink, cut } = arrow(weight, spacing);
  if (orientation === "flat") {
    // Crop to the ink, so a layout sizes and aligns the drawing, not empty margins.
    const xs = ink.flat().map((p) => p[0]);
    const ys = ink.flat().map((p) => p[1]);
    const x0 = Math.floor(Math.min(...xs));
    const y0 = Math.floor(Math.min(...ys));
    const w = Math.ceil(Math.max(...xs)) - x0;
    const h = Math.ceil(Math.max(...ys)) - y0;
    return {
      viewBox: `${x0} ${y0} ${w} ${h}`,
      width: w,
      height: h,
      ink: pathData(ink, 1),
      cut: pathData(cut, 1),
    };
  }
  const turn = (p: Point) => rotate(p, -Math.PI / 4, [NOCK, Y]);
  const tInk = transform(ink, turn);
  const tCut = transform(cut, turn);
  const xs = tInk.flat().map((p) => p[0]);
  const ys = tInk.flat().map((p) => p[1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const size = Math.ceil(Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY));
  const grow = fit === "corners" ? 1.13 : 1;
  // The tip sits at (size, 0); scaling about it keeps the tip in its corner.
  const fitBounds = (p: Point): Point => [
    size + (p[0] - minX - size) * grow,
    (p[1] - minY) * grow,
  ];
  return {
    viewBox: `0 0 ${size} ${size}`,
    width: size,
    height: size,
    ink: pathData(transform(tInk, fitBounds), 1),
    cut: pathData(transform(tCut, fitBounds), 1),
  };
}
