/**
 * The few pieces the site's static plots need: a scale from data to SVG
 * coordinates, and path data for a polyline. The plots are drawn at build
 * time, so the page ships SVG and no plotting code.
 */

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** A base-2 logarithmic scale; the domain is in data units. */
export function log2(domain: [number, number], range: [number, number]): Scale {
  const inner = linear([Math.log2(domain[0]), Math.log2(domain[1])], range);
  const f = ((v: number) => inner(Math.log2(v))) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

const round = (v: number) => Math.round(v * 10) / 10;

/** Path data through the points, in order. */
export function polyline(points: Array<[number, number]>): string {
  return points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${round(x)} ${round(y)}`).join(" ");
}

/** A small triangle or square marker centred on (x, y). */
export function marker(shape: "triangle" | "square", x: number, y: number, size = 4): string {
  if (shape === "square") {
    const h = size * 0.85;
    return `M${round(x - h)} ${round(y - h)} h${round(2 * h)} v${round(2 * h)} h${round(-2 * h)} Z`;
  }
  const h = size * 1.1;
  return `M${round(x)} ${round(y - h)} L${round(x + h)} ${round(y + h * 0.8)} L${round(x - h)} ${round(y + h * 0.8)} Z`;
}

export { round };
