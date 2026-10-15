import type { APIRoute } from "astro";
import { cutMark } from "../lib/brand";

// The header mark: the arrow laid flat, cut at header weight.
export const GET: APIRoute = () => {
  const mark = cutMark({ weight: 1.6, spacing: 1.4 });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${mark.viewBox}"><path d="${mark.ink}" fill="#e0b66e"/><path d="${mark.cut}" fill="#0b0b0a"/></svg>`;
  return new Response(svg, { headers: { "Content-Type": "image/svg+xml" } });
};
