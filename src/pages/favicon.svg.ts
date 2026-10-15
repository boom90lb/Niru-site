import type { APIRoute } from "astro";
import { cutMark } from "../lib/brand";

// The arrow runs corner to corner on a full-bleed ground, so it fills even
// a 16-pixel tab. Built from the same drawing as every other mark.
export const GET: APIRoute = () => {
  const mark = cutMark({ weight: 2.6, orientation: "diagonal", fit: "corners", spacing: 2.6 });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${mark.viewBox}"><rect width="100%" height="100%" fill="#0b0b0a"/><path d="${mark.ink}" fill="#e0b66e"/><path d="${mark.cut}" fill="#0b0b0a"/></svg>`;
  return new Response(svg, { headers: { "Content-Type": "image/svg+xml" } });
};
