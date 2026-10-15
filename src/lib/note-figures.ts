/**
 * A note places a figure with an HTML comment on a line of its own, such as
 * `<!-- figure:loss -->`. The note page splits the rendered text at each mark
 * and draws the figure there.
 */

export const NOTE_FIGURES = ["loss", "certificate", "kv"] as const;
export type NoteFigure = (typeof NOTE_FIGURES)[number];
export type NotePart =
  { html: string; figure?: undefined } | { figure: NoteFigure; html?: undefined };

const MARK = /<!--\s*figure:([a-z]+)\s*-->/;

export function splitFigures(html: string): NotePart[] {
  return html
    .split(MARK)
    .map((chunk, i): NotePart => {
      if (i % 2 === 0) return { html: chunk };
      if (!NOTE_FIGURES.includes(chunk as NoteFigure)) throw new Error(`unknown figure ${chunk}`);
      return { figure: chunk as NoteFigure };
    })
    .filter((part) => part.figure !== undefined || part.html.trim() !== "");
}
