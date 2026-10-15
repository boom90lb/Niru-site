import { expect, test } from "@playwright/test";
import { frame, REF, toSvg, type Prim } from "../../src/ink/engine.ts";
import { FIGURES, loadAll, MODELS, view } from "../../src/ink/models.ts";
import { cameraBetween, rewriteFrame } from "../../src/ink/morph.ts";
import { framing, type View } from "../../src/ink/view.ts";
import { waitForStablePage } from "./helpers";
import { lint } from "./ink-lint.mjs";

/**
 * Every view reachable from either model's training step, and every step of every figure with
 * the drawing it is rewritten from, which the window shows as the rewrite begins.
 */
function reachable(): string[] {
  const seen = new Set<string>();
  const queue = [...MODELS.flatMap((m) => [...m.path]), ...FIGURES.flatMap((f) => [...f.steps])];
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const v = view(id);
    for (const o of v.sketch.openables) queue.push(o.child);
    if (v.before) queue.push(v.before);
  }
  return [...seen];
}

const FILMS = {
  rewrite: () => import("../../src/ink/film.ts"),
  functors: () => import("../../src/ink/functors.ts"),
};

/**
 * A film's frames the lint checks: every beat a quarter, half and three quarters of the way
 * and at its end, every chapter's last frame with its ledger, and every jump between chapters
 * part way, forward from a chapter's end and back from its start.
 */
async function filmFrames(figure: keyof typeof FILMS = "rewrite") {
  const { film, frameAt, chapterEnd } = await FILMS[figure]();
  const { chapters, starts } = film();
  const frames: Array<{
    name: string;
    a: { prims: Prim[]; from: View; to: View; u: number };
    b?: { prims: Prim[]; from: View; to: View; u: number };
    u?: number;
  }> = [];
  chapters.forEach((c, ci) => {
    let t = starts[ci]!;
    for (const [bi, b] of c.beats.entries()) {
      for (const f of [0.25, 0.5, 0.75, 1])
        frames.push({
          name: `chapter ${ci + 1}, beat ${bi + 1} at ${f}`,
          a: frameAt(t + b.ms * f),
        });
      t += b.ms;
    }
    frames.push({
      name: `chapter ${ci + 1}, its ledger`,
      a: frameAt(chapterEnd(ci), { details: true }),
    });
    if (ci + 1 < chapters.length)
      for (const u of [0.25, 0.5, 0.75]) {
        frames.push({
          name: `Next from chapter ${ci + 1} at ${u}`,
          a: frameAt(chapterEnd(ci)),
          b: frameAt(starts[ci + 1]! + 0.5),
          u,
        });
        frames.push({
          name: `Back to chapter ${ci + 1} at ${u}`,
          a: frameAt(starts[ci + 1]! + 0.5),
          b: frameAt(starts[ci]! + 0.5),
          u,
        });
      }
  });
  return frames;
}

/** Set a drawing at the design size into the page as an SVG, ready for the lint. */
async function place(
  page: import("@playwright/test").Page,
  prims: Prim[],
  cam: ReturnType<typeof frame>["cam"],
  v: View,
) {
  const svg = toSvg(
    prims,
    cam,
    { width: REF.width, height: REF.height },
    { id: "lint", title: v.title, desc: v.caption },
  );
  await page.evaluate(
    ([markup, w, h]) => {
      document.querySelector("#ink-lint")?.remove();
      const box = document.createElement("div");
      box.id = "ink-lint";
      box.style.width = `${w}px`;
      box.style.height = `${h}px`;
      box.innerHTML = markup;
      document.body.append(box);
    },
    [svg, REF.width, REF.height] as const,
  );
}

// Each drawing is designed at the plate's full width and scales as one picture, so its text
// is checked once, at that size, with the site's own fonts: no label touches another label, a
// line or a dot, none breaks out of its box, and each keeps a margin from the frame.
test("every drawing's lettering is clear at its design size", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-1440", "one browser width suffices");
  test.setTimeout(120_000);
  await loadAll();
  await page.goto("/");
  await waitForStablePage(page);
  const problems: string[] = [];
  const ids = reachable();
  expect(ids.length).toBeGreaterThan(40);
  let labels = 0;
  for (const id of ids) {
    const v = view(id);
    const { cam } = frame(framing(v), v.yaw, v.pitch, REF.width, REF.height, v.maxScale);
    await place(page, v.sketch.prims, cam, v);
    labels += await page.locator("#ink-lint text").count();
    const found: string[] = await page.evaluate(lint, {
      margin: 12,
      selector: "#ink-lint svg",
      mark: false,
    });
    problems.push(...found.map((p) => `${id}: ${p}`));
  }
  expect(labels).toBeGreaterThan(1000);
  expect(problems).toEqual([]);
});

// The film is drawn frame by frame, effects and all, so its frames are drawings too. Each is
// checked with ink still fading in or out (below 30% opacity) left aside.
/** Lint frames of a film at the design size, as the window draws them. */
async function lintFrames(
  page: import("@playwright/test").Page,
  frames: Awaited<ReturnType<typeof filmFrames>>,
) {
  const problems: string[] = [];
  const cam = (f: { from: View; to: View; u: number }) =>
    cameraBetween(
      frame(framing(f.from), f.from.yaw, f.from.pitch, REF.width, REF.height, f.from.maxScale).cam,
      frame(framing(f.to), f.to.yaw, f.to.pitch, REF.width, REF.height, f.to.maxScale).cam,
      f.u,
    );
  for (const f of frames) {
    const prims = f.b ? rewriteFrame(f.a.prims, f.b.prims, f.u!) : f.a.prims;
    const c = f.b ? cameraBetween(cam(f.a), cam(f.b), f.u!) : cam(f.a);
    await place(page, prims, c, f.a.to);
    const found: string[] = await page.evaluate(lint, {
      margin: 12,
      selector: "#ink-lint svg",
      mark: false,
      minOpacity: 0.3,
    });
    problems.push(...found.map((p) => `${f.name}: ${p}`));
  }
  return problems;
}

test("every frame of the film, and every jump between its chapters, is clear", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-1440", "one browser width suffices");
  test.setTimeout(240_000);
  await loadAll();
  await page.goto("/");
  await waitForStablePage(page);
  const frames = await filmFrames();
  expect(frames.length).toBeGreaterThan(250);
  expect(await lintFrames(page, frames)).toEqual([]);
});

// C plays once; at rest the reader hides its bands and sets its policy. Each state is clear.
test("every frame of C's film, and each state the reader sets at its end, is clear", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-1440", "one browser width suffices");
  test.setTimeout(240_000);
  await loadAll();
  await page.goto("/");
  await waitForStablePage(page);
  const frames = await filmFrames("functors");
  expect(frames.length).toBeGreaterThan(80);
  const { film, frameAt, WALLS } = await FILMS.functors();
  const { total } = film();
  for (const policy of ["StoreAll", "Remat"] as const)
    for (const hide of [[], ...WALLS.map((w) => [...w.keys])])
      frames.push({
        name: `at rest, ${policy}, hiding ${hide.join(", ") || "nothing"}`,
        a: frameAt(total, { policy, hide }),
      });
  expect(await lintFrames(page, frames)).toEqual([]);
});
