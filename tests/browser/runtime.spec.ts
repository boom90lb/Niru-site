import { gzipSync } from "node:zlib";
import { expect, test } from "@playwright/test";
import { CORE_ROUTES, RESULT_ROUTES } from "./routes";

for (const route of RESULT_ROUTES) {
  test(`${route.path} is excluded from indexing`, async ({ page }) => {
    await page.goto(route.path);
    await expect(page.locator('meta[name="robots"]')).toHaveAttribute(
      "content",
      "noindex,nofollow,noarchive",
    );
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
      "href",
      `https://niru.ml${route.path}`,
    );
  });
}

for (const route of CORE_ROUTES) {
  test(`${route.path} makes no third-party requests and sets no cookies`, async ({
    page,
    context,
  }) => {
    const requests: string[] = [];
    page.on("request", (request) => {
      requests.push(request.url());
    });
    await page.goto(route.path, { waitUntil: "networkidle" });
    const pageOrigin = new URL(page.url()).origin;
    expect(requests.filter((url) => new URL(url).origin !== pageOrigin)).toEqual([]);
    expect(await context.cookies()).toEqual([]);
  });
}

test("route payloads stay inside practical static-site budgets", async ({ page }) => {
  for (const route of CORE_ROUTES) {
    const responses: Array<Promise<{ url: string; type: string; bytes: number } | undefined>> = [];
    page.on("response", (response) => {
      responses.push(
        response
          .body()
          .then((body) => ({
            url: response.url(),
            type: response.headers()["content-type"] ?? "",
            bytes: body.byteLength,
          }))
          .catch(() => undefined),
      );
    });

    await page.goto(route.path, { waitUntil: "networkidle" });
    // The home page's window plays the search's film once it is in view: measure with it. The
    // film's chunk pulls in its own chunks after it, and the page reached network idle long
    // before, so wait for the film's controls: they show only once every chunk has loaded.
    const algebra = page.locator("[data-ink-window]");
    if ((await algebra.count()) > 0) {
      await algebra.scrollIntoViewIfNeeded();
      const activate = algebra.locator(".ink-mobile-open");
      if (await activate.isVisible()) {
        await activate.click();
        await algebra.locator('.ink-figure button[data-figure="rewrite"]').click();
      }
      await expect(algebra.locator(".ink-steps")).toBeVisible();
    }
    const resources = (await Promise.all(responses)).filter(
      (resource): resource is { url: string; type: string; bytes: number } => Boolean(resource),
    );
    const unique = Array.from(new Map(resources.map((r) => [r.url, r])).values());
    const totalBytes = unique.reduce((sum, r) => sum + r.bytes, 0);
    const javascriptBytes = unique
      .filter((r) => /javascript/.test(r.type) || /\.js(?:\?|$)/.test(r.url))
      .reduce((sum, r) => sum + r.bytes, 0);
    const domNodes = await page.locator("body *").count();

    // Text, one stylesheet and its fonts. Plots are inline SVG and formulas
    // are MathML. The home page ships the optional algebra window,
    // which replaces its own still drawing and plays the search's film as it
    // comes into view, so the home page may weigh 480 KiB, film included: about 461 KiB
    // today, most of it Source Serif with its optical-size axis, kept on purpose.
    // Its JavaScript has its own budget, gzipped, below.
    expect(unique.length, `${route.path} request count`).toBeLessThanOrEqual(12);
    expect(totalBytes, `${route.path} total bytes`).toBeLessThanOrEqual(
      (route.path === "/" ? 480 : 400) * 1024,
    );
    if (!["/", "/lab/"].includes(route.path))
      expect(javascriptBytes, `${route.path} JavaScript bytes`).toBe(0);
    // The scientific note retains all 60 per-layer rows in native HTML for no-JS inspection.
    // Keep a separate bound for that disclosure; payload and request budgets still apply.
    expect(domNodes, `${route.path} DOM nodes`).toBeLessThanOrEqual(
      route.path === "/research/spectral-diagnostics/" ? 1200 : 800,
    );
  }
});

// The window's JavaScript, as a reader who opens every figure loads it: gzipped, as it travels.
test("the window's JavaScript stays inside its gzipped budget with every figure opened", async ({
  page,
}) => {
  const js = new Map<string, number>();
  const bodies: Array<Promise<void>> = [];
  page.on("response", (response) => {
    if (!/\.js(?:\?|$)/.test(response.url())) return;
    bodies.push(
      response
        .body()
        .then((body) => {
          js.set(response.url(), gzipSync(body, { level: 9 }).byteLength);
        })
        .catch(() => undefined),
    );
  });
  const chunk = (name: string) =>
    page.waitForResponse((r) => new RegExp(`/${name}\\.[\\w-]+\\.js$`).test(r.url()));
  const settled = async () => {
    await page.waitForLoadState("networkidle");
    await Promise.all(bodies);
  };
  await page.goto("/", { waitUntil: "networkidle" });
  const win = page.locator("[data-ink-window]");
  // B plays once the window is in view; D opens from the model view, C from the block.
  const film = chunk("film");
  await win.scrollIntoViewIfNeeded();
  const activate = win.locator(".ink-mobile-open");
  if (await activate.isVisible()) {
    await activate.click();
    await win.locator('.ink-figure button[data-figure="rewrite"]').click();
  }
  await film;
  await win.locator(".ink-stage").focus();
  await page.keyboard.press("Escape");
  const family = chunk("fam");
  await win.locator('.ink-figure button[data-figure="family"]').click();
  await family;
  await expect(win.locator(".ink-family")).toBeVisible();
  await page.keyboard.press("Escape");
  await win.locator(".ink-open-buttons button", { hasText: "layer" }).first().click();
  const functors = chunk("functors");
  await win.locator('.ink-figure button[data-figure="functors"]').click();
  await functors;
  await settled();
  const opened = [...js.values()].reduce((sum, n) => sum + n, 0);
  expect(opened, "JavaScript gzipped: arrival, B, C and D").toBeLessThanOrEqual(64 * 1024);
  // DeepSeek's views load when the model switch first chooses it.
  const before = new Set(js.keys());
  const deepseek = chunk("deepseek");
  await win.locator('.ink-models button[data-model="dsv41"]').click();
  await deepseek;
  await settled();
  const switched = [...js].filter(([url]) => !before.has(url)).reduce((sum, [, n]) => sum + n, 0);
  expect(switched, "DeepSeek's chunk gzipped").toBeGreaterThan(0);
  expect(switched, "DeepSeek's chunk gzipped").toBeLessThanOrEqual(14 * 1024);
  test.info().annotations.push({
    type: "gzipped JavaScript",
    description: `arrival, B, C and D: ${opened} bytes; DeepSeek's chunk: ${switched} bytes`,
  });
});
