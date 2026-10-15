import { expect, test } from "@playwright/test";

test("the phone home page starts with a fitted drawing overview and loads details only on request", async ({
  page,
}) => {
  test.skip((page.viewportSize()?.width ?? 1440) > 640, "Phone-specific behavior");
  const optional: string[] = [];
  page.on("request", (request) => {
    if (/\/(film|fam|functors|deepseek)\.[\w-]+\.js$/.test(request.url()))
      optional.push(request.url());
  });
  await page.goto("/", { waitUntil: "networkidle" });
  const win = page.locator("[data-ink-window]");
  await win.scrollIntoViewIfNeeded();
  await expect(win.locator("canvas")).toBeHidden();
  expect(await win.locator(".ink-stage").evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
    true,
  );
  expect(optional).toEqual([]);
  await win.getByRole("button", { name: "Open the interactive drawing" }).click();
  await expect(win.locator("canvas")).toBeVisible();
  await win.locator('button[data-figure="family"]').click();
  await expect(win.locator(".ink-counts")).toBeVisible();
  await expect(win.locator(".ink-counts")).toContainText("parameters");
});

test("an unavailable optional drawing offers visible recovery", async ({ page }) => {
  test.skip(
    (page.viewportSize()?.width ?? 0) !== 1440,
    "One desktop error-injection pass per color scheme",
  );
  await page.route(/\/deepseek\.[\w-]+\.js$/, (route) => route.abort());
  await page.goto("/");
  const win = page.locator("[data-ink-window]");
  // In view, the window loads and plays its film; a model click during that load is dropped.
  await win.scrollIntoViewIfNeeded();
  await expect(win.locator(".ink-steps")).toBeVisible();
  await win.locator('button[data-model="dsv41"]').click();
  await expect(win.locator(".ink-load-error")).toBeVisible();
  await expect(win.getByRole("button", { name: "Reload page" })).toBeVisible();
  await page.unroute(/\/deepseek\.[\w-]+\.js$/);
  await win.getByRole("button", { name: "Reload page" }).click();
  await win.locator('button[data-model="dsv41"]').click();
  await expect(win.locator('button[data-model="dsv41"]')).toHaveAttribute("aria-pressed", "true");
  await expect(win.locator(".ink-load-error")).toBeHidden();
});

test("the opening, the status and the descent table read without JavaScript", async ({
  browser,
  baseURL,
}) => {
  if (!baseURL) throw new Error("Browser verification requires its configured base URL");
  const context = await browser.newContext({ javaScriptEnabled: false, baseURL });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "Thirty million hours, or one",
  );
  await expect(page.getByText("AMLO is a research prototype.", { exact: false })).toBeVisible();
  await expect(page.locator(".descent-table tbody tr")).toHaveCount(9);
  await expect(page.locator(".descent-step--hour")).toHaveCount(1);
  await context.close();
});

test("the static tablet drawing is keyboard scrollable without JavaScript", async ({
  browser,
  baseURL,
}, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-768", "One no-JavaScript tablet pass");
  if (!baseURL) throw new Error("Browser verification requires its configured base URL");
  const context = await browser.newContext({
    javaScriptEnabled: false,
    baseURL,
    viewport: { width: 800, height: 900 },
  });
  const page = await context.newPage();
  await page.goto("/");
  const stage = page.locator(".ink-stage");
  await expect(stage).toHaveAttribute("tabindex", "0");
  await stage.focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => stage.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
  await context.close();
});
