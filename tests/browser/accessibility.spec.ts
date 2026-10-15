import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { expectNoHorizontalOverflow, expectSemanticHeadingOrder } from "./helpers";
import { CORE_ROUTES } from "./routes";

for (const route of CORE_ROUTES) {
  test(`${route.path} has no automated WCAG 2.1 A/AA violations`, async ({ page }) => {
    await page.goto(route.path);
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();

    expect(
      results.violations,
      results.violations
        .map(
          (violation) =>
            `${violation.id}: ${violation.help}\n${violation.nodes
              .map((node) => `  ${node.target.join(" ")}: ${node.failureSummary}`)
              .join("\n")}`,
        )
        .join("\n"),
    ).toEqual([]);
  });

  test(`${route.path} has one coherent document structure`, async ({ page }) => {
    const response = await page.goto(route.path);
    expect(response?.ok(), `${route.path} did not return a successful response`).toBe(true);
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("main#main-content")).toHaveCount(1);
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toHaveCount(1);
    await expect(page.locator("h1")).toHaveCount(1);
    await expectSemanticHeadingOrder(page, route.path);

    for (const image of await page.locator("img").all()) {
      expect((await image.getAttribute("alt"))?.trim().length).toBeGreaterThan(0);
    }
  });

  test(`${route.path} reflows without horizontal page scrolling`, async ({ page }) => {
    await page.goto(route.path);
    await expectNoHorizontalOverflow(page, `${route.path} at project width`);
  });
}
