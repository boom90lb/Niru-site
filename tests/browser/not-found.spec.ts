import { expect, test } from "@playwright/test";
import { expectNoHorizontalOverflow, expectSemanticHeadingOrder } from "./helpers";

test("the static 404 is a branded, recoverable route", async ({ page }) => {
  const response = await page.goto("/404.html");

  expect(response?.ok()).toBe(true);
  await expect(page).toHaveTitle("Page not found - Niru ML");
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute(
    "content",
    "noindex,nofollow,noarchive",
  );
  await expect(
    page.getByRole("heading", { level: 1, name: "This page does not exist." }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "Return home" })).toHaveAttribute("href", "/");
  await expect(
    page
      .getByRole("navigation", { name: "Primary navigation" })
      .getByRole("link", { name: "About" }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("navigation", { name: "Primary navigation" })
      .getByRole("link", { name: "Research" }),
  ).toBeVisible();
  await expectSemanticHeadingOrder(page, "/404.html");
  await expectNoHorizontalOverflow(page, "/404.html at project width");
});
