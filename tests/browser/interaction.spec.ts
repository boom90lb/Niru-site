import { expect, test } from "@playwright/test";

test("the skip link is first, visible on focus, and moves focus to main", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: "Skip to content" });
  await expect(skipLink).toBeFocused();
  await expect(skipLink).toBeInViewport();

  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main-content$/);
  await expect(page.locator("#main-content")).toBeFocused();
});

test("all tab stops are reachable in DOM order with a visible focus indicator", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "chromium-1440", "One desktop keyboard pass is sufficient");
  await page.goto("/preview/");

  const expectedOrder = await page.evaluate(() => {
    const candidates = Array.from(
      document.querySelectorAll<HTMLElement>(
        "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]",
      ),
    ).filter((element) => {
      if (element.tabIndex < 0) return false;
      const style = getComputedStyle(element);
      return style.display !== "none" && style.visibility !== "hidden";
    });
    return candidates.map((element, index) => {
      element.dataset.browserTabOrder = String(index);
      return String(index);
    });
  });

  const actualOrder: string[] = [];
  for (let index = 0; index < expectedOrder.length; index += 1) {
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    actualOrder.push((await focused.getAttribute("data-browser-tab-order")) ?? "missing");
    const style = await focused.evaluate((element) => {
      const computed = getComputedStyle(element);
      return {
        outlineStyle: computed.outlineStyle,
        outlineWidth: Number.parseFloat(computed.outlineWidth),
      };
    });
    expect(
      style.outlineStyle !== "none" && style.outlineWidth >= 1,
      `Tab stop ${index + 1} lacks a visible outline: ${await focused.evaluate((element) => element.outerHTML)}`,
    ).toBe(true);
  }
  expect(actualOrder).toEqual(expectedOrder);
});

test("primary touch controls expose at least a 44 by 44 CSS pixel target", async ({ page }) => {
  for (const path of ["/", "/research/", "/preview/"]) {
    await page.goto(path);
    const targets = page.locator(
      [
        ".site-header a",
        ".site-footer nav a",
        ".button",
        ".note-list a",
        ".preview-form input:not([type='checkbox']):not([tabindex='-1'])",
        ".preview-form select",
        ".preview-form textarea",
        ".preview-form button",
        ".preview-form .check-row",
      ].join(", "),
    );
    const undersized: string[] = [];
    for (const target of await targets.all()) {
      if (!(await target.isVisible())) continue;
      const box = await target.boundingBox();
      if (!box || (box.width >= 44 && box.height >= 44)) continue;
      undersized.push(
        `${await target.evaluate((element) => element.outerHTML.slice(0, 80))} ${box.width}x${box.height}`,
      );
    }
    expect(undersized, `${path} has undersized touch targets`).toEqual([]);
  }
});
