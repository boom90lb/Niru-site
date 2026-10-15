import { test } from "@playwright/test";
import { waitForStablePage } from "./helpers";
import { SCREENSHOT_ROUTES } from "./routes";

for (const route of SCREENSHOT_ROUTES) {
  test(`${route.path} representative screenshot`, async ({ page }, testInfo) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(route.path);
    await waitForStablePage(page);
    await page.screenshot({
      path: testInfo.outputPath(
        `screenshots/${route.slug}-${testInfo.project.name.replace("chromium-", "")}.png`,
      ),
      fullPage: true,
      animations: "disabled",
    });
  });
}
