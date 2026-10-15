import { expect, type Page } from "@playwright/test";

export async function waitForStablePage(page: Page): Promise<void> {
  await page.waitForLoadState("networkidle");
  await page.evaluate(async () => {
    await document.fonts.ready;
  });
}

export async function expectNoHorizontalOverflow(page: Page, context: string): Promise<void> {
  const geometry = await page.evaluate(() => {
    const root = document.documentElement;
    const width = root.clientWidth;
    const offenders = Array.from(document.querySelectorAll<HTMLElement>("body *"))
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          selector:
            element.id && !element.id.includes(" ")
              ? `#${element.id}`
              : `${element.tagName.toLowerCase()}.${Array.from(element.classList).join(".")}`,
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
        };
      })
      .filter(
        ({ left, right, width: elementWidth }) =>
          elementWidth > 1 && (left < -1 || right > width + 1),
      )
      .slice(0, 8);
    return { clientWidth: width, scrollWidth: root.scrollWidth, offenders };
  });

  expect(
    geometry.scrollWidth,
    `${context} overflows horizontally: ${JSON.stringify(geometry)}`,
  ).toBeLessThanOrEqual(geometry.clientWidth + 1);
}

export async function expectSemanticHeadingOrder(page: Page, context: string): Promise<void> {
  const headings = await page.locator("h1, h2, h3, h4, h5, h6").evaluateAll((elements) =>
    elements.map((element) => ({
      level: Number(element.tagName.slice(1)),
      text: element.textContent?.trim().replace(/\s+/g, " ").slice(0, 80) ?? "",
    })),
  );

  expect(headings.length, `${context} needs a heading`).toBeGreaterThan(0);
  expect(headings[0]?.level, `${context} must begin its hierarchy at h1`).toBe(1);
  for (let index = 1; index < headings.length; index += 1) {
    const previous = headings[index - 1]!;
    const current = headings[index]!;
    expect(
      current.level,
      `${context} skips from h${previous.level} “${previous.text}” to h${current.level} “${current.text}”`,
    ).toBeLessThanOrEqual(previous.level + 1);
  }
}
