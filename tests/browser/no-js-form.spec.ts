import { expect, test } from "@playwright/test";

test.use({ javaScriptEnabled: false });

test("contact is honest and usable without JavaScript or a form endpoint", async ({ page }) => {
  const posts: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") posts.push(request.url());
  });
  await page.goto("/preview/");
  await expect(
    page.getByRole("heading", { name: "Arrange a conversation", level: 1 }),
  ).toBeVisible();
  await expect(page.locator("form")).toHaveCount(0);
  await expect(page.locator(".conversation .button")).toHaveAttribute(
    "href",
    "mailto:brendonrt@niru.ml?subject=Conversation%20about%20Niru%20ML",
  );
  await expect(
    page.getByText("AMLO is private and under development.", { exact: false }),
  ).toBeVisible();
  expect(posts).toEqual([]);
});
