import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:4417";

export default defineConfig({
  testDir: "./tests/browser",
  outputDir: "./test-results",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 4,
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "line",
  expect: {
    timeout: 5_000,
  },
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    locale: "en-US",
    colorScheme: "light",
    serviceWorkers: "block",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium-320",
      use: { viewport: { width: 320, height: 760 } },
    },
    {
      name: "chromium-768",
      use: { viewport: { width: 768, height: 900 } },
    },
    {
      name: "chromium-1440",
      use: { viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "chromium-1440-dark",
      use: { viewport: { width: 1440, height: 1000 }, colorScheme: "dark" },
    },
  ],
  ...(process.env.PLAYWRIGHT_BASE_URL
    ? {}
    : {
        webServer: {
          command:
            "node ./node_modules/astro/bin/astro.mjs build && python3 -m http.server 4417 --bind 127.0.0.1 --directory dist 2>/dev/null",
          url: baseURL,
          reuseExistingServer: false,
          timeout: 30_000,
        },
      }),
});
