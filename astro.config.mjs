import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://niru.ml",
  output: "static",
  trailingSlash: "always",
  build: {
    format: "directory",
  },
  vite: {
    build: {
      // Keep every font as a same-origin file. Inlined data-font URLs would
      // silently exceed the deliberately narrow `font-src 'self'` contract.
      assetsInlineLimit: 0,
    },
  },
  server: {
    host: "127.0.0.1",
  },
});
