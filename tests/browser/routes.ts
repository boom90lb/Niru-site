export interface CoreRoute {
  path: string;
  slug: string;
  kind: "content" | "result";
}

export const CORE_ROUTES: readonly CoreRoute[] = [
  { path: "/", slug: "home", kind: "content" },
  { path: "/lab/", slug: "lab", kind: "content" },
  { path: "/research/", slug: "research", kind: "content" },
  { path: "/research/compute-fairness/", slug: "research-compute-fairness", kind: "content" },
  { path: "/research/three-expansions/", slug: "research-three-expansions", kind: "content" },
  {
    path: "/research/spectral-diagnostics/",
    slug: "research-spectral-diagnostics",
    kind: "content",
  },
  { path: "/about/", slug: "about", kind: "content" },
  { path: "/preview/", slug: "preview", kind: "content" },
  { path: "/privacy/", slug: "privacy", kind: "content" },
  { path: "/preview/submitted/", slug: "preview-submitted", kind: "result" },
  { path: "/preview/invalid/", slug: "preview-invalid", kind: "result" },
  { path: "/preview/temporary-error/", slug: "preview-temporary-error", kind: "result" },
] as const;

export const RESULT_ROUTES = CORE_ROUTES.filter((route) => route.kind === "result");

export const SCREENSHOT_ROUTES = CORE_ROUTES.filter((route) =>
  ["home", "research-spectral-diagnostics", "about", "preview"].includes(route.slug),
);
