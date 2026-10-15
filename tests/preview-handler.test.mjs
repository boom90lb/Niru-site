import assert from "node:assert/strict";
import test from "node:test";

import {
  PREVIEW_AGE_ACKNOWLEDGEMENT_VALUE,
  PREVIEW_MAX_BODY_BYTES,
  PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION,
  PREVIEW_READINESS_VERSION,
} from "../src/lib/preview-contract.ts";
import { derivePreviewSource, handlePreviewRequest } from "../functions/_shared/preview-core.ts";

const ORIGIN = "https://niru.ml";
const ENDPOINT = `${ORIGIN}/api/preview`;
const FIXED_NOW = new Date("2026-08-16T18:30:00.000Z");

const baseValues = Object.freeze({
  email: "preview-test@example.com",
  name: "Test Researcher",
  organization: "Example Lab",
  role: "researcher",
  use_case: "Evaluate a bounded single-GPU reference.",
  hardware_envelope: "single_gpu_over_12gb",
  preview_outcome: "evaluate_single_gpu_reference",
  processing_acknowledgement: PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION,
  age_acknowledgement: PREVIEW_AGE_ACKNOWLEDGEMENT_VALUE,
  website: "",
});

const productionEnvironment = Object.freeze({
  PREVIEW_DEPLOYMENT_ENVIRONMENT: "production",
  PREVIEW_PUBLIC_ORIGIN: ORIGIN,
  PREVIEW_WRITES_ENABLED: "true",
  PREVIEW_READINESS_VERSION,
  MAILERLITE_API_TOKEN: "test-token-never-sent-to-a-vendor",
  MAILERLITE_PREVIEW_GROUP_ID: "test_group_123",
});

function encode(values = baseValues) {
  return new URLSearchParams(values).toString();
}

function requestFor(body = encode(), headers = {}, url = ENDPOINT) {
  return new Request(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Origin: ORIGIN,
      Referer: `${ORIGIN}/preview/?src=linkedin`,
      ...headers,
    },
    body,
  });
}

function assertRedirect(response, path) {
  assert.equal(response.status, 303);
  assert.equal(response.headers.get("location"), `${ORIGIN}${path}`);
  assert.equal(response.headers.get("cache-control"), "no-store");
  assert.equal(response.headers.get("referrer-policy"), "no-referrer");
}

test("non-POST requests fail without calling a vendor", async () => {
  let calls = 0;
  const response = await handlePreviewRequest(new Request(ENDPOINT), productionEnvironment, {
    fetch: async () => {
      calls += 1;
      return new Response(null, { status: 201 });
    },
  });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "POST");
  assert.equal(calls, 0);
});

test("origin, content type, size, unknown-field, and duplicate-field failures are generic", async (t) => {
  const cases = [
    ["missing origin", requestFor(encode(), { Origin: "" })],
    ["cross-origin", requestFor(encode(), { Origin: "https://attacker.invalid" })],
    ["wrong content type", requestFor(encode(), { "Content-Type": "application/json" })],
    [
      "declared oversized",
      requestFor(encode(), { "Content-Length": String(PREVIEW_MAX_BODY_BYTES + 1) }),
    ],
    ["actual oversized", requestFor(`use_case=${"x".repeat(PREVIEW_MAX_BODY_BYTES + 1)}`)],
    ["unknown field", requestFor(`${encode()}&utm_source=raw`)],
    ["duplicate field", requestFor(`${encode()}&email=second%40example.com`)],
  ];

  for (const [name, request] of cases) {
    await t.test(name, async () => {
      let calls = 0;
      const response = await handlePreviewRequest(request, productionEnvironment, {
        fetch: async () => {
          calls += 1;
          return new Response(null, { status: 201 });
        },
      });
      assertRedirect(response, "/preview/invalid/");
      assert.equal(calls, 0);
    });
  }
});

test("required values, enums, acknowledgements, and lengths are enforced", async (t) => {
  const invalidValues = [
    ["email", { ...baseValues, email: "not-an-email" }],
    ["email too long", { ...baseValues, email: `${"a".repeat(244)}@example.com` }],
    ["name too long", { ...baseValues, name: "n".repeat(101) }],
    ["organization too long", { ...baseValues, organization: "o".repeat(161) }],
    ["single-line control", { ...baseValues, organization: "Example\nLab" }],
    ["role", { ...baseValues, role: "founder" }],
    ["use case missing", { ...baseValues, use_case: "" }],
    ["use case too long", { ...baseValues, use_case: "x".repeat(1_001) }],
    ["hardware", { ...baseValues, hardware_envelope: "eight_h100s" }],
    ["outcome", { ...baseValues, preview_outcome: "guaranteed_access" }],
    ["notice", { ...baseValues, processing_acknowledgement: "yes" }],
    ["age", { ...baseValues, age_acknowledgement: "yes" }],
  ];

  for (const [name, values] of invalidValues) {
    await t.test(name, async () => {
      const response = await handlePreviewRequest(
        requestFor(encode(values)),
        productionEnvironment,
        { fetch: async () => new Response(null, { status: 201 }) },
      );
      assertRedirect(response, "/preview/invalid/");
    });
  }
});

test("local and preview modes cannot write even when production flags and secrets are present", async (t) => {
  for (const environment of [undefined, "local", "preview"]) {
    await t.test(environment ?? "unset", async () => {
      let calls = 0;
      const response = await handlePreviewRequest(
        requestFor(),
        {
          ...productionEnvironment,
          PREVIEW_DEPLOYMENT_ENVIRONMENT: environment,
        },
        {
          fetch: async () => {
            calls += 1;
            return new Response(null, { status: 201 });
          },
        },
      );
      assertRedirect(response, "/preview/temporary-error/");
      assert.equal(calls, 0);
    });
  }
});

test("a populated honeypot returns neutral success and never writes", async () => {
  let calls = 0;
  const response = await handlePreviewRequest(
    requestFor(encode({ ...baseValues, website: "https://bot.invalid" })),
    productionEnvironment,
    {
      fetch: async () => {
        calls += 1;
        return new Response(null, { status: 500 });
      },
    },
  );
  assertRedirect(response, "/preview/submitted/");
  assert.equal(calls, 0);
});

test("production requires every independent write gate", async (t) => {
  const gated = [
    ["missing origin binding", { PREVIEW_PUBLIC_ORIGIN: undefined }],
    ["origin has slash", { PREVIEW_PUBLIC_ORIGIN: `${ORIGIN}/` }],
    ["writes false", { PREVIEW_WRITES_ENABLED: "false" }],
    ["writes wrong case", { PREVIEW_WRITES_ENABLED: "TRUE" }],
    ["readiness missing", { PREVIEW_READINESS_VERSION: undefined }],
    ["readiness stale", { PREVIEW_READINESS_VERSION: "preview-readiness-old" }],
    ["token missing", { MAILERLITE_API_TOKEN: undefined }],
    ["group missing", { MAILERLITE_PREVIEW_GROUP_ID: undefined }],
    ["group malformed", { MAILERLITE_PREVIEW_GROUP_ID: "bad group id" }],
  ];

  for (const [name, changed] of gated) {
    await t.test(name, async () => {
      let calls = 0;
      const response = await handlePreviewRequest(
        requestFor(),
        { ...productionEnvironment, ...changed },
        {
          fetch: async () => {
            calls += 1;
            return new Response(null, { status: 201 });
          },
        },
      );
      assertRedirect(response, "/preview/temporary-error/");
      assert.equal(calls, 0);
    });
  }
});

test("an accepted vendor response receives only the approved additive payload", async () => {
  const calls = [];
  const response = await handlePreviewRequest(requestFor(), productionEnvironment, {
    now: () => FIXED_NOW,
    fetch: async (...args) => {
      calls.push(args);
      return new Response(null, { status: 201 });
    },
  });

  assertRedirect(response, "/preview/submitted/");
  assert.equal(calls.length, 1);
  const [url, options] = calls[0];
  assert.equal(url, "https://connect.mailerlite.com/api/subscribers");
  assert.equal(options.method, "POST");
  assert.equal(options.redirect, "manual");
  assert.equal(
    options.headers.Authorization,
    `Bearer ${productionEnvironment.MAILERLITE_API_TOKEN}`,
  );

  const payload = JSON.parse(options.body);
  assert.deepEqual(Object.keys(payload).sort(), ["email", "fields", "groups"]);
  assert.equal(payload.email, baseValues.email);
  assert.deepEqual(payload.groups, [productionEnvironment.MAILERLITE_PREVIEW_GROUP_ID]);
  assert.deepEqual(payload.fields, {
    preview_age_eligibility_confirmed: "yes",
    preview_hardware_envelope: baseValues.hardware_envelope,
    preview_name: baseValues.name,
    preview_organization: baseValues.organization,
    preview_outcome: baseValues.preview_outcome,
    preview_processing_acknowledgement_version: PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION,
    preview_received_at: FIXED_NOW.toISOString(),
    preview_retention_due_at: "2026-11-14T18:30:00.000Z",
    preview_role: baseValues.role,
    preview_source: "linkedin",
    preview_state: "received",
    preview_use_case: baseValues.use_case,
  });
  assert.equal("status" in payload, false);
  assert.equal("ip" in payload.fields, false);
  assert.equal(JSON.stringify(payload).includes("/preview/?src="), false);
});

test("optional name and organization are omitted when blank", async () => {
  let payload;
  const response = await handlePreviewRequest(
    requestFor(encode({ ...baseValues, name: "", organization: "" })),
    productionEnvironment,
    {
      now: () => FIXED_NOW,
      fetch: async (_url, options) => {
        payload = JSON.parse(options.body);
        return new Response(null, { status: 200 });
      },
    },
  );
  assertRedirect(response, "/preview/submitted/");
  assert.equal("preview_name" in payload.fields, false);
  assert.equal("preview_organization" in payload.fields, false);
});

test("only HTTP 200 and 201 are success and failures are never retried", async (t) => {
  for (const status of [200, 201, 202, 400, 500]) {
    await t.test(String(status), async () => {
      let calls = 0;
      const response = await handlePreviewRequest(requestFor(), productionEnvironment, {
        fetch: async () => {
          calls += 1;
          return new Response(null, { status });
        },
      });
      assertRedirect(
        response,
        status === 200 || status === 201 ? "/preview/submitted/" : "/preview/temporary-error/",
      );
      assert.equal(calls, 1);
    });
  }

  await t.test("network rejection", async () => {
    let calls = 0;
    const response = await handlePreviewRequest(requestFor(), productionEnvironment, {
      fetch: async () => {
        calls += 1;
        throw new Error("synthetic network failure");
      },
    });
    assertRedirect(response, "/preview/temporary-error/");
    assert.equal(calls, 1);
  });
});

test("vendor timeout fails closed without a retry", async () => {
  let calls = 0;
  const response = await handlePreviewRequest(requestFor(), productionEnvironment, {
    vendorTimeoutMs: 5,
    fetch: async (_url, options) => {
      calls += 1;
      return await new Promise((_resolve, reject) => {
        options.signal.addEventListener(
          "abort",
          () => reject(new DOMException("Aborted", "AbortError")),
          { once: true },
        );
      });
    },
  });
  assertRedirect(response, "/preview/temporary-error/");
  assert.equal(calls, 1);
});

test("source mapping stores an allowlisted enum and discards raw or foreign values", () => {
  assert.equal(derivePreviewSource(`${ORIGIN}/preview/?src=event`, ORIGIN), "event");
  assert.equal(
    derivePreviewSource(`${ORIGIN}/preview/?src=unknown-campaign`, ORIGIN),
    "direct_or_unknown",
  );
  assert.equal(
    derivePreviewSource(`${ORIGIN}/preview/?src=event&src=partner`, ORIGIN),
    "direct_or_unknown",
  );
  assert.equal(
    derivePreviewSource("https://attacker.invalid/?src=linkedin", ORIGIN),
    "direct_or_unknown",
  );
  assert.equal(derivePreviewSource(`${ORIGIN}/privacy/?src=partner`, ORIGIN), "direct_or_unknown");
  assert.equal(derivePreviewSource(null, ORIGIN), "direct_or_unknown");
});
