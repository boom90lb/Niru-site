import {
  PREVIEW_AGE_ACKNOWLEDGEMENT_VALUE,
  PREVIEW_FIELD_LIMITS,
  PREVIEW_FORM_FIELD_NAMES,
  PREVIEW_HARDWARE_ENVELOPES,
  PREVIEW_MAX_BODY_BYTES,
  PREVIEW_OUTCOMES,
  PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION,
  PREVIEW_READINESS_VERSION,
  PREVIEW_RETENTION_DAYS,
  PREVIEW_ROLES,
  PREVIEW_SOURCES,
  PREVIEW_SUBMITTED_PATH,
  PREVIEW_INVALID_PATH,
  PREVIEW_TEMPORARY_ERROR_PATH,
  type PreviewHardwareEnvelope,
  type PreviewOutcome,
  type PreviewRole,
  type PreviewSource,
} from "../../src/lib/preview-contract.ts";

const MAILERLITE_SUBSCRIBERS_ENDPOINT = "https://connect.mailerlite.com/api/subscribers";
const VENDOR_TIMEOUT_MS = 5_000;

const ROLE_VALUES = new Set(PREVIEW_ROLES.map(({ value }) => value));
const HARDWARE_VALUES = new Set(PREVIEW_HARDWARE_ENVELOPES.map(({ value }) => value));
const OUTCOME_VALUES = new Set(PREVIEW_OUTCOMES.map(({ value }) => value));
const SOURCE_VALUES = new Set(PREVIEW_SOURCES);
const ALLOWED_FORM_FIELDS: ReadonlySet<string> = new Set(Object.values(PREVIEW_FORM_FIELD_NAMES));

export interface PreviewEnvironment {
  PREVIEW_DEPLOYMENT_ENVIRONMENT?: string;
  PREVIEW_PUBLIC_ORIGIN?: string;
  PREVIEW_WRITES_ENABLED?: string;
  PREVIEW_READINESS_VERSION?: string;
  MAILERLITE_API_TOKEN?: string;
  MAILERLITE_PREVIEW_GROUP_ID?: string;
}

export interface PreviewDependencies {
  fetch?: typeof globalThis.fetch;
  now?: () => Date;
  vendorTimeoutMs?: number;
}

export interface ValidatedPreviewApplication {
  email: string;
  name?: string;
  organization?: string;
  role: PreviewRole;
  useCase: string;
  hardwareEnvelope: PreviewHardwareEnvelope;
  outcome: PreviewOutcome;
}

export interface MailerLitePreviewPayload {
  email: string;
  fields: Record<string, string>;
  groups: [string];
}

type ParsedForm = { ok: true; values: ReadonlyMap<string, string> } | { ok: false };

type ValidationResult = { ok: true; application: ValidatedPreviewApplication } | { ok: false };

function deploymentEnvironment(env: PreviewEnvironment): "production" | "preview" | "local" {
  if (env.PREVIEW_DEPLOYMENT_ENVIRONMENT === "production") {
    return "production";
  }
  if (env.PREVIEW_DEPLOYMENT_ENVIRONMENT === "preview") {
    return "preview";
  }
  return "local";
}

function configuredPublicOrigin(value: string | undefined): string | null {
  if (!value) {
    return null;
  }

  try {
    const url = new URL(value);
    if (
      url.protocol !== "https:" ||
      url.username ||
      url.password ||
      url.pathname !== "/" ||
      url.search ||
      url.hash ||
      value !== url.origin
    ) {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

function plainResponse(status: number, message: string): Response {
  return new Response(message, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "text/plain; charset=utf-8",
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
      ...(status === 405 ? { Allow: "POST" } : {}),
    },
  });
}

function stateRedirect(requestUrl: string, path: string): Response {
  const destination = new URL(path, requestUrl);
  destination.search = "";
  destination.hash = "";

  return new Response(null, {
    status: 303,
    headers: {
      "Cache-Control": "no-store",
      Location: destination.toString(),
      "Referrer-Policy": "no-referrer",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function invalidRequest(request: Request): Response {
  return stateRedirect(request.url, PREVIEW_INVALID_PATH);
}

function temporaryFailure(request: Request): Response {
  return stateRedirect(request.url, PREVIEW_TEMPORARY_ERROR_PATH);
}

function hasExpectedContentType(request: Request): boolean {
  const contentType = request.headers.get("Content-Type") ?? "";
  return /^application\/x-www-form-urlencoded(?:\s*;\s*charset=(?:utf-8|"utf-8"))?$/iu.test(
    contentType,
  );
}

function hasExpectedOrigin(request: Request, expectedOrigin: string): boolean {
  try {
    const requestOrigin = new URL(request.url).origin;
    const suppliedOrigin = request.headers.get("Origin");
    if (!suppliedOrigin) {
      return false;
    }
    const originUrl = new URL(suppliedOrigin);
    return (
      requestOrigin === expectedOrigin &&
      suppliedOrigin === originUrl.origin &&
      originUrl.origin === expectedOrigin
    );
  } catch {
    return false;
  }
}

async function readBoundedBody(request: Request, maximumBytes: number): Promise<string | null> {
  const contentLength = request.headers.get("Content-Length");
  if (contentLength) {
    if (!/^\d+$/u.test(contentLength) || Number(contentLength) > maximumBytes) {
      return null;
    }
  }

  if (!request.body) {
    return "";
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      totalBytes += value.byteLength;
      if (totalBytes > maximumBytes) {
        await reader.cancel();
        return null;
      }
      chunks.push(value);
    }
  } catch {
    return null;
  }

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

function parseForm(body: string): ParsedForm {
  const params = new URLSearchParams(body);
  const values = new Map<string, string>();

  for (const [name, value] of params) {
    if (
      !ALLOWED_FORM_FIELDS.has(name) ||
      values.has(name) ||
      name.includes("\uFFFD") ||
      value.includes("\uFFFD")
    ) {
      return { ok: false };
    }
    values.set(name, value);
  }

  return { ok: true, values };
}

function normalizeSingleLine(value: string, maximumLength: number): string | null {
  const normalized = value.normalize("NFC").trim();
  if (normalized.length > maximumLength || /[\u0000-\u001F\u007F\uFFFD]/u.test(normalized)) {
    return null;
  }
  return normalized;
}

function normalizeMultiline(value: string, maximumLength: number): string | null {
  const normalized = value.replace(/\r\n?/gu, "\n").normalize("NFC").trim();
  if (
    normalized.length > maximumLength ||
    /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\uFFFD]/u.test(normalized)
  ) {
    return null;
  }
  return normalized;
}

function validEmail(value: string): boolean {
  if (value.length === 0 || value.length > PREVIEW_FIELD_LIMITS.email) {
    return false;
  }
  const at = value.lastIndexOf("@");
  if (at <= 0 || at !== value.indexOf("@") || at === value.length - 1) {
    return false;
  }
  const domain = value.slice(at + 1);
  return (
    domain.includes(".") && !domain.startsWith(".") && !domain.endsWith(".") && !/\s/u.test(value)
  );
}

function validateApplication(values: ReadonlyMap<string, string>): ValidationResult {
  const email = normalizeSingleLine(
    values.get(PREVIEW_FORM_FIELD_NAMES.email) ?? "",
    PREVIEW_FIELD_LIMITS.email,
  );
  const name = normalizeSingleLine(
    values.get(PREVIEW_FORM_FIELD_NAMES.name) ?? "",
    PREVIEW_FIELD_LIMITS.name,
  );
  const organization = normalizeSingleLine(
    values.get(PREVIEW_FORM_FIELD_NAMES.organization) ?? "",
    PREVIEW_FIELD_LIMITS.organization,
  );
  const useCase = normalizeMultiline(
    values.get(PREVIEW_FORM_FIELD_NAMES.useCase) ?? "",
    PREVIEW_FIELD_LIMITS.useCase,
  );
  const role = values.get(PREVIEW_FORM_FIELD_NAMES.role) ?? "";
  const hardwareEnvelope = values.get(PREVIEW_FORM_FIELD_NAMES.hardwareEnvelope) ?? "";
  const outcome = values.get(PREVIEW_FORM_FIELD_NAMES.outcome) ?? "";

  if (
    email === null ||
    !validEmail(email) ||
    name === null ||
    organization === null ||
    useCase === null ||
    useCase.length === 0 ||
    !ROLE_VALUES.has(role as PreviewRole) ||
    !HARDWARE_VALUES.has(hardwareEnvelope as PreviewHardwareEnvelope) ||
    !OUTCOME_VALUES.has(outcome as PreviewOutcome) ||
    values.get(PREVIEW_FORM_FIELD_NAMES.processingAcknowledgement) !==
      PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION ||
    values.get(PREVIEW_FORM_FIELD_NAMES.ageAcknowledgement) !== PREVIEW_AGE_ACKNOWLEDGEMENT_VALUE
  ) {
    return { ok: false };
  }

  return {
    ok: true,
    application: {
      email,
      ...(name ? { name } : {}),
      ...(organization ? { organization } : {}),
      role: role as PreviewRole,
      useCase,
      hardwareEnvelope: hardwareEnvelope as PreviewHardwareEnvelope,
      outcome: outcome as PreviewOutcome,
    },
  };
}

export function derivePreviewSource(
  referer: string | null,
  expectedOrigin: string,
): PreviewSource {
  if (!referer) {
    return "direct_or_unknown";
  }

  try {
    const refererUrl = new URL(referer);
    if (
      refererUrl.origin !== expectedOrigin ||
      refererUrl.pathname !== "/preview/" ||
      refererUrl.username ||
      refererUrl.password ||
      refererUrl.hash
    ) {
      return "direct_or_unknown";
    }
    const sourceValues = refererUrl.searchParams.getAll("src");
    if (sourceValues.length !== 1) {
      return "direct_or_unknown";
    }
    const source = sourceValues[0];
    return SOURCE_VALUES.has(source as PreviewSource)
      ? (source as PreviewSource)
      : "direct_or_unknown";
  } catch {
    return "direct_or_unknown";
  }
}

export function buildMailerLitePayload(
  application: ValidatedPreviewApplication,
  groupId: string,
  source: PreviewSource,
  receivedAt: Date,
): MailerLitePreviewPayload {
  const retentionDueAt = new Date(
    receivedAt.getTime() + PREVIEW_RETENTION_DAYS * 24 * 60 * 60 * 1_000,
  );
  const fields: Record<string, string> = {
    preview_role: application.role,
    preview_use_case: application.useCase,
    preview_hardware_envelope: application.hardwareEnvelope,
    preview_outcome: application.outcome,
    preview_processing_acknowledgement_version: PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION,
    preview_age_eligibility_confirmed: "yes",
    preview_received_at: receivedAt.toISOString(),
    preview_source: source,
    preview_state: "received",
    preview_retention_due_at: retentionDueAt.toISOString(),
  };

  if (application.name) {
    fields.preview_name = application.name;
  }
  if (application.organization) {
    fields.preview_organization = application.organization;
  }

  return {
    email: application.email,
    fields,
    groups: [groupId],
  };
}

function validProductionConfiguration(env: PreviewEnvironment): boolean {
  return Boolean(
    env.PREVIEW_WRITES_ENABLED === "true" &&
    env.PREVIEW_READINESS_VERSION === PREVIEW_READINESS_VERSION &&
    env.MAILERLITE_API_TOKEN?.trim() &&
    env.MAILERLITE_PREVIEW_GROUP_ID?.trim() &&
    /^[A-Za-z0-9_-]{1,128}$/u.test(env.MAILERLITE_PREVIEW_GROUP_ID),
  );
}

async function submitToMailerLite(
  payload: MailerLitePreviewPayload,
  env: PreviewEnvironment,
  dependencies: PreviewDependencies,
): Promise<boolean> {
  const fetcher = dependencies.fetch ?? globalThis.fetch;
  const controller = new AbortController();
  const configuredTimeout = dependencies.vendorTimeoutMs ?? VENDOR_TIMEOUT_MS;
  const timeoutMs = Math.max(1, Math.min(configuredTimeout, VENDOR_TIMEOUT_MS));
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetcher(MAILERLITE_SUBSCRIBERS_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${env.MAILERLITE_API_TOKEN ?? ""}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      redirect: "manual",
      signal: controller.signal,
    });
    return response.status === 200 || response.status === 201;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

export async function handlePreviewRequest(
  request: Request,
  env: PreviewEnvironment,
  dependencies: PreviewDependencies = {},
): Promise<Response> {
  if (request.method !== "POST") {
    return plainResponse(405, "Method not allowed.");
  }

  const environment = deploymentEnvironment(env);
  const requestOrigin = new URL(request.url).origin;
  const configuredOrigin = configuredPublicOrigin(env.PREVIEW_PUBLIC_ORIGIN);
  if (
    (env.PREVIEW_PUBLIC_ORIGIN && !configuredOrigin) ||
    (environment === "production" && !configuredOrigin)
  ) {
    return temporaryFailure(request);
  }
  const expectedOrigin = configuredOrigin ?? requestOrigin;

  if (!hasExpectedOrigin(request, expectedOrigin) || !hasExpectedContentType(request)) {
    return invalidRequest(request);
  }

  const body = await readBoundedBody(request, PREVIEW_MAX_BODY_BYTES);
  if (body === null) {
    return invalidRequest(request);
  }
  const parsed = parseForm(body);
  if (!parsed.ok) {
    return invalidRequest(request);
  }

  const honeypot = parsed.values.get(PREVIEW_FORM_FIELD_NAMES.honeypot) ?? "";
  if (honeypot.trim()) {
    return stateRedirect(request.url, PREVIEW_SUBMITTED_PATH);
  }

  const validated = validateApplication(parsed.values);
  if (!validated.ok) {
    return invalidRequest(request);
  }

  if (environment !== "production") {
    return stateRedirect(request.url, PREVIEW_TEMPORARY_ERROR_PATH);
  }

  if (!validProductionConfiguration(env)) {
    return temporaryFailure(request);
  }

  const now = dependencies.now?.() ?? new Date();
  if (!Number.isFinite(now.getTime())) {
    return temporaryFailure(request);
  }
  const source = derivePreviewSource(request.headers.get("Referer"), expectedOrigin);
  const payload = buildMailerLitePayload(
    validated.application,
    env.MAILERLITE_PREVIEW_GROUP_ID as string,
    source,
    now,
  );

  const accepted = await submitToMailerLite(payload, env, dependencies);
  return accepted ? stateRedirect(request.url, PREVIEW_SUBMITTED_PATH) : temporaryFailure(request);
}
