/**
 * Public, non-secret contract for AMLO technical-preview applications.
 *
 * The browser form and the same-origin processing boundary both consume these
 * values. Changing a field, option, acknowledgement, or retention rule also
 * requires a privacy and operations review.
 */

export const PREVIEW_PROCESSING_ACKNOWLEDGEMENT_VERSION =
  "preview-processing-notice-2026-08-16-v1";

export const PREVIEW_AGE_ACKNOWLEDGEMENT_VALUE = "eligible";

/**
 * This exact value may be configured only after the production-readiness
 * checklist for this notice/boundary version has been signed off outside Git.
 */
export const PREVIEW_READINESS_VERSION = "preview-readiness-2026-08-16-v1";

export const PREVIEW_GROUP_NAME = "AMLO Technical Preview — Applications";

export const PREVIEW_RETENTION_DAYS = 90;
export const PREVIEW_MAX_BODY_BYTES = 16_384;

export const PREVIEW_FIELD_LIMITS = Object.freeze({
  email: 254,
  name: 100,
  organization: 160,
  useCase: 1_000,
});

export const PREVIEW_FORM_FIELD_NAMES = Object.freeze({
  email: "email",
  name: "name",
  organization: "organization",
  role: "role",
  useCase: "use_case",
  hardwareEnvelope: "hardware_envelope",
  outcome: "preview_outcome",
  processingAcknowledgement: "processing_acknowledgement",
  ageAcknowledgement: "age_acknowledgement",
  honeypot: "website",
} as const);

export const PREVIEW_ROLES = Object.freeze([
  { value: "researcher", label: "Researcher" },
  { value: "ml_engineer", label: "ML engineer" },
  {
    value: "platform_infrastructure_engineer",
    label: "Platform/infrastructure engineer",
  },
  { value: "lab_team_lead", label: "Lab/team lead" },
  { value: "other", label: "Other" },
] as const);

export const PREVIEW_HARDWARE_ENVELOPES = Object.freeze([
  { value: "cpu_exploration", label: "CPU exploration" },
  { value: "single_gpu_up_to_12gb", label: "One GPU up to 12 GB" },
  { value: "single_gpu_over_12gb", label: "One GPU over 12 GB" },
  { value: "multi_gpu_single_node", label: "Multi-GPU, single node" },
  {
    value: "multi_node_distributed_planning",
    label: "Multi-node/distributed planning",
  },
  { value: "other_not_sure", label: "Other or not sure" },
] as const);

export const PREVIEW_OUTCOMES = Object.freeze([
  {
    value: "evaluate_single_gpu_reference",
    label: "Evaluate the reference path",
  },
  {
    value: "follow_distributed_validation",
    label: "Follow the research program as it develops",
  },
  {
    value: "lab_platform_evaluation",
    label: "Discuss a lab or platform evaluation",
  },
  {
    value: "contribute_research_collaboration",
    label: "Contribute or discuss research collaboration",
  },
] as const);

export const PREVIEW_SOURCES = Object.freeze([
  "amlo_github",
  "linkedin",
  "research_or_notes",
  "event",
  "partner",
  "other_campaign",
  "direct_or_unknown",
] as const);

export type PreviewRole = (typeof PREVIEW_ROLES)[number]["value"];
export type PreviewHardwareEnvelope = (typeof PREVIEW_HARDWARE_ENVELOPES)[number]["value"];
export type PreviewOutcome = (typeof PREVIEW_OUTCOMES)[number]["value"];
export type PreviewSource = (typeof PREVIEW_SOURCES)[number];

export const PREVIEW_SUBMITTED_PATH = "/preview/submitted/";
export const PREVIEW_INVALID_PATH = "/preview/invalid/";
export const PREVIEW_TEMPORARY_ERROR_PATH = "/preview/temporary-error/";
