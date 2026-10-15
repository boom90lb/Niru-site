"""Parse Niru's Cloudflare `_headers` contract and compare a route's headers against it."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

PUBLIC_ROUTES = (
    "/",
    "/lab/",
    "/research/",
    "/research/compute-fairness/",
    "/research/three-expansions/",
    "/research/spectral-diagnostics/",
    "/about/",
    "/preview/",
    "/privacy/",
)
RESULT_ROUTES = (
    "/preview/submitted/",
    "/preview/invalid/",
    "/preview/temporary-error/",
)
CHECKED_ROUTES = PUBLIC_ROUTES + RESULT_ROUTES

REQUIRED_SECURITY_HEADERS = (
    "content-security-policy",
    "cross-origin-opener-policy",
    "cross-origin-resource-policy",
    "permissions-policy",
    "referrer-policy",
    "strict-transport-security",
    "x-content-type-options",
    "x-frame-options",
)


class HeaderContractError(ValueError):
    """Raised when a `_headers` file cannot be parsed unambiguously."""


@dataclass(frozen=True)
class HeaderRule:
    pattern: str
    headers: Mapping[str, str]


def parse_headers_contract(text: str) -> tuple[HeaderRule, ...]:
    """Parse the small subset of Cloudflare `_headers` syntax the site uses."""

    rules: list[HeaderRule] = []
    pattern: str | None = None
    headers: dict[str, str] = {}

    def finish() -> None:
        nonlocal pattern, headers
        if pattern is not None:
            rules.append(HeaderRule(pattern, dict(headers)))
        pattern, headers = None, {}

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if raw[:1].isspace():
            if pattern is None or ":" not in line:
                raise HeaderContractError(f"line {number}: header outside a path rule")
            name, value = (part.strip() for part in line.split(":", 1))
            if not name or not value or name.lower() in headers:
                raise HeaderContractError(f"line {number}: empty or duplicate header")
            headers[name.lower()] = value
            continue
        finish()
        if not line.startswith("/") or any(rule.pattern == line for rule in rules):
            raise HeaderContractError(f"line {number}: invalid or duplicate path rule")
        pattern = line
    finish()

    if not any(rule.pattern == "/*" for rule in rules):
        raise HeaderContractError("contract is missing the common /* rule")
    return tuple(rules)


def effective_headers(rules: Sequence[HeaderRule], path: str) -> dict[str, str]:
    """Apply matching rules in file order, the way Cloudflare does."""

    result: dict[str, str] = {}
    for rule in rules:
        expression = "^" + re.escape(rule.pattern).replace(r"\*", ".*") + "$"
        if re.fullmatch(expression, path):
            result.update(rule.headers)
    return result


def validate_route_headers(
    path: str,
    contract: Mapping[str, str],
    actual: Mapping[str, str],
    *,
    expect_preview_noindex: bool = False,
) -> list[str]:
    """Return the ways a route's live headers fall short of the contract."""

    errors: list[str] = []
    normalized = {name.lower(): value.strip() for name, value in actual.items()}
    for name in REQUIRED_SECURITY_HEADERS:
        if name not in contract:
            errors.append(f"contract does not declare {name}")
    for name, expected in contract.items():
        if name not in normalized:
            errors.append(f"missing {name}")
        elif normalized[name] != expected:
            errors.append(f"{name} differs from the contract")
    if (expect_preview_noindex or path in RESULT_ROUTES) and "noindex" not in normalized.get(
        "x-robots-tag", ""
    ).lower():
        errors.append("missing noindex X-Robots-Tag")
    if "set-cookie" in normalized:
        errors.append("unexpected set-cookie")
    return errors
