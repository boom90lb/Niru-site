"""Unit tests for the `_headers` parser and the deployed-header comparison."""

from __future__ import annotations

import pytest

from scripts.check_deployed_headers import (
    UnsafeRedirectError,
    normalized_origin,
    validate_redirect,
)
from scripts.header_contract import (
    HeaderContractError,
    effective_headers,
    parse_headers_contract,
    validate_route_headers,
)

CONTRACT = """
/*
  X-Frame-Options: DENY
  Referrer-Policy: same-origin

/preview/submitted/*
  X-Robots-Tag: noindex, nofollow

/_astro/*
  Cache-Control: public, max-age=31536000, immutable
"""


def test_parser_applies_common_and_specific_rules_in_order() -> None:
    rules = parse_headers_contract(CONTRACT)
    assert effective_headers(rules, "/") == {
        "x-frame-options": "DENY",
        "referrer-policy": "same-origin",
    }
    assert effective_headers(rules, "/preview/submitted/")["x-robots-tag"] == "noindex, nofollow"
    assert "cache-control" in effective_headers(rules, "/_astro/app.css")


@pytest.mark.parametrize(
    "contract",
    [
        "  X-Frame-Options: DENY\n",
        "/*\n  X-Frame-Options: DENY\n/*\n  X-Frame-Options: DENY\n",
        "/*\n  X-Frame-Options: DENY\n  X-Frame-Options: DENY\n",
        "/about/\n  X-Frame-Options: DENY\n",
    ],
)
def test_parser_rejects_ambiguous_contracts(contract: str) -> None:
    with pytest.raises(HeaderContractError):
        parse_headers_contract(contract)


def test_live_headers_must_match_the_contract_exactly() -> None:
    contract = {
        "content-security-policy": "default-src 'none'",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-resource-policy": "same-origin",
        "permissions-policy": "camera=()",
        "referrer-policy": "same-origin",
        "strict-transport-security": "max-age=300",
        "x-content-type-options": "nosniff",
        "x-frame-options": "DENY",
    }
    assert validate_route_headers("/", contract, dict(contract)) == []

    drifted = {**contract, "X-Frame-Options": "SAMEORIGIN"}
    del drifted["x-frame-options"]
    assert validate_route_headers("/", contract, drifted) == [
        "x-frame-options differs from the contract"
    ]

    missing = dict(contract)
    del missing["strict-transport-security"]
    assert validate_route_headers("/", contract, missing) == ["missing strict-transport-security"]

    assert "unexpected set-cookie" in validate_route_headers(
        "/", contract, {**contract, "Set-Cookie": "a=b"}
    )


def test_result_routes_and_preview_deployments_require_noindex() -> None:
    assert validate_route_headers("/preview/submitted/", {}, {}) == [
        f"contract does not declare {name}"
        for name in (
            "content-security-policy",
            "cross-origin-opener-policy",
            "cross-origin-resource-policy",
            "permissions-policy",
            "referrer-policy",
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
        )
    ] + ["missing noindex X-Robots-Tag"]
    assert "missing noindex X-Robots-Tag" in validate_route_headers(
        "/", {}, {}, expect_preview_noindex=True
    )
    assert "missing noindex X-Robots-Tag" not in validate_route_headers(
        "/", {}, {"X-Robots-Tag": "noindex"}, expect_preview_noindex=True
    )


def test_deployed_checker_only_follows_same_origin_https_redirects() -> None:
    origin = normalized_origin("https://preview.example.pages.dev")
    assert validate_redirect("https://preview.example.pages.dev/about", "/about/", origin) == (
        "https://preview.example.pages.dev/about/"
    )
    for target in ("http://preview.example.pages.dev/about/", "https://attacker.invalid/"):
        with pytest.raises(UnsafeRedirectError):
            validate_redirect("https://preview.example.pages.dev/about", target, origin)
    with pytest.raises(ValueError):
        normalized_origin("https://user:pass@preview.example.pages.dev/")
