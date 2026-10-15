#!/usr/bin/env python3
"""Check a live deployment's routes and response headers against `public/_headers`."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from scripts.header_contract import (
        CHECKED_ROUTES,
        effective_headers,
        parse_headers_contract,
        validate_route_headers,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from header_contract import (
        CHECKED_ROUTES,
        effective_headers,
        parse_headers_contract,
        validate_route_headers,
    )

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ORIGIN = "https://niru.ml"
USER_AGENT = "niru-deployment-verifier/2"
MAX_BODY_BYTES = 1_048_576


class UnsafeRedirectError(RuntimeError):
    """Raised when a response tries to leave the verified HTTPS origin."""


class CanonicalCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag.lower() == "link" and "canonical" in attributes.get("rel", "").lower().split():
            self.targets.append(attributes.get("href", ""))


def normalized_origin(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("deployment URL must use HTTPS and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("deployment URL must not contain credentials")
    return (parsed.hostname.lower(), parsed.port or 443)


def validate_redirect(source_url: str, target_url: str, origin: tuple[str, int]) -> str:
    resolved = urljoin(source_url, target_url)
    try:
        if normalized_origin(resolved) != origin:
            raise ValueError
    except ValueError as error:
        raise UnsafeRedirectError("redirect left the verified HTTPS origin") from error
    return resolved


class SameOriginRedirectHandler(HTTPRedirectHandler):
    def __init__(self, origin: tuple[str, int]) -> None:
        super().__init__()
        self.origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url = validate_redirect(req.full_url, newurl, self.origin)
        return super().redirect_request(req, fp, code, msg, headers, safe_url)


def check_route(opener, base_url, path, contract_rules, *, timeout, expect_preview_noindex):
    request = Request(
        urljoin(base_url, path.lstrip("/")),
        headers={"Accept": "text/html", "User-Agent": USER_AGENT},
    )
    with opener.open(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if status != 200:
            return [f"unexpected HTTP status {status}"]
        errors: list[str] = []
        final = urlsplit(response.geturl())
        if final.path != path or final.query or final.fragment:
            errors.append("final URL does not preserve the requested route")
        headers = {name.lower(): value.strip() for name, value in response.headers.items()}
        errors.extend(
            validate_route_headers(
                path,
                effective_headers(contract_rules, path),
                headers,
                expect_preview_noindex=expect_preview_noindex,
            )
        )
        if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/html":
            errors.append("content-type must identify an HTML document")
        body = response.read(MAX_BODY_BYTES + 1)
        if len(body) > MAX_BODY_BYTES:
            errors.append("HTML page exceeds the verifier body limit")
        else:
            collector = CanonicalCollector()
            collector.feed(body.decode("utf-8", errors="replace"))
            if collector.targets != [f"{CANONICAL_ORIGIN}{path}"]:
                errors.append(f"HTML document must contain exactly one canonical for {path}")
        return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a Niru deployment against the header contract (makes real requests)."
    )
    parser.add_argument("--base-url", required=True, help="HTTPS deployment origin to verify")
    parser.add_argument(
        "--expect-preview-noindex",
        action="store_true",
        help="require X-Robots-Tag noindex on every checked route",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="per-request timeout")
    args = parser.parse_args(argv)

    try:
        origin = normalized_origin(args.base_url)
    except ValueError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    base_url = f"https://{origin[0]}" + (f":{origin[1]}" if origin[1] != 443 else "") + "/"
    contract_rules = parse_headers_contract((ROOT / "public/_headers").read_text("utf-8"))
    opener = build_opener(SameOriginRedirectHandler(origin))

    failures = 0
    for path in CHECKED_ROUTES:
        try:
            errors = check_route(
                opener,
                base_url,
                path,
                contract_rules,
                timeout=args.timeout,
                expect_preview_noindex=args.expect_preview_noindex,
            )
        except (HTTPError, URLError, TimeoutError, UnsafeRedirectError, ValueError) as error:
            errors = [str(error)]
        if errors:
            failures += 1
            print(f"FAIL {path}: {'; '.join(errors)}", file=sys.stderr)
        else:
            print(f"PASS {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
