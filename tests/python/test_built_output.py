"""Behavioral checks over the production build in dist/ and the GitHub Pages projection."""

from __future__ import annotations

import base64
from html.parser import HTMLParser
import hashlib
from pathlib import Path
import re
from xml.etree import ElementTree

import pytest

from scripts.header_contract import (
    CHECKED_ROUTES,
    PUBLIC_ROUTES,
    RESULT_ROUTES,
    effective_headers,
    parse_headers_contract,
    validate_route_headers,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public"
DIST = ROOT / "dist"
DOCS = ROOT / "docs"
ORIGIN = "https://niru.ml"


def output_path(route: str) -> Path:
    return DIST / "index.html" if route == "/" else DIST / route.strip("/") / "index.html"


class HeadCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: list[str] = []
        self.robots: list[str] = []
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {name.lower(): value or "" for name, value in attrs}
        if tag == "link" and a.get("rel", "").lower() == "canonical":
            self.canonicals.append(a.get("href", ""))
        if tag == "meta" and a.get("name", "").lower() == "robots":
            self.robots.append(a.get("content", ""))
        for attribute in ("href", "src"):
            if a.get(attribute, "").startswith("/"):
                self.references.append(a[attribute])


def head(path: Path) -> HeadCollector:
    collector = HeadCollector()
    collector.feed(path.read_text(encoding="utf-8"))
    return collector


def test_every_route_builds() -> None:
    assert DIST.is_dir(), "dist/ missing; run the production build first"
    assert [route for route in CHECKED_ROUTES if not output_path(route).is_file()] == []


def test_fallback_projection_matches_the_static_build() -> None:
    def tree(root: Path) -> dict[str, bytes]:
        return {
            p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }

    built, fallback = tree(DIST), tree(DOCS)
    assert fallback == built
    for output in (built, fallback):
        contact = output["preview/index.html"].decode()
        assert "<form" not in contact and "/api/preview" not in contact
        assert "mailto:brendonrt@niru.ml" in contact
        assert "under development" in contact
    assert (DOCS / ".nojekyll").is_file()


def test_header_contract_is_copied_and_satisfied_for_every_route() -> None:
    assert (DIST / "_headers").read_bytes() == (PUBLIC / "_headers").read_bytes()
    rules = parse_headers_contract((DIST / "_headers").read_text(encoding="utf-8"))
    failures = {}
    for route in CHECKED_ROUTES:
        contract = effective_headers(rules, route)
        errors = validate_route_headers(route, contract, contract)
        if errors:
            failures[route] = errors
    assert failures == {}


def test_legacy_redirects_are_copied_unchanged() -> None:
    assert (DIST / "_redirects").read_bytes() == (PUBLIC / "_redirects").read_bytes()
    redirects = {}
    for line in (DIST / "_redirects").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            source, destination, status = line.split()
            redirects[source] = (destination, status)
    assert redirects == {
        "/index.html": ("/", "301"),
        "/about.html": ("/about/", "301"),
        "/pre-register.html": ("/preview/", "301"),
        "/system/": ("/", "301"),
    }


@pytest.mark.parametrize("route", PUBLIC_ROUTES)
def test_public_route_has_exact_canonical(route: str) -> None:
    assert head(output_path(route)).canonicals == [f"{ORIGIN}{route}"]


@pytest.mark.parametrize("route", RESULT_ROUTES)
def test_result_route_is_noindex(route: str) -> None:
    assert any("noindex" in value.lower() for value in head(output_path(route)).robots)


def test_every_internal_reference_resolves() -> None:
    missing = []
    for html in sorted(DIST.rglob("*.html")):
        for reference in head(html).references:
            path = re.split(r"[?#]", reference, maxsplit=1)[0]
            target = DIST / path.lstrip("/")
            if path.endswith("/"):
                target /= "index.html"
            if not target.is_file():
                missing.append(f"{html.relative_to(DIST)}: {reference}")
    assert missing == []
    for asset in ("NIRUML.ico", "icon-32.png", "apple-touch-icon.png", "niru-social-card.png"):
        assert (DIST / asset).is_file(), asset


def test_csp_hashes_match_every_inline_script() -> None:
    rules = parse_headers_contract((DIST / "_headers").read_text(encoding="utf-8"))
    csp = effective_headers(rules, "/")["content-security-policy"]
    script_src = next(d.split()[1:] for d in csp.split(";") if d.split()[:1] == ["script-src"])
    declared = {token for token in script_src if token.startswith("'sha256-")}

    inline = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", re.IGNORECASE)
    computed = set()
    for html in sorted(DIST.rglob("*.html")):
        for body in inline.findall(html.read_text(encoding="utf-8")):
            digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode()
            computed.add(f"'sha256-{digest}'")
    assert declared == computed


def test_sitemap_lists_exactly_the_public_routes() -> None:
    root = ElementTree.parse(DIST / "sitemap.xml").getroot()
    namespace = re.match(r"\{([^}]+)\}", root.tag)
    locs = (
        root.findall("s:url/s:loc", {"s": namespace.group(1)})
        if namespace
        else root.findall("url/loc")
    )
    assert {element.text for element in locs} == {f"{ORIGIN}{route}" for route in PUBLIC_ROUTES}
    robots = (DIST / "robots.txt").read_text(encoding="utf-8")
    assert f"Sitemap: {ORIGIN}/sitemap.xml" in robots and "Disallow: /" not in robots
