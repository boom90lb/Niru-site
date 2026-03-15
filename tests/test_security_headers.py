from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

# --- Expected HTML meta http-equiv values ---
# CSP directives are compared as a set of stripped lines.
EXPECTED_META = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "content-security-policy": {
        directive.strip()
        for directive in dedent(
            """\
            default-src 'none';
            img-src 'self' data: https://www.linkedin.com https://assets.mlcdn.com https://assets.mailerlite.com;
            style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://assets.mlcdn.com https://fonts.mlcdn.com;
            script-src 'self' 'unsafe-inline' 'unsafe-eval' https://groot.mailerlite.com https://assets.mailerlite.com https://assets.mlcdn.com;
            font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com https://assets.mlcdn.com https://fonts.mlcdn.com;
            connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://assets.mailerlite.com https://groot.mailerlite.com https://assets.mlcdn.com https://fonts.mlcdn.com;
            frame-src https://assets.mailerlite.com https://groot.mailerlite.com;
            child-src https://assets.mailerlite.com https://groot.mailerlite.com;
            form-action 'self' https://assets.mailerlite.com https://groot.mailerlite.com;
            """
        )
        .strip()
        .splitlines()
    },
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}

# --- Expected Netlify/Cloudflare _headers file values ---
EXPECTED_HEADERS = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "content-security-policy": (
        "default-src 'none'; "
        "img-src 'self' data: https://www.linkedin.com https://assets.mlcdn.com https://assets.mailerlite.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://assets.mlcdn.com https://fonts.mlcdn.com; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://groot.mailerlite.com https://assets.mailerlite.com https://assets.mlcdn.com; "
        "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com https://assets.mlcdn.com https://fonts.mlcdn.com; "
        "connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://assets.mailerlite.com https://groot.mailerlite.com https://assets.mlcdn.com https://fonts.mlcdn.com; "
        "frame-src https://assets.mailerlite.com https://groot.mailerlite.com; "
        "child-src https://assets.mailerlite.com https://groot.mailerlite.com; "
        "form-action 'self' https://assets.mailerlite.com https://groot.mailerlite.com"
    ),
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
}

# All HTML pages that are expected to have security meta tags.
HTML_FILES = [
    DOCS_DIR / "index.html",
    DOCS_DIR / "about.html",
    DOCS_DIR / "pre-register.html",
]


class MetaCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "meta":
            self.meta.append({name.lower(): value or "" for name, value in attrs})


def collect_meta(file_path: Path) -> list[dict[str, str]]:
    parser = MetaCollector()
    parser.feed(file_path.read_text())
    return parser.meta


def assert_expected_meta(meta_tags: list[dict[str, str]]) -> None:
    lookup: dict[str, list[str]] = {}
    for tag in meta_tags:
        http_equiv = tag.get("http-equiv")
        content = tag.get("content")
        if http_equiv and content:
            lookup.setdefault(http_equiv.lower(), []).append(content.strip())

    for header, expected_content in EXPECTED_META.items():  # type: ignore[union-attr]
        assert header in lookup, f"Missing meta http-equiv='{header}'"
        if isinstance(expected_content, set):
            actual_sets = [
                {
                    directive.strip()
                    for directive in content.split("\n")
                    if directive.strip()
                }
                for content in lookup[header]
            ]
            assert any(
                directives == expected_content for directives in actual_sets
            ), f"Meta http-equiv='{header}' directives do not match expected content"
        else:
            assert any(
                content == expected_content for content in lookup[header]
            ), f"Meta http-equiv='{header}' does not match expected content"


def test_security_meta_present():
    """Every HTML page must contain the expected security meta tags."""
    for html_file in HTML_FILES:
        assert html_file.exists(), f"HTML file missing: {html_file.name}"
        meta_tags = collect_meta(html_file)
        assert_expected_meta(meta_tags)


def parse_headers_file(
    headers_path: Path,
) -> dict[str, dict[str, str]]:
    headers_by_path: dict[str, dict[str, str]] = {}
    current_path = None

    for raw_line in headers_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
            current_path = stripped
            headers_by_path.setdefault(current_path, {})
            continue

        if ":" not in stripped or current_path is None:
            continue

        name, value = stripped.split(":", 1)
        headers_by_path[current_path][name.strip().lower()] = value.strip()

    return headers_by_path


def test_headers_file_contains_security_headers():
    """The _headers file must contain all expected security headers for /*."""
    headers_path = DOCS_DIR / "_headers"
    assert headers_path.exists(), "_headers file missing"

    headers_by_path = parse_headers_file(headers_path)
    root_headers = headers_by_path.get("/*")
    assert root_headers is not None, "Root path headers not defined"

    for header, expected_value in EXPECTED_HEADERS.items():
        actual_value = root_headers.get(header)
        assert (
            actual_value == expected_value
        ), f"Header '{header}' expected '{expected_value}' but found '{actual_value}'"


def test_no_tailwind_cdn_references():
    """No HTML file should reference the Tailwind CDN (removed dependency)."""
    for html_file in HTML_FILES:
        content = html_file.read_text()
        assert (
            "cdn.tailwindcss.com" not in content
        ), f"{html_file.name} still references cdn.tailwindcss.com"


def test_shared_stylesheet_referenced():
    """Every HTML page must link to the shared styles.css."""
    for html_file in HTML_FILES:
        content = html_file.read_text()
        assert (
            'href="styles.css"' in content
        ), f"{html_file.name} does not reference styles.css"


def test_shared_components_referenced():
    """Every HTML page must include the shared components.js."""
    for html_file in HTML_FILES:
        content = html_file.read_text()
        assert (
            'src="components.js"' in content
        ), f"{html_file.name} does not reference components.js"


def test_seo_meta_tags_present():
    """Every HTML page must have description and Open Graph meta tags."""
    for html_file in HTML_FILES:
        parser = MetaCollector()
        parser.feed(html_file.read_text())

        meta_names = {
            tag.get("name", tag.get("property", "")).lower() for tag in parser.meta
        }
        assert (
            "description" in meta_names
        ), f"{html_file.name} missing meta name='description'"
        assert (
            "og:title" in meta_names
        ), f"{html_file.name} missing meta property='og:title'"
        assert (
            "og:description" in meta_names
        ), f"{html_file.name} missing meta property='og:description'"


def test_copyright_year():
    """Footer copyright should reference 2026."""
    for html_file in HTML_FILES:
        content = html_file.read_text()
        assert (
            "2026" in content
        ), f"{html_file.name} does not contain copyright year 2026"
