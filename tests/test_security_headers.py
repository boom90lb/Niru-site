from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
EXPECTED_META = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "content-security-policy": {
        directive.strip()
        for directive in dedent(
            """
            default-src 'none';
            img-src 'self' data: https://www.linkedin.com;
            style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com;
            script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com;
            font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com;
            connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.tailwindcss.com;
            """
        ).strip().splitlines()
    },
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}

EXPECTED_HEADERS = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "content-security-policy": "default-src 'none'; img-src 'self' data: https://www.linkedin.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.tailwindcss.com; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com; font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.tailwindcss.com",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
}


class MetaCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "meta":
            self.meta.append({name.lower(): value for name, value in attrs})


def collect_meta(file_path: Path):
    parser = MetaCollector()
    parser.feed(file_path.read_text())
    return parser.meta


def assert_expected_meta(meta_tags):
    lookup = {}
    for tag in meta_tags:
        http_equiv = tag.get("http-equiv")
        content = tag.get("content")
        if http_equiv and content:
            lookup.setdefault(http_equiv.lower(), []).append(content.strip())

    for header, expected_content in EXPECTED_META.items():
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
    html_files = [DOCS_DIR / "index.html", DOCS_DIR / "about.html"]
    for html_file in html_files:
        meta_tags = collect_meta(html_file)
        assert_expected_meta(meta_tags)


def parse_headers_file(headers_path: Path):
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
    headers_path = DOCS_DIR / "_headers"
    assert headers_path.exists(), "Netlify _headers file missing"

    headers_by_path = parse_headers_file(headers_path)
    root_headers = headers_by_path.get("/*")
    assert root_headers is not None, "Root path headers not defined"

    for header, expected_value in EXPECTED_HEADERS.items():
        actual_value = root_headers.get(header)
        assert (
            actual_value == expected_value
        ), f"Header '{header}' expected '{expected_value}' but found '{actual_value}'"
