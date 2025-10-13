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
            script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com;
            font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com;
            connect-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.tailwindcss.com;
            """
        ).strip().splitlines()
    },
    "x-content-type-options": "nosniff",
    "x-frame-options": "SAMEORIGIN",
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
