import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).parents[1]
LEARNING = ROOT / "docs" / "learning"
CHAPTERS = [
    ("01-money-assets-cash.html", "Money, Assets, And Cash"),
    ("02-prices-dividends-returns.html", "Prices, Dividends, And Returns"),
    ("03-risk-downside-losses.html", "Risk And Downside Losses"),
    ("04-market-regimes-persistence.html", "Market Regimes And Persistence"),
    ("05-data-parity-proxy-replication.html", "Data Parity And Proxy Replication"),
    (
        "06-reproducibility-sealed-evidence.html",
        "Reproducibility And Sealed Evidence",
    ),
    ("07-backtesting-without-future.html", "Backtesting Without Seeing The Future"),
    ("08-returns-to-model-features.html", "From Returns To Model Features"),
    ("09-hidden-markov-models.html", "Hidden Markov Models From Zero"),
    (
        "10-clustering-statistical-jump-models.html",
        "Clustering And Statistical Jump Models",
    ),
    (
        "11-dynamic-programming-online-inference.html",
        "Dynamic Programming And Online Inference",
    ),
    (
        "12-walk-forward-selection-performance.html",
        "Walk-Forward Selection And Performance Measurement",
    ),
]
REQUIRED_SECTIONS = {
    "question",
    "vocabulary",
    "paper-connection",
    "project-connection",
    "evidence-limitations",
    "practice",
    "advisor",
}
APPROVED_DEPENDENCIES = {
    "https://cdn.jsdelivr.net/npm/mathjax@4.1.3/mml-chtml.js": (
        "sha384-GYAeDZjH9w23NyL4cfS+ZrgWmGvta6VNs4p+/vtuF+RWe30fQhmP8pHbzRmlwmAK"
    ),
    "https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js": (
        "sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ"
    ),
}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class DocumentParser(HTMLParser):
    """Collect the structural facts used by the authored-course contract."""

    def __init__(self) -> None:
        super().__init__()
        self.html_lang = None
        self.ids = []
        self.hrefs = []
        self.scripts = []
        self.section_ids = set()
        self.class_counts = {}
        self.visible_text = []
        self._excluded_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.html_lang = attributes.get("lang")
        if element_id := attributes.get("id"):
            self.ids.append(element_id)
            if tag == "section":
                self.section_ids.add(element_id)
        if href := attributes.get("href"):
            self.hrefs.append(href)
        if tag == "script" and (src := attributes.get("src")):
            self.scripts.append((src, attributes.get("integrity")))
        for class_name in (attributes.get("class") or "").split():
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        if self._excluded_depth:
            if tag not in VOID_ELEMENTS:
                self._excluded_depth += 1
        elif tag in {"script", "style", "nav", "footer"}:
            self._excluded_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._excluded_depth:
            self._excluded_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._excluded_depth:
            self.visible_text.append(data)


def parse(path: Path) -> DocumentParser:
    parser = DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def local_target(page: Path, reference: str) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (page.parent / parsed.path).resolve()


@pytest.mark.parametrize(("filename", "title"), CHAPTERS)
def test_authored_chapter_contract(filename: str, title: str) -> None:
    path = LEARNING / filename
    source = path.read_text(encoding="utf-8")
    document = parse(path)
    visible_words = len(" ".join(document.visible_text).split())

    assert document.html_lang == "en"
    assert 2_500 <= visible_words <= 3_500
    assert REQUIRED_SECTIONS <= document.section_ids
    assert len(document.ids) == len(set(document.ids))
    assert document.class_counts.get("worked-example", 0) >= 2
    assert document.class_counts.get("remember", 0) >= 1
    assert document.class_counts.get("lab", 0) == 1
    assert document.class_counts.get("quiz", 0) == 1
    assert "common failure" in source.lower()
    assert "$100" in source
    assert f"<h1>{title}</h1>" in source
    assert "data-reset" in source
    assert "<math" in source
    assert "mermaid" not in source.lower()


def test_authored_chapter_order_and_navigation() -> None:
    index = (LEARNING / "index.html").read_text(encoding="utf-8")
    positions = []
    for chapter_number, (filename, title) in enumerate(CHAPTERS, start=1):
        positions.append(index.index(f'href="{filename}"'))
        assert f'<span class="chapter-number">{chapter_number:02d}</span>' in index
        assert f"<strong>{title}</strong>" in index
    assert positions == sorted(positions)
    assert 'class="chapter-card planned" href=' not in index

    for chapter_index, (filename, _) in enumerate(CHAPTERS):
        hrefs = parse(LEARNING / filename).hrefs
        expected_previous = (
            "index.html" if chapter_index == 0 else CHAPTERS[chapter_index - 1][0]
        )
        expected_next = (
            "index.html"
            if chapter_index == len(CHAPTERS) - 1
            else CHAPTERS[chapter_index + 1][0]
        )
        assert expected_previous in hrefs
        assert expected_next in hrefs


def test_learning_documents_have_valid_local_references() -> None:
    pages = [
        LEARNING / "index.html",
        *(LEARNING / filename for filename, _ in CHAPTERS),
    ]
    for page in pages:
        document = parse(page)
        references = [*document.hrefs, *(source for source, _ in document.scripts)]
        for reference in references:
            target = local_target(page, reference)
            if target is not None:
                assert target.exists(), f"{page}: missing local reference {reference}"


def test_browser_dependencies_are_exactly_pinned_and_allowlisted() -> None:
    manifest = json.loads((LEARNING / "browser-dependencies.json").read_text())
    recorded = {item["url"]: item["integrity"] for item in manifest["dependencies"]}
    assert manifest["schema_version"] == 1
    assert recorded == APPROVED_DEPENDENCIES

    external_scripts = []
    for filename, _ in CHAPTERS:
        external_scripts.extend(
            script
            for script in parse(LEARNING / filename).scripts
            if urlsplit(script[0]).scheme
        )
    assert external_scripts
    for source, integrity in external_scripts:
        assert source in APPROVED_DEPENDENCIES
        assert integrity == APPROVED_DEPENDENCIES[source]


def test_learning_stack_contains_no_mermaid() -> None:
    authored_files = [
        *LEARNING.glob("*.html"),
        LEARNING / "course.css",
        LEARNING / "course.js",
        LEARNING / "browser-dependencies.json",
    ]
    assert all(
        "mermaid" not in path.read_text(encoding="utf-8").lower()
        for path in authored_files
    )
