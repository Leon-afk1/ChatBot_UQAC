"""Tests for HTML/PDF loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chatbot_uqac.ingest.loaders as loaders


def test_clean_text_normalizes_whitespace() -> None:
    text = " a \n\n b\t c "
    assert loaders.clean_text(text) == "a b c"


def test_fetch_html_extracts_title_and_target_blocks(monkeypatch) -> None:
    html = """
    <html>
      <head><title>My Title</title></head>
      <body>
        <div class="entry-header">Header section</div>
        <div class="entry-content">Main content</div>
        <div class="other">Should not be included</div>
      </body>
    </html>
    """

    def fake_get(url: str, headers: dict[str, str], timeout: int):
        return SimpleNamespace(
            text=html,
            apparent_encoding="utf-8",
            encoding="utf-8",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(loaders.requests, "get", fake_get)
    title, content = loaders.fetch_html("https://x", headers={}, timeout=5)

    assert title == "My Title"
    assert "Header section" in content
    assert "Main content" in content
    assert "Should not be included" not in content


def test_fetch_pdf_reads_pages_and_cleans_temp_file(monkeypatch) -> None:
    created_paths: list[Path] = []

    class FakeReader:
        def __init__(self, file_path: str):
            created_paths.append(Path(file_path))
            self.pages = [
                SimpleNamespace(extract_text=lambda: "Page one"),
                SimpleNamespace(extract_text=lambda: "Page two"),
            ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int):
            _ = chunk_size
            yield b"%PDF-fake"

    def fake_get(url: str, headers: dict[str, str], timeout: int, stream: bool):
        _ = (url, headers, timeout, stream)
        return FakeResponse()

    monkeypatch.setattr(loaders, "PdfReader", FakeReader)
    monkeypatch.setattr(loaders.requests, "get", fake_get)

    title, content = loaders.fetch_pdf("https://x/doc.pdf", headers={}, timeout=5)

    assert "tmp" in title.lower()
    assert content == "Page one Page two"
    assert created_paths, "Expected a temporary PDF path to be created."
    assert not created_paths[0].exists(), "Temporary PDF file should be removed."
