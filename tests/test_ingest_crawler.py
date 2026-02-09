"""Tests for crawler URL filtering and traversal behavior."""

from __future__ import annotations

from types import SimpleNamespace

import chatbot_uqac.ingest.crawler as crawler
from chatbot_uqac.ingest.crawler import CrawlConfig


def test_normalize_url_and_pdf_detection() -> None:
    assert crawler.normalize_url("https://x/y/#part") == "https://x/y"
    assert crawler.is_pdf("https://x/a.pdf") is True
    assert crawler.is_pdf("https://x/a.html") is False


def test_is_allowed_url_checks_domain_scheme_path_and_extension() -> None:
    base_netloc = "www.uqac.ca"
    base_path = "/mgestion"
    prefixes = ("/wp-content/",)

    assert crawler.is_allowed_url(
        "https://www.uqac.ca/mgestion/page",
        base_netloc,
        base_path,
        prefixes,
    )
    assert not crawler.is_allowed_url(
        "ftp://www.uqac.ca/mgestion/page",
        base_netloc,
        base_path,
        prefixes,
    )
    assert not crawler.is_allowed_url(
        "https://example.com/mgestion/page",
        base_netloc,
        base_path,
        prefixes,
    )
    assert not crawler.is_allowed_url(
        "https://www.uqac.ca/mgestion/image.png",
        base_netloc,
        base_path,
        prefixes,
    )
    assert crawler.is_allowed_url(
        "https://www.uqac.ca/wp-content/doc.pdf",
        base_netloc,
        base_path,
        prefixes,
    )


def test_extract_links_resolves_relative_urls() -> None:
    html = '<a href="/a">A</a><a href="https://x/b">B</a>'
    links = list(crawler.extract_links(html, "https://x/base"))
    assert "https://x/a" in links
    assert "https://x/b" in links


def test_crawl_site_bfs_and_skip_pdf_fetch(monkeypatch) -> None:
    html_by_url = {
        "https://www.uqac.ca/mgestion": """
            <a href="/mgestion/a">A</a>
            <a href="/mgestion/file.pdf">PDF</a>
            <a href="/outside">Outside</a>
        """,
        "https://www.uqac.ca/mgestion/a": '<a href="/mgestion/b">B</a>',
        "https://www.uqac.ca/mgestion/b": "",
    }
    calls: list[str] = []

    def fake_get(url: str, headers: dict, timeout: int):
        calls.append(url.rstrip("/"))
        body = html_by_url.get(url.rstrip("/"), "")
        return SimpleNamespace(
            status_code=200,
            text=body,
            apparent_encoding="utf-8",
            encoding="utf-8",
        )

    monkeypatch.setattr(crawler.requests, "get", fake_get)

    config = CrawlConfig(
        base_url="https://www.uqac.ca/mgestion/",
        max_pages=10,
        timeout=3,
        user_agent="test-agent",
    )
    urls = crawler.crawl_site(config)

    assert "https://www.uqac.ca/mgestion" in urls
    assert "https://www.uqac.ca/mgestion/a" in urls
    assert "https://www.uqac.ca/mgestion/b" in urls
    assert "https://www.uqac.ca/mgestion/file.pdf" in urls
    assert "https://www.uqac.ca/outside" not in urls
    assert "https://www.uqac.ca/mgestion/file.pdf" not in calls
