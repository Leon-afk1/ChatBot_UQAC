"""URL crawling utilities for the UQAC management guide."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# Skip non-text assets to keep the crawl focused and fast.
_SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".zip",
    ".rar",
    ".7z",
    ".mp4",
    ".mp3",
    ".wav",
}


@dataclass(frozen=True)
class CrawlConfig:
    """Crawler configuration for scope and limits."""

    base_url: str
    max_pages: int
    timeout: int
    user_agent: str
    extra_path_prefixes: tuple[str, ...] = ("/wp-content/",)


def normalize_url(url: str) -> str:
    """Normalize URL by dropping fragments and trailing slash."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


def is_pdf(url: str) -> bool:
    """Return True when the URL points to a PDF file."""
    return url.lower().endswith(".pdf")


def _is_allowed_path(path: str, base_path: str, extra_prefixes: tuple[str, ...]) -> bool:
    """Check whether a path is within the allowed crawl scope."""
    # Restrict crawl scope to the base path plus explicit allowlisted prefixes.
    if not base_path or base_path == "/":
        return True
    normalized = path.rstrip("/")
    if normalized == base_path or path.startswith(f"{base_path}/"):
        return True
    for prefix in extra_prefixes:
        if normalized == prefix.rstrip("/") or path.startswith(prefix):
            return True
    return False


def is_allowed_url(
    url: str, base_netloc: str, base_path: str, extra_prefixes: tuple[str, ...]
) -> bool:
    """Validate URL based on scheme, host, scope, and file type."""
    # Enforce same-domain and scoped-path constraints.
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != base_netloc:
        return False
    if not _is_allowed_path(parsed.path or "", base_path, extra_prefixes):
        return False
    suffix = (parsed.path or "").lower()
    if not suffix:
        return True
    ext = suffix[suffix.rfind(".") :] if "." in suffix else ""
    if ext in _SKIP_EXTENSIONS:
        return False
    return True


def extract_links(html: str, base_url: str) -> Iterable[str]:
    """Extract absolute URLs from anchor tags in a page."""
    # Resolve relative links to absolute URLs.
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link.get("href", "").strip()
        if not href:
            continue
        yield urljoin(base_url, href)


def crawl_site(config: CrawlConfig) -> list[str]:
    """Breadth-first crawl to collect URLs within scope."""
    base_url = normalize_url(config.base_url)
    base_netloc = urlparse(base_url).netloc
    base_path = urlparse(base_url).path.rstrip("/")
    seen: set[str] = set()
    queue: deque[str] = deque([base_url])
    results: list[str] = []

    headers = {"User-Agent": config.user_agent}

    # Breadth-first crawl to avoid deep single-path exploration.
    while queue and len(results) < config.max_pages:
        url = normalize_url(queue.popleft())
        if url in seen:
            continue
        seen.add(url)

        if not is_allowed_url(
            url, base_netloc, base_path, config.extra_path_prefixes
        ):
            continue

        results.append(url)
        if is_pdf(url):
            continue

        # Best-effort crawling: skip transient errors to keep progress moving.
        try:
            response = requests.get(url, headers=headers, timeout=config.timeout)
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        response.encoding = response.apparent_encoding or response.encoding
        for link in extract_links(response.text, url):
            normalized = normalize_url(link)
            if normalized not in seen and is_allowed_url(
                normalized, base_netloc, base_path, config.extra_path_prefixes
            ):
                queue.append(normalized)

    return results
