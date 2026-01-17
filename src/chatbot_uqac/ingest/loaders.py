"""Load and clean HTML and PDF content."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace and trim output text."""
    # Normalize whitespace to keep chunks compact and consistent.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_html(url: str, headers: dict[str, str], timeout: int) -> tuple[str, str]:
    """Download a HTML page and extract relevant text blocks."""
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Only keep content from the specified manual sections.
    blocks = []
    for class_name in ("entry-header", "entry-content"):
        for div in soup.find_all("div", class_=class_name):
            blocks.append(div.get_text(separator=" ", strip=True))

    content = clean_text(" ".join(blocks))
    return title, content


def fetch_pdf(url: str, headers: dict[str, str], timeout: int) -> tuple[str, str]:
    """Download a PDF and extract text from all pages."""
    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
    response.raise_for_status()

    # Use a temp file to support pypdf's file-based reader.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                tmp_file.write(chunk)
        temp_path = Path(tmp_file.name)

    try:
        reader = PdfReader(str(temp_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        content = clean_text(" ".join(pages))
        title = temp_path.name
        return title, content
    finally:
        # Always remove the temporary file.
        temp_path.unlink(missing_ok=True)
