"""Orchestrate crawling, extraction, and indexing."""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress

from chatbot_uqac.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    MAX_PAGES,
    REQUEST_TIMEOUT,
    UQAC_BASE_URL,
    USER_AGENT,
)
from chatbot_uqac.ingest.crawler import CrawlConfig, crawl_site, is_pdf
from chatbot_uqac.ingest.loaders import fetch_html, fetch_pdf
from chatbot_uqac.ingest.store import DocumentStore
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter for chunking."""
    # Chunking keeps context sizes manageable for embedding and retrieval.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def main() -> None:
    """Run ingestion from the command line."""
    parser = argparse.ArgumentParser(description="Ingest UQAC management guide.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args()

    console = Console()
    console.print("Crawling site...", style="bold cyan")

    crawl_config = CrawlConfig(
        base_url=UQAC_BASE_URL,
        max_pages=args.max_pages,
        timeout=REQUEST_TIMEOUT,
        user_agent=USER_AGENT,
    )
    # Crawl first so we can show progress on ingestion.
    urls = crawl_site(crawl_config)

    if not urls:
        console.print("No URLs found. Check network access or base URL.", style="red")
        return

    console.print(f"Found {len(urls)} URLs.", style="bold green")

    store = DocumentStore(DB_PATH)
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    splitter = build_splitter()

    headers = {"User-Agent": USER_AGENT}

    with Progress() as progress:
        task = progress.add_task("Ingesting", total=len(urls))
        for url in urls:
            progress.advance(task)

            # Skip pages that fail to download or parse.
            try:
                if is_pdf(url):
                    title, content = fetch_pdf(url, headers, REQUEST_TIMEOUT)
                else:
                    title, content = fetch_html(url, headers, REQUEST_TIMEOUT)
            except Exception:
                continue

            if not content:
                continue

            # Avoid re-indexing content that has not changed.
            existing = store.get(url)
            new_hash = store.compute_hash(content)
            if existing and existing.content_hash == new_hash:
                continue

            store.upsert(url, title, content)

            # Remove any old chunks for this URL before re-adding.
            try:
                vectorstore.delete(where={"url": url})
            except Exception:
                pass

            documents = []
            for chunk in splitter.split_text(content):
                documents.append(
                    Document(page_content=chunk, metadata={"url": url, "title": title})
                )
            if documents:
                vectorstore.add_documents(documents)

    # Some vectorstore clients auto-persist; call persist when available.
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    console.print("Ingestion complete.", style="bold green")


if __name__ == "__main__":
    main()
