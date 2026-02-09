"""Tests for SQLite document storage."""

from __future__ import annotations

from pathlib import Path

from chatbot_uqac.ingest.store import DocumentStore


def test_compute_hash_is_stable() -> None:
    left = DocumentStore.compute_hash("same content")
    right = DocumentStore.compute_hash("same content")
    assert left == right


def test_upsert_and_get_round_trip(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "docs.sqlite3")
    url = "https://example.com/page"

    created = store.upsert(url, "Title 1", "Content 1")
    fetched = store.get(url)

    assert fetched is not None
    assert fetched.url == created.url
    assert fetched.title == "Title 1"
    assert fetched.content == "Content 1"
    assert fetched.content_hash == DocumentStore.compute_hash("Content 1")


def test_upsert_updates_existing_record(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "docs.sqlite3")
    url = "https://example.com/page"
    store.upsert(url, "Title 1", "Content 1")

    updated = store.upsert(url, "Title 2", "Content 2")
    fetched = store.get(url)

    assert fetched is not None
    assert fetched.title == "Title 2"
    assert fetched.content == "Content 2"
    assert fetched.content_hash == updated.content_hash
