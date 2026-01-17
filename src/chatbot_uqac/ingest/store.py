"""SQLite persistence for raw documents."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    """Stored document record."""

    url: str
    title: str
    content: str
    content_hash: str
    updated_at: str


class DocumentStore:
    """Simple SQLite-backed document store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection to the local database."""
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create tables when they do not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    content_hash TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return a stable hash for content comparisons."""
        # Content hash is used to skip unchanged pages on re-ingest.
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, url: str) -> DocumentRecord | None:
        """Fetch a document by URL."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT url, title, content, content_hash, updated_at FROM documents WHERE url = ?",
                (url,),
            ).fetchone()
        if not row:
            return None
        return DocumentRecord(*row)

    def upsert(self, url: str, title: str, content: str) -> DocumentRecord:
        """Insert or update a document record."""
        content_hash = self.compute_hash(content)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (url, title, content, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                (url, title, content, content_hash, updated_at),
            )
            conn.commit()
        return DocumentRecord(url, title, content, content_hash, updated_at)
