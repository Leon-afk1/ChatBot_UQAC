"""Lexical retrieval helpers backed by SQLite (FTS5 + LIKE fallback)."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from chatbot_uqac.config import DB_PATH


logger = logging.getLogger(__name__)

_LEXICAL_MIN_TERM_LEN = 3
_DEBUG_DOC_PREVIEW = 1
_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "et",
    "est",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "mais",
    "me",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
}
_FTS_READY = False


def _short_text(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_retrieved_docs_debug(source_label: str, question: str, docs: list) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "%s: docs retrieved question=%r count=%s",
        source_label,
        _short_text(question),
        len(docs),
    )
    for idx, doc in enumerate(docs[:_DEBUG_DOC_PREVIEW], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        logger.debug(
            "%s: doc[%s] title=%r url=%r",
            source_label,
            idx,
            metadata.get("title", ""),
            metadata.get("url", ""),
        )
    if len(docs) > _DEBUG_DOC_PREVIEW:
        logger.debug(
            "%s: ... %s more docs omitted in debug output",
            source_label,
            len(docs) - _DEBUG_DOC_PREVIEW,
        )


def extract_query_terms(question: str, limit: int = 8) -> list[str]:
    """Extract normalized lexical terms from a user query."""
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[\w'-]+\b", (question or "").lower()):
        for part in re.split(r"[-']", token):
            part = part.strip("-'")
            if len(part) < _LEXICAL_MIN_TERM_LEN or part in _STOPWORDS:
                continue
            if part in seen:
                continue
            seen.add(part)
            terms.append(part)
            if len(terms) >= limit:
                return terms
    return terms


def _fetch_lexical_docs_fts(conn: sqlite3.Connection, question: str, limit: int) -> list:
    global _FTS_READY
    if not _FTS_READY:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(url UNINDEXED, title, content)"
        )
        fts_count = conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
        if fts_count == 0:
            conn.execute(
                "INSERT INTO documents_fts (url, title, content) SELECT url, title, content FROM documents"
            )
            conn.commit()
        _FTS_READY = True

    terms = extract_query_terms(question)
    match_parts = [f"{term}*" for term in terms]
    if not match_parts:
        return []

    rows = conn.execute(
        """
        SELECT url, title, content, bm25(documents_fts) AS rank
        FROM documents_fts
        WHERE documents_fts MATCH ?
        ORDER BY rank ASC
        LIMIT ?
        """,
        (" OR ".join(match_parts), limit),
    ).fetchall()
    docs: list = []
    for row in rows:
        docs.append(
            SimpleNamespace(
                page_content=(row[2] or "")[:1200],
                metadata={"url": row[0] or "", "title": row[1] or "", "retrieval_source": "lexical_fts"},
            )
        )
    return docs


def _fetch_lexical_docs_like(conn: sqlite3.Connection, question: str, limit: int) -> list:
    terms = extract_query_terms(question)
    if not terms:
        return []

    conditions: list[str] = []
    score_parts: list[str] = []
    condition_params: list[str] = []
    score_params: list[str] = []
    for term in terms:
        pattern = f"%{term}%"
        conditions.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
        condition_params.extend([pattern, pattern])
        score_parts.append("(CASE WHEN LOWER(title) LIKE ? THEN 3 ELSE 0 END)")
        score_parts.append("(CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END)")
        score_params.extend([pattern, pattern])

    sql = f"""
        SELECT url, title, content, ({' + '.join(score_parts)}) AS lexical_score
        FROM documents
        WHERE {' OR '.join(conditions)}
        ORDER BY lexical_score DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*score_params, *condition_params, limit)).fetchall()
    docs: list = []
    for row in rows:
        docs.append(
            SimpleNamespace(
                page_content=(row[2] or "")[:1200],
                metadata={"url": row[0] or "", "title": row[1] or "", "retrieval_source": "lexical_like"},
            )
        )
    return docs


def retrieve_lexical_docs(question: str, limit: int, *, source_label: str = "Router") -> list:
    """Retrieve lexical candidates via FTS5, with LIKE fallback when needed."""
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            docs = _fetch_lexical_docs_fts(conn, question, limit)
            if docs:
                _log_retrieved_docs_debug(f"{source_label}:lexical_fts", question, docs)
                return docs
    except sqlite3.OperationalError as exc:
        logger.debug(
            "FTS lexical retrieval failed (%s); using LIKE fallback.",
            str(exc),
        )
    except sqlite3.Error as exc:
        logger.debug(
            "FTS lexical retrieval error (%s); using LIKE fallback.",
            str(exc),
        )

    try:
        with sqlite3.connect(db_path) as conn:
            docs = _fetch_lexical_docs_like(conn, question, limit)
            _log_retrieved_docs_debug(f"{source_label}:lexical_like", question, docs)
            return docs
    except sqlite3.Error as exc:
        logger.debug("LIKE lexical retrieval failed (%s).", str(exc))
        return []
