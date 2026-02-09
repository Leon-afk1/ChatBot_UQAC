"""Vectorstore and embedding helpers."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from chatbot_uqac.config import CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def _ensure_writable_dir(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(
            f"{label} is not writable: {path}. "
            "Fix permissions (e.g. `chmod -R u+w data`) or set DATA_DIR/CHROMA_DIR to a writable location."
        ) from e

    # Need both write + execute (traverse) on directories.
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"{label} is not writable: {path}. "
            "Fix permissions (e.g. `chmod -R u+w data`) or set DATA_DIR/CHROMA_DIR to a writable location."
        )

    probe = path / ".chatbot_uqac_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"{label} is not writable: {path}. "
            "Chroma needs to create lock/journal files next to its SQLite DB. "
            "Fix permissions (e.g. `chmod -R u+w data`) or set DATA_DIR/CHROMA_DIR to a writable location."
        ) from e


def build_embeddings() -> OllamaEmbeddings:
    """Create an embeddings client backed by Ollama."""
    # Embeddings are generated locally via Ollama.
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def load_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    """Open or create the persistent Chroma collection."""
    # Persist vectors on disk so ingestion is not required every run.
    _ensure_writable_dir(CHROMA_DIR, "Chroma persistence directory")
    return Chroma(
        collection_name="uqac_mgestion",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )