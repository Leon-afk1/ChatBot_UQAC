"""Vectorstore and embedding helpers."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from chatbot_uqac.config import CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def build_embeddings() -> OllamaEmbeddings:
    """Create an embeddings client backed by Ollama."""
    # Embeddings are generated locally via Ollama.
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def load_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    """Open or create the persistent Chroma collection."""
    # Persist vectors on disk so ingestion is not required every run.
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="uqac_mgestion",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
