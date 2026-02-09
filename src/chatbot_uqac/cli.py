"""CLI entry point for the UQAC chatbot."""

from __future__ import annotations

import sqlite3

from chatbot_uqac.compat import ensure_supported_python


ensure_supported_python()


from rich.console import Console
from rich.prompt import Prompt

from chatbot_uqac.config import (
    CHROMA_DIR,
    DB_PATH,
    RETRIEVAL_K,
    RETRIEVAL_SCORE_THRESHOLD,
    OLLAMA_CHAT_MODEL,
    STREAMING_ENABLED,
)
from chatbot_uqac.logging_config import setup_logging
from chatbot_uqac.rag.engine import RagChat, build_llm, extract_sources
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


def main() -> None:
    """Run an interactive chat loop in the terminal."""
    setup_logging()
    console = Console()
    if not DB_PATH.exists() or not CHROMA_DIR.exists():
        # Avoid running the chat without an indexed corpus.
        console.print(
            "Missing local data. Run: python -m chatbot_uqac.ingest",
            style="yellow",
        )
        return

    try:
        embeddings = build_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        llm = build_llm()
        chat = RagChat(
            retriever,
            llm,
            retrieval_k=RETRIEVAL_K,
            score_threshold=RETRIEVAL_SCORE_THRESHOLD,
        )
    except RuntimeError as e:
        console.print(str(e), style="red")
        return

    try:
        chunk_count = vectorstore._collection.count()
    except Exception:
        chunk_count = None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except Exception:
        doc_count = None

    console.print("ChatBot UQAC (type /exit to quit)", style="bold cyan")
    if doc_count is not None or chunk_count is not None:
        parts = []
        if doc_count is not None:
            parts.append(f"Documents: {doc_count}")
        if chunk_count is not None:
            parts.append(f"Chunks: {chunk_count}")
        console.print("Dataset size: " + " | ".join(parts), style="dim")
    console.print(f"Chat model: {OLLAMA_CHAT_MODEL}", style="dim")

    while True:
        question = Prompt.ask("You")
        if question.strip().lower() in {"/exit", "/quit"}:
            break
        if not question.strip():
            continue

        if STREAMING_ENABLED:
            console.print("")
            def _write_chunk(text: str) -> None:
                console.print(text, end="")

            answer, docs = chat.ask(question, stream=True, on_chunk=_write_chunk)
            console.print("\n")
        else:
            answer, docs = chat.ask(question)
            console.print(f"\n{answer}\n")

        # Only show sources that were cited by the model.
        sources = extract_sources(docs, answer)
        if sources:
            console.print("Sources:", style="bold")
            for url in sources:
                console.print(f"- {url}")
        console.print("")


if __name__ == "__main__":
    main()
