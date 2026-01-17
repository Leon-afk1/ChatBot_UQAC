"""CLI entry point for the UQAC chatbot."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from chatbot_uqac.config import CHROMA_DIR, DB_PATH
from chatbot_uqac.rag.engine import RagChat, build_llm, extract_sources
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


def main() -> None:
    """Run an interactive chat loop in the terminal."""
    console = Console()
    if not DB_PATH.exists() or not CHROMA_DIR.exists():
        # Avoid running the chat without an indexed corpus.
        console.print(
            "Missing local data. Run: python -m chatbot_uqac.ingest",
            style="yellow",
        )
        return

    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()
    chat = RagChat(retriever, llm)

    console.print("ChatBot UQAC (type /exit to quit)", style="bold cyan")

    while True:
        question = Prompt.ask("You")
        if question.strip().lower() in {"/exit", "/quit"}:
            break
        if not question.strip():
            continue

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
