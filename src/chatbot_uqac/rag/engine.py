"""RAG prompting and chat session orchestration."""

from __future__ import annotations

import re
from typing import Iterable

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chatbot_uqac.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


SYSTEM_PROMPT = (
    "You are a helpful assistant for the UQAC management guide. "
    "Use only the provided context. If the answer is not in the context, "
    "say you do not know. Keep answers concise. "
    "Cite sources using square brackets like [1] based on the context numbering."
)


def build_llm() -> ChatOllama:
    """Create the local chat model served by Ollama."""
    # Local chat model served by Ollama.
    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def build_prompt() -> ChatPromptTemplate:
    """Build the prompt template with context and history."""
    # Inject context + chat history into a single prompt.
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )


def format_docs(docs: Iterable) -> str:
    """Format retrieved docs into a numbered context block."""
    # Number each chunk so the model can cite it as [n].
    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        header_bits = []
        if title:
            header_bits.append(f"Title: {title}")
        if url:
            header_bits.append(f"Source: {url}")
        header = " | ".join(header_bits)
        if header:
            parts.append(f"[{index}] {header}\n{doc.page_content}")
        else:
            parts.append(f"[{index}] {doc.page_content}")
    return "\n\n".join(parts)


class RagChat:
    """Session-scoped RAG chat with short-term memory."""

    def __init__(self, retriever, llm: ChatOllama, max_history_messages: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = build_prompt()
        self.history: list[BaseMessage] = []
        self.max_history_messages = max_history_messages

    def _get_docs(self, question: str) -> list:
        # Support both old and new retriever interfaces.
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(question)
        return self.retriever.get_relevant_documents(question)

    def ask(self, question: str) -> tuple[str, list]:
        """Retrieve context and generate a cited answer."""
        # Retrieve context and generate an answer with citations.
        docs = self._get_docs(question)
        context = format_docs(docs)
        messages = self.prompt.format_messages(
            input=question, context=context, history=self.history
        )
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        self._append_history(question, answer)
        return answer, docs

    def _append_history(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        # Keep a bounded history to avoid prompt bloat.
        excess = len(self.history) - self.max_history_messages * 2
        if excess > 0:
            self.history = self.history[excess:]


def _extract_cited_indices(answer: str) -> set[int]:
    # Parse citations like [1] or [1, 2] from the model output.
    indices: set[int] = set()
    for block in re.findall(r"\[([0-9,\\s]+)\\]", answer):
        for number in re.findall(r"\\d+", block):
            indices.add(int(number))
    return indices


def extract_sources(docs: Iterable, answer: str | None = None) -> list[str]:
    """Return source URLs, optionally filtered to cited indices."""
    # If the answer cites sources, only return those URLs.
    cited = _extract_cited_indices(answer) if answer else set()
    sources: list[str] = []
    seen = set()
    for idx, doc in enumerate(docs, start=1):
        if cited and idx not in cited:
            continue
        url = (doc.metadata or {}).get("url")
        if url and url not in seen:
            sources.append(url)
            seen.add(url)
    return sources
