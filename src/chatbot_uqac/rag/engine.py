"""RAG prompting and chat session orchestration."""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chatbot_uqac.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from chatbot_uqac.rag.routing import (
    answer_from_chitchat,
    answer_from_memory_only,
    retrieve_docs,
    route,
    summarize_history,
)


SYSTEM_PROMPT = (
    "You are a helpful assistant for the UQAC management guide. "
    "Use only the provided context. If the answer is not in the context, "
    "say you do not know. Keep answers concise. "
    "Cite sources using square brackets like [1] based on the context numbering. "
    "Do not write 'Source:' or document titles; only use bracketed citations."
)

logger = logging.getLogger(__name__)


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


def _strip_source_lines(answer: str) -> str:
    cleaned_lines = []
    for line in answer.splitlines():
        if re.match(r"^\s*(source|sources)\s*[:：]", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\s*\[\d+(?:,\s*\d+)*\]\s*source\s*[:：]", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


class RagChat:
    """Session-scoped RAG chat with short-term memory and periodic summarization."""

    def __init__(
        self,
        retriever,
        llm: ChatOllama,
        max_history_messages: int = 5,
        summarize_threshold: int = 10,
        keep_recent: int = 6,
        retrieval_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = build_prompt()
        self.history: list[BaseMessage] = []
        self.max_history_messages = max_history_messages
        self.summarize_threshold = summarize_threshold  # Threshold to trigger summarization
        self.keep_recent = keep_recent  # Number of recent messages to keep (must be even)
        self.retrieval_k = retrieval_k
        self.score_threshold = score_threshold

    def _get_docs(self, question: str) -> list:
        return retrieve_docs(
            self.retriever,
            question,
            retrieval_k=self.retrieval_k,
            score_threshold=self.score_threshold,
            source_label="RagChat",
        )
    
    def ask(
        self,
        question: str,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list]:
        """Retrieve context and generate a cited answer."""
        # Decide the answering path before running full RAG.
        mode, payload = route(
            question,
            self.history,
            self.retriever,
            llm=self.llm,
            retrieval_k=self.retrieval_k,
            score_threshold=self.score_threshold,
        )

        # 1) Query is too vague -> ask for clarification.
        if mode == "clarify":
            answer = payload["answer"]
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []

        # 2) Meta query -> history-only answer (no retrieval).
        if mode == "memory":
            answer = answer_from_memory_only(question, self.history, self.llm)
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []

        # 3) Social small-talk -> answer without retrieval.
        if mode == "chitchat":
            answer = answer_from_chitchat(question, self.llm)
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []

        # 4) No relevant docs -> return a fixed fallback message.
        if mode == "no_docs":
            answer = payload["answer"]
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []

        # 5) Standard RAG path, reusing documents retrieved by the router.
        docs = payload["docs"]

        # Existing RAG generation flow.
        context = format_docs(docs)
        messages = self.prompt.format_messages(
            input=question, context=context, history=self.history
        )

        if stream:
            parts: list[str] = []
            for chunk in self.llm.stream(messages):
                text = getattr(chunk, "content", str(chunk))
                if not text:
                    continue
                parts.append(text)
                if on_chunk:
                    on_chunk(text)
            answer = "".join(parts)
        else:
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

        answer = _strip_source_lines(answer)
        self._append_history(question, answer)
        return answer, docs


    def _append_history(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("History length: %s messages", len(self.history))
            for index, msg in enumerate(self.history):
                if isinstance(msg, HumanMessage):
                    msg_type = "user"
                elif isinstance(msg, AIMessage):
                    msg_type = "assistant"
                else:
                    msg_type = "summary"
                logger.debug("History[%s] %s: %s", index, msg_type, msg.content)

        # Trigger summarization if history exceeds threshold
        if len(self.history) > self.summarize_threshold:
            logger.info(
                "History length %s exceeds threshold %s; summarizing.",
                len(self.history),
                self.summarize_threshold,
            )
            self._summarize_and_compress_history()
        # Fallback: simple truncation if no summary was created
        elif len(self.history) > self.max_history_messages * 2:
            excess = len(self.history) - self.max_history_messages * 2
            self.history = self.history[excess:]

    def _summarize_and_compress_history(self) -> None:
        """Summarize old messages and keep only recent ones + summary."""
        # Keep only the most recent messages
        recent_messages = self.history[-self.keep_recent:]
        old_messages = self.history[:-self.keep_recent]

        # Check if we already have a summary at the beginning
        existing_summary = None
        if old_messages and isinstance(old_messages[0], SystemMessage):
            existing_summary = old_messages[0]
            old_messages = old_messages[1:]

        # Generate a summary of the old messages
        if old_messages:
            summary_text = summarize_history(old_messages, self.llm)

            # If there's an existing summary, combine them
            if existing_summary:
                combined_summary = (
                    f"Previous conversation summary: {existing_summary.content} "
                    f"Recent topics: {summary_text}"
                )
                summary_message = SystemMessage(content=combined_summary)
            else:
                summary_message = SystemMessage(
                    content=f"Conversation summary: {summary_text}"
                )

            # Replace history with summary + recent messages
            self.history = [summary_message] + recent_messages

            logger.info(
                "History compressed from %s to %s messages.",
                len(old_messages) + len(recent_messages),
                len(self.history),
            )


def _extract_cited_indices(answer: str) -> set[int]:
    # Parse citations like [1] or [1, 2] from the model output.
    indices: set[int] = set()
    for block in re.findall(r"\[([0-9,\s]+)\]", answer):
        for number in re.findall(r"\d+", block):
            indices.add(int(number))
    return indices


def extract_source_refs(docs: Iterable, answer: str | None = None) -> list[tuple[int, str]]:
    """Return (citation_index, URL) pairs, optionally filtered to cited indices."""
    cited = _extract_cited_indices(answer) if answer else set()
    refs: list[tuple[int, str]] = []
    for idx, doc in enumerate(docs, start=1):
        if cited and idx not in cited:
            continue
        url = (doc.metadata or {}).get("url")
        if url:
            refs.append((idx, url))
    return refs


def group_source_refs(source_refs: Iterable[tuple[int, str]]) -> list[tuple[list[int], str]]:
    """Group citation indices by URL while preserving URL order."""
    grouped: dict[str, list[int]] = {}
    ordered_urls: list[str] = []

    for idx, url in source_refs:
        if not url:
            continue
        if url not in grouped:
            grouped[url] = []
            ordered_urls.append(url)
        if idx not in grouped[url]:
            grouped[url].append(idx)

    return [(grouped[url], url) for url in ordered_urls]


def extract_grouped_source_refs(
    docs: Iterable,
    answer: str | None = None,
) -> list[tuple[list[int], str]]:
    """Return grouped citation indices by unique URL."""
    return group_source_refs(extract_source_refs(docs, answer))


def extract_sources(docs: Iterable, answer: str | None = None) -> list[str]:
    """Return unique source URLs, optionally filtered to cited indices."""
    refs = extract_source_refs(docs, answer)
    sources: list[str] = []
    seen = set()
    for _, url in refs:
        if url and url not in seen:
            sources.append(url)
            seen.add(url)
    return sources
