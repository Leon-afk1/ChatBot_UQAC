"""Tests for RAG engine formatting, extraction, and orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import chatbot_uqac.rag.engine as engine


def _doc(content: str, *, url: str = "", title: str = "") -> SimpleNamespace:
    metadata = {}
    if url:
        metadata["url"] = url
    if title:
        metadata["title"] = title
    return SimpleNamespace(page_content=content, metadata=metadata)


def test_format_docs_numbers_chunks_and_headers() -> None:
    docs = [
        _doc("Chunk A", url="https://a", title="Doc A"),
        _doc("Chunk B"),
    ]

    text = engine.format_docs(docs)

    assert "[1]" in text and "[2]" in text
    assert "Title: Doc A" in text
    assert "Source: https://a" in text
    assert "Chunk B" in text


def test_strip_source_lines_removes_source_labels() -> None:
    raw = "Answer line\nSources: x\n[1] Source: y\nFinal line"
    cleaned = engine._strip_source_lines(raw)

    assert "Sources:" not in cleaned
    assert "[1] Source:" not in cleaned
    assert "Answer line" in cleaned and "Final line" in cleaned


def test_extract_cited_indices_and_sources() -> None:
    docs = [
        _doc("A", url="https://a"),
        _doc("B", url="https://b"),
        _doc("C", url="https://c"),
    ]
    answer = "Based on [1, 3], details..."

    indices = engine._extract_cited_indices(answer)
    source_refs = engine.extract_source_refs(docs, answer)
    sources = engine.extract_sources(docs, answer)

    assert indices == {1, 3}
    assert source_refs == [(1, "https://a"), (3, "https://c")]
    assert sources == ["https://a", "https://c"]


def test_extract_source_refs_preserves_citation_indices_with_duplicate_urls() -> None:
    docs = [
        _doc("A", url="https://same"),
        _doc("B", url="https://same"),
        _doc("C", url="https://other"),
    ]
    answer = "Supported by [1, 2, 3]."

    refs = engine.extract_source_refs(docs, answer)

    assert refs == [(1, "https://same"), (2, "https://same"), (3, "https://other")]


def test_group_source_refs_merges_duplicate_urls() -> None:
    refs = [(1, "https://same"), (2, "https://same"), (3, "https://other")]

    grouped = engine.group_source_refs(refs)

    assert grouped == [([1, 2], "https://same"), ([3], "https://other")]


def test_extract_grouped_source_refs_uses_cited_indices() -> None:
    docs = [
        _doc("A", url="https://same"),
        _doc("B", url="https://same"),
        _doc("C", url="https://other"),
    ]
    answer = "Supported by [2, 3]."

    grouped = engine.extract_grouped_source_refs(docs, answer)

    assert grouped == [([2], "https://same"), ([3], "https://other")]


def test_ragchat_ask_runs_rag_path_and_updates_history() -> None:
    docs = [_doc("Policy text", url="https://manual/policy", title="Policy")]
    retriever = Mock()
    retriever.invoke.return_value = docs
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Answer from context [1]\nSources: nope")

    chat = engine.RagChat(retriever, llm, summarize_threshold=99)
    answer, out_docs = chat.ask("What is the policy?")

    assert "Sources:" not in answer
    assert out_docs == docs
    assert isinstance(chat.history[-2], HumanMessage)
    assert isinstance(chat.history[-1], AIMessage)


def test_ragchat_ask_streaming_path() -> None:
    docs = [_doc("Policy text", url="https://manual/policy")]
    retriever = Mock()
    retriever.invoke.return_value = docs

    llm = Mock()
    llm.stream.return_value = [
        SimpleNamespace(content="Hello "),
        SimpleNamespace(content="world"),
    ]

    chunks: list[str] = []
    chat = engine.RagChat(retriever, llm, summarize_threshold=99)
    answer, _ = chat.ask("What is policy?", stream=True, on_chunk=chunks.append)

    assert answer == "Hello world"
    assert chunks == ["Hello ", "world"]


def test_ragchat_ask_memory_mode_does_not_retrieve(monkeypatch) -> None:
    retriever = Mock()
    llm = Mock()
    chat = engine.RagChat(retriever, llm, summarize_threshold=99)

    def _fake_route(*args, **kwargs):
        return "memory", {"reason": "meta_query"}

    monkeypatch.setattr(engine, "route", _fake_route)
    monkeypatch.setattr(engine, "answer_from_memory_only", lambda *_: "Memory answer")

    answer, docs = chat.ask("what was my last question?")

    assert answer == "Memory answer"
    assert docs == []
    retriever.invoke.assert_not_called()


def test_ragchat_ask_chitchat_mode_does_not_retrieve(monkeypatch) -> None:
    retriever = Mock()
    llm = Mock()
    chat = engine.RagChat(retriever, llm, summarize_threshold=99)

    def _fake_route(*args, **kwargs):
        return "chitchat", {"reason": "llm_intent"}

    monkeypatch.setattr(engine, "route", _fake_route)
    monkeypatch.setattr(engine, "answer_from_chitchat", lambda *_: "Salut! Je vais bien.")

    answer, docs = chat.ask("Salut ca va ?")

    assert answer == "Salut! Je vais bien."
    assert docs == []
    retriever.invoke.assert_not_called()


def test_append_history_triggers_summarization(monkeypatch) -> None:
    retriever = Mock()
    llm = Mock()
    chat = engine.RagChat(
        retriever,
        llm,
        summarize_threshold=2,
        keep_recent=2,
        max_history_messages=10,
    )

    monkeypatch.setattr(engine, "summarize_history", lambda *_: "summary text")

    chat._append_history("Q1", "A1")
    chat._append_history("Q2", "A2")

    assert isinstance(chat.history[0], SystemMessage)
    assert "summary" in chat.history[0].content.lower()
