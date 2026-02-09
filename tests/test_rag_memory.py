"""Tests for memory summarization and memory-only answers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage

import chatbot_uqac.rag.routing as routing


def test_summarize_history_omits_citations_in_prompt() -> None:
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Summary output")
    history = [
        HumanMessage(content="Q1"),
        AIMessage(content="A1 [1]"),
    ]

    summary = routing.summarize_history(history, llm)

    assert summary == "Summary output"
    call_messages = llm.invoke.call_args[0][0]
    prompt_text = call_messages[0].content
    assert "[1]" not in prompt_text
    assert "User: Q1" in prompt_text


def test_answer_from_memory_only_returns_empty_session_message() -> None:
    llm = Mock()
    answer = routing.answer_from_memory_only("what was my last answer?", [], llm)
    assert "pas encore d’historique" in answer


def test_answer_from_memory_only_calls_llm_with_transcript() -> None:
    history = [HumanMessage(content="Q1"), AIMessage(content="A1")]
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Memory-only answer")

    answer = routing.answer_from_memory_only("what was my last answer?", history, llm)

    assert answer == "Memory-only answer"
    sent_messages = llm.invoke.call_args[0][0]
    assert "Conversation history:" in sent_messages[1].content
    assert "User: Q1" in sent_messages[1].content
    assert "Assistant: A1" in sent_messages[1].content
