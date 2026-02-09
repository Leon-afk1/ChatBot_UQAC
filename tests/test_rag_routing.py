"""Tests for routing and retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from chatbot_uqac.rag.routing import (
    classify_intent,
    retrieve_docs,
    route,
)


def test_retrieve_docs_uses_invoke_without_threshold() -> None:
    retriever = Mock()
    retriever.invoke.return_value = ["doc1"]

    docs = retrieve_docs(
        retriever,
        "question",
        retrieval_k=4,
        score_threshold=None,
    )

    assert docs == ["doc1"]
    retriever.invoke.assert_called_once_with("question")


def test_retrieve_docs_filters_with_score_threshold() -> None:
    keep = SimpleNamespace(page_content="keep", metadata={})
    drop = SimpleNamespace(page_content="drop", metadata={})
    vectorstore = Mock()
    vectorstore.similarity_search_with_score.return_value = [(keep, 0.2), (drop, 0.9)]
    retriever = SimpleNamespace(vectorstore=vectorstore, search_kwargs={"k": 7})

    docs = retrieve_docs(
        retriever,
        "question",
        retrieval_k=None,
        score_threshold=0.5,
    )

    assert docs == [keep]
    vectorstore.similarity_search_with_score.assert_called_once_with("question", k=7)


def test_route_returns_memory_when_llm_classifier_is_confident() -> None:
    retriever = Mock()
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"intent":"memory","confidence":0.97}'
    )

    mode, payload = route(
        "what was my last question",
        history=[],
        retriever=retriever,
        llm=llm,
    )

    assert mode == "memory"
    assert payload["reason"] == "llm_intent"
    retriever.invoke.assert_not_called()


def test_route_prefers_rag_when_docs_exist() -> None:
    retriever = Mock()
    retriever.invoke.return_value = ["doc1"]

    mode, payload = route(
        "Absence maladie",
        history=[],
        retriever=retriever,
    )

    assert mode == "rag"
    assert payload["docs"] == ["doc1"]


def test_route_returns_clarify_when_unclear_intent_confident() -> None:
    retriever = Mock()
    retriever.invoke.return_value = []
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"intent":"unclear","confidence":0.92}'
    )

    mode, payload = route(
        "budget",
        history=[],
        retriever=retriever,
        llm=llm,
    )

    assert mode == "clarify"
    assert "vague" in payload["answer"].lower()
    retriever.invoke.assert_not_called()


def test_route_returns_no_docs_when_no_docs_and_specific() -> None:
    retriever = Mock()
    retriever.invoke.return_value = []

    mode, payload = route(
        "Quelle est la procédure d'approbation des congés annuels?",
        history=[],
        retriever=retriever,
    )

    assert mode == "no_docs"
    assert "manuel" in payload["answer"].lower()


def test_classify_intent_parses_llm_json_payload() -> None:
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"intent":"chitchat","confidence":0.91}'
    )

    intent, confidence = classify_intent("Salut ca va ?", history=[], llm=llm)

    assert intent == "chitchat"
    assert confidence == 0.91


def test_route_returns_chitchat_when_llm_classifier_is_confident() -> None:
    retriever = Mock()
    retriever.invoke.return_value = []
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"intent":"chitchat","confidence":0.95}'
    )

    mode, payload = route(
        "Salut ca va ?",
        history=[],
        retriever=retriever,
        llm=llm,
    )

    assert mode == "chitchat"
    assert payload["reason"] == "llm_intent"
