"""Tests for routing and retrieval helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage

import chatbot_uqac.rag.routing as routing
from chatbot_uqac.rag.routing import (
    classify_intent,
    retrieve_docs_hybrid,
    rewrite_retrieval_query,
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


def test_route_prefers_rag_when_docs_exist(monkeypatch) -> None:
    retriever = Mock()
    retriever.invoke.return_value = [
        SimpleNamespace(page_content="dense", metadata={"url": "https://dense"})
    ]
    monkeypatch.setattr(routing, "_retrieve_lexical_docs", lambda *_args, **_kwargs: [])

    mode, payload = route(
        "Absence maladie",
        history=[],
        retriever=retriever,
    )

    assert mode == "rag"
    assert len(payload["docs"]) == 1
    assert payload["docs"][0].metadata["url"] == "https://dense"


def test_rewrite_retrieval_query_makes_followup_standalone() -> None:
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="Quels sont les principes directeurs de la politique linguistique de l'UQAC ?"
    )
    history = [
        HumanMessage(content="Quelle est la politique linguistique de l'UQAC ?"),
        AIMessage(content="Résumé de la politique ..."),
    ]

    rewritten = rewrite_retrieval_query(
        "Et quels sont ces principes directeurs ?",
        history,
        llm,
    )

    assert rewritten == "Quels sont les principes directeurs de la politique linguistique de l'UQAC ?"


def test_rewrite_retrieval_query_falls_back_on_exception() -> None:
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("failure")
    history = [HumanMessage(content="Q1"), AIMessage(content="A1")]

    rewritten = rewrite_retrieval_query("Question de suivi", history, llm)

    assert rewritten == "Question de suivi"


def test_route_uses_rewritten_query_for_retrieval(monkeypatch) -> None:
    retriever = Mock()
    retriever.invoke.return_value = [
        SimpleNamespace(page_content="dense", metadata={"url": "https://dense"})
    ]
    llm = Mock()
    llm.invoke.side_effect = [
        SimpleNamespace(content='{"intent":"domain","confidence":0.95}'),
        SimpleNamespace(
            content="Quels sont les principes directeurs de la politique linguistique de l'UQAC ?"
        ),
    ]
    history = [
        HumanMessage(content="Quelle est la politique linguistique de l'UQAC ?"),
        AIMessage(content="Résumé ..."),
    ]
    monkeypatch.setattr(routing, "_retrieve_lexical_docs", lambda *_args, **_kwargs: [])

    mode, payload = route(
        "Et quels sont ces principes directeurs ?",
        history=history,
        retriever=retriever,
        llm=llm,
    )

    assert mode == "rag"
    assert len(payload["docs"]) >= 1
    called_queries = [call.args[0] for call in retriever.invoke.call_args_list]
    assert "Et quels sont ces principes directeurs ?" in called_queries
    assert "Quels sont les principes directeurs de la politique linguistique de l'UQAC ?" in called_queries


def test_retrieve_docs_hybrid_combines_dense_and_lexical(monkeypatch) -> None:
    retriever = Mock()
    dense_doc = SimpleNamespace(page_content="dense", metadata={"url": "https://dense"})
    retriever.invoke.return_value = [dense_doc]
    lexical_doc = SimpleNamespace(page_content="lexical", metadata={"url": "https://lexical"})

    monkeypatch.setattr(routing, "_retrieve_lexical_docs", lambda *_args, **_kwargs: [lexical_doc])

    docs = retrieve_docs_hybrid(
        retriever,
        question="question originale",
        rewritten_question="question reformulee",
        retrieval_k=4,
        score_threshold=None,
        source_label="Test",
    )

    urls = {(doc.metadata or {}).get("url") for doc in docs}
    assert "https://dense" in urls
    assert "https://lexical" in urls


def test_extract_query_terms_splits_apostrophes() -> None:
    terms = routing._extract_query_terms("Quels sont les dons que peut accepter l'université ?")

    assert "université" in terms
    assert all("'" not in term for term in terms)


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


def test_route_returns_no_docs_when_no_docs_and_specific(monkeypatch) -> None:
    retriever = Mock()
    retriever.invoke.return_value = []
    monkeypatch.setattr(routing, "_retrieve_lexical_docs", lambda *_args, **_kwargs: [])

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
