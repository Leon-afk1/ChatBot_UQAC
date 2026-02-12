"""Routing and memory-only helpers for the RAG chat engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chatbot_uqac.rag.lexical_retrieval import (
    extract_query_terms as _extract_query_terms,
    retrieve_lexical_docs as _retrieve_lexical_docs,
)
from chatbot_uqac.rag.retrieval import (
    retrieve_docs as _retrieve_docs_impl,
    retrieve_docs_hybrid as _retrieve_docs_hybrid_impl,
)


logger = logging.getLogger(__name__)

ROUTER_INTENT_CONFIDENCE = 0.65
_ALLOWED_INTENTS = {"domain", "memory", "chitchat", "unclear"}


def _short_text(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def summarize_history(history: list[BaseMessage], llm: ChatOllama) -> str:
    """Generate a factual summary of the conversation history."""
    if not history:
        return ""

    conversation_text: list[str] = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", msg.content)
            conversation_text.append(f"Assistant: {content}")

    summary_prompt = (
        "You are summarizing a conversation. Create a concise, factual summary "
        "of the following conversation. Do NOT include any sources, URLs, or citations. "
        "Focus only on the key topics discussed and important information exchanged. "
        "Keep it brief (2-3 sentences max).\n\n"
        f"Conversation:\n{chr(10).join(conversation_text)}\n\n"
        "Summary:"
    )

    response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary = response.content if hasattr(response, "content") else str(response)
    return summary.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if isinstance(text, (bytes, bytearray)):
        raw = text.decode("utf-8", errors="ignore").strip()
    elif isinstance(text, str):
        raw = text.strip()
    else:
        return None

    if not raw:
        return None

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def classify_intent(
    question: str,
    history: list[BaseMessage],
    llm: ChatOllama,
) -> tuple[str, float]:
    """Classify a user query into routing intents using the LLM."""
    q = (question or "").strip()
    if not q:
        return "unclear", 0.0

    recent_history: list[str] = []
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            recent_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            recent_history.append(f"Assistant: {msg.content}")
    history_text = "\n".join(recent_history) if recent_history else "(no history)"

    prompt = (
        "Classify the user's latest message for a RAG assistant.\n"
        "Return JSON only with this schema: "
        '{"intent":"domain|memory|chitchat|unclear","confidence":0.0_to_1.0}.\n'
        "Intent rules:\n"
        "- memory: asks about prior chat turns/history/last question or answer.\n"
        "- memory: asks to reformulate/simplify/shorten/translate a previous assistant answer.\n"
        "- chitchat: greetings, thanks, social talk, small talk.\n"
        "- domain: asks information related to UQAC management manual content.\n"
        "- unclear: ambiguous intent.\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f'Latest message: "{q}"\n'
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.debug("Intent classification failed; defaulting to domain.", exc_info=True)
        return "domain", 0.0

    response_text = response.content if hasattr(response, "content") else str(response)
    payload = _extract_json_object(response_text)
    if not payload:
        return "domain", 0.0

    intent = str(payload.get("intent", "domain")).strip().lower()
    if intent not in _ALLOWED_INTENTS:
        intent = "domain"

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    logger.info(
        "Routing intent classified intent=%s confidence=%.2f question=%r",
        intent,
        confidence,
        _short_text(q),
    )
    return intent, confidence


def rewrite_retrieval_query(
    question: str,
    history: list[BaseMessage],
    llm: ChatOllama | None,
) -> str:
    """Rewrite follow-up questions into a standalone retrieval query."""
    q = (question or "").strip()
    if not q or llm is None or not history:
        return q

    recent_history: list[str] = []
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            recent_history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            recent_history.append(f"Assistant: {msg.content}")

    history_text = "\n".join(recent_history).strip() or "(no history)"
    prompt_messages = [
        SystemMessage(
            content=(
                "You rewrite follow-up user questions into standalone retrieval queries. "
                "Target domain: UQAC management manual. "
                "Preserve the original language and intent. "
                "If the latest question is already standalone, return it unchanged. "
                "Output only the final query text, no JSON, no explanation."
            )
        ),
        HumanMessage(
            content=(
                f"Recent conversation:\n{history_text}\n\n"
                f"Latest user question:\n{q}\n\n"
                "Standalone retrieval query:"
            )
        ),
    ]

    try:
        response = llm.invoke(prompt_messages)
    except Exception:
        logger.debug(
            "Retrieval query rewriting failed; fallback to original question.",
            exc_info=True,
        )
        return q

    rewritten = response.content if hasattr(response, "content") else str(response)
    rewritten_query = (rewritten or "").strip()
    if not rewritten_query:
        return q

    rewritten_query = rewritten_query.strip("`").strip()
    rewritten_query = re.sub(
        r"^\s*(rewritten(\s+retrieval)?\s+query|query)\s*:\s*",
        "",
        rewritten_query,
        flags=re.IGNORECASE,
    ).strip()
    rewritten_query = rewritten_query.splitlines()[0].strip()
    if not rewritten_query:
        return q

    if rewritten_query != q:
        logger.info(
            "Router retrieval query rewritten original=%r rewritten=%r",
            _short_text(q),
            _short_text(rewritten_query),
        )
    else:
        logger.debug("Router retrieval query unchanged=%r", _short_text(q))
    return rewritten_query


def retrieve_docs(
    retriever: Any,
    question: str,
    *,
    retrieval_k: int | None,
    score_threshold: float | None,
    source_label: str = "Retriever",
) -> list:
    """Backward-compatible wrapper around retrieval helpers."""
    return _retrieve_docs_impl(
        retriever,
        question,
        retrieval_k=retrieval_k,
        score_threshold=score_threshold,
        source_label=source_label,
    )


def retrieve_docs_hybrid(
    retriever: Any,
    *,
    question: str,
    rewritten_question: str,
    retrieval_k: int | None,
    score_threshold: float | None,
    source_label: str = "Router",
) -> list:
    """Backward-compatible wrapper around hybrid retrieval helpers."""
    return _retrieve_docs_hybrid_impl(
        retriever,
        question=question,
        rewritten_question=rewritten_question,
        retrieval_k=retrieval_k,
        score_threshold=score_threshold,
        source_label=source_label,
        lexical_retriever=lambda q, lim: _retrieve_lexical_docs(
            q,
            lim,
            source_label=source_label,
        ),
    )


def route(
    question: str,
    history: list[BaseMessage],
    retriever: Any,
    *,
    llm: ChatOllama | None = None,
    retrieval_k: int | None = None,
    score_threshold: float | None = None,
    intent_conf_threshold: float = ROUTER_INTENT_CONFIDENCE,
) -> tuple[str, dict[str, Any]]:
    """
    Returns (mode, payload) where mode in:
      - "clarify": intent is unclear -> ask for clarification
      - "chitchat": social message -> answer without retrieval context
      - "memory": meta question about history -> memory-only (no retrieval)
      - "no_docs": no relevant docs (score too low) -> fixed message
      - "rag": proceed with RAG using retrieved docs (payload contains "docs")
    """
    q = (question or "").strip()

    intent = "domain"
    confidence = 0.0
    if llm is not None:
        intent, confidence = classify_intent(q, history, llm)
        if intent == "memory" and confidence >= intent_conf_threshold:
            logger.info(
                "Routing decision mode=memory reason=llm_intent confidence=%.2f question=%r",
                confidence,
                _short_text(q),
            )
            return "memory", {"reason": "llm_intent", "confidence": confidence}
        if intent == "chitchat" and confidence >= intent_conf_threshold:
            logger.info(
                "Routing decision mode=chitchat reason=llm_intent confidence=%.2f question=%r",
                confidence,
                _short_text(q),
            )
            return "chitchat", {"reason": "llm_intent", "confidence": confidence}
        if intent == "unclear" and confidence >= intent_conf_threshold:
            logger.info(
                "Routing decision mode=clarify reason=llm_intent confidence=%.2f question=%r",
                confidence,
                _short_text(q),
            )
            return "clarify", {
                "reason": "llm_intent",
                "confidence": confidence,
                "answer": (
                    "Ta question est un peu vague. Peux-tu préciser ce que tu cherches "
                    "(thème, chapitre, contexte) dans le manuel de gestion UQAC ?"
                ),
            }

    retrieval_query = rewrite_retrieval_query(q, history, llm)
    docs = retrieve_docs_hybrid(
        retriever,
        question=q,
        rewritten_question=retrieval_query,
        retrieval_k=retrieval_k,
        score_threshold=score_threshold,
        source_label="Router",
    )
    if docs:
        logger.info(
            "Routing decision mode=rag reason=docs_found docs=%s question=%r",
            len(docs),
            _short_text(q),
        )
        return "rag", {"docs": docs}

    if intent == "chitchat":
        logger.info(
            "Routing decision mode=chitchat reason=llm_intent_weak confidence=%.2f question=%r",
            confidence,
            _short_text(q),
        )
        return "chitchat", {"reason": "llm_intent_weak", "confidence": confidence}

    logger.info(
        "Routing decision mode=no_docs reason=no_relevant_docs question=%r",
        _short_text(q),
    )
    return "no_docs", {"answer": "Je ne trouve pas l’information dans le manuel."}


def answer_from_chitchat(question: str, llm: ChatOllama) -> str:
    """Answer social small-talk without retrieval."""
    q = (question or "").strip()
    if not q:
        return "Bonjour! Comment puis-je vous aider avec le manuel de gestion UQAC?"

    messages = [
        SystemMessage(
            content=(
                "You are a friendly assistant for UQAC's management guide. "
                "For social messages and greetings, respond naturally in the user's language "
                "in 1-2 short sentences, then gently invite a management-guide question."
            )
        ),
        HumanMessage(content=q),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    return (text or "").strip() or "Bonjour! Comment puis-je vous aider aujourd’hui?"


def answer_from_memory_only(
    question: str,
    history: list[BaseMessage],
    llm: ChatOllama,
) -> str:
    """Answer using chat history only (no retrieval and no rule-based parsing)."""
    if not history:
        return "Je n’ai pas encore d’historique dans cette session."

    transcript_lines: list[str] = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            transcript_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            transcript_lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            transcript_lines.append(f"System: {msg.content}")

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        return "Je ne trouve pas d’éléments exploitables dans l’historique."

    prompt_messages = [
        SystemMessage(
            content=(
                "You answer only from the provided conversation history. "
                "Do not use external knowledge. "
                "If the requested information is not present in the history, say you do not know. "
                "Do not provide source URLs or bracket citations."
            )
        ),
        HumanMessage(
            content=(
                f"Conversation history:\n{transcript}\n\n"
                f"User question about this history:\n{question}"
            )
        ),
    ]

    response = llm.invoke(prompt_messages)
    text = response.content if hasattr(response, "content") else str(response)
    return (text or "").strip() or "Je ne sais pas à partir de l’historique disponible."
