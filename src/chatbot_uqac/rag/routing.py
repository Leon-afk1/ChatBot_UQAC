"""Routing and memory-only helpers for the RAG chat engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


logger = logging.getLogger(__name__)

ROUTER_INTENT_CONFIDENCE = 0.65
_ALLOWED_INTENTS = {"domain", "memory", "chitchat", "unclear"}


def _short_text(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_retrieved_docs_debug(
    source_label: str,
    question: str,
    docs: list,
) -> None:
    """Emit per-document retrieval details when DEBUG logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "%s: docs retrieved question=%r count=%s",
        source_label,
        _short_text(question),
        len(docs),
    )
    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        logger.debug(
            "%s: doc[%s] title=%r url=%r",
            source_label,
            idx,
            title,
            url,
        )


def summarize_history(history: list[BaseMessage], llm: ChatOllama) -> str:
    """Generate a factual summary of the conversation history."""
    if not history:
        return ""

    # Build a compact transcript for summarization.
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


def retrieve_docs(
    retriever: Any,
    question: str,
    *,
    retrieval_k: int | None,
    score_threshold: float | None,
    source_label: str = "Retriever",
) -> list:
    """Retrieve documents with optional score filtering."""
    if score_threshold is None:
        if hasattr(retriever, "invoke"):
            docs = list(retriever.invoke(question))
        else:
            docs = list(retriever.get_relevant_documents(question))
        _log_retrieved_docs_debug(source_label, question, docs)
        return docs

    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore and hasattr(vectorstore, "similarity_search_with_score"):
        k = retrieval_k if retrieval_k is not None else getattr(
            retriever, "search_kwargs", {}
        ).get("k", 4)
        results = vectorstore.similarity_search_with_score(question, k=k)
        docs = [doc for doc, score in results if score <= score_threshold]
        if not docs and results:
            best_score = min(score for _, score in results)
            logger.info(
                "%s: no docs under score threshold %s (best=%s).",
                source_label,
                score_threshold,
                best_score,
            )
        logger.info(
            "%s: retrieved %s documents after score filtering.",
            source_label,
            len(docs),
        )
        _log_retrieved_docs_debug(source_label, question, docs)
        return docs

    if hasattr(retriever, "invoke"):
        docs = list(retriever.invoke(question))
    else:
        docs = list(retriever.get_relevant_documents(question))
    _log_retrieved_docs_debug(source_label, question, docs)
    return docs


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
    _ = history  # Reserved for future history-aware routing rules.
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

    # Try retrieval first. If relevant docs exist, proceed with RAG even for short queries.
    docs = retrieve_docs(
        retriever,
        q,
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

    # Build a compact plain-text transcript for memory-only answering.
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
        )
    ]

    response = llm.invoke(prompt_messages)
    text = response.content if hasattr(response, "content") else str(response)
    return (text or "").strip() or "Je ne sais pas à partir de l’historique disponible."
