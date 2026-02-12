"""Routing and memory-only helpers for the RAG chat engine."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chatbot_uqac.config import DB_PATH

logger = logging.getLogger(__name__)

ROUTER_INTENT_CONFIDENCE = 0.65
_ALLOWED_INTENTS = {"domain", "memory", "chitchat", "unclear"}
_DEFAULT_RETRIEVAL_K = 4
_RRF_K = 60
_DENSE_POOL_MULTIPLIER = 2
_MAX_CHUNKS_PER_URL = 2
_LEXICAL_CANDIDATE_LIMIT = 8
_DEBUG_DOC_PREVIEW_DEFAULT = 1
_DEBUG_DOC_PREVIEW_FINAL = 4
_LEXICAL_MIN_TERM_LEN = 3
_STOPWORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "et",
    "est",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "lui",
    "mais",
    "me",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
}
_FTS_READY = False


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
    preview_count = _DEBUG_DOC_PREVIEW_FINAL if source_label.endswith(":hybrid") else _DEBUG_DOC_PREVIEW_DEFAULT
    for idx, doc in enumerate(docs[:preview_count], start=1):
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
    if len(docs) > preview_count:
        logger.debug(
            "%s: ... %s more docs omitted in debug output",
            source_label,
            len(docs) - preview_count,
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


def _extract_query_terms(question: str, limit: int = 8) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[\w'-]+\b", (question or "").lower()):
        for part in re.split(r"[-']", token):
            part = part.strip("-'")
            if len(part) < _LEXICAL_MIN_TERM_LEN or part in _STOPWORDS:
                continue
            if part in seen:
                continue
            seen.add(part)
            terms.append(part)
            if len(terms) >= limit:
                return terms
    return terms


def _doc_key(doc: Any) -> tuple[str, str]:
    metadata = getattr(doc, "metadata", {}) or {}
    url = str(metadata.get("url", "")).strip()
    content = str(getattr(doc, "page_content", "") or "")
    return url, content[:240]


def _with_retrieval_source(doc: Any, source: str, score: float | None = None) -> Any:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["retrieval_source"] = source
    if score is not None:
        metadata["hybrid_score"] = round(score, 6)
    content = getattr(doc, "page_content", "")
    return SimpleNamespace(page_content=content, metadata=metadata)


def _fetch_lexical_docs_fts(conn: sqlite3.Connection, question: str, limit: int) -> list:
    global _FTS_READY
    if not _FTS_READY:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(url UNINDEXED, title, content)"
        )
        fts_count = conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
        if fts_count == 0:
            conn.execute(
                "INSERT INTO documents_fts (url, title, content) SELECT url, title, content FROM documents"
            )
            conn.commit()
        _FTS_READY = True

    terms = _extract_query_terms(question)
    match_parts = [f"{term}*" for term in terms]
    if not match_parts:
        return []

    rows = conn.execute(
        """
        SELECT url, title, content, bm25(documents_fts) AS rank
        FROM documents_fts
        WHERE documents_fts MATCH ?
        ORDER BY rank ASC
        LIMIT ?
        """,
        (" OR ".join(match_parts), limit),
    ).fetchall()
    docs: list = []
    for row in rows:
        docs.append(
            SimpleNamespace(
                page_content=(row[2] or "")[:1200],
                metadata={
                    "url": row[0] or "",
                    "title": row[1] or "",
                    "retrieval_source": "lexical_fts",
                },
            )
        )
    return docs


def _fetch_lexical_docs_like(conn: sqlite3.Connection, question: str, limit: int) -> list:
    terms = _extract_query_terms(question)
    if not terms:
        return []

    conditions: list[str] = []
    score_parts: list[str] = []
    condition_params: list[str] = []
    score_params: list[str] = []
    for term in terms:
        pattern = f"%{term}%"
        conditions.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
        condition_params.extend([pattern, pattern])
        score_parts.append("(CASE WHEN LOWER(title) LIKE ? THEN 3 ELSE 0 END)")
        score_parts.append("(CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END)")
        score_params.extend([pattern, pattern])

    if not conditions:
        return []

    sql = f"""
        SELECT url, title, content, ({' + '.join(score_parts)}) AS lexical_score
        FROM documents
        WHERE {' OR '.join(conditions)}
        ORDER BY lexical_score DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (*score_params, *condition_params, limit)).fetchall()
    docs: list = []
    for row in rows:
        docs.append(
            SimpleNamespace(
                page_content=(row[2] or "")[:1200],
                metadata={
                    "url": row[0] or "",
                    "title": row[1] or "",
                    "retrieval_source": "lexical_like",
                },
            )
        )
    return docs


def _retrieve_lexical_docs(question: str, limit: int) -> list:
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            docs = _fetch_lexical_docs_fts(conn, question, limit)
            if docs:
                _log_retrieved_docs_debug("Router:lexical_fts", question, docs)
                return docs
    except sqlite3.OperationalError as exc:
        logger.debug(
            "FTS lexical retrieval failed (%s); using LIKE fallback.",
            str(exc),
        )
    except sqlite3.Error as exc:
        logger.debug(
            "FTS lexical retrieval error (%s); using LIKE fallback.",
            str(exc),
        )

    try:
        with sqlite3.connect(db_path) as conn:
            docs = _fetch_lexical_docs_like(conn, question, limit)
            _log_retrieved_docs_debug("Router:lexical_like", question, docs)
            return docs
    except sqlite3.Error as exc:
        logger.debug("LIKE lexical retrieval failed (%s).", str(exc))
        return []


def _fuse_rrf(
    rank_lists: list[tuple[str, list]],
    *,
    final_k: int,
    max_chunks_per_url: int,
) -> list:
    fused: dict[tuple[str, str], dict[str, Any]] = {}

    for label, docs in rank_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            if key not in fused:
                fused[key] = {"doc": doc, "score": 0.0, "signals": set()}
            fused[key]["score"] += 1.0 / (_RRF_K + rank)
            fused[key]["signals"].add(label)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    selected: list = []
    per_url: dict[str, int] = {}
    for item in ranked:
        doc = item["doc"]
        metadata = getattr(doc, "metadata", {}) or {}
        url = str(metadata.get("url", "")).strip()
        if url:
            count = per_url.get(url, 0)
            if count >= max_chunks_per_url:
                continue
            per_url[url] = count + 1
        sources = "+".join(sorted(item["signals"]))
        selected.append(_with_retrieval_source(doc, sources, score=item["score"]))
        if len(selected) >= final_k:
            break
    return selected


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


def retrieve_docs_hybrid(
    retriever: Any,
    *,
    question: str,
    rewritten_question: str,
    retrieval_k: int | None,
    score_threshold: float | None,
    source_label: str = "Router",
) -> list:
    """Hybrid retrieval: dense over multiple queries + lexical fallback + RRF."""
    base_k = retrieval_k or _DEFAULT_RETRIEVAL_K
    dense_pool_k = max(base_k, _DEFAULT_RETRIEVAL_K) * _DENSE_POOL_MULTIPLIER

    queries: list[str] = []
    for query in (question, rewritten_question):
        q = (query or "").strip()
        if q and q not in queries:
            queries.append(q)
    if not queries:
        return []

    dense_rank_lists: list[tuple[str, list]] = []
    for idx, query in enumerate(queries, start=1):
        dense_docs = retrieve_docs(
            retriever,
            query,
            retrieval_k=dense_pool_k,
            score_threshold=score_threshold,
            source_label=f"{source_label}:dense[{idx}]",
        )
        dense_rank_lists.append((f"dense_q{idx}", dense_docs))

    lexical_limit = max(base_k, _LEXICAL_CANDIDATE_LIMIT)
    lexical_docs = _retrieve_lexical_docs(rewritten_question or question, lexical_limit)
    rank_lists = dense_rank_lists + [("lexical", lexical_docs)]
    fused_docs = _fuse_rrf(
        rank_lists,
        final_k=base_k,
        max_chunks_per_url=_MAX_CHUNKS_PER_URL,
    )

    logger.info(
        "%s: hybrid retrieval dense_queries=%s dense_docs=%s lexical_docs=%s fused_docs=%s",
        source_label,
        len(queries),
        sum(len(docs) for _, docs in dense_rank_lists),
        len(lexical_docs),
        len(fused_docs),
    )
    _log_retrieved_docs_debug(f"{source_label}:hybrid", rewritten_question or question, fused_docs)
    return fused_docs


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
    # Try retrieval first. If relevant docs exist, proceed with RAG even for short queries.
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
