"""Dense and hybrid retrieval helpers."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any, Callable

from chatbot_uqac.rag.lexical_retrieval import retrieve_lexical_docs


logger = logging.getLogger(__name__)

_DEFAULT_RETRIEVAL_K = 4
_RRF_K = 60
_DENSE_POOL_MULTIPLIER = 2
_MAX_CHUNKS_PER_URL = 2
_LEXICAL_CANDIDATE_LIMIT = 8
_DEBUG_DOC_PREVIEW_DEFAULT = 1
_DEBUG_DOC_PREVIEW_FINAL = 4


def _short_text(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _log_retrieved_docs_debug(source_label: str, question: str, docs: list) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "%s: docs retrieved question=%r count=%s",
        source_label,
        _short_text(question),
        len(docs),
    )
    preview_count = (
        _DEBUG_DOC_PREVIEW_FINAL if source_label.endswith(":hybrid") else _DEBUG_DOC_PREVIEW_DEFAULT
    )
    for idx, doc in enumerate(docs[:preview_count], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        logger.debug(
            "%s: doc[%s] title=%r url=%r",
            source_label,
            idx,
            metadata.get("title", ""),
            metadata.get("url", ""),
        )
    if len(docs) > preview_count:
        logger.debug(
            "%s: ... %s more docs omitted in debug output",
            source_label,
            len(docs) - preview_count,
        )


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


def retrieve_docs_hybrid(
    retriever: Any,
    *,
    question: str,
    rewritten_question: str,
    retrieval_k: int | None,
    score_threshold: float | None,
    source_label: str = "Router",
    lexical_retriever: Callable[[str, int], list] | None = None,
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
    lexical_fetch = lexical_retriever or (
        lambda q, lim: retrieve_lexical_docs(q, lim, source_label=source_label)
    )
    lexical_docs = lexical_fetch(rewritten_question or question, lexical_limit)
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
    _log_retrieved_docs_debug(
        f"{source_label}:hybrid",
        rewritten_question or question,
        fused_docs,
    )
    return fused_docs
