"""RAG prompting and chat session orchestration."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chatbot_uqac.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


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


def summarize_history(history: list[BaseMessage], llm: ChatOllama) -> str:
    """Generate a factual summary of the conversation history."""
    if not history:
        return ""
    
    # Build a prompt to summarize the conversation
    conversation_text = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Remove citations from the summary
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


def _strip_source_lines(answer: str) -> str:
    cleaned_lines = []
    for line in answer.splitlines():
        if re.match(r"^\s*(source|sources)\s*[:：]", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\s*\[\d+(?:,\s*\d+)*\]\s*source\s*[:：]", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

# ----------------------------
# Simple router before RAG
# ----------------------------

ROUTER_MIN_CHARS = 12
ROUTER_MIN_WORDS = 3

_META_PATTERNS: tuple[str, ...] = (
    # FR
    r"\b(résume|resume|résumer|resumer)\b.*\b(derni(è|e)re|dernier|last)\b",
    r"\b(historique|historique\s+du\s+chat|history)\b",
    r"\b(ma|mon|mes)\b.*\b(derni(è|e)re|dernier|last)\b.*\b(question|réponse|reponse|message)\b",
    # EN
    r"\b(summarize|summary)\b.*\b(last|previous)\b",
    r"\bwhat\s+did\s+i\s+ask\b|\bwhat\s+was\s+my\s+last\b",
)


def _is_meta_query(text: str) -> bool:
    q = (text or "").strip().lower()
    return any(re.search(p, q) for p in _META_PATTERNS)


def _is_too_vague(text: str) -> bool:
    q = (text or "").strip()
    if len(q) < ROUTER_MIN_CHARS:
        return True
    words = re.findall(r"\w+", q, flags=re.UNICODE)
    return len(words) < ROUTER_MIN_WORDS


def _get_docs_preflight(
    retriever: Any,
    question: str,
    *,
    retrieval_k: int | None,
    score_threshold: float | None,
) -> list:
    """Même logique que RagChat._get_docs, mais utilisable par le routeur (hors classe)."""
    if score_threshold is None:
        if hasattr(retriever, "invoke"):
            return retriever.invoke(question)
        return retriever.get_relevant_documents(question)

    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore and hasattr(vectorstore, "similarity_search_with_score"):
        k = retrieval_k if retrieval_k is not None else getattr(retriever, "search_kwargs", {}).get("k", 4)
        results = vectorstore.similarity_search_with_score(question, k=k)
        docs = [doc for doc, score in results if score <= score_threshold]
        if not docs and results:
            best_score = min(score for _, score in results)
            logger.info(
                "Router: no docs under score threshold %s (best=%s).",
                score_threshold,
                best_score,
            )
        return docs

    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


def route(
    question: str,
    history: list[BaseMessage],
    retriever: Any,
    *,
    retrieval_k: int | None = None,
    score_threshold: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Returns (mode, payload) where mode in:
      - "clarify": question too vague -> ask for clarification
      - "memory": meta question about history -> memory-only (no retrieval)
      - "no_docs": no relevant docs (score too low) -> fixed message
      - "rag": proceed with RAG using retrieved docs (payload contains "docs")
    """
    q = (question or "").strip()

    if _is_meta_query(q):
        return "memory", {"reason": "meta_query"}

    if _is_too_vague(q):
        return "clarify", {
            "answer": (
                "Ta question est un peu vague. Peux-tu préciser ce que tu cherches "
                "(thème, chapitre, contexte) dans le manuel de gestion UQAC ?"
            )
        }

    docs = _get_docs_preflight(
        retriever,
        q,
        retrieval_k=retrieval_k,
        score_threshold=score_threshold,
    )

    if not docs:
        return "no_docs", {"answer": "Je ne trouve pas l’information dans le manuel."}

    return "rag", {"docs": docs}


def answer_from_memory_only(question: str, history: list[BaseMessage], llm: ChatOllama) -> str:
    """Répond uniquement à partir de l'historique (pas de retrieval)."""
    if not history:
        return "Je n’ai pas encore d’historique dans cette session."

    q = (question or "").strip().lower()

    # Si l'utilisateur demande un résumé de la conversation entière
    if re.search(r"\b(résume|resume|summary|summarize)\b", q) and re.search(
        r"\b(conversation|chat|historique|history)\b", q
    ):
        return summarize_history(history, llm)

    # Sinon, on renvoie la dernière question/réponse.
    last_user = None
    last_ai = None
    for msg in reversed(history):
        if last_ai is None and isinstance(msg, AIMessage):
            # on enlève [1], [2]... pour la lisibilité
            last_ai = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", msg.content).strip()
        if last_user is None and isinstance(msg, HumanMessage):
            last_user = msg.content
        if last_user and last_ai:
            break

    if not last_user and not last_ai:
        return "Je ne trouve pas d’éléments exploitables dans l’historique."

    wants_q = bool(re.search(r"\b(question|ask)\b", q))
    wants_a = bool(re.search(r"\b(réponse|reponse|answer)\b", q))

    if wants_q and not wants_a:
        return f"Dernière question : {last_user}" if last_user else "Je ne trouve pas de dernière question."
    if wants_a and not wants_q:
        return f"Dernière réponse : {last_ai}" if last_ai else "Je ne trouve pas de dernière réponse."

    if last_user and last_ai:
        return f"Dernière question : {last_user}\n\nDernière réponse : {last_ai}"
    return last_user or last_ai


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
        # Support both old and new retriever interfaces.
        if self.score_threshold is None:
            if hasattr(self.retriever, "invoke"):
                return self.retriever.invoke(question)
            return self.retriever.get_relevant_documents(question)

        vectorstore = getattr(self.retriever, "vectorstore", None)
        if vectorstore and hasattr(vectorstore, "similarity_search_with_score"):
            if self.retrieval_k is not None:
                k = self.retrieval_k
            else:
                k = getattr(self.retriever, "search_kwargs", {}).get("k", 4)
            results = vectorstore.similarity_search_with_score(question, k=k)
            docs = [doc for doc, score in results if score <= self.score_threshold]
            if not docs and results:
                best_score = min(score for _, score in results)
                logger.info(
                    "No documents under score threshold %s (best=%s).",
                    self.score_threshold,
                    best_score,
                )
            logger.info("Retrieved %s documents after applying score threshold.", len(docs))
            return docs

        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(question)
        return self.retriever.get_relevant_documents(question)

    # def ask(
    #     self,
    #     question: str,
    #     stream: bool = False,
    #     on_chunk: Callable[[str], None] | None = None,
    # ) -> tuple[str, list]:
    #     """Retrieve context and generate a cited answer."""
    #     # Retrieve context and generate an answer with citations.
    #     docs = self._get_docs(question)
    #     context = format_docs(docs)
    #     messages = self.prompt.format_messages(
    #         input=question, context=context, history=self.history
    #     )
    #     if stream:
    #         parts: list[str] = []
    #         for chunk in self.llm.stream(messages):
    #             text = getattr(chunk, "content", str(chunk))
    #             if not text:
    #                 continue
    #             parts.append(text)
    #             if on_chunk:
    #                 on_chunk(text)
    #         answer = "".join(parts)
    #     else:
    #         response = self.llm.invoke(messages)
    #         answer = response.content if hasattr(response, "content") else str(response)
    #     answer = _strip_source_lines(answer)
    #     self._append_history(question, answer)
    #     return answer, docs
    
    def ask(
        self,
        question: str,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list]:
        """Retrieve context and generate a cited answer."""
        # --- NEW: decide path before running full RAG
        mode, payload = route(
            question,
            self.history,
            self.retriever,
            retrieval_k=self.retrieval_k,
            score_threshold=self.score_threshold,
        )

        # 1) Question vague -> clarification (pas de retrieval, pas de LLM)
        if mode == "clarify":
            answer = payload["answer"]
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []

        # 2) Requête méta -> mémoire-seule (pas de retrieval)
        if mode == "memory":
            answer = answer_from_memory_only(question, self.history, self.llm)
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []
        # 3) Pas de docs pertinents -> phrase fixe (pas de LLM)
        if mode == "no_docs":
            answer = payload["answer"]
            if stream and on_chunk:
                on_chunk(answer)
            self._append_history(question, answer)
            return answer, []
        # 4) RAG normal: on réutilise les docs déjà récupérés par le routeur
        docs = payload["docs"]

        # --- Existing RAG code (identique à avant)
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
