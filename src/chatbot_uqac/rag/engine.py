"""RAG prompting and chat session orchestration."""

from __future__ import annotations

import re
from typing import Iterable

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chatbot_uqac.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


SYSTEM_PROMPT = (
    "You are a helpful assistant for the UQAC management guide. "
    "Use only the provided context. If the answer is not in the context, "
    "say you do not know. Keep answers concise. "
    "Cite sources using square brackets like [1] based on the context numbering."
)


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


class RagChat:
    """Session-scoped RAG chat with short-term memory and periodic summarization."""

    def __init__(
        self, 
        retriever, 
        llm: ChatOllama, 
        max_history_messages: int = 5,
        summarize_threshold: int = 10,
        keep_recent: int = 6
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt = build_prompt()
        self.history: list[BaseMessage] = []
        self.max_history_messages = max_history_messages
        self.summarize_threshold = summarize_threshold  # Threshold to trigger summarization
        self.keep_recent = keep_recent  # Number of recent messages to keep (must be even)

    def _get_docs(self, question: str) -> list:
        # Support both old and new retriever interfaces.
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(question)
        return self.retriever.get_relevant_documents(question)

    def ask(self, question: str) -> tuple[str, list]:
        """Retrieve context and generate a cited answer."""
        # Retrieve context and generate an answer with citations.
        docs = self._get_docs(question)
        context = format_docs(docs)
        messages = self.prompt.format_messages(
            input=question, context=context, history=self.history
        )
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        self._append_history(question, answer)
        return answer, docs

    def _append_history(self, question: str, answer: str) -> None:
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        
        # Print history status
        print(f"\n{'='*70}")
        print(f"📝 HISTORIQUE ACTUEL: {len(self.history)} messages")
        print(f"{'='*70}")
        for i, msg in enumerate(self.history):
            msg_type = "👤 USER" if isinstance(msg, HumanMessage) else "🤖 ASSISTANT" if isinstance(msg, AIMessage) else "📋 RÉSUMÉ"
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"[{i}] {msg_type}: {content}")
        print(f"{'='*70}\n")
        
        # Trigger summarization if history exceeds threshold
        if len(self.history) > self.summarize_threshold:
            print(f"⚠️  Seuil dépassé ({len(self.history)} > {self.summarize_threshold}), déclenchement du résumé...\n")
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
            
            print(f"✨ COMPRESSION EFFECTUÉE: {len(old_messages) + len(recent_messages)} → {len(self.history)} messages")
            print(f"   Résumé créé + {len(recent_messages)} messages récents conservés\n")


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
