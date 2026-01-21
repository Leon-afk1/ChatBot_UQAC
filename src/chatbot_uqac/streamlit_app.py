"""Streamlit UI for the UQAC chatbot."""

from __future__ import annotations

import sqlite3

import streamlit as st

from chatbot_uqac.config import (
    CHROMA_DIR,
    DB_PATH,
    HISTORY_MAX_MESSAGES,
    KEEP_RECENT_MESSAGES,
    RETRIEVAL_K,
    RETRIEVAL_SCORE_THRESHOLD,
    OLLAMA_CHAT_MODEL,
    STREAMING_ENABLED,
    SUMMARIZE_THRESHOLD,
)
from chatbot_uqac.logging_config import setup_logging
from chatbot_uqac.rag.engine import RagChat, build_llm, extract_sources
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


setup_logging()
st.set_page_config(page_title="ChatBot UQAC")
st.title("ChatBot UQAC")

if not DB_PATH.exists() or not CHROMA_DIR.exists():
    # Avoid running the chat without an indexed corpus.
    st.warning("Missing local data. Run: python -m chatbot_uqac.ingest")
    st.stop()

if "chat" not in st.session_state:
    # Build the retriever/LLM once per session.
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = build_llm()
    st.session_state.chat = RagChat(
        retriever,
        llm,
        max_history_messages=HISTORY_MAX_MESSAGES,
        summarize_threshold=SUMMARIZE_THRESHOLD,
        keep_recent=KEEP_RECENT_MESSAGES,
        retrieval_k=RETRIEVAL_K,
        score_threshold=RETRIEVAL_SCORE_THRESHOLD,
    )
    st.session_state.messages = []
    st.session_state.busy = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "dataset_info" not in st.session_state:
    try:
        chunk_count = st.session_state.chat.retriever.vectorstore._collection.count()
    except Exception:
        chunk_count = None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except Exception:
        doc_count = None
    st.session_state.dataset_info = (doc_count, chunk_count)

doc_count, chunk_count = st.session_state.dataset_info
if doc_count is not None or chunk_count is not None:
    parts = []
    if doc_count is not None:
        parts.append(f"Documents: {doc_count}")
    if chunk_count is not None:
        parts.append(f"Chunks: {chunk_count}")
    st.caption("Dataset size: " + " | ".join(parts))
st.caption(f"Chat model: {OLLAMA_CHAT_MODEL}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        sources = message.get("sources")
        if sources:
            st.caption("Sources:")
            st.write("\n".join(sources))

processing = st.session_state.pending_question is not None
st.session_state.busy = processing

question = st.chat_input(
    "Ask a question about the UQAC guide...",
    disabled=processing,
)
if question and not processing:
    st.session_state.pending_question = question
    st.session_state.busy = True
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()

if processing:
    question = st.session_state.pending_question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if STREAMING_ENABLED:
                    placeholder = st.empty()
                    buffer: list[str] = []

                    def _write_chunk(text: str) -> None:
                        buffer.append(text)
                        placeholder.markdown("".join(buffer))

                    answer, docs = st.session_state.chat.ask(
                        question, stream=True, on_chunk=_write_chunk
                    )
                    placeholder.markdown(answer)
                else:
                    answer, docs = st.session_state.chat.ask(question)
                    st.write(answer)
            finally:
                st.session_state.busy = False
                st.session_state.pending_question = None
        # Only show sources that were cited by the model.
        sources = extract_sources(docs, answer)
        if sources:
            st.caption("Sources:")
            st.write("\n".join(sources))
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()
