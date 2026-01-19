"""Streamlit UI for the UQAC chatbot."""

from __future__ import annotations

import streamlit as st

from chatbot_uqac.config import (
    CHROMA_DIR,
    DB_PATH,
    HISTORY_MAX_MESSAGES,
    KEEP_RECENT_MESSAGES,
    SUMMARIZE_THRESHOLD,
)
from chatbot_uqac.rag.engine import RagChat, build_llm, extract_sources
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()
    st.session_state.chat = RagChat(
        retriever, 
        llm,
        max_history_messages=HISTORY_MAX_MESSAGES,
        summarize_threshold=SUMMARIZE_THRESHOLD,
        keep_recent=KEEP_RECENT_MESSAGES
    )
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        sources = message.get("sources")
        if sources:
            st.caption("Sources:")
            st.write("\n".join(sources))

question = st.chat_input("Ask a question about the UQAC guide...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, docs = st.session_state.chat.ask(question)
        st.write(answer)
        # Only show sources that were cited by the model.
        sources = extract_sources(docs, answer)
        if sources:
            st.caption("Sources:")
            st.write("\n".join(sources))
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
