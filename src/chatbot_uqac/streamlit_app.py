"""Streamlit UI for the UQAC chatbot."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from chatbot_uqac.compat import ensure_supported_python


ensure_supported_python()

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
from chatbot_uqac.rag.engine import RagChat, build_llm, extract_grouped_source_refs
from chatbot_uqac.rag.vectorstore import build_embeddings, load_vectorstore


setup_logging()
st.set_page_config(
    page_title="Assistant Virtuel UQAC",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _sources_links_html(sources: list | None) -> str:
    if not sources:
        return ""

    rows: list[str] = []
    for entry in sources:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        indices, url = entry
        if not url:
            continue
        if isinstance(indices, int):
            label = str(indices)
        elif isinstance(indices, (list, tuple)):
            label = ",".join(str(i) for i in indices)
        else:
            label = str(indices)
        rows.append(
            f'[{label}] <a href="{url}" target="_blank" style="color: #548427;">{url}</a>'
        )
    return "<br>".join(rows)

# Chargement du CSS personnalisé
css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# JavaScript pour styliser les messages utilisateur
st.markdown("""
<script>
function styleUserMessages() {
    const messages = document.querySelectorAll('[data-testid^="stChatMessage"]');
    messages.forEach(msg => {
        const avatar = msg.querySelector('[data-testid="chatAvatarIcon-user"]');
        if (avatar) {
            msg.classList.add('user-message-styled');
        }
    });
}
// Run on load and observe for changes
const observer = new MutationObserver(styleUserMessages);
observer.observe(document.body, { childList: true, subtree: true });
setTimeout(styleUserMessages, 500);
</script>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    # Logo UQAC
    logo_path = Path(__file__).parent / "assets" / "UQAC_Logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=200)
    
    st.markdown("### Assistant Virtuel")
    st.markdown(
        """
        Cet assistant utilise l'IA pour répondre à vos questions sur les guides de gestion de l'UQAC.
        
        **Sources:**
        - Guides officiels
        - Documents administratifs
        """
    )
    
    st.divider()
    
    # Dataset Info in Sidebar
    if "dataset_info" not in st.session_state:
        # Initial check logic preserved
        pass 
       
    # We'll populate dataset info later in the script but user sees it here
    placeholder_info = st.empty()


# --- Main Content ---
st.markdown('''
<h1 style="text-align: center; color: #548427; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; padding-bottom: 0.5rem; border-bottom: 1px solid #ddd; margin-bottom: 2rem;">
    Assistant UQAC
</h1>
''', unsafe_allow_html=True)

if not DB_PATH.exists() or not CHROMA_DIR.exists():
    # Avoid running the chat without an indexed corpus.
    st.warning("⚠️ Données locales manquantes. Lancez: `python -m chatbot_uqac.ingest`")
    st.stop()

if "chat" not in st.session_state:
    # Build the retriever/LLM once per session.
    with st.spinner("Initialisation de l'assistant..."):
        try:
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
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

# Dataset Info Logic (Preserved)
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

# Display dataset info in sidebar
doc_count, chunk_count = st.session_state.dataset_info
if doc_count is not None or chunk_count is not None:
    info_text = ""
    if doc_count: info_text += f"📄 **Documents:** {doc_count}\n\n"
    if chunk_count: info_text += f"🧩 **Fragments:** {chunk_count}"
    placeholder_info.markdown(info_text)

# Chat Interface
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
        <div style="background-color: #548427; border-left: 4px solid #3d611c; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 12px;">
            <div style="background-color: white; border: 2px solid #3d611c; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#548427" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            </div>
            <p style="color: white; margin: 0; flex-grow: 1;">{message["content"]}</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        sources = message.get("sources")
        sources_html = ""
        if sources:
            sources_links = _sources_links_html(sources)
            if sources_links:
                sources_html = f'<div style="font-size: 0.85rem; margin-top: 12px; padding-top: 8px; border-top: 1px solid #ddd;">📚 <b>Sources:</b><br>{sources_links}</div>'
        st.markdown(f'''
        <div style="background-color: #f4f4f4; border-left: 4px solid #548427; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 12px;">
            <div style="background-color: #548427; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>
            </div>
            <div style="color: #333; margin: 0; flex-grow: 1;">{message["content"]}{sources_html}</div>
        </div>
        ''', unsafe_allow_html=True)

processing = st.session_state.get("pending_question") is not None
st.session_state.busy = processing

question = st.chat_input(
    "Posez votre question sur l'UQAC...",
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
    st.markdown(f'''
    <div style="background-color: #548427; border-left: 4px solid #3d611c; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
        <div style="background-color: white; border: 2px solid #3d611c; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#548427" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
        </div>
        <p style="color: white; margin: 0; flex-grow: 1;">{question}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Marqueur pour le style CSS du conteneur assistant
    st.markdown('<div class="assistant-loading-marker"></div>', unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        with st.spinner("Traitement de votre demande..."):
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
        
        # Show cited sources grouped by unique URL (e.g., [1,2]).
        sources = extract_grouped_source_refs(docs, answer)
        if sources:
            sources_links = _sources_links_html(sources)
            if sources_links:
                st.markdown(f'<div class="source-container">📚 <b>Sources:</b><br>{sources_links}</div>', unsafe_allow_html=True)
             
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()
