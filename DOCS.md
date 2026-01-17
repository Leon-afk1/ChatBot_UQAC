# ChatBot UQAC - How it works

This document explains the current architecture and flow: ingestion, RAG, memory,
and the two user interfaces.

## Overview

The project ingests the UQAC management guide, extracts content from HTML and
PDF pages, stores it locally, and builds a vector index for retrieval. A local
Ollama model answers questions using LangChain, with sources shown as URLs.

## Data flow (high level)

1) Crawl the site to collect relevant URLs.
2) Download each URL and extract text.
3) Persist raw text in SQLite for reuse.
4) Split text into chunks and embed with Ollama.
5) Store embeddings in a local Chroma vector database.
6) At query time, retrieve top chunks and ask the LLM to answer.
7) Display the answer with the URLs used as sources.

## Ingestion and scraping

Files: `src/chatbot_uqac/ingest/crawler.py`, `src/chatbot_uqac/ingest/loaders.py`,
`src/chatbot_uqac/ingest/store.py`, `src/chatbot_uqac/ingest/runner.py`

- The crawler starts at `UQAC_BASE_URL` and breadth-first explores links.
- Only links within the same domain are kept, and paths are restricted to the
  base path (plus `/wp-content/` for attachments). Assets and media are skipped.
- HTML pages are parsed and only the text inside:
  - `div.entry-header`
  - `div.entry-content`
  is retained.
- PDF files are downloaded to a temporary file and extracted with `pypdf`.

## Local persistence

Files: `src/chatbot_uqac/ingest/store.py`, `src/chatbot_uqac/config.py`

- Raw content is stored in `data/documents.sqlite3`.
- Each URL has a content hash so unchanged pages are skipped on re-ingest.
- The vector database is stored under `data/chroma/`.

## Chunking and embeddings

Files: `src/chatbot_uqac/ingest.py`, `src/chatbot_uqac/rag/vectorstore.py`

- Text is split with a recursive character splitter:
  - `CHUNK_SIZE` and `CHUNK_OVERLAP` control chunk boundaries.
- Each chunk is embedded using the Ollama embeddings model.
- The Chroma vectorstore is persisted on disk.

## Retrieval and RAG answer generation

Files: `src/chatbot_uqac/rag/engine.py`

- The retriever fetches the most relevant chunks (default `k=4`).
- A system prompt tells the model to use only the provided context.
- The answer is produced by a "stuff" chain (all retrieved docs concatenated).
- Sources are extracted from chunk metadata and printed in the UI.

## Memory behavior

Files: `src/chatbot_uqac/rag/engine.py`

- A simple chat memory is maintained as a list of messages.
- The last N turns (default 5) are injected into the prompt as `history`.
- Memory is local to a session (CLI process or Streamlit session).

## User interfaces

Files: `src/chatbot_uqac/cli.py`, `src/chatbot_uqac/streamlit_app.py`

- CLI uses Rich for a simple interactive chat loop.
- Streamlit UI keeps the chat session in `st.session_state`.
- Both UIs check for existing data and ask to run ingestion if missing.

## Configuration

Files: `src/chatbot_uqac/config.py`, `.env.example`

Important settings:

- `OLLAMA_BASE_URL`: where Ollama is running.
- `OLLAMA_CHAT_MODEL`: chat model name.
- `OLLAMA_EMBED_MODEL`: embedding model name.
- `UQAC_BASE_URL`: start URL for crawling.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: chunking behavior.
- `MAX_PAGES`: crawl limit.

## Possible Extension points (prioritized)

Priority 0 (required for a reliable, successful project):

- Ingestion QA and logs: track counts, failures, and per-URL status; export a
  simple report to verify coverage and catch missing pages.
- Crawl scope controls: make allowed paths explicit in config, and document
  how to include or exclude specific sections.
- Data refresh workflow: add a safe re-ingest mode with cache validation and a
  clear "rebuild index" procedure.
- Retrieval correctness: add basic evaluation questions and confirm that the
  cited sources match the answer content.

Priority 1 (quality improvements):

- Retrieval upgrades: MMR or reranking, and metadata filters to limit results
  to the most relevant sections.
- Memory upgrades: summarize chat history to keep prompts small and relevant.
- LangGraph: introduce a simple router (QA vs. "I do not know" vs. clarification).
- UI polish: show ingestion status and dataset size in CLI and Streamlit.

Priority 2 (nice-to-have):

- Scheduled re-ingestion with change detection.
- User feedback loop (thumbs up/down + notes) to tune prompts and retrieval.
- Advanced analytics (top questions, failure cases).
