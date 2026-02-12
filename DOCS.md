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

Files: `src/chatbot_uqac/rag/engine.py`, `src/chatbot_uqac/rag/routing.py`

- Retrieval now uses a lightweight hybrid pipeline:
  - Dense retrieval from Chroma.
  - Lexical retrieval from SQLite (`documents`) using FTS5 when available,
    otherwise a `LIKE` fallback.
  - Fusion with Reciprocal Rank Fusion (RRF), then URL deduplication.
- For dense retrieval, the system can use both:
  - the original user question,
  - and a rewritten standalone query for follow-up questions.
- `RETRIEVAL_K` still controls the final number of chunks sent to generation.
- When a score threshold is configured, dense candidates are filtered with
  score <= threshold (Chroma distance: lower is better).
- Retrieval helper logic is centralized in `routing.py` (`retrieve_docs`,
  `retrieve_docs_hybrid`) and reused by the router.
- A system prompt tells the model to use only the provided context.
- The answer is produced by a "stuff" chain (all retrieved docs concatenated).
- Sources are extracted from chunk metadata and printed in the UI.

## Query routing layer

Files: `src/chatbot_uqac/rag/routing.py`, `src/chatbot_uqac/rag/engine.py`

- Before standard RAG generation, each question goes through a lightweight
  router (`route`).
- The router can select one of four modes:
  - `chitchat`: social message (greetings/small talk), answered directly by the
    model without retrieval context.
  - `memory`: for explicit chat-memory requests (for example, last question or
    conversation summary), answered from history only.
  - `rag`: standard retrieval + generation when relevant docs are found.
  - `clarify`: ask the user to clarify when intent is classified as `unclear`
    with sufficient confidence (no document retrieval in that branch).
  - `no_docs`: return a fixed fallback when no relevant document is found.
- Routing is hybrid: deterministic checks + LLM intent classification
  (`domain|memory|chitchat|unclear`) with confidence-based decisions.
- For domain queries, the router rewrites follow-up questions into a
  standalone retrieval query (`rewrite_retrieval_query`) before retrieval.
- Domain retrieval uses `retrieve_docs_hybrid` (dense + lexical + RRF), which
  improves robustness when semantic retrieval misses exact keywords.
- This routing step improves behavior for social queries while preserving
  retrieval-first grounding for domain questions.
- Routing observability is logged in `routing.py` (intent, confidence, selected
  mode, reason, query rewrite, and hybrid retrieval stats). Use
  `LOG_LEVEL=INFO` (or `DEBUG`) to inspect decisions.

## Memory behavior

Files: `src/chatbot_uqac/rag/engine.py`, `src/chatbot_uqac/rag/routing.py`

- Chat memory is stored as a list of messages (human/assistant).
- When the history exceeds a threshold, older messages are summarized into a
  single `SystemMessage`, and only the most recent exchanges are kept.
- Summaries are factual and strip citations/URLs; they are prepended to the
  prompt history.
- The last N turns are retained even after summarization to keep recent detail.
- Memory is local to a session (CLI process or Streamlit session).
- For memory-only queries, `answer_from_memory_only` forwards the conversation
  transcript + user request to the LLM, with no retrieval step.

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
- `REQUEST_TIMEOUT`: HTTP timeout (seconds) for crawling and downloads.
- `USER_AGENT`: user agent string used for HTTP requests.
- `RETRIEVAL_K`: number of chunks to retrieve per query.
- `RETRIEVAL_SCORE_THRESHOLD`: distance threshold for retrieval filtering.
- `STREAMING_ENABLED`: enable streaming responses in CLI and Streamlit.
- `HISTORY_MAX_MESSAGES`: max turns kept when summarization does not trigger.
- `SUMMARIZE_THRESHOLD`: number of messages that triggers history summarization.
- `KEEP_RECENT_MESSAGES`: number of recent messages kept after summarization.
- `LOG_LEVEL`: logging verbosity (e.g. `DEBUG`, `INFO`).
