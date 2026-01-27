# ChatBot UQAC (RAG)

Local RAG chatbot for the UQAC management guide. Uses:

- Ollama for the local LLM and embeddings
- LangChain for RAG
- Rich for CLI
- Streamlit for a simple web UI

## Quick start

1) Install Ollama: https://ollama.com/download

2) Create a virtual env and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3) Install the project package (needed for `python -m chatbot_uqac.*`):

```bash
python -m pip install -e .
```

4) Pull Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

5) Configure optional env vars:

```bash
copy .env.example .env
```

6) Ingest the UQAC management guide:

```bash
python -m chatbot_uqac.ingest
```

7) Run the CLI:

```bash
python -m chatbot_uqac.cli
```

8) Run the Streamlit app:

```bash
python -m streamlit run src/chatbot_uqac/streamlit_app.py
```

## Notes

- Data is persisted under `data/` (SQLite + Chroma).
- If you change chunking, embedding settings, or crawl scope, delete `data/` and re-run ingestion.
- Ollama must be running locally on `http://localhost:11434`.
- Detailed architecture and extension points are in `DOCS.md`.
- Contribution guidelines are in `CONTRIBUTING.md`.
