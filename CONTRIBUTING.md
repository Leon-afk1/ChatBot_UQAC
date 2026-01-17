# Contributing

## Setup

1) Create a virtual env and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pip install -e .
```

2) (Optional) Run ingestion if you need fresh data:

```bash
python -m chatbot_uqac.ingest
```

## Guidelines

- Keep the codebase simple and readable.
- Prefer small, focused changes.
- Use a new branch for each change and commit regularly with clear messages.
- Do not commit `data/` or other generated artifacts.
- Update `DOCS.md` when you change architecture or behavior.

## Quick checks

- CLI: `python -m chatbot_uqac.cli`
- Streamlit: `streamlit run src/chatbot_uqac/streamlit_app.py`
