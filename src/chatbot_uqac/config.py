"""Configuration and environment settings for the project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "documents.sqlite3"
CHROMA_DIR = DATA_DIR / "chroma"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

UQAC_BASE_URL = os.getenv("UQAC_BASE_URL", "https://www.uqac.ca/mgestion/")
MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
USER_AGENT = os.getenv(
    "USER_AGENT", "ChatBotUQAC/0.1 (+https://www.uqac.ca/mgestion/)"
)

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
_threshold_raw = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
RETRIEVAL_SCORE_THRESHOLD = float(_threshold_raw) if _threshold_raw else None
STREAMING_ENABLED = _env_bool("STREAMING_ENABLED", True)

# Memory and summarization settings
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "5"))
SUMMARIZE_THRESHOLD = int(os.getenv("SUMMARIZE_THRESHOLD", "10"))
KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "6"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
