"""Shared test fixtures and test-path bootstrap."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class FakeDoc:
    """Simple stand-in for LangChain document objects."""

    page_content: str
    metadata: dict = field(default_factory=dict)
