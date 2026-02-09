"""Runtime compatibility checks.

This project depends on libraries that currently require Python < 3.14.
In particular, some of the LangChain/Chroma dependency chain relies on
Pydantic v1 compatibility layers which break on Python 3.14+.
"""

from __future__ import annotations

import sys


MIN_PYTHON = (3, 10)
MAX_EXCLUSIVE_PYTHON = (3, 14)


def ensure_supported_python() -> None:
    """Fail fast with a clear message on unsupported Python versions."""
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            "Unsupported Python version. "
            f"This project requires Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}. "
            f"Detected: {sys.version_info.major}.{sys.version_info.minor}."
        )

    if sys.version_info >= MAX_EXCLUSIVE_PYTHON:
        raise RuntimeError(
            "Unsupported Python version. "
            "Python 3.14+ is not supported by the current dependency set "
            "(LangChain/Chroma/Pydantic v1 compatibility). "
            "Please use Python 3.12 or 3.13 and recreate the virtualenv. "
            f"Detected: {sys.version_info.major}.{sys.version_info.minor}."
        )
