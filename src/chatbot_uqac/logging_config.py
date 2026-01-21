"""Logging configuration helpers."""

from __future__ import annotations

import logging

from chatbot_uqac.config import LOG_LEVEL


_LOGGING_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once for the application."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        if level:
            logging.getLogger().setLevel(level.upper())
        return

    log_level = (level or LOG_LEVEL or "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGING_CONFIGURED = True
