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
            _configure_noisy_loggers()
        return

    log_level = (level or LOG_LEVEL or "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _configure_noisy_loggers()
    _LOGGING_CONFIGURED = True


def _configure_noisy_loggers() -> None:
    """Reduce noise from third-party loggers in DEBUG mode."""
    # Streamlit uses watchdog file observers that can flood DEBUG logs with inotify events.
    logging.getLogger("watchdog").setLevel(logging.INFO)
    # Networking and image decoding libraries can flood DEBUG logs in Streamlit mode.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.WARNING)
