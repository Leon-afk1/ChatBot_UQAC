"""Tests for compatibility and logging helpers."""

from __future__ import annotations

import logging

import pytest

import chatbot_uqac.compat as compat
import chatbot_uqac.logging_config as logging_config


def test_ensure_supported_python_raises_when_below_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "MIN_PYTHON", (99, 0))
    with pytest.raises(RuntimeError, match="Unsupported Python version"):
        compat.ensure_supported_python()


def test_ensure_supported_python_raises_when_above_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "MIN_PYTHON", (3, 0))
    monkeypatch.setattr(compat, "MAX_EXCLUSIVE_PYTHON", (0, 0))
    with pytest.raises(RuntimeError, match="Python 3.14\\+ is not supported"):
        compat.ensure_supported_python()


def test_setup_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_basic_config(**kwargs):
        calls.append(("basic", kwargs))

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(logging_config, "_LOGGING_CONFIGURED", False)

    logging_config.setup_logging("debug")
    logging_config.setup_logging("info")

    assert len(calls) == 1
    assert calls[0][1]["level"] == "DEBUG"
