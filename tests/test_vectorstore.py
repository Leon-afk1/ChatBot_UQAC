"""Tests for vectorstore helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest

import chatbot_uqac.rag.vectorstore as vectorstore


def test_ensure_writable_dir_creates_directory_and_probe(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    vectorstore._ensure_writable_dir(target, "Chroma directory")
    assert target.exists()
    assert target.is_dir()


def test_ensure_writable_dir_raises_when_access_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "chroma"
    target.mkdir()

    monkeypatch.setattr(vectorstore.os, "access", lambda *_: False)
    with pytest.raises(RuntimeError, match="not writable"):
        vectorstore._ensure_writable_dir(target, "Chroma directory")


def test_build_embeddings_uses_configured_model_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeEmbeddings:
        def __init__(self, model: str, base_url: str):
            captured["model"] = model
            captured["base_url"] = base_url

    monkeypatch.setattr(vectorstore, "OllamaEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(vectorstore, "OLLAMA_EMBED_MODEL", "embed-model-test")
    monkeypatch.setattr(vectorstore, "OLLAMA_BASE_URL", "http://ollama-test")

    vectorstore.build_embeddings()

    assert captured == {
        "model": "embed-model-test",
        "base_url": "http://ollama-test",
    }


def test_load_vectorstore_initializes_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str] = {}

    def fake_ensure(path: Path, label: str) -> None:
        calls["ensure_path"] = str(path)
        calls["ensure_label"] = label

    class FakeChroma:
        def __init__(
            self,
            collection_name: str,
            embedding_function,
            persist_directory: str,
        ):
            calls["collection_name"] = collection_name
            calls["embedding_function"] = embedding_function
            calls["persist_directory"] = persist_directory

    fake_embeddings = object()
    monkeypatch.setattr(vectorstore, "_ensure_writable_dir", fake_ensure)
    monkeypatch.setattr(vectorstore, "Chroma", FakeChroma)
    monkeypatch.setattr(vectorstore, "CHROMA_DIR", Path("/tmp/chroma-test"))

    store = vectorstore.load_vectorstore(fake_embeddings)

    assert isinstance(store, FakeChroma)
    assert calls["collection_name"] == "uqac_mgestion"
    assert calls["embedding_function"] is fake_embeddings
    assert calls["persist_directory"] == "/tmp/chroma-test"
