"""Tests for the model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_coreml.registry import add_model, get_model, get_registry_path, list_models, remove_model

if TYPE_CHECKING:
    from pathlib import Path


class TestRegistry:
    def test_registry_path_inside_user_dir(self, registry_dir: Path) -> None:
        assert get_registry_path() == registry_dir / "llm-coreml.json"

    def test_list_empty(self, registry_dir: Path) -> None:
        assert list_models() == {}

    def test_add_and_get(self, registry_dir: Path) -> None:
        add_model("test", "/path/to/model.mlpackage", "org/tokenizer")
        model = get_model("test")
        assert model is not None
        assert model["path"] == "/path/to/model.mlpackage"
        assert model["tokenizer"] == "org/tokenizer"

    def test_add_overwrites(self, registry_dir: Path) -> None:
        add_model("test", "/old/path", "org/old")
        add_model("test", "/new/path", "org/new")
        model = get_model("test")
        assert model is not None
        assert model["path"] == "/new/path"

    def test_list_multiple(self, registry_dir: Path) -> None:
        add_model("a", "/a", "org/a")
        add_model("b", "/b", "org/b")
        models = list_models()
        assert len(models) == 2
        assert "a" in models
        assert "b" in models

    def test_remove_existing(self, registry_dir: Path) -> None:
        add_model("test", "/path", "org/tok")
        assert remove_model("test") is True
        assert get_model("test") is None

    def test_remove_nonexistent(self, registry_dir: Path) -> None:
        assert remove_model("nope") is False

    def test_get_nonexistent(self, registry_dir: Path) -> None:
        assert get_model("nope") is None
