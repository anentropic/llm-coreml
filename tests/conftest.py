"""Shared pytest fixtures for llm_coreml test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the llm user_dir to a temp directory for registry tests."""
    monkeypatch.setattr("llm.user_dir", lambda: str(tmp_path))
    return tmp_path
