"""JSON registry for CoreML model configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import llm


def get_registry_path() -> Path:
    """Return the path to the registry JSON file."""
    return Path(llm.user_dir()) / "llm-coreml.json"


def _read_registry() -> dict[str, dict[str, str]]:
    path = get_registry_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _write_registry(data: dict[str, Any]) -> None:
    path = get_registry_path()
    path.write_text(json.dumps(data, indent=2) + "\n")


def add_model(
    name: str, path: str, tokenizer: str, compute_units: str = "all"
) -> None:
    """Register a model with the given name, path, tokenizer ID, and compute units."""
    registry = _read_registry()
    registry[name] = {
        "path": path,
        "tokenizer": tokenizer,
        "compute_units": compute_units,
    }
    _write_registry(registry)


def remove_model(name: str) -> bool:
    """Remove a model by name. Returns True if it existed."""
    registry = _read_registry()
    if name not in registry:
        return False
    del registry[name]
    _write_registry(registry)
    return True


def list_models() -> dict[str, dict[str, str]]:
    """Return all registered models."""
    return _read_registry()


def get_model(name: str) -> dict[str, str] | None:
    """Return config for a single model, or None if not found."""
    return _read_registry().get(name)
