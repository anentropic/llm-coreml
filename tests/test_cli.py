"""Tests for the CLI commands."""

from __future__ import annotations

from pathlib import Path

import click
from click.testing import CliRunner

from llm_coreml import register_commands
from llm_coreml.registry import get_model, list_models


def _make_cli() -> click.Group:
    """Create a minimal Click group and register our commands on it."""

    @click.group()
    def cli() -> None:
        pass

    register_commands(cli)
    return cli


class TestCLI:
    def test_add_model(self, registry_dir: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()
        cli = _make_cli()

        result = runner.invoke(
            cli,
            ["coreml", "add", "test", str(model_path), "--tokenizer", "org/tok"],
        )
        assert result.exit_code == 0
        assert "Added model coreml/test" in result.output

        model = get_model("test")
        assert model is not None
        assert model["tokenizer"] == "org/tok"

    def test_add_resolves_path(self, registry_dir: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()
        cli = _make_cli()

        result = runner.invoke(
            cli,
            ["coreml", "add", "test", str(model_path), "--tokenizer", "org/tok"],
        )
        assert result.exit_code == 0

        model = get_model("test")
        assert model is not None
        assert Path(model["path"]).is_absolute()

    def test_list_empty(self, registry_dir: Path) -> None:
        runner = CliRunner()
        cli = _make_cli()

        result = runner.invoke(cli, ["coreml", "list"])
        assert result.exit_code == 0
        assert "No CoreML models registered" in result.output

    def test_list_shows_models(self, registry_dir: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()
        cli = _make_cli()

        runner.invoke(cli, ["coreml", "add", "test", str(model_path), "--tokenizer", "org/tok"])
        result = runner.invoke(cli, ["coreml", "list"])
        assert result.exit_code == 0
        assert "coreml/test" in result.output
        assert "org/tok" in result.output

    def test_remove_existing(self, registry_dir: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        model_path = tmp_path / "model.mlpackage"
        model_path.mkdir()
        cli = _make_cli()

        runner.invoke(cli, ["coreml", "add", "test", str(model_path), "--tokenizer", "org/tok"])
        result = runner.invoke(cli, ["coreml", "remove", "test"])
        assert result.exit_code == 0
        assert "Removed model coreml/test" in result.output
        assert list_models() == {}

    def test_remove_nonexistent(self, registry_dir: Path) -> None:
        runner = CliRunner()
        cli = _make_cli()

        result = runner.invoke(cli, ["coreml", "remove", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.output
