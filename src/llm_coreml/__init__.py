"""A plugin for llm that runs local CoreML .mlpackage model files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
import llm

from llm_coreml.registry import add_model, list_models, remove_model

if TYPE_CHECKING:
    from collections.abc import Callable


@llm.hookimpl
def register_models(register: Callable[..., object]) -> None:
    """Register all CoreML models from the local registry."""
    from llm_coreml.model import CoreMLModel

    for name, config in list_models().items():
        register(
            CoreMLModel(
                model_id=f"coreml/{name}",
                model_path=config["path"],
                tokenizer_id=config["tokenizer"],
                compute_units=config.get("compute_units", "all"),
            ),
        )


@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register the `llm coreml` command group."""

    @cli.group(name="coreml")
    def coreml_group() -> None:
        """Manage locally-registered CoreML models."""

    @coreml_group.command(name="add")
    @click.argument("name")
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--tokenizer", required=True, help="HuggingFace tokenizer model ID")
    @click.option(
        "--compute-units",
        type=click.Choice(["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"]),
        default="all",
        help="CoreML compute units to use (default: all)",
    )
    def add_cmd(name: str, path: str, tokenizer: str, compute_units: str) -> None:  # pyright: ignore[reportUnusedFunction]
        """
        Register a CoreML model.

        NAME is the model name (used as coreml/NAME).
        PATH is the path to the .mlpackage directory.
        """
        resolved = str(Path(path).resolve())
        add_model(name, resolved, tokenizer, compute_units)
        click.echo(f"Added model coreml/{name}")

    @coreml_group.command(name="list")
    def list_cmd() -> None:  # pyright: ignore[reportUnusedFunction]
        """List registered CoreML models."""
        models = list_models()
        if not models:
            click.echo("No CoreML models registered.")
            return
        for name, config in models.items():
            click.echo(f"coreml/{name}")
            click.echo(f"  Path:           {config['path']}")
            click.echo(f"  Tokenizer:      {config['tokenizer']}")
            click.echo(f"  Compute units:  {config.get('compute_units', 'all')}")

    @coreml_group.command(name="remove")
    @click.argument("name")
    def remove_cmd(name: str) -> None:  # pyright: ignore[reportUnusedFunction]
        """Remove a registered CoreML model."""
        if remove_model(name):
            click.echo(f"Removed model coreml/{name}")
        else:
            click.echo(f"Model '{name}' not found.", err=True)
            raise SystemExit(1)
