"""CoreML model implementation for llm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import llm
from pydantic import Field

from llm_coreml.inference import CoreMLInferenceEngine

if TYPE_CHECKING:
    from collections.abc import Iterator


class CoreMLOptions(llm.Options):
    """Options for CoreML model inference."""

    max_tokens: int | None = Field(
        description="Maximum number of tokens to generate",
        default=200,
    )
    temperature: float | None = Field(
        description="Sampling temperature (0 for greedy)",
        default=0.0,
    )
    top_p: float | None = Field(
        description="Top-p nucleus sampling threshold",
        default=1.0,
    )


class CoreMLModel(llm.Model):
    """An llm Model backed by a local CoreML .mlpackage."""

    can_stream = True
    model_id: str
    model_path: str
    tokenizer_id: str
    Options = CoreMLOptions  # type: ignore[assignment]

    def __init__(
        self,
        model_id: str,
        model_path: str,
        tokenizer_id: str,
        compute_units: str = "all",
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.tokenizer_id = tokenizer_id
        self.compute_units = compute_units
        self._engine: CoreMLInferenceEngine | None = None
        self._tokenizer: Any = None

    def _get_engine(self) -> CoreMLInferenceEngine:
        if self._engine is None:
            self._engine = CoreMLInferenceEngine(self.model_path, self.compute_units)
        return self._engine

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)  # pyright: ignore[reportUnknownMemberType]
        return self._tokenizer  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None,
    ) -> Iterator[str]:
        tokenizer = self._get_tokenizer()
        engine = self._get_engine()

        messages = _build_messages(prompt, conversation)
        input_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        opts: CoreMLOptions = prompt.options  # type: ignore[assignment]
        token_count = 0
        for token_id in engine.generate(
            input_ids,
            max_tokens=opts.max_tokens or 200,
            temperature=opts.temperature or 0.0,
            top_p=opts.top_p or 1.0,
            eos_token_id=tokenizer.eos_token_id,
        ):
            token_count += 1
            yield tokenizer.decode([token_id])  # type: ignore[no-any-return]

        response.set_usage(input=len(input_ids), output=token_count)  # pyright: ignore[reportUnknownMemberType]


def _build_messages(
    prompt: llm.Prompt,
    conversation: llm.Conversation | None,
) -> list[dict[str, str]]:
    """Reconstruct a chat message list from the conversation history."""
    messages: list[dict[str, str]] = []

    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})

    if conversation is not None:
        for prev in conversation.responses:
            messages.append({"role": "user", "content": prev.prompt.prompt or ""})
            messages.append(
                {"role": "assistant", "content": prev.text() or ""},  # type: ignore[union-attr]
            )

    messages.append({"role": "user", "content": prompt.prompt or ""})
    return messages
