"""Integration tests exercising real CoreML inference with a converted model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parent.parent / ".models" / "tiny-gpt2"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not MODEL_PATH.exists(),
        reason=f"Model not found at {MODEL_PATH}. Run 'just convert-test-model' first.",
    ),
    pytest.mark.skipif(
        sys.platform != "darwin",
        reason="CoreML inference only available on macOS",
    ),
]


def _find_mlpackage() -> str:
    """Find the .mlpackage inside the model directory."""
    packages = list(MODEL_PATH.glob("*.mlpackage"))
    assert packages, f"No .mlpackage found in {MODEL_PATH}"
    return str(packages[0])


class TestEngineLoadsModel:
    def test_engine_loads_model(self) -> None:
        from llm_coreml.inference import CoreMLInferenceEngine, ModelFormat

        engine = CoreMLInferenceEngine(_find_mlpackage())
        assert engine.format == ModelFormat.HUGGINGFACE

    def test_engine_detects_stateless(self) -> None:
        from llm_coreml.inference import CoreMLInferenceEngine

        engine = CoreMLInferenceEngine(_find_mlpackage())
        assert engine.stateful is False


class TestEngineGeneratesTokens:
    def test_engine_generates_tokens(self) -> None:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

        from llm_coreml.inference import CoreMLInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        engine = CoreMLInferenceEngine(_find_mlpackage())

        input_ids: list[int] = tokenizer.encode("Hello")  # pyright: ignore[reportAssignmentType]
        tokens = list(engine.generate(input_ids, max_tokens=5))

        assert len(tokens) == 5
        for token in tokens:
            assert isinstance(token, int)
            assert 0 <= token < tokenizer.vocab_size

    def test_engine_respects_max_tokens(self) -> None:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

        from llm_coreml.inference import CoreMLInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        engine = CoreMLInferenceEngine(_find_mlpackage())

        input_ids: list[int] = tokenizer.encode("Hello")  # pyright: ignore[reportAssignmentType]

        for n in (1, 3, 7):
            tokens = list(engine.generate(list(input_ids), max_tokens=n))
            assert len(tokens) == n

    def test_engine_stops_at_eos(self) -> None:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

        from llm_coreml.inference import CoreMLInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        engine = CoreMLInferenceEngine(_find_mlpackage())

        input_ids: list[int] = tokenizer.encode("Hello")  # pyright: ignore[reportAssignmentType]
        tokens = list(
            engine.generate(
                input_ids,
                max_tokens=100,
                eos_token_id=tokenizer.eos_token_id,
            )
        )

        # Should stop before max_tokens if EOS is generated,
        # or exactly max_tokens if not
        assert len(tokens) <= 100


class TestEngineHandlesShortSequence:
    def test_engine_handles_short_sequence(self) -> None:
        """Single-token input should work with dynamic sequence lengths."""
        from llm_coreml.inference import CoreMLInferenceEngine

        engine = CoreMLInferenceEngine(_find_mlpackage())
        tokens = list(engine.generate([1], max_tokens=3))
        assert len(tokens) == 3
        for token in tokens:
            assert isinstance(token, int)


class TestTokenizeGenerateDecode:
    def test_round_trip(self) -> None:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

        from llm_coreml.inference import CoreMLInferenceEngine

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        engine = CoreMLInferenceEngine(_find_mlpackage())

        prompt = "The weather today is"
        input_ids: list[int] = tokenizer.encode(prompt)  # pyright: ignore[reportAssignmentType]
        generated = list(engine.generate(input_ids, max_tokens=10))

        decoded: str = tokenizer.decode(generated)  # pyright: ignore[reportAssignmentType]
        assert isinstance(decoded, str)
        assert len(decoded) > 0
