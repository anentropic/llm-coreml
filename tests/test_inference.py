"""Tests for inference engine format detection and sampling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_coreml.inference import (
    ModelFormat,
    _build_attention_mask,
    _build_causal_mask,
    detect_format,
    is_stateful,
    sample_token,
)


def _make_spec(input_names: list[str], *, stateful: bool = False) -> MagicMock:
    """Create a mock model spec with the given input names."""
    spec = MagicMock()
    spec.description.input = [SimpleNamespace(name=n) for n in input_names]
    spec.description.stateDescriptions = ["kv"] if stateful else []
    return spec


class TestDetectFormat:
    def test_apple_format(self) -> None:
        spec = _make_spec(["inputIds", "causalMask"])
        assert detect_format(spec) == ModelFormat.APPLE

    def test_huggingface_format(self) -> None:
        spec = _make_spec(["input_ids", "attention_mask"])
        assert detect_format(spec) == ModelFormat.HUGGINGFACE

    def test_unknown_format_raises(self) -> None:
        spec = _make_spec(["tokens", "mask"])
        with pytest.raises(ValueError, match="Cannot detect model format"):
            detect_format(spec)


class TestIsStateful:
    def test_stateful(self) -> None:
        spec = _make_spec(["inputIds"], stateful=True)
        assert is_stateful(spec) is True

    def test_stateless(self) -> None:
        spec = _make_spec(["inputIds"], stateful=False)
        assert is_stateful(spec) is False


class TestBuildMasks:
    def test_causal_mask_shape(self) -> None:
        mask = _build_causal_mask(1, 5)
        assert mask.shape == (1, 1, 1, 5)
        assert mask.dtype == np.float16

    def test_causal_mask_values(self) -> None:
        mask = _build_causal_mask(3, 3)
        # First token can only see itself
        assert mask[0, 0, 0, 0] == 0.0
        assert np.isinf(mask[0, 0, 0, 1])
        # Last token can see all
        assert mask[0, 0, 2, 0] == 0.0
        assert mask[0, 0, 2, 2] == 0.0

    def test_attention_mask_shape(self) -> None:
        mask = _build_attention_mask(5)
        assert mask.shape == (1, 5)
        assert mask.dtype == np.int32
        assert np.all(mask == 1)


class TestSampleToken:
    def test_greedy(self) -> None:
        logits = np.array([1.0, 5.0, 2.0])
        assert sample_token(logits, temperature=0.0) == 1

    def test_temperature_zero_picks_argmax(self) -> None:
        logits = np.array([0.0, 0.0, 10.0])
        assert sample_token(logits, temperature=0.0) == 2

    def test_with_temperature(self) -> None:
        logits = np.array([0.0, 0.0, 100.0])
        # With very peaked distribution, should almost always pick index 2
        results = [sample_token(logits, temperature=0.1) for _ in range(100)]
        assert all(r == 2 for r in results)

    def test_top_p_filtering(self) -> None:
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        # top_p=0.5 with very peaked distribution should always pick 0
        results = [sample_token(logits, temperature=1.0, top_p=0.5) for _ in range(50)]
        assert all(r == 0 for r in results)
