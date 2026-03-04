"""CoreML inference engine with format auto-detection and autoregressive generation."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


class ModelFormat(enum.Enum):
    """Detected input format of a CoreML model."""

    APPLE = "apple"
    HUGGINGFACE = "huggingface"


def detect_format(spec: Any) -> ModelFormat:
    """Detect whether a model uses Apple (camelCase) or HuggingFace (snake_case) inputs."""
    input_names: set[str] = {inp.name for inp in spec.description.input}
    if "inputIds" in input_names:
        return ModelFormat.APPLE
    if "input_ids" in input_names:
        return ModelFormat.HUGGINGFACE
    msg = f"Cannot detect model format. Input names: {input_names}"
    raise ValueError(msg)


def is_stateful(spec: Any) -> bool:
    """Check if the model supports stateful KV-cache inference."""
    state_descs = getattr(spec.description, "stateDescriptions", None)
    return state_descs is not None and len(state_descs) > 0


class CoreMLInferenceEngine:
    """Loads a CoreML model and runs autoregressive text generation."""

    COMPUTE_UNIT_MAP: dict[str, str] = {
        "all": "ALL",
        "cpu_only": "CPU_ONLY",
        "cpu_and_gpu": "CPU_AND_GPU",
        "cpu_and_ne": "CPU_AND_NE",
    }

    def __init__(self, model_path: str, compute_units: str = "all") -> None:
        self.model_path = model_path
        self.compute_units = compute_units
        self._model: Any = None
        self._format: ModelFormat | None = None
        self._stateful: bool | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        import coremltools as ct  # pyright: ignore[reportMissingTypeStubs]

        cu_name = self.COMPUTE_UNIT_MAP.get(self.compute_units, "ALL")
        cu = getattr(ct.ComputeUnit, cu_name)
        self._model = ct.models.MLModel(
            self.model_path,
            compute_units=cu,
        )
        spec = self._model.get_spec()
        self._format = detect_format(spec)
        self._stateful = is_stateful(spec)
        return self._model

    @property
    def format(self) -> ModelFormat:
        """Model input format (loads model if needed)."""
        self._load()
        assert self._format is not None
        return self._format

    @property
    def stateful(self) -> bool:
        """Whether the model supports stateful KV-cache."""
        self._load()
        assert self._stateful is not None
        return self._stateful

    def generate(
        self,
        input_ids: list[int],
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Iterator[int]:
        """Generate tokens autoregressively, yielding one token ID at a time."""
        model = self._load()

        if self._stateful:
            yield from self._generate_stateful(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )
        else:
            yield from self._generate_stateless(
                model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )

    def _generate_stateful(
        self,
        model: Any,
        input_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int | None,
    ) -> Iterator[int]:
        state = model.make_state()

        # Prefill: process all prompt tokens
        output: dict[str, np.ndarray] = {}
        for i, token_id in enumerate(input_ids):
            feed = self._make_input(
                [token_id],
                seq_len=len(input_ids),
                position=i,
            )
            output = model.predict(feed, state=state)

        # Decode
        next_token = self._extract_next_token(output, temperature, top_p)
        for _ in range(max_tokens):
            if eos_token_id is not None and next_token == eos_token_id:
                break
            yield next_token

            feed = self._make_input(
                [next_token],
                seq_len=len(input_ids) + 1,
                position=len(input_ids),
            )
            input_ids.append(next_token)
            output = model.predict(feed, state=state)
            next_token = self._extract_next_token(output, temperature, top_p)

    def _generate_stateless(
        self,
        model: Any,
        input_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int | None,
    ) -> Iterator[int]:
        seq = list(input_ids)
        for _ in range(max_tokens):
            feed = self._make_input(seq, seq_len=len(seq), position=0)
            output: dict[str, np.ndarray] = model.predict(feed)
            next_token = self._extract_next_token(output, temperature, top_p)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            yield next_token
            seq.append(next_token)

    def _make_input(
        self,
        token_ids: list[int],
        *,
        seq_len: int,
        position: int,
    ) -> dict[str, np.ndarray]:
        assert self._format is not None
        ids = np.array([token_ids], dtype=np.int32)

        if self._format == ModelFormat.APPLE:
            return {
                "inputIds": ids,
                "causalMask": _build_causal_mask(len(token_ids), seq_len),
            }
        return {
            "input_ids": ids,
            "attention_mask": _build_attention_mask(seq_len),
        }

    def _extract_next_token(
        self,
        output: dict[str, np.ndarray],
        temperature: float,
        top_p: float,
    ) -> int:
        logits_key = "logits" if "logits" in output else next(iter(output))
        logits = np.array(output[logits_key], dtype=np.float32)
        # Take last token's logits
        if logits.ndim == 3:
            logits = logits[0, -1, :]
        elif logits.ndim == 2:
            logits = logits[-1, :]
        return sample_token(logits, temperature=temperature, top_p=top_p)


def _build_causal_mask(query_len: int, kv_len: int) -> np.ndarray:
    """Build a float16 causal mask for Apple-format models."""
    mask = np.full((1, 1, query_len, kv_len), -np.inf, dtype=np.float16)
    for i in range(query_len):
        pos = kv_len - query_len + i
        mask[0, 0, i, : pos + 1] = 0.0
    return mask


def _build_attention_mask(seq_len: int) -> np.ndarray:
    """Build an int32 attention mask for HuggingFace-format models."""
    return np.ones((1, seq_len), dtype=np.int32)


def sample_token(
    logits: np.ndarray,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits using temperature and top-p nucleus sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    # Numerical stability
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    if top_p < 1.0:
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, top_p) + 1
        # Zero out tokens beyond cutoff
        mask = np.zeros_like(probs)
        mask[sorted_indices[:cutoff]] = 1.0
        probs = probs * mask
        probs = probs / np.sum(probs)

    return int(np.random.choice(len(probs), p=probs))
