"""Tests for CoreMLModel."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_coreml.model import CoreMLModel, _build_messages


class TestCoreMLModel:
    def test_model_id_prefix(self) -> None:
        model = CoreMLModel("coreml/test", "/path", "org/tok")
        assert model.model_id == "coreml/test"

    def test_can_stream(self) -> None:
        model = CoreMLModel("coreml/test", "/path", "org/tok")
        assert model.can_stream is True


class TestBuildMessages:
    def test_simple_prompt(self) -> None:
        prompt = MagicMock()
        prompt.prompt = "Hello"
        prompt.system = None
        messages = _build_messages(prompt, None)
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_system_prompt(self) -> None:
        prompt = MagicMock()
        prompt.prompt = "Hello"
        prompt.system = "You are helpful."
        messages = _build_messages(prompt, None)
        assert messages == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    def test_with_conversation_history(self) -> None:
        prev_response = MagicMock()
        prev_response.prompt.prompt = "Hi"
        prev_response.text.return_value = "Hello!"

        conversation = MagicMock()
        conversation.responses = [prev_response]

        prompt = MagicMock()
        prompt.prompt = "How are you?"
        prompt.system = None

        messages = _build_messages(prompt, conversation)
        assert messages == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
