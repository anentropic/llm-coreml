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


class TestExecuteTokenization:
    """Test chat template vs plain tokenization fallback."""

    def test_uses_plain_encode_when_no_chat_template(self) -> None:
        """Completion models (e.g. GPT-2) without chat_template should use encode()."""
        model = CoreMLModel("coreml/test", "/path", "org/tok")

        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.eos_token_id = 0
        tokenizer.decode.return_value = "tok"
        model._tokenizer = tokenizer

        engine = MagicMock()
        engine.generate.return_value = [4, 5]
        model._engine = engine

        prompt = MagicMock()
        prompt.prompt = "Hello"
        prompt.system = None
        prompt.options = MagicMock()
        prompt.options.max_tokens = 10
        prompt.options.temperature = 0.0
        prompt.options.top_p = 1.0

        response = MagicMock()
        list(model.execute(prompt, stream=True, response=response, conversation=None))

        tokenizer.encode.assert_called_once_with("Hello")
        tokenizer.apply_chat_template.assert_not_called()

    def test_uses_chat_template_when_available(self) -> None:
        """Chat models with chat_template should use apply_chat_template()."""
        model = CoreMLModel("coreml/test", "/path", "org/tok")

        tokenizer = MagicMock()
        tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        tokenizer.eos_token_id = 0
        tokenizer.decode.return_value = "tok"
        model._tokenizer = tokenizer

        engine = MagicMock()
        engine.generate.return_value = [4, 5]
        model._engine = engine

        prompt = MagicMock()
        prompt.prompt = "Hello"
        prompt.system = None
        prompt.options = MagicMock()
        prompt.options.max_tokens = 10
        prompt.options.temperature = 0.0
        prompt.options.top_p = 1.0

        response = MagicMock()
        list(model.execute(prompt, stream=True, response=response, conversation=None))

        tokenizer.apply_chat_template.assert_called_once()
        tokenizer.encode.assert_not_called()


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
