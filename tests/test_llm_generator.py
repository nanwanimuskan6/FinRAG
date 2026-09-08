"""Tests for the Gemini generation adapter without calling the external API."""

from types import SimpleNamespace

import pytest

from src.generation.llm_generator import DEFAULT_MODEL_NAME, LLMGenerator


class FakeModels:
    def __init__(self, text: str = "Supported answer.") -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def test_gemini_generator_sends_deterministic_grounded_configuration() -> None:
    models = FakeModels()
    generator = LLMGenerator(client=SimpleNamespace(models=models))

    assert generator.model_name == DEFAULT_MODEL_NAME
    assert generator.generate("Evidence: revenue was 100.") == "Supported answer."
    assert models.calls[0]["config"] == {
        "system_instruction": (
            "You are a financial document question-answering assistant. Answer only "
            "from the supplied evidence. Never use outside knowledge or invent a "
            "number. If the answer is not explicitly supported by the evidence, say "
            "that it cannot be found. Give a concise answer."
        ),
        "temperature": 0.0,
        "max_output_tokens": 512,
    }


def test_gemini_generator_validates_empty_prompts_and_answers() -> None:
    generator = LLMGenerator(client=SimpleNamespace(models=FakeModels("   ")))

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generator.generate("  ")
    with pytest.raises(RuntimeError, match="empty answer"):
        generator.generate("Evidence")


def test_gemini_generator_requires_a_key_without_an_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        LLMGenerator()
