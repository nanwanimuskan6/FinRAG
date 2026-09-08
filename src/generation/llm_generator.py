"""Grounded answer generation through the Google Gemini API."""

from __future__ import annotations

import os
from typing import Any, Protocol


# Google's current highest-capability Gemini text model. Keep this configurable
# because preview model availability and pricing can change.
DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"

SYSTEM_INSTRUCTION = (
    "You are a financial document question-answering assistant. Answer only "
    "from the supplied evidence. Never use outside knowledge or invent a "
    "number. If the answer is not explicitly supported by the evidence, say "
    "that it cannot be found. Give a concise answer."
)


class GeminiClient(Protocol):
    """Subset of the Google GenAI client used by this adapter."""

    models: Any


class LLMGenerator:
    """Generate deterministic, evidence-grounded answers with Gemini.

    Set ``GEMINI_API_KEY`` in the environment before running the application.
    The Google SDK import is deliberately delayed so retrieval-only workflows
    and unit tests do not require network access or an API key.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        api_key: str | None = None,
        client: GeminiClient | None = None,
    ) -> None:
        self.model_name = model_name
        key = api_key or os.environ.get("GEMINI_API_KEY")

        if client is None:
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is required. Add it to your environment before "
                    "starting FinRAG."
                )
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError(
                    "Google GenAI SDK is not installed. Run: pip install -r requirements.txt"
                ) from error
            client = genai.Client(api_key=key)

        self._client = client

    def generate(self, prompt: str) -> str:
        """Return a concise answer generated strictly from supplied evidence."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0.0,
                "max_output_tokens": 512,
            },
        )
        answer = getattr(response, "text", None)
        if not answer or not str(answer).strip():
            raise RuntimeError("Gemini returned an empty answer")
        return str(answer).strip()
