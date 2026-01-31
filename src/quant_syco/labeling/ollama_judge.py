"""LLM-as-judge implementation using Ollama."""

import json
import re
from dataclasses import dataclass
from typing import Any

import ollama
from rich.console import Console

from ..config import DEFAULT_OLLAMA_MODEL
from .prompts import build_evaluation_prompt

console = Console()


@dataclass
class LabelResult:
    """Result of labeling a single response."""

    sycophancy: int | None
    politeness: int | None
    reasoning: str
    raw_response: str
    success: bool
    error: str | None = None


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling common issues.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If parsing fails.
    """
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in response
    json_match = re.search(r"\{[^{}]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")


def validate_scores(data: dict[str, Any]) -> tuple[int | None, int | None, str]:
    """
    Validate and extract scores from parsed response.

    Returns:
        (sycophancy_score, politeness_score, reasoning)
    """
    sycophancy = data.get("sycophancy")
    politeness = data.get("politeness")
    reasoning = data.get("reasoning", "") or data.get("sycophancy_reasoning", "") or ""

    # Validate sycophancy
    if sycophancy is not None:
        try:
            sycophancy = int(sycophancy)
            if not 0 <= sycophancy <= 3:
                sycophancy = None
        except (ValueError, TypeError):
            sycophancy = None

    # Validate politeness
    if politeness is not None:
        try:
            politeness = int(politeness)
            if not 0 <= politeness <= 3:
                politeness = None
        except (ValueError, TypeError):
            politeness = None

    return sycophancy, politeness, reasoning


class OllamaJudge:
    """LLM-as-judge for sycophancy evaluation using Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = 60.0,
    ):
        """
        Initialize the Ollama judge.

        Args:
            model: Ollama model name (e.g., "llama3.2:8b").
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.timeout = timeout
        self._client = ollama.Client()

    def _generate(self, prompt: str) -> str:
        """Generate a response from Ollama."""
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.0,  # Deterministic for consistency
                "num_predict": 256,  # Limit response length
            },
        )
        return response["response"]

    def label(
        self,
        user_message: str,
        assistant_response: str,
        max_retries: int = 2,
    ) -> LabelResult:
        """
        Label a single user-assistant pair for sycophancy and politeness.

        Args:
            user_message: The user's message.
            assistant_response: The assistant's response.
            max_retries: Number of retries on parse failure.

        Returns:
            LabelResult with scores and metadata.
        """
        prompt = build_evaluation_prompt(user_message, assistant_response)

        for attempt in range(max_retries + 1):
            try:
                raw_response = self._generate(prompt)

                data = parse_json_response(raw_response)
                sycophancy, politeness, reasoning = validate_scores(data)

                if sycophancy is None:
                    raise ValueError("Failed to extract valid sycophancy score")

                return LabelResult(
                    sycophancy=sycophancy,
                    politeness=politeness,
                    reasoning=reasoning,
                    raw_response=raw_response,
                    success=True,
                )

            except Exception as e:
                if attempt == max_retries:
                    return LabelResult(
                        sycophancy=None,
                        politeness=None,
                        reasoning="",
                        raw_response=raw_response if "raw_response" in dir() else "",
                        success=False,
                        error=str(e),
                    )
                continue

        # Should not reach here
        return LabelResult(
            sycophancy=None,
            politeness=None,
            reasoning="",
            raw_response="",
            success=False,
            error="Unknown error",
        )


def label_single_response(
    user_message: str,
    assistant_response: str,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> LabelResult:
    """
    Convenience function to label a single response.

    Args:
        user_message: The user's message.
        assistant_response: The assistant's response.
        model: Ollama model to use.

    Returns:
        LabelResult with scores.
    """
    judge = OllamaJudge(model=model)
    return judge.label(user_message, assistant_response)
