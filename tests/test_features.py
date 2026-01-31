"""Tests for feature extraction modules."""

import pandas as pd
import pytest

from quant_syco.features.lexical import (
    count_phrase_matches,
    compute_lexical_features,
    SYCOPHANCY_PHRASES,
)
from quant_syco.features.topics import classify_topic, compute_topic_features


class TestLexicalFeatures:
    def test_count_flattery_phrases(self):
        text = "Great question! You're absolutely right about that."
        counts = count_phrase_matches(text)

        assert counts["flattery"] >= 2  # "great question" and "absolutely right"

    def test_count_validation_seeking(self):
        text = "I hope this helps! Does that make sense?"
        counts = count_phrase_matches(text)

        assert counts["validation_seeking"] >= 2

    def test_count_empty_text(self):
        counts = count_phrase_matches("")

        for category in SYCOPHANCY_PHRASES:
            assert counts[category] == 0

    def test_compute_lexical_features_dataframe(self):
        df = pd.DataFrame({
            "assistant_a": [
                "Great question! Here's the answer.",
                "The answer is 42.",
                "You're absolutely right! Brilliant!",
            ]
        })

        result = compute_lexical_features(df, "assistant_a")

        assert "lex_flattery_count" in result.columns
        assert "lex_sycophancy_total" in result.columns
        assert "lex_sycophancy_any" in result.columns

        # First and third rows have sycophantic phrases
        assert result.iloc[0]["lex_sycophancy_any"] == 1
        assert result.iloc[1]["lex_sycophancy_any"] == 0
        assert result.iloc[2]["lex_sycophancy_any"] == 1


class TestTopicClassification:
    def test_classify_coding(self):
        prompts = [
            "Write a Python function to sort a list",
            "How do I debug this JavaScript error?",
            "Explain how Git branching works",
        ]

        for prompt in prompts:
            assert classify_topic(prompt) == "coding"

    def test_classify_math(self):
        prompts = [
            "Calculate the derivative of x^2",
            "Solve this equation: 2x + 5 = 15",
            "What is the probability of rolling a 6?",
        ]

        for prompt in prompts:
            assert classify_topic(prompt) == "math"

    def test_classify_advice(self):
        prompts = [
            "Should I accept this job offer?",
            "What should I do about my neighbor?",
            "Can you recommend a good laptop?",
        ]

        for prompt in prompts:
            assert classify_topic(prompt) == "advice"

    def test_classify_other(self):
        # Gibberish or unclassifiable
        assert classify_topic("asdfghjkl") == "other"
        assert classify_topic("") == "other"

    def test_compute_topic_features(self):
        df = pd.DataFrame({
            "user_prompt": [
                "Write Python code for sorting",
                "What is 5 + 5?",
                "Tell me a story",
            ]
        })

        result = compute_topic_features(df, "user_prompt")

        assert "topic" in result.columns
        assert result.iloc[0]["topic"] == "coding"
        assert result.iloc[1]["topic"] == "math"
        assert result.iloc[2]["topic"] == "creative_writing"
