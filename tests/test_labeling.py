"""Tests for the labeling module."""

import pytest

from quant_syco.labeling.prompts import build_evaluation_prompt, COMBINED_PROMPT
from quant_syco.labeling.ollama_judge import parse_json_response, validate_scores


class TestPrompts:
    def test_build_evaluation_prompt(self):
        prompt = build_evaluation_prompt(
            user_message="What is 2+2?",
            assistant_response="Great question! 2+2 equals 4.",
        )

        assert "What is 2+2?" in prompt
        assert "Great question! 2+2 equals 4." in prompt
        assert "sycophancy" in prompt.lower()
        assert "politeness" in prompt.lower()

    def test_prompt_has_rating_scale(self):
        assert "0 = " in COMBINED_PROMPT
        assert "1 = " in COMBINED_PROMPT
        assert "2 = " in COMBINED_PROMPT
        assert "3 = " in COMBINED_PROMPT


class TestJsonParsing:
    def test_parse_valid_json(self):
        response = '{"sycophancy": 2, "politeness": 3, "reasoning": "test"}'
        result = parse_json_response(response)

        assert result["sycophancy"] == 2
        assert result["politeness"] == 3
        assert result["reasoning"] == "test"

    def test_parse_json_with_whitespace(self):
        response = '  \n  {"sycophancy": 1, "politeness": 2}  \n  '
        result = parse_json_response(response)

        assert result["sycophancy"] == 1
        assert result["politeness"] == 2

    def test_parse_json_in_markdown(self):
        response = '''Here's my evaluation:
        ```json
        {"sycophancy": 0, "politeness": 2, "reasoning": "honest response"}
        ```
        '''
        result = parse_json_response(response)

        assert result["sycophancy"] == 0

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_json_response("not json at all")


class TestValidateScores:
    def test_validate_valid_scores(self):
        data = {"sycophancy": 2, "politeness": 3, "reasoning": "test"}
        syco, polite, reason = validate_scores(data)

        assert syco == 2
        assert polite == 3
        assert reason == "test"

    def test_validate_out_of_range(self):
        data = {"sycophancy": 5, "politeness": -1}
        syco, polite, _ = validate_scores(data)

        assert syco is None  # 5 is out of range
        assert polite is None  # -1 is out of range

    def test_validate_string_numbers(self):
        data = {"sycophancy": "2", "politeness": "1"}
        syco, polite, _ = validate_scores(data)

        assert syco == 2
        assert polite == 1

    def test_validate_missing_keys(self):
        data = {"other": "value"}
        syco, polite, reason = validate_scores(data)

        assert syco is None
        assert polite is None
        assert reason == ""
