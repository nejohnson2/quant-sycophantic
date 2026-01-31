"""Topic classification for domain-based analysis.

Uses keyword-based heuristics for fast classification.
Can be enhanced with embeddings or LLM-based classification.
"""

import re

import pandas as pd

from ..config import TOPIC_CATEGORIES

# Keyword patterns for topic classification
TOPIC_KEYWORDS = {
    "coding": [
        r"\bcode\b",
        r"\bpython\b",
        r"\bjavascript\b",
        r"\bjava\b",
        r"\bc\+\+\b",
        r"\bfunction\b",
        r"\bclass\b",
        r"\bvariable\b",
        r"\bdebug\b",
        r"\berror\b",
        r"\bapi\b",
        r"\bsql\b",
        r"\bhtml\b",
        r"\bcss\b",
        r"\bgit\b",
        r"\bprogram\b",
        r"\balgorithm\b",
        r"\bdata structure\b",
    ],
    "creative_writing": [
        r"\bwrite (a |me )?(story|poem|essay|script)\b",
        r"\bcreative\b",
        r"\bfiction\b",
        r"\bcharacter\b",
        r"\bplot\b",
        r"\bstory\b",
        r"\bpoem\b",
        r"\bnovel\b",
        r"\bnarrative\b",
    ],
    "math": [
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bequation\b",
        r"\bmath\b",
        r"\balgebra\b",
        r"\bcalculus\b",
        r"\bintegral\b",
        r"\bderivative\b",
        r"\bprobability\b",
        r"\bstatistics\b",
        r"\b\d+\s*[\+\-\*\/]\s*\d+\b",  # arithmetic expressions
    ],
    "factual_qa": [
        r"\bwhat is\b",
        r"\bwho is\b",
        r"\bwhen did\b",
        r"\bwhere is\b",
        r"\bhow does\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bdescribe\b",
        r"\bhistory of\b",
        r"\bfacts? about\b",
    ],
    "advice": [
        r"\bshould i\b",
        r"\bwhat should\b",
        r"\bhow should\b",
        r"\badvice\b",
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\bhelp me (decide|choose)\b",
        r"\bbest way to\b",
    ],
    "opinion": [
        r"\bwhat do you think\b",
        r"\bdo you (think|believe|agree)\b",
        r"\byour opinion\b",
        r"\bbetter[,:]?\s*(a|b|option)\b",
        r"\bwhich is better\b",
        r"\bpros and cons\b",
        r"\bcompare\b",
    ],
}

# Compile patterns
_COMPILED_TOPIC_PATTERNS: dict[str, list[re.Pattern]] = {}


def _get_topic_patterns() -> dict[str, list[re.Pattern]]:
    """Lazily compile topic patterns."""
    global _COMPILED_TOPIC_PATTERNS
    if not _COMPILED_TOPIC_PATTERNS:
        _COMPILED_TOPIC_PATTERNS = {
            topic: [re.compile(p, re.IGNORECASE) for p in patterns]
            for topic, patterns in TOPIC_KEYWORDS.items()
        }
    return _COMPILED_TOPIC_PATTERNS


def classify_topic(text: str) -> str:
    """
    Classify a user prompt into a topic category.

    Uses keyword matching with priority ordering.

    Args:
        text: User prompt text.

    Returns:
        Topic category string.
    """
    if not text:
        return "other"

    patterns = _get_topic_patterns()

    # Score each topic by number of keyword matches
    scores = {}
    for topic, topic_patterns in patterns.items():
        score = sum(len(p.findall(text)) for p in topic_patterns)
        if score > 0:
            scores[topic] = score

    if not scores:
        return "other"

    # Return topic with highest score
    return max(scores, key=scores.get)


def compute_topic_features(
    df: pd.DataFrame,
    prompt_col: str = "user_prompt",
) -> pd.DataFrame:
    """
    Add topic classification to DataFrame.

    Args:
        df: DataFrame with user prompts.
        prompt_col: Column name containing user prompts.

    Returns:
        DataFrame with 'topic' column added.
    """
    result = df.copy()

    if prompt_col not in result.columns:
        raise ValueError(f"Column '{prompt_col}' not found in DataFrame")

    result["topic"] = result[prompt_col].apply(classify_topic)

    return result


def get_topic_distribution(df: pd.DataFrame, topic_col: str = "topic") -> pd.Series:
    """Get distribution of topics in the dataset."""
    return df[topic_col].value_counts()
