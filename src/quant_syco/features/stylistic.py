"""Additional stylistic features for response quality control.

These features capture formatting, confidence, and readability signals
that could confound the sycophancy-preference relationship. Used in
extended robustness specifications (Specs 7-8) to show that
domain-heterogeneous effects persist beyond length and politeness.
"""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting features
# ---------------------------------------------------------------------------

_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s", re.MULTILINE)
_BULLET_POINT = re.compile(r"^[\s]*[-*+]\s", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^[\s]*\d+[.)]\s", re.MULTILINE)
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`]+`")
_BOLD = re.compile(r"\*\*[^*]+\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)[^*]+\*(?!\*)")


def compute_formatting_features(
    df: pd.DataFrame,
    text_col: str = "assistant_a",
    prefix: str = "",
) -> pd.DataFrame:
    """Extract markdown formatting features from responses.

    Args:
        df: DataFrame with response text.
        text_col: Column containing the response.
        prefix: Optional prefix for output column names.

    Returns:
        DataFrame with formatting feature columns added.
    """
    result = df.copy()
    p = f"{prefix}" if prefix else f"{text_col}_"

    def _count(pattern, text):
        return len(pattern.findall(str(text))) if text else 0

    result[f"{p}n_headers"] = result[text_col].apply(lambda x: _count(_MARKDOWN_HEADER, x))
    result[f"{p}n_bullets"] = result[text_col].apply(lambda x: _count(_BULLET_POINT, x))
    result[f"{p}n_numbered"] = result[text_col].apply(lambda x: _count(_NUMBERED_LIST, x))
    result[f"{p}n_code_blocks"] = result[text_col].apply(lambda x: _count(_CODE_BLOCK, x))
    result[f"{p}n_inline_code"] = result[text_col].apply(lambda x: _count(_INLINE_CODE, x))
    result[f"{p}n_bold"] = result[text_col].apply(lambda x: _count(_BOLD, x))

    # Aggregate formatting score
    fmt_cols = [
        f"{p}n_headers", f"{p}n_bullets", f"{p}n_numbered",
        f"{p}n_code_blocks", f"{p}n_bold",
    ]
    result[f"{p}formatting_score"] = result[fmt_cols].sum(axis=1)

    # Binary: uses any formatting
    result[f"{p}has_formatting"] = (result[f"{p}formatting_score"] > 0).astype(int)

    return result


# ---------------------------------------------------------------------------
# Confidence / hedging features
# ---------------------------------------------------------------------------

_HEDGING_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bprobably\b",
        r"\bperhaps\b",
        r"\bit'?s? possible\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bi'?m? not (entirely |completely )?sure\b",
        r"\bit depends\b",
        r"\bgenerally speaking\b",
        r"\bin my (opinion|view)\b",
    ]
]

_CONFIDENCE_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcertainly\b",
        r"\bdefinitely\b",
        r"\babsolutely\b",
        r"\bwithout (a )?doubt\b",
        r"\bthe answer is\b",
        r"\bthe (?:correct|right) answer\b",
        r"\bclearly\b",
        r"\bobviously\b",
        r"\bundoubtedly\b",
        r"\bin fact\b",
    ]
]


def compute_confidence_features(
    df: pd.DataFrame,
    text_col: str = "assistant_a",
    prefix: str = "",
) -> pd.DataFrame:
    """Extract confidence and hedging language features.

    Args:
        df: DataFrame with response text.
        text_col: Column containing the response.
        prefix: Optional prefix for output column names.

    Returns:
        DataFrame with confidence/hedging columns added.
    """
    result = df.copy()
    p = f"{prefix}" if prefix else f"{text_col}_"

    def _count_patterns(patterns, text):
        if not text:
            return 0
        text = str(text)
        return sum(len(pat.findall(text)) for pat in patterns)

    result[f"{p}n_hedging"] = result[text_col].apply(
        lambda x: _count_patterns(_HEDGING_PHRASES, x)
    )
    result[f"{p}n_confidence"] = result[text_col].apply(
        lambda x: _count_patterns(_CONFIDENCE_PHRASES, x)
    )

    # Net confidence: confidence - hedging (positive = more assertive)
    result[f"{p}net_confidence"] = (
        result[f"{p}n_confidence"] - result[f"{p}n_hedging"]
    )

    return result


# ---------------------------------------------------------------------------
# Readability features
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+|[.!?]+$")


def compute_readability_features(
    df: pd.DataFrame,
    text_col: str = "assistant_a",
    prefix: str = "",
) -> pd.DataFrame:
    """Extract readability features from responses.

    Computes Flesch-Kincaid grade level (approximation), average sentence
    length, and type-token ratio as proxies for complexity and quality.

    Args:
        df: DataFrame with response text.
        text_col: Column containing the response.
        prefix: Optional prefix for output column names.

    Returns:
        DataFrame with readability columns added.
    """
    result = df.copy()
    p = f"{prefix}" if prefix else f"{text_col}_"

    def _syllable_count(word: str) -> int:
        """Rough syllable count heuristic."""
        word = word.lower().strip(".,!?;:")
        if len(word) <= 2:
            return 1
        # Remove trailing silent e
        if word.endswith("e") and len(word) > 2:
            word = word[:-1]
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)

    def _readability_stats(text):
        if not text:
            return {"avg_sentence_len": 0, "fk_grade": 0, "ttr": 0}
        text = str(text)
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        words = text.split()
        n_words = len(words)
        n_sentences = max(len(sentences), 1)

        if n_words == 0:
            return {"avg_sentence_len": 0, "fk_grade": 0, "ttr": 0}

        avg_sentence_len = n_words / n_sentences
        n_syllables = sum(_syllable_count(w) for w in words)
        avg_syllables = n_syllables / n_words

        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59

        # Type-token ratio (vocabulary diversity)
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / n_words

        return {
            "avg_sentence_len": avg_sentence_len,
            "fk_grade": fk_grade,
            "ttr": ttr,
        }

    stats = result[text_col].apply(_readability_stats)
    stats_df = pd.DataFrame(stats.tolist(), index=result.index)
    result[f"{p}avg_sentence_len"] = stats_df["avg_sentence_len"]
    result[f"{p}fk_grade"] = stats_df["fk_grade"]
    result[f"{p}ttr"] = stats_df["ttr"]

    return result


# ---------------------------------------------------------------------------
# All stylistic features in one pass
# ---------------------------------------------------------------------------


def compute_all_stylistic_features(
    df: pd.DataFrame,
    text_cols: Sequence[str] = ("assistant_a", "assistant_b"),
) -> pd.DataFrame:
    """Compute all stylistic features for given text columns.

    Runs formatting, confidence, and readability extraction for each
    specified text column. Computes differentials (A - B) if both sides
    are provided.

    Args:
        df: DataFrame with response text columns.
        text_cols: Columns to extract features from.

    Returns:
        DataFrame with all stylistic features added.
    """
    result = df.copy()

    for col in text_cols:
        if col not in result.columns:
            continue
        result = compute_formatting_features(result, text_col=col)
        result = compute_confidence_features(result, text_col=col)
        result = compute_readability_features(result, text_col=col)

    # Compute differentials if both sides present
    if "assistant_a" in text_cols and "assistant_b" in text_cols:
        a, b = "assistant_a_", "assistant_b_"
        for feat in ["formatting_score", "net_confidence", "fk_grade", "ttr", "avg_sentence_len"]:
            if f"{a}{feat}" in result.columns and f"{b}{feat}" in result.columns:
                result[f"{feat}_diff"] = result[f"{a}{feat}"] - result[f"{b}{feat}"]

    return result
