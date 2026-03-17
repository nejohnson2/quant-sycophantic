"""Feature extraction utilities."""

from .lexical import compute_lexical_features, SYCOPHANCY_PHRASES
from .metrics import (
    add_ts_utc,
    approx_tokens,
    build_all_metrics,
    build_battle_level_table,
    compute_return_metrics,
    compute_session_metrics,
    compute_user_summary,
    extract_first_assistant_response,
    extract_turn_table,
)
from .stylistic import (
    compute_all_stylistic_features,
    compute_confidence_features,
    compute_formatting_features,
    compute_readability_features,
)
from .topics import classify_topic, compute_topic_features
from .topics_embedding import classify_topics_ollama, compare_classifiers

__all__ = [
    # Metrics
    "add_ts_utc",
    "approx_tokens",
    "build_all_metrics",
    "build_battle_level_table",
    "compute_return_metrics",
    "compute_session_metrics",
    "compute_user_summary",
    "extract_first_assistant_response",
    "extract_turn_table",
    # Lexical
    "compute_lexical_features",
    "SYCOPHANCY_PHRASES",
    # Stylistic
    "compute_all_stylistic_features",
    "compute_formatting_features",
    "compute_confidence_features",
    "compute_readability_features",
    # Topics
    "classify_topic",
    "compute_topic_features",
    "classify_topics_ollama",
    "compare_classifiers",
]
