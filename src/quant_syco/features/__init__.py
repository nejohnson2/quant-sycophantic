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
from .topics import classify_topic, compute_topic_features

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
    # Topics
    "classify_topic",
    "compute_topic_features",
]
