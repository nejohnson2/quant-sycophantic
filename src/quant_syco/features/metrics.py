import math
import json
import pandas as pd
import numpy as np

# ---------------------------
# 1) Timestamp helpers
# ---------------------------

def to_ts_utc(tstamp_value) -> pd.Timestamp:
    """
    Convert LMSYS `tstamp` (unix seconds, often float) to pandas UTC timestamp.
    """
    if tstamp_value is None or (isinstance(tstamp_value, float) and math.isnan(tstamp_value)):
        return pd.NaT
    # Some dumps store as string
    if isinstance(tstamp_value, str):
        tstamp_value = tstamp_value.strip()
        if tstamp_value == "":
            return pd.NaT
        try:
            tstamp_value = float(tstamp_value)
        except ValueError:
            # Try ISO parse
            return pd.to_datetime(tstamp_value, utc=True, errors="coerce")
    # Treat as unix seconds
    return pd.to_datetime(float(tstamp_value), unit="s", utc=True)


def add_ts_utc(df: pd.DataFrame, tstamp_col: str = "tstamp") -> pd.DataFrame:
    """
    Adds/overwrites `ts_utc` column from `tstamp`.
    """
    out = df.copy()
    out["ts_utc"] = out[tstamp_col].apply(to_ts_utc)
    return out


# ---------------------------
# 2) Conversation extraction
# ---------------------------

def _normalize_messages(obj):
    """
    Normalize conversation_a / conversation_b into list of dicts with keys: role, content.
    Handles:
      - list of dicts
      - JSON-encoded strings
    """
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []

    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                obj = json.loads(s)
            except Exception:
                return [{"role": "unknown", "content": s}]
        else:
            return [{"role": "unknown", "content": s}]

    if isinstance(obj, dict):
        # Sometimes wrapped
        if "messages" in obj:
            return _normalize_messages(obj["messages"])
        if "conversation" in obj:
            return _normalize_messages(obj["conversation"])
        # single message dict
        role = obj.get("role", "unknown")
        content = obj.get("content", obj.get("text", ""))
        return [{"role": str(role), "content": str(content)}]

    if isinstance(obj, list):
        out = []
        for m in obj:
            if isinstance(m, dict):
                role = m.get("role") or m.get("from") or m.get("speaker") or "unknown"
                content = m.get("content") or m.get("text") or m.get("message") or ""
                out.append({"role": str(role), "content": str(content)})
            else:
                out.append({"role": "unknown", "content": str(m)})
        return out

    return [{"role": "unknown", "content": str(obj)}]


def extract_turn_table(
    df: pd.DataFrame,
    conversation_col: str,
    side_label: str,  # "a" or "b"
    id_col: str = "question_id",
    user_col: str = "judge",
    model_col: str = None,  # e.g., "model_a" or "model_b"
) -> pd.DataFrame:
    """
    Returns one row per message turn for a given conversation side (A or B).
    Output columns:
      question_id, judge, model, side, ts_utc,
      turn_index, role, content
    """
    rows = []
    for _, r in df.iterrows():
        qid = r.get(id_col)
        user = r.get(user_col)
        ts = r.get("ts_utc", pd.NaT)
        model = r.get(model_col) if model_col else None

        msgs = _normalize_messages(r.get(conversation_col))
        for j, m in enumerate(msgs):
            rows.append(
                {
                    "question_id": qid,
                    "judge": user,
                    "model": model,
                    "side": side_label,
                    "ts_utc": ts,
                    "turn_index": j,
                    "role": str(m.get("role", "unknown")).lower(),
                    "content": m.get("content", ""),
                }
            )
    return pd.DataFrame(rows)


def approx_tokens(text: str) -> int:
    """
    Cheap proxy token count. Replace with a tokenizer later if you want.
    """
    if text is None:
        return 0
    return len(str(text).split())


def compute_session_metrics(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conversation-level metrics from a turn table.
    Groups by (question_id, judge, side, model, ts_utc).
    """
    t = turns.copy()
    t["approx_tokens"] = t["content"].map(approx_tokens)

    def _count_role(s, role_name):
        return int((s == role_name).sum())

    gcols = ["question_id", "judge", "side", "model", "ts_utc"]
    metrics = (
        t.groupby(gcols, dropna=False)
        .agg(
            n_turns=("turn_index", "count"),
            n_user_turns=("role", lambda s: _count_role(s, "user")),
            n_assistant_turns=("role", lambda s: _count_role(s, "assistant")),
            approx_tokens_total=("approx_tokens", "sum"),
        )
        .reset_index()
    )
    return metrics


def extract_first_assistant_response(df: pd.DataFrame, conversation_col: str) -> pd.Series:
    """
    Extracts the first assistant message content from conversation list.
    Works even if conversation has multiple turns.
    Returns a Series aligned to df.
    """
    def _first_assistant(conv):
        msgs = _normalize_messages(conv)
        for m in msgs:
            if str(m.get("role", "")).lower() == "assistant":
                return m.get("content", "")
        return ""

    return df[conversation_col].apply(_first_assistant)


def build_battle_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a battle-level table with side-by-side A/B responses and metadata.
    Useful for:
      - labeling sycophancy_a / sycophancy_b
      - modeling winner vs sycophancy difference
    """
    out = df.copy()
    if "ts_utc" not in out.columns:
        out = add_ts_utc(out, "tstamp")

    out["user_id"] = out["judge"]

    out["user_prompt"] = out["conversation_a"].apply(
        lambda c: next((m.get("content", "") for m in _normalize_messages(c)
                        if str(m.get("role", "")).lower() == "user"), "")
    )

    out["assistant_a"] = extract_first_assistant_response(out, "conversation_a")
    out["assistant_b"] = extract_first_assistant_response(out, "conversation_b")

    keep = [
        "question_id", "user_id", "judge", "ts_utc",
        "model_a", "model_b", "winner", "language",
        "user_prompt", "assistant_a", "assistant_b",
        "turn", "anony",
        "openai_moderation", "toxic_chat_tag",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep]


# ---------------------------
# 3) Return metrics by user
# ---------------------------

def compute_return_metrics(
    df: pd.DataFrame,
    user_col: str = "judge",
    time_col: str = "ts_utc",
    id_col: str = "question_id",
    windows_days=(1, 7, 30),
) -> pd.DataFrame:
    """
    Compute return metrics at the session level: does the user have a next session within {1,7,30} days?

    Input df: one row per battle/session.
    Required columns: user_col, time_col, id_col
    Output includes:
      next_ts_utc, next_gap_hours, returned_{Nd}, is_last_observation
    """
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"Missing {time_col}. Call add_ts_utc(df) first or pass the right time_col.")

    # Some rows may have missing timestamps
    out = out.dropna(subset=[user_col, time_col]).copy()

    out = out.sort_values([user_col, time_col, id_col])
    out["next_ts_utc"] = out.groupby(user_col)[time_col].shift(-1)
    out["next_gap_hours"] = (out["next_ts_utc"] - out[time_col]).dt.total_seconds() / 3600.0
    out["is_last_observation"] = out["next_ts_utc"].isna()

    for d in windows_days:
        out[f"returned_{d}d"] = out["next_gap_hours"].apply(
            lambda h: False if pd.isna(h) else (h <= 24.0 * d)
        )
    return out


def compute_user_summary(
    sessions_with_returns: pd.DataFrame,
    user_col: str = "judge",
    time_col: str = "ts_utc",
    id_col: str = "question_id",
    windows_days=(7, 30),
) -> pd.DataFrame:
    """
    User-level aggregates for description/diagnostics.
    """
    out = sessions_with_returns.copy()
    agg = {
        "n_sessions": (id_col, "count"),
        "first_ts": (time_col, "min"),
        "last_ts": (time_col, "max"),
    }
    for d in windows_days:
        col = f"returned_{d}d"
        if col in out.columns:
            agg[f"p_return_{d}d"] = (col, "mean")

    return out.groupby(user_col, dropna=False).agg(**agg).reset_index()


# ---------------------------
# 4) Convenience wrapper: build everything
# ---------------------------

def build_all_metrics(df: pd.DataFrame) -> dict:
    """
    One-stop function to produce:
      - turns_a, turns_b
      - session_metrics_a, session_metrics_b
      - battle_table (side-by-side)
      - returns_table (per battle)
      - user_summary
    """
    df2 = add_ts_utc(df, "tstamp")

    turns_a = extract_turn_table(df2, "conversation_a", "a", model_col="model_a")
    turns_b = extract_turn_table(df2, "conversation_b", "b", model_col="model_b")

    metrics_a = compute_session_metrics(turns_a)
    metrics_b = compute_session_metrics(turns_b)

    battle = build_battle_level_table(df2)

    # Return metrics at the battle level (user = judge)
    returns = compute_return_metrics(df2, user_col="judge", time_col="ts_utc", id_col="question_id")

    user_sum = compute_user_summary(returns, user_col="judge", time_col="ts_utc", id_col="question_id")

    return {
        "turns_a": turns_a,
        "turns_b": turns_b,
        "session_metrics_a": metrics_a,
        "session_metrics_b": metrics_b,
        "battle_table": battle,
        "returns_table": returns,
        "user_summary": user_sum,
    }