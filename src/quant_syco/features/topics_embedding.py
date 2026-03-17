"""Improved topic classification using LLM zero-shot or embeddings.

Provides alternatives to the keyword-based classifier for robustness checks.
Primary approach: use Ollama (locally available) for zero-shot classification.
Fallback: use transformers zero-shot NLI if installed.
"""

import json
import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

TOPIC_LABELS = [
    "coding",
    "creative_writing",
    "math",
    "factual_qa",
    "advice",
    "opinion",
    "other",
]

TOPIC_DESCRIPTIONS = {
    "coding": "programming, software development, debugging, or code-related questions",
    "creative_writing": "writing stories, poems, essays, or other creative content",
    "math": "mathematical problems, equations, proofs, or numerical reasoning",
    "factual_qa": "factual questions seeking objective information or explanations",
    "advice": "seeking recommendations, guidance, or help making decisions",
    "opinion": "asking for opinions, comparisons, or subjective evaluations",
    "other": "general conversation or topics not fitting the above categories",
}

_CLASSIFY_PROMPT = """Classify the following user prompt into EXACTLY ONE of these categories:
- coding: {coding}
- creative_writing: {creative_writing}
- math: {math}
- factual_qa: {factual_qa}
- advice: {advice}
- opinion: {opinion}
- other: {other}

User prompt: "{prompt}"

Respond with ONLY a JSON object: {{"topic": "<category>"}}""".format(
    **TOPIC_DESCRIPTIONS, prompt="{prompt}"
)


def classify_topics_ollama(
    df: pd.DataFrame,
    prompt_col: str = "user_prompt",
    model: str = "llama3.2:latest",
    batch_size: int = 50,
    sample: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Classify topics using Ollama LLM zero-shot classification.

    Args:
        df: DataFrame with user prompts.
        prompt_col: Column containing prompts to classify.
        model: Ollama model to use.
        batch_size: Number of prompts per progress update.
        sample: If set, only classify this many prompts (random sample).
        seed: Random seed for sampling.

    Returns:
        DataFrame with 'topic_llm' column added.
    """
    import requests

    result = df.copy()

    if sample and sample < len(df):
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(df), sample, replace=False)
        classify_mask = pd.Series(False, index=df.index)
        classify_mask.iloc[sample_idx] = True
    else:
        classify_mask = pd.Series(True, index=df.index)

    topics = pd.Series("", index=df.index)
    prompts_to_classify = df.loc[classify_mask, prompt_col]

    logger.info("Classifying %d prompts with %s", len(prompts_to_classify), model)

    for idx in tqdm(prompts_to_classify.index, desc="LLM topic classification"):
        prompt_text = str(prompts_to_classify[idx])[:500]  # Truncate long prompts
        classify_prompt = _CLASSIFY_PROMPT.format(prompt=prompt_text)

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": classify_prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["response"]

            # Parse JSON from response
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                parsed = json.loads(match.group())
                topic = parsed.get("topic", "other").lower().strip()
                if topic in TOPIC_LABELS:
                    topics[idx] = topic
                else:
                    topics[idx] = "other"
            else:
                topics[idx] = "other"
        except Exception as e:
            logger.debug("Classification failed for idx %s: %s", idx, e)
            topics[idx] = "other"

    result["topic_llm"] = topics
    return result


def classify_topics_zero_shot(
    df: pd.DataFrame,
    prompt_col: str = "user_prompt",
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Classify topics using HuggingFace zero-shot NLI model.

    Requires: pip install transformers torch

    Args:
        df: DataFrame with user prompts.
        prompt_col: Column containing prompts.
        model_name: HuggingFace model for zero-shot classification.
        batch_size: Batch size for inference.

    Returns:
        DataFrame with 'topic_zs' and probability columns.
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers is required for zero-shot classification. "
            "Install with: pip install transformers torch"
        )

    classifier = pipeline("zero-shot-classification", model=model_name, device=-1)

    candidate_labels = [
        TOPIC_DESCRIPTIONS[t] for t in TOPIC_LABELS if t != "other"
    ]
    label_to_topic = {
        TOPIC_DESCRIPTIONS[t]: t for t in TOPIC_LABELS if t != "other"
    }

    result = df.copy()
    prompts = df[prompt_col].fillna("").str[:512].tolist()

    all_results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Zero-shot classification"):
        batch = prompts[i : i + batch_size]
        batch_results = classifier(batch, candidate_labels, multi_label=False)
        if not isinstance(batch_results, list):
            batch_results = [batch_results]
        all_results.extend(batch_results)

    topics = []
    probs = {t: [] for t in TOPIC_LABELS if t != "other"}
    for r in all_results:
        top_label = r["labels"][0]
        top_topic = label_to_topic.get(top_label, "other")
        top_score = r["scores"][0]

        if top_score < 0.3:
            topics.append("other")
        else:
            topics.append(top_topic)

        for label, score in zip(r["labels"], r["scores"]):
            topic = label_to_topic.get(label)
            if topic:
                probs[topic].append(score)

    result["topic_zs"] = topics
    for topic, scores in probs.items():
        if scores:
            result[f"topic_prob_{topic}"] = scores

    return result


def compare_classifiers(
    df: pd.DataFrame,
    keyword_col: str = "topic",
    llm_col: str = "topic_llm",
) -> dict:
    """Compare keyword and LLM-based topic classifiers.

    Args:
        df: DataFrame with both classifier outputs.
        keyword_col: Column with keyword-based topics.
        llm_col: Column with LLM-based topics.

    Returns:
        Dict with agreement statistics and confusion matrix.
    """
    df_clean = df.dropna(subset=[keyword_col, llm_col])
    df_clean = df_clean[df_clean[llm_col] != ""]

    if len(df_clean) == 0:
        return {"error": "No overlapping classifications found"}

    # Overall agreement
    agreement = (df_clean[keyword_col] == df_clean[llm_col]).mean()

    # Confusion matrix
    confusion = pd.crosstab(
        df_clean[keyword_col],
        df_clean[llm_col],
        rownames=["Keyword"],
        colnames=["LLM"],
    )

    # Per-topic agreement
    per_topic = {}
    for topic in TOPIC_LABELS:
        kw_mask = df_clean[keyword_col] == topic
        llm_mask = df_clean[llm_col] == topic
        tp = (kw_mask & llm_mask).sum()
        kw_n = kw_mask.sum()
        llm_n = llm_mask.sum()
        per_topic[topic] = {
            "keyword_n": int(kw_n),
            "llm_n": int(llm_n),
            "agree": int(tp),
            "keyword_precision": tp / llm_n if llm_n > 0 else 0,
            "keyword_recall": tp / kw_n if kw_n > 0 else 0,
        }

    # Cohen's kappa for nominal agreement
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(df_clean[keyword_col], df_clean[llm_col])

    return {
        "n_items": len(df_clean),
        "overall_agreement": agreement,
        "cohen_kappa": kappa,
        "confusion_matrix": confusion,
        "per_topic": per_topic,
    }
