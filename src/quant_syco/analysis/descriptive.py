"""Descriptive statistics and exploratory analysis."""

import pandas as pd
import numpy as np
from scipy import stats


def compute_summary_statistics(
    df: pd.DataFrame,
    score_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for sycophancy and politeness scores.

    Args:
        df: DataFrame with score columns.
        score_cols: Columns to summarize. Defaults to sycophancy/politeness.

    Returns:
        DataFrame with summary statistics.
    """
    if score_cols is None:
        score_cols = [c for c in df.columns if "sycophancy" in c or "politeness" in c]
        score_cols = [c for c in score_cols if df[c].dtype in [np.int64, np.float64]]

    summaries = []
    for col in score_cols:
        data = df[col].dropna()
        summaries.append(
            {
                "column": col,
                "n": len(data),
                "mean": data.mean(),
                "std": data.std(),
                "median": data.median(),
                "min": data.min(),
                "max": data.max(),
                "q25": data.quantile(0.25),
                "q75": data.quantile(0.75),
            }
        )

    return pd.DataFrame(summaries)


def compute_sycophancy_by_model(
    df: pd.DataFrame,
    model_col: str = "model_a",
    score_col: str = "sycophancy_a",
) -> pd.DataFrame:
    """
    Compute sycophancy statistics by model.

    Args:
        df: DataFrame with model and score columns.
        model_col: Column containing model names.
        score_col: Column containing sycophancy scores.

    Returns:
        DataFrame with per-model statistics.
    """
    grouped = (
        df.groupby(model_col)[score_col]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(
            columns={
                "count": "n_samples",
                "mean": "mean_sycophancy",
                "std": "std_sycophancy",
                "median": "median_sycophancy",
            }
        )
        .sort_values("mean_sycophancy", ascending=False)
    )

    return grouped


def compute_correlations(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute correlations between feature columns and outcome columns.

    Args:
        df: DataFrame with data.
        x_cols: Predictor columns.
        y_cols: Outcome columns.
        method: Correlation method ("spearman" or "pearson").

    Returns:
        DataFrame with correlation matrix and p-values.
    """
    results = []

    for x_col in x_cols:
        for y_col in y_cols:
            data = df[[x_col, y_col]].dropna()
            if len(data) < 3:
                continue

            if method == "spearman":
                corr, pval = stats.spearmanr(data[x_col], data[y_col])
            else:
                corr, pval = stats.pearsonr(data[x_col], data[y_col])

            results.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "correlation": corr,
                    "p_value": pval,
                    "n": len(data),
                    "significant_05": pval < 0.05,
                    "significant_01": pval < 0.01,
                }
            )

    return pd.DataFrame(results)


def compute_win_rate_by_sycophancy(
    df: pd.DataFrame,
    sycophancy_col: str = "sycophancy_a",
    winner_col: str = "winner",
) -> pd.DataFrame:
    """
    Compute win rate as a function of sycophancy score.

    Args:
        df: DataFrame with sycophancy scores and winner labels.
        sycophancy_col: Column with sycophancy scores.
        winner_col: Column with winner labels ("model_a", "model_b", "tie").

    Returns:
        DataFrame with win rates by sycophancy level.
    """
    df_clean = df.dropna(subset=[sycophancy_col, winner_col])

    # Determine side from column name
    side = "a" if "_a" in sycophancy_col else "b"
    win_label = f"model_{side}"

    # Compute win rate by sycophancy level
    results = (
        df_clean.groupby(sycophancy_col)
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "n_wins": (x[winner_col] == win_label).sum(),
                    "n_ties": (x[winner_col] == "tie").sum(),
                    "n_losses": (x[winner_col] == f"model_{'b' if side == 'a' else 'a'}").sum(),
                }
            )
        )
        .reset_index()
    )

    results["win_rate"] = results["n_wins"] / results["n"]
    results["win_or_tie_rate"] = (results["n_wins"] + results["n_ties"]) / results["n"]

    return results


def compute_sycophancy_differential_effect(
    df: pd.DataFrame,
    sycophancy_a_col: str = "sycophancy_a",
    sycophancy_b_col: str = "sycophancy_b",
    winner_col: str = "winner",
) -> pd.DataFrame:
    """
    Analyze how the difference in sycophancy between A and B affects winner.

    Args:
        df: DataFrame with sycophancy scores for both sides.
        sycophancy_a_col: Column with A's sycophancy.
        sycophancy_b_col: Column with B's sycophancy.
        winner_col: Column with winner labels.

    Returns:
        DataFrame with win rates by sycophancy differential.
    """
    df_clean = df.dropna(subset=[sycophancy_a_col, sycophancy_b_col, winner_col])

    # Compute differential
    df_clean = df_clean.copy()
    df_clean["syco_diff"] = df_clean[sycophancy_a_col] - df_clean[sycophancy_b_col]

    # Bin differentials
    df_clean["syco_diff_bin"] = pd.cut(
        df_clean["syco_diff"],
        bins=[-4, -2, -1, 0, 1, 2, 4],
        labels=["A much less", "A less", "A slightly less", "A slightly more", "A more", "A much more"],
    )

    results = (
        df_clean.groupby("syco_diff_bin", observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "a_win_rate": (x[winner_col] == "model_a").mean(),
                    "b_win_rate": (x[winner_col] == "model_b").mean(),
                    "tie_rate": (x[winner_col] == "tie").mean(),
                }
            )
        )
        .reset_index()
    )

    return results
