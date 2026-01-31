"""Inter-rater reliability metrics for sycophancy labels."""

import numpy as np
import pandas as pd
from scipy import stats


def compute_weighted_kappa(
    rater1: np.ndarray,
    rater2: np.ndarray,
    weights: str = "quadratic",
) -> dict:
    """
    Compute weighted Cohen's kappa for ordinal ratings.

    Args:
        rater1: Array of ratings from rater 1.
        rater2: Array of ratings from rater 2.
        weights: "linear" or "quadratic" weighting.

    Returns:
        Dict with kappa, interpretation, and confidence interval.
    """
    # Remove pairs with missing values
    mask = ~(np.isnan(rater1) | np.isnan(rater2))
    r1 = rater1[mask].astype(int)
    r2 = rater2[mask].astype(int)

    if len(r1) == 0:
        return {"kappa": np.nan, "n": 0, "interpretation": "No valid pairs"}

    # Get unique categories
    categories = np.unique(np.concatenate([r1, r2]))
    n_cat = len(categories)
    n = len(r1)

    # Build confusion matrix
    cat_map = {c: i for i, c in enumerate(categories)}
    confusion = np.zeros((n_cat, n_cat))
    for i, j in zip(r1, r2):
        confusion[cat_map[i], cat_map[j]] += 1

    confusion /= n

    # Marginals
    sum_rows = confusion.sum(axis=1)
    sum_cols = confusion.sum(axis=0)

    # Weight matrix
    if weights == "quadratic":
        weight_mat = np.array(
            [[(i - j) ** 2 for j in range(n_cat)] for i in range(n_cat)]
        ) / ((n_cat - 1) ** 2)
    else:  # linear
        weight_mat = np.array(
            [[abs(i - j) for j in range(n_cat)] for i in range(n_cat)]
        ) / (n_cat - 1)

    # Observed and expected disagreement
    observed = np.sum(weight_mat * confusion)
    expected = np.sum(weight_mat * np.outer(sum_rows, sum_cols))

    # Kappa
    if expected == 0:
        kappa = 1.0 if observed == 0 else 0.0
    else:
        kappa = 1 - (observed / expected)

    # Interpretation (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Poor"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"

    # Bootstrap CI
    n_bootstrap = 1000
    kappas = []
    indices = np.arange(n)

    for _ in range(n_bootstrap):
        idx = np.random.choice(indices, n, replace=True)
        try:
            k = compute_weighted_kappa(r1[idx].astype(float), r2[idx].astype(float), weights)
            kappas.append(k["kappa"])
        except Exception:
            continue

    if kappas:
        ci_lower = np.percentile(kappas, 2.5)
        ci_upper = np.percentile(kappas, 97.5)
    else:
        ci_lower = ci_upper = np.nan

    return {
        "kappa": kappa,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "interpretation": interpretation,
        "weights": weights,
    }


def compute_irr(
    labels1: pd.DataFrame,
    labels2: pd.DataFrame,
    score_col: str = "sycophancy_a",
    id_col: str = "question_id",
) -> dict:
    """
    Compute inter-rater reliability between two sets of labels.

    Args:
        labels1: Labels from first rater/model.
        labels2: Labels from second rater/model.
        score_col: Column with scores to compare.
        id_col: Column to merge on.

    Returns:
        Dict with reliability metrics.
    """
    # Merge on ID
    merged = labels1[[id_col, score_col]].merge(
        labels2[[id_col, score_col]],
        on=id_col,
        suffixes=("_1", "_2"),
    )

    r1 = merged[f"{score_col}_1"].values
    r2 = merged[f"{score_col}_2"].values

    # Weighted kappa
    kappa_result = compute_weighted_kappa(r1, r2, weights="quadratic")

    # Exact agreement
    exact_agreement = np.mean(r1 == r2)

    # Within-1 agreement
    within_1_agreement = np.mean(np.abs(r1 - r2) <= 1)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(r1, r2)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(r1, r2)

    # Mean absolute difference
    mad = np.mean(np.abs(r1 - r2))

    return {
        **kappa_result,
        "exact_agreement": exact_agreement,
        "within_1_agreement": within_1_agreement,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mean_abs_diff": mad,
        "n_pairs": len(merged),
    }


def compute_model_agreement_matrix(
    labels_dict: dict[str, pd.DataFrame],
    score_col: str = "sycophancy_a",
    id_col: str = "question_id",
) -> pd.DataFrame:
    """
    Compute pairwise agreement matrix between multiple models.

    Args:
        labels_dict: Dict mapping model name to labels DataFrame.
        score_col: Column with scores.
        id_col: Column to merge on.

    Returns:
        DataFrame with pairwise kappa values.
    """
    models = list(labels_dict.keys())
    n_models = len(models)

    matrix = np.zeros((n_models, n_models))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                result = compute_irr(
                    labels_dict[m1],
                    labels_dict[m2],
                    score_col=score_col,
                    id_col=id_col,
                )
                matrix[i, j] = result["kappa"]
                matrix[j, i] = result["kappa"]

    return pd.DataFrame(matrix, index=models, columns=models)
