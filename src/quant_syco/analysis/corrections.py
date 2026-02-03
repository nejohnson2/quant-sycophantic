"""Multiple comparison corrections for hypothesis testing."""

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: np.ndarray | list, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        pvalues: Array of raw p-values.
        alpha: Significance level for FDR control.

    Returns:
        Tuple of (adjusted_pvalues, significant_mask).
        - adjusted_pvalues: FDR-adjusted p-values
        - significant_mask: Boolean array where True = significant after correction
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort p-values and track original indices
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # Compute BH adjusted p-values
    # adjusted_p[i] = p[i] * n / (rank[i])
    # Then enforce monotonicity from the end
    ranks = np.arange(1, n + 1)
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity: adjusted[i] <= adjusted[i+1]
    # Work backwards to ensure this
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)

    # Restore original order
    result = np.empty(n)
    result[sorted_idx] = adjusted

    return result, result < alpha


def bonferroni(pvalues: np.ndarray | list, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction for multiple comparisons.

    More conservative than BH, controls family-wise error rate (FWER).

    Args:
        pvalues: Array of raw p-values.
        alpha: Significance level.

    Returns:
        Tuple of (adjusted_pvalues, significant_mask).
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    adjusted = np.clip(pvalues * n, 0, 1)
    return adjusted, adjusted < alpha


def holm_bonferroni(pvalues: np.ndarray | list, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni, still controls FWER.

    Args:
        pvalues: Array of raw p-values.
        alpha: Significance level.

    Returns:
        Tuple of (adjusted_pvalues, significant_mask).
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # Holm: adjusted_p[i] = max(p[j] * (n - j) for j <= i)
    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_pvals[i] * (n - i)

    # Enforce monotonicity (step-down)
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.clip(adjusted, 0, 1)

    # Restore original order
    result = np.empty(n)
    result[sorted_idx] = adjusted

    return result, result < alpha


def apply_correction_to_dataframe(
    df: pd.DataFrame,
    pvalue_col: str,
    method: str = "bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply multiple comparison correction to a DataFrame.

    Args:
        df: DataFrame containing p-values.
        pvalue_col: Name of column with raw p-values.
        method: Correction method ('bh', 'bonferroni', 'holm').
        alpha: Significance level.

    Returns:
        DataFrame with added columns:
        - p_adjusted: Corrected p-values
        - significant_corrected: Boolean significance after correction
    """
    df = df.copy()

    pvalues = df[pvalue_col].values

    if method == "bh":
        adjusted, significant = benjamini_hochberg(pvalues, alpha)
    elif method == "bonferroni":
        adjusted, significant = bonferroni(pvalues, alpha)
    elif method == "holm":
        adjusted, significant = holm_bonferroni(pvalues, alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bh', 'bonferroni', or 'holm'.")

    df["p_adjusted"] = adjusted
    df["significant_corrected"] = significant

    return df
