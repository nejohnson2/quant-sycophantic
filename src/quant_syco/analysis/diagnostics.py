"""Diagnostic functions for propensity score analysis and data quality."""

import numpy as np
import pandas as pd
from typing import Optional


def check_propensity_overlap(
    df: pd.DataFrame,
    ps_col: str = "propensity_score",
    treatment_col: str = "high_sycophancy",
) -> dict:
    """
    Check common support assumption for propensity scores.

    Good overlap means both treated and control groups have similar
    distributions of propensity scores, enabling valid causal inference.

    Args:
        df: DataFrame with propensity scores and treatment indicator.
        ps_col: Column name for propensity scores.
        treatment_col: Column name for binary treatment.

    Returns:
        Dict with overlap statistics and recommendations.
    """
    df_clean = df.dropna(subset=[ps_col, treatment_col]).copy()

    treated = df_clean[df_clean[treatment_col] == 1][ps_col]
    control = df_clean[df_clean[treatment_col] == 0][ps_col]

    # Compute overlap region
    common_min = max(treated.min(), control.min())
    common_max = min(treated.max(), control.max())

    # Count observations outside common support
    n_treated_outside = ((treated < common_min) | (treated > common_max)).sum()
    n_control_outside = ((control < common_min) | (control > common_max)).sum()

    # Compute overlap statistics
    overlap_stats = {
        "n_treated": len(treated),
        "n_control": len(control),
        "treated_ps_min": treated.min(),
        "treated_ps_max": treated.max(),
        "treated_ps_mean": treated.mean(),
        "treated_ps_std": treated.std(),
        "control_ps_min": control.min(),
        "control_ps_max": control.max(),
        "control_ps_mean": control.mean(),
        "control_ps_std": control.std(),
        "common_support_min": common_min,
        "common_support_max": common_max,
        "n_treated_outside_support": n_treated_outside,
        "n_control_outside_support": n_control_outside,
        "pct_treated_outside": 100 * n_treated_outside / len(treated) if len(treated) > 0 else 0,
        "pct_control_outside": 100 * n_control_outside / len(control) if len(control) > 0 else 0,
    }

    # Assess overlap quality
    total_outside = n_treated_outside + n_control_outside
    total_n = len(treated) + len(control)
    pct_outside = 100 * total_outside / total_n if total_n > 0 else 0

    if pct_outside < 5:
        overlap_stats["overlap_quality"] = "Good"
        overlap_stats["recommendation"] = "Proceed with IPW estimation"
    elif pct_outside < 15:
        overlap_stats["overlap_quality"] = "Moderate"
        overlap_stats["recommendation"] = "Consider trimming extreme propensity scores"
    else:
        overlap_stats["overlap_quality"] = "Poor"
        overlap_stats["recommendation"] = "Causal estimates may be unreliable; consider alternative methods"

    return overlap_stats


def compute_covariate_balance(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    ps_col: Optional[str] = None,
    weighted: bool = False,
) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMD) for covariate balance.

    SMD = (mean_treated - mean_control) / pooled_std

    Rule of thumb: |SMD| < 0.1 indicates good balance.

    Args:
        df: DataFrame with treatment and covariates.
        treatment_col: Binary treatment column.
        covariates: List of covariate columns to check.
        ps_col: Propensity score column (for weighted balance).
        weighted: If True, compute IPW-weighted balance.

    Returns:
        DataFrame with SMD for each covariate.
    """
    df_clean = df.dropna(subset=[treatment_col] + covariates).copy()

    treated_mask = df_clean[treatment_col] == 1
    control_mask = df_clean[treatment_col] == 0

    results = []

    for cov in covariates:
        # Check if column is numeric using pandas utility
        is_numeric = pd.api.types.is_numeric_dtype(df_clean[cov])

        if not is_numeric:
            # For categorical, compute balance for each level
            for level in df_clean[cov].unique():
                binary_col = (df_clean[cov] == level).astype(float)

                if weighted and ps_col:
                    ps = df_clean[ps_col].clip(0.05, 0.95)
                    weights_t = 1 / ps
                    weights_c = 1 / (1 - ps)

                    mean_t = np.average(binary_col[treated_mask], weights=weights_t[treated_mask])
                    mean_c = np.average(binary_col[control_mask], weights=weights_c[control_mask])
                else:
                    mean_t = binary_col[treated_mask].mean()
                    mean_c = binary_col[control_mask].mean()

                # Pooled std for binary variable
                pooled_var = (binary_col[treated_mask].var() + binary_col[control_mask].var()) / 2
                pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1

                smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

                results.append({
                    "covariate": f"{cov}={level}",
                    "mean_treated": mean_t,
                    "mean_control": mean_c,
                    "smd": smd,
                    "abs_smd": abs(smd),
                    "balanced": abs(smd) < 0.1,
                })
        else:
            # Numeric covariate
            if weighted and ps_col:
                ps = df_clean[ps_col].clip(0.05, 0.95)
                weights_t = 1 / ps
                weights_c = 1 / (1 - ps)

                mean_t = np.average(df_clean.loc[treated_mask, cov], weights=weights_t[treated_mask])
                mean_c = np.average(df_clean.loc[control_mask, cov], weights=weights_c[control_mask])
            else:
                mean_t = df_clean.loc[treated_mask, cov].mean()
                mean_c = df_clean.loc[control_mask, cov].mean()

            # Pooled standard deviation
            var_t = df_clean.loc[treated_mask, cov].var()
            var_c = df_clean.loc[control_mask, cov].var()
            pooled_std = np.sqrt((var_t + var_c) / 2) if (var_t + var_c) > 0 else 1

            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0

            results.append({
                "covariate": cov,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": smd,
                "abs_smd": abs(smd),
                "balanced": abs(smd) < 0.1,
            })

    return pd.DataFrame(results)


def run_trimming_sensitivity(
    df: pd.DataFrame,
    treatment_col: str = "high_sycophancy",
    outcome_col: str = "a_wins",
    ps_col: str = "propensity_score",
    thresholds: list[float] = None,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """
    Test sensitivity of ATE to propensity score trimming thresholds.

    Trims observations with extreme propensity scores (too close to 0 or 1).
    Shows how stable the estimate is across different trimming choices.

    Args:
        df: DataFrame with treatment, outcome, and propensity scores.
        treatment_col: Binary treatment column.
        outcome_col: Binary outcome column.
        ps_col: Propensity score column.
        thresholds: List of trimming thresholds to test (default: 0.01 to 0.20).
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        DataFrame with ATE estimates at each trimming threshold.
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]

    df_clean = df.dropna(subset=[treatment_col, outcome_col, ps_col]).copy()

    results = []

    for threshold in thresholds:
        # Trim observations outside [threshold, 1-threshold]
        ps = df_clean[ps_col]
        in_support = (ps >= threshold) & (ps <= 1 - threshold)
        df_trimmed = df_clean[in_support].copy()

        if len(df_trimmed) < 100:
            continue

        T = df_trimmed[treatment_col].values
        Y = df_trimmed[outcome_col].values
        ps_trimmed = df_trimmed[ps_col].values

        # Clip to avoid division issues
        ps_trimmed = np.clip(ps_trimmed, threshold, 1 - threshold)

        # IPW estimator
        treated_weight = T / ps_trimmed
        control_weight = (1 - T) / (1 - ps_trimmed)
        ate = np.mean(Y * treated_weight) - np.mean(Y * control_weight)

        # Bootstrap CI
        n = len(df_trimmed)
        ate_bootstrap = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            T_b, Y_b, ps_b = T[idx], Y[idx], ps_trimmed[idx]
            ate_b = np.mean(Y_b * T_b / ps_b) - np.mean(Y_b * (1 - T_b) / (1 - ps_b))
            ate_bootstrap.append(ate_b)

        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)

        results.append({
            "trim_threshold": threshold,
            "n_retained": len(df_trimmed),
            "pct_retained": 100 * len(df_trimmed) / len(df_clean),
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se": np.std(ate_bootstrap),
            "significant": (ci_lower > 0) or (ci_upper < 0),
        })

    return pd.DataFrame(results)


def compute_ate_stabilized(
    df: pd.DataFrame,
    treatment_col: str = "high_sycophancy",
    outcome_col: str = "a_wins",
    ps_col: str = "propensity_score",
    trim_threshold: float = 0.05,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute ATE using stabilized inverse propensity weights.

    Stabilized weights: w = P(T) / P(T|X) instead of 1 / P(T|X)
    This reduces variance when propensity scores are extreme.

    Args:
        df: DataFrame with treatment, outcome, and propensity scores.
        treatment_col: Binary treatment column.
        outcome_col: Binary outcome column.
        ps_col: Propensity score column.
        trim_threshold: Clip PS to [threshold, 1-threshold].
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        Dict with ATE estimate, CI, and comparison to unstabilized.
    """
    df_clean = df.dropna(subset=[treatment_col, outcome_col, ps_col]).copy()

    T = df_clean[treatment_col].values
    Y = df_clean[outcome_col].values
    ps = np.clip(df_clean[ps_col].values, trim_threshold, 1 - trim_threshold)

    # Marginal probability of treatment
    p_treat = T.mean()

    # Unstabilized weights
    w_unstab = np.where(T == 1, 1 / ps, 1 / (1 - ps))

    # Stabilized weights
    w_stab = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    # ATE with unstabilized weights
    ate_unstab = np.mean(Y * T / ps) - np.mean(Y * (1 - T) / (1 - ps))

    # ATE with stabilized weights (Horvitz-Thompson style)
    ate_stab = (
        np.sum(Y * T * w_stab) / np.sum(T * w_stab) -
        np.sum(Y * (1 - T) * w_stab) / np.sum((1 - T) * w_stab)
    )

    # Bootstrap for stabilized
    n = len(df_clean)
    ate_bootstrap = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        T_b, Y_b, ps_b = T[idx], Y[idx], ps[idx]
        w_b = np.where(T_b == 1, p_treat / ps_b, (1 - p_treat) / (1 - ps_b))
        ate_b = (
            np.sum(Y_b * T_b * w_b) / np.sum(T_b * w_b) -
            np.sum(Y_b * (1 - T_b) * w_b) / np.sum((1 - T_b) * w_b)
        )
        ate_bootstrap.append(ate_b)

    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)

    return {
        "ate_stabilized": ate_stab,
        "ate_unstabilized": ate_unstab,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": np.std(ate_bootstrap),
        "weight_variance_unstab": np.var(w_unstab),
        "weight_variance_stab": np.var(w_stab),
        "variance_reduction": 1 - np.var(w_stab) / np.var(w_unstab) if np.var(w_unstab) > 0 else 0,
        "n_treated": T.sum(),
        "n_control": (1 - T).sum(),
        "p_treatment": p_treat,
    }


def track_sample_exclusions(
    df_raw: pd.DataFrame,
    df_labeled: pd.DataFrame,
    df_final: pd.DataFrame,
    label_col: str = "sycophancy_a",
) -> dict:
    """
    Track sample sizes through the analysis pipeline.

    Documents all exclusions for transparency reporting.

    Args:
        df_raw: Raw battle data.
        df_labeled: After merging with labels.
        df_final: Final analysis sample.
        label_col: A label column to check for missingness.

    Returns:
        Dict with sample sizes and exclusion reasons.
    """
    n_raw = len(df_raw)
    n_labeled = len(df_labeled)
    n_final = len(df_final)

    # Count specific exclusions
    n_no_label = n_raw - n_labeled

    # Ties
    n_ties = 0
    n_bothbad = 0
    if "winner" in df_labeled.columns:
        n_ties = (df_labeled["winner"] == "tie").sum()
        n_bothbad = (df_labeled["winner"] == "tie (bothbad)").sum()

    # Missing labels
    n_missing_labels = df_labeled[label_col].isna().sum() if label_col in df_labeled.columns else 0

    return {
        "n_raw": n_raw,
        "n_after_label_merge": n_labeled,
        "n_after_tie_exclusion": n_labeled - n_ties - n_bothbad,
        "n_final": n_final,
        "exclusions": {
            "no_label_available": n_no_label,
            "tie": n_ties,
            "tie_bothbad": n_bothbad,
            "missing_values": n_labeled - n_ties - n_bothbad - n_final,
        },
        "retention_rate": n_final / n_raw if n_raw > 0 else 0,
        "pct_retained": 100 * n_final / n_raw if n_raw > 0 else 0,
    }
