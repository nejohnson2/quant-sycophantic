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


# ---------------------------------------------------------------------------
# Sensitivity analysis for unmeasured confounding
# ---------------------------------------------------------------------------


def compute_e_value(
    estimate: float,
    ci_bound: Optional[float] = None,
    estimate_type: str = "OR",
) -> dict:
    """Compute the E-value for sensitivity to unmeasured confounding.

    The E-value (VanderWeele & Ding, 2017) answers: how strong would an
    unmeasured confounder need to be, in terms of its association with both
    treatment and outcome, to fully explain away the observed effect?

    A large E-value means the effect is robust; a small E-value means a
    modest confounder could explain it away.

    Args:
        estimate: Point estimate (odds ratio or risk ratio).
        ci_bound: The CI bound closest to the null (1.0). If provided,
            computes the E-value for the CI bound as well.
        estimate_type: "OR" for odds ratio, "RR" for risk ratio, or
            "beta" for log-odds coefficient (will be exponentiated).

    Returns:
        Dict with E-value for point estimate and CI bound.
    """
    if estimate_type == "beta":
        estimate = np.exp(estimate)
        if ci_bound is not None:
            ci_bound = np.exp(ci_bound)
        estimate_type = "OR"

    # For rare outcomes, OR ≈ RR. For common outcomes, convert using
    # the square-root approximation: RR_approx ≈ sqrt(OR) when prevalence
    # is moderate. We use the OR directly with the standard formula,
    # which is conservative (the true E-value may be larger).
    rr = estimate

    def _e_value(rr_val: float) -> float:
        """E-value formula for a risk/odds ratio."""
        if rr_val < 1:
            rr_val = 1 / rr_val
        return rr_val + np.sqrt(rr_val * (rr_val - 1))

    e_point = _e_value(rr)

    result = {
        "estimate": estimate,
        "estimate_type": estimate_type,
        "e_value_point": e_point,
        "interpretation": (
            f"An unmeasured confounder would need to be associated with both "
            f"treatment and outcome by a factor of {e_point:.2f} each "
            f"(beyond measured covariates) to explain away the observed effect."
        ),
    }

    if ci_bound is not None:
        if ci_bound <= 1.0 and estimate > 1.0:
            # CI crosses null — effect is not robust
            result["e_value_ci"] = 1.0
            result["ci_interpretation"] = (
                "The confidence interval includes the null; E-value for CI = 1.0."
            )
        elif ci_bound >= 1.0 and estimate < 1.0:
            result["e_value_ci"] = 1.0
            result["ci_interpretation"] = (
                "The confidence interval includes the null; E-value for CI = 1.0."
            )
        else:
            e_ci = _e_value(ci_bound)
            result["e_value_ci"] = e_ci
            result["ci_interpretation"] = (
                f"To move the CI bound to the null requires a confounder "
                f"associated by a factor of {e_ci:.2f} with both treatment "
                f"and outcome."
            )

    return result


def compute_e_values_table(
    results: list[dict],
) -> pd.DataFrame:
    """Compute E-values for a set of regression results.

    Args:
        results: List of dicts, each with keys:
            - label: description of the estimate
            - or_estimate: odds ratio (or beta if log-odds)
            - ci_lower: lower CI bound
            - ci_upper: upper CI bound

    Returns:
        DataFrame with E-values for each estimate.
    """
    rows = []
    for r in results:
        or_est = r["or_estimate"]
        ci_lower = r.get("ci_lower")
        ci_upper = r.get("ci_upper")

        # The CI bound closest to 1.0
        if or_est >= 1.0:
            ci_null = ci_lower
        else:
            ci_null = ci_upper

        e = compute_e_value(or_est, ci_null, estimate_type="OR")

        rows.append({
            "label": r["label"],
            "OR": or_est,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper,
            "E_value_point": e["e_value_point"],
            "E_value_CI": e.get("e_value_ci", np.nan),
        })

    return pd.DataFrame(rows)


def rosenbaum_sensitivity(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity_scores: np.ndarray,
    gamma_range: Optional[list[float]] = None,
    n_pairs: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Rosenbaum sensitivity analysis for hidden bias.

    Tests how robust the treatment-outcome association is to an unobserved
    confounder that changes the odds of treatment assignment by a factor of
    Gamma. Uses a matched-pair approach.

    At Gamma = 1.0, there is no hidden bias and the p-value should be small
    (confirming the observed effect). As Gamma increases, the p-value
    increases; the critical Gamma at which the p-value crosses 0.05
    indicates the study's sensitivity to hidden bias.

    Args:
        treatment: Binary treatment array (0/1).
        outcome: Binary outcome array (0/1).
        propensity_scores: Estimated propensity scores.
        gamma_range: List of Gamma values to test.
        n_pairs: Number of matched pairs to form.
        seed: Random seed for matching.

    Returns:
        DataFrame with Gamma, p_upper, p_lower, and significance.
    """
    from scipy.stats import norm

    if gamma_range is None:
        gamma_range = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    rng = np.random.default_rng(seed)
    treatment = np.asarray(treatment, dtype=int)
    outcome = np.asarray(outcome, dtype=float)
    ps = np.asarray(propensity_scores, dtype=float)

    # --- Nearest-neighbor matching on propensity score ---
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return pd.DataFrame()

    # Match each treated unit to nearest control (without replacement)
    pairs = []
    available_controls = set(control_idx.tolist())
    control_ps = ps[control_idx]

    for t_idx in rng.permutation(treated_idx):
        if not available_controls or len(pairs) >= n_pairs:
            break
        avail = np.array(sorted(available_controls))
        distances = np.abs(ps[t_idx] - ps[avail])
        best = avail[np.argmin(distances)]
        pairs.append((t_idx, best))
        available_controls.discard(best)

    if len(pairs) < 10:
        return pd.DataFrame()

    # Compute outcome differences within pairs
    diffs = np.array([outcome[t] - outcome[c] for t, c in pairs])
    n_matched = len(pairs)

    results = []
    for gamma in gamma_range:
        # Under hidden bias Gamma, the probability that the treated unit
        # has a higher outcome can range from 1/(1+Gamma) to Gamma/(1+Gamma)
        p_plus = gamma / (1 + gamma)  # upper bound on P(treated has higher outcome)
        p_minus = 1 / (1 + gamma)     # lower bound

        # Number of concordant pairs (treated outcome > control outcome)
        n_positive = np.sum(diffs > 0)
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero == 0:
            results.append({
                "gamma": gamma,
                "n_pairs": n_matched,
                "p_upper": 1.0,
                "p_lower": 1.0,
                "significant_upper": False,
            })
            continue

        # McNemar-style test statistic under Gamma
        # Upper bound: assume hidden bias maximally inflates concordant pairs
        expected_upper = n_nonzero * p_plus
        var_upper = n_nonzero * p_plus * (1 - p_plus)
        z_upper = (n_positive - expected_upper) / np.sqrt(var_upper) if var_upper > 0 else 0
        p_upper = 1 - norm.cdf(z_upper)

        # Lower bound: assume hidden bias maximally deflates concordant pairs
        expected_lower = n_nonzero * p_minus
        var_lower = n_nonzero * p_minus * (1 - p_minus)
        z_lower = (n_positive - expected_lower) / np.sqrt(var_lower) if var_lower > 0 else 0
        p_lower = 1 - norm.cdf(z_lower)

        results.append({
            "gamma": gamma,
            "n_pairs": n_matched,
            "n_concordant": int(n_positive),
            "n_discordant": int(n_nonzero - n_positive),
            "p_upper": p_upper,
            "p_lower": p_lower,
            "significant_upper": p_upper < 0.05,
        })

    df_results = pd.DataFrame(results)

    # Find critical Gamma: the first Gamma where p_upper > 0.05
    significant = df_results[df_results["significant_upper"]]
    if len(significant) == len(df_results):
        df_results.attrs["critical_gamma"] = f"> {gamma_range[-1]}"
    elif len(significant) == 0:
        df_results.attrs["critical_gamma"] = "< 1.0 (effect not significant)"
    else:
        last_sig = significant["gamma"].max()
        df_results.attrs["critical_gamma"] = last_sig

    return df_results
