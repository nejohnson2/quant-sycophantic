"""Causal inference models for sycophancy effects."""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list[str],
    treatment_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Compute propensity scores for high vs low sycophancy.

    Args:
        df: DataFrame with treatment and covariates.
        treatment_col: Sycophancy score column.
        covariate_cols: Confounding variables to control for.
        treatment_threshold: Score above which counts as "high sycophancy".

    Returns:
        DataFrame with propensity scores added.
    """
    df_clean = df.dropna(subset=[treatment_col] + covariate_cols).copy()

    # Binary treatment: high vs low sycophancy
    df_clean["high_sycophancy"] = (df_clean[treatment_col] >= treatment_threshold).astype(int)

    # Prepare features
    X = df_clean[covariate_cols].copy()

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression for propensity
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, df_clean["high_sycophancy"])

    df_clean["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    return df_clean


def compute_ate(
    df: pd.DataFrame,
    treatment_col: str = "high_sycophancy",
    outcome_col: str = "returned_7d",
    propensity_col: str = "propensity_score",
) -> dict:
    """
    Compute Average Treatment Effect using inverse propensity weighting.

    Args:
        df: DataFrame with treatment, outcome, and propensity scores.
        treatment_col: Binary treatment column.
        outcome_col: Binary outcome column.
        propensity_col: Propensity score column.

    Returns:
        Dict with ATE estimate and confidence interval.
    """
    df_clean = df.dropna(subset=[treatment_col, outcome_col, propensity_col]).copy()

    T = df_clean[treatment_col].values
    Y = df_clean[outcome_col].values
    ps = df_clean[propensity_col].values

    # Clip propensity scores to avoid extreme weights
    ps = np.clip(ps, 0.05, 0.95)

    # IPW estimator
    treated_weight = T / ps
    control_weight = (1 - T) / (1 - ps)

    ate = np.mean(Y * treated_weight) - np.mean(Y * control_weight)

    # Bootstrap for confidence interval
    n_bootstrap = 1000
    ate_bootstrap = []
    n = len(df_clean)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        T_b, Y_b, ps_b = T[idx], Y[idx], ps[idx]
        ate_b = np.mean(Y_b * T_b / ps_b) - np.mean(Y_b * (1 - T_b) / (1 - ps_b))
        ate_bootstrap.append(ate_b)

    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)

    return {
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_treated": T.sum(),
        "n_control": (1 - T).sum(),
    }


def run_outcome_regression(
    df: pd.DataFrame,
    outcome_col: str,
    sycophancy_col: str,
    control_cols: list[str],
    logistic: bool = True,
) -> dict:
    """
    Run regression of outcome on sycophancy with controls.

    Args:
        df: DataFrame with outcome and predictors.
        outcome_col: Dependent variable (e.g., "returned_7d").
        sycophancy_col: Sycophancy score column.
        control_cols: Control variables.
        logistic: If True, use logistic regression for binary outcome.

    Returns:
        Dict with model summary and key coefficients.
    """
    all_cols = [outcome_col, sycophancy_col] + control_cols
    df_clean = df.dropna(subset=all_cols).copy()

    # Prepare X
    X = df_clean[[sycophancy_col] + control_cols].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X)

    y = df_clean[outcome_col]

    if logistic:
        model = sm.Logit(y, X).fit(disp=0)
    else:
        model = sm.OLS(y, X).fit()

    # Extract sycophancy coefficient
    syco_coef = model.params.get(sycophancy_col, np.nan)
    syco_pval = model.pvalues.get(sycophancy_col, np.nan)
    syco_se = model.bse.get(sycophancy_col, np.nan)

    return {
        "model": model,
        "sycophancy_coef": syco_coef,
        "sycophancy_pval": syco_pval,
        "sycophancy_se": syco_se,
        "sycophancy_ci_lower": syco_coef - 1.96 * syco_se,
        "sycophancy_ci_upper": syco_coef + 1.96 * syco_se,
        "pseudo_r2": getattr(model, "prsquared", None) or model.rsquared,
        "n_obs": len(df_clean),
        "summary": model.summary(),
    }


def run_winner_regression(
    df: pd.DataFrame,
    sycophancy_a_col: str = "sycophancy_a",
    sycophancy_b_col: str = "sycophancy_b",
    politeness_a_col: str = "politeness_a",
    politeness_b_col: str = "politeness_b",
    control_cols: list[str] = None,
    include_length: bool = False,
    length_a_col: str = "assistant_a_word_count",
    length_b_col: str = "assistant_b_word_count",
    length_nonlinear: bool = False,
    cluster_col: str = None,
) -> dict:
    """
    Model winner as a function of sycophancy differential.

    Outcome: 1 = A wins, 0 = B wins (ties excluded).

    Args:
        df: DataFrame with battle data.
        sycophancy_a_col: A's sycophancy score.
        sycophancy_b_col: B's sycophancy score.
        politeness_a_col: A's politeness score.
        politeness_b_col: B's politeness score.
        control_cols: Additional control variables.
        include_length: If True, add length differential as control.
        length_a_col: Column for A's response length.
        length_b_col: Column for B's response length.
        length_nonlinear: If True, add quadratic length term.
        cluster_col: Column to cluster standard errors by (e.g., 'user_id', 'model_a').

    Returns:
        Dict with regression results.
    """
    control_cols = control_cols or []

    # Exclude ties
    df_clean = df[df["winner"].isin(["model_a", "model_b"])].copy()

    # Drop rows with missing values in required columns
    required_cols = [sycophancy_a_col, sycophancy_b_col] + control_cols
    if include_length:
        required_cols += [length_a_col, length_b_col]
    df_clean = df_clean.dropna(subset=required_cols)

    # Create outcome: 1 = A wins
    df_clean["a_wins"] = (df_clean["winner"] == "model_a").astype(int)

    # Create differentials
    df_clean["syco_diff"] = df_clean[sycophancy_a_col] - df_clean[sycophancy_b_col]

    predictors = ["syco_diff"]

    if politeness_a_col in df_clean.columns and politeness_b_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[politeness_a_col, politeness_b_col])
        df_clean["polite_diff"] = df_clean[politeness_a_col] - df_clean[politeness_b_col]
        predictors.append("polite_diff")

    # Add length controls if requested
    if include_length:
        df_clean["length_diff"] = df_clean[length_a_col] - df_clean[length_b_col]
        predictors.append("length_diff")
        if length_nonlinear:
            df_clean["length_diff_sq"] = df_clean["length_diff"] ** 2
            predictors.append("length_diff_sq")

    predictors += control_cols

    # Prepare X - convert categoricals to dummies and ensure numeric types
    X = df_clean[predictors].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = sm.add_constant(X)

    y = df_clean["a_wins"]

    # Fit model with optional clustering
    if cluster_col and cluster_col in df_clean.columns:
        model = sm.Logit(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_clean[cluster_col]},
            disp=0,
        )
    else:
        model = sm.Logit(y, X).fit(disp=0)

    return {
        "model": model,
        "syco_diff_coef": model.params.get("syco_diff", np.nan),
        "syco_diff_pval": model.pvalues.get("syco_diff", np.nan),
        "syco_diff_se": model.bse.get("syco_diff", np.nan),
        "syco_diff_or": np.exp(model.params.get("syco_diff", np.nan)),  # Odds ratio
        "polite_diff_coef": model.params.get("polite_diff", np.nan),
        "polite_diff_pval": model.pvalues.get("polite_diff", np.nan),
        "length_diff_coef": model.params.get("length_diff", np.nan),
        "length_diff_pval": model.pvalues.get("length_diff", np.nan),
        "pseudo_r2": model.prsquared,
        "n_obs": len(df_clean),
        "clustered": cluster_col is not None,
        "summary": model.summary(),
    }


def run_heterogeneous_effects(
    df: pd.DataFrame,
    outcome_col: str,
    sycophancy_col: str,
    moderator_col: str,
    control_cols: list[str] = None,
    fdr_correction: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate heterogeneous treatment effects by a moderator variable.

    Args:
        df: DataFrame with data.
        outcome_col: Outcome variable.
        sycophancy_col: Sycophancy score.
        moderator_col: Variable to stratify by (e.g., topic).
        control_cols: Control variables.
        fdr_correction: If True, apply Benjamini-Hochberg FDR correction.
        alpha: Significance level for FDR correction.

    Returns:
        DataFrame with effects by moderator level, including corrected p-values.
    """
    control_cols = control_cols or []
    results = []

    for level in df[moderator_col].unique():
        subset = df[df[moderator_col] == level]
        if len(subset) < 50:
            continue

        try:
            result = run_outcome_regression(
                subset,
                outcome_col=outcome_col,
                sycophancy_col=sycophancy_col,
                control_cols=control_cols,
            )
            results.append(
                {
                    "moderator": moderator_col,
                    "level": level,
                    "n": result["n_obs"],
                    "sycophancy_coef": result["sycophancy_coef"],
                    "sycophancy_pval": result["sycophancy_pval"],
                    "sycophancy_ci_lower": result["sycophancy_ci_lower"],
                    "sycophancy_ci_upper": result["sycophancy_ci_upper"],
                }
            )
        except Exception:
            continue

    df_results = pd.DataFrame(results)

    # Apply FDR correction if requested and there are results
    if fdr_correction and len(df_results) > 0:
        from .corrections import benjamini_hochberg

        pvals = df_results["sycophancy_pval"].values
        adjusted, significant = benjamini_hochberg(pvals, alpha)
        df_results["p_adjusted"] = adjusted
        df_results["significant_fdr"] = significant

    return df_results


def run_coefficient_stability(
    df: pd.DataFrame,
    sycophancy_a_col: str = "sycophancy_a",
    sycophancy_b_col: str = "sycophancy_b",
    politeness_a_col: str = "politeness_a",
    politeness_b_col: str = "politeness_b",
    length_a_col: str = "assistant_a_word_count",
    length_b_col: str = "assistant_b_word_count",
    topic_col: str = "topic",
    model_col: str = "model_a",
) -> pd.DataFrame:
    """
    Run series of regressions to show coefficient stability across specifications.

    Tests whether sycophancy effect is robust to adding controls for:
    - Politeness
    - Response length (linear and quadratic)
    - Topic fixed effects
    - Model fixed effects

    Args:
        df: DataFrame with battle data.
        sycophancy_a_col, sycophancy_b_col: Sycophancy score columns.
        politeness_a_col, politeness_b_col: Politeness score columns.
        length_a_col, length_b_col: Response length columns.
        topic_col: Topic column for fixed effects.
        model_col: Model column for fixed effects.

    Returns:
        DataFrame with coefficient estimates across specifications.
    """
    specs = []

    # Spec 1: Sycophancy only
    try:
        result = run_winner_regression(
            df,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=None,
            politeness_b_col=None,
            include_length=False,
        )
        specs.append({
            "specification": "1. Sycophancy only",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    # Spec 2: + Politeness
    try:
        result = run_winner_regression(
            df,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=politeness_a_col,
            politeness_b_col=politeness_b_col,
            include_length=False,
        )
        specs.append({
            "specification": "2. + Politeness",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    # Spec 3: + Length (linear)
    try:
        result = run_winner_regression(
            df,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=politeness_a_col,
            politeness_b_col=politeness_b_col,
            include_length=True,
            length_a_col=length_a_col,
            length_b_col=length_b_col,
            length_nonlinear=False,
        )
        specs.append({
            "specification": "3. + Length (linear)",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    # Spec 4: + Length (quadratic)
    try:
        result = run_winner_regression(
            df,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=politeness_a_col,
            politeness_b_col=politeness_b_col,
            include_length=True,
            length_a_col=length_a_col,
            length_b_col=length_b_col,
            length_nonlinear=True,
        )
        specs.append({
            "specification": "4. + Length (quadratic)",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    # Spec 5: + Topic FE
    try:
        result = run_winner_regression(
            df,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=politeness_a_col,
            politeness_b_col=politeness_b_col,
            include_length=True,
            length_a_col=length_a_col,
            length_b_col=length_b_col,
            length_nonlinear=True,
            control_cols=[topic_col],
        )
        specs.append({
            "specification": "5. + Topic FE",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    # Spec 6: + Model FE (top 10 models only)
    try:
        top_models = df[model_col].value_counts().head(10).index
        df_top = df[df[model_col].isin(top_models)].copy()

        result = run_winner_regression(
            df_top,
            sycophancy_a_col=sycophancy_a_col,
            sycophancy_b_col=sycophancy_b_col,
            politeness_a_col=politeness_a_col,
            politeness_b_col=politeness_b_col,
            include_length=True,
            length_a_col=length_a_col,
            length_b_col=length_b_col,
            length_nonlinear=True,
            control_cols=[topic_col, model_col],
        )
        specs.append({
            "specification": "6. + Model FE (top 10)",
            "syco_diff_coef": result["syco_diff_coef"],
            "syco_diff_se": result["syco_diff_se"],
            "syco_diff_pval": result["syco_diff_pval"],
            "syco_diff_or": result["syco_diff_or"],
            "pseudo_r2": result["pseudo_r2"],
            "n_obs": result["n_obs"],
        })
    except Exception:
        pass

    return pd.DataFrame(specs)


# ---------------------------------------------------------------------------
# Formal mediation analysis (Imai, Keele & Tingley 2010)
# ---------------------------------------------------------------------------


def compute_mediation_decomposition(
    df: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    covariate_cols: list[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Causal mediation decomposition for binary outcomes.

    Implements the potential-outcomes mediation framework of Imai, Keele &
    Tingley (2010). Decomposes the total effect of treatment on outcome into:

    - ACME (Average Causal Mediation Effect): the indirect pathway
      through the mediator (e.g., sycophancy -> length -> win)
    - ADE (Average Direct Effect): the effect NOT through the mediator
      (e.g., sycophancy -> win, holding length constant)
    - Proportion mediated: ACME / Total Effect

    This replaces the informal "87% attenuation" language with formal
    mediation quantities and bootstrap confidence intervals.

    Args:
        df: DataFrame with all required columns.
        treatment_col: Continuous or binary treatment (e.g., syco_diff).
        mediator_col: Mediator variable (e.g., length_diff).
        outcome_col: Binary outcome (e.g., a_wins).
        covariate_cols: Additional covariates to control for.
        n_bootstrap: Number of bootstrap iterations for CIs.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ACME, ADE, total effect, proportion mediated, and CIs.
    """
    covariate_cols = covariate_cols or []
    all_cols = [treatment_col, mediator_col, outcome_col] + covariate_cols
    df_clean = df.dropna(subset=all_cols).copy()

    if len(df_clean) < 100:
        raise ValueError(f"Too few observations after dropping NAs: {len(df_clean)}")

    rng = np.random.default_rng(seed)

    def _estimate_mediation(data: pd.DataFrame) -> dict:
        """Estimate mediation quantities on a single (possibly bootstrapped) sample."""
        # Step 1: Mediator model (OLS) — M = a0 + a1*T + a2*X + e
        X_med = data[[treatment_col] + covariate_cols].copy()
        X_med = pd.get_dummies(X_med, drop_first=True, dtype=float)
        X_med = sm.add_constant(X_med)
        M = data[mediator_col]

        try:
            med_model = sm.OLS(M, X_med).fit(disp=0)
        except Exception:
            return None

        # Step 2: Outcome model (Logit) — Y = b0 + b1*T + b2*M + b3*X + e
        X_out = data[[treatment_col, mediator_col] + covariate_cols].copy()
        X_out = pd.get_dummies(X_out, drop_first=True, dtype=float)
        X_out = sm.add_constant(X_out)
        Y = data[outcome_col]

        try:
            out_model = sm.Logit(Y, X_out).fit(disp=0, maxiter=100)
        except Exception:
            return None

        # Step 3: Simulate potential mediator values under T=t and T=t'
        # Using the approach from Imai et al. (2010):
        # For each observation, predict M(t=1) and M(t=0)
        n = len(data)

        # Get treatment coefficient in mediator model
        a_treatment = med_model.params.get(treatment_col, 0)
        sigma_m = np.sqrt(med_model.mse_resid)

        # Predicted mediator under T=actual and T=counterfactual
        M_actual = med_model.predict(X_med)
        # Counterfactual: shift mediator by -a_treatment * T_actual
        # (what M would have been if T were 0 instead of actual)
        M_counterfactual = M_actual - a_treatment * data[treatment_col].values

        # Step 4: Compute potential outcomes
        # Y(t=1, M(1)) - observed
        # Y(t=1, M(0)) - counterfactual mediator
        # Y(t=0, M(0)) - counterfactual treatment and mediator

        # Create counterfactual design matrices
        X_out_actual = X_out.copy()

        X_out_counter_m = X_out.copy()
        X_out_counter_m[mediator_col] = M_counterfactual

        # Predicted probabilities
        Y_t1_m1 = out_model.predict(X_out_actual)  # Y(t, M(t))

        Y_t1_m0 = out_model.predict(X_out_counter_m)  # Y(t, M(t'))

        # ACME = E[Y(t, M(1)) - Y(t, M(0))]
        acme = np.mean(Y_t1_m1 - Y_t1_m0)

        # ADE = E[Y(1, M(t')) - Y(0, M(t'))]
        # We approximate by the coefficient difference
        b_treatment = out_model.params.get(treatment_col, 0)
        b_mediator = out_model.params.get(mediator_col, 0)

        # Total effect (from model without mediator)
        X_total = data[[treatment_col] + covariate_cols].copy()
        X_total = pd.get_dummies(X_total, drop_first=True, dtype=float)
        X_total = sm.add_constant(X_total)
        try:
            total_model = sm.Logit(Y, X_total).fit(disp=0, maxiter=100)
            total_coef = total_model.params.get(treatment_col, 0)
        except Exception:
            total_coef = b_treatment + b_mediator * a_treatment

        # Direct effect coefficient (from full model)
        direct_coef = b_treatment

        # Indirect via product of coefficients
        indirect_coef = a_treatment * b_mediator

        # Proportion mediated (on coefficient scale)
        total = direct_coef + indirect_coef
        prop_mediated = indirect_coef / total if abs(total) > 1e-10 else 0

        return {
            "acme": acme,
            "ade_coef": direct_coef,
            "indirect_coef": indirect_coef,
            "total_coef": total_coef,
            "direct_coef": direct_coef,
            "prop_mediated": prop_mediated,
            "a_path": a_treatment,  # treatment -> mediator
            "b_path": b_mediator,   # mediator -> outcome (controlling T)
            "c_prime": direct_coef, # direct effect of treatment
            "c_total": total_coef,  # total effect of treatment
        }

    # Point estimate
    point = _estimate_mediation(df_clean)
    if point is None:
        raise RuntimeError("Mediation estimation failed on full sample")

    # Bootstrap CIs
    n = len(df_clean)
    boot_results = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_data = df_clean.iloc[idx].reset_index(drop=True)
        boot_est = _estimate_mediation(boot_data)
        if boot_est is not None:
            boot_results.append(boot_est)

    if len(boot_results) < n_bootstrap * 0.5:
        logger.warning(
            "Only %d/%d bootstrap samples converged",
            len(boot_results), n_bootstrap,
        )

    def _ci(key):
        vals = [b[key] for b in boot_results if key in b]
        if len(vals) < 10:
            return (np.nan, np.nan)
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    return {
        # Mediation decomposition
        "acme": point["acme"],
        "acme_ci": _ci("acme"),
        "ade_coef": point["ade_coef"],
        "ade_ci": _ci("ade_coef"),
        "total_coef": point["total_coef"],
        "total_ci": _ci("total_coef"),
        "prop_mediated": point["prop_mediated"],
        "prop_mediated_ci": _ci("prop_mediated"),
        # Path coefficients (for reporting)
        "a_path": point["a_path"],          # T -> M
        "a_path_ci": _ci("a_path"),
        "b_path": point["b_path"],          # M -> Y | T
        "b_path_ci": _ci("b_path"),
        "c_prime": point["c_prime"],        # T -> Y | M (direct)
        "c_prime_ci": _ci("c_prime"),
        "c_total": point["c_total"],        # T -> Y (total)
        "c_total_ci": _ci("c_total"),
        # Metadata
        "n_obs": len(df_clean),
        "n_bootstrap": n_bootstrap,
        "n_converged": len(boot_results),
        "treatment": treatment_col,
        "mediator": mediator_col,
        "outcome": outcome_col,
    }
