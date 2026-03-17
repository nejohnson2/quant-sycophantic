"""Analysis modules for sycophancy research."""

from .causal import (
    compute_ate,
    compute_mediation_decomposition,
    compute_propensity_scores,
    run_outcome_regression,
    run_winner_regression,
)
from .causal_framework import CausalDAG, build_sycophancy_dag
from .descriptive import (
    compute_correlations,
    compute_sycophancy_by_model,
    compute_summary_statistics,
    compute_win_rate_by_sycophancy,
)
from .diagnostics import compute_e_value, compute_e_values_table, rosenbaum_sensitivity
from .reliability import compute_irr, compute_weighted_kappa

__all__ = [
    # Descriptive
    "compute_summary_statistics",
    "compute_sycophancy_by_model",
    "compute_correlations",
    "compute_win_rate_by_sycophancy",
    # Causal
    "compute_propensity_scores",
    "compute_ate",
    "run_outcome_regression",
    "run_winner_regression",
    "compute_mediation_decomposition",
    # Causal framework
    "CausalDAG",
    "build_sycophancy_dag",
    # Diagnostics / Sensitivity
    "compute_e_value",
    "compute_e_values_table",
    "rosenbaum_sensitivity",
    # Reliability
    "compute_irr",
    "compute_weighted_kappa",
]
