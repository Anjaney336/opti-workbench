"""
Analysis utilities for portfolio optimization experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import monte_carlo_evaluate
from src.utils import compute_return_and_variance


def evaluate_portfolio(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_scenarios: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Evaluate a portfolio analytically and via Monte Carlo simulation.
    """
    analytical_return, analytical_variance = compute_return_and_variance(x, mu, Sigma)
    mc_stats = monte_carlo_evaluate(x, mu, Sigma, n_scenarios=n_scenarios, seed=seed)

    return {
        "analytical_return": float(analytical_return),
        "analytical_variance": float(analytical_variance),
        "mc_mean_return": float(mc_stats["mean_return"]),
        "mc_variance": float(mc_stats["variance"]),
        "mc_std_return": float(mc_stats["std_return"]),
        "mc_sharpe_ratio": float(mc_stats["sharpe_ratio"]),
        "mc_var_95": float(mc_stats["var_95"]),
        "mc_cvar_95": float(mc_stats["cvar_95"]),
        "mc_loss_var_95": float(mc_stats["loss_var_95"]),
        "mc_loss_cvar_95": float(mc_stats["loss_cvar_95"]),
        "diversification_index": float(mc_stats["diversification_index"]),
    }


def apply_mu_noise(mu: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """
    Apply multiplicative noise to expected returns:

        mu_noisy = mu * (1 + epsilon), epsilon ~ U(-noise_level, noise_level)
    """
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-noise_level, noise_level, size=mu.shape)
    return mu * (1.0 + noise)


def build_hybrid_convergence(
    qp_objective: float,
    pso_history: dict[str, list[float]],
    ga_history: dict[str, list[float]],
) -> dict[str, list[float]]:
    """
    Merge QP, PSO, and GA histories into a single best-so-far hybrid curve.
    """
    objective_curve = [float(qp_objective)]
    diversity_curve = [np.nan]
    running_best = float(qp_objective)

    pso_best = pso_history.get("best_fitness", [])
    pso_diversity = pso_history.get("population_diversity", [])
    ga_best = ga_history.get("best_fitness", [])
    ga_diversity = ga_history.get("population_diversity", [])

    for idx, value in enumerate(pso_best):
        running_best = max(running_best, float(value))
        objective_curve.append(running_best)
        diversity_curve.append(float(pso_diversity[idx]) if idx < len(pso_diversity) else np.nan)

    for idx, value in enumerate(ga_best):
        running_best = max(running_best, float(value))
        objective_curve.append(running_best)
        diversity_curve.append(float(ga_diversity[idx]) if idx < len(ga_diversity) else np.nan)

    return {
        "objective": objective_curve,
        "population_diversity": diversity_curve,
    }


def summarize_baseline_results(results_df: pd.DataFrame, baseline_lambda: float) -> pd.DataFrame:
    """
    Compact baseline comparison table for reporting.
    """
    baseline = results_df[
        (results_df["analysis_type"] == "baseline")
        & np.isclose(results_df["lambda"], baseline_lambda)
    ].copy()

    columns = [
        "method",
        "objective",
        "mc_mean_return",
        "mc_std_return",
        "mc_cvar_95",
        "runtime_sec",
        "feasible",
    ]
    return baseline[columns].sort_values("objective", ascending=False).reset_index(drop=True)


def summarize_robustness(results_df: pd.DataFrame, baseline_lambda: float) -> pd.DataFrame:
    """
    Mean performance degradation under noisy mu perturbations.
    """
    baseline = results_df[
        (results_df["analysis_type"] == "baseline")
        & np.isclose(results_df["lambda"], baseline_lambda)
    ][["method", "objective", "mc_mean_return", "mc_std_return"]].rename(
        columns={
            "objective": "baseline_objective",
            "mc_mean_return": "baseline_return",
            "mc_std_return": "baseline_risk",
        }
    )

    robustness = results_df[results_df["analysis_type"] == "robustness"].copy()
    merged = robustness.merge(baseline, on="method", how="left")
    merged["objective_degradation"] = merged["baseline_objective"] - merged["objective"]
    merged["return_degradation"] = merged["baseline_return"] - merged["mc_mean_return"]
    merged["risk_increase"] = merged["mc_std_return"] - merged["baseline_risk"]

    summary = (
        merged.groupby(["method", "noise_level"], observed=True)[
            ["objective_degradation", "return_degradation", "risk_increase"]
        ]
        .mean()
        .reset_index()
    )
    return summary.sort_values(["noise_level", "method"]).reset_index(drop=True)
