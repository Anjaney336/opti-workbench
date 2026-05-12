"""
Experiment runner for the portfolio optimization system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from analysis import apply_mu_noise, build_hybrid_convergence, evaluate_portfolio
from src.data import generate_portfolio_data
from src.ga import optimize_with_ga
from src.pso import optimize_with_pso
from src.qp import solve_quadratic_program
from src.utils import check_feasibility, project_constraints


METHOD_LABELS = {
    "QP": "QP",
    "PSO": "PSO",
    "GA": "GA",
    "Hybrid": "Hybrid",
}


@dataclass
class ExperimentConfig:
    n_assets: int = 30
    lambda_values: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    baseline_lambda: float = 0.5
    expected_return_range: tuple[float, float] = (0.05, 0.15)
    budget: float = 1.0
    max_allocation: float = 0.3
    n_mc_scenarios: int = 2000
    pso_particles: int = 24
    pso_iterations: int = 80
    ga_population: int = 32
    ga_generations: int = 80
    robustness_noise_levels: tuple[float, ...] = (0.05, 0.10)
    robustness_trials: int = 3
    seed: int = 42
    output_dir: str = "results"
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def utility_objective(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, lambda_param: float) -> float:
    return float(np.dot(x, mu) - lambda_param * (x @ Sigma @ x))


def make_objective_fn(mu: np.ndarray, Sigma: np.ndarray, lambda_param: float):
    def objective(x: np.ndarray) -> float:
        return utility_objective(x, mu, Sigma, lambda_param)

    return objective


def make_constraint_projection(data: dict):
    def constraint_proj(x: np.ndarray) -> np.ndarray:
        return project_constraints(x, data["c"], data["B"], data["alpha"])

    return constraint_proj


def evaluate_solution(
    method: str,
    x: np.ndarray,
    objective: float,
    runtime_sec: float,
    lambda_param: float,
    data: dict,
    config: ExperimentConfig,
    analysis_type: str,
    noise_level: float | None = None,
    trial: int = 0,
) -> dict[str, Any]:
    portfolio_metrics = evaluate_portfolio(
        x=x,
        mu=data["mu"],
        Sigma=data["Sigma"],
        n_scenarios=config.n_mc_scenarios,
        seed=config.seed + trial,
    )
    feasibility = check_feasibility(x, data["c"], data["B"], data["alpha"])

    return {
        "analysis_type": analysis_type,
        "method": method,
        "lambda": lambda_param,
        "noise_level": noise_level if noise_level is not None else 0.0,
        "trial": trial,
        "objective": float(objective),
        "runtime_sec": float(runtime_sec),
        "feasible": bool(feasibility["feasible"]),
        "n_assets": int(data["n_assets"]),
        **portfolio_metrics,
    }


def run_single_experiment(
    data: dict,
    lambda_param: float,
    config: ExperimentConfig,
    seed: int,
    analysis_type: str,
    trial: int = 0,
    noise_level: float | None = None,
    capture_convergence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run QP, PSO, GA, and Hybrid on a fixed portfolio instance.
    """
    n_assets = data["n_assets"]
    objective_fn = make_objective_fn(data["mu"], data["Sigma"], lambda_param)
    constraint_proj_fn = make_constraint_projection(data)

    rows: list[dict[str, Any]] = []
    convergence: dict[str, Any] = {}

    start = perf_counter()
    qp_result = solve_quadratic_program(
        mu=data["mu"],
        Sigma=data["Sigma"],
        c=data["c"],
        B=data["B"],
        alpha=data["alpha"],
        lambda_param=lambda_param,
        verbose=False,
    )
    qp_runtime = perf_counter() - start
    x_qp = project_constraints(
        qp_result["x"] if qp_result["x"] is not None else np.ones(n_assets) / n_assets,
        data["c"],
        data["B"],
        data["alpha"],
    )
    qp_objective = utility_objective(x_qp, data["mu"], data["Sigma"], lambda_param)
    rows.append(
        evaluate_solution(
            method="QP",
            x=x_qp,
            objective=qp_objective,
            runtime_sec=qp_runtime,
            lambda_param=lambda_param,
            data=data,
            config=config,
            analysis_type=analysis_type,
            noise_level=noise_level,
            trial=trial,
        )
    )

    start = perf_counter()
    pso_result = optimize_with_pso(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        x0=None,
        n_dimensions=n_assets,
        n_particles=config.pso_particles,
        n_iterations=config.pso_iterations,
        verbose=False,
        seed=seed,
    )
    pso_runtime = perf_counter() - start
    x_pso = np.asarray(pso_result["x"])
    rows.append(
        evaluate_solution(
            method="PSO",
            x=x_pso,
            objective=float(pso_result["fitness"]),
            runtime_sec=pso_runtime,
            lambda_param=lambda_param,
            data=data,
            config=config,
            analysis_type=analysis_type,
            noise_level=noise_level,
            trial=trial,
        )
    )

    start = perf_counter()
    ga_result = optimize_with_ga(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        x0=None,
        n_dimensions=n_assets,
        population_size=config.ga_population,
        n_generations=config.ga_generations,
        verbose=False,
        seed=seed,
    )
    ga_runtime = perf_counter() - start
    x_ga = np.asarray(ga_result["x"])
    rows.append(
        evaluate_solution(
            method="GA",
            x=x_ga,
            objective=float(ga_result["fitness"]),
            runtime_sec=ga_runtime,
            lambda_param=lambda_param,
            data=data,
            config=config,
            analysis_type=analysis_type,
            noise_level=noise_level,
            trial=trial,
        )
    )

    start = perf_counter()
    hybrid_pso = optimize_with_pso(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        x0=x_qp,
        n_dimensions=n_assets,
        n_particles=config.pso_particles,
        n_iterations=config.pso_iterations,
        verbose=False,
        seed=seed,
    )
    hybrid_ga = optimize_with_ga(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        x0=np.asarray(hybrid_pso["x"]),
        n_dimensions=n_assets,
        population_size=config.ga_population,
        n_generations=config.ga_generations,
        verbose=False,
        seed=seed,
    )
    hybrid_runtime = perf_counter() - start

    candidate_allocations = {
        "QP": x_qp,
        "PSO": np.asarray(hybrid_pso["x"]),
        "GA": np.asarray(hybrid_ga["x"]),
    }
    candidate_objectives = {
        name: utility_objective(x, data["mu"], data["Sigma"], lambda_param)
        for name, x in candidate_allocations.items()
    }
    hybrid_stage = max(candidate_objectives, key=candidate_objectives.get)
    x_hybrid = candidate_allocations[hybrid_stage]
    hybrid_objective = candidate_objectives[hybrid_stage]

    rows.append(
        evaluate_solution(
            method="Hybrid",
            x=x_hybrid,
            objective=hybrid_objective,
            runtime_sec=hybrid_runtime + qp_runtime,
            lambda_param=lambda_param,
            data=data,
            config=config,
            analysis_type=analysis_type,
            noise_level=noise_level,
            trial=trial,
        )
    )

    if capture_convergence:
        convergence = {
            "PSO": {
                "objective": list(pso_result["history"]["best_fitness"]),
                "population_diversity": list(pso_result["history"].get("population_diversity", [])),
            },
            "GA": {
                "objective": list(ga_result["history"]["best_fitness"]),
                "population_diversity": list(ga_result["history"].get("population_diversity", [])),
            },
            "Hybrid": build_hybrid_convergence(
                qp_objective=qp_objective,
                pso_history=pso_result["history"],
                ga_history=hybrid_ga["history"],
            ),
            "meta": {
                "lambda": lambda_param,
                "analysis_type": analysis_type,
                "noise_level": noise_level if noise_level is not None else 0.0,
                "trial": trial,
            },
        }

    return rows, convergence


def run_full_experiments(config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any], dict]:
    """
    Run baseline, sensitivity, and robustness experiments.
    """
    base_data = generate_portfolio_data(
        n_assets=config.n_assets,
        expected_return_range=config.expected_return_range,
        budget=config.budget,
        max_allocation=config.max_allocation,
        seed=config.seed,
    )

    all_rows: list[dict[str, Any]] = []
    convergence_data: dict[str, Any] = {}

    baseline_rows, baseline_convergence = run_single_experiment(
        data=base_data,
        lambda_param=config.baseline_lambda,
        config=config,
        seed=config.seed,
        analysis_type="baseline",
        trial=0,
        capture_convergence=True,
    )
    all_rows.extend(baseline_rows)
    convergence_data["baseline"] = baseline_convergence

    for lambda_param in config.lambda_values:
        sensitivity_rows, _ = run_single_experiment(
            data=base_data,
            lambda_param=lambda_param,
            config=config,
            seed=config.seed,
            analysis_type="sensitivity",
            trial=0,
            capture_convergence=False,
        )
        all_rows.extend(sensitivity_rows)

    for noise_level in config.robustness_noise_levels:
        for trial in range(config.robustness_trials):
            noisy_data = dict(base_data)
            noisy_data["mu"] = apply_mu_noise(base_data["mu"], noise_level, seed=config.seed + 100 * (trial + 1))
            robustness_rows, _ = run_single_experiment(
                data=noisy_data,
                lambda_param=config.baseline_lambda,
                config=config,
                seed=config.seed + trial + 1,
                analysis_type="robustness",
                trial=trial,
                noise_level=noise_level,
                capture_convergence=False,
            )
            all_rows.extend(robustness_rows)

    results_df = pd.DataFrame(all_rows)
    results_df["method"] = pd.Categorical(
        results_df["method"],
        categories=["QP", "PSO", "GA", "Hybrid"],
        ordered=True,
    )
    return results_df, convergence_data, base_data
