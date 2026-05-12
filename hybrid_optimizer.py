"""
Hybrid GA + PSO + QP benchmark utilities for convex quadratic programs.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ga import GAConfig, solve_with_ga
from pso import PSOConfig, solve_with_pso
from qp_solver import (
    QuadraticProgram,
    constraint_violation_norm,
    is_feasible,
    objective_value,
    solve_qp_active_set,
    unconstrained_optimum,
)


METHOD_ORDER = ["ga", "pso", "qp", "ga_pso", "pso_qp", "ga_pso_qp"]
METHOD_LABELS = {
    "ga": "GA",
    "pso": "PSO",
    "qp": "QP",
    "ga_pso": "GA + PSO",
    "pso_qp": "PSO + QP",
    "ga_pso_qp": "GA + PSO + QP",
}


def build_ga_seeded_swarm(
    ranked_population: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    """
    Build a PSO initialization cloud from the best GA individuals.
    """
    ranked_population = np.asarray(ranked_population, dtype=float)
    rng = np.random.default_rng(seed)
    elite_count = min(len(ranked_population), max(4, n_particles // 4))
    elites = ranked_population[:elite_count]

    seeded = np.empty((n_particles, ranked_population.shape[1]), dtype=float)
    seeded[:elite_count] = elites

    spread = 0.10 * (upper_bound - lower_bound)
    for idx in range(elite_count, n_particles):
        parent = elites[idx % elite_count]
        child = parent + rng.normal(loc=0.0, scale=spread, size=ranked_population.shape[1])
        seeded[idx] = np.clip(child, lower_bound, upper_bound)

    return seeded


def generate_benchmark_problem(
    dimension: int,
    seed: int,
    n_inequalities: int = 3,
    n_equalities: int = 1,
    search_radius: float = 3.0,
) -> tuple[QuadraticProgram, np.ndarray, np.ndarray]:
    """
    Generate a strongly convex quadratic program whose analytical unconstrained
    optimum x* = -Q^{-1} c is also feasible for all constraints.
    """
    rng = np.random.default_rng(seed)

    basis = rng.normal(size=(dimension, dimension))
    Q = basis.T @ basis + (0.5 + dimension / 20.0) * np.eye(dimension)
    c = rng.normal(size=dimension)
    x_true = -np.linalg.solve(Q, c)

    if n_inequalities > 0:
        A = rng.normal(size=(n_inequalities, dimension))
        slacks = 0.5 + rng.random(n_inequalities)
        b = A @ x_true + slacks
    else:
        A = np.zeros((0, dimension))
        b = np.zeros(0)

    if n_equalities > 0:
        E = rng.normal(size=(n_equalities, dimension))
        d = E @ x_true
    else:
        E = np.zeros((0, dimension))
        d = np.zeros(0)

    lower_bound = x_true - search_radius
    upper_bound = x_true + search_radius

    problem = QuadraticProgram(
        Q=Q,
        c=c,
        A=A,
        b=b,
        E=E,
        d=d,
        x_true=x_true,
        name=f"benchmark_d{dimension}_s{seed}",
    )
    return problem, lower_bound, upper_bound


def summarize_solution(
    problem: QuadraticProgram,
    x: np.ndarray,
    elapsed_seconds: float,
    iterations: int,
    candidates_checked: int = -1,
) -> dict[str, float | bool | int | np.ndarray]:
    x_true = problem.x_true if problem.x_true is not None else unconstrained_optimum(problem)
    true_objective = objective_value(problem, x_true)
    objective = objective_value(problem, x)
    return {
        "x": np.asarray(x, dtype=float),
        "objective": float(objective),
        "position_error": float(np.linalg.norm(x - x_true)),
        "objective_gap": float(objective - true_objective),
        "constraint_violation": float(constraint_violation_norm(problem, x)),
        "feasible": bool(is_feasible(problem, x)),
        "elapsed_seconds": float(elapsed_seconds),
        "iterations": int(iterations),
        "candidates_checked": int(candidates_checked),
    }


def estimate_convergence_rate(sequence: list[float] | np.ndarray, optimum: float) -> float:
    values = np.asarray(sequence, dtype=float)
    if values.size < 4:
        return np.nan

    gaps = np.maximum(values - optimum, 1e-14)
    start = values.size // 2
    x_axis = np.arange(start, values.size)
    y_axis = np.log(gaps[start:])
    slope, _ = np.polyfit(x_axis, y_axis, deg=1)
    return float(np.exp(slope))


def _copy_history(history: dict[str, object]) -> dict[str, list[float]]:
    copied: dict[str, list[float]] = {}
    for key, values in history.items():
        copied[key] = [float(v) if np.isscalar(v) else v for v in values]
    return copied


def _combine_histories(
    histories: list[dict[str, object]],
    final_objective: float,
    final_error: float,
) -> dict[str, list[float]]:
    combined_objective: list[float] = []
    combined_error: list[float] = []
    combined_diversity: list[float] = []

    for history in histories:
        if "best_objective_value" in history:
            combined_objective.extend(float(v) for v in history["best_objective_value"])
        if "position_error" in history:
            combined_error.extend(float(v) for v in history["position_error"])
        if "population_diversity" in history:
            combined_diversity.extend(float(v) for v in history["population_diversity"])

    combined_objective.append(float(final_objective))
    combined_error.append(float(final_error))

    return {
        "best_objective_value": combined_objective,
        "position_error": combined_error,
        "population_diversity": combined_diversity,
    }


def _augment_summary(
    summary: dict[str, object],
    history: dict[str, object],
    true_objective: float,
    **extras: object,
) -> dict[str, object]:
    return {
        **summary,
        "history": history,
        "convergence_rate": estimate_convergence_rate(history["best_objective_value"], true_objective),
        **extras,
    }


def run_benchmark_experiment(
    problem: QuadraticProgram,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    ga_config: GAConfig,
    pso_config: PSOConfig,
    verbose: bool = False,
) -> dict[str, object]:
    """
    Compare:
        1. GA
        2. PSO
        3. QP
        4. GA + PSO
        5. PSO + QP
        6. GA + PSO + QP
    """
    x_true = problem.x_true if problem.x_true is not None else unconstrained_optimum(problem)
    true_objective = float(objective_value(problem, x_true))

    start = perf_counter()
    ga_result = solve_with_ga(problem, lower_bound, upper_bound, ga_config, verbose=verbose)
    ga_elapsed = perf_counter() - start

    start = perf_counter()
    pso_result = solve_with_pso(problem, lower_bound, upper_bound, pso_config, verbose=verbose)
    pso_elapsed = perf_counter() - start

    start = perf_counter()
    ga_seeded_swarm = build_ga_seeded_swarm(
        ranked_population=np.asarray(ga_result["ranked_population"]),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_particles=pso_config.n_particles,
        seed=pso_config.seed + 101,
    )
    ga_pso_result = solve_with_pso(
        problem,
        lower_bound,
        upper_bound,
        pso_config,
        initial_positions=ga_seeded_swarm,
        verbose=verbose,
    )
    ga_pso_elapsed = perf_counter() - start

    start = perf_counter()
    qp_result = solve_qp_active_set(problem, x0=None)
    qp_elapsed = perf_counter() - start

    if qp_result["x"] is None:
        raise RuntimeError("KKT solver failed on benchmark problem.")

    start = perf_counter()
    pso_qp_result = solve_qp_active_set(problem, x0=np.asarray(pso_result["x"]))
    pso_qp_elapsed = perf_counter() - start

    start = perf_counter()
    ga_pso_qp_result = solve_qp_active_set(problem, x0=np.asarray(ga_pso_result["x"]))
    ga_pso_qp_elapsed = perf_counter() - start

    if pso_qp_result["x"] is None or ga_pso_qp_result["x"] is None:
        raise RuntimeError("Hybrid KKT refinement failed on benchmark problem.")

    ga_summary = summarize_solution(
        problem,
        np.asarray(ga_result["x"]),
        ga_elapsed,
        iterations=ga_config.generations,
    )
    pso_summary = summarize_solution(
        problem,
        np.asarray(pso_result["x"]),
        pso_elapsed,
        iterations=pso_config.iterations,
    )
    qp_summary = summarize_solution(
        problem,
        np.asarray(qp_result["x"]),
        qp_elapsed,
        iterations=int(qp_result["candidates_checked"]),
        candidates_checked=int(qp_result["candidates_checked"]),
    )
    ga_pso_summary = summarize_solution(
        problem,
        np.asarray(ga_pso_result["x"]),
        ga_elapsed + ga_pso_elapsed,
        iterations=ga_config.generations + pso_config.iterations,
    )
    pso_qp_summary = summarize_solution(
        problem,
        np.asarray(pso_qp_result["x"]),
        pso_elapsed + pso_qp_elapsed,
        iterations=pso_config.iterations + 1,
        candidates_checked=int(pso_qp_result["candidates_checked"]),
    )
    ga_pso_qp_summary = summarize_solution(
        problem,
        np.asarray(ga_pso_qp_result["x"]),
        ga_elapsed + ga_pso_elapsed + ga_pso_qp_elapsed,
        iterations=ga_config.generations + pso_config.iterations + 1,
        candidates_checked=int(ga_pso_qp_result["candidates_checked"]),
    )

    ga_history = _copy_history(ga_result["history"])
    pso_history = _copy_history(pso_result["history"])
    ga_pso_history = _combine_histories(
        [_copy_history(ga_result["history"]), _copy_history(ga_pso_result["history"])],
        final_objective=float(ga_pso_summary["objective"]),
        final_error=float(ga_pso_summary["position_error"]),
    )
    pso_qp_history = _combine_histories(
        [_copy_history(pso_result["history"])],
        final_objective=float(pso_qp_summary["objective"]),
        final_error=float(pso_qp_summary["position_error"]),
    )
    ga_pso_qp_history = _combine_histories(
        [_copy_history(ga_result["history"]), _copy_history(ga_pso_result["history"])],
        final_objective=float(ga_pso_qp_summary["objective"]),
        final_error=float(ga_pso_qp_summary["position_error"]),
    )

    methods = {
        "ga": _augment_summary(
            ga_summary,
            ga_history,
            true_objective,
            penalized_objective=float(ga_result["penalized_objective"]),
            penalty=float(ga_result["penalty"]),
            feasible_selection=str(ga_result["selected_from"]),
            initial_diversity=float(ga_result["initial_diversity"]),
            final_diversity=float(ga_result["final_diversity"]),
        ),
        "pso": _augment_summary(
            pso_summary,
            pso_history,
            true_objective,
            penalized_objective=float(pso_result["penalized_objective"]),
            penalty=float(pso_result["penalty"]),
            feasible_selection=str(pso_result["selected_from"]),
            initial_diversity=float(pso_result["initial_diversity"]),
            final_diversity=float(pso_result["final_diversity"]),
        ),
        "qp": {
            **qp_summary,
            "active_set": tuple(qp_result["active_set"]),
            "kkt_residual": float(qp_result["kkt_residual"]),
        },
        "ga_pso": _augment_summary(
            ga_pso_summary,
            ga_pso_history,
            true_objective,
            penalized_objective=float(ga_pso_result["penalized_objective"]),
            penalty=float(ga_pso_result["penalty"]),
            feasible_selection=str(ga_pso_result["selected_from"]),
            initial_diversity=float(ga_pso_result["initial_diversity"]),
            final_diversity=float(ga_pso_result["final_diversity"]),
            ga_seed_diversity=float(ga_result["final_diversity"]),
            warm_start_distance=float(np.linalg.norm(np.asarray(ga_result["x"]) - np.asarray(ga_pso_result["x"]))),
        ),
        "pso_qp": _augment_summary(
            pso_qp_summary,
            pso_qp_history,
            true_objective,
            active_set=tuple(pso_qp_result["active_set"]),
            kkt_residual=float(pso_qp_result["kkt_residual"]),
            warm_start_distance=float(np.linalg.norm(np.asarray(pso_result["x"]) - np.asarray(pso_qp_result["x"]))),
        ),
        "ga_pso_qp": _augment_summary(
            ga_pso_qp_summary,
            ga_pso_qp_history,
            true_objective,
            active_set=tuple(ga_pso_qp_result["active_set"]),
            kkt_residual=float(ga_pso_qp_result["kkt_residual"]),
            warm_start_distance=float(np.linalg.norm(np.asarray(ga_pso_result["x"]) - np.asarray(ga_pso_qp_result["x"]))),
            ga_seed_diversity=float(ga_result["final_diversity"]),
        ),
    }

    validation = {
        "qp_matches_exact": bool(np.linalg.norm(np.asarray(qp_result["x"]) - x_true) < 1e-8),
        "pso_qp_matches_exact": bool(np.linalg.norm(np.asarray(pso_qp_result["x"]) - x_true) < 1e-8),
        "ga_pso_qp_matches_exact": bool(np.linalg.norm(np.asarray(ga_pso_qp_result["x"]) - x_true) < 1e-8),
        "ga_pso_improves_ga": bool(float(methods["ga_pso"]["objective_gap"]) <= float(methods["ga"]["objective_gap"]) + 1e-10),
        "ga_pso_improves_pso": bool(float(methods["ga_pso"]["objective_gap"]) <= float(methods["pso"]["objective_gap"]) + 1e-10),
        "pso_qp_improves_pso": bool(float(methods["pso_qp"]["objective_gap"]) <= float(methods["pso"]["objective_gap"]) + 1e-10),
        "ga_pso_qp_improves_ga_pso": bool(float(methods["ga_pso_qp"]["objective_gap"]) <= float(methods["ga_pso"]["objective_gap"]) + 1e-10),
        "ga_population_has_diversity": bool(float(ga_result["final_diversity"]) > 0.0),
        "ga_seeded_swarm_has_diversity": bool(float(ga_pso_result["initial_diversity"]) > 0.0),
    }

    return {
        "problem": problem,
        "x_true": x_true,
        "true_objective": true_objective,
        "methods": methods,
        "validation": validation,
    }


def run_stress_suite(
    dimensions: list[int],
    trials_per_dimension: int,
    ga_factory,
    pso_factory,
    base_seed: int = 100,
) -> list[dict[str, object]]:
    """
    Stress-test all six methods across dimensions and seeds.
    """
    rows: list[dict[str, object]] = []

    for dimension in dimensions:
        for trial in range(trials_per_dimension):
            seed = base_seed + 31 * dimension + trial
            problem, lower_bound, upper_bound = generate_benchmark_problem(
                dimension=dimension,
                seed=seed,
                n_inequalities=3,
                n_equalities=1,
                search_radius=3.5,
            )
            result = run_benchmark_experiment(
                problem,
                lower_bound,
                upper_bound,
                ga_factory(dimension, seed),
                pso_factory(dimension, seed),
                verbose=False,
            )

            row: dict[str, object] = {"dimension": dimension, "trial": trial, "seed": seed}
            for method in METHOD_ORDER:
                summary = result["methods"][method]
                row[f"{method}_error"] = float(summary["position_error"])
                row[f"{method}_gap"] = float(summary["objective_gap"])
                row[f"{method}_time"] = float(summary["elapsed_seconds"])
            rows.append(row)

    return rows
