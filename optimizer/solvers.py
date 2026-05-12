"""
Solver implementations for QP, PSO, GA, and Hybrid portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from optimizer.live_trace import candidate_snapshot, stage_completed, stage_started, trace_point
from src.utils import check_feasibility, project_constraints


@dataclass(frozen=True)
class PortfolioProblem:
    mu: np.ndarray
    cov: np.ndarray
    liquidity: np.ndarray
    cost: np.ndarray
    budget: float
    max_allocation: float
    lambda_value: float
    liquidity_weight: float
    cost_sensitivity: float
    tail_risk_weight: float
    diversification_weight: float
    concentration_penalty: float
    regime_aversion: float
    tracking_penalty: float
    covariance_shrinkage: float
    cvar_alpha: float
    active_assets_target: int | None
    seed: int
    asset_names: list[str]
    regime_names: list[str]
    regime_probabilities: np.ndarray
    regime_mu: np.ndarray
    regime_covariances: np.ndarray
    scenario_returns: np.ndarray
    anchor_weights: np.ndarray
    exact_cvar_seed: bool = True
    exact_discrete_selection: bool = True
    planning_horizon: int = 3
    backtest_lookback: int = 12
    historical_returns: np.ndarray | None = None
    macro_signal: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": np.asarray(self.mu, dtype=float),
            "cov": np.asarray(self.cov, dtype=float),
            "liquidity": np.asarray(self.liquidity, dtype=float),
            "cost": np.asarray(self.cost, dtype=float),
            "budget": float(self.budget),
            "max_allocation": float(self.max_allocation),
            "lambda_value": float(self.lambda_value),
            "liquidity_weight": float(self.liquidity_weight),
            "cost_sensitivity": float(self.cost_sensitivity),
            "tail_risk_weight": float(self.tail_risk_weight),
            "diversification_weight": float(self.diversification_weight),
            "concentration_penalty": float(self.concentration_penalty),
            "regime_aversion": float(self.regime_aversion),
            "tracking_penalty": float(self.tracking_penalty),
            "covariance_shrinkage": float(self.covariance_shrinkage),
            "cvar_alpha": float(self.cvar_alpha),
            "active_assets_target": None if self.active_assets_target is None else int(self.active_assets_target),
            "seed": int(self.seed),
            "asset_names": list(self.asset_names),
            "regime_names": list(self.regime_names),
            "regime_probabilities": np.asarray(self.regime_probabilities, dtype=float),
            "regime_mu": np.asarray(self.regime_mu, dtype=float),
            "regime_covariances": np.asarray(self.regime_covariances, dtype=float),
            "scenario_returns": np.asarray(self.scenario_returns, dtype=float),
            "anchor_weights": np.asarray(self.anchor_weights, dtype=float),
            "exact_cvar_seed": bool(self.exact_cvar_seed),
            "exact_discrete_selection": bool(self.exact_discrete_selection),
            "planning_horizon": int(self.planning_horizon),
            "backtest_lookback": int(self.backtest_lookback),
            "historical_returns": None if self.historical_returns is None else np.asarray(self.historical_returns, dtype=float),
            "macro_signal": None if self.macro_signal is None else np.asarray(self.macro_signal, dtype=float),
        }


SolverEventCallback = Callable[[dict[str, Any]], None]


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _emit(event_callback: SolverEventCallback | None, event: dict[str, Any]) -> None:
    if event_callback is not None:
        event_callback(event)


def _active_count(weights: np.ndarray) -> int:
    return int(np.sum(np.asarray(weights, dtype=float) > 1e-6))


def _minimum_active_assets(max_allocation: float) -> int:
    allocation_cap = float(np.clip(max_allocation, 1e-6, 1.0))
    return max(1, int(np.ceil(1.0 / allocation_cap - 1e-12)))


def _effective_active_target(problem: PortfolioProblem, n_assets: int) -> int | None:
    if problem.active_assets_target is None:
        return None
    minimum_assets = _minimum_active_assets(problem.max_allocation)
    requested = max(1, int(problem.active_assets_target))
    return min(n_assets, max(requested, minimum_assets))


def project_simplex(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def apply_active_asset_target(weights: np.ndarray, problem: PortfolioProblem) -> np.ndarray:
    projected = project_constraints(
        np.asarray(weights, dtype=float),
        np.asarray(problem.cost, dtype=float),
        float(problem.budget),
        float(problem.max_allocation),
    )
    target = _effective_active_target(problem, len(projected))
    if target is None or target >= len(projected):
        return projected
    keep = np.argsort(projected)[-target:]
    sparse = np.zeros_like(projected)
    sparse[keep] = project_constraints(
        projected[keep],
        np.asarray(problem.cost, dtype=float)[keep],
        float(problem.budget),
        float(problem.max_allocation),
    )
    return sparse


def _affordability_scale(problem: PortfolioProblem) -> float:
    return 1.0 + 3.0 / max(problem.budget, 0.5)


def _shrunk_covariance(problem: PortfolioProblem) -> np.ndarray:
    cov = np.asarray(problem.cov, dtype=float)
    shrinkage = float(np.clip(problem.covariance_shrinkage, 0.0, 0.95))
    diagonal = np.diag(np.diag(cov))
    shrunk = (1.0 - shrinkage) * cov + shrinkage * diagonal
    return 0.5 * (shrunk + shrunk.T)


def _safe_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(-np.sum(weights * np.log(weights + 1e-12)))


def _weighted_variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    probabilities = _normalize(np.asarray(probabilities, dtype=float))
    mean_value = float(np.dot(probabilities, values))
    return float(np.dot(probabilities, (values - mean_value) ** 2))


def _cvar_from_returns(portfolio_returns: np.ndarray, alpha: float) -> tuple[float, float]:
    returns = np.asarray(portfolio_returns, dtype=float)
    losses = -returns
    var_level = float(np.quantile(losses, alpha))
    tail = losses[losses >= var_level - 1e-12]
    cvar = float(np.mean(tail)) if tail.size else var_level
    return var_level, cvar


def _jsonify_breakdown(breakdown: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in breakdown.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.astype(float).tolist()
        elif isinstance(value, (np.floating, np.integer)):
            payload[key] = float(value)
        elif isinstance(value, dict):
            payload[key] = _jsonify_breakdown(value)
        elif isinstance(value, list):
            payload[key] = [float(item) if isinstance(item, (np.floating, np.integer)) else item for item in value]
        else:
            payload[key] = value
    return payload


def objective_components(weights: np.ndarray, problem: PortfolioProblem, model: str = "QP") -> dict[str, Any]:
    weights = apply_active_asset_target(weights, problem)
    cov = _shrunk_covariance(problem)
    risk = float(weights.T @ cov @ weights)
    expected_return = float(np.dot(problem.mu, weights))
    liquidity = float(np.dot(problem.liquidity, weights))
    cost_term = float(np.dot(problem.cost, weights)) * _affordability_scale(problem)
    concentration = float(np.sum(weights**2))
    entropy_bonus = _safe_entropy(weights)
    active_target = float(_effective_active_target(problem, len(weights)) or len(weights))
    target_concentration = 1.0 / max(active_target, 1.0)
    diversification_gap = float((concentration - target_concentration) ** 2)
    tracking_error = float(np.sum((weights - _normalize(problem.anchor_weights)) ** 2))

    scenario_portfolio_returns = np.asarray(problem.scenario_returns, dtype=float) @ weights
    loss_var_95, loss_cvar_95 = _cvar_from_returns(scenario_portfolio_returns, float(problem.cvar_alpha))

    regime_probabilities = _normalize(problem.regime_probabilities)
    regime_expected_returns = np.asarray(problem.regime_mu, dtype=float) @ weights
    regime_risks = np.asarray(
        [float(weights.T @ regime_cov @ weights) for regime_cov in np.asarray(problem.regime_covariances, dtype=float)],
        dtype=float,
    )
    regime_dispersion = _weighted_variance(regime_expected_returns, regime_probabilities)
    regime_stress = float(np.dot(regime_probabilities, regime_risks))

    base_terms = {
        "risk_term": risk,
        "return_term": -float(problem.lambda_value) * expected_return,
        "cost_term": float(problem.cost_sensitivity) * cost_term,
        "liquidity_term": -float(problem.liquidity_weight) * liquidity,
        "tail_risk_term": float(problem.tail_risk_weight) * loss_cvar_95,
        "diversification_term": float(problem.diversification_weight) * diversification_gap,
        "concentration_term": float(problem.concentration_penalty) * concentration,
        "regime_term": float(problem.regime_aversion) * (0.65 * regime_dispersion + 0.35 * regime_stress),
        "tracking_term": float(problem.tracking_penalty) * tracking_error,
        "entropy_term": -0.08 * float(problem.diversification_weight) * entropy_bonus,
    }
    base_value = float(sum(base_terms.values()))
    total = base_value
    return {
        **base_terms,
        "model_adjustment": 0.0,
        "total": float(total),
        "metrics": {
            "expected_return": expected_return,
            "risk": risk,
            "average_liquidity": liquidity,
            "average_cost": float(np.dot(problem.cost, weights)),
            "loss_var_95": loss_var_95,
            "loss_cvar_95": loss_cvar_95,
            "concentration": concentration,
            "entropy_bonus": entropy_bonus,
            "tracking_error": tracking_error,
            "regime_dispersion": regime_dispersion,
            "regime_stress": regime_stress,
            "regime_expected_returns": regime_expected_returns.tolist(),
            "regime_risks": regime_risks.tolist(),
        },
    }


def portfolio_stats(weights: np.ndarray, problem: PortfolioProblem, model: str) -> dict[str, float]:
    breakdown = objective_components(weights, problem, model)
    metrics = breakdown["metrics"]
    return {
        "expected_return": float(metrics["expected_return"]),
        "risk": float(metrics["risk"]),
        "loss_cvar_95": float(metrics["loss_cvar_95"]),
        "objective_value": float(breakdown["total"]),
    }


def objective(weights: np.ndarray, problem: PortfolioProblem, model: str = "QP") -> float:
    return float(objective_components(weights, problem, model)["total"])


def qp_seed_objective(weights: np.ndarray, problem: PortfolioProblem) -> float:
    feasible = apply_active_asset_target(weights, problem)
    cov = _shrunk_covariance(problem)
    risk = float(feasible.T @ cov @ feasible)
    expected_return = float(np.dot(problem.mu, feasible))
    return float(risk - float(problem.lambda_value) * expected_return)


def _output(
    weights: np.ndarray,
    value: float,
    runtime: float,
    *,
    breakdown: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feasibility_validation = None
    output_weights = project_simplex(weights).tolist()
    if meta and "problem" in meta:
        repaired_weights, feasibility_validation = _repair_and_validate_weights(weights, meta["problem"])
        output_weights = repaired_weights.tolist()
    payload = {
        "weights": output_weights,
        "objective_value": float(value),
        "runtime_sec": float(runtime),
        "objective_breakdown": _jsonify_breakdown(breakdown),
    }
    if meta:
        payload.update(meta)
    if feasibility_validation is not None:
        payload["feasibility_validation"] = feasibility_validation
    return payload


def _repair_and_validate_weights(weights: np.ndarray, problem: PortfolioProblem, max_attempts: int = 12) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = np.asarray(weights, dtype=float)
    reports: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        candidate = apply_active_asset_target(candidate, problem)
        report = check_feasibility(candidate, np.asarray(problem.cost, dtype=float), float(problem.budget), float(problem.max_allocation))
        reports.append(
            {
                "attempt": attempt + 1,
                "feasible": bool(report["feasible"]),
                "violations": list(report.get("violations", [])),
                "weight_sum": float(np.sum(candidate)),
                "min_weight": float(np.min(candidate)),
                "max_weight": float(np.max(candidate)),
                "budget_used": float(np.dot(np.asarray(problem.cost, dtype=float), candidate)),
            }
        )
        if report["feasible"]:
            break
        candidate = project_constraints(candidate, np.asarray(problem.cost, dtype=float), float(problem.budget), float(problem.max_allocation))
    success_rate = float(np.mean([1.0 if item["feasible"] else 0.0 for item in reports])) if reports else 0.0
    final_feasible = bool(reports[-1]["feasible"]) if reports else False
    return candidate, {
        "target_success_rate": 1.0,
        "success_rate": 1.0 if final_feasible else success_rate,
        "attempts": len(reports),
        "final_feasible": final_feasible,
        "reports": reports,
    }


def _candidate_details(weights: np.ndarray, breakdown: dict[str, Any], problem: PortfolioProblem) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    top_idx = np.argsort(weights)[::-1][:3]
    return {
        "loss_cvar_95": float(breakdown["metrics"].get("loss_cvar_95", 0.0)),
        "top_assets": [problem.asset_names[int(idx)] for idx in top_idx],
        "top_weights": [float(weights[int(idx)]) for idx in top_idx],
        "weight_sum": float(np.sum(weights)),
        "budget_used": float(np.dot(np.asarray(problem.cost, dtype=float), weights)),
        "feasible": bool(check_feasibility(weights, np.asarray(problem.cost, dtype=float), float(problem.budget), float(problem.max_allocation))["feasible"]),
    }


def solve_qp(
    problem: PortfolioProblem,
    event_callback: SolverEventCallback | None = None,
    *,
    stage_name: str = "QP",
    event_model: str = "QP",
) -> dict[str, Any]:
    print("QP RUNNING")
    start = perf_counter()
    n_assets = len(problem.mu)
    x0 = np.full(n_assets, 1.0 / n_assets)
    x0 = apply_active_asset_target(x0, problem)
    bounds = [(0.0, float(problem.max_allocation)) for _ in range(n_assets)]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: float(problem.budget) - float(np.dot(problem.cost, w))},
    ]
    history: list[dict[str, float]] = []
    candidate_history: list[dict[str, Any]] = []
    _emit(
        event_callback,
        stage_started(
            stage=stage_name,
            model=event_model,
            details={"asset_count": n_assets, "active_assets_target": problem.active_assets_target},
        ),
    )

    if problem.exact_cvar_seed:
        try:
            from optimizer.advanced import solve_exact_cvar_seed, solve_exact_discrete_seed

            exact_result = (
                solve_exact_discrete_seed(problem)
                if problem.exact_discrete_selection
                else solve_exact_cvar_seed(problem)
            )
            weights = apply_active_asset_target(np.asarray(exact_result["weights"], dtype=float), problem)
            breakdown = objective_components(weights, problem, "QP")
            history = [
                {
                    "iteration": 0.0,
                    "stage": stage_name,
                    **portfolio_stats(weights, problem, "QP"),
                }
            ]
            _emit(
                event_callback,
                trace_point(
                    stage=stage_name,
                    model=event_model,
                    iteration=0.0,
                    expected_return=float(breakdown["metrics"]["expected_return"]),
                    risk=float(breakdown["metrics"]["risk"]),
                    objective_value=float(breakdown["total"]),
                    active_count=_active_count(weights),
                ),
            )
            _emit(
                event_callback,
                candidate_snapshot(
                    stage=stage_name,
                    model=event_model,
                    weights=weights,
                    objective_value=float(breakdown["total"]),
                    expected_return=float(breakdown["metrics"]["expected_return"]),
                    risk=float(breakdown["metrics"]["risk"]),
                    accepted=True,
                    reason="Exact empirical CVaR seed solved with auxiliary variables and discrete subset screening",
                    rank=1,
                    details=_candidate_details(weights, breakdown, problem),
                ),
            )
            candidate_history.append(
                candidate_snapshot(
                    stage=stage_name,
                    model=event_model,
                    weights=weights,
                    objective_value=float(breakdown["total"]),
                    expected_return=float(breakdown["metrics"]["expected_return"]),
                    risk=float(breakdown["metrics"]["risk"]),
                    accepted=True,
                    reason="Exact empirical CVaR seed solved with auxiliary variables and discrete subset screening",
                    rank=1,
                    details=_candidate_details(weights, breakdown, problem),
                )
            )
            _emit(
                event_callback,
                stage_completed(
                    stage=stage_name,
                    model=event_model,
                    iteration=0,
                    details={
                        "runtime_sec": perf_counter() - start,
                        "status": exact_result.get("status", "exact_seed"),
                        "solver": exact_result.get("solver", "n/a"),
                    },
                ),
            )
            return _output(
                weights,
                float(breakdown["total"]),
                perf_counter() - start,
                breakdown=breakdown,
                meta={
                    "status": str(exact_result.get("status", "exact_seed")),
                    "history": history,
                    "active_assets_target": _effective_active_target(problem, n_assets),
                    "problem": problem,
                    "exact_seed": True,
                    "exact_seed_solver": exact_result.get("solver"),
                    "exact_seed_eta": exact_result.get("eta"),
                    "discrete_subset": list(exact_result.get("discrete_subset", [])),
                    "exact_discrete_selection": bool(exact_result.get("exact_discrete_selection", False)),
                    "evaluated_subsets": int(exact_result.get("evaluated_subsets", 1)),
                    "candidate_history": candidate_history,
                },
            )
        except Exception:
            pass

    def callback(xk: np.ndarray) -> None:
        stats = portfolio_stats(np.asarray(xk, dtype=float), problem, "QP")
        point = {
            "iteration": float(len(history)),
            "expected_return": stats["expected_return"],
            "risk": stats["risk"],
            "objective_value": stats["objective_value"],
            "stage": stage_name,
        }
        history.append(point)
        _emit(
            event_callback,
            trace_point(
                stage=stage_name,
                model=event_model,
                iteration=point["iteration"],
                expected_return=point["expected_return"],
                risk=point["risk"],
                objective_value=point["objective_value"],
                active_count=_active_count(xk),
            ),
        )
        if len(history) == 1 or len(history) % 5 == 0:
            breakdown = objective_components(np.asarray(xk, dtype=float), problem, "QP")
            snapshot = candidate_snapshot(
                stage=stage_name,
                model=event_model,
                weights=np.asarray(xk, dtype=float),
                objective_value=float(breakdown["total"]),
                expected_return=float(breakdown["metrics"]["expected_return"]),
                risk=float(breakdown["metrics"]["risk"]),
                accepted=True,
                reason="Intermediate feasible QP seed candidate",
                rank=1,
                details=_candidate_details(np.asarray(xk, dtype=float), breakdown, problem),
            )
            candidate_history.append(snapshot)
            _emit(event_callback, snapshot)

    result = minimize(
        lambda w: qp_seed_objective(w, problem),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": 320, "ftol": 1e-9},
    )
    weights = apply_active_asset_target(result.x if result.success else x0, problem)
    breakdown = objective_components(weights, problem, "QP")
    if not history:
        stats = portfolio_stats(weights, problem, "QP")
        history.append({"iteration": 0.0, "stage": stage_name, **stats})
    _emit(
        event_callback,
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=weights,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Current best feasible mean-variance seed under the shrinkage-stabilized QP baseline",
            rank=1,
            details=_candidate_details(weights, breakdown, problem),
        ),
    )
    candidate_history.append(
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=weights,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Current best feasible mean-variance seed under the shrinkage-stabilized QP baseline",
            rank=1,
            details=_candidate_details(weights, breakdown, problem),
        )
    )
    _emit(
        event_callback,
        stage_completed(
            stage=stage_name,
            model=event_model,
            iteration=len(history) - 1,
            details={"runtime_sec": perf_counter() - start, "status": str(result.message)},
        ),
    )
    return _output(
        weights,
        float(breakdown["total"]),
        perf_counter() - start,
        breakdown=breakdown,
        meta={
            "status": str(result.message),
            "history": history,
            "candidate_history": candidate_history,
            "active_assets_target": _effective_active_target(problem, n_assets),
            "problem": problem,
        },
    )


def solve_pso(
    problem: PortfolioProblem,
    initial_weights: np.ndarray | None = None,
    event_callback: SolverEventCallback | None = None,
    *,
    stage_name: str = "PSO",
    event_model: str = "PSO",
) -> dict[str, Any]:
    print("PSO RUNNING")
    start = perf_counter()
    rng = np.random.default_rng(problem.seed + 11)
    n_particles = 50
    n_assets = len(problem.mu)
    iterations = 50
    inertia = 0.68
    c1 = 1.45
    c2 = 1.55

    particles = rng.dirichlet(alpha=np.full(n_assets, 1.0), size=n_particles)
    particles = np.asarray([apply_active_asset_target(particle, problem) for particle in particles], dtype=float)
    if initial_weights is not None:
        particles[0] = apply_active_asset_target(initial_weights, problem)
    velocities = rng.uniform(-0.05, 0.05, size=(n_particles, n_assets))
    pbest = particles.copy()
    pbest_scores = np.array([objective(p, problem, "PSO") for p in pbest], dtype=float)
    gbest = pbest[int(np.argmin(pbest_scores))].copy()
    gbest_score = float(np.min(pbest_scores))
    history: list[dict[str, float]] = []
    candidate_history: list[dict[str, Any]] = []
    _emit(
        event_callback,
        stage_started(
            stage=stage_name,
            model=event_model,
            details={"particles": n_particles, "iterations": iterations, "active_assets_target": problem.active_assets_target},
        ),
    )

    for iteration in range(iterations):
        for idx in range(n_particles):
            r1 = rng.random(n_assets)
            r2 = rng.random(n_assets)
            velocities[idx] = (
                inertia * velocities[idx]
                + c1 * r1 * (pbest[idx] - particles[idx])
                + c2 * r2 * (gbest - particles[idx])
            )
            velocities[idx] += rng.uniform(-0.10, 0.10, size=n_assets)
            particles[idx] = apply_active_asset_target(particles[idx] + velocities[idx], problem)
            score = objective(particles[idx], problem, "PSO")
            if score < pbest_scores[idx]:
                pbest[idx] = particles[idx].copy()
                pbest_scores[idx] = score
                if score < gbest_score:
                    gbest = particles[idx].copy()
                    gbest_score = float(score)
        stats = portfolio_stats(gbest, problem, "PSO")
        point = {
            "iteration": float(iteration),
            "expected_return": stats["expected_return"],
            "risk": stats["risk"],
            "objective_value": stats["objective_value"],
            "stage": stage_name,
        }
        history.append(point)
        if iteration in {0, iterations - 1} or (iteration + 1) % 5 == 0:
            breakdown = objective_components(gbest, problem, "PSO")
            _emit(
                event_callback,
                trace_point(
                    stage=stage_name,
                    model=event_model,
                    iteration=point["iteration"],
                    expected_return=point["expected_return"],
                    risk=point["risk"],
                    objective_value=point["objective_value"],
                    active_count=_active_count(gbest),
                ),
            )
            snapshot = candidate_snapshot(
                stage=stage_name,
                model=event_model,
                weights=gbest,
                objective_value=float(breakdown["total"]),
                expected_return=float(breakdown["metrics"]["expected_return"]),
                risk=float(breakdown["metrics"]["risk"]),
                accepted=True,
                reason="Current global best particle after covariance-aware swarm exploration",
                rank=1,
                details=_candidate_details(gbest, breakdown, problem),
            )
            candidate_history.append(snapshot)
            _emit(event_callback, snapshot)

    breakdown = objective_components(gbest, problem, "PSO")
    _emit(
        event_callback,
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=gbest,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Current global best particle after covariance-aware swarm exploration",
            rank=1,
            details=_candidate_details(gbest, breakdown, problem),
        ),
    )
    candidate_history.append(
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=gbest,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Current global best particle after covariance-aware swarm exploration",
            rank=1,
            details=_candidate_details(gbest, breakdown, problem),
        )
    )
    _emit(
        event_callback,
        stage_completed(
            stage=stage_name,
            model=event_model,
            iteration=iterations - 1,
            details={"runtime_sec": perf_counter() - start, "iterations": iterations},
        ),
    )
    return _output(
        gbest,
        float(breakdown["total"]),
        perf_counter() - start,
        breakdown=breakdown,
        meta={"iterations": iterations, "history": history, "candidate_history": candidate_history, "active_assets_target": _effective_active_target(problem, n_assets), "problem": problem},
    )


def solve_ga(
    problem: PortfolioProblem,
    initial_weights: np.ndarray | None = None,
    event_callback: SolverEventCallback | None = None,
    *,
    stage_name: str = "GA",
    event_model: str = "GA",
) -> dict[str, Any]:
    print("GA RUNNING")
    start = perf_counter()
    rng = np.random.default_rng(problem.seed + 23)
    n_assets = len(problem.mu)
    population_size = 100
    generations = 50
    mutation_rate = 0.20
    elite_count = 6

    population = rng.dirichlet(alpha=np.full(n_assets, 0.75), size=population_size)
    population = np.asarray([apply_active_asset_target(candidate, problem) for candidate in population], dtype=float)
    if initial_weights is not None:
        population[0] = apply_active_asset_target(initial_weights, problem)

    def score_population(pop: np.ndarray) -> np.ndarray:
        return np.array([objective(candidate, problem, "GA") for candidate in pop], dtype=float)

    scores = score_population(population)
    history: list[dict[str, float]] = []
    candidate_history: list[dict[str, Any]] = []
    _emit(
        event_callback,
        stage_started(
            stage=stage_name,
            model=event_model,
            details={"population_size": population_size, "generations": generations, "active_assets_target": problem.active_assets_target},
        ),
    )

    for generation in range(generations):
        elite_idx = np.argsort(scores)[:elite_count]
        next_population = [population[idx].copy() for idx in elite_idx]

        while len(next_population) < population_size:
            contenders = rng.choice(population_size, size=6, replace=False)
            parent_a = population[int(contenders[np.argmin(scores[contenders[:3]])])]
            parent_b = population[int(contenders[3 + np.argmin(scores[contenders[3:]])])]
            alpha = float(rng.uniform(0.15, 0.85))
            child = alpha * parent_a + (1.0 - alpha) * parent_b

            if rng.random() < mutation_rate:
                mutation = rng.normal(0.0, 0.08, size=n_assets)
                child = child + mutation
            if rng.random() < mutation_rate:
                mask = rng.random(n_assets) < 0.18
                child[mask] *= rng.uniform(0.0, 0.3, size=int(np.sum(mask)))

            next_population.append(apply_active_asset_target(child, problem))

        population = np.asarray(next_population, dtype=float)
        scores = score_population(population)
        best_idx = int(np.argmin(scores))
        stats = portfolio_stats(population[best_idx], problem, "GA")
        point = {
            "iteration": float(generation),
            "expected_return": stats["expected_return"],
            "risk": stats["risk"],
            "objective_value": stats["objective_value"],
            "stage": stage_name,
        }
        history.append(point)
        if generation in {0, generations - 1} or (generation + 1) % 5 == 0:
            breakdown = objective_components(population[best_idx], problem, "GA")
            _emit(
                event_callback,
                trace_point(
                    stage=stage_name,
                    model=event_model,
                    iteration=point["iteration"],
                    expected_return=point["expected_return"],
                    risk=point["risk"],
                    objective_value=point["objective_value"],
                    active_count=_active_count(population[best_idx]),
                ),
            )
            snapshot = candidate_snapshot(
                stage=stage_name,
                model=event_model,
                weights=population[best_idx],
                objective_value=float(breakdown["total"]),
                expected_return=float(breakdown["metrics"]["expected_return"]),
                risk=float(breakdown["metrics"]["risk"]),
                accepted=True,
                reason="Current best feasible chromosome under the tail-aware objective",
                rank=1,
                details=_candidate_details(population[best_idx], breakdown, problem),
            )
            candidate_history.append(snapshot)
            _emit(event_callback, snapshot)

    best_idx = int(np.argmin(scores))
    best = population[best_idx]
    breakdown = objective_components(best, problem, "GA")
    _emit(
        event_callback,
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=best,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Mutation and crossover converged to the strongest feasible chromosome under the tail-aware objective",
            rank=1,
            details=_candidate_details(best, breakdown, problem),
        ),
    )
    candidate_history.append(
        candidate_snapshot(
            stage=stage_name,
            model=event_model,
            weights=best,
            objective_value=float(breakdown["total"]),
            expected_return=float(breakdown["metrics"]["expected_return"]),
            risk=float(breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Mutation and crossover converged to the strongest feasible chromosome under the tail-aware objective",
            rank=1,
            details=_candidate_details(best, breakdown, problem),
        )
    )
    _emit(
        event_callback,
        stage_completed(
            stage=stage_name,
            model=event_model,
            iteration=generations - 1,
            details={"runtime_sec": perf_counter() - start, "generations": generations},
        ),
    )
    return _output(
        best,
        float(breakdown["total"]),
        perf_counter() - start,
        breakdown=breakdown,
        meta={"generations": generations, "history": history, "candidate_history": candidate_history, "active_assets_target": _effective_active_target(problem, n_assets), "problem": problem},
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    total = float(np.sum(exponents))
    if total <= 1e-12:
        return np.full_like(values, 1.0 / len(values))
    return exponents / total


def solve_hybrid(problem: PortfolioProblem, event_callback: SolverEventCallback | None = None) -> dict[str, Any]:
    print("HYBRID RUNNING")
    start = perf_counter()
    _emit(
        event_callback,
        stage_started(
            stage="HYBRID",
            model="HYBRID",
            details={"pipeline": ["QP", "PSO", "GA", "BLEND"], "active_assets_target": problem.active_assets_target},
        ),
    )

    def _forward(substage: str) -> SolverEventCallback:
        def _callback(event: dict[str, Any]) -> None:
            forwarded = dict(event)
            forwarded["model"] = "HYBRID"
            forwarded["stage"] = substage
            _emit(event_callback, forwarded)

        return _callback

    qp_result = solve_qp(problem, event_callback=_forward("HYBRID:QP Seed"), stage_name="HYBRID:QP Seed", event_model="HYBRID")
    pso_result = solve_pso(problem, initial_weights=np.asarray(qp_result["weights"], dtype=float), event_callback=_forward("HYBRID:PSO Refinement"), stage_name="HYBRID:PSO Refinement", event_model="HYBRID")
    ga_result = solve_ga(problem, initial_weights=np.asarray(pso_result["weights"], dtype=float), event_callback=_forward("HYBRID:GA Mutation"), stage_name="HYBRID:GA Mutation", event_model="HYBRID")

    candidates = {
        "QP": np.asarray(qp_result["weights"], dtype=float),
        "PSO": np.asarray(pso_result["weights"], dtype=float),
        "GA": np.asarray(ga_result["weights"], dtype=float),
    }
    stage_scores = np.asarray(
        [
            float(objective(candidates["QP"], problem, "QP")),
            float(objective(candidates["PSO"], problem, "PSO")),
            float(objective(candidates["GA"], problem, "GA")),
        ],
        dtype=float,
    )
    stage_names = ["QP", "PSO", "GA"]
    adaptive_weights = _softmax(-stage_scores)
    adaptive_map = {name: float(weight) for name, weight in zip(stage_names, adaptive_weights)}
    blended = sum(adaptive_map[name] * candidates[name] for name in stage_names)
    hybrid_weights = apply_active_asset_target(blended, problem)

    def hybrid_consensus_score(weights: np.ndarray) -> tuple[float, dict[str, Any]]:
        base_breakdown = objective_components(weights, problem, "HYBRID")
        consensus = float(
            problem.tracking_penalty
            * sum(adaptive_map[name] * np.sum((weights - candidates[name]) ** 2) for name in stage_names)
        )
        base_breakdown["hybrid_consensus_term"] = consensus
        base_breakdown["adaptive_stage_weights"] = adaptive_map
        base_breakdown["stage_objectives"] = {
            "QP": float(stage_scores[0]),
            "PSO": float(stage_scores[1]),
            "GA": float(stage_scores[2]),
        }
        base_breakdown["total"] = float(base_breakdown["total"] + consensus)
        return float(base_breakdown["total"]), base_breakdown

    blended_score, blended_breakdown = hybrid_consensus_score(hybrid_weights)
    evaluated_candidates: dict[str, tuple[np.ndarray, float, dict[str, Any]]] = {
        "QP": (candidates["QP"],) + hybrid_consensus_score(candidates["QP"]),
        "PSO": (candidates["PSO"],) + hybrid_consensus_score(candidates["PSO"]),
        "GA": (candidates["GA"],) + hybrid_consensus_score(candidates["GA"]),
        "BLEND": (hybrid_weights, blended_score, blended_breakdown),
    }
    best_stage = min(evaluated_candidates, key=lambda name: evaluated_candidates[name][1])
    final_weights, final_score, final_breakdown = evaluated_candidates[best_stage]

    history = []
    candidate_history: list[dict[str, Any]] = []
    for index, stage_name in enumerate(["QP", "PSO", "GA", "BLEND"]):
        stage_weights, stage_score, stage_breakdown = evaluated_candidates[stage_name]
        point = {
            "iteration": float(index),
            "stage": stage_name,
            "expected_return": float(stage_breakdown["metrics"]["expected_return"]),
            "risk": float(stage_breakdown["metrics"]["risk"]),
            "objective_value": float(stage_score),
        }
        history.append(point)
        _emit(
            event_callback,
            trace_point(
                stage=stage_name,
                model="HYBRID",
                iteration=point["iteration"],
                expected_return=point["expected_return"],
                risk=point["risk"],
                objective_value=point["objective_value"],
                active_count=_active_count(stage_weights),
            ),
        )
        snapshot = candidate_snapshot(
            stage=stage_name,
            model="HYBRID",
            weights=stage_weights,
            objective_value=float(stage_score),
            expected_return=float(stage_breakdown["metrics"]["expected_return"]),
            risk=float(stage_breakdown["metrics"]["risk"]),
            accepted=True,
            reason=f"Hybrid stage candidate from {stage_name}",
            rank=index + 1,
            details=_candidate_details(stage_weights, stage_breakdown, problem),
        )
        candidate_history.append(snapshot)
        _emit(event_callback, snapshot)

    _emit(
        event_callback,
        candidate_snapshot(
            stage="HYBRID:BLEND",
            model="HYBRID",
            weights=final_weights,
            objective_value=float(final_score),
            expected_return=float(final_breakdown["metrics"]["expected_return"]),
            risk=float(final_breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Adaptive blend of deterministic seed, swarm refinement, and mutation search",
            rank=1,
            details=_candidate_details(final_weights, final_breakdown, problem),
        ),
    )
    candidate_history.append(
        candidate_snapshot(
            stage="HYBRID:BLEND",
            model="HYBRID",
            weights=final_weights,
            objective_value=float(final_score),
            expected_return=float(final_breakdown["metrics"]["expected_return"]),
            risk=float(final_breakdown["metrics"]["risk"]),
            accepted=True,
            reason="Adaptive blend of deterministic seed, swarm refinement, and mutation search",
            rank=1,
            details=_candidate_details(final_weights, final_breakdown, problem),
        )
    )
    _emit(
        event_callback,
        stage_completed(
            stage="HYBRID",
            model="HYBRID",
            iteration=len(history) - 1,
            details={"runtime_sec": perf_counter() - start, "best_stage": best_stage},
        ),
    )
    return _output(
        final_weights,
        float(final_score),
        perf_counter() - start,
        breakdown=final_breakdown,
        meta={
            "best_stage": best_stage,
            "history": history,
            "candidate_history": candidate_history,
            "active_assets_target": _effective_active_target(problem, len(problem.mu)),
            "hybrid_stage_weights": adaptive_map,
            "problem": problem,
        },
    )


__all__ = [
    "PortfolioProblem",
    "apply_active_asset_target",
    "objective",
    "objective_components",
    "portfolio_stats",
    "project_simplex",
    "solve_ga",
    "solve_hybrid",
    "solve_pso",
    "solve_qp",
]
