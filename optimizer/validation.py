"""
Validation and analytics helpers for portfolio outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.solvers import PortfolioProblem, solve_ga, solve_hybrid, solve_pso, solve_qp


def validate_weights(weights: np.ndarray | list[float], atol: float = 1e-6) -> bool:
    weights = np.asarray(weights, dtype=float)
    return bool(np.all(weights >= -atol) and np.isclose(np.sum(weights), 1.0, atol=atol))


def diversification_count(weights: np.ndarray | list[float], threshold: float = 1e-6) -> int:
    weights = np.asarray(weights, dtype=float)
    return int(np.sum(weights > threshold))


def compare_weight_sets(model_outputs: dict[str, dict[str, Any]], threshold: float = 0.03) -> bool:
    items = list(model_outputs.items())
    for index, (_, left) in enumerate(items):
        left_w = np.asarray(left["weights"], dtype=float)
        for _, right in items[index + 1 :]:
            right_w = np.asarray(right["weights"], dtype=float)
            if float(np.linalg.norm(left_w - right_w, ord=1)) <= threshold:
                return False
    return True


def monte_carlo_summary(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    paths: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scenarios = rng.multivariate_normal(mu, cov, size=paths)
    portfolio_returns = scenarios @ np.asarray(weights, dtype=float)
    losses = -portfolio_returns
    loss_var_95 = float(np.quantile(losses, 0.95))
    tail_losses = losses[losses >= loss_var_95 - 1e-12]
    return {
        "mean_return": float(np.mean(portfolio_returns)),
        "std_return": float(np.std(portfolio_returns)),
        "p05": float(np.quantile(portfolio_returns, 0.05)),
        "p95": float(np.quantile(portfolio_returns, 0.95)),
        "loss_var_95": loss_var_95,
        "loss_cvar_95": float(np.mean(tail_losses)) if tail_losses.size else loss_var_95,
        "samples": [float(value) for value in portfolio_returns[: min(200, len(portfolio_returns))]],
    }


def frontier_points(problem: PortfolioProblem, points: int = 12) -> list[dict[str, float]]:
    lambdas = np.linspace(0.2, 2.0, points)
    frontier: list[dict[str, float]] = []
    for idx, lambda_value in enumerate(lambdas):
        variant = PortfolioProblem(**{**problem.to_dict(), "lambda_value": float(lambda_value), "seed": problem.seed + idx})
        result = solve_qp(variant)
        weights = np.asarray(result["weights"], dtype=float)
        frontier.append(
            {
                "lambda_value": float(lambda_value),
                "expected_return": float(np.dot(weights, variant.mu)),
                "risk": float(weights.T @ variant.cov @ weights),
            }
        )
    return frontier


def sensitivity_points(
    problem: PortfolioProblem,
    *,
    method: str,
    lambda_values: np.ndarray,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for idx, lambda_value in enumerate(np.asarray(lambda_values, dtype=float)):
        variant = PortfolioProblem(**{**problem.to_dict(), "lambda_value": float(lambda_value), "seed": problem.seed + 31 + idx})
        if method == "QP":
            result = solve_qp(variant)
        elif method == "PSO":
            result = solve_pso(variant)
        elif method == "GA":
            result = solve_ga(variant)
        else:
            result = solve_hybrid(variant)
        weights = np.asarray(result["weights"], dtype=float)
        points.append(
            {
                "lambda_value": float(lambda_value),
                "expected_return": float(np.dot(weights, variant.mu)),
                "risk": float(weights.T @ variant.cov @ weights),
            }
        )
    return points
