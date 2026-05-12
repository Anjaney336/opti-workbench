"""
Advanced optimization helpers: regime calibration, exact CVaR seed solving,
discrete subset search, and rolling multi-period validation.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    total = float(np.sum(probs))
    if total <= 1e-12:
        return np.full_like(probs, 1.0 / len(probs))
    return probs / total


def _effective_active_target(active_assets_target: int | None, max_allocation: float, n_assets: int) -> int:
    minimum_assets = max(1, int(np.ceil(1.0 / max(float(max_allocation), 1e-6) - 1e-12)))
    if active_assets_target is None:
        return min(n_assets, minimum_assets)
    requested = max(1, int(active_assets_target))
    return min(n_assets, max(requested, minimum_assets))


def _shrunk_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    raw = np.asarray(cov, dtype=float)
    diagonal = np.diag(np.diag(raw))
    shrunk = (1.0 - float(np.clip(shrinkage, 0.0, 0.95))) * raw + float(np.clip(shrinkage, 0.0, 0.95)) * diagonal
    return 0.5 * (shrunk + shrunk.T)


def _affordability_scale(budget: float) -> float:
    return 1.0 + 3.0 / max(float(budget), 0.5)


def _regime_dispersion_matrix(regime_mu: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    regime_mu = np.asarray(regime_mu, dtype=float)
    probabilities = _normalize_probabilities(np.asarray(probabilities, dtype=float))
    center = np.diag(probabilities) - np.outer(probabilities, probabilities)
    matrix = regime_mu.T @ center @ regime_mu
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.maximum(evals, 0.0)
    repaired = evecs @ np.diag(evals) @ evecs.T
    return 0.5 * (repaired + repaired.T)


def calibrate_regimes_from_history(
    historical_returns: np.ndarray,
    *,
    macro_signal: np.ndarray | None = None,
    fallback_mu: np.ndarray | None = None,
    fallback_cov: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate Bear/Neutral/Bull regimes from historical asset returns.

    If `macro_signal` is absent, the cross-sectional market return at each time
    step is used as the regime score.
    """
    history = np.asarray(historical_returns, dtype=float)
    if history.ndim != 2 or history.shape[0] < 12:
        raise ValueError("historical_returns must be a 2D array with at least 12 observations.")

    if macro_signal is None:
        score = np.mean(history, axis=1)
    else:
        score = np.asarray(macro_signal, dtype=float)
        if score.shape[0] != history.shape[0]:
            raise ValueError("macro_signal must have the same length as historical_returns.")

    lower, upper = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    masks = [
        score <= lower,
        (score > lower) & (score < upper),
        score >= upper,
    ]
    names = ["Bear", "Neutral", "Bull"]
    probabilities = np.asarray([np.mean(mask) for mask in masks], dtype=float)

    default_mu = np.asarray(fallback_mu if fallback_mu is not None else np.mean(history, axis=0), dtype=float)
    default_cov = np.asarray(
        fallback_cov if fallback_cov is not None else np.cov(history, rowvar=False),
        dtype=float,
    )
    regime_mu: list[np.ndarray] = []
    regime_covariances: list[np.ndarray] = []
    scenario_chunks: list[np.ndarray] = []

    for mask in masks:
        subset = history[mask]
        if subset.shape[0] < 3:
            subset = history
        regime_mu.append(np.mean(subset, axis=0) if subset.size else default_mu)
        cov = np.cov(subset, rowvar=False) if subset.shape[0] > 1 else default_cov
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        diag_scale = max(float(np.trace(cov)) / max(len(cov), 1), 1e-6)
        cov = cov + np.eye(cov.shape[0]) * (1e-4 * diag_scale)
        regime_covariances.append(cov)
        scenario_chunks.append(np.asarray(subset, dtype=float))

    return names, probabilities, np.vstack(regime_mu), np.asarray(regime_covariances), np.vstack(scenario_chunks)


def solve_exact_cvar_seed(
    problem: Any,
    *,
    asset_indices: tuple[int, ...] | None = None,
    solver_order: tuple[str, ...] = ("OSQP", "CLARABEL", "SCS"),
) -> dict[str, Any]:
    """
    Solve a convex seed problem with exact empirical CVaR via auxiliary variables.

    The seed keeps the submitted model structure as closely as possible while
    staying convex:
    - shrinkage covariance risk
    - return
    - cost and liquidity
    - exact empirical CVaR with eta/u variables
    - concentration penalty
    - regime dispersion/stress
    - tracking penalty
    - convex concentration-gap surrogate
    """
    start = perf_counter()
    indices = tuple(asset_indices) if asset_indices is not None else tuple(range(len(problem.mu)))
    mu = np.asarray(problem.mu, dtype=float)[list(indices)]
    cost = np.asarray(problem.cost, dtype=float)[list(indices)]
    liquidity = np.asarray(problem.liquidity, dtype=float)[list(indices)]
    anchor = np.asarray(problem.anchor_weights, dtype=float)[list(indices)]
    anchor_total = float(np.sum(anchor))
    anchor = anchor / anchor_total if anchor_total > 1e-12 else np.full_like(anchor, 1.0 / len(anchor))
    cov = _shrunk_covariance(np.asarray(problem.cov, dtype=float)[np.ix_(indices, indices)], problem.covariance_shrinkage)
    regime_mu = np.asarray(problem.regime_mu, dtype=float)[:, list(indices)]
    regime_covariances = np.asarray(problem.regime_covariances, dtype=float)[:, list(indices), :][:, :, list(indices)]
    scenario_returns = np.asarray(problem.scenario_returns, dtype=float)[:, list(indices)]
    probabilities = _normalize_probabilities(np.asarray(problem.regime_probabilities, dtype=float))

    n_assets = len(indices)
    target_active = _effective_active_target(problem.active_assets_target, problem.max_allocation, len(problem.mu))
    target_concentration = 1.0 / max(target_active, 1.0)
    alpha = max(1e-8, 1.0 - float(problem.cvar_alpha))

    x = cp.Variable(n_assets)
    eta = cp.Variable()
    u = cp.Variable(scenario_returns.shape[0], nonneg=True)
    concentration_slack = cp.Variable(nonneg=True)

    concentration = cp.sum_squares(x)
    tracking = cp.sum_squares(x - anchor)
    losses = -scenario_returns @ x
    cvar_expr = eta + cp.sum(u) / max(1.0, scenario_returns.shape[0]) / alpha
    regime_dispersion = cp.quad_form(x, _regime_dispersion_matrix(regime_mu, probabilities))
    regime_stress = sum(
        float(probabilities[idx]) * cp.quad_form(x, np.asarray(regime_covariances[idx], dtype=float))
        for idx in range(len(probabilities))
    )

    objective = cp.Minimize(
        cp.quad_form(x, cov)
        - float(problem.lambda_value) * (mu @ x)
        + float(problem.cost_sensitivity) * _affordability_scale(problem.budget) * (cost @ x)
        - float(problem.liquidity_weight) * (liquidity @ x)
        + float(problem.tail_risk_weight) * cvar_expr
        + float(problem.diversification_weight) * cp.square(concentration_slack)
        + float(problem.concentration_penalty) * concentration
        + float(problem.regime_aversion) * (0.65 * regime_dispersion + 0.35 * regime_stress)
        + float(problem.tracking_penalty) * tracking
    )

    constraints = [
        cp.sum(x) == 1.0,
        x >= 0.0,
        x <= float(problem.max_allocation),
        cost @ x <= float(problem.budget),
        u >= losses - eta,
        u >= 0.0,
        concentration - target_concentration <= concentration_slack,
    ]

    cp_problem = cp.Problem(objective, constraints)
    last_error = None
    for solver_name in solver_order:
        try:
            cp_problem.solve(solver=solver_name, warm_start=True, verbose=False)
            if x.value is not None and cp_problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                weights = np.zeros(len(problem.mu), dtype=float)
                weights[list(indices)] = np.asarray(x.value, dtype=float)
                return {
                    "weights": weights,
                    "objective_value": float(cp_problem.value),
                    "runtime_sec": perf_counter() - start,
                    "status": str(cp_problem.status),
                    "solver": solver_name,
                    "eta": float(eta.value) if eta.value is not None else None,
                    "exact_seed": True,
                    "discrete_subset": indices,
                }
        except Exception as exc:  # pragma: no cover - solver-specific failure path
            last_error = exc
            continue

    raise RuntimeError(f"Exact CVaR seed solve failed. Last error: {last_error}")


def solve_exact_discrete_seed(
    problem: Any,
    *,
    shortlist_limit: int = 10,
    max_exact_subsets: int = 700,
) -> dict[str, Any]:
    """
    Exact cardinality-constrained seed solve by enumerating small subsets.

    For 10-15 assets and K around 4, this is a practical exact discrete upgrade.
    """
    n_assets = len(problem.mu)
    target = _effective_active_target(problem.active_assets_target, problem.max_allocation, n_assets)
    if target >= n_assets:
        return solve_exact_cvar_seed(problem)

    full_indices = tuple(range(n_assets))
    subset_count = 0
    try_indices = full_indices
    total_subsets = sum(1 for _ in combinations(full_indices, target))
    if total_subsets > max_exact_subsets:
        mu = np.asarray(problem.mu, dtype=float)
        liquidity = np.asarray(problem.liquidity, dtype=float)
        cost = np.asarray(problem.cost, dtype=float)
        sigma = np.sqrt(np.maximum(np.diag(np.asarray(problem.cov, dtype=float)), 1e-12))
        score = 0.45 * mu + 0.20 * liquidity - 0.20 * cost - 0.15 * sigma
        shortlist = np.argsort(score)[-max(target + 2, min(shortlist_limit, n_assets)) :]
        try_indices = tuple(int(idx) for idx in shortlist)

    best_result: dict[str, Any] | None = None
    best_value = float("inf")
    for subset in combinations(try_indices, target):
        subset_count += 1
        try:
            result = solve_exact_cvar_seed(problem, asset_indices=tuple(sorted(subset)))
        except Exception:
            continue
        if float(result["objective_value"]) < best_value:
            best_result = result
            best_value = float(result["objective_value"])

    if best_result is None:
        raise RuntimeError("Exact discrete seed search failed on all candidate subsets.")
    best_result["evaluated_subsets"] = subset_count
    best_result["exact_discrete_selection"] = True
    return best_result


def _historical_variant(problem: Any, history_window: np.ndarray, step: int, macro_window: np.ndarray | None = None) -> Any:
    mu = np.mean(history_window, axis=0)
    cov = np.cov(history_window, rowvar=False)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    diag_scale = max(float(np.trace(cov)) / max(len(cov), 1), 1e-6)
    cov = cov + np.eye(cov.shape[0]) * (1e-4 * diag_scale)
    regime_names, regime_probabilities, regime_mu, regime_covariances, scenario_returns = calibrate_regimes_from_history(
        history_window,
        macro_signal=macro_window,
        fallback_mu=mu,
        fallback_cov=cov,
    )
    return replace(
        problem,
        mu=mu,
        cov=cov,
        regime_names=regime_names,
        regime_probabilities=regime_probabilities,
        regime_mu=regime_mu,
        regime_covariances=regime_covariances,
        scenario_returns=scenario_returns,
        seed=int(problem.seed + 1000 + step),
        exact_discrete_selection=False,
    )


def rolling_backtest(
    problem: Any,
    historical_returns: np.ndarray,
    *,
    method: str,
    lookback: int = 12,
    macro_signal: np.ndarray | None = None,
    solver_callback: Any,
) -> dict[str, Any]:
    """
    Rolling out-of-sample validation on historical return paths.
    """
    history = np.asarray(historical_returns, dtype=float)
    if history.ndim != 2 or history.shape[0] <= lookback:
        raise ValueError("historical_returns must have more rows than lookback.")
    macro = None if macro_signal is None else np.asarray(macro_signal, dtype=float)
    if macro is not None and macro.shape[0] != history.shape[0]:
        raise ValueError("macro_signal must align with historical_returns.")

    records: list[dict[str, Any]] = []
    wealth = 1.0
    previous_weights = np.asarray(problem.anchor_weights, dtype=float)
    for step in range(lookback, history.shape[0]):
        window = history[step - lookback : step]
        macro_window = None if macro is None else macro[step - lookback : step]
        variant = _historical_variant(problem, window, step, macro_window)
        raw = solver_callback(method, variant)
        weights = np.asarray(raw["weights"], dtype=float)
        realized_return = float(history[step] @ weights)
        turnover = float(np.sum(np.abs(weights - previous_weights)))
        wealth *= 1.0 + realized_return
        records.append(
            {
                "step": int(step),
                "realized_return": realized_return,
                "wealth": float(wealth),
                "turnover": turnover,
                "weights": weights.tolist(),
            }
        )
        previous_weights = weights

    realized_returns = np.asarray([row["realized_return"] for row in records], dtype=float)
    return {
        "lookback": int(lookback),
        "periods": int(len(records)),
        "cumulative_return": float(wealth - 1.0),
        "mean_return": float(np.mean(realized_returns)) if realized_returns.size else 0.0,
        "volatility": float(np.std(realized_returns)) if realized_returns.size else 0.0,
        "loss_cvar_95": float(np.mean(np.sort(-realized_returns)[-max(1, int(np.ceil(0.05 * len(realized_returns)))) :]))
        if realized_returns.size
        else 0.0,
        "path": records,
    }


def multi_period_receding_horizon(
    problem: Any,
    historical_returns: np.ndarray,
    *,
    horizon: int,
    method: str,
    macro_signal: np.ndarray | None = None,
    solver_callback: Any,
) -> dict[str, Any]:
    """
    Practical dynamic multi-period allocation via rolling re-optimization.
    """
    plan = rolling_backtest(
        problem,
        historical_returns,
        method=method,
        lookback=max(12, int(horizon)),
        macro_signal=macro_signal,
        solver_callback=solver_callback,
    )
    plan["planning_horizon"] = int(horizon)
    plan["policy"] = "receding_horizon_reoptimization"
    return plan
