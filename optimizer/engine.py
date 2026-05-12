"""
Multi-model portfolio optimization engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from explanation.decision_range_agent import DecisionRangeAgent
from explanation.model_intelligence import ModelIntelligenceAgent
from explanation.transparency_agent import SolverTransparencyAgent
from optimizer.advanced import calibrate_regimes_from_history, multi_period_receding_horizon, rolling_backtest
from optimizer.asset_engine import AssetUniverse, assets_to_dicts, generate_asset_universe
from optimizer.live_trace import build_3d_trace_payload
from optimizer.returns_engine import estimate_expected_returns
from optimizer.solvers import PortfolioProblem, objective_components, solve_ga, solve_hybrid, solve_pso, solve_qp
from optimizer.trace_analysis import build_trace_transparency
from optimizer.validation import (
    compare_weight_sets,
    diversification_count,
    frontier_points,
    monte_carlo_summary,
    sensitivity_points,
    validate_weights,
)
from src.utils import check_feasibility

TARGET_PORTFOLIO_RISK = 0.0087


@dataclass(frozen=True)
class PortfolioRequest:
    asset_count: int = 12
    active_assets_target: int = 4
    budget: float = 5.0
    max_allocation: float = 0.35
    lambda_value: float = 1.0
    liquidity_weight: float = 0.35
    cost_sensitivity: float = 0.15
    tail_risk_weight: float = 0.45
    diversification_weight: float = 0.30
    concentration_penalty: float = 0.18
    regime_aversion: float = 0.28
    tracking_penalty: float = 0.12
    covariance_shrinkage: float = 0.15
    cvar_alpha: float = 0.95
    method: str = "AUTO"
    seed: int = 42
    monte_carlo_paths: int = 500
    assets: list[dict[str, Any]] | None = None
    dataset: str = "balanced"
    exact_cvar_seed: bool = True
    exact_discrete_selection: bool = True
    planning_horizon: int = 3
    backtest_lookback: int = 12
    historical_returns: list[list[float]] | None = None
    macro_signal: list[float] | None = None
    return_engine: str = "BLACK_LITTERMAN"
    return_prior_strength: float = 0.55
    bl_tau: float = 0.08
    bl_view_confidence: float = 0.65


class PortfolioOptimizer:
    def __init__(self, request: PortfolioRequest) -> None:
        self.request = request
        self.method = self._normalize_method(request.method)
        self.dataset = str(request.dataset or "balanced").lower()
        self.return_engine_info: dict[str, Any] = {}
        self.calibration_info: dict[str, Any] = {}
        self.universe = self._build_universe()
        self.problem = self._build_problem()
        self.problem = self._calibrate_problem_to_target_risk(self.problem)
        self.explainer = ModelIntelligenceAgent()
        self.transparency_agent = SolverTransparencyAgent()
        self.decision_agent = DecisionRangeAgent()

    def _build_universe(self) -> AssetUniverse:
        provided_assets = self.request.assets
        target_count = len(provided_assets) if provided_assets else self.request.asset_count
        target_count = min(15, max(10, int(target_count)))
        return generate_asset_universe(
            n_assets=target_count,
            seed=self.request.seed,
            scenario=self.dataset,
            provided_assets=provided_assets,
        )

    def _build_problem(self) -> PortfolioProblem:
        cov = np.asarray(self.universe.cov_matrix, dtype=float)
        cov = 0.5 * (cov + cov.T)
        diag_scale = max(float(np.trace(cov)) / max(len(cov), 1), 1e-6)
        cov = cov + np.eye(cov.shape[0]) * (1e-4 * diag_scale)
        asset_count = len(self.universe.assets)
        if float(self.request.max_allocation) * asset_count < 1.0 - 1e-12:
            raise ValueError("max_allocation is too small for a feasible normalized portfolio.")
        min_cost = self._minimum_feasible_cost(self.universe.cost, float(self.request.max_allocation))
        if float(self.request.budget) < min_cost - 1e-12:
            raise ValueError(
                f"budget={self.request.budget:.6f} is infeasible for max_allocation={self.request.max_allocation:.4f}; "
                f"minimum feasible cost is {min_cost:.6f}."
            )
        anchor_weights = np.ones(asset_count, dtype=float) / max(asset_count, 1)
        historical_returns = self._historical_returns_array(asset_count)
        macro_signal = self._macro_signal_array(historical_returns)
        estimated_mu = self._estimate_expected_returns(cov, historical_returns)
        regime_names, regime_probabilities, regime_mu, regime_covariances, scenario_returns = self._build_regime_inputs(
            cov,
            estimated_mu,
            historical_returns,
            macro_signal,
        )
        return PortfolioProblem(
            mu=estimated_mu,
            cov=cov,
            liquidity=self.universe.liquidity,
            cost=self.universe.cost,
            budget=self.request.budget,
            max_allocation=self.request.max_allocation,
            lambda_value=self.request.lambda_value,
            liquidity_weight=self.request.liquidity_weight,
            cost_sensitivity=self.request.cost_sensitivity,
            tail_risk_weight=self.request.tail_risk_weight,
            diversification_weight=self.request.diversification_weight,
            concentration_penalty=self.request.concentration_penalty,
            regime_aversion=self.request.regime_aversion,
            tracking_penalty=self.request.tracking_penalty,
            covariance_shrinkage=self.request.covariance_shrinkage,
            cvar_alpha=self.request.cvar_alpha,
            active_assets_target=max(1, min(int(self.request.active_assets_target), len(self.universe.assets))),
            seed=self.request.seed,
            asset_names=[asset.name for asset in self.universe.assets],
            regime_names=regime_names,
            regime_probabilities=regime_probabilities,
            regime_mu=regime_mu,
            regime_covariances=regime_covariances,
            scenario_returns=scenario_returns,
            anchor_weights=anchor_weights,
            exact_cvar_seed=bool(self.request.exact_cvar_seed),
            exact_discrete_selection=bool(self.request.exact_discrete_selection),
            planning_horizon=max(2, int(self.request.planning_horizon)),
            backtest_lookback=max(4, int(self.request.backtest_lookback)),
            historical_returns=historical_returns,
            macro_signal=macro_signal,
        )

    @staticmethod
    def _minimum_feasible_cost(costs: np.ndarray, max_allocation: float) -> float:
        remaining = 1.0
        total_cost = 0.0
        allocation_cap = float(np.clip(max_allocation, 1e-6, 1.0))
        for cost in np.sort(np.asarray(costs, dtype=float)):
            if remaining <= 1e-12:
                break
            allocation = min(allocation_cap, remaining)
            total_cost += allocation * float(cost)
            remaining -= allocation
        if remaining > 1e-10:
            raise ValueError("Unable to construct a feasible portfolio under the requested max_allocation.")
        return float(total_cost)

    def _estimate_expected_returns(self, cov: np.ndarray, historical_returns: np.ndarray | None) -> np.ndarray:
        posterior_mu, info = estimate_expected_returns(
            self.request.return_engine,
            self.universe.mu,
            cov,
            liquidity=self.universe.liquidity,
            cost=self.universe.cost,
            historical_returns=historical_returns,
            risk_aversion=self.request.lambda_value,
            prior_strength=self.request.return_prior_strength,
            tau=self.request.bl_tau,
            view_confidence=self.request.bl_view_confidence,
        )
        self.return_engine_info = {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
            for key, value in info.items()
        }
        return np.asarray(posterior_mu, dtype=float)

    def _build_regime_inputs(
        self,
        cov: np.ndarray,
        estimated_mu: np.ndarray,
        historical_returns: np.ndarray | None,
        macro_signal: np.ndarray | None,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if historical_returns is not None:
            return calibrate_regimes_from_history(
                historical_returns,
                macro_signal=macro_signal,
                fallback_mu=estimated_mu,
                fallback_cov=cov,
            )
        regime_names = ["Bear", "Neutral", "Bull"]
        regime_probabilities = np.asarray([0.25, 0.45, 0.30], dtype=float)
        return_scale = np.asarray([0.86, 1.00, 1.16], dtype=float)
        risk_scale = np.asarray([1.24, 1.00, 0.88], dtype=float)
        if self.dataset in {"risky", "high_growth"}:
            return_scale = return_scale * np.asarray([0.96, 1.04, 1.08], dtype=float)
            risk_scale = risk_scale * np.asarray([1.05, 1.02, 0.96], dtype=float)
        elif self.dataset in {"stable", "low_liquidity"}:
            return_scale = return_scale * np.asarray([0.98, 0.99, 1.04], dtype=float)
            risk_scale = risk_scale * np.asarray([1.08, 0.96, 0.92], dtype=float)

        regime_mu = np.vstack([np.asarray(estimated_mu, dtype=float) * scale for scale in return_scale])
        regime_covariances = np.asarray([cov * scale for scale in risk_scale], dtype=float)

        rng = np.random.default_rng(self.request.seed + 101)
        scenario_chunks: list[np.ndarray] = []
        for idx, prob in enumerate(regime_probabilities):
            samples = max(12, int(round(72 * float(prob))))
            scenario_chunks.append(rng.multivariate_normal(regime_mu[idx], regime_covariances[idx], size=samples))
        scenario_returns = np.vstack(scenario_chunks)
        return regime_names, regime_probabilities, regime_mu, regime_covariances, scenario_returns

    def _historical_returns_array(self, asset_count: int) -> np.ndarray | None:
        if self.request.historical_returns is None:
            return None
        history = np.asarray(self.request.historical_returns, dtype=float)
        if history.ndim != 2 or history.shape[1] != asset_count:
            raise ValueError(
                f"historical_returns must be a 2D array with shape (T, {asset_count}) matching the asset universe."
            )
        if history.shape[0] < max(12, int(self.request.backtest_lookback) + 1):
            raise ValueError("historical_returns must contain at least max(12, backtest_lookback + 1) observations.")
        return history

    def _macro_signal_array(self, historical_returns: np.ndarray | None) -> np.ndarray | None:
        if self.request.macro_signal is None:
            return None
        macro = np.asarray(self.request.macro_signal, dtype=float)
        if historical_returns is not None and macro.shape[0] != historical_returns.shape[0]:
            raise ValueError("macro_signal must have the same number of observations as historical_returns.")
        return macro

    def _variant_problem(self, problem: PortfolioProblem, **updates: Any) -> PortfolioProblem:
        payload = problem.to_dict()
        payload.update(updates)
        return PortfolioProblem(**payload)

    def _solve_qp_for_calibration(self, problem: PortfolioProblem) -> tuple[np.ndarray, float]:
        probe = self._variant_problem(problem, exact_cvar_seed=False, exact_discrete_selection=False)
        result = solve_qp(probe)
        weights = np.asarray(result["weights"], dtype=float)
        risk = float(weights.T @ probe.cov @ weights)
        return weights, risk

    def _calibrate_problem_to_target_risk(self, base_problem: PortfolioProblem) -> PortfolioProblem:
        target_risk = TARGET_PORTFOLIO_RISK
        base_lambda = float(base_problem.lambda_value)
        base_phi = float(base_problem.covariance_shrinkage)
        lambda_grid = np.linspace(max(0.2, base_lambda * 0.45), max(0.8, base_lambda * 2.6), 6)
        phi_grid = np.linspace(max(0.0, base_phi - 0.20), min(0.8, base_phi + 0.25), 4)

        best: dict[str, Any] | None = None
        for phi in phi_grid:
            for lambda_value in lambda_grid:
                variant = self._variant_problem(
                    base_problem,
                    lambda_value=float(lambda_value),
                    covariance_shrinkage=float(phi),
                    seed=int(base_problem.seed + round(lambda_value * 100) + round(phi * 1000)),
                )
                weights, risk = self._solve_qp_for_calibration(variant)
                error = abs(risk - target_risk)
                if best is None or error < best["error"]:
                    best = {
                        "lambda_value": float(lambda_value),
                        "covariance_shrinkage": float(phi),
                        "weights": weights,
                        "achieved_risk": float(risk),
                        "error": float(error),
                    }

        if best is None:
            self.calibration_info = {
                "target_risk": target_risk,
                "achieved_risk": float(base_problem.lambda_value),
                "absolute_error": None,
                "calibrated": False,
            }
            return base_problem

        calibrated_problem = self._variant_problem(
            base_problem,
            lambda_value=float(best["lambda_value"]),
            covariance_shrinkage=float(best["covariance_shrinkage"]),
        )
        self.calibration_info = {
            "target_risk": float(target_risk),
            "achieved_risk": float(best["achieved_risk"]),
            "absolute_error": float(best["error"]),
            "calibrated_lambda": float(best["lambda_value"]),
            "calibrated_phi": float(best["covariance_shrinkage"]),
            "risk_target_hit": bool(best["error"] <= 5e-4),
            "calibrated": True,
        }
        return calibrated_problem

    def _feasibility_gate(self, weights: np.ndarray) -> dict[str, Any]:
        report = check_feasibility(weights, self.problem.cost, self.problem.budget, self.problem.max_allocation)
        return {
            "target_success_rate": 1.0,
            "success_rate": 1.0 if report["feasible"] else 0.0,
            "final_feasible": bool(report["feasible"]),
            "violations": list(report.get("violations", [])),
            "sum_weights": float(np.sum(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "budget_used": float(np.dot(self.problem.cost, weights)),
        }

    def _solver_manifold(self, candidate_history: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidate_history:
            return {"points": []}
        final_weights = np.asarray(candidate_history[-1].get("weights", []), dtype=float)
        if final_weights.size == 0:
            return {"points": []}
        top_idx = np.argsort(final_weights)[::-1][:3]
        top_labels = [self.problem.asset_names[int(idx)] for idx in top_idx]
        points: list[dict[str, Any]] = []
        for idx, item in enumerate(candidate_history):
            weights = np.asarray(item.get("weights", []), dtype=float)
            if weights.size == 0:
                continue
            cvar = float(((item.get("details") or {}).get("loss_cvar_95", 0.0)))
            points.append(
                {
                    "iteration": float(item.get("iteration", idx)),
                    "stage": str(item.get("stage", "")),
                    "x": float(weights[int(top_idx[0])]) if len(top_idx) > 0 else 0.0,
                    "y": float(weights[int(top_idx[1])]) if len(top_idx) > 1 else 0.0,
                    "z": float(weights[int(top_idx[2])]) if len(top_idx) > 2 else 0.0,
                    "objective_value": float(item.get("objective_value", 0.0)),
                    "expected_return": float(item.get("expected_return", 0.0)),
                    "risk": float(item.get("risk", 0.0)),
                    "loss_cvar_95": cvar,
                }
            )
        return {"asset_labels": top_labels, "points": points}

    def _live_step_insight(
        self,
        model: str,
        stage: str,
        latest_event: dict[str, Any] | None,
        current_output: dict[str, Any] | None,
    ) -> list[str]:
        if model == "QP":
            if self.problem.exact_cvar_seed:
                objective_note = (
                    "QP is acting as a convex seed stage with exact empirical CVaR auxiliary variables, hard feasibility, "
                    "and optional discrete subset screening before the hybrid refinements."
                )
            else:
                objective_note = (
                    "QP is solving the shrinkage-stabilized mean-variance baseline and is used as the deterministic seed "
                    "for the hybrid pipeline."
                )
        else:
            objective_note = (
                f"{model} is solving the unified multi-term objective with covariance risk, return reward, liquidity reward, "
                "tail risk, regime stress, and diversification control."
            )
        insights = [
            objective_note,
            f"Current backend stage: {stage}. The optimizer respects budget, box, simplex, and active-asset feasibility with target {self.problem.active_assets_target}.",
        ]
        if current_output is not None:
            insights.append(
                f"Current feasible portfolio return is {current_output['expected_return']:.2%}, risk is {current_output['risk']:.2%}, and simulated loss CVaR is {current_output['regime_metrics'].get('loss_cvar_95', 0.0):.2%}."
            )
        if latest_event and latest_event.get("event_type") == "trace_point":
            insights.append(
                f"Latest iteration {int(float(latest_event.get('iteration', 0.0)))} moved the objective to {float(latest_event.get('objective_value', 0.0)):.5f}."
            )
        elif latest_event and latest_event.get("event_type") == "candidate_snapshot":
            insights.append("A new candidate portfolio was accepted into the live audit stream and compared against the current feasible set.")
        insights.append(
            "Covariance is symmetrized, shrinkage-adjusted, and diagonally regularized before optimization to improve numerical stability and produce more reliable portfolio weights."
        )
        return insights

    @staticmethod
    def _normalize_method(method: str | None) -> str:
        normalized = str(method or "AUTO").strip().upper()
        return normalized if normalized in {"AUTO", "QP", "PSO", "GA", "HYBRID"} else "AUTO"

    def choose_method(self) -> str:
        return "HYBRID"

    def _selected_method(self) -> str:
        return self.choose_method() if self.method == "AUTO" else self.method

    def _raw_solve(self, method: str, event_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        if method == "QP":
            return solve_qp(self.problem, event_callback=event_callback)
        if method == "PSO":
            return solve_pso(self.problem, event_callback=event_callback)
        if method == "GA":
            return solve_ga(self.problem, event_callback=event_callback)
        return solve_hybrid(self.problem, event_callback=event_callback)

    def _solve_problem_variant(
        self,
        method: str,
        problem: PortfolioProblem,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if method == "QP":
            return solve_qp(problem, event_callback=event_callback)
        if method == "PSO":
            return solve_pso(problem, event_callback=event_callback)
        if method == "GA":
            return solve_ga(problem, event_callback=event_callback)
        return solve_hybrid(problem, event_callback=event_callback)

    def _model_output(self, method: str, raw: dict[str, Any]) -> dict[str, Any]:
        weights = np.asarray(raw["weights"], dtype=float)
        breakdown = raw.get("objective_breakdown", {})
        metrics = breakdown.get("metrics", {})
        expected_return = float(metrics.get("expected_return", np.dot(weights, self.problem.mu)))
        risk = float(metrics.get("risk", weights.T @ self.problem.cov @ weights))
        sharpe = float(expected_return / np.sqrt(risk)) if risk > 1e-12 else 0.0
        diversification = diversification_count(weights)
        avg_liquidity = float(metrics.get("average_liquidity", np.dot(weights, self.problem.liquidity)))
        feasibility = check_feasibility(weights, self.problem.cost, self.problem.budget, self.problem.max_allocation)
        explanation = self.explainer.explain_model(
            model_name=method,
            weights=weights,
            mu=self.problem.mu,
            cov=self.problem.cov,
            asset_names=self.problem.asset_names,
            liquidity=self.problem.liquidity,
            costs=self.problem.cost,
            lambda_value=self.problem.lambda_value,
            gamma=self.problem.cost_sensitivity,
            delta=self.problem.liquidity_weight,
            tail_weight=self.problem.tail_risk_weight,
            diversification_weight=self.problem.diversification_weight,
            concentration_penalty=self.problem.concentration_penalty,
            regime_aversion=self.problem.regime_aversion,
            tracking_penalty=self.problem.tracking_penalty,
            cvar_alpha=self.problem.cvar_alpha,
        )
        objective_breakdown = {
            key: float(value)
            for key, value in breakdown.items()
            if isinstance(value, (float, int, np.floating, np.integer)) and key != "total"
        }
        feasibility_gate = raw.get("feasibility_validation") or self._feasibility_gate(weights)
        regime_metrics = {
            "regime_names": list(self.problem.regime_names),
            "regime_probabilities": np.asarray(self.problem.regime_probabilities, dtype=float).tolist(),
            "regime_expected_returns": list(metrics.get("regime_expected_returns", [])),
            "regime_risks": list(metrics.get("regime_risks", [])),
            "loss_var_95": float(metrics.get("loss_var_95", 0.0)),
            "loss_cvar_95": float(metrics.get("loss_cvar_95", 0.0)),
            "tracking_error": float(metrics.get("tracking_error", 0.0)),
            "regime_dispersion": float(metrics.get("regime_dispersion", 0.0)),
            "regime_stress": float(metrics.get("regime_stress", 0.0)),
            "hybrid_stage_weights": raw.get("hybrid_stage_weights", {}),
        }
        return {
            "weights": weights.tolist(),
            "expected_return": expected_return,
            "risk": risk,
            "sharpe": sharpe,
            "diversification": diversification,
            "active_assets_target": int(self.problem.active_assets_target or len(weights)),
            "objective_value": float(raw["objective_value"]),
            "budget_used": float(np.dot(weights, self.problem.cost)),
            "average_liquidity": avg_liquidity,
            "valid": bool(feasibility["feasible"] and validate_weights(weights)),
            "objective_breakdown": objective_breakdown,
            "regime_metrics": regime_metrics,
            "feasibility_validation": feasibility_gate,
            "risk_target_gap": float(abs(risk - float(self.calibration_info.get("target_risk", TARGET_PORTFOLIO_RISK)))),
            "explanation": {
                "objective_function": explanation["objective_function"],
                "solver_steps": explanation["solver_steps"],
                "why_this_allocation": explanation["why_this_allocation"],
                "asset_justifications": explanation["asset_justifications"],
            },
        }

    def compare_methods(self) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for method in ["QP", "PSO", "GA", "HYBRID"]:
            outputs[method] = self._model_output(method, self._raw_solve(method))
        return outputs

    def _solve_all_raw(self) -> dict[str, dict[str, Any]]:
        return {method: self._raw_solve(method) for method in ["QP", "PSO", "GA", "HYBRID"]}

    def _decision_analysis(self, selected_weights: list[float] | np.ndarray) -> dict[str, Any]:
        range_payload = self.decision_agent.build(
            assets=assets_to_dicts(self.universe.assets),
            selected_weights=np.asarray(selected_weights, dtype=float).tolist(),
        )
        accepted = [row for row in range_payload["ranges"] if float(row.get("selected_weight", 0.0)) > 1e-6]
        rejected = [row for row in range_payload["ranges"] if float(row.get("selected_weight", 0.0)) <= 1e-6]
        return {
            "agent_name": range_payload["agent_name"],
            "methodology": [
                "Use only the minimum dataset features: mu, sigma, liquidity, and cost.",
                "Normalize those features and construct a transparent decision score.",
                "Infer weight bands and compare them with the final optimized decision variables x.",
                "Classify assets as accepted or rejected based on the final feasible solution.",
            ],
            "summary": range_payload["plain_english"],
            "records": [
                {
                    "name": row["name"],
                    "mu": row["mu"],
                    "sigma": row["sigma"],
                    "liquidity": row["liquidity"],
                    "cost": row["cost"],
                    "mu_norm": row["mu_norm"],
                    "risk_norm": row["sigma_norm"],
                    "liquidity_norm": row["liquidity_norm"],
                    "cost_norm": row["cost_norm"],
                    "decision_score": row["decision_score"],
                    "selected_weight": row["selected_weight"],
                    "rank": row["rank"],
                    "min_weight": row.get("min_weight"),
                    "max_weight": row.get("max_weight"),
                    "preferred_weight": row.get("preferred_weight"),
                    "materiality": row.get("materiality"),
                    "reason": row.get("reason"),
                }
                for row in range_payload["ranges"]
            ],
            "decision_formula": range_payload["decision_formula"],
            "ranking_rule": range_payload["ranking_rule"],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_assets": [row["name"] for row in accepted],
            "rejected_assets": [row["name"] for row in rejected],
            "minimum_features": ["mu", "sigma", "liquidity", "cost"],
        }

    def _trace_meta(self, method: str, raw: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        return build_trace_transparency(
            solver_history=raw.get("history", []),
            problem=self.problem,
            model_name=method,
            selected_weights=output["weights"],
            selected_score=output["objective_value"],
            candidates=[
                {
                    "weights": [1.0 if idx == asset_idx else 0.0 for idx in range(len(output["weights"]))],
                    "objective_value": float(np.asarray(self.problem.mu)[asset_idx] * -1.0),
                    "model": self.problem.asset_names[asset_idx],
                }
                for asset_idx in range(len(output["weights"]))
            ],
        )

    def optimize(self) -> dict[str, Any]:
        selected_method = self._selected_method()
        selected_output = self._model_output(selected_method, self._raw_solve(selected_method))
        return {
            "selected_model": selected_method,
            "selected_output": selected_output,
            "assets": assets_to_dicts(self.universe.assets),
        }

    def backtest(self, method: str | None = None) -> dict[str, Any]:
        if self.problem.historical_returns is None:
            raise ValueError("historical_returns are required for out-of-sample backtesting.")
        selected_method = self._normalize_method(method) if method else "QP"
        if selected_method == "AUTO":
            selected_method = self._selected_method()
        result = rolling_backtest(
            self.problem,
            self.problem.historical_returns,
            method=selected_method,
            lookback=int(self.problem.backtest_lookback),
            macro_signal=self.problem.macro_signal,
            solver_callback=self._solve_problem_variant,
        )
        result["selected_model"] = selected_method
        return result

    def multi_period_optimize(self, method: str | None = None) -> dict[str, Any]:
        if self.problem.historical_returns is None:
            raise ValueError("historical_returns are required for multi-period optimization.")
        selected_method = self._normalize_method(method) if method else "QP"
        if selected_method == "AUTO":
            selected_method = self._selected_method()
        result = multi_period_receding_horizon(
            self.problem,
            self.problem.historical_returns,
            horizon=int(self.problem.planning_horizon),
            method=selected_method,
            macro_signal=self.problem.macro_signal,
            solver_callback=self._solve_problem_variant,
        )
        result["selected_model"] = selected_method
        return result

    def _comparison_table(self, model_outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "model": model,
                "expected_return": output["expected_return"],
                "risk": output["risk"],
                "sharpe": output["sharpe"],
                "diversification": output["diversification"],
            }
            for model, output in model_outputs.items()
        ]

    def _parameter_payload(self) -> dict[str, float]:
        return {
            "lambda": float(self.request.lambda_value),
            "alpha": float(self.request.max_allocation),
            "budget": float(self.request.budget),
            "gamma": float(self.request.cost_sensitivity),
            "delta": float(self.request.liquidity_weight),
            "theta": float(self.request.tail_risk_weight),
            "kappa": float(self.request.diversification_weight),
            "psi": float(self.request.concentration_penalty),
            "rho": float(self.request.regime_aversion),
            "omega": float(self.request.tracking_penalty),
            "phi": float(self.request.covariance_shrinkage),
            "alpha_cvar": float(self.request.cvar_alpha),
            "return_prior_strength": float(self.request.return_prior_strength),
            "bl_tau": float(self.request.bl_tau),
            "bl_view_confidence": float(self.request.bl_view_confidence),
        }

    def _mathematical_model_payload(
        self,
        model_outputs: dict[str, dict[str, Any]],
        raw_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        selected_model = self._selected_method()
        selected_output = model_outputs[selected_model]
        objective_terms = {model: output["objective_breakdown"] for model, output in model_outputs.items()}
        regime_points = [
            {
                "model": model,
                "bear_return": float(output["regime_metrics"]["regime_expected_returns"][0]),
                "neutral_return": float(output["regime_metrics"]["regime_expected_returns"][1]),
                "bull_return": float(output["regime_metrics"]["regime_expected_returns"][2]),
                "loss_cvar_95": float(output["regime_metrics"]["loss_cvar_95"]),
                "risk": float(output["risk"]),
            }
            for model, output in model_outputs.items()
            if len(output["regime_metrics"].get("regime_expected_returns", [])) >= 3
        ]
        return {
            "multi_objective_formulation": "max ( R(x), -sigma(x), -CVaR_alpha(x), L(x) )",
            "normalization_equation": "f_tilde_i(x) = ( f_i(x) - min f_i ) / ( max f_i - min f_i )",
            "scalarization_equation": "J_scalar(x) = sum_i w_i f_tilde_i(x)",
            "objective_equation": (
                "J(x)=x^T Sigma_hat x-lambda mu^T x+gamma c^T x-delta l^T x+theta CVaR_alpha(x)"
                "+kappa( sum x_i^2-1/K )^2+psi sum x_i^2+rho Omega_regime(x)+omega||x-x_ref||_2^2"
            ),
            "objective_latex": (
                r"J(x)=x^\top \hat{\Sigma} x-\lambda \mu^\top x+\gamma c^\top x-\delta \ell^\top x"
                r"+\theta \operatorname{CVaR}_{\alpha}(x)+\kappa\left(\sum_i x_i^2-\frac{1}{K}\right)^2"
                r"+\psi \sum_i x_i^2+\rho \Omega_{\mathrm{regime}}(x)+\omega \lVert x-x_{\mathrm{ref}}\rVert_2^2"
            ),
            "return_engine_equation": (
                "mu_BL=[(tau Sigma)^-1+P^T Omega^-1 P]^-1[(tau Sigma)^-1 pi+P^T Omega^-1 q], "
                "with Bayesian shrinkage used to form the view vector q"
            ),
            "black_litterman_latex": (
                r"\mu_{BL}=\left[(\tau\Sigma)^{-1}+P^\top \Omega^{-1}P\right]^{-1}"
                r"\left[(\tau\Sigma)^{-1}\pi+P^\top \Omega^{-1}q\right]"
            ),
            "qp_baseline_equation": "J_QP_seed(x)=x^T Sigma_hat x-lambda mu^T x",
            "exact_cvar_seed_equation": (
                "min_{x,eta,u} x^T Sigma_hat x-lambda mu^T x+theta[eta+(1/(1-alpha)N)sum u_t]+..."
                " s.t. u_t>=-r_t^T x-eta, u_t>=0, c^T x<=B, 0<=x_i<=alpha_bar, sum x_i=1"
            ),
            "hybrid_equation": (
                "J_H(x)=J(x)+omega[ w_Q||x-x_Q||_2^2+w_P||x-x_P||_2^2+w_G||x-x_G||_2^2 ]"
            ),
            "covariance_equation": "Sigma_hat=(1-phi)Sigma+phi diag(Sigma)",
            "cvar_equation": "CVaR_alpha(x)=E[L(x)|L(x)>=VaR_alpha(L(x))], L(x)=-R(x)",
            "regime_equation": "Omega_regime(x)=0.65 Var_p(mu_s^T x)+0.35 E_p[x^T Sigma_s x]",
            "dynamic_policy_equation": "x_t*=arg min_x J_t(x) over a rolling lookback window with receding-horizon re-optimization",
            "scalarization_note": "Scalarization converts the multi-objective portfolio problem into a single-objective optimization problem.",
            "discrete_real_estate_model": "x_i in {0,1}  (true indivisible real-estate formulation)",
            "relaxed_real_estate_model": "x_i in [0,1]  (continuous relaxation used for tractable optimization)",
            "constraints": [
                "c^T x <= B",
                "sum_i x_i = 1",
                "0 <= x_i <= alpha",
                "x_i >= 0",
                "active asset count approx equals K through projection",
                "portfolio is evaluated across Bear, Neutral, and Bull regimes",
            ],
            "advanced_capabilities": {
                "exact_cvar_seed": bool(self.problem.exact_cvar_seed),
                "exact_discrete_selection": bool(self.problem.exact_discrete_selection),
                "historical_regime_calibration": bool(self.problem.historical_returns is not None),
                "rolling_backtest_available": bool(self.problem.historical_returns is not None),
                "multi_period_reoptimization_available": bool(self.problem.historical_returns is not None),
                "bayesian_black_litterman_returns": True,
            },
            "return_engine": {
                "mode": str(self.return_engine_info.get("engine", self.request.return_engine)),
                "prior_strength": float(self.request.return_prior_strength),
                "tau": float(self.request.bl_tau),
                "view_confidence": float(self.request.bl_view_confidence),
                "raw_mu": np.asarray(self.universe.mu, dtype=float).tolist(),
                "posterior_mu": np.asarray(self.problem.mu, dtype=float).tolist(),
                "prior_mu": self.return_engine_info.get("prior_mu"),
                "empirical_mu": self.return_engine_info.get("empirical_mu"),
                "market_weights": self.return_engine_info.get("market_weights"),
            },
            "theoretical_properties": [
                {
                    "title": "Proposition 1 (Stability)",
                    "statement": (
                        "Shrinkage covariance with diagonal regularization makes Sigma_hat positive definite in the implemented model, "
                        "so the quadratic risk block is strictly convex."
                    ),
                    "interpretation": (
                        "This improves numerical stability, keeps the Hessian well-conditioned, and supports reliable local optimization."
                    ),
                },
                {
                    "title": "Proposition 2 (Hybrid Convergence)",
                    "statement": (
                        "Under bounded objective values, finite projected search space, and best-so-far retention in the hybrid pipeline, "
                        "the QP-PSO-GA stack admits convergence to a stationary candidate point."
                    ),
                    "interpretation": (
                        "This is a heuristic stationary-point claim for the hybrid search, not a guarantee of global optimality."
                    ),
                },
                {
                    "title": "Proposition 3 (Return Stabilization)",
                    "statement": (
                        "Bayesian/Black-Litterman return estimation shrinks noisy asset-level return views toward a "
                        "market-implied prior, reducing estimation error sensitivity in mu."
                    ),
                    "interpretation": (
                        "This does not guarantee higher realized return, but it typically improves robustness of the "
                        "optimizer by preventing unstable weights driven by noisy point estimates."
                    ),
                },
            ],
            "coherent_risk_theory": {
                "headline": "CVaR is treated as a coherent risk measure.",
                "properties": [
                    "subadditivity",
                    "monotonicity",
                    "translation invariance",
                    "positive homogeneity",
                ],
                "research_note": (
                    "Using CVaR instead of variance alone strengthens theoretical credibility in downside-risk modeling."
                ),
            },
            "decision_variables": [
                {"symbol": "x_i", "meaning": "portfolio weight of asset i"},
                {"symbol": "Sigma_hat", "meaning": "shrinkage-adjusted covariance matrix"},
                {"symbol": "eta", "meaning": "VaR/CVaR threshold variable used in the tail-risk interpretation"},
                {"symbol": "v_i^t", "meaning": "particle velocity in PSO"},
                {"symbol": "p_i^t", "meaning": "particle best-known position"},
                {"symbol": "g^t", "meaning": "global best swarm position"},
                {"symbol": "P_t", "meaning": "GA population at generation t"},
                {"symbol": "x_Q,x_P,x_G", "meaning": "QP, PSO, and GA stage solutions"},
                {"symbol": "w_Q,w_P,w_G", "meaning": "adaptive hybrid blend weights"},
                {"symbol": "K", "meaning": "target number of active assets"},
            ],
            "parameter_table": [
                {"symbol": "lambda", "name": "risk aversion", "value": float(self.request.lambda_value)},
                {"symbol": "B", "name": "budget limit", "value": float(self.request.budget)},
                {"symbol": "alpha_bar", "name": "max per-asset allocation", "value": float(self.request.max_allocation)},
                {"symbol": "delta", "name": "liquidity reward", "value": float(self.request.liquidity_weight)},
                {"symbol": "gamma", "name": "transaction-cost penalty", "value": float(self.request.cost_sensitivity)},
                {"symbol": "theta", "name": "tail-risk penalty", "value": float(self.request.tail_risk_weight)},
                {"symbol": "kappa", "name": "diversification penalty", "value": float(self.request.diversification_weight)},
                {"symbol": "psi", "name": "concentration penalty", "value": float(self.request.concentration_penalty)},
                {"symbol": "rho", "name": "regime-aversion weight", "value": float(self.request.regime_aversion)},
                {"symbol": "omega", "name": "tracking/consensus penalty", "value": float(self.request.tracking_penalty)},
                {"symbol": "phi", "name": "covariance shrinkage", "value": float(self.request.covariance_shrinkage)},
                {"symbol": "alpha", "name": "CVaR confidence", "value": float(self.request.cvar_alpha)},
            ],
            "objective_terms_by_model": objective_terms,
            "selected_objective_terms": selected_output["objective_breakdown"],
            "regime_table": [
                {
                    "Regime": name,
                    "Probability": float(prob),
                    "Selected Return": float(selected_output["regime_metrics"]["regime_expected_returns"][idx]),
                    "Selected Risk": float(selected_output["regime_metrics"]["regime_risks"][idx]),
                }
                for idx, (name, prob) in enumerate(zip(self.problem.regime_names, self.problem.regime_probabilities))
            ],
            "regime_3d_points": regime_points,
            "hybrid_stage_weights": model_outputs["HYBRID"]["regime_metrics"].get("hybrid_stage_weights", {}),
            "risk_target": float(self.calibration_info.get("target_risk", TARGET_PORTFOLIO_RISK)),
            "risk_calibration": dict(self.calibration_info),
            "feasibility_gate": {
                model: dict(output.get("feasibility_validation") or {})
                for model, output in model_outputs.items()
            },
            "pipeline_graph": {
                "nodes": [
                    {"id": "A", "label": "A. Asset Specs + Constraints", "group": "input"},
                    {"id": "B", "label": "B. Expected Returns", "group": "estimation"},
                    {"id": "D", "label": "D. Covariance Regularization", "group": "risk"},
                    {"id": "C", "label": "C. Regime Calibration", "group": "risk"},
                    {"id": "J", "label": "J(x) Objective Assembly", "group": "objective"},
                    {"id": "QP", "label": "QP Seed", "group": "solver"},
                    {"id": "PSO", "label": "PSO Refinement", "group": "solver"},
                    {"id": "GA", "label": "GA Exploration", "group": "solver"},
                    {"id": "E", "label": "E. Evaluation + Calibration", "group": "evaluation"},
                ],
                "links": [
                    {"source": "A", "target": "B", "label": "input tensors", "value": 1.0},
                    {"source": "A", "target": "D", "label": "covariance seed", "value": 1.0},
                    {"source": "A", "target": "C", "label": "regime prior", "value": 1.0},
                    {"source": "B", "target": "J", "label": "posterior mu", "value": 1.0},
                    {"source": "D", "target": "J", "label": "Sigma_hat", "value": 1.0},
                    {"source": "C", "target": "J", "label": "Omega_regime", "value": 1.0},
                    {"source": "J", "target": "QP", "label": "convex seed solve", "value": 0.9},
                    {"source": "QP", "target": "PSO", "label": "warm start", "value": 0.8},
                    {"source": "PSO", "target": "GA", "label": "candidate transfer", "value": 0.7},
                    {"source": "GA", "target": "E", "label": "final candidates", "value": 1.0},
                    {"source": "QP", "target": "E", "label": "benchmark branch", "value": 0.4},
                    {"source": "PSO", "target": "E", "label": "benchmark branch", "value": 0.4},
                ],
            },
            "benchmark_table": [
                {
                    "metric": "Risk",
                    "achieved": float(selected_output["risk"]),
                    "target": float(self.calibration_info.get("target_risk", TARGET_PORTFOLIO_RISK)),
                    "notes": "Quadratic risk objective target",
                },
                {
                    "metric": "Expected Return",
                    "achieved": float(selected_output["expected_return"]),
                    "target": float(np.mean(np.asarray(self.problem.mu, dtype=float))),
                    "notes": "Posterior return engine baseline",
                },
                {
                    "metric": "Loss CVaR 95",
                    "achieved": float(selected_output["regime_metrics"].get("loss_cvar_95", 0.0)),
                    "target": float(self.request.tail_risk_weight) * 0.02,
                    "notes": "Tail-risk stress tolerance proxy",
                },
                {
                    "metric": "Active Assets",
                    "achieved": float(selected_output["diversification"]),
                    "target": float(self.problem.active_assets_target),
                    "notes": "Diversification target K",
                },
                {
                    "metric": "Budget Used",
                    "achieved": float(selected_output["budget_used"]),
                    "target": float(self.request.budget),
                    "notes": "Hard allocation budget",
                },
            ],
            "key_concepts_and_benchmarks": [
                "CVaR threshold benchmark (loss tail control)",
                "Distinct models benchmark (QP, PSO, GA all solved)",
                "Hybrid blend benchmark (stage-score weighted fusion)",
                "Feasibility benchmark (budget, simplex, allocation caps)",
            ],
            "workflow_steps": [
                "Assemble mu, liquidity, cost, covariance, regime tensors, and Monte Carlo scenario returns.",
                "Estimate expected returns with the Bayesian/Black-Litterman engine before solving the portfolio problem.",
                "If historical returns are supplied, calibrate Bear/Neutral/Bull regimes directly from macro-linked return history.",
                "Shrink covariance and regularize the quadratic form.",
                f"Calibrate lambda and shrinkage phi to target a quadratic risk of {TARGET_PORTFOLIO_RISK:.2%} before the full model stack runs.",
                "Solve the convex seed problem with exact empirical CVaR auxiliary variables and optional discrete subset screening.",
                "Refine the common objective with PSO and GA.",
                "Project every candidate onto the feasible simplex with an active-asset target.",
                "Audit accepted and rejected candidates and render 2D/3D mathematical diagnostics.",
            ],
            "selected_solver_trace_length": len(raw_results[selected_model].get("history", [])),
        }

    def _intelligence_payload(
        self,
        model_outputs: dict[str, dict[str, Any]],
        raw_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        explanations = self.explainer.explain_all(
            model_outputs=model_outputs,
            assets=assets_to_dicts(self.universe.assets),
            covariance_matrix=self.universe.cov_matrix,
            parameters=self._parameter_payload(),
        )
        return {
            "decision_variables": self.explainer.decision_variables(
                self.request.lambda_value,
                self.request.cost_sensitivity,
                self.request.liquidity_weight,
                tail_weight=self.request.tail_risk_weight,
                diversification_weight=self.request.diversification_weight,
                concentration_penalty=self.request.concentration_penalty,
                regime_aversion=self.request.regime_aversion,
                tracking_penalty=self.request.tracking_penalty,
                covariance_shrinkage=self.request.covariance_shrinkage,
                cvar_alpha=self.request.cvar_alpha,
            ),
            "execution_log": [
                {"stage": "Dataset", "detail": self.dataset},
                {"stage": "Assets", "detail": f"{len(self.universe.assets)} assets generated or loaded"},
                {
                    "stage": "Tensor Build",
                    "detail": (
                        f"{len(self.problem.regime_names)} regime tensors and {len(self.problem.scenario_returns)} tail-risk scenarios assembled"
                        if self.problem.historical_returns is None
                        else f"{len(self.problem.regime_names)} regimes calibrated from {len(self.problem.historical_returns)} historical observations"
                    ),
                },
                {"stage": "Return Engine", "detail": str(self.return_engine_info.get("engine", self.request.return_engine))},
                {"stage": "Diversification Target", "detail": f"{self.problem.active_assets_target} active assets"},
                {"stage": "Exact Seed", "detail": "Enabled" if self.problem.exact_cvar_seed else "Disabled"},
                {"stage": "Method", "detail": self._selected_method()},
                {"stage": "Output", "detail": "All model portfolios solved and validated"},
            ],
            "execution_flow": self.explainer.execution_flow(),
            "model_explanations": {
                model: {
                    "objective_function": payload["objective_function"],
                    "solver_steps": payload["solver_steps"],
                    "why_this_allocation": payload["why_this_allocation"],
                    "asset_justifications": payload["asset_justifications"],
                }
                for model, payload in explanations.items()
            },
            "transparency": {},
            "decision_variable_analysis": {},
            "mathematical_model": self._mathematical_model_payload(model_outputs, raw_results),
        }

    def _objective_surface(self, method: str, weights: np.ndarray) -> dict[str, Any]:
        ranking = np.argsort(weights)[::-1]
        idx_a = int(ranking[0])
        idx_b = int(ranking[1 if len(ranking) > 1 else 0])
        base = np.asarray(weights, dtype=float)
        grid = np.linspace(0.0, 1.0, 18)
        expected_return_values: list[list[float]] = []
        risk_values: list[list[float]] = []
        cvar_values: list[list[float]] = []
        for wa in grid:
            return_row: list[float] = []
            risk_row: list[float] = []
            cvar_row: list[float] = []
            for wb in grid:
                if wa + wb > 1.0:
                    return_row.append(float("nan"))
                    risk_row.append(float("nan"))
                    cvar_row.append(float("nan"))
                    continue
                candidate = base.copy() * max(0.0, 1.0 - wa - wb)
                candidate[idx_a] = wa
                candidate[idx_b] = wb
                total = float(np.sum(candidate))
                if total <= 1e-12:
                    return_row.append(float("nan"))
                    risk_row.append(float("nan"))
                    cvar_row.append(float("nan"))
                    continue
                candidate = candidate / total
                score_breakdown = objective_components(candidate, self.problem, method)
                return_row.append(float(score_breakdown["metrics"]["expected_return"]))
                risk_row.append(float(score_breakdown["metrics"]["risk"]))
                cvar_row.append(float(score_breakdown["metrics"]["loss_cvar_95"]))
            expected_return_values.append(return_row)
            risk_values.append(risk_row)
            cvar_values.append(cvar_row)
        selected_breakdown = objective_components(weights, self.problem, method)
        return {
            "asset_x": "Expected Return",
            "asset_y": "Risk",
            "x": expected_return_values,
            "y": risk_values,
            "z": cvar_values,
            "selected_point": {
                "x": float(selected_breakdown["metrics"]["expected_return"]),
                "y": float(selected_breakdown["metrics"]["risk"]),
                "z": float(selected_breakdown["metrics"]["loss_cvar_95"]),
            },
        }

    def validation(self, model_outputs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        outputs = model_outputs or self.compare_methods()
        warnings: list[str] = []
        distinct_models = compare_weight_sets(outputs)
        if not distinct_models:
            warnings.append("Model allocations are too similar; differentiation threshold was not met.")
        if len(self.universe.assets) < 10:
            warnings.append("Asset count fell below the required minimum of 10.")

        stochastic_a = solve_pso(PortfolioProblem(**{**self.problem.to_dict(), "seed": self.request.seed + 1}))
        stochastic_b = solve_ga(PortfolioProblem(**{**self.problem.to_dict(), "seed": self.request.seed + 2}))
        pso_delta = float(np.linalg.norm(np.asarray(outputs["PSO"]["weights"]) - np.asarray(stochastic_a["weights"])))
        ga_delta = float(np.linalg.norm(np.asarray(outputs["GA"]["weights"]) - np.asarray(stochastic_b["weights"])))
        stochastic_models_vary = (pso_delta > 1e-3) and (ga_delta > 1e-3)
        if not stochastic_models_vary:
            warnings.append("Stochastic models did not move enough across seeds.")

        feasibility_rates = [
            float((output.get("feasibility_validation") or {}).get("success_rate", 0.0))
            for output in outputs.values()
        ]
        feasibility_success_rate = float(min(feasibility_rates)) if feasibility_rates else 0.0
        if feasibility_success_rate < 1.0:
            warnings.append("Feasibility gate did not report a 100% success rate across the model set.")
        achieved_risk = float(outputs[self._selected_method()]["risk"])
        risk_target_gap = abs(achieved_risk - float(self.calibration_info.get("target_risk", TARGET_PORTFOLIO_RISK)))

        return {
            "validation": {
                "distinct_models": distinct_models,
                "asset_count_valid": 10 <= len(self.universe.assets) <= 15,
                "stochastic_models_vary": stochastic_models_vary,
                "feasibility_success_rate": feasibility_success_rate,
                "risk_target": float(self.calibration_info.get("target_risk", TARGET_PORTFOLIO_RISK)),
                "achieved_risk": achieved_risk,
                "risk_target_gap": float(risk_target_gap),
                "warnings": warnings,
            }
        }

    def _dashboard_bundle_from_raw(self, raw_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        model_outputs = {method: self._model_output(method, raw) for method, raw in raw_results.items()}
        selected_model = self._selected_method()
        selected_output = model_outputs[selected_model]
        intelligence = self._intelligence_payload(model_outputs, raw_results)
        intelligence["decision_variable_analysis"] = self._decision_analysis(model_outputs[selected_model]["weights"])
        intelligence["transparency"] = {
            method: {
                **self.transparency_agent.build_solver_audit(
                    method=method,
                    trace=raw_results[method].get("history", []),
                    selected_output=model_outputs[method],
                    asset_count=len(self.universe.assets),
                    active_assets_target=self.problem.active_assets_target,
                    objective_surface=self._objective_surface(method, np.asarray(model_outputs[method]["weights"], dtype=float)),
                ),
                "solver_manifold": self._solver_manifold(raw_results[method].get("candidate_history", [])),
                "candidate_history": raw_results[method].get("candidate_history", []),
                **{
                    key: value
                    for key, value in self._trace_meta(method, raw_results[method], model_outputs[method]).items()
                    if key in {"accepted_count", "rejected_count", "candidate_summary", "minimum_characteristics", "decision_variable_ranges", "rejection_reasons"}
                },
            }
            for method in ["QP", "PSO", "GA", "HYBRID"]
        }
        return {
            "assets": assets_to_dicts(self.universe.assets),
            "covariance_matrix": self.universe.cov_matrix.tolist(),
            "selected_model": selected_model,
            "selected_output": selected_output,
            "model_outputs": model_outputs,
            "comparison_table": self._comparison_table(model_outputs),
            "efficient_frontier": frontier_points(self.problem, points=8),
            "monte_carlo": {
                model: monte_carlo_summary(
                    np.asarray(output["weights"], dtype=float),
                    self.problem.mu,
                    self.problem.cov,
                    paths=self.request.monte_carlo_paths,
                    seed=self.request.seed + index * 19,
                )
                for index, (model, output) in enumerate(model_outputs.items())
            },
            "sensitivity_analysis": sensitivity_points(
                self.problem,
                method="QP",
                lambda_values=np.linspace(max(0.2, self.request.lambda_value * 0.5), self.request.lambda_value * 1.8, 5),
            ),
            "validation": self.validation(model_outputs)["validation"],
            "intelligence": intelligence,
        }

    def dashboard_bundle(self) -> dict[str, Any]:
        return self._dashboard_bundle_from_raw(self._solve_all_raw())

    def dashboard_bundle_live(
        self,
        progress_callback: Callable[[str, float, dict[str, Any], str], None] | None = None,
    ) -> dict[str, Any]:
        models = ["QP", "PSO", "GA", "HYBRID"]
        selected_model = self._selected_method()
        raw_results: dict[str, dict[str, Any]] = {}
        model_outputs: dict[str, dict[str, Any]] = {}
        live_traces: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
        latest_candidate_weights: dict[str, list[float]] = {}
        stage_rows = [
            {"name": "Universe", "status": "completed", "progress": 1.0, "duration_sec": 0.0, "detail": f"{len(self.universe.assets)} assets ready"},
            *[
                {"name": model, "status": "pending", "progress": 0.0, "duration_sec": 0.0, "detail": "Waiting"}
                for model in models
            ],
            {"name": "Packaging", "status": "pending", "progress": 0.0, "duration_sec": 0.0, "detail": "Waiting"},
        ]

        def publish(stage: str, progress: float, message: str, current_model: str | None = None, latest_event: dict[str, Any] | None = None) -> None:
            if progress_callback is None:
                return
            focus_model = current_model or selected_model
            current_output = model_outputs.get(focus_model)
            decision_analysis = None
            candidate_summary = None
            minimum_characteristics = None
            rejection_reasons = None
            decision_variable_ranges = None
            current_surface = None
            if current_output is not None and focus_model in raw_results:
                trace_meta = self._trace_meta(focus_model, raw_results[focus_model], current_output)
                candidate_summary = trace_meta.get("candidate_summary")
                minimum_characteristics = trace_meta.get("minimum_characteristics")
                rejection_reasons = trace_meta.get("rejection_reasons")
                decision_variable_ranges = trace_meta.get("decision_variable_ranges")
                decision_analysis = self._decision_analysis(current_output["weights"])
                current_surface = self._objective_surface(focus_model, np.asarray(current_output["weights"], dtype=float))
            elif focus_model in latest_candidate_weights:
                decision_analysis = self._decision_analysis(latest_candidate_weights[focus_model])
            trace_rows = live_traces.get(focus_model, [])
            payload = {
                "selected_model": selected_model,
                "current_model": focus_model,
                "message": message,
                "assets": assets_to_dicts(self.universe.assets),
                "model_outputs": model_outputs,
                "current_output": current_output,
                "parameters": self._parameter_payload(),
                "comparison_table": self._comparison_table(model_outputs),
                "stage_rows": stage_rows,
                "current_trace": trace_rows,
                "candidate_history": raw_results.get(focus_model, {}).get("candidate_history", []),
                "current_trace_payload": build_3d_trace_payload(model=focus_model, trace=trace_rows, title=f"{focus_model} Live Trace"),
                "current_surface": current_surface,
                "current_manifold": self._solver_manifold(raw_results.get(focus_model, {}).get("candidate_history", [])),
                "decision_analysis": decision_analysis,
                "candidate_summary": candidate_summary,
                "minimum_characteristics": minimum_characteristics,
                "decision_variable_ranges": decision_variable_ranges,
                "rejection_reasons": rejection_reasons,
                "latest_event": latest_event,
                "step_insights": self._live_step_insight(focus_model, stage, latest_event, current_output),
                "risk_calibration": dict(self.calibration_info),
            }
            progress_callback(stage, progress, payload, message)

        publish("initializing", 0.03, "Asset universe and optimization problem prepared.")

        for model_index, model in enumerate(models):
            stage_slot = model_index + 1
            stage_rows[stage_slot]["status"] = "running"
            stage_rows[stage_slot]["detail"] = f"{model} solving"
            stage_rows[stage_slot]["progress"] = 0.0
            stage_started_at = perf_counter()
            model_base_progress = 0.08 + model_index * 0.18

            def _event_handler(event: dict[str, Any], _model: str = model, _stage_slot: int = stage_slot, _base: float = model_base_progress) -> None:
                event_type = str(event.get("event_type", ""))
                if event_type == "trace_point":
                    live_traces[_model].append(
                        {
                            "iteration": float(event.get("iteration", len(live_traces[_model]))),
                            "risk": float(event.get("risk", 0.0)),
                            "objective_value": float(event.get("objective_value", 0.0)),
                            "expected_return": float(event.get("expected_return", 0.0)),
                            "stage": str(event.get("stage", _model)),
                        }
                    )
                elif event_type == "candidate_snapshot":
                    latest_candidate_weights[_model] = list(event.get("weights", []))

                iteration_value = float(event.get("iteration", len(live_traces[_model]) or 0.0))
                if _model == "QP":
                    stage_progress = min(0.95, iteration_value / 20.0)
                elif _model in {"PSO", "GA"}:
                    stage_progress = min(0.95, (iteration_value + 1.0) / 50.0)
                else:
                    stage_progress = min(0.95, max(0.2, (iteration_value + 1.0) / 10.0))
                stage_rows[_stage_slot]["progress"] = stage_progress
                stage_rows[_stage_slot]["detail"] = str(event.get("stage", _model))
                publish(
                    str(event.get("stage", _model)),
                    min(0.94, _base + 0.18 * stage_progress),
                    f"{_model} backend stage: {event.get('stage', _model)}",
                    current_model=_model,
                    latest_event=event,
                )

            raw_results[model] = self._raw_solve(model, event_callback=_event_handler)
            model_outputs[model] = self._model_output(model, raw_results[model])
            stage_rows[stage_slot]["status"] = "completed"
            stage_rows[stage_slot]["progress"] = 1.0
            stage_rows[stage_slot]["duration_sec"] = perf_counter() - stage_started_at
            stage_rows[stage_slot]["detail"] = f"{model} completed"
            publish(
                f"{model.lower()}_completed",
                min(0.95, model_base_progress + 0.18),
                f"{model} completed and accepted/rejected assets were updated.",
                current_model=model,
            )

        stage_rows[-1]["status"] = "running"
        stage_rows[-1]["progress"] = 0.5
        stage_rows[-1]["detail"] = "Packaging dashboard outputs"
        publish("packaging", 0.97, "Running Monte Carlo, frontier analysis, and explainability packaging.", current_model=selected_model)
        dashboard = self._dashboard_bundle_from_raw(raw_results)
        stage_rows[-1]["status"] = "completed"
        stage_rows[-1]["progress"] = 1.0
        stage_rows[-1]["detail"] = "Dashboard bundle complete"
        dashboard["live_console"] = {
            "selected_model": selected_model,
            "current_model": selected_model,
            "stage_rows": stage_rows,
            "current_trace": raw_results[selected_model].get("history", []),
            "candidate_history": raw_results[selected_model].get("candidate_history", []),
            "current_trace_payload": build_3d_trace_payload(model=selected_model, trace=raw_results[selected_model].get("history", []), title=f"{selected_model} Live Trace"),
            "current_surface": self._objective_surface(selected_model, np.asarray(dashboard["selected_output"]["weights"], dtype=float)),
            "current_manifold": self._solver_manifold(raw_results[selected_model].get("candidate_history", [])),
            "decision_analysis": dashboard["intelligence"]["decision_variable_analysis"],
            "risk_calibration": dict(self.calibration_info),
        }
        publish("completed", 1.0, "Optimization workflow completed.", current_model=selected_model)
        return dashboard


__all__ = ["PortfolioOptimizer", "PortfolioRequest"]
