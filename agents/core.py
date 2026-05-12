"""
Multi-agent portfolio intelligence components.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
import warnings

import cvxpy as cp
import numpy as np

from src.ga import optimize_with_ga
from src.pso import optimize_with_pso


class ScenarioAgent:
    def __init__(self) -> None:
        self._lambda_map = {
            "safe": 1.0,
            "low": 1.0,
            "balanced": 0.5,
            "medium": 0.5,
            "aggressive": 0.1,
            "high": 0.1,
        }
        self._scenario_aliases = {
            "positive": "positive",
            "bull": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "bear": "negative",
        }
        self._goal_aliases = {
            "growth": "growth",
            "max growth": "growth",
            "balanced": "balanced",
            "income": "income",
            "stable income": "income",
        }

    def normalize_risk(self, value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "safe": "safe",
            "low": "safe",
            "balanced": "balanced",
            "medium": "balanced",
            "aggressive": "aggressive",
            "high": "aggressive",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported risk level: {value}")
        return aliases[normalized]

    def normalize_goal(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in self._goal_aliases:
            raise ValueError(f"Unsupported investment goal: {value}")
        return self._goal_aliases[normalized]

    def normalize_scenario(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in self._scenario_aliases:
            raise ValueError(f"Unsupported market scenario: {value}")
        return self._scenario_aliases[normalized]

    def normalize_method(self, value: str) -> str:
        aliases = {"auto": "AUTO", "qp": "QP", "pso": "PSO", "ga": "GA", "hybrid": "HYBRID"}
        normalized = value.strip().lower()
        if normalized not in aliases:
            raise ValueError(f"Unsupported optimization method: {value}")
        return aliases[normalized]

    def lambda_from_risk(self, risk_level: str) -> float:
        return float(self._lambda_map[risk_level.strip().lower()])

    def transform(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        liquidity: np.ndarray,
        sector_labels: np.ndarray,
        risk_level: str,
        investment_goal: str,
        scenario: str,
    ) -> dict[str, Any]:
        risk_level = self.normalize_risk(risk_level)
        investment_goal = self.normalize_goal(investment_goal)
        scenario = self.normalize_scenario(scenario)

        mu_adj = np.asarray(mu, dtype=float).copy()
        sigma_adj = np.asarray(sigma, dtype=float).copy()
        liquidity_adj = np.asarray(liquidity, dtype=float).copy()
        volatility = np.sqrt(np.maximum(np.diag(sigma_adj), 1e-12))

        if risk_level == "aggressive" or investment_goal == "growth":
            top_return = mu_adj >= np.quantile(mu_adj, 0.70)
            mu_adj[top_return] *= 1.18
            mu_adj[~top_return] *= 0.97
        elif risk_level == "safe" or investment_goal == "income":
            high_risk = volatility >= np.quantile(volatility, 0.65)
            low_risk = volatility <= np.quantile(volatility, 0.35)
            mu_adj[high_risk] *= 0.78
            mu_adj[low_risk] *= 1.07
            liquidity_adj[high_risk] *= 0.95
        else:
            high_quality = (mu_adj >= np.quantile(mu_adj, 0.55)) & (volatility <= np.quantile(volatility, 0.55))
            mu_adj[high_quality] *= 1.08

        scenario_return = {
            "positive": {"Residential": 1.08, "Commercial": 1.16, "Land": 1.24},
            "neutral": {"Residential": 1.00, "Commercial": 1.00, "Land": 1.00},
            "negative": {"Residential": 0.92, "Commercial": 0.80, "Land": 0.72},
        }
        scenario_risk = {
            "positive": {"Residential": 0.92, "Commercial": 0.88, "Land": 0.85},
            "neutral": {"Residential": 1.00, "Commercial": 1.00, "Land": 1.00},
            "negative": {"Residential": 1.08, "Commercial": 1.18, "Land": 1.28},
        }
        scenario_liquidity = {
            "positive": {"Residential": 1.04, "Commercial": 1.07, "Land": 1.02},
            "neutral": {"Residential": 1.00, "Commercial": 1.00, "Land": 1.00},
            "negative": {"Residential": 0.96, "Commercial": 0.92, "Land": 0.86},
        }

        sector_scale = np.array([scenario_return[scenario][label] for label in sector_labels], dtype=float)
        mu_adj = mu_adj * sector_scale
        liquidity_adj = np.clip(
            liquidity_adj * np.array([scenario_liquidity[scenario][label] for label in sector_labels], dtype=float),
            0.05,
            1.0,
        )
        risk_diag = np.array([np.sqrt(scenario_risk[scenario][label]) for label in sector_labels], dtype=float)
        sigma_adj = np.diag(risk_diag) @ sigma_adj @ np.diag(risk_diag)

        uncertainty = {"positive": 0.04, "neutral": 0.06, "negative": 0.12}[scenario]
        return {
            "mu": mu_adj,
            "Sigma": sigma_adj,
            "liquidity": liquidity_adj,
            "lambda_value": self.lambda_from_risk(risk_level),
            "scenario": scenario,
            "risk_level": risk_level,
            "investment_goal": investment_goal,
            "uncertainty_radius": uncertainty,
            "scenario_label": scenario.title(),
        }


class AllocationAgent:
    def __init__(self, sector_labels: np.ndarray, sector_limits: dict[str, float], alpha: float) -> None:
        self.sector_labels = np.asarray(sector_labels, dtype=object)
        self.sector_limits = sector_limits
        self.alpha = alpha

    @staticmethod
    def sector_limits_for_risk(risk_level: str) -> dict[str, float]:
        normalized = risk_level.strip().lower()
        if normalized in {"safe", "low"}:
            return {"Residential": 0.55, "Commercial": 0.35, "Land": 0.20}
        if normalized in {"aggressive", "high"}:
            return {"Residential": 0.40, "Commercial": 0.35, "Land": 0.35}
        return {"Residential": 0.45, "Commercial": 0.40, "Land": 0.25}

    def sector_exposures(self, allocation: np.ndarray) -> dict[str, float]:
        return {
            sector: float(np.sum(allocation[self.sector_labels == sector]))
            for sector in ["Residential", "Commercial", "Land"]
        }

    def project(self, allocation: np.ndarray) -> np.ndarray:
        candidate = np.clip(np.asarray(allocation, dtype=float), 0.0, self.alpha)
        total = float(np.sum(candidate))
        if total <= 1e-12:
            candidate = np.full_like(candidate, 1.0 / len(candidate))
        else:
            candidate = candidate / total

        for _ in range(8):
            exposures = self.sector_exposures(candidate)
            for sector, limit in self.sector_limits.items():
                if exposures[sector] > limit + 1e-10:
                    mask = self.sector_labels == sector
                    candidate[mask] *= limit / exposures[sector]

            candidate = np.clip(candidate, 0.0, self.alpha)
            total = float(np.sum(candidate))
            if total <= 1e-12:
                break

            deficit = 1.0 - total
            if abs(deficit) <= 1e-8:
                candidate /= total
                break

            exposures = self.sector_exposures(candidate)
            rooms = {sector: max(0.0, self.sector_limits[sector] - exposures[sector]) for sector in self.sector_limits}
            room_total = sum(rooms.values())
            if deficit > 0 and room_total > 1e-10:
                for sector, room in rooms.items():
                    if room <= 0:
                        continue
                    mask = np.where(self.sector_labels == sector)[0]
                    if len(mask) == 0:
                        continue
                    weights = candidate[mask] + 1e-6
                    weights = weights / np.sum(weights)
                    candidate[mask] += deficit * (room / room_total) * weights
            else:
                candidate /= total

            candidate = np.clip(candidate, 0.0, self.alpha)
            total = float(np.sum(candidate))
            if total > 1e-12:
                candidate /= total

        return candidate / np.sum(candidate)

    def summarize(self, allocation: np.ndarray) -> list[dict[str, float | str]]:
        exposures = self.sector_exposures(allocation)
        return [{"type": sector, "allocation": float(exposures[sector])} for sector in ["Residential", "Commercial", "Land"]]

    def diversification_score(self, allocation: np.ndarray) -> float:
        weights = np.asarray(allocation, dtype=float)
        return float(np.clip(1.0 - np.sum(weights**2), 0.0, 1.0))

    def sector_constraint_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        bounds = []
        for sector in ["Residential", "Commercial", "Land"]:
            row = (self.sector_labels == sector).astype(float)
            rows.append(row)
            bounds.append(self.sector_limits[sector])
        return np.vstack(rows), np.asarray(bounds, dtype=float)


class MathematicalAgent:
    def normalize_vector(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if abs(max_value - min_value) <= 1e-12:
            return np.ones_like(values)
        return (values - min_value) / (max_value - min_value)

    def normalize_covariance(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        scale = float(np.max(np.abs(sigma)))
        if scale <= 1e-12:
            return sigma + 1e-8 * np.eye(sigma.shape[0])
        normalized = sigma / scale
        return 0.5 * (normalized + normalized.T) + 1e-8 * np.eye(sigma.shape[0])

    def solve_qp(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        c: np.ndarray,
        B: float,
        alpha: float,
        lambda_value: float,
        liquidity: np.ndarray,
        transaction_costs: np.ndarray,
        liquidity_weight: float,
        transaction_weight: float,
        sector_matrix: np.ndarray,
        sector_bounds: np.ndarray,
        project_fn: Callable[[np.ndarray], np.ndarray],
    ) -> dict[str, Any]:
        n = len(mu)
        x = cp.Variable(n)
        effective_mu = mu + liquidity_weight * liquidity - transaction_weight * transaction_costs
        sigma_psd = cp.psd_wrap(0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n))
        objective = cp.Maximize(effective_mu @ x - lambda_value * cp.quad_form(x, sigma_psd))
        constraints = [cp.sum(x) == 1, x >= 0, x <= alpha, c @ x <= B]
        if sector_matrix.size > 0:
            constraints.append(sector_matrix @ x <= sector_bounds)
        problem = cp.Problem(objective, constraints)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)

        solution = x.value if x.value is not None else np.ones(n) / n
        solution = project_fn(np.asarray(solution, dtype=float))
        return {
            "x": solution,
            "status": problem.status,
            "optimal": problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE},
            "objective": float(problem.value) if problem.value is not None else float("nan"),
        }

    def solve_efficient_frontier(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        c: np.ndarray,
        B: float,
        alpha: float,
        sector_matrix: np.ndarray,
        sector_bounds: np.ndarray,
        project_fn: Callable[[np.ndarray], np.ndarray],
        n_points: int = 12,
    ) -> dict[str, list[float]]:
        n = len(mu)
        x = cp.Variable(n)
        sigma_psd = cp.psd_wrap(0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n))
        target_grid = np.linspace(float(np.min(mu) + 0.15 * (np.max(mu) - np.min(mu))), float(np.max(mu) * 0.98), n_points)
        risks: list[float] = []
        returns: list[float] = []
        targets: list[float] = []
        for target_return in target_grid:
            constraints = [cp.sum(x) == 1, x >= 0, x <= alpha, c @ x <= B, mu @ x == target_return]
            if sector_matrix.size > 0:
                constraints.append(sector_matrix @ x <= sector_bounds)
            problem = cp.Problem(cp.Minimize(cp.quad_form(x, sigma_psd)), constraints)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            if x.value is None:
                continue
            solution = project_fn(np.asarray(x.value, dtype=float))
            returns.append(float(mu @ solution))
            risks.append(float(solution @ sigma @ solution))
            targets.append(float(target_return))
        return {"target_return": targets, "return": returns, "risk": risks}

    def solve_pso(
        self,
        objective_fn: Callable[[np.ndarray], float],
        project_fn: Callable[[np.ndarray], np.ndarray],
        n_dimensions: int,
        seed: int,
        x0: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return optimize_with_pso(
            objective_fn=objective_fn,
            constraint_proj_fn=project_fn,
            x0=x0,
            n_dimensions=n_dimensions,
            n_particles=max(20, min(36, n_dimensions + 6)),
            n_iterations=80,
            verbose=False,
            seed=seed,
        )

    def solve_ga(
        self,
        objective_fn: Callable[[np.ndarray], float],
        project_fn: Callable[[np.ndarray], np.ndarray],
        n_dimensions: int,
        seed: int,
        x0: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return optimize_with_ga(
            objective_fn=objective_fn,
            constraint_proj_fn=project_fn,
            x0=x0,
            n_dimensions=n_dimensions,
            population_size=max(26, min(46, n_dimensions + 12)),
            n_generations=80,
            verbose=False,
            seed=seed,
        )


class ExplanationAgent:
    def build(self, request: Any, result: dict[str, Any]) -> dict[str, str]:
        sector_mix = result.get("portfolio", [])
        sector_summary = ", ".join(f"{item['type']} {item['allocation']:.0%}" for item in sector_mix)
        top_assets = result.get("top_allocations", [])[:3]
        top_names = ", ".join(str(item["asset"]) for item in top_assets) or "a balanced set of properties"
        diversification = result.get("diversification_score", 0.0)

        if diversification >= 0.75:
            diversification_text = "strong"
        elif diversification >= 0.55:
            diversification_text = "balanced"
        else:
            diversification_text = "concentrated"

        portfolio_summary = (
            f"We recommend a {diversification_text} real estate mix across sectors: {sector_summary}. "
            f"The portfolio reflects your {result['risk_level']} risk profile under a {result['scenario_label'].lower()} market view."
        )
        explanation = (
            f"The engine selected {top_names} because they fit the target return profile after accounting for liquidity, "
            f"transaction drag, sector caps, and correlation risk."
        )
        risk_explanation = (
            f"Risk is controlled with a full covariance model, explicit sector limits, and downside-tail evaluation. "
            f"This prevents the recommendation from overloading one property type while still monitoring severe-loss scenarios."
        )

        growth_tradeoff = result.get("tradeoff_analysis", {}).get("more_growth")
        stability_tradeoff = result.get("tradeoff_analysis", {}).get("more_stability")
        pieces = []
        if growth_tradeoff:
            pieces.append(
                f"If you take more risk, expected return changes by {growth_tradeoff['delta_return']:+.2%} "
                f"and risk changes by {growth_tradeoff['delta_risk']:+.2%}."
            )
        if stability_tradeoff:
            pieces.append(
                f"If you move safer, expected return changes by {stability_tradeoff['delta_return']:+.2%} "
                f"and risk changes by {stability_tradeoff['delta_risk']:+.2%}."
            )
        tradeoffs = " ".join(pieces)

        constraint_explanation = (
            f"Sector caps, budget discipline, and a {request.max_allocation:.0%} single-asset cap keep the allocation investable "
            f"and more resilient than a pure return-maximizing portfolio."
        )
        return {
            "portfolio_summary": portfolio_summary,
            "explanation": explanation,
            "risk_explanation": risk_explanation,
            "tradeoffs": tradeoffs,
            "constraint_explanation": constraint_explanation,
        }


class TestingAgent:
    def run_sensitivity_tests(
        self,
        base_request: Any,
        base_result: dict[str, Any],
        run_callback: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        risk_cases = {}
        for risk in ["safe", "balanced", "aggressive"]:
            request_variant = replace(base_request, risk_level=risk, method="qp")
            result = run_callback(request_variant)
            risk_cases[risk] = self._snapshot(result)

        scenario_cases = {}
        for scenario in ["positive", "neutral", "negative"]:
            request_variant = replace(base_request, scenario=scenario, method="qp")
            result = run_callback(request_variant)
            scenario_cases[scenario] = self._snapshot(result)

        risk_movement = self._max_portfolio_delta(risk_cases)
        scenario_movement = self._max_portfolio_delta(scenario_cases)
        warnings: list[str] = []
        if risk_movement < 0.05:
            warnings.append("Risk sensitivity is weaker than expected.")
        if scenario_movement < 0.05:
            warnings.append("Scenario sensitivity is weaker than expected.")

        qp_tail = run_callback(replace(base_request, method="qp"))
        hybrid_tail = run_callback(replace(base_request, method="hybrid"))
        cvar_improves = float(hybrid_tail["cvar_95"]) <= float(qp_tail["cvar_95"]) + 1e-6
        if not cvar_improves:
            warnings.append("The hybrid search did not improve simulated tail risk versus QP in this run.")

        impact_view = {
            "higher_risk": {
                "delta_return": float(risk_cases["aggressive"]["expected_return"] - base_result["expected_return"]),
                "delta_risk": float(risk_cases["aggressive"]["risk"] - base_result["risk"]),
            },
            "lower_risk": {
                "delta_return": float(risk_cases["safe"]["expected_return"] - base_result["expected_return"]),
                "delta_risk": float(risk_cases["safe"]["risk"] - base_result["risk"]),
            },
        }

        return {
            "risk_variation_detected": risk_movement >= 0.05,
            "scenario_variation_detected": scenario_movement >= 0.05,
            "input_responsiveness_score": float(min(1.0, 0.5 * risk_movement + 0.5 * scenario_movement)),
            "warnings": warnings,
            "risk_sensitivity": risk_cases,
            "scenario_sensitivity": scenario_cases,
            "impact_view": impact_view,
            "cvar_validation": {
                "qp_cvar": float(qp_tail["cvar_95"]),
                "hybrid_cvar": float(hybrid_tail["cvar_95"]),
                "hybrid_improves_tail_risk": bool(cvar_improves),
            },
        }

    def _snapshot(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "expected_return": float(result["expected_return"]),
            "risk": float(result["risk"]),
            "cvar_95": float(result["cvar_95"]),
            "diversification_score": float(result["diversification_score"]),
            "portfolio": result["portfolio"],
        }

    def _max_portfolio_delta(self, cases: dict[str, dict[str, Any]]) -> float:
        keys = list(cases.keys())
        deltas = []
        for idx, left in enumerate(keys):
            for right in keys[idx + 1 :]:
                deltas.append(self._portfolio_delta(cases[left]["portfolio"], cases[right]["portfolio"]))
        return max(deltas) if deltas else 0.0

    def _portfolio_delta(self, left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
        left_map = {item["type"]: float(item["allocation"]) for item in left}
        right_map = {item["type"]: float(item["allocation"]) for item in right}
        return float(sum(abs(left_map.get(sector, 0.0) - right_map.get(sector, 0.0)) for sector in {"Residential", "Commercial", "Land"}))
