"""
Simulation and what-if analysis helpers.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from simulation.scenarios import MarketScenario
from src.utils import compute_return_and_variance


def normal_cvar_gap(mean_return: float, risk: float, alpha: float = 0.95) -> tuple[float, float]:
    """
    Estimate downside CVaR gap under a normal approximation.
    """
    tail_probability = max(1e-6, 1.0 - alpha)
    z_value = NormalDist().inv_cdf(tail_probability)
    pdf_value = float(np.exp(-0.5 * z_value**2) / np.sqrt(2.0 * np.pi))
    tail_expectation = float(mean_return - risk * pdf_value / tail_probability)
    return float(max(0.0, mean_return - tail_expectation)), tail_expectation


def empirical_cvar(losses: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """
    Empirical CVaR using the Rockafellar-Uryasev representation.

    Returns:
    - cvar value
    - eta (empirical VaR threshold)
    """
    sorted_losses = np.sort(np.asarray(losses, dtype=float))
    tail_count = max(1, int(np.ceil((1.0 - alpha) * len(sorted_losses))))
    eta = float(sorted_losses[-tail_count])
    cvar = float(eta + np.mean(np.maximum(sorted_losses - eta, 0.0)) / max(1e-8, 1.0 - alpha))
    return cvar, eta


class SimulationEngine:
    def __init__(self, seed: int = 42, n_scenarios: int = 1000) -> None:
        self.seed = seed
        self.n_scenarios = n_scenarios

    def evaluate_portfolio(
        self,
        allocation: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        liquidity: np.ndarray,
        cvar_alpha: float = 0.95,
    ) -> dict[str, float | list[float]]:
        expected_return, variance = compute_return_and_variance(allocation, mu, sigma)
        risk = float(variance)
        rng = np.random.default_rng(self.seed)
        simulated_returns = rng.multivariate_normal(mu, sigma, size=self.n_scenarios)
        portfolio_returns = simulated_returns @ allocation
        losses = -portfolio_returns
        cvar_value, eta = empirical_cvar(losses, cvar_alpha)
        return {
            "expected_return": float(expected_return),
            "variance": float(variance),
            "risk": risk,
            "liquidity_score": float(np.dot(allocation, liquidity)),
            "confidence_interval": [
                float(np.quantile(portfolio_returns, 0.05)),
                float(np.quantile(portfolio_returns, 0.95)),
            ],
            "monte_carlo_return": float(np.mean(portfolio_returns)),
            "monte_carlo_risk": float(np.var(portfolio_returns)),
            "cvar_95": float(cvar_value),
            "tail_expectation": float(-eta),
        }

    def evaluate_across_scenarios(
        self,
        allocation: np.ndarray,
        base_mu: np.ndarray,
        base_sigma: np.ndarray,
        liquidity: np.ndarray,
        goal_name: str,
        lambda_value: float,
    ) -> dict[str, dict[str, float]]:
        goal_mean = float(np.mean(base_mu))
        centered = base_mu - goal_mean
        if goal_name == "growth":
            goal_mu = 1.10 * base_mu + 0.03 * centered
        elif goal_name == "income":
            goal_mu = 0.82 * base_mu + 0.18 * goal_mean
        else:
            goal_mu = base_mu

        analysis: dict[str, dict[str, float]] = {}
        for scenario_name in MarketScenario.all_names():
            scenario = MarketScenario.resolve(scenario_name)
            mu = goal_mu * scenario.return_multiplier
            sigma = base_sigma * scenario.risk_multiplier
            stats = self.evaluate_portfolio(
                allocation=allocation,
                mu=mu,
                sigma=sigma,
                liquidity=np.clip(liquidity * scenario.liquidity_multiplier, 0.05, 1.0),
            )
            analysis[scenario_name] = {
                "expected_return": float(stats["expected_return"]),
                "risk": float(stats["risk"]),
                "objective_value": float(stats["expected_return"] - lambda_value * stats["variance"]),
            }
        return analysis

    def stress_test(
        self,
        allocation: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        liquidity: np.ndarray,
        mu_noise: float = 0.08,
        sigma_shock: float = 0.10,
    ) -> dict[str, dict[str, float | list[float]]]:
        shocked_mu = mu * (1.0 - mu_noise)
        shocked_sigma = sigma * (1.0 + sigma_shock)
        return {
            "baseline": self.evaluate_portfolio(allocation, mu, sigma, liquidity),
            "stressed": self.evaluate_portfolio(allocation, shocked_mu, shocked_sigma, liquidity),
        }


class DecisionSimulationEngine:
    def __init__(self) -> None:
        self._risk_order = ["safe", "balanced", "aggressive"]

    def _step_risk(self, current: str, direction: int) -> str:
        index = self._risk_order.index(current)
        index = min(max(index + direction, 0), len(self._risk_order) - 1)
        return self._risk_order[index]

    def simulate_variants(
        self,
        base_request: Any,
        base_result: dict[str, Any],
        optimize_callback: Callable[[Any], dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        variants = {
            "higher_budget": (
                "If you increase your budget",
                replace(base_request, budget=base_request.budget * 1.20),
            ),
            "more_conservative": (
                "If you shift to a safer profile",
                replace(base_request, risk_level=self._step_risk(base_result["risk_level"], -1)),
            ),
            "more_aggressive": (
                "If you push for higher growth",
                replace(base_request, risk_level=self._step_risk(base_result["risk_level"], 1)),
            ),
            "market_downturn": (
                "If the market outlook weakens",
                replace(base_request, scenario="negative"),
            ),
        }

        analysis: dict[str, dict[str, Any]] = {}
        for key, (label, request_variant) in variants.items():
            variant_result = optimize_callback(request_variant)
            delta_return = float(variant_result["expected_return"] - base_result["expected_return"])
            delta_risk = float(variant_result["risk"] - base_result["risk"])
            analysis[key] = {
                "label": label,
                "expected_return": float(variant_result["expected_return"]),
                "risk": float(variant_result["risk"]),
                "delta_return": delta_return,
                "delta_risk": delta_risk,
                "allocation": variant_result["allocation"],
                "summary": (
                    f"{label}: return {'+' if delta_return >= 0 else ''}{delta_return:.2%}, "
                    f"risk {'+' if delta_risk >= 0 else ''}{delta_risk:.2%}."
                ),
            }
        return analysis
