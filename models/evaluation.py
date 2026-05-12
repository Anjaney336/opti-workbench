"""
Compatibility wrappers for the product-facing optimization engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import PortfolioOptimizer, PortfolioRequest


@dataclass(frozen=True)
class OptimizationParams:
    budget: float
    lambda_value: float
    max_allocation: float
    method: str = "HYBRID"
    n_assets: int = 30
    seed: int = 42
    mc_scenarios: int = 500
    expected_return_min: float = 0.05
    expected_return_max: float = 0.15


def _legacy_to_request(params: OptimizationParams) -> PortfolioRequest:
    if params.lambda_value >= 0.8:
        risk_level = "low"
    elif params.lambda_value >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "high"

    return PortfolioRequest(
        budget=params.budget,
        risk_level=risk_level,
        investment_goal="balanced",
        scenario="neutral",
        method=params.method.lower(),
        max_allocation=params.max_allocation,
        n_assets=params.n_assets,
        seed=params.seed,
        mc_scenarios=params.mc_scenarios,
        expected_return_min=params.expected_return_min,
        expected_return_max=params.expected_return_max,
    )


def compare_methods(base_params: OptimizationParams) -> dict[str, dict]:
    """
    Backward-compatible comparison helper for the old demo wrappers.
    """
    optimizer = PortfolioOptimizer(_legacy_to_request(base_params))
    return optimizer.compare_methods()
