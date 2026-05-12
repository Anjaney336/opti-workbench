"""
PSO wrapper backed by the central PortfolioOptimizer engine.
"""

from __future__ import annotations

from .engine import PortfolioOptimizer
from .evaluation import OptimizationParams, _legacy_to_request


def optimize_pso(params: OptimizationParams) -> dict:
    optimizer = PortfolioOptimizer(_legacy_to_request(params))
    return optimizer.optimize_pso()
