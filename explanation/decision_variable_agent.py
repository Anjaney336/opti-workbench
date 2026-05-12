"""
Decision-variable agent that derives model inputs from minimal asset characteristics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


class DecisionVariableAgent:
    def build(
        self,
        *,
        assets: list[dict[str, Any]],
        selected_weights: list[float],
        lambda_value: float,
        gamma: float,
        delta: float,
    ) -> dict[str, Any]:
        mu = np.asarray([asset["mu"] for asset in assets], dtype=float)
        sigma = np.asarray([asset["sigma"] for asset in assets], dtype=float)
        liquidity = np.asarray([asset["liquidity"] for asset in assets], dtype=float)
        cost = np.asarray([asset["cost"] for asset in assets], dtype=float)
        weights = np.asarray(selected_weights, dtype=float)

        mu_norm = _minmax(mu)
        sigma_norm = _minmax(sigma)
        liquidity_norm = _minmax(liquidity)
        cost_norm = _minmax(cost)
        stability_score = 1.0 - sigma_norm
        cost_efficiency = 1.0 - cost_norm
        decision_score = lambda_value * mu_norm + 0.55 * stability_score + delta * liquidity_norm + 0.35 * cost_efficiency - gamma * cost_norm

        ranking = np.argsort(-decision_score)
        rank_map = {int(idx): rank + 1 for rank, idx in enumerate(ranking)}

        records = []
        for idx, asset in enumerate(assets):
            records.append(
                {
                    "name": asset["name"],
                    "mu": float(mu[idx]),
                    "sigma": float(sigma[idx]),
                    "liquidity": float(liquidity[idx]),
                    "cost": float(cost[idx]),
                    "mu_norm": float(mu_norm[idx]),
                    "risk_norm": float(sigma_norm[idx]),
                    "liquidity_norm": float(liquidity_norm[idx]),
                    "cost_norm": float(cost_norm[idx]),
                    "decision_score": float(decision_score[idx]),
                    "selected_weight": float(weights[idx]),
                    "rank": int(rank_map[idx]),
                }
            )

        records.sort(key=lambda row: row["rank"])
        return {
            "agent_name": "Decision Variable Agent",
            "methodology": [
                "Use the minimum input characteristics: expected return, volatility, liquidity, and cost.",
                "Normalize each characteristic to a comparable [0, 1] scale.",
                "Construct a decision score from return reward, stability reward, liquidity reward, and cost penalty.",
                "Rank assets by decision score and compare those ranks against the final optimized allocation weights.",
            ],
            "summary": (
                "The optimizer treats x_i as the decision variable for each asset. "
                "Those variables are driven by the minimum asset descriptors mu, sigma, liquidity, and cost after normalization and scalarization."
            ),
            "records": records,
        }

