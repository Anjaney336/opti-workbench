"""
Decision-range agent for interpretable, JSON-ready feature ranking and banding.

This module turns minimal asset characteristics into:
- normalized feature vectors
- decision scores
- allocation bands / ranges
- rank logic
- plain-English mathematical explanations

It deliberately uses only mu, sigma, liquidity, and cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = ["DecisionRangeAgent", "DecisionRangeResult"]


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def _band_from_score(score: float, position: int, count: int) -> dict[str, float | str]:
    # Higher scores receive wider and higher bands.
    base = 1.0 / max(count, 1)
    rank_scale = (count - position) / max(count - 1, 1) if count > 1 else 1.0
    preferred = base * (0.55 + 0.90 * score)
    width = base * (0.20 + 0.25 * (1.0 - score))
    lower = max(0.0, preferred - width)
    upper = min(1.0, preferred + width)
    if score >= 0.75:
        label = "core"
    elif score >= 0.50:
        label = "support"
    else:
        label = "satellite"
    return {
        "preferred_weight": float(preferred),
        "min_weight": float(lower),
        "max_weight": float(upper),
        "materiality": label,
        "rank_scale": float(rank_scale),
    }


@dataclass(frozen=True)
class DecisionRangeResult:
    agent_name: str
    feature_summary: dict[str, float]
    normalization: dict[str, str]
    decision_formula: str
    ranking_rule: str
    ranges: list[dict[str, Any]]
    plain_english: str
    bands_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "feature_summary": self.feature_summary,
            "normalization": self.normalization,
            "decision_formula": self.decision_formula,
            "ranking_rule": self.ranking_rule,
            "ranges": self.ranges,
            "plain_english": self.plain_english,
            "bands_summary": self.bands_summary,
        }


class DecisionRangeAgent:
    """
    Convert minimal asset features into a transparent decision-variable range model.

    The scoring rule is intentionally simple and interpretable:

        score = w_mu * mu_norm + w_liq * liquidity_norm + w_stab * (1 - sigma_norm) - w_cost * cost_norm

    where all normalized features lie in [0, 1].
    """

    def __init__(
        self,
        *,
        w_mu: float = 0.42,
        w_sigma: float = 0.28,
        w_liquidity: float = 0.22,
        w_cost: float = 0.18,
    ) -> None:
        self.w_mu = float(w_mu)
        self.w_sigma = float(w_sigma)
        self.w_liquidity = float(w_liquidity)
        self.w_cost = float(w_cost)

    def build(
        self,
        *,
        assets: list[dict[str, Any]],
        selected_weights: Iterable[float] | None = None,
    ) -> dict[str, Any]:
        if not assets:
            return DecisionRangeResult(
                agent_name="Decision Range Agent",
                feature_summary={},
                normalization={},
                decision_formula=self.decision_formula(),
                ranking_rule=self.ranking_rule(),
                ranges=[],
                plain_english="No assets were provided, so no decision ranges could be derived.",
                bands_summary={"active_asset_count": 0, "top_asset": None},
            ).to_dict()

        mu = _as_float_array(asset["mu"] for asset in assets)
        sigma = _as_float_array(asset["sigma"] for asset in assets)
        liquidity = _as_float_array(asset["liquidity"] for asset in assets)
        cost = _as_float_array(asset["cost"] for asset in assets)
        weights = _as_float_array(selected_weights) if selected_weights is not None else np.zeros(len(assets), dtype=float)
        if weights.size and weights.size != len(assets):
            raise ValueError("selected_weights must match the number of assets.")

        mu_n = _minmax(mu)
        sigma_n = _minmax(sigma)
        liquidity_n = _minmax(liquidity)
        cost_n = _minmax(cost)
        stability_n = 1.0 - sigma_n

        scores = (
            self.w_mu * mu_n
            + self.w_liquidity * liquidity_n
            + self.w_sigma * stability_n
            - self.w_cost * cost_n
        )
        scores = _minmax(scores)
        ranking = np.argsort(-scores)

        asset_ranges: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranking, start=1):
            band = _band_from_score(float(scores[idx]), rank, len(assets))
            asset_ranges.append(
                {
                    "name": str(assets[idx].get("name", f"Asset {idx + 1}")),
                    "sector": str(assets[idx].get("sector", "Unknown")),
                    "mu": float(mu[idx]),
                    "sigma": float(sigma[idx]),
                    "liquidity": float(liquidity[idx]),
                    "cost": float(cost[idx]),
                    "mu_norm": float(mu_n[idx]),
                    "sigma_norm": float(sigma_n[idx]),
                    "liquidity_norm": float(liquidity_n[idx]),
                    "cost_norm": float(cost_n[idx]),
                    "decision_score": float(scores[idx]),
                    "rank": int(rank),
                    **band,
                    "selected_weight": float(weights[idx]) if weights.size else 0.0,
                    "reason": self._asset_reason(
                        name=str(assets[idx].get("name", f"Asset {idx + 1}")),
                        mu=float(mu[idx]),
                        sigma=float(sigma[idx]),
                        liquidity=float(liquidity[idx]),
                        cost=float(cost[idx]),
                        score=float(scores[idx]),
                        selected_weight=float(weights[idx]) if weights.size else 0.0,
                    ),
                }
            )

        active_count = int(np.sum(scores >= np.quantile(scores, 0.60)))
        if weights.size:
            active_count = max(active_count, int(np.sum(weights > 1e-6)))

        selected_names = [row["name"] for row in asset_ranges[: min(3, len(asset_ranges))]]
        english = (
            "The model first normalizes expected return, volatility, liquidity, and cost onto a common [0, 1] scale. "
            "It then computes a transparent score that rewards higher return and liquidity, rewards lower volatility, and penalizes higher cost. "
            f"The top assets by that score are {', '.join(selected_names)}. "
            "Those assets receive wider allocation bands, while lower-ranked assets are assigned tighter satellite bands or near-zero weight bands."
        )

        return DecisionRangeResult(
            agent_name="Decision Range Agent",
            feature_summary={
                "mean_mu": float(np.mean(mu)),
                "mean_sigma": float(np.mean(sigma)),
                "mean_liquidity": float(np.mean(liquidity)),
                "mean_cost": float(np.mean(cost)),
                "active_asset_count": float(active_count),
            },
            normalization={
                "mu": "min-max normalized to [0, 1]",
                "sigma": "min-max normalized to [0, 1], then inverted as stability = 1 - sigma_norm",
                "liquidity": "min-max normalized to [0, 1]",
                "cost": "min-max normalized to [0, 1], then treated as a penalty",
            },
            decision_formula=self.decision_formula(),
            ranking_rule=self.ranking_rule(),
            ranges=asset_ranges,
            plain_english=english,
            bands_summary={
                "active_asset_count": active_count,
                "top_asset": selected_names[0] if selected_names else None,
                "top_band_type": asset_ranges[0]["materiality"] if asset_ranges else None,
            },
        ).to_dict()

    def decision_formula(self) -> str:
        return (
            "score_i = 0.42 * mu_norm_i + 0.28 * (1 - sigma_norm_i) + "
            "0.22 * liquidity_norm_i - 0.18 * cost_norm_i"
        )

    def ranking_rule(self) -> str:
        return "Rank assets by descending decision_score; wider allocation bands are assigned to higher-ranked assets."

    def _asset_reason(
        self,
        *,
        name: str,
        mu: float,
        sigma: float,
        liquidity: float,
        cost: float,
        score: float,
        selected_weight: float,
    ) -> str:
        direction = "accepted" if score >= 0.5 else "down-weighted"
        if selected_weight <= 1e-6:
            chosen = "its final weight is near zero"
        elif selected_weight < 0.05:
            chosen = "it is a satellite position"
        else:
            chosen = "it is a material allocation"
        return (
            f"{name} is {direction} because mu={mu:.4f}, sigma={sigma:.4f}, liquidity={liquidity:.4f}, "
            f"and cost={cost:.4f} combine into score={score:.4f}; {chosen}."
        )

