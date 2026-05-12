"""
Human-readable explanation generation for portfolio decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ExplanationEngine:
    def build(self, request: Any, result: dict[str, Any]) -> dict[str, str]:
        allocation = np.asarray(result["allocation"], dtype=float)
        active_assets = int(np.sum(allocation > 0.015))
        concentration = float(np.sum(allocation**2))
        top_assets = result.get("top_allocations", [])[:3]
        top_names = ", ".join(str(item["asset"]) for item in top_assets) or "a diversified property basket"

        diversification_text = "well diversified" if concentration < 0.11 else "moderately concentrated"
        risk_label = self._risk_label(result["risk"])
        scenario_label = str(result.get("scenario_label", str(result["scenario"]).title()))

        portfolio_summary = (
            f"We recommend investing across {active_assets} properties with a {diversification_text} structure "
            f"built for a {result['risk_level']} investor in a {scenario_label.lower()} market."
        )

        explanation = (
            f"High-conviction exposure is placed in {top_names}. The allocation favors assets that support a "
            f"{result['investment_goal']} objective while preserving portfolio balance through scenario-aware diversification."
        )

        risk_explanation = (
            f"Your portfolio risk is {risk_label} because capital is spread across {active_assets} holdings and the "
            f"largest positions stay within the allocation cap. This limits single-property concentration while still "
            f"allowing stronger opportunities to influence performance."
        )

        tradeoff = self._tradeoff_text(result.get("tradeoff_analysis", {}))
        constraint_explanation = (
            f"Budget and allocation constraints prevent over-allocating into premium assets. "
            f"The {request.max_allocation:.0%} cap forces diversification and keeps the recommendation implementable."
        )

        return {
            "portfolio_summary": portfolio_summary,
            "explanation": explanation,
            "risk_explanation": risk_explanation,
            "tradeoffs": tradeoff,
            "constraint_explanation": constraint_explanation,
        }

    def _tradeoff_text(self, tradeoffs: dict[str, dict[str, Any]]) -> str:
        growth = tradeoffs.get("more_growth")
        stability = tradeoffs.get("more_stability")

        parts: list[str] = []
        if growth:
            parts.append(
                f"Moving to a more aggressive profile would change expected return by {growth['delta_return']:+.2%} "
                f"and risk by {growth['delta_risk']:+.2%}."
            )
        if stability:
            parts.append(
                f"Moving to a safer profile would change expected return by {stability['delta_return']:+.2%} "
                f"and risk by {stability['delta_risk']:+.2%}."
            )
        return " ".join(parts)

    def _risk_label(self, risk: float) -> str:
        if risk < 0.12:
            return "low"
        if risk < 0.19:
            return "moderate"
        return "elevated"
