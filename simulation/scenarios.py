"""
Market scenario definitions for portfolio simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketScenario:
    name: str
    label: str
    return_multiplier: float
    risk_multiplier: float
    liquidity_multiplier: float
    uncertainty_radius: float

    @staticmethod
    def resolve(name: str) -> "MarketScenario":
        normalized = name.strip().lower()
        aliases = {
            "bull": "positive",
            "positive": "positive",
            "neutral": "neutral",
            "bear": "negative",
            "negative": "negative",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported market scenario: {name}")
        key = aliases[normalized]
        scenarios = {
            "positive": MarketScenario(
                name="positive",
                label="Positive",
                return_multiplier=1.15,
                risk_multiplier=0.88,
                liquidity_multiplier=1.05,
                uncertainty_radius=0.04,
            ),
            "neutral": MarketScenario(
                name="neutral",
                label="Neutral",
                return_multiplier=1.00,
                risk_multiplier=1.00,
                liquidity_multiplier=1.00,
                uncertainty_radius=0.06,
            ),
            "negative": MarketScenario(
                name="negative",
                label="Negative",
                return_multiplier=0.80,
                risk_multiplier=1.28,
                liquidity_multiplier=0.94,
                uncertainty_radius=0.10,
            ),
        }
        return scenarios[key]

    @staticmethod
    def all_names() -> list[str]:
        return ["positive", "neutral", "negative"]
