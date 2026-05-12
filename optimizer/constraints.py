"""
Constraint abstraction for portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import check_feasibility, project_constraints


@dataclass(frozen=True)
class ConstraintConfig:
    budget: float
    max_allocation: float
    liquidity_threshold: float | None = None
    integer_constraint: bool = False


class ConstraintEngine:
    """
    Applies convex portfolio constraints and optional demo-mode extensions.
    """

    def __init__(
        self,
        c: np.ndarray,
        B: float,
        alpha: float,
        liquidity_scores: np.ndarray,
        config: ConstraintConfig,
    ) -> None:
        self.c = c
        self.B = B
        self.alpha = alpha
        self.liquidity_scores = liquidity_scores
        self.config = config

    def project(self, x: np.ndarray) -> np.ndarray:
        candidate = np.asarray(x, dtype=float)
        if self.config.liquidity_threshold is not None:
            allowed = self.liquidity_scores >= self.config.liquidity_threshold
            if np.any(allowed):
                candidate = np.where(allowed, candidate, 0.0)

        candidate = project_constraints(candidate, self.c, self.B, self.alpha)

        if self.config.integer_constraint:
            candidate = self._project_integer_style(candidate)

        return candidate

    def _project_integer_style(self, x: np.ndarray) -> np.ndarray:
        """
        Demo-only sparse selection surrogate for integer allocation logic.
        """
        n_keep = min(len(x), max(4, len(x) // 5))
        mask = np.zeros_like(x, dtype=bool)
        mask[np.argsort(-x)[:n_keep]] = True
        candidate = np.where(mask, x, 0.0)
        return project_constraints(candidate, self.c, self.B, self.alpha)

    def is_complex(self) -> bool:
        return self.config.integer_constraint or self.config.liquidity_threshold is not None

    def check(self, x: np.ndarray) -> dict:
        return check_feasibility(x, self.c, self.B, self.alpha)
