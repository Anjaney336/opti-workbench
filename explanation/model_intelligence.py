"""
Research-grade explainability layer for multi-model portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MODEL_DESCRIPTIONS = {
    "QP": "smooth and deterministic",
    "PSO": "stochastic and exploratory",
    "GA": "mutation-driven and irregular",
    "HYBRID": "progressive and balanced",
}


@dataclass(frozen=True)
class ParameterDescriptions:
    x: str = "allocation vector across assets"
    mu: str = "expected return vector"
    Sigma: str = "covariance matrix capturing joint risk"
    Sigma_hat: str = "shrinkage-adjusted covariance matrix"
    lambda_value: str = "risk aversion weight"
    gamma: str = "cost penalty weight"
    delta: str = "liquidity reward weight"
    theta: str = "tail-risk CVaR penalty weight"
    kappa: str = "diversification-gap penalty weight"
    psi: str = "concentration penalty weight"
    rho: str = "regime-aversion weight"
    omega: str = "tracking-consensus penalty weight"
    phi: str = "covariance shrinkage intensity"
    alpha: str = "CVaR confidence level"


class ModelIntelligenceAgent:
    """Explain optimization results using equations, flow, and per-asset logic."""

    def __init__(self) -> None:
        self.parameters = ParameterDescriptions()

    def explain_model(
        self,
        model_name: str,
        weights: Any,
        mu: Any,
        cov: Any,
        *,
        asset_names: list[str] | None = None,
        liquidity: Any | None = None,
        costs: Any | None = None,
        lambda_value: float = 1.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        tail_weight: float = 0.0,
        diversification_weight: float = 0.0,
        concentration_penalty: float = 0.0,
        regime_aversion: float = 0.0,
        tracking_penalty: float = 0.0,
        covariance_shrinkage: float = 0.0,
        cvar_alpha: float = 0.95,
    ) -> dict[str, Any]:
        model = str(model_name).upper()
        weights_arr = self._as_vector(weights)
        mu_arr = self._as_vector(mu)
        cov_arr = self._as_matrix(cov)
        liquidity_arr = self._as_vector(liquidity) if liquidity is not None else None
        cost_arr = self._as_vector(costs) if costs is not None else None
        asset_names = asset_names or [f"Asset {i + 1}" for i in range(len(weights_arr))]

        metrics = self._portfolio_metrics(weights_arr, mu_arr, cov_arr, liquidity_arr, cost_arr)
        objective = self._objective_string(
            model,
            lambda_value,
            gamma,
            delta,
            tail_weight,
            diversification_weight,
            concentration_penalty,
            regime_aversion,
            tracking_penalty,
            covariance_shrinkage,
            cvar_alpha,
        )
        steps = self._solver_steps(model)
        why = self._why_allocation(model, weights_arr, mu_arr, cov_arr, liquidity_arr, cost_arr, asset_names)
        asset_justifications = self._asset_justifications(weights_arr, mu_arr, cov_arr, liquidity_arr, cost_arr, asset_names)

        return {
            "model_name": model,
            "model_behavior": MODEL_DESCRIPTIONS.get(model, "data-driven"),
            "objective_function": objective,
            "solver_steps": steps,
            "why_this_allocation": why,
            "asset_justifications": asset_justifications,
            "parameter_descriptions": self.parameter_descriptions(),
            "execution_flow": self.execution_flow(),
            "decision_variables": self.decision_variables(
                lambda_value,
                gamma,
                delta,
                tail_weight=tail_weight,
                diversification_weight=diversification_weight,
                concentration_penalty=concentration_penalty,
                regime_aversion=regime_aversion,
                tracking_penalty=tracking_penalty,
                covariance_shrinkage=covariance_shrinkage,
                cvar_alpha=cvar_alpha,
            ),
            "metrics": metrics,
        }

    def explain_all(
        self,
        model_outputs: dict[str, dict[str, Any]],
        assets: list[dict[str, Any]] | None = None,
        covariance_matrix: Any | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        assets = assets or []
        asset_names = [str(item.get("name", f"Asset {i + 1}")) for i, item in enumerate(assets)]
        mu = [float(item.get("mu", 0.0)) for item in assets] if assets else None
        liquidity = [float(item.get("liquidity", 0.0)) for item in assets] if assets else None
        costs = [float(item.get("cost", 0.0)) for item in assets] if assets else None
        cov = covariance_matrix if covariance_matrix is not None else np.eye(len(asset_names), dtype=float)
        params = parameters or {}

        return {
            model: self.explain_model(
                model,
                payload.get("weights", payload.get("allocation", [])),
                mu if mu is not None else payload.get("mu", []),
                cov,
                asset_names=asset_names or payload.get("asset_names"),
                liquidity=liquidity,
                costs=costs,
                lambda_value=float(params.get("lambda", params.get("lambda_value", 1.0))),
                gamma=float(params.get("gamma", 0.0)),
                delta=float(params.get("delta", 0.0)),
                tail_weight=float(params.get("theta", params.get("tail_weight", 0.0))),
                diversification_weight=float(params.get("kappa", params.get("diversification_weight", 0.0))),
                concentration_penalty=float(params.get("psi", params.get("concentration_penalty", 0.0))),
                regime_aversion=float(params.get("rho", params.get("regime_aversion", 0.0))),
                tracking_penalty=float(params.get("omega", params.get("tracking_penalty", 0.0))),
                covariance_shrinkage=float(params.get("phi", params.get("covariance_shrinkage", 0.0))),
                cvar_alpha=float(params.get("alpha_cvar", params.get("cvar_alpha", 0.95))),
            )
            for model, payload in model_outputs.items()
        }

    def parameter_descriptions(self) -> dict[str, str]:
        return {
            "x": self.parameters.x,
            "mu": self.parameters.mu,
            "Sigma": self.parameters.Sigma,
            "Sigma_hat": self.parameters.Sigma_hat,
            "lambda": self.parameters.lambda_value,
            "gamma": self.parameters.gamma,
            "delta": self.parameters.delta,
            "theta": self.parameters.theta,
            "kappa": self.parameters.kappa,
            "psi": self.parameters.psi,
            "rho": self.parameters.rho,
            "omega": self.parameters.omega,
            "phi": self.parameters.phi,
            "alpha": self.parameters.alpha,
        }

    def execution_flow(self) -> list[str]:
        return [
            "User inputs are converted into decision variables, regime controls, and feasibility constraints.",
            "The asset matrix is assembled from expected return, covariance, liquidity, cost, and scenario tensors.",
            "Covariance is shrinkage-adjusted before optimization.",
            "The selected optimizer computes a feasible portfolio allocation with deterministic, stochastic, or hybrid search.",
            "The allocation is normalized, validated, decomposed by objective terms, and translated into explanations.",
        ]

    def build(self, request: Any, result: dict[str, Any]) -> dict[str, str]:
        allocation = self._as_vector(result.get("allocation", []))
        top_assets = result.get("top_allocations", [])[:3]
        top_names = ", ".join(str(item.get("asset", "asset")) for item in top_assets) or "a diversified basket"
        concentration = float(np.sum(allocation**2)) if allocation.size else 0.0
        diversified = "well diversified" if concentration < 0.12 else "moderately concentrated"
        risk_label = self._risk_label(float(result.get("risk", 0.0)))

        return {
            "portfolio_summary": (
                f"We recommend a {diversified} allocation across {max(1, len(allocation))} assets "
                f"for a {result.get('risk_level', 'balanced')} investor."
            ),
            "explanation": (
                f"The optimizer concentrates weight in {top_names} because they provide stronger return, lower covariance, "
                f"or better liquidity relative to the rest of the universe."
            ),
            "risk_explanation": (
                f"Risk is {risk_label} because the allocation spreads capital across multiple assets and avoids over-concentration."
            ),
            "tradeoffs": (
                "Increasing return now also changes tail risk, regime sensitivity, and concentration, not only variance."
            ),
            "constraint_explanation": (
                "Simplex feasibility, active-asset targeting, and tracking-consensus terms keep the portfolio implementable."
            ),
        }

    def _objective_string(
        self,
        model: str,
        lambda_value: float,
        gamma: float,
        delta: float,
        tail_weight: float,
        diversification_weight: float,
        concentration_penalty: float,
        regime_aversion: float,
        tracking_penalty: float,
        covariance_shrinkage: float,
        cvar_alpha: float,
    ) -> str:
        model_label = MODEL_DESCRIPTIONS.get(model, "data-driven")
        return (
            f"Minimize x^T Sigma_hat x - {lambda_value:.3f} mu^T x + {gamma:.3f} C(x) - {delta:.3f} L(x) "
            f"+ {tail_weight:.3f} CVaR_{cvar_alpha:.3f}(x) + {diversification_weight:.3f} D_gap(x) "
            f"+ {concentration_penalty:.3f} HHI(x) + {regime_aversion:.3f} Omega_regime(x) "
            f"+ {tracking_penalty:.3f} T(x), with Sigma_hat using shrinkage phi={covariance_shrinkage:.3f}. "
            f"{model} behaves as {model_label} search."
        )

    def _solver_steps(self, model: str) -> list[str]:
        common = [
            "Form the return vector mu, covariance matrix Sigma, and shrinkage-adjusted covariance Sigma_hat.",
            "Assemble liquidity, transaction-cost, CVaR-scenario, and regime-conditioned tensors.",
            "Set the objective and enforce nonnegative weights that sum to 1.",
            "Solve the model-specific optimization problem.",
            "Project the solution back to the feasible simplex if needed.",
            "Validate the allocation and compute summary metrics.",
        ]
        if model == "QP":
            return ["Compute a local deterministic minimum of the advanced portfolio objective."] + common
        if model == "PSO":
            return ["Initialize a swarm of candidate portfolios and update velocity-position dynamics."] + common
        if model == "GA":
            return ["Initialize a population of portfolios and evolve it with crossover and mutation."] + common
        if model == "HYBRID":
            return ["Solve a QP baseline, refine with PSO, mutate with GA, then form an adaptive hybrid consensus solution."] + common
        return common

    def _why_allocation(
        self,
        model: str,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        liquidity: np.ndarray | None,
        costs: np.ndarray | None,
        asset_names: list[str],
    ) -> str:
        if weights.size == 0:
            return "No allocation was provided to explain."

        top_idx = int(np.argmax(weights))
        top_name = asset_names[top_idx]
        top_mu = float(mu[top_idx]) if top_idx < mu.size else 0.0
        top_risk = float(cov[top_idx, top_idx]) if cov.size else 0.0
        top_liquidity = float(liquidity[top_idx]) if liquidity is not None and top_idx < liquidity.size else None
        top_cost = float(costs[top_idx]) if costs is not None and top_idx < costs.size else None

        parts = [
            f"{model} assigns the largest weight to {top_name} because it offers a strong trade-off across return, covariance risk, and tail-risk control.",
            f"Its expected return is {top_mu:.4f} and its standalone variance contribution is {top_risk:.4f}.",
        ]
        if top_liquidity is not None:
            parts.append(f"The asset also has liquidity {top_liquidity:.4f}, which supports implementability.")
        if top_cost is not None:
            parts.append(f"Its transaction cost is {top_cost:.4f}, which affects how aggressively the optimizer can size it.")
        if model == "QP":
            parts.append("QP concentrates mass where the deterministic curvature of the advanced objective is most efficient.")
        elif model == "PSO":
            parts.append("PSO allows exploratory sampling, so nearby alternatives can retain weight if they improve the multi-term swarm objective.")
        elif model == "GA":
            parts.append("GA is mutation-driven, so sparse and irregular allocations can persist if they survive selection pressure.")
        elif model == "HYBRID":
            parts.append("HYBRID combines the best structural, exploratory, and mutation-driven candidates into a consensus allocation.")
        return " ".join(parts)

    def _asset_justifications(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        liquidity: np.ndarray | None,
        costs: np.ndarray | None,
        asset_names: list[str],
    ) -> list[str]:
        if weights.size == 0:
            return []

        justifications: list[str] = []
        for idx, weight in enumerate(weights):
            name = asset_names[idx] if idx < len(asset_names) else f"Asset {idx + 1}"
            if weight <= 1e-6:
                justifications.append(f"{name}: weight is near zero because it is dominated on return, covariance, liquidity, cost, or tail-risk contribution.")
                continue

            covariance_exposure = float(np.dot(cov[idx], weights)) if cov.size else 0.0
            text = [
                f"{name}: weight {weight:.2%}",
                f"mu={mu[idx]:.4f}" if idx < mu.size else "mu=n/a",
                f"covariance exposure={covariance_exposure:.4f}",
            ]
            if liquidity is not None and idx < liquidity.size:
                text.append(f"liquidity={liquidity[idx]:.4f}")
            if costs is not None and idx < costs.size:
                text.append(f"cost={costs[idx]:.4f}")
            if weight >= 0.10:
                text.append("chosen because it materially contributes to return and/or stabilizes the diversified objective")
            else:
                text.append("kept as a satellite position for balance, regime resilience, or feasibility")
            justifications.append("; ".join(text) + ".")
        return justifications

    def _portfolio_metrics(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        liquidity: np.ndarray | None,
        costs: np.ndarray | None,
    ) -> dict[str, float]:
        if weights.size == 0:
            return {"expected_return": 0.0, "risk": 0.0, "diversification": 0.0, "average_liquidity": 0.0, "average_cost": 0.0}
        expected_return = float(np.dot(weights, mu)) if mu.size else 0.0
        risk = float(weights.T @ cov @ weights) if cov.size else 0.0
        diversification = float(1.0 - np.sum(weights**2))
        average_liquidity = float(np.dot(weights, liquidity)) if liquidity is not None else 0.0
        average_cost = float(np.dot(weights, costs)) if costs is not None else 0.0
        return {
            "expected_return": expected_return,
            "risk": risk,
            "diversification": diversification,
            "average_liquidity": average_liquidity,
            "average_cost": average_cost,
        }

    def decision_variables(
        self,
        lambda_value: float,
        gamma: float,
        delta: float,
        *,
        tail_weight: float = 0.0,
        diversification_weight: float = 0.0,
        concentration_penalty: float = 0.0,
        regime_aversion: float = 0.0,
        tracking_penalty: float = 0.0,
        covariance_shrinkage: float = 0.0,
        cvar_alpha: float = 0.95,
    ) -> dict[str, str | float]:
        return {
            "x": self.parameters.x,
            "mu": self.parameters.mu,
            "Sigma": self.parameters.Sigma,
            "Sigma_hat": self.parameters.Sigma_hat,
            "lambda": float(lambda_value),
            "gamma": float(gamma),
            "delta": float(delta),
            "theta": float(tail_weight),
            "kappa": float(diversification_weight),
            "psi": float(concentration_penalty),
            "rho": float(regime_aversion),
            "omega": float(tracking_penalty),
            "phi": float(covariance_shrinkage),
            "alpha": float(cvar_alpha),
        }

    def _risk_label(self, risk: float) -> str:
        if risk < 0.02:
            return "low"
        if risk < 0.06:
            return "moderate"
        return "elevated"

    @staticmethod
    def _as_vector(value: Any) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        return arr.astype(float, copy=False)

    @staticmethod
    def _as_matrix(value: Any) -> np.ndarray:
        if value is None:
            return np.asarray([[]], dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = np.diag(arr)
        return arr.astype(float, copy=False)


__all__ = ["ModelIntelligenceAgent", "ParameterDescriptions", "MODEL_DESCRIPTIONS"]
