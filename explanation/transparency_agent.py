"""
Transparency agent for solver traces and mathematical audit summaries.
"""

from __future__ import annotations

from typing import Any


class SolverTransparencyAgent:
    def build_solver_audit(
        self,
        *,
        method: str,
        trace: list[dict[str, Any]],
        selected_output: dict[str, Any],
        asset_count: int,
        active_assets_target: int | None,
        objective_surface: dict[str, Any],
    ) -> dict[str, Any]:
        method_name = str(method).upper()
        return {
            "agent_name": "Solver Transparency Agent",
            "headline": self._headline(method_name),
            "mathematical_walkthrough": self._walkthrough(method_name, asset_count),
            "solver_trace": trace,
            "convergence_summary": self._convergence_summary(method_name, trace, selected_output),
            "accuracy_statement": self._accuracy_statement(selected_output),
            "solver_agent_notes": self._solver_agent_notes(method_name, selected_output, active_assets_target),
            "objective_surface": objective_surface,
        }

    def _headline(self, method: str) -> str:
        if method == "QP":
            return "The optimizer follows a deterministic constrained minimum of a quadratic objective."
        if method == "PSO":
            return "The optimizer updates a swarm of candidate portfolios and tracks the best feasible particle."
        if method == "GA":
            return "The optimizer evolves a portfolio population using selection, crossover, and mutation."
        return "The optimizer composes deterministic and stochastic stages to balance stability and exploration."

    def _walkthrough(self, method: str, asset_count: int) -> list[str]:
        base = [
            f"Construct the decision vector x over {asset_count} assets.",
            "Form expected return vector mu, covariance matrix Sigma, and shrinkage-adjusted covariance Sigma_hat.",
            "Assemble liquidity, transaction-cost, regime, and CVaR scenario tensors.",
            "Enforce feasibility with x_i >= 0 and sum(x_i) = 1.",
        ]
        if method == "QP":
            return base + [
                "Solve min x^T Sigma_hat x - lambda mu^T x + gamma C(x) - delta L(x) + theta CVaR + kappa D_gap + psi HHI + rho Omega_regime + omega T(x).",
                "Record each feasible callback iterate and select the final optimizer output.",
            ]
        if method == "PSO":
            return base + [
                "Initialize 50 particles on the simplex.",
                "Update velocity using inertia, cognitive, and social terms plus random noise.",
                "Project each particle back to feasibility and track the global best iterate.",
            ]
        if method == "GA":
            return base + [
                "Initialize a population of 100 candidate portfolios.",
                "Apply tournament-style survival pressure through elite retention and crossover.",
                "Inject Gaussian and sparsity mutation, then keep the best generation member.",
            ]
        return base + [
            "Solve a deterministic QP baseline.",
            "Refine that baseline using PSO exploration.",
            "Mutate the refined candidate with GA and blend the stage outputs into the final portfolio with adaptive hybrid weights.",
        ]

    def _convergence_summary(self, method: str, trace: list[dict[str, Any]], selected_output: dict[str, Any]) -> str:
        if not trace:
            return "No trace history was recorded for this run."
        first = trace[0]
        last = trace[-1]
        return (
            f"{method} moved from objective {first.get('objective_value', 0.0):.4f} "
            f"to {last.get('objective_value', selected_output['objective_value']):.4f} while ending at "
            f"return {selected_output['expected_return']:.4f} and risk {selected_output['risk']:.4f}."
        )

    def _accuracy_statement(self, selected_output: dict[str, Any]) -> str:
        validity = "satisfies" if selected_output.get("valid") else "does not satisfy"
        return (
            f"The final portfolio {validity} the feasibility checks, uses budget {selected_output['budget_used']:.2f}, "
            f"and reaches objective value {selected_output['objective_value']:.4f}."
        )

    def _solver_agent_notes(self, method: str, selected_output: dict[str, Any], active_assets_target: int | None) -> list[str]:
        notes = [
            f"The user requested a diversification target of {active_assets_target or 'all'} active assets.",
            f"The solver returned {selected_output['diversification']} materially active positions after feasibility projection.",
            f"Final expected return is {selected_output['expected_return']:.4f} with risk {selected_output['risk']:.4f}.",
            f"Objective terms include tail risk, concentration, regime dispersion, and tracking penalties in addition to covariance risk and return.",
        ]
        if method == "QP":
            notes.append("Because QP is deterministic, its path is governed by the local quadratic curvature of the objective.")
        elif method == "PSO":
            notes.append("Because PSO is stochastic, its path reflects noisy exploration before settling on a strong particle.")
        elif method == "GA":
            notes.append("Because GA uses mutation, irregular and sparse allocations can survive when they improve objective quality.")
        else:
            notes.append("The hybrid stack uses QP for structure, PSO for exploration, and GA for mutation-driven refinement.")
        return notes
