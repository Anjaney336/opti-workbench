"""
Quadratic programming utilities derived from first principles.

The primal problem is

    minimize    f(x) = 0.5 x^T Q x + c^T x
    subject to  A x <= b
                E x  = d

with Q symmetric positive definite.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass
class QuadraticProgram:
    """
    Structured quadratic program data.

    Attributes
    ----------
    Q : ndarray, shape (D, D)
        Symmetric positive definite Hessian.
    c : ndarray, shape (D,)
        Linear term.
    A : ndarray, shape (m, D)
        Inequality matrix in A x <= b form.
    b : ndarray, shape (m,)
        Inequality right-hand side.
    E : ndarray, shape (p, D)
        Equality matrix in E x = d form.
    d : ndarray, shape (p,)
        Equality right-hand side.
    x_true : ndarray, shape (D,), optional
        Ground-truth unconstrained minimizer x* = -Q^{-1} c when available.
    name : str
        Human-readable label for reporting.
    """

    Q: Array
    c: Array
    A: Array | None = None
    b: Array | None = None
    E: Array | None = None
    d: Array | None = None
    x_true: Array | None = None
    name: str = "quadratic_program"

    def __post_init__(self) -> None:
        self.Q = np.asarray(self.Q, dtype=float)
        self.c = np.asarray(self.c, dtype=float)

        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        if self.c.shape != (self.Q.shape[0],):
            raise ValueError("c must be a vector with the same dimension as Q.")

        dimension = self.Q.shape[0]

        if self.A is None:
            self.A = np.zeros((0, dimension), dtype=float)
            self.b = np.zeros(0, dtype=float)
        else:
            self.A = np.asarray(self.A, dtype=float)
            self.b = np.asarray(self.b, dtype=float)
            if self.A.ndim != 2 or self.A.shape[1] != dimension:
                raise ValueError("A must have shape (m, dimension).")
            if self.b.shape != (self.A.shape[0],):
                raise ValueError("b must have one entry per inequality row.")

        if self.E is None:
            self.E = np.zeros((0, dimension), dtype=float)
            self.d = np.zeros(0, dtype=float)
        else:
            self.E = np.asarray(self.E, dtype=float)
            self.d = np.asarray(self.d, dtype=float)
            if self.E.ndim != 2 or self.E.shape[1] != dimension:
                raise ValueError("E must have shape (p, dimension).")
            if self.d.shape != (self.E.shape[0],):
                raise ValueError("d must have one entry per equality row.")

        if self.x_true is not None:
            self.x_true = np.asarray(self.x_true, dtype=float)
            if self.x_true.shape != (dimension,):
                raise ValueError("x_true must match problem dimension.")

        if not np.allclose(self.Q, self.Q.T, atol=1e-10):
            raise ValueError("Q must be symmetric.")

    @property
    def dimension(self) -> int:
        return self.Q.shape[0]

    @property
    def num_inequalities(self) -> int:
        return self.A.shape[0]

    @property
    def num_equalities(self) -> int:
        return self.E.shape[0]


def objective_value(problem: QuadraticProgram, x: Array) -> float:
    """Return 0.5 x^T Q x + c^T x."""
    return 0.5 * float(x @ problem.Q @ x) + float(problem.c @ x)


def objective_gradient(problem: QuadraticProgram, x: Array) -> Array:
    """Return Q x + c."""
    return problem.Q @ x + problem.c


def unconstrained_optimum(problem: QuadraticProgram) -> Array:
    """Ground truth for the strictly convex unconstrained quadratic problem."""
    return -np.linalg.solve(problem.Q, problem.c)


def constraint_residuals(problem: QuadraticProgram, x: Array) -> dict[str, Array]:
    """Return raw residuals for A x <= b and E x = d."""
    return {
        "inequality": problem.A @ x - problem.b,
        "equality": problem.E @ x - problem.d,
    }


def constraint_violation_norm(problem: QuadraticProgram, x: Array) -> float:
    """
    Squared-penalty norm used by the PSO penalty function.

    Penalizes only positive inequality violations and all equality residuals.
    """
    residuals = constraint_residuals(problem, x)
    ineq_violation = np.maximum(residuals["inequality"], 0.0)
    eq_violation = residuals["equality"]
    return float(np.linalg.norm(ineq_violation) ** 2 + np.linalg.norm(eq_violation) ** 2)


def is_feasible(problem: QuadraticProgram, x: Array, tol: float = 1e-6) -> bool:
    """Check primal feasibility."""
    residuals = constraint_residuals(problem, x)
    return bool(
        np.all(residuals["inequality"] <= tol)
        and np.all(np.abs(residuals["equality"]) <= tol)
    )


def _solve_linear_system(matrix: Array, rhs: Array) -> Array:
    """Solve a dense linear system with a least-squares fallback."""
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def solve_kkt_system(
    problem: QuadraticProgram,
    active_set: Iterable[int] = (),
) -> dict[str, Array | tuple[int, ...] | float]:
    """
    Solve the KKT system for a fixed active inequality set.

    For active inequalities I, we solve

        Q x + c + A_I^T lambda_I + E^T mu = 0
        A_I x = b_I
        E x   = d
    """
    active_set = tuple(sorted(active_set))
    active_rows = problem.A[list(active_set)] if active_set else np.zeros((0, problem.dimension))
    active_rhs = problem.b[list(active_set)] if active_set else np.zeros(0)

    constraint_matrix = np.vstack([active_rows, problem.E])
    constraint_rhs = np.concatenate([active_rhs, problem.d])

    if constraint_matrix.shape[0] == 0:
        x = unconstrained_optimum(problem)
        return {
            "x": x,
            "lambda_active": np.zeros(0),
            "mu": np.zeros(0),
            "active_set": active_set,
        }

    zero_block = np.zeros((constraint_matrix.shape[0], constraint_matrix.shape[0]))
    kkt_matrix = np.block([
        [problem.Q, constraint_matrix.T],
        [constraint_matrix, zero_block],
    ])
    rhs = np.concatenate([-problem.c, constraint_rhs])

    solution = _solve_linear_system(kkt_matrix, rhs)
    x = solution[: problem.dimension]
    dual = solution[problem.dimension :]
    lambda_active = dual[: len(active_set)]
    mu = dual[len(active_set) :]

    return {
        "x": x,
        "lambda_active": lambda_active,
        "mu": mu,
        "active_set": active_set,
    }


def kkt_residual(
    problem: QuadraticProgram,
    x: Array,
    lambdas: Array,
    mu: Array,
) -> float:
    """Maximum absolute KKT residual across stationarity and feasibility."""
    stationarity = objective_gradient(problem, x) + problem.A.T @ lambdas + problem.E.T @ mu
    primal_ineq = np.maximum(problem.A @ x - problem.b, 0.0)
    primal_eq = problem.E @ x - problem.d
    dual_feasibility = np.minimum(lambdas, 0.0)
    complementarity = lambdas * (problem.A @ x - problem.b)

    residual_vector = np.concatenate([
        stationarity,
        primal_ineq,
        primal_eq,
        dual_feasibility,
        complementarity,
    ])
    return float(np.max(np.abs(residual_vector))) if residual_vector.size else 0.0


def _candidate_active_sets(problem: QuadraticProgram, x0: Array | None) -> list[tuple[int, ...]]:
    """Enumerate active sets, optionally prioritizing inequalities near x0."""
    m = problem.num_inequalities
    all_sets = [
        combo
        for size in range(m + 1)
        for combo in combinations(range(m), size)
    ]

    if x0 is None or m == 0:
        return all_sets

    residuals = np.abs(problem.A @ x0 - problem.b)
    ranked = tuple(sorted(range(m), key=lambda idx: residuals[idx]))
    preferred = tuple(sorted(idx for idx in ranked[: min(2, m)] if residuals[idx] <= 5e-2))

    ordered: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for candidate in [preferred]:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    for size in range(m + 1):
        for combo in combinations(ranked, size):
            candidate = tuple(sorted(combo))
            if candidate not in seen:
                ordered.append(candidate)
                seen.add(candidate)

    return ordered


def solve_qp_active_set(
    problem: QuadraticProgram,
    x0: Array | None = None,
    tol: float = 1e-8,
) -> dict[str, Array | tuple[int, ...] | float | int | bool | str]:
    """
    Solve the quadratic program by enumerating KKT systems for candidate active sets.

    This is intended for research-grade small constrained problems where the
    mathematics should be explicit and inspectable.
    """
    if problem.num_inequalities > 12:
        raise ValueError(
            "Active-set enumeration is intended for small inequality counts; "
            "received more than 12 inequality constraints."
        )

    best_solution: dict[str, Array | tuple[int, ...] | float | int | bool | str] | None = None
    candidates_checked = 0

    for active_set in _candidate_active_sets(problem, x0):
        candidate = solve_kkt_system(problem, active_set)
        x = np.asarray(candidate["x"], dtype=float)
        lambda_active = np.asarray(candidate["lambda_active"], dtype=float)
        mu = np.asarray(candidate["mu"], dtype=float)
        candidates_checked += 1

        if not np.all(np.isfinite(x)):
            continue

        full_lambda = np.zeros(problem.num_inequalities)
        if active_set:
            full_lambda[list(active_set)] = lambda_active

        residuals = constraint_residuals(problem, x)
        inactive = [idx for idx in range(problem.num_inequalities) if idx not in active_set]

        inactive_ok = True
        if inactive:
            inactive_ok = bool(np.all(residuals["inequality"][inactive] <= tol))

        active_ok = True
        if active_set:
            active_ok = bool(np.all(np.abs(residuals["inequality"][list(active_set)]) <= tol))

        if not inactive_ok or not active_ok:
            continue
        if not np.all(np.abs(residuals["equality"]) <= tol):
            continue
        if not np.all(lambda_active >= -tol):
            continue

        objective = objective_value(problem, x)
        residual_norm = kkt_residual(problem, x, full_lambda, mu)

        if best_solution is None or objective < float(best_solution["objective"]):
            best_solution = {
                "x": x,
                "objective": objective,
                "lambda": full_lambda,
                "mu": mu,
                "active_set": tuple(active_set),
                "kkt_residual": residual_norm,
                "candidates_checked": candidates_checked,
                "optimal": True,
                "status": "optimal",
                "method": "kkt_active_set_enumeration",
            }

    if best_solution is None:
        return {
            "x": None,
            "objective": np.nan,
            "lambda": np.zeros(problem.num_inequalities),
            "mu": np.zeros(problem.num_equalities),
            "active_set": tuple(),
            "kkt_residual": np.inf,
            "candidates_checked": candidates_checked,
            "optimal": False,
            "status": "infeasible_or_unsolved",
            "method": "kkt_active_set_enumeration",
        }

    return best_solution
