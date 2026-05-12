"""
Utility functions for constraint handling and numerical operations.
Provides reusable projection and normalization functions.
"""

import numpy as np
from typing import Tuple


def project_capped_simplex(x: np.ndarray,
                           alpha: float,
                           target_sum: float = 1.0,
                           tol: float = 1e-10,
                           max_iterations: int = 100) -> np.ndarray:
    """
    Project onto the capped simplex: sum(x)=target_sum and 0<=x_i<=alpha.
    """
    n = len(x)
    if alpha * n < target_sum - tol:
        raise ValueError("Infeasible capped simplex: alpha is too small for target sum.")

    lower = np.min(x) - alpha
    upper = np.max(x)

    for _ in range(max_iterations):
        threshold = 0.5 * (lower + upper)
        projected = np.clip(x - threshold, 0.0, alpha)
        projected_sum = projected.sum()

        if abs(projected_sum - target_sum) <= tol:
            return projected

        if projected_sum > target_sum:
            lower = threshold
        else:
            upper = threshold

    projected = np.clip(x - upper, 0.0, alpha)
    current_sum = projected.sum()

    if current_sum <= tol:
        return np.full(n, target_sum / n)

    residual = target_sum - current_sum
    free = np.where((projected > tol) & (projected < alpha - tol))[0]

    if len(free) > 0:
        projected[free] += residual / len(free)
    else:
        slack = np.where(projected < alpha - tol)[0]
        if residual > 0 and len(slack) > 0:
            for idx in slack:
                delta = min(residual, alpha - projected[idx])
                projected[idx] += delta
                residual -= delta
                if abs(residual) <= tol:
                    break
        elif residual < 0:
            active = np.where(projected > tol)[0]
            for idx in active:
                delta = min(-residual, projected[idx])
                projected[idx] -= delta
                residual += delta
                if abs(residual) <= tol:
                    break

    return np.clip(projected, 0.0, alpha)


def _enforce_budget_constraint(x: np.ndarray,
                               c: np.ndarray,
                               B: float,
                               alpha: float,
                               tol: float = 1e-10,
                               max_iterations: int = 1000) -> np.ndarray:
    """
    Reduce portfolio cost while preserving feasibility on the capped simplex.
    """
    x = np.copy(x)

    for _ in range(max_iterations):
        cost = np.dot(c, x)
        if cost <= B + tol:
            return x

        donors = np.argsort(-c)
        receivers = np.argsort(c)
        moved = False

        for donor in donors:
            if x[donor] <= tol:
                continue

            for receiver in receivers:
                if c[donor] <= c[receiver] + tol:
                    continue
                if x[receiver] >= alpha - tol:
                    continue

                cost_gap = cost - B
                delta_needed = cost_gap / (c[donor] - c[receiver])
                delta = min(delta_needed, x[donor], alpha - x[receiver])

                if delta <= tol:
                    continue

                x[donor] -= delta
                x[receiver] += delta
                moved = True
                break

            if moved:
                break

        if not moved:
            break

    return x


def project_constraints(x: np.ndarray,
                       c: np.ndarray,
                       B: float,
                       alpha: float,
                       max_iterations: int = 10) -> np.ndarray:
    """
    Project vector x onto feasible region using iterative scaling.
    
    Constraints:
    - c^T x ≤ B (budget constraint)
    - 0 ≤ x_i ≤ α (box constraints)
    - sum(x) = 1 (normalization)
    
    Parameters:
    -----------
    x : np.ndarray
        Allocation vector to project
    c : np.ndarray
        Cost vector
    B : float
        Budget limit
    alpha : float
        Maximum per-asset allocation
    max_iterations : int
        Number of projection iterations
    
    Returns:
    --------
    np.ndarray : Projected feasible allocation
    """
    x = project_capped_simplex(np.copy(x), alpha=alpha, target_sum=1.0, max_iterations=max_iterations * 20)

    if np.dot(c, x) > B + 1e-10:
        x = _enforce_budget_constraint(x, c, B, alpha, max_iterations=max_iterations * 100)
        x = project_capped_simplex(x, alpha=alpha, target_sum=1.0, max_iterations=max_iterations * 20)

    return x


def project_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project onto probability simplex: sum(x)=1, x≥0.
    Uses Duchi et al. (2008) algorithm.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    np.ndarray : Projection onto simplex
    """
    n = len(x)
    x_sorted = np.sort(x)[::-1]
    
    # Find threshold
    cumsum = np.cumsum(x_sorted)
    threshold_idx = np.where(x_sorted > (cumsum - 1) / (np.arange(1, n+1)))[0]
    
    if len(threshold_idx) > 0:
        threshold = (cumsum[threshold_idx[-1]] - 1) / (threshold_idx[-1] + 1)
    else:
        threshold = (np.sum(x) - 1) / n
    
    return np.maximum(x - threshold, 0)


def check_feasibility(x: np.ndarray,
                     c: np.ndarray,
                     B: float,
                     alpha: float,
                     tol: float = 1e-6) -> dict:
    """
    Check feasibility of allocation vector.
    
    Parameters:
    -----------
    x : np.ndarray
        Allocation vector
    c : np.ndarray
        Cost vector
    B : float
        Budget limit
    alpha : float
        Max allocation
    tol : float
        Tolerance for constraint violations
    
    Returns:
    --------
    dict : Feasibility report
    """
    report = {
        'feasible': True,
        'violations': []
    }
    
    # Check sum = 1
    sum_violation = abs(np.sum(x) - 1.0)
    if sum_violation > tol:
        report['feasible'] = False
        report['violations'].append(f"Sum violation: {sum_violation:.2e}")
    
    # Check box constraints
    min_val = np.min(x)
    if min_val < -tol:
        report['feasible'] = False
        report['violations'].append(f"Lower bound violation: min={min_val:.2e}")
    
    max_val = np.max(x)
    if max_val > alpha + tol:
        report['feasible'] = False
        report['violations'].append(f"Upper bound violation: max={max_val:.4f} (α={alpha})")
    
    # Check budget
    cost = np.dot(c, x)
    if cost > B + tol:
        report['feasible'] = False
        report['violations'].append(f"Budget violation: {cost:.4f} > {B:.4f}")
    
    return report


def normalize_vector(x: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """
    Normalize vector to sum to target value.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    target_sum : float
        Target sum
    
    Returns:
    --------
    np.ndarray : Normalized vector
    """
    current_sum = np.sum(x)
    if current_sum > 1e-10:
        return x * (target_sum / current_sum)
    else:
        return np.ones_like(x) * (target_sum / len(x))


def compute_return_and_variance(x: np.ndarray,
                               mu: np.ndarray,
                               Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Compute expected return and variance of portfolio.
    
    Parameters:
    -----------
    x : np.ndarray
        Portfolio weights
    mu : np.ndarray
        Expected returns
    Sigma : np.ndarray
        Covariance matrix
    
    Returns:
    --------
    tuple : (expected_return, variance)
    """
    expected_return = np.dot(x, mu)
    variance = x @ Sigma @ x
    return expected_return, variance


def compute_portfolio_metrics(x: np.ndarray,
                              mu: np.ndarray,
                              Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Backward-compatible alias for analytical portfolio metrics.
    
    Parameters:
    -----------
    x : np.ndarray
        Portfolio weights
    mu : np.ndarray
        Expected returns
    Sigma : np.ndarray
        Covariance matrix
    
    Returns:
    --------
    tuple : (expected_return, variance)
    """
    return compute_return_and_variance(x, mu, Sigma)


def is_positive_semidefinite(A: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Check if matrix is positive semi-definite.
    
    Parameters:
    -----------
    A : np.ndarray
        Square matrix
    tol : float
        Tolerance for eigenvalue check
    
    Returns:
    --------
    bool : True if PSD
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        # If Cholesky fails, check eigenvalues
        evals = np.linalg.eigvals(A)
        return np.all(evals >= -tol)


if __name__ == '__main__':
    # Test projection
    x = np.array([0.5, 0.6, -0.1, 0.2])
    c = np.array([1.0, 1.0, 1.0, 1.0])
    B = 1.0
    alpha = 0.4
    
    x_proj = project_constraints(x, c, B, alpha)
    print("Original:", x)
    print("Projected:", x_proj)
    
    report = check_feasibility(x_proj, c, B, alpha)
    print("Feasible:", report['feasible'])
    print("Violations:", report['violations'])
