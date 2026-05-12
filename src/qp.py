"""
Quadratic Programming solver for risk-adjusted portfolio optimization.
Uses CVXPY for convex optimization baseline.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict


def solve_quadratic_program(mu: np.ndarray,
                           Sigma: np.ndarray,
                           c: np.ndarray,
                           B: float,
                           alpha: float,
                           lambda_param: float = 1.0,
                           verbose: bool = False) -> Dict:
    """
    Solve portfolio optimization via convex quadratic programming.
    
    Problem:
    max:  x^T μ - λ x^T Σ x
    
    Subject to:
    c^T x ≤ B          (budget constraint)
    0 ≤ x_i ≤ α        (box constraints)
    sum(x) = 1         (allocation constraint)
    
    Parameters:
    -----------
    mu : np.ndarray
        Expected return vector (n_assets,)
    Sigma : np.ndarray
        Covariance matrix (n_assets, n_assets)
    c : np.ndarray
        Cost vector (n_assets,)
    B : float
        Budget limit
    alpha : float
        Maximum allocation per asset
    lambda_param : float
        Risk aversion parameter
    verbose : bool
        Print solver output
    
    Returns:
    --------
    dict : Contains optimal allocation, objective value, and solver status
    
    Notes:
    ------
    Solves in standard form using CVXPY.
    Uses SCS or OSQP solver for numerical stability.
    """
    n = len(mu)
    
    # Decision variable
    x = cp.Variable(n)
    
    # Objective: maximize x^T μ - λ x^T Σ x
    # Converted to minimization: minimize λ x^T Σ x - x^T μ
    objective = cp.Minimize(
        lambda_param * cp.quad_form(x, Sigma) - cp.sum(mu @ x)
    )
    
    # Constraints
    constraints = [
        cp.sum(x) == 1,              # Allocation normalization
        x >= 0,                       # Non-negativity
        x <= alpha,                   # Upper bound
        c @ x <= B                    # Budget
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=verbose, max_iters=200)
        
        return {
            'x': x.value,
            'objective': -problem.value if x.value is not None else np.nan,
            'status': problem.status,
            'optimal': problem.status == cp.OPTIMAL,
            'solver_info': {
                'solver': 'ECOS',
                'num_iterations': None,
            }
        }
    except:
        # Fallback to SCS if ECOS fails
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS, verbose=verbose, max_iters=5000)
            return {
                'x': x.value,
                'objective': -problem.value if x.value is not None else np.nan,
                'status': problem.status,
                'optimal': problem.status == cp.OPTIMAL,
                'solver_info': {
                    'solver': 'SCS',
                    'num_iterations': None,
                }
            }
        except Exception as e:
            return {
                'x': None,
                'objective': np.nan,
                'status': 'SOLVER_FAILED',
                'optimal': False,
                'error': str(e)
            }


def solve_markowitz(mu: np.ndarray,
                   Sigma: np.ndarray,
                   target_return: float,
                   c: np.ndarray,
                   B: float,
                   alpha: float) -> Dict:
    """
    Solve minimum variance portfolio with target return (Markowitz).
    
    Problem:
    min: x^T Σ x
    
    Subject to:
    x^T μ ≥ target_return  (return constraint)
    c^T x ≤ B              (budget constraint)
    0 ≤ x_i ≤ α            (box constraints)
    sum(x) = 1             (allocation)
    
    Parameters:
    -----------
    mu, Sigma, c, B, alpha : Portfolio parameters
    target_return : float
        Minimum expected return
    
    Returns:
    --------
    dict : Optimization results
    """
    n = len(mu)
    x = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(x, Sigma))
    
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= alpha,
        c @ x <= B,
        mu @ x >= target_return
    ]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False, max_iters=200)
        return {
            'x': x.value,
            'variance': problem.value if x.value is not None else np.nan,
            'status': problem.status,
            'optimal': problem.status == cp.OPTIMAL
        }
    except:
        return {
            'x': None,
            'variance': np.nan,
            'status': 'FAILED',
            'optimal': False
        }


def efficient_frontier(mu: np.ndarray,
                      Sigma: np.ndarray,
                      c: np.ndarray,
                      B: float,
                      alpha: float,
                      n_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute efficient frontier: returns and variances for optimal portfolios.
    
    Parameters:
    -----------
    mu, Sigma, c, B, alpha : Portfolio parameters
    n_points : int
        Number of frontier points
    
    Returns:
    --------
    tuple : (expected_returns, variances)
    """
    min_return = np.min(mu)
    max_return = np.max(mu)
    
    returns = np.linspace(min_return, max_return, n_points)
    variances = []
    
    for target in returns:
        result = solve_markowitz(mu, Sigma, target * 0.9, c, B, alpha)
        if result['optimal'] and result['x'] is not None:
            variances.append(result['variance'])
        else:
            variances.append(np.nan)
    
    return returns, np.array(variances)


if __name__ == '__main__':
    # Test QP solver
    from data import generate_portfolio_data
    
    data = generate_portfolio_data(n_assets=20, seed=42)
    
    result = solve_quadratic_program(
        mu=data['mu'],
        Sigma=data['Sigma'],
        c=data['c'],
        B=data['B'],
        alpha=data['alpha'],
        lambda_param=2.0,
        verbose=True
    )
    
    print("\nQP Results:")
    print(f"Status: {result['status']}")
    print(f"Optimal: {result['optimal']}")
    print(f"Objective: {result['objective']:.6f}")
    print(f"Allocation (first 5): {result['x'][:5]}")
