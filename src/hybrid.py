"""
Hybrid optimization framework.
Pipeline: QP → PSO (warm start) → GA (warm start) → final solution
"""

import numpy as np
from typing import Dict
import copy


def hybrid_optimize(mu: np.ndarray,
                   Sigma: np.ndarray,
                   c: np.ndarray,
                   B: float,
                   alpha: float,
                   lambda_param: float = 1.0,
                   verbose: bool = False,
                   seed: int = 42) -> Dict:
    """
    Complete hybrid optimization pipeline.
    
    Pipeline:
    1. Solve convex QP for baseline
    2. Initialize PSO with QP solution (warm start)
    3. Initialize GA with best PSO solution (warm start)
    4. Return all solutions and metrics
    
    Parameters:
    -----------
    mu : np.ndarray
        Expected returns
    Sigma : np.ndarray
        Covariance matrix
    c : np.ndarray
        Costs
    B : float
        Budget
    alpha : float
        Max allocation
    lambda_param : float
        Risk aversion parameter
    verbose : bool
        Print progress
    seed : int
        Random seed
    
    Returns:
    --------
    dict : Contains x_qp, x_pso, x_ga, x_final with results
    """
    
    from src.qp import solve_quadratic_program
    from src.pso import optimize_with_pso
    from src.ga import optimize_with_ga
    from src.utils import project_constraints, compute_return_and_variance
    from src.evaluation import monte_carlo_evaluate
    
    np.random.seed(seed)
    
    # ====== STAGE 1: Quadratic Programming ======
    if verbose:
        print("\n" + "="*70)
        print("HYBRID OPTIMIZATION PIPELINE")
        print("="*70)
        print("\n[STAGE 1] QUADRATIC PROGRAMMING (Baseline)")
        print("-" * 70)
    
    qp_result = solve_quadratic_program(
        mu=mu, Sigma=Sigma, c=c, B=B, alpha=alpha,
        lambda_param=lambda_param, verbose=False
    )
    
    if not qp_result['optimal']:
        if verbose:
            print(f"[WARN] QP Status: {qp_result['status']}")
        x_qp = np.ones(len(mu)) / len(mu)
    else:
        x_qp = qp_result['x']
    
    # Ensure feasibility
    x_qp = project_constraints(x_qp, c, B, alpha)
    
    ret_qp, var_qp = compute_return_and_variance(x_qp, mu, Sigma)
    
    if verbose:
        print(f"[OK] QP Solution:")
        print(f"  Expected Return: {ret_qp:.6f}")
        print(f"  Variance: {var_qp:.6f}")
        print(f"  Utility (return - lambda*var): {ret_qp - lambda_param * var_qp:.6f}")
    
    # ====== STAGE 2: Particle Swarm Optimization ======
    if verbose:
        print("\n[STAGE 2] PARTICLE SWARM OPTIMIZATION (Warm Start from QP)")
        print("-" * 70)
    
    def objective_fn(x):
        return np.dot(x, mu) - lambda_param * (x @ Sigma @ x)
    
    def constraint_proj(x):
        return project_constraints(x, c, B, alpha)
    
    pso_result = optimize_with_pso(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj,
        x0=x_qp,
        n_particles=30,
        n_iterations=150,
        verbose=verbose,
        seed=seed
    )
    
    x_pso = pso_result['x']
    ret_pso, var_pso = compute_return_and_variance(x_pso, mu, Sigma)
    
    if verbose:
        print(f"\n[OK] PSO Solution:")
        print(f"  Expected Return: {ret_pso:.6f}")
        print(f"  Variance: {var_pso:.6f}")
        print(f"  Utility: {ret_pso - lambda_param * var_pso:.6f}")
        improvement = pso_result['fitness'] - qp_result['objective']
        print(f"  Improvement over QP: {improvement:+.6f}")
    
    # ====== STAGE 3: Genetic Algorithm ======
    if verbose:
        print("\n[STAGE 3] GENETIC ALGORITHM (Warm Start from PSO)")
        print("-" * 70)
    
    ga_result = optimize_with_ga(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj,
        x0=x_pso,
        population_size=40,
        n_generations=150,
        verbose=verbose,
        seed=seed
    )
    
    x_ga = ga_result['x']
    ret_ga, var_ga = compute_return_and_variance(x_ga, mu, Sigma)
    
    if verbose:
        print(f"\n[OK] GA Solution:")
        print(f"  Expected Return: {ret_ga:.6f}")
        print(f"  Variance: {var_ga:.6f}")
        print(f"  Utility: {ret_ga - lambda_param * var_ga:.6f}")
        improvement = ga_result['fitness'] - pso_result['fitness']
        print(f"  Improvement over PSO: {improvement:+.6f}")
    
    # ====== FINAL DECISION ======
    if verbose:
        print("\n" + "="*70)
        print("FINAL SOLUTION SELECTION")
        print("="*70)
    
    # Select best overall solution based on utility
    utility_qp = ret_qp - lambda_param * var_qp
    utility_pso = ret_pso - lambda_param * var_pso
    utility_ga = ret_ga - lambda_param * var_ga
    
    utilities = {
        'QP': utility_qp,
        'PSO': utility_pso,
        'GA': utility_ga
    }
    
    best_method = max(utilities, key=utilities.get)
    
    if best_method == 'QP':
        x_final = x_qp
    elif best_method == 'PSO':
        x_final = x_pso
    else:
        x_final = x_ga
    
    if verbose:
        print(f"\nBest Method: {best_method}")
        print(f"Utilities: QP={utility_qp:.6f}, PSO={utility_pso:.6f}, GA={utility_ga:.6f}")
        print(f"\n[OK] Final Solution Metrics:")
        ret_final, var_final = compute_return_and_variance(x_final, mu, Sigma)
        print(f"  Expected Return: {ret_final:.6f}")
        print(f"  Variance: {var_final:.6f}")
        print(f"  Utility: {ret_final - lambda_param * var_final:.6f}")
    
    # Compile results
    result = {
        'x_qp': x_qp,
        'x_pso': x_pso,
        'x_ga': x_ga,
        'x_final': x_final,
        'best_method': best_method,
        
        'metrics': {
            'QP': {
                'x': x_qp,
                'return': ret_qp,
                'variance': var_qp,
                'utility': utility_qp,
                'objective': qp_result['objective']
            },
            'PSO': {
                'x': x_pso,
                'return': ret_pso,
                'variance': var_pso,
                'utility': utility_pso,
                'objective': pso_result['fitness'],
                'n_particles': pso_result['n_particles'],
                'n_iterations': pso_result['n_iterations']
            },
            'GA': {
                'x': x_ga,
                'return': ret_ga,
                'variance': var_ga,
                'utility': utility_ga,
                'objective': ga_result['fitness'],
                'population_size': ga_result['population_size'],
                'n_generations': ga_result['n_generations']
            }
        },
        'convergence': {
            'PSO': pso_result['history'],
            'GA': ga_result['history']
        }
    }
    
    return result


if __name__ == '__main__':
    from data import generate_portfolio_data
    
    # Generate test data
    data = generate_portfolio_data(n_assets=30, seed=42)
    
    # Run hybrid pipeline
    result = hybrid_optimize(
        mu=data['mu'],
        Sigma=data['Sigma'],
        c=data['c'],
        B=data['B'],
        alpha=data['alpha'],
        lambda_param=2.0,
        verbose=True,
        seed=42
    )
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nBest Method: {result['best_method']}")
    print(f"Final allocation (first 5 assets): {result['x_final'][:5]}")
