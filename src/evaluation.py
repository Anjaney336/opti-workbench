"""
Monte Carlo simulation for portfolio evaluation.
Provides unbiased return and risk metrics via stochastic sampling.
"""

import numpy as np
from typing import Tuple, Dict, List, Literal


def diversification_index(x: np.ndarray) -> float:
    """
    Herfindahl-style diversification score:

        D(x) = 1 - sum_i x_i^2

    Larger values indicate lower concentration.
    """
    weights = np.asarray(x, dtype=float)
    return float(1.0 - np.sum(weights ** 2))


def monte_carlo_evaluate(x: np.ndarray,
                         mu: np.ndarray,
                         Sigma: np.ndarray,
                         n_scenarios: int = 1000,
                         seed: int = None) -> Dict:
    """
    Evaluate portfolio via Monte Carlo simulation.
    
    Parameters:
    -----------
    x : np.ndarray
        Portfolio allocation (n_assets,)
    mu : np.ndarray
        Expected returns (n_assets,)
    Sigma : np.ndarray
        Covariance matrix (n_assets, n_assets)
    n_scenarios : int
        Number of Monte Carlo samples (≥1000 recommended)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Statistics including mean return, variance, quantiles, Sharpe ratio,
    return-tail metrics, and explicit loss-based VaR/CVaR metrics
    
    Notes:
    ------
    Samples R ~ N(μ, Σ) for n_scenarios realizations.
    Computes empirical moments and tail statistics.

    Note:
    -----
    The report includes both:
    - return-tail quantities (`var_95`, `cvar_95`) for backward compatibility
    - loss-tail quantities (`loss_var_95`, `loss_cvar_95`) aligned with the
      standard financial definition used in CVaR optimization
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample asset returns
    R = np.random.multivariate_normal(mu, Sigma, size=n_scenarios)
    
    # Portfolio returns
    portfolio_returns = R @ x
    
    # Compute statistics
    mean_return = np.mean(portfolio_returns)
    variance = np.var(portfolio_returns)
    std_return = np.sqrt(variance)
    
    # Quantiles
    quantiles = np.quantile(portfolio_returns, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = mean_return / (std_return + 1e-10)

    # Return-tail metrics kept for backward compatibility with existing callers.
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

    # Loss-tail metrics aligned with the canonical CVaR definition.
    losses = -portfolio_returns
    loss_var_95 = np.percentile(losses, 95)
    loss_cvar_95 = np.mean(losses[losses >= loss_var_95])

    # Skewness and Kurtosis
    skewness = np.mean((portfolio_returns - mean_return) ** 3) / (std_return ** 3 + 1e-10)
    excess_kurtosis = np.mean((portfolio_returns - mean_return) ** 4) / (variance ** 2 + 1e-10) - 3

    report = {
        'mean_return': mean_return,
        'variance': variance,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'max_return': np.max(portfolio_returns),
        'min_return': np.min(portfolio_returns),
        'quantiles': {
            '1%': quantiles[0],
            '5%': quantiles[1],
            '25%': quantiles[2],
            '50%': quantiles[3],
            '75%': quantiles[4],
            '95%': quantiles[5],
            '99%': quantiles[6]
        },
        'var_95': var_95,
        'cvar_95': cvar_95,  # Mean of worst 5% returns (legacy return-tail metric)
        'loss_var_95': loss_var_95,
        'loss_cvar_95': loss_cvar_95,
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
        'diversification_index': diversification_index(x),
        'n_scenarios': n_scenarios,
        'returns_sample': portfolio_returns[:100]  # First 100 samples
    }
    
    return report


def compare_portfolios(allocations: Dict[str, np.ndarray],
                      mu: np.ndarray,
                      Sigma: np.ndarray,
                      n_scenarios: int = 2000,
                      seed: int = 42) -> Dict:
    """
    Compare multiple portfolio allocations via Monte Carlo.
    
    Parameters:
    -----------
    allocations : dict
        {'name': allocation_vector}
    mu, Sigma : Portfolio parameters
    n_scenarios : int
        MC samples
    seed : int
        Random seed
    
    Returns:
    --------
    dict : Results for each portfolio
    """
    results = {}
    
    for name, x in allocations.items():
        if x is None:
            results[name] = {'error': 'Allocation is None'}
            continue
        
        results[name] = monte_carlo_evaluate(x, mu, Sigma, n_scenarios, seed)
    
    return results


def portfolio_statistics_table(results: Dict) -> str:
    """
    Format comparison results as readable table.
    
    Parameters:
    -----------
    results : dict
        Output from compare_portfolios
    
    Returns:
    --------
    str : Formatted table
    """
    table_width = 160
    lines = []
    lines.append("\n" + "=" * table_width)
    lines.append("PORTFOLIO PERFORMANCE COMPARISON (Monte Carlo, 2000 scenarios)")
    lines.append("=" * table_width)
    
    # Header
    metrics = ['Mean Return', 'Std Dev', 'Sharpe Ratio', 'VaR(95%)', 'CVaR(95%)',
               'Loss CVaR', 'Diversif.', 
               'Skewness', 'Excess Kurt.', 'Min/Max Return']
    header = "Portfolio".ljust(15)
    for m in metrics:
        header += f" | {m:>12}"
    lines.append(header)
    lines.append("-" * table_width)
    
    # Data
    for name, stats in results.items():
        if 'error' in stats:
            lines.append(f"{name:15} | ERROR: {stats['error']}")
            continue
        
        row = f"{name:15}"
        row += f" | {stats['mean_return']:>12.6f}"
        row += f" | {stats['std_return']:>12.6f}"
        row += f" | {stats['sharpe_ratio']:>12.4f}"
        row += f" | {stats['var_95']:>12.6f}"
        row += f" | {stats['cvar_95']:>12.6f}"
        row += f" | {stats['loss_cvar_95']:>12.6f}"
        row += f" | {stats['diversification_index']:>12.4f}"
        row += f" | {stats['skewness']:>12.4f}"
        row += f" | {stats['excess_kurtosis']:>12.4f}"
        row += f" | {stats['min_return']:>7.4f}/{stats['max_return']:>7.4f}"
        
        lines.append(row)
    
    lines.append("=" * table_width)
    return "\n".join(lines)


def compute_portfolio_metrics(x: np.ndarray,
                             mu: np.ndarray,
                             Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Compute analytical portfolio expected return and variance.
    
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


def compare_against_reference_model(
    model_scope: Literal['src', 'optimizer'] = 'optimizer',
    reference_name: str = 'Anjaney Malhotra',
) -> Dict:
    """
    Compare the implemented portfolio model against the reference formulation.

    Parameters
    ----------
    model_scope : {'src', 'optimizer'}
        'src' compares against the simplified mean-variance implementation in
        the src/ package.
        'optimizer' compares against the richer production optimizer package.
    reference_name : str
        Human-readable label for the external reference model.

    Returns
    -------
    dict
        Structured comparison with objectives, features, scores, and verdicts.
    """
    implemented = _implemented_model_spec(model_scope)
    reference = _reference_model_spec(reference_name)
    criteria = _comparison_criteria(model_scope)

    weighted_implemented = sum(item['weight'] * item['implemented_score'] for item in criteria)
    weighted_reference = sum(item['weight'] * item['reference_score'] for item in criteria)

    if model_scope == 'src':
        overall_verdict = (
            f"{reference_name}'s model is mathematically stronger than the current src model "
            "because it explicitly adds tail-risk control, liquidity utility, transaction penalties, "
            "and regime dependence."
        )
        deployment_verdict = (
            "The current src model is easier to solve and validate, but it is too close to classical "
            "Markowitz to dominate the reference on real-estate realism."
        )
    else:
        overall_verdict = (
            "Neither model strictly dominates. The optimizer package is stronger as an implemented, "
            "deployable engineering system with explicit sampled tail-risk control, while the reference "
            "model remains cleaner as a paper-style stochastic formulation."
        )
        deployment_verdict = (
            "For production use today, the implemented optimizer is better because liquidity, cost, "
            "sampled CVaR, sparsity control, and hybrid search are all operational. For a research-paper "
            "objective, the reference remains attractive if exact regime stochastic programming or exact "
            "mixed-integer asset selection is required."
        )

    return {
        'implemented_model': implemented,
        'reference_model': reference,
        'criteria': criteria,
        'score_summary': {
            'implemented_weighted_score': float(weighted_implemented),
            'reference_weighted_score': float(weighted_reference),
        },
        'verdict': {
            'overall': overall_verdict,
            'deployment': deployment_verdict,
            'recommended_next_step': (
                "Keep the implemented objective, but tighten the mathematical story around sampled CVaR, "
                "regime averaging, and approximate sparsity. If deeper research rigor is needed, add exact "
                "mixed-integer asset selection and a more formal stochastic-programming regime layer."
            ),
        },
    }


def format_reference_model_comparison(comparison: Dict) -> str:
    """
    Format a comparison payload from compare_against_reference_model() as text.
    """
    implemented = comparison['implemented_model']
    reference = comparison['reference_model']
    score_summary = comparison['score_summary']

    lines = []
    lines.append("=" * 110)
    lines.append("MATHEMATICAL MODEL COMPARISON")
    lines.append("=" * 110)
    lines.append(f"Implemented model scope : {implemented['scope']}")
    lines.append(f"Implemented objective   : {implemented['objective']}")
    lines.append(f"Reference model        : {reference['name']}")
    lines.append(f"Reference objective    : {reference['objective']}")
    lines.append("-" * 110)
    lines.append("Criteria".ljust(28) + " | Impl | Ref  | Winner | Rationale")
    lines.append("-" * 110)

    for item in comparison['criteria']:
        lines.append(
            f"{item['criterion'][:28].ljust(28)} | "
            f"{item['implemented_score']:>4} | "
            f"{item['reference_score']:>4} | "
            f"{item['winner'][:6].ljust(6)} | "
            f"{item['rationale']}"
        )

    lines.append("-" * 110)
    lines.append(
        f"Weighted score          | {score_summary['implemented_weighted_score']:>4.2f} | "
        f"{score_summary['reference_weighted_score']:>4.2f}"
    )
    lines.append(f"Overall verdict         : {comparison['verdict']['overall']}")
    lines.append(f"Deployment verdict      : {comparison['verdict']['deployment']}")
    lines.append(f"Recommended next step   : {comparison['verdict']['recommended_next_step']}")
    lines.append("=" * 110)
    return "\n".join(lines)


def _implemented_model_spec(model_scope: Literal['src', 'optimizer']) -> Dict:
    if model_scope == 'src':
        return {
            'scope': 'src (simplified Markowitz + budget constraints)',
            'objective': r"max_x x^T mu - lambda x^T Sigma x",
            'constraints': [
                r"c^T x <= B",
                r"0 <= x_i <= alpha",
                r"sum_i x_i = 1",
            ],
            'features': {
                'expected_return': True,
                'variance_risk': True,
                'liquidity_term_in_objective': False,
                'transaction_penalty_in_objective': False,
                'tail_risk_in_objective': False,
                'regime_modeling': False,
                'diversification_metric': False,
                'discrete_or_sparse_control': False,
                'hybrid_optimization': True,
            },
            'caveats': [
                'Tail risk is only evaluated ex post in Monte Carlo; it is not optimized.',
                'No explicit regime averaging or stochastic-programming layer is present.',
                'Real-estate illiquidity is not represented directly in the objective.',
            ],
        }

    return {
        'scope': 'optimizer (production multi-term objective)',
        'objective': (
            r"min_x x^T Sigma_hat x - lambda mu^T x + gamma c^T x - delta l^T x "
            r"+ theta CVaR_alpha(x) + rho Omega_regime(x) + diversification/concentration regularization"
        ),
        'constraints': [
            r"c^T x <= B",
            r"sum_i x_i = 1",
            r"0 <= x_i <= alpha",
            r"x_i >= 0",
            'active-asset target enforced by projection',
            'sampled scenarios and regime-weighted stress terms in the broader engine',
        ],
        'features': {
            'expected_return': True,
            'variance_risk': True,
            'bayesian_return_estimation': True,
            'liquidity_term_in_objective': True,
            'transaction_penalty_in_objective': True,
            'tail_risk_in_objective': True,
            'regime_modeling': True,
            'diversification_metric': True,
            'discrete_or_sparse_control': True,
            'hybrid_optimization': True,
        },
        'caveats': [
            'Expected returns are now estimated through a Bayesian/Black-Litterman layer, but the quality of the posterior still depends on the quality of the supplied inputs and views.',
            'Exact empirical CVaR with auxiliary eta/u variables is now available in the convex seed stage, while PSO/GA still refine the sampled scalar objective.',
            'Historical regime calibration and rolling validation are supported when return history is supplied, but the platform does not ship with proprietary macro data.',
            'Discrete subset screening is exact over practical shortlist sizes, but full mixed-integer branch-and-bound is not implemented.',
        ],
    }


def _reference_model_spec(reference_name: str) -> Dict:
    return {
        'name': reference_name,
        'objective': (
            r"max_x x^T mu - lambda x^T Sigma x + beta x^T L - tau c^T x - gamma CVaR_alpha(x)"
        ),
        'constraints': [
            r"c^T x <= B",
            r"0 <= x_i <= alpha",
            r"sum_i x_i = 1",
        ],
        'features': {
            'expected_return': True,
            'variance_risk': True,
            'liquidity_term_in_objective': True,
            'transaction_penalty_in_objective': True,
            'tail_risk_in_objective': True,
            'regime_modeling': True,
            'diversification_metric': True,
            'discrete_or_sparse_control': 'conceptual only',
            'hybrid_optimization': True,
        },
        'caveats': [
            'The formulation is theoretically richer, but implementation details for exact discrete handling are not fully specified.',
            'Regime modeling is presented conceptually as stochastic programming; calibration and scenario probabilities are not specified.',
            'Without a concrete solver formulation, computational tractability and reproducibility remain unproven.',
        ],
    }


def _comparison_criteria(model_scope: Literal['src', 'optimizer']) -> List[Dict]:
    if model_scope == 'src':
        rows = [
            ('Mean-variance baseline', 5, 5, 0.9, 'tie', 'Both models retain the Markowitz return-risk core.'),
            ('Liquidity realism', 1, 5, 1.1, 'ref', 'The reference has an explicit liquidity utility term; src does not.'),
            ('Transaction-cost realism', 2, 4, 1.0, 'ref', 'src only uses cost as a budget constraint, not an objective penalty.'),
            ('Tail-risk control', 1, 5, 1.2, 'ref', 'The reference includes CVaR directly in the optimization objective.'),
            ('Regime awareness', 1, 5, 1.1, 'ref', 'The reference explicitly models bull, bear, and neutral regimes.'),
            ('Diversification control', 1, 4, 0.8, 'ref', 'The reference adds D(x)=1-sum x_i^2; src leaves diversification implicit.'),
            ('Solver tractability', 5, 3, 0.8, 'impl', 'The simpler src formulation is easier to solve robustly.'),
            ('Implementation maturity', 5, 2, 1.0, 'impl', 'The src model is coded and reproducible; the reference is mostly conceptual.'),
        ]
    else:
        rows = [
            ('Mean-variance baseline', 5, 5, 0.9, 'tie', 'Both models retain the core expected-return and covariance-risk trade-off.'),
            ('Expected-return estimation', 5, 4, 1.1, 'impl', 'The optimizer now feeds the allocation engine with Bayesian/Black-Litterman posterior returns instead of raw point estimates alone.'),
            ('Liquidity realism', 4, 5, 1.0, 'ref', 'Both address liquidity, but the reference states it more cleanly in the objective.'),
            ('Transaction-cost realism', 4, 4, 0.9, 'tie', 'Both penalize transaction/cost effects in economically sensible ways.'),
            ('Tail-risk control', 5, 5, 1.3, 'tie', 'The optimizer now supports exact empirical CVaR auxiliary-variable seeding and retains sampled tail control in the full objective.'),
            ('Regime awareness', 5, 5, 1.0, 'tie', 'The optimizer can now calibrate bear/neutral/bull regimes from supplied historical or macro-linked data.'),
            ('Sparse/discrete behavior', 5, 3, 0.9, 'impl', 'The optimizer now performs exact discrete subset screening at practical portfolio sizes, while the reference remains conceptual here.'),
            ('Diversification control', 4, 4, 0.8, 'tie', 'Both include anti-concentration logic, though by different mechanisms.'),
            ('Hybrid search practicality', 5, 4, 1.0, 'impl', 'The optimizer has an executable QP-PSO-GA workflow; the reference only sketches the hybrid idea.'),
            ('Deployment readiness', 5, 2, 1.2, 'impl', 'The optimizer is calibrated, testable, supports rolling backtests, and is already wired into the application.'),
        ]

    criteria = []
    for criterion, implemented_score, reference_score, weight, winner, rationale in rows:
        criteria.append(
            {
                'criterion': criterion,
                'implemented_score': int(implemented_score),
                'reference_score': int(reference_score),
                'weight': float(weight),
                'winner': winner,
                'rationale': rationale,
            }
        )
    return criteria


if __name__ == '__main__':
    from data import generate_portfolio_data
    
    data = generate_portfolio_data(n_assets=30, seed=42)
    
    # Test allocation
    x = np.ones(30) / 30  # Equal weight
    
    results = monte_carlo_evaluate(x, data['mu'], data['Sigma'], n_scenarios=2000, seed=42)
    
    print("\nPortfolio Evaluation:")
    print(f"Mean Return: {results['mean_return']:.6f}")
    print(f"Std Dev: {results['std_return']:.6f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"VaR (95%): {results['var_95']:.6f}")
    print(f"Loss CVaR (95%): {results['loss_cvar_95']:.6f}")

    comparison = compare_against_reference_model(model_scope='optimizer')
    print()
    print(format_reference_model_comparison(comparison))
