"""
Data generation module for real estate portfolio optimization.
Generates correlated asset data using stochastic models.
"""

import numpy as np
import pandas as pd
from pathlib import Path


SYNTHETIC_DATASET_SPECS = {
    "high_growth": {
        "label": "High Growth",
        "return_range": (0.10, 0.20),
        "covariance_scale": 1.18,
        "sector_probs": [0.25, 0.35, 0.40],
        "sector_return": {"Residential": 0.98, "Commercial": 1.10, "Land": 1.25},
        "sector_risk": {"Residential": 0.98, "Commercial": 1.10, "Land": 1.28},
        "sector_liquidity": {"Residential": 0.82, "Commercial": 0.72, "Land": 0.48},
        "cost_scale": {"Residential": 0.98, "Commercial": 1.04, "Land": 1.15},
        "tx_scale": {"Residential": 0.011, "Commercial": 0.017, "Land": 0.028},
    },
    "stable_income": {
        "label": "Stable Income",
        "return_range": (0.05, 0.11),
        "covariance_scale": 0.72,
        "sector_probs": [0.58, 0.30, 0.12],
        "sector_return": {"Residential": 1.05, "Commercial": 0.98, "Land": 0.88},
        "sector_risk": {"Residential": 0.78, "Commercial": 0.92, "Land": 1.12},
        "sector_liquidity": {"Residential": 0.92, "Commercial": 0.80, "Land": 0.55},
        "cost_scale": {"Residential": 0.96, "Commercial": 1.03, "Land": 1.08},
        "tx_scale": {"Residential": 0.010, "Commercial": 0.014, "Land": 0.022},
    },
    "risky_momentum": {
        "label": "Risky Momentum",
        "return_range": (0.09, 0.18),
        "covariance_scale": 1.45,
        "sector_probs": [0.22, 0.33, 0.45],
        "sector_return": {"Residential": 0.96, "Commercial": 1.08, "Land": 1.30},
        "sector_risk": {"Residential": 1.02, "Commercial": 1.15, "Land": 1.35},
        "sector_liquidity": {"Residential": 0.78, "Commercial": 0.66, "Land": 0.42},
        "cost_scale": {"Residential": 0.99, "Commercial": 1.05, "Land": 1.20},
        "tx_scale": {"Residential": 0.011, "Commercial": 0.018, "Land": 0.030},
    },
    "low_liquidity": {
        "label": "Low Liquidity",
        "return_range": (0.06, 0.14),
        "covariance_scale": 1.10,
        "sector_probs": [0.35, 0.32, 0.33],
        "sector_return": {"Residential": 0.99, "Commercial": 1.02, "Land": 1.12},
        "sector_risk": {"Residential": 0.96, "Commercial": 1.06, "Land": 1.18},
        "sector_liquidity": {"Residential": 0.55, "Commercial": 0.42, "Land": 0.25},
        "cost_scale": {"Residential": 1.00, "Commercial": 1.06, "Land": 1.18},
        "tx_scale": {"Residential": 0.014, "Commercial": 0.020, "Land": 0.034},
    },
    "commercial_heavy": {
        "label": "Commercial Heavy",
        "return_range": (0.07, 0.15),
        "covariance_scale": 1.00,
        "sector_probs": [0.20, 0.60, 0.20],
        "sector_return": {"Residential": 0.95, "Commercial": 1.16, "Land": 1.02},
        "sector_risk": {"Residential": 0.88, "Commercial": 1.10, "Land": 1.08},
        "sector_liquidity": {"Residential": 0.82, "Commercial": 0.78, "Land": 0.50},
        "cost_scale": {"Residential": 0.98, "Commercial": 1.07, "Land": 1.10},
        "tx_scale": {"Residential": 0.010, "Commercial": 0.016, "Land": 0.024},
    },
    "residential_defensive": {
        "label": "Residential Defensive",
        "return_range": (0.05, 0.10),
        "covariance_scale": 0.68,
        "sector_probs": [0.70, 0.20, 0.10],
        "sector_return": {"Residential": 1.08, "Commercial": 0.95, "Land": 0.82},
        "sector_risk": {"Residential": 0.70, "Commercial": 0.96, "Land": 1.08},
        "sector_liquidity": {"Residential": 0.95, "Commercial": 0.76, "Land": 0.48},
        "cost_scale": {"Residential": 0.94, "Commercial": 1.02, "Land": 1.12},
        "tx_scale": {"Residential": 0.009, "Commercial": 0.014, "Land": 0.022},
    },
    "land_speculative": {
        "label": "Land Speculative",
        "return_range": (0.08, 0.17),
        "covariance_scale": 1.35,
        "sector_probs": [0.18, 0.22, 0.60],
        "sector_return": {"Residential": 0.92, "Commercial": 1.00, "Land": 1.35},
        "sector_risk": {"Residential": 0.94, "Commercial": 1.04, "Land": 1.40},
        "sector_liquidity": {"Residential": 0.75, "Commercial": 0.68, "Land": 0.30},
        "cost_scale": {"Residential": 0.97, "Commercial": 1.04, "Land": 1.24},
        "tx_scale": {"Residential": 0.011, "Commercial": 0.017, "Land": 0.035},
    },
    "inflation_hedge": {
        "label": "Inflation Hedge",
        "return_range": (0.07, 0.16),
        "covariance_scale": 0.94,
        "sector_probs": [0.34, 0.28, 0.38],
        "sector_return": {"Residential": 1.02, "Commercial": 1.04, "Land": 1.18},
        "sector_risk": {"Residential": 0.88, "Commercial": 0.98, "Land": 1.12},
        "sector_liquidity": {"Residential": 0.84, "Commercial": 0.72, "Land": 0.46},
        "cost_scale": {"Residential": 0.97, "Commercial": 1.04, "Land": 1.16},
        "tx_scale": {"Residential": 0.010, "Commercial": 0.015, "Land": 0.026},
    },
    "diversified_core": {
        "label": "Diversified Core",
        "return_range": (0.06, 0.13),
        "covariance_scale": 0.90,
        "sector_probs": [0.44, 0.34, 0.22],
        "sector_return": {"Residential": 1.01, "Commercial": 1.04, "Land": 1.08},
        "sector_risk": {"Residential": 0.86, "Commercial": 0.98, "Land": 1.12},
        "sector_liquidity": {"Residential": 0.88, "Commercial": 0.77, "Land": 0.54},
        "cost_scale": {"Residential": 0.96, "Commercial": 1.03, "Land": 1.10},
        "tx_scale": {"Residential": 0.010, "Commercial": 0.015, "Land": 0.024},
    },
}


def available_dataset_names() -> list[str]:
    return ["auto", *SYNTHETIC_DATASET_SPECS.keys()]


def resolve_dataset_name(dataset: str, scenario: str, seed: int) -> str:
    dataset = (dataset or "auto").lower().strip()
    if dataset in SYNTHETIC_DATASET_SPECS:
        return dataset

    scenario_buckets = {
        "positive": ["high_growth", "commercial_heavy", "land_speculative"],
        "neutral": ["diversified_core", "stable_income", "inflation_hedge"],
        "negative": ["residential_defensive", "low_liquidity", "risky_momentum"],
    }
    bucket = scenario_buckets.get(scenario, scenario_buckets["neutral"])
    return bucket[int(seed) % len(bucket)]


def generate_portfolio_data(n_assets: int = 40, 
                           expected_return_range: tuple = (0.05, 0.15),
                           budget: float = 1.0,
                           max_allocation: float = 0.3,
                           seed: int = 42) -> dict:
    """
    Generate correlated asset data for portfolio optimization.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets (30-50 recommended)
    expected_return_range : tuple
        Range for expected returns (min, max)
    budget : float
        Total budget constraint (typically 1.0 for normalized)
    max_allocation : float
        Maximum allocation per asset
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Contains μ (expected returns), Σ (covariance), c (costs), B (budget)
    
    Notes:
    ------
    Covariance matrix generated via: Σ = A A^T where A is random.
    This ensures positive semi-definiteness by construction.
    """
    np.random.seed(seed)
    
    # 1. Expected returns: random samples from uniform distribution
    mu = np.random.uniform(expected_return_range[0], 
                          expected_return_range[1], 
                          n_assets)
    
    # 2. Generate positive semi-definite covariance via random matrix A
    #    Σ = A A^T is guaranteed PSD
    A = np.random.randn(n_assets, n_assets)
    A = np.tril(A)  # Lower triangular to reduce redundancy
    Sigma = A @ A.T
    
    # 3. Normalize covariance matrix (scale to reasonable magnitude)
    # Target: typical variance around 0.01-0.04
    trace = np.trace(Sigma)
    Sigma = Sigma / (trace / (n_assets * 0.02))
    
    # Ensure positive semi-definite by cleaning up numerical errors
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-8)
    Sigma = evecs @ np.diag(evals) @ evecs.T
    
    # 4. Costs: normally distributed around 1.0 (unit cost assets)
    c = np.abs(np.random.normal(1.0, 0.2, n_assets))
    
    # 5. Budget and constraints
    B = budget * np.sum(c)  # Budget in monetary units
    alpha = max_allocation  # Max allocation per asset
    
    data = {
        'mu': mu,                      # Expected returns (n_assets,)
        'Sigma': Sigma,                # Covariance matrix (n_assets, n_assets)
        'c': c,                        # Costs (n_assets,)
        'B': B,                        # Budget constraint
        'alpha': alpha,                # Max allocation per asset
        'n_assets': n_assets,
        'seed': seed
    }
    
    return data


def save_portfolio_data(data: dict, output_dir: str = 'data') -> None:
    """
    Save portfolio data to CSV and NumPy files.
    
    Parameters:
    -----------
    data : dict
        Portfolio data dictionary from generate_portfolio_data
    output_dir : str
        Directory to save files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save vectors to CSV
    df = pd.DataFrame({
        'expected_return': data['mu'],
        'cost': data['c']
    })
    df.to_csv(f"{output_dir}/asset_parameters.csv", index_label='asset_id')
    
    # Save covariance matrix
    np.save(f"{output_dir}/covariance_matrix.npy", data['Sigma'])
    
    # Save metadata
    metadata = {
        'n_assets': data['n_assets'],
        'budget': float(data['B']),
        'max_allocation': float(data['alpha']),
        'seed': data['seed']
    }
    
    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Data saved to {output_dir}/")
    print(f"  - Covariance matrix: {data['Sigma'].shape} (PSD)")
    print(f"  - Assets: {data['n_assets']}")
    print(f"  - Budget: {data['B']:.4f}")
    print(f"  - Max allocation: {data['alpha']:.4f}")


def load_portfolio_data(data_dir: str = 'data') -> dict:
    """
    Load portfolio data from saved files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing saved data files
        
    Returns:
    --------
    dict : Loaded portfolio data
    """
    import json
    
    df = pd.read_csv(f"{data_dir}/asset_parameters.csv", index_col='asset_id')
    mu = df['expected_return'].values
    c = df['cost'].values
    Sigma = np.load(f"{data_dir}/covariance_matrix.npy")
    
    with open(f"{data_dir}/metadata.json") as f:
        metadata = json.load(f)
    
    data = {
        'mu': mu,
        'Sigma': Sigma,
        'c': c,
        'B': metadata['budget'],
        'alpha': metadata['max_allocation'],
        'n_assets': metadata['n_assets'],
        'seed': metadata['seed']
    }
    
    return data


if __name__ == '__main__':
    # Example usage
    data = generate_portfolio_data(n_assets=40, seed=42)
    save_portfolio_data(data, output_dir='data')
    
    # Verify PSD
    evals = np.linalg.eigvals(data['Sigma'])
    print(f"\nCovariance matrix properties:")
    print(f"  Min eigenvalue: {np.min(evals):.2e}")
    print(f"  Max eigenvalue: {np.max(evals):.4f}")
    print(f"  Condition number: {np.max(evals) / np.min(evals):.2e}")
