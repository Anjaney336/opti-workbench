# Hybrid Portfolio Optimization System

Research-grade implementation of a hybrid optimization system for real estate portfolio allocation combining Quadratic Programming (QP), Particle Swarm Optimization (PSO), and Genetic Algorithms (GA).

## Overview

This system provides:

1. **Data Generation** - Correlated stochastic asset models with positive semi-definite covariance matrices
2. **Convex Baseline** - CVXPY-based Quadratic Programming solver for risk-adjusted portfolio optimization
3. **Custom Metaheuristics** - Implementations of PSO and GA from scratch (no black-box libraries)
4. **Hybrid Pipeline** - Sequential optimization: QP → PSO (warm start) → GA (warm start)
5. **Monte Carlo Evaluation** - Unbiased performance assessment via stochastic simulation
6. **Constraint Handling** - Robust projection for budget, allocation, and normalization constraints
7. **Convergence Analysis** - Detailed tracking of optimization progress

## Mathematical Framework

### Optimization Problem

**Maximize:**
$$x^T \mu - \lambda x^T \Sigma x$$

**Subject to:**
- $c^T x \leq B$ (budget constraint)
- $0 \leq x_i \leq \alpha$ (box constraints)
- $\sum_i x_i = 1$ (normalization)

Where:
- $x$: portfolio allocation vector
- $\mu$: expected returns
- $\Sigma$: covariance matrix (guaranteed PSD via $\Sigma = AA^T$)
- $c$: asset costs
- $B$: budget limit
- $\alpha$: maximum per-asset allocation
- $\lambda$: risk aversion parameter

### Methods

1. **Quadratic Programming (QP)**
   - Solver: CVXPY (ECOS/SCS backend)
   - Provides convex baseline solution
   - Guaranteed globally optimal for QP formulation

2. **Particle Swarm Optimization (PSO)**
   - 30 particles, 150 iterations
   - Inertia weight decay: 0.9 → 0.4
   - Cognitive/social components: $c_1 = c_2 = 1.5$
   - Warm start from QP solution
   - Constraint projection after each update

3. **Genetic Algorithm (GA)**
   - Population: 50, generations: 150
   - Tournament selection (size 3)
   - Blend crossover (BLX-α) with α=0.5
   - Gaussian mutation: σ=0.05
   - Elitism: top 15% preserved
   - Warm start from PSO solution

## Project Structure

```
opti/
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
├── data/                   # Generated portfolio data (CSV, NPY)
├── results/                # Optimization results & plots
└── src/
    ├── __init__.py
    ├── data.py             # Data generation & I/O
    ├── utils.py            # Constraint projection, utilities
    ├── qp.py               # Quadratic programming solver
    ├── evaluation.py       # Monte Carlo evaluation
    ├── pso.py              # PSO implementation
    ├── ga.py               # GA implementation
    └── hybrid.py           # Hybrid pipeline coordinator
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
cd c:\Users\hp\Documents\opti
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computations
- `scipy` - Scientific algorithms
- `cvxpy` - Convex optimization
- `matplotlib` - Plotting
- `pandas` - Data I/O

## Execution

### Quick Start

```bash
cd c:\Users\hp\Documents\opti
python main.py
```

This runs the complete pipeline:
1. Generates 40 correlated assets
2. Solves QP optimization
3. Refines with PSO (warm start from QP)
4. Further refines with GA (warm start from PSO)
5. Evaluates all solutions via Monte Carlo (2000 scenarios)
6. Generates convergence plots

### Configuration

Edit `main.py` to customize:

```python
result, data, mc_results = run_complete_pipeline(
    n_assets=40,              # Number of portfolio assets
    lambda_param=2.0,         # Risk aversion parameter
    n_mc_scenarios=2000,      # Monte Carlo samples
    generate_plots=True,      # Create plots
    verbose=True              # Detailed output
)
```

### Custom Usage

#### Generate Portfolio Data Only

```python
from src.data import generate_portfolio_data, save_portfolio_data

data = generate_portfolio_data(n_assets=50, seed=42)
save_portfolio_data(data, output_dir='data')
```

#### Solve QP Only

```python
from src.qp import solve_quadratic_program
from src.data import generate_portfolio_data

data = generate_portfolio_data(n_assets=40, seed=42)
result = solve_quadratic_program(
    mu=data['mu'],
    Sigma=data['Sigma'],
    c=data['c'],
    B=data['B'],
    alpha=data['alpha'],
    lambda_param=2.0
)
print(f"Optimal allocation: {result['x']}")
```

#### Run PSO Only

```python
from src.pso import optimize_with_pso
from src.utils import project_constraints

def objective(x):
    return np.dot(x, mu) - 2.0 * (x @ Sigma @ x)

def constraint_proj(x):
    return project_constraints(x, c, B, alpha)

result = optimize_with_pso(
    objective_fn=objective,
    constraint_proj_fn=constraint_proj,
    x0=x_initial,
    n_particles=30,
    n_iterations=150
)
```

#### Run GA Only

```python
from src.ga import optimize_with_ga

result = optimize_with_ga(
    objective_fn=objective,
    constraint_proj_fn=constraint_proj,
    x0=x_initial,
    population_size=50,
    n_generations=150
)
```

#### Complete Hybrid Pipeline

```python
from src.hybrid import hybrid_optimize

result = hybrid_optimize(
    mu=data['mu'],
    Sigma=data['Sigma'],
    c=data['c'],
    B=data['B'],
    alpha=data['alpha'],
    lambda_param=2.0,
    verbose=True
)

# Access results
x_qp = result['x_qp']          # QP solution
x_pso = result['x_pso']        # PSO solution
x_ga = result['x_ga']          # GA solution
x_final = result['x_final']    # Best overall
print(f"Best method: {result['best_method']}")
```

#### Monte Carlo Evaluation

```python
from src.evaluation import monte_carlo_evaluate, compare_portfolios

# Evaluate single portfolio
stats = monte_carlo_evaluate(x, mu, Sigma, n_scenarios=2000)
print(f"Mean return: {stats['mean_return']:.6f}")
print(f"Sharpe ratio: {stats['sharpe_ratio']:.4f}")
print(f"VaR (95%): {stats['var_95']:.6f}")

# Compare multiple allocations
allocations = {'QP': x_qp, 'PSO': x_pso, 'GA': x_ga}
comparison = compare_portfolios(allocations, mu, Sigma, n_scenarios=2000)
```

## Output

### Console Output Example

```
================================================================================
             RESEARCH-GRADE HYBRID PORTFOLIO OPTIMIZATION
================================================================================

[STAGE 0] DATA GENERATION
────────────────────────────────────────────────────────────────────────────────

✓ Portfolio Data Generated:
  Assets: 40
  Expected return range: [0.0512, 0.1497]
  Budget constraint: 40.0000
  Max allocation: 0.3000

✓ Covariance Matrix Properties:
  Min eigenvalue: 1.24e-08 (PSD: ✓)
  Max eigenvalue: 3.187643
  Condition number: 2.57e+08
  Trace: 0.800000

[STAGE 1] QUADRATIC PROGRAMMING (Baseline)
────────────────────────────────────────────────────────────────────────────────
...
```

### Generated Artifacts

**Data Files** (`data/`):
- `asset_parameters.csv` - Expected returns and costs
- `covariance_matrix.npy` - PSD covariance matrix
- `metadata.json` - Portfolio configuration

**Results** (`results/`):
- `convergence.png` - PSO and GA convergence curves
- `allocations.png` - Allocation comparison chart

## Key Features

### Numerical Stability
- PSD covariance via Cholesky decomposition
- Eigenvalue clipping to prevent rank deficiency
- Iterative constraint projection
- Velocity clipping in PSO for stability

### Constraint Handling
- Budget constraint: $c^T x \leq B$
- Box constraints: $0 \leq x_i \leq \alpha$
- Normalization: $\sum x = 1$
- Robust projection with feasibility verification

### Vectorization
- All core computations use NumPy broadcasting
- Efficient matrix operations (Σ x, x^T Σ x)
- Batch Monte Carlo evaluation

### Reproducibility
- Fixed random seeds for all components
- Deterministic algorithm flow
- Detailed logging and history tracking

## Performance

**Typical Runtime:**
- Data generation: < 1s
- QP solving: < 2s
- PSO (30 particles, 150 iter): 5-10s
- GA (50 pop, 150 gen): 10-15s
- MC evaluation (2000 scenarios): 2-3s
- **Total: ~25-30s for full pipeline**

**Problem Sizes Tested:**
- Assets: 20-100
- Budget constraint satisfaction: ✓ (verified)
- Allocation normalization: ✓ (≤1e-8 error)

## Mathematical Verification

### Covariance Matrix
```python
# Guaranteed PSD by construction
A = random matrix
Σ = A A^T  ✓ Always PSD

# Eigenvalue properties verified
evals = np.linalg.eigvals(Sigma)
assert all(evals >= -1e-9)  # All non-negative
```

### Feasibility
```python
# All solutions verified for constraint satisfaction
assert abs(sum(x) - 1.0) < 1e-6
assert all(x >= -1e-6)
assert all(x <= alpha + 1e-6)
assert c @ x <= B + 1e-6
```

### Convergence
```python
# PSO and GA show monotonic improvement (with high probability)
# Convergence plots saved to results/convergence.png
```

## References

1. **Markowitz, H.** (1952). "Portfolio Selection". Journal of Finance.
2. **Kennedy, J. & Eberhart, R.** (1995). "Particle Swarm Optimization". ICNN.
3. **Goldberg, D.E.** (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning".
4. **Boyd, S. & Vandenberghe, L.** (2004). "Convex Optimization". Cambridge University Press.

## License

MIT License

## Notes

- This implementation prioritizes **mathematical correctness** and **reproducibility** over computational speed
- Suitable for research, academic validation, and production deployment
- Can be extended with additional constraints or alternative objective functions
- PSO and GA implementations are fully custom—no reliance on scikit-optimize, DEAP, or similar libraries

## Support

For issues or questions:
1. Verify dependencies: `pip install -r requirements.txt`
2. Check random seeds for reproducibility
3. Review console output for constraint violations
4. Inspect convergence plots for optimizer behavior
5. Validate covariance matrix properties via eigenvalue analysis
