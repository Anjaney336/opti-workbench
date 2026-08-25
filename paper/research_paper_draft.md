# Current Hybrid Portfolio Optimization Engine: Black-Litterman Returns, Exact-CVaR QP Seeding, and Adaptive Metaheuristic Refinement

**Author:** [Your Name]  
**Affiliation:** [Your Institution or Organization]  
**Date:** August 3, 2026

## Abstract

This paper documents the portfolio-optimization engine currently implemented in this repository, replacing earlier writeups that mixed together older synthetic experiments and a separate research pipeline. The live system is a constrained long-only optimizer that combines a Black-Litterman-style expected-return engine, covariance conditioning, regime-aware tail-risk modeling, an exact empirical-CVaR quadratic-program seed, warm-started Particle Swarm Optimization (PSO), warm-started Genetic Algorithm (GA) refinement, and an adaptive hybrid consensus layer. The implemented objective is a multi-term engineering objective rather than a plain Markowitz quadratic. It jointly accounts for covariance risk, expected return, transaction cost, liquidity reward, empirical CVaR, diversification-gap control, concentration penalty, regime stress, tracking penalty, and an entropy-style diversification bonus. In the latest successful live run on July 12, 2026, the system solved a 12-asset balanced universe with a 4-asset active target and selected the `HYBRID` workflow. Within that workflow, the strongest candidate by objective came from the GA stage, while the final blended hybrid portfolio remained nearly tied. The selected published hybrid output allocated capital to Asset 3 (`22.16%`), Asset 4 (`31.86%`), Asset 6 (`24.41%`), and Asset 11 (`21.56%`), with expected return `0.08807`, risk `0.00655`, Sharpe `1.0880`, and loss CVaR at `95%` confidence of `0.08844`. The measured results show that all four solver pathways are close on objective value, with QP offering the best Sharpe in this run and GA achieving the numerically lowest objective. The engineering conclusion is therefore narrow and defensible: the current hybrid stack is best understood as a controlled refinement architecture over a strong feasible convex seed, not as proof that a metaheuristic is universally superior.

**Keywords:** portfolio optimization, Black-Litterman, CVaR, quadratic programming, particle swarm optimization, genetic algorithm, hybrid optimization, regime-aware optimization

## 1. Introduction

The previous paper associated with this project no longer matched the optimizer that is actually running in the codebase. Earlier drafts described either a simpler synthetic mean-variance study or a broader `HouseTS` research workflow, while the live engine now solves a more specific engineering problem: how to produce a feasible, interpretable, risk-aware portfolio under a multi-term objective with deterministic seeding and stochastic refinement.

That distinction matters. A technically credible paper should describe the implemented system, not an earlier architecture. The current engine is neither a pure textbook Markowitz optimizer nor a black-box search routine. It is a layered optimization pipeline with three goals:

1. produce a feasible allocation under budget, box, simplex, and active-asset constraints
2. optimize a richer portfolio score than variance minus return alone
3. preserve traceability so each optimization stage can be inspected and defended

The current repository achieves that by combining a convex seed stage with two stochastic refinement stages and a final hybrid selection layer. The seed stage provides feasibility and structure. The PSO stage explores neighboring feasible regions. The GA stage introduces mutation-driven search pressure. The hybrid wrapper then evaluates the seed, swarm, mutation, and blended candidates under a common hybrid score and publishes the strongest feasible result.

## 2. System Scope

This paper is intentionally scoped to the current live optimizer implementation in:

- `optimizer/engine.py`
- `optimizer/returns_engine.py`
- `optimizer/solvers.py`
- `optimizer/advanced.py`
- `results/live_runs/d3858571df2749b8ada6c26629a872e0/`

It does not treat older experimental notebooks, prior synthetic studies, or the separate broader real-estate research pipeline as authoritative for the claims made here.

## 3. Problem Definition

The engine solves a long-only portfolio-allocation problem over a small curated asset universe. In the latest successful live run, the input payload used:

- `12` assets
- active-assets target `K = 4`
- budget `B = 5.0`
- per-asset maximum allocation `0.35`
- seed `42`
- Monte Carlo paths `300`
- method `AUTO`, which resolves to the `HYBRID` workflow
- return engine `BLACK_LITTERMAN`

The portfolio decision vector is:

```text
x = [x1, x2, ..., xn]
```

with constraints:

```text
sum_i x_i = 1
0 <= x_i <= 0.35
c^T x <= B
```

and with an enforced active-asset target applied through feasibility projection and sparse support selection.

## 4. Return Engine

The current engine does not optimize directly on the raw input `mu` vector. It first passes returns through a Bayesian/Black-Litterman layer implemented in `optimizer/returns_engine.py`.

The return procedure is:

1. build an investability score from liquidity divided by cost
2. normalize that score into pseudo-market weights
3. form a market-implied prior `pi = lambda * Sigma * w_market`
4. blend empirical and prior views using Bayesian shrinkage
5. solve an absolute-view Black-Litterman posterior return estimate
6. rescale the posterior to the reference range of the raw return vector

The implemented posterior is summarized in the code as:

```text
mu_BL = [(tau Sigma)^-1 + P^T Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P^T Omega^-1 q]
```

with the latest run using:

- prior strength `0.55`
- `tau = 0.08`
- view confidence `0.65`

This is an engineering prior-smoothing device. It should be read as a stabilization layer over the optimizer inputs, not as a claim that the Black-Litterman posterior is uniquely correct for all datasets.

## 5. Covariance Conditioning and Regime Construction

Before optimization, the covariance matrix is:

1. symmetrized
2. diagonally regularized using a trace-scaled ridge
3. shrinkage-adjusted toward its diagonal

The full objective uses:

```text
Sigma_hat = (1 - phi) Sigma + phi diag(Sigma)
```

where the request-level shrinkage input is `phi = 0.15`, although the calibration sweep in the latest run selected `calibrated_phi = 0.0` for the probe problem.

If historical returns are available, the engine calibrates Bear, Neutral, and Bull regimes from return history. If they are absent, as in the latest live run, the engine constructs three scenario regimes from scaled return and covariance tensors. Scenario returns are then sampled from those regime-conditioned distributions and used to compute empirical loss CVaR.

## 6. Implemented Objective

The live objective in `optimizer/solvers.py` is a multi-term minimization:

```text
J(x) =
    x^T Sigma_hat x
  - lambda mu^T x
  + gamma C(x)
  - delta L(x)
  + theta CVaR_alpha(x)
  + kappa D_gap(x)
  + psi HHI(x)
  + rho Omega_regime(x)
  + omega ||x - x_ref||_2^2
  - epsilon H(x)
```

where:

- `x^T Sigma_hat x` is covariance risk
- `mu^T x` is expected return reward
- `C(x)` is the weighted transaction-cost term
- `L(x)` is the weighted liquidity reward
- `CVaR_alpha(x)` is exact empirical downside tail risk from scenario returns
- `D_gap(x)` penalizes deviation from the active-assets concentration target
- `HHI(x) = sum_i x_i^2` penalizes concentration
- `Omega_regime(x)` combines regime dispersion and regime stress
- `||x - x_ref||_2^2` is tracking error to the anchor portfolio
- `H(x)` is an entropy-style diversification bonus

The latest run used the following coefficients:

- `lambda = 1.0` at request level
- liquidity weight `0.35`
- cost sensitivity `0.15`
- tail-risk weight `0.45`
- diversification weight `0.30`
- concentration penalty `0.18`
- regime aversion `0.28`
- tracking penalty `0.12`
- `alpha = 0.95` for CVaR

One detail matters for scientific accuracy: the exact QP seed solves a convex approximation of the common objective while preserving exact empirical CVaR through auxiliary variables. The seed stage therefore remains faithful to the live score structure without attempting to force every downstream nonconvex preference into one closed-form convex program.

## 7. Solver Architecture

### 7.1 Exact-CVaR QP Seed

The QP stage in the current engine is stronger than a standard mean-variance seed. When enabled, it uses:

- exact empirical CVaR auxiliary variables `eta` and `u`
- convex concentration-gap slack
- regime dispersion and regime stress terms
- tracking penalty
- optional exact discrete subset enumeration for the active set

For the current problem size, the engine can enumerate small active subsets and solve an exact discrete seed rather than only a relaxed dense seed. This is materially different from the earlier paper, which described QP only as a plain quadratic baseline.

### 7.2 PSO Refinement

The PSO stage is warm-started from the QP solution. In the current implementation it uses:

- `50` particles
- `50` iterations
- inertia `0.68`
- cognitive coefficient `1.45`
- social coefficient `1.55`
- random velocity perturbations
- feasibility projection after every update

The swarm searches under the full multi-term objective, not under a separate toy score.

### 7.3 GA Refinement

The GA stage is warm-started from the PSO result. The current implementation uses:

- population size `100`
- `50` generations
- mutation rate `0.20`
- elite count `6`
- tournament-style parent selection
- convex mixing crossover
- Gaussian and sparse-mask mutations
- feasibility projection after offspring creation

### 7.4 Hybrid Consensus

The hybrid wrapper runs `QP -> PSO -> GA`, then:

1. evaluates each stage output under its model score
2. forms adaptive weights via a softmax over negative stage objectives
3. creates a blended candidate
4. adds a consensus penalty that discourages drift from strong stage solutions
5. selects the best candidate among `QP`, `PSO`, `GA`, and `BLEND`

This means the published `HYBRID` output is not necessarily the arithmetic blend. It is the best candidate under the hybrid consensus score.

## 8. Feasibility and Validation Logic

Every candidate is projected and validated against:

- simplex feasibility
- nonnegativity
- maximum allocation
- budget feasibility
- active-asset support

The current engine also records:

- candidate histories
- stage traces
- objective-surface diagnostics
- decision-range analysis
- Monte Carlo summaries
- sensitivity points
- risk-target gap

This is an engineering decision-support system, not just a single optimization call.

## 9. Latest Live Results

The latest successful end-to-end live run completed on **July 12, 2026** under job id `d3858571df2749b8ada6c26629a872e0`.

### 9.1 Model Comparison

| Method | Expected Return | Risk | Sharpe | Objective | Loss CVaR 95 |
|---|---:|---:|---:|---:|---:|
| QP | 0.086893 | 0.006079 | 1.114507 | -0.237571 | 0.084487 |
| PSO | 0.088314 | 0.006645 | 1.083387 | -0.237928 | 0.090019 |
| GA | 0.088015 | 0.006530 | 1.089150 | **-0.237977** | 0.088368 |
| HYBRID | 0.088070 | 0.006552 | 1.088004 | -0.237932 | 0.088442 |

Three observations are important:

1. all four methods are very close on the objective, so inflated claims of decisive optimizer dominance are not supported
2. the standalone `GA` output is numerically best on the implemented objective in this run
3. the `QP` seed remains highly competitive and in fact has the best Sharpe ratio in this run

That is exactly the kind of result a strong engineering paper should report directly rather than overselling the hybrid label.

### 9.2 Selected Hybrid Output

The published selected output under the `HYBRID` workflow was:

- Asset 3: `22.16%`
- Asset 4: `31.86%`
- Asset 6: `24.41%`
- Asset 11: `21.56%`

with:

- expected return `0.08807`
- risk `0.00655`
- Sharpe `1.08800`
- diversification count `4`
- budget used `0.01819`
- average liquidity `0.81089`
- loss VaR 95 `0.05157`
- loss CVaR 95 `0.08844`

The hybrid stage weights used inside the consensus logic were almost equal:

- QP: `0.333247`
- PSO: `0.333369`
- GA: `0.333383`

Despite that near-equal weighting, the hybrid selector still preferred the GA-stage candidate over the blended candidate in this run.

### 9.3 Objective Breakdown of the Published Hybrid Output

The final hybrid objective breakdown was:

| Term | Value |
|---|---:|
| risk term | 0.006552 |
| return term | -0.039631 |
| cost term | 0.004366 |
| liquidity term | -0.283810 |
| tail-risk term | 0.039799 |
| diversification term | 0.000014 |
| concentration term | 0.046211 |
| regime term | 0.000675 |
| tracking term | 0.020807 |
| entropy term | -0.032962 |
| hybrid consensus term | 0.000047 |

The largest magnitude contributor is the liquidity reward, followed by return reward, tail risk, concentration, and tracking. That is consistent with the current design: the optimizer is not merely minimizing variance, and its chosen allocation is heavily shaped by liquidity and diversification structure.

## 10. Risk Calibration Behavior

The engine includes a calibration sweep intended to move the seed-stage risk toward a target portfolio risk of:

```text
0.0087
```

In the latest run, the calibration record reports:

- target risk `0.0087`
- probe achieved risk `0.015947`
- absolute calibration error `0.007247`
- calibrated lambda `0.45`
- calibrated phi `0.0`
- `risk_target_hit = false`

The final solved model outputs then landed between roughly `0.00608` and `0.00664` realized risk depending on the solver path. The correct interpretation is not that calibration is broken, but that the current probe-based sweep is an approximate tuning layer whose target was not exactly achieved in this run. A high-grade engineering paper should say that explicitly.

## 11. Discussion

### 11.1 What the Results Support

The current evidence supports four narrow claims:

1. the exact-CVaR QP seed produces a strong feasible baseline
2. warm-started PSO and GA can refine the common multi-term objective without violating feasibility
3. the hybrid wrapper provides a disciplined mechanism for comparing and blending stage outputs
4. on the latest run, solver differences are real but small, with the strongest solutions clustered tightly together

### 11.2 What the Results Do Not Support

The current run does not support:

- universal superiority of hybrid search over convex optimization
- a claim that blending always beats the best stage-specific candidate
- a claim that calibration always hits the requested risk target
- broad out-of-sample generalization from one live run

### 11.3 Engineering Interpretation

The strongest engineering takeaway is that the system architecture is sound even though the optimizer ranking is close. That is a good result, not a weak one. If a convex seed, a swarm refinement, a genetic refinement, and a hybrid selector all converge to almost the same region under the same objective, then the pipeline is behaving coherently. The implementation is adding search robustness and traceability more than it is manufacturing large artificial performance gaps.

## 12. Limitations

This paper is limited by the current implementation context.

- The latest live result is a single run on a 12-asset balanced universe.
- Historical returns and macro signals were `null` in that run, so regime inputs were generated from fallback regime scaling rather than true history calibration.
- The objective is an engineering utility, not a direct investor-indifference function estimated from market data.
- The liquidity and cost inputs in the live run were supplied asset attributes, not directly audited execution records.
- The paper documents the implemented engine faithfully, but it is not a large-sample empirical asset-pricing study.

## 13. Recommended Next Improvements

The most justified next steps are:

1. run repeated-seed studies on the current engine and report mean and dispersion, not only one live run
2. feed historical return matrices through the same live path so regime calibration and rolling backtests are exercised in production mode
3. improve the risk-calibration loop so the achieved probe risk is closer to the target more consistently
4. separate seed-objective, refinement-objective, and publication-objective reporting more explicitly in the artifacts
5. generate a manuscript-ready results table automatically from `result.json` to prevent documentation drift

## 14. Conclusion

The current optimizer in this repository is a materially more advanced system than the earlier paper described. It uses Black-Litterman-style return conditioning, covariance regularization, exact empirical-CVaR convex seeding, optional discrete subset screening, warm-started PSO, warm-started GA, adaptive hybrid blending, and strict feasibility repair. In the latest successful live run dated July 12, 2026, the `HYBRID` workflow was selected, but the strongest candidate inside that workflow came from the GA stage, while QP remained the best Sharpe performer and the final blended candidate was slightly weaker than the best stage-specific candidate.

That is the correct high-grade engineering conclusion: the system is credible because it is internally consistent, constrained, inspectable, and honest about close results. The hybrid stack should be presented as a robust optimization architecture built on top of a strong convex seed, not as a marketing claim that one metaheuristic categorically defeats the rest.

## References

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77-91.

Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal, 48*(5), 28-43.

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*, 1942-1948.

Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.

Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk, 2*(3), 21-41.
