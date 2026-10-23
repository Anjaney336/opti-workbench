# Opti-Workbench — calculation and decision guide

Do independent implementations solve the same tail-risk allocation problem?

## Match the problem

Both solvers minimise empirical 95% CVaR with identical long-only, fully invested weights and a 30% per-asset cap. This is a continuous optimisation problem; there are no integer positions or cardinality constraints.

## Check numerical correctness

Compare feasibility and objective values across 12 training windows. Maximum objective disagreement is approximately 9.73e-10. Solver agreement validates the implementation, not the realism of the data.

## Evaluate after training

Use four chronological test folds for each of three synthetic seeds, accounting for weight drift and 10bp turnover costs. Equal weighting is a feasible benchmark; retain both winning and losing test paths.

## Calculation

CVaRα(w) = minη {η + mean[max(−Rw − η, 0)] / (1 − α)}; α = 0.95, sum(w) = 1, 0 ≤ w ≤ 0.30.

## Graph interpretation

### Matched solver evaluation

Read solver agreement separately from held-out investment performance. Two implementations can agree exactly and still lose money out of sample. The three seeded panels are synthetic and do not establish an empirical market edge.

## Review evidence

matched_benchmark.py · test_matched_benchmark.py · results/matched/solver_audit.csv

Demonstrates: Constrained optimisation, tail-risk measurement and independent solver verification.

All values are existing recorded outputs. No new performance experiment was run for this presentation update. Refer to the README and result JSON for data sources, access dates, units and limitations.
