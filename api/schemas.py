"""
Schemas for the multi-model portfolio optimization platform.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssetSpec(BaseModel):
    name: str
    mu: float = Field(..., ge=0.0)
    sigma: float = Field(..., gt=0.0)
    liquidity: float = Field(..., ge=0.0, le=1.0)
    cost: float = Field(..., gt=0.0)


class OptimizeRequest(BaseModel):
    asset_count: int = Field(10, ge=5, le=15)
    active_assets_target: int = Field(4, ge=2, le=15)
    budget: float = Field(5.0, gt=0.0)
    max_allocation: float = Field(0.35, gt=0.0, le=1.0)
    lambda_value: float = Field(1.0, gt=0.0, le=5.0)
    liquidity_weight: float = Field(0.35, ge=0.0, le=2.0)
    cost_sensitivity: float = Field(0.15, ge=0.0, le=2.0)
    tail_risk_weight: float = Field(0.45, ge=0.0, le=3.0)
    diversification_weight: float = Field(0.30, ge=0.0, le=3.0)
    concentration_penalty: float = Field(0.18, ge=0.0, le=3.0)
    regime_aversion: float = Field(0.28, ge=0.0, le=3.0)
    tracking_penalty: float = Field(0.12, ge=0.0, le=3.0)
    covariance_shrinkage: float = Field(0.15, ge=0.0, le=0.95)
    cvar_alpha: float = Field(0.95, ge=0.80, le=0.995)
    method: str = Field("AUTO", description="AUTO | QP | PSO | GA | HYBRID")
    dataset: str = Field("balanced")
    seed: int = Field(42, ge=0)
    monte_carlo_paths: int = Field(500, ge=200, le=5000)
    assets: list[AssetSpec] | None = None
    exact_cvar_seed: bool = True
    exact_discrete_selection: bool = True
    planning_horizon: int = Field(3, ge=2, le=24)
    backtest_lookback: int = Field(12, ge=4, le=120)
    historical_returns: list[list[float]] | None = None
    macro_signal: list[float] | None = None
    return_engine: str = Field("BLACK_LITTERMAN", description="RAW | BAYESIAN | BLACK_LITTERMAN")
    return_prior_strength: float = Field(0.55, ge=0.0, le=0.95)
    bl_tau: float = Field(0.08, gt=0.0, le=1.0)
    bl_view_confidence: float = Field(0.65, gt=0.0, lt=1.0)


class ModelExplanation(BaseModel):
    objective_function: str
    solver_steps: list[str]
    why_this_allocation: str
    asset_justifications: list[str]


class ModelOutput(BaseModel):
    weights: list[float]
    expected_return: float
    risk: float
    sharpe: float
    diversification: int
    active_assets_target: int
    objective_value: float
    budget_used: float
    average_liquidity: float
    valid: bool
    objective_breakdown: dict[str, float]
    regime_metrics: dict[str, Any]
    feasibility_validation: dict[str, Any] | None = None
    risk_target_gap: float | None = None
    explanation: ModelExplanation


class ComparisonRow(BaseModel):
    model: str
    expected_return: float
    risk: float
    sharpe: float
    diversification: int


class FrontierPoint(BaseModel):
    lambda_value: float
    expected_return: float
    risk: float


class MonteCarloSummary(BaseModel):
    mean_return: float
    std_return: float
    p05: float
    p95: float
    loss_var_95: float | None = None
    loss_cvar_95: float | None = None
    samples: list[float]


class SensitivityPoint(BaseModel):
    lambda_value: float
    expected_return: float
    risk: float


class ValidationSummary(BaseModel):
    distinct_models: bool
    asset_count_valid: bool
    stochastic_models_vary: bool
    feasibility_success_rate: float | None = None
    risk_target: float | None = None
    achieved_risk: float | None = None
    risk_target_gap: float | None = None
    warnings: list[str]


class TracePoint(BaseModel):
    iteration: float
    expected_return: float
    risk: float
    objective_value: float
    stage: str | None = None


class TransparencyPayload(BaseModel):
    agent_name: str
    headline: str
    mathematical_walkthrough: list[str]
    solver_trace: list[TracePoint]
    convergence_summary: str
    accuracy_statement: str
    solver_agent_notes: list[str]
    objective_surface: dict[str, Any]
    solver_manifold: dict[str, Any] | None = None
    candidate_history: list[dict[str, Any]] | None = None
    accepted_count: int | None = None
    rejected_count: int | None = None
    candidate_summary: str | None = None
    minimum_characteristics: dict[str, float] | None = None
    decision_variable_ranges: dict[str, Any] | None = None
    rejection_reasons: list[str] | None = None


class DecisionVariableRecord(BaseModel):
    name: str
    mu: float
    sigma: float
    liquidity: float
    cost: float
    mu_norm: float
    risk_norm: float
    liquidity_norm: float
    cost_norm: float
    decision_score: float
    selected_weight: float
    rank: int
    min_weight: float | None = None
    max_weight: float | None = None
    preferred_weight: float | None = None
    materiality: str | None = None
    reason: str | None = None


class DecisionVariablePayload(BaseModel):
    agent_name: str
    methodology: list[str]
    summary: str
    records: list[DecisionVariableRecord]
    decision_formula: str | None = None
    ranking_rule: str | None = None
    accepted_count: int | None = None
    rejected_count: int | None = None
    accepted_assets: list[str] | None = None
    rejected_assets: list[str] | None = None
    minimum_features: list[str] | None = None


class IntelligencePayload(BaseModel):
    decision_variables: dict[str, str | float]
    execution_log: list[dict[str, str]]
    execution_flow: list[str]
    model_explanations: dict[str, ModelExplanation]
    transparency: dict[str, TransparencyPayload]
    decision_variable_analysis: DecisionVariablePayload
    mathematical_model: dict[str, Any]


class DashboardResponse(BaseModel):
    assets: list[AssetSpec]
    covariance_matrix: list[list[float]]
    selected_model: str
    selected_output: ModelOutput
    model_outputs: dict[str, ModelOutput]
    comparison_table: list[ComparisonRow]
    efficient_frontier: list[FrontierPoint]
    monte_carlo: dict[str, MonteCarloSummary]
    sensitivity_analysis: list[SensitivityPoint]
    validation: ValidationSummary
    intelligence: IntelligencePayload


class CompareResponse(BaseModel):
    results: dict[str, ModelOutput]
    comparison_table: list[ComparisonRow]


class ValidationResponse(BaseModel):
    validation: ValidationSummary


class OptimizeResponse(BaseModel):
    selected_model: str
    selected_output: ModelOutput
    assets: list[AssetSpec]
