"""
Reusable trace-analysis helpers for solver transparency.

This module is intentionally backend-agnostic. It accepts solver histories,
candidate weights, and portfolio problem inputs, then returns JSON-ready
structures that can be consumed by a frontend transparency panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TraceAnalysisConfig:
    acceptance_quantile: float = 0.35
    rejection_margin: float = 0.08
    range_samples: int = 64


def _as_array(values: Any, *, ndim: int = 1) -> np.ndarray:
    if values is None:
        if ndim == 1:
            return np.asarray([], dtype=float)
        return np.asarray([[]], dtype=float)
    arr = np.asarray(values, dtype=float)
    if ndim == 1:
        return arr.reshape(-1).astype(float, copy=False)
    if arr.ndim == 1:
        arr = np.diag(arr)
    return arr.astype(float, copy=False)


def _as_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _safe_quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, q))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def build_decision_variable_ranges(
    assets: Sequence[Mapping[str, Any]],
    *,
    active_assets_target: int | None = None,
) -> dict[str, Any]:
    """
    Derive mathematically motivated variable ranges from the minimum asset data.

    Returns ranges for:
    - return score
    - risk score
    - liquidity score
    - cost score
    - suggested weight interval
    """

    if not assets:
        return {
            "asset_count": 0,
            "active_assets_target": active_assets_target,
            "ranges": {},
            "summary": "No assets were provided.",
        }

    mu = _as_array([asset.get("mu", 0.0) for asset in assets])
    sigma = _as_array([asset.get("sigma", 0.0) for asset in assets])
    liquidity = _as_array([asset.get("liquidity", 0.0) for asset in assets])
    cost = _as_array([asset.get("cost", 0.0) for asset in assets])
    names = [str(asset.get("name", f"Asset {i + 1}")) for i, asset in enumerate(assets)]

    mu_norm = _minmax(mu)
    sigma_norm = _minmax(sigma)
    liquidity_norm = _minmax(liquidity)
    cost_norm = _minmax(cost)

    target = active_assets_target or max(2, min(4, len(assets) // 3))
    target = max(1, min(int(target), len(assets)))
    base_weight = 1.0 / target
    lower = max(0.0, base_weight * 0.40)
    upper = min(1.0, base_weight * 1.85)

    records = []
    for idx, name in enumerate(names):
        decision_score = 0.50 * mu_norm[idx] + 0.25 * liquidity_norm[idx] + 0.20 * (1.0 - sigma_norm[idx]) - 0.15 * cost_norm[idx]
        records.append(
            {
                "name": name,
                "mu": float(mu[idx]),
                "sigma": float(sigma[idx]),
                "liquidity": float(liquidity[idx]),
                "cost": float(cost[idx]),
                "mu_norm": float(mu_norm[idx]),
                "risk_norm": float(sigma_norm[idx]),
                "liquidity_norm": float(liquidity_norm[idx]),
                "cost_norm": float(cost_norm[idx]),
                "decision_score": float(decision_score),
                "weight_lower_bound": float(lower),
                "weight_upper_bound": float(upper),
            }
        )

    records.sort(key=lambda row: row["decision_score"], reverse=True)
    return {
        "asset_count": len(assets),
        "active_assets_target": target,
        "ranges": {
            "mu": [float(np.min(mu)), float(np.max(mu))],
            "sigma": [float(np.min(sigma)), float(np.max(sigma))],
            "liquidity": [float(np.min(liquidity)), float(np.max(liquidity))],
            "cost": [float(np.min(cost)), float(np.max(cost))],
            "weight_interval": [float(lower), float(upper)],
        },
        "records": records,
        "summary": (
            f"Decision variables are derived from normalized mu, sigma, liquidity, and cost. "
            f"The solver should concentrate mass on roughly {target} assets, with individual weights typically in "
            f"[{lower:.3f}, {upper:.3f}] before projection."
        ),
    }


def summarize_rejection_reason(
    candidate_weights: Sequence[float],
    problem: Any,
    *,
    model_name: str | None = None,
    score: float | None = None,
    acceptance_threshold: float | None = None,
) -> str:
    """
    Produce a concise explanation for why a candidate would be rejected.
    """

    weights = _as_array(candidate_weights)
    if weights.size == 0:
        return "Rejected because the candidate portfolio is empty."

    active_target = int(getattr(problem, "active_assets_target", len(weights)) or len(weights))
    active_count = int(np.sum(weights > 1e-6))
    if not np.isclose(np.sum(weights), 1.0, atol=1e-5):
        return "Rejected because the weights do not sum to 1 after feasibility projection."
    if np.any(weights < -1e-8):
        return "Rejected because the candidate contains negative allocations."
    if active_count != active_target:
        return f"Rejected because it uses {active_count} active positions while the diversification target is {active_target}."

    if acceptance_threshold is not None and score is not None and score > acceptance_threshold:
        return f"Rejected because the objective value {score:.4f} is above the acceptance threshold {acceptance_threshold:.4f}."

    model = f"{model_name} " if model_name else ""
    return f"Rejected due to weaker {model}return-risk-liquidity trade-off relative to accepted candidates."


def build_candidate_analysis(
    candidates: Sequence[Mapping[str, Any]] | Sequence[Sequence[float]],
    problem: Any,
    *,
    model_name: str | None = None,
    config: TraceAnalysisConfig | None = None,
) -> dict[str, Any]:
    """
    Split candidates into accepted and rejected groups using objective values,
    feasibility, and diversification target checks.
    """

    config = config or TraceAnalysisConfig()
    weights_list: list[np.ndarray] = []
    candidate_records: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping):
            weights = _as_array(candidate.get("weights", candidate.get("allocation", [])))
            score = candidate.get("objective_value", candidate.get("score"))
            trace = dict(candidate)
        else:
            weights = _as_array(candidate)
            score = None
            trace = {"weights": _as_list(weights)}
        weights_list.append(weights)
        trace["weights"] = _as_list(weights)
        trace["active_count"] = int(np.sum(weights > 1e-6))
        trace["feasible"] = bool(np.isclose(np.sum(weights), 1.0, atol=1e-5) and np.all(weights >= -1e-8))
        trace["score"] = None if score is None else float(score)
        trace["model"] = str(model_name or trace.get("model", "candidate"))
        candidate_records.append(trace)

    if not candidate_records:
        return {
            "accepted_candidates": [],
            "rejected_candidates": [],
            "rejection_reasons": [],
            "summary": "No candidates were provided.",
        }

    scores = np.asarray([float(item["score"]) if item["score"] is not None else np.inf for item in candidate_records], dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    acceptance_cutoff = _safe_quantile(finite_scores, config.acceptance_quantile, default=np.inf)
    active_target = int(getattr(problem, "active_assets_target", len(candidate_records[0]["weights"])) or len(candidate_records[0]["weights"]))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: list[dict[str, str]] = []

    for candidate in candidate_records:
        weights = _as_array(candidate["weights"])
        score = candidate["score"]
        feasible = bool(candidate["feasible"])
        active_count = int(candidate["active_count"])

        if not feasible:
            reason = "Rejected because it violates simplex feasibility."
            rejected.append(candidate)
            rejection_reasons.append({"candidate": candidate.get("model", "candidate"), "reason": reason})
            continue

        if active_count != active_target:
            reason = (
                f"Rejected because it uses {active_count} active positions while the target is {active_target}."
            )
            candidate["rejection_reason"] = reason
            rejected.append(candidate)
            rejection_reasons.append({"candidate": candidate.get("model", "candidate"), "reason": reason})
            continue

        if score is not None and score > acceptance_cutoff + config.rejection_margin:
            reason = (
                f"Rejected because score {score:.4f} exceeds the acceptance cutoff {acceptance_cutoff:.4f} "
                f"by more than the margin {config.rejection_margin:.4f}."
            )
            candidate["rejection_reason"] = reason
            rejected.append(candidate)
            rejection_reasons.append({"candidate": candidate.get("model", "candidate"), "reason": reason})
            continue

        candidate["accepted_reason"] = "Feasible, diversified, and competitive on objective value."
        accepted.append(candidate)

    if not accepted and candidate_records:
        best_idx = int(np.nanargmin(scores)) if np.any(np.isfinite(scores)) else 0
        candidate = candidate_records[best_idx]
        candidate["accepted_reason"] = "Promoted as the best available candidate for transparency display."
        accepted.append(candidate)
        rejected = [item for idx, item in enumerate(candidate_records) if idx != best_idx]
        rejection_reasons = [
            {"candidate": item.get("model", "candidate"), "reason": item.get("rejection_reason", "Rejected by ranking logic.")}
            for item in rejected
        ]

    return {
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "rejection_reasons": rejection_reasons,
        "acceptance_cutoff": float(acceptance_cutoff) if np.isfinite(acceptance_cutoff) else None,
        "summary": (
            f"Accepted {len(accepted)} of {len(candidate_records)} candidates using feasibility, diversification, "
            f"and objective ranking."
        ),
    }


def build_trace_transparency(
    solver_history: Sequence[Mapping[str, Any]] | Sequence[Any],
    problem: Any,
    *,
    model_name: str,
    selected_weights: Sequence[float],
    selected_score: float | None = None,
    candidates: Sequence[Mapping[str, Any]] | Sequence[Sequence[float]] | None = None,
    config: TraceAnalysisConfig | None = None,
) -> dict[str, Any]:
    """
    Build a single JSON-ready transparency payload for a solver run.
    """

    config = config or TraceAnalysisConfig()
    weights = _as_array(selected_weights)
    mu = _as_array(getattr(problem, "mu", None))
    cov = _as_array(getattr(problem, "cov", None), ndim=2)
    liquidity = _as_array(getattr(problem, "liquidity", None))
    cost = _as_array(getattr(problem, "cost", None))
    active_target = int(getattr(problem, "active_assets_target", len(weights)) or len(weights))

    ranges = build_decision_variable_ranges(
        [
            {
                "name": name,
                "mu": float(mu[idx]) if idx < mu.size else 0.0,
                "sigma": float(np.sqrt(max(cov[idx, idx], 0.0))) if cov.size else 0.0,
                "liquidity": float(liquidity[idx]) if idx < liquidity.size else 0.0,
                "cost": float(cost[idx]) if idx < cost.size else 0.0,
            }
            for idx, name in enumerate(getattr(problem, "asset_names", [f"Asset {i + 1}" for i in range(len(weights))]))
        ],
        active_assets_target=active_target,
    )

    history = [
        {
            "iteration": float(item.get("iteration", idx)) if isinstance(item, Mapping) else float(idx),
            "expected_return": float(item.get("expected_return", 0.0)) if isinstance(item, Mapping) else 0.0,
            "risk": float(item.get("risk", 0.0)) if isinstance(item, Mapping) else 0.0,
            "objective_value": float(item.get("objective_value", 0.0)) if isinstance(item, Mapping) else 0.0,
            "stage": str(item.get("stage")) if isinstance(item, Mapping) and item.get("stage") is not None else None,
        }
        for idx, item in enumerate(solver_history)
    ]

    candidate_payload = build_candidate_analysis(
        candidates or [],
        problem,
        model_name=model_name,
        config=config,
    ) if candidates is not None else {
        "accepted_candidates": [],
        "rejected_candidates": [],
        "rejection_reasons": [],
        "summary": "No candidate set was provided.",
    }

    accepted_count = len(candidate_payload["accepted_candidates"])
    rejected_count = len(candidate_payload["rejected_candidates"])
    rejection_reasons = [
        str(item.get("reason", "")) for item in candidate_payload.get("rejection_reasons", [])
    ]

    decision_variable_records = ranges["records"]
    feasible_ranges = {
        "weight_interval": ranges["ranges"]["weight_interval"],
        "top_asset_rank": [record["name"] for record in decision_variable_records[:active_target]],
        "expected_return_range": [
            float(np.min(mu)) if mu.size else 0.0,
            float(np.max(mu)) if mu.size else 0.0,
        ],
        "risk_range": [
            float(np.min(np.sqrt(np.maximum(np.diag(cov), 0.0)))) if cov.size else 0.0,
            float(np.max(np.sqrt(np.maximum(np.diag(cov), 0.0)))) if cov.size else 0.0,
        ],
        "liquidity_range": [
            float(np.min(liquidity)) if liquidity.size else 0.0,
            float(np.max(liquidity)) if liquidity.size else 0.0,
        ],
        "cost_range": [
            float(np.min(cost)) if cost.size else 0.0,
            float(np.max(cost)) if cost.size else 0.0,
        ],
    }

    summary = (
        f"{model_name} solved a {len(weights)}-variable portfolio with target {active_target} active assets. "
        f"{accepted_count} candidates were accepted and {rejected_count} were rejected."
    )
    if selected_score is not None:
        summary += f" Selected score: {float(selected_score):.4f}."

    return {
        "model_name": str(model_name).upper(),
        "summary": summary,
        "decision_variable_ranges": ranges,
        "feasible_ranges": feasible_ranges,
        "accepted_candidates": candidate_payload["accepted_candidates"],
        "rejected_candidates": candidate_payload["rejected_candidates"],
        "rejection_reasons": rejection_reasons,
        "solver_history": history,
        "selected_weights": _as_list(weights),
        "selected_score": None if selected_score is None else float(selected_score),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "minimum_characteristics": {
            "mu_min": float(np.min(mu)) if mu.size else 0.0,
            "sigma_min": float(np.min(np.sqrt(np.maximum(np.diag(cov), 0.0)))) if cov.size else 0.0,
            "liquidity_min": float(np.min(liquidity)) if liquidity.size else 0.0,
            "cost_min": float(np.min(cost)) if cost.size else 0.0,
        },
        "candidate_summary": candidate_payload.get("summary", ""),
    }


__all__ = [
    "TraceAnalysisConfig",
    "build_candidate_analysis",
    "build_decision_variable_ranges",
    "build_trace_transparency",
    "summarize_rejection_reason",
]

