"""
Live optimization event utilities.

This module is intentionally backend-agnostic. It turns solver progress,
candidate states, and 3D plot data into JSON-ready dictionaries that can be
emitted while the optimizer is running and consumed by the frontend.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "stage_started",
    "stage_completed",
    "trace_point",
    "candidate_snapshot",
    "build_3d_trace_payload",
    "build_objective_surface_payload",
    "build_progress_snapshot",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _vector(values: Sequence[float] | np.ndarray | None) -> list[float]:
    if values is None:
        return []
    return [float(item) for item in np.asarray(values, dtype=float).reshape(-1)]


def _matrix(values: Sequence[Sequence[float]] | np.ndarray | None) -> list[list[float]]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = np.diag(arr)
    return [[float(cell) for cell in row] for row in arr.tolist()]


def _base_event(
    event_type: str,
    *,
    stage: str,
    model: str,
    iteration: int | float | None = None,
    status: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_type": event_type,
        "stage": str(stage),
        "model": str(model).upper(),
        "timestamp": _utc_now(),
    }
    if iteration is not None:
        payload["iteration"] = float(iteration)
    if status is not None:
        payload["status"] = str(status)
    if details:
        payload["details"] = _jsonable(details)
    return payload


def stage_started(
    *,
    stage: str,
    model: str,
    iteration: int | float | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_event("stage_started", stage=stage, model=model, iteration=iteration, status="started", details=details)


def stage_completed(
    *,
    stage: str,
    model: str,
    iteration: int | float | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _base_event("stage_completed", stage=stage, model=model, iteration=iteration, status="completed", details=details)


def trace_point(
    *,
    stage: str,
    model: str,
    iteration: int | float,
    expected_return: float,
    risk: float,
    objective_value: float,
    accepted: bool | None = None,
    active_count: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _base_event("trace_point", stage=stage, model=model, iteration=iteration, details=details)
    payload.update(
        {
            "expected_return": float(expected_return),
            "risk": float(risk),
            "objective_value": float(objective_value),
        }
    )
    if accepted is not None:
        payload["accepted"] = bool(accepted)
    if active_count is not None:
        payload["active_count"] = int(active_count)
    return payload


def candidate_snapshot(
    *,
    stage: str,
    model: str,
    weights: Sequence[float] | np.ndarray,
    objective_value: float,
    expected_return: float,
    risk: float,
    accepted: bool,
    reason: str | None = None,
    rank: int | None = None,
    decision_variables: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _base_event("candidate_snapshot", stage=stage, model=model, details=details)
    payload.update(
        {
            "weights": _vector(weights),
            "objective_value": float(objective_value),
            "expected_return": float(expected_return),
            "risk": float(risk),
            "accepted": bool(accepted),
        }
    )
    if reason is not None:
        payload["reason"] = str(reason)
    if rank is not None:
        payload["rank"] = int(rank)
    if decision_variables is not None:
        payload["decision_variables"] = _jsonable(decision_variables)
    return payload


def build_3d_trace_payload(
    *,
    model: str,
    trace: Sequence[Mapping[str, Any]],
    color_by: str = "expected_return",
    title: str | None = None,
) -> dict[str, Any]:
    points = []
    for idx, item in enumerate(trace):
        points.append(
            {
                "iteration": float(item.get("iteration", idx)),
                "risk": float(item.get("risk", 0.0)),
                "objective_value": float(item.get("objective_value", 0.0)),
                "expected_return": float(item.get("expected_return", 0.0)),
                "stage": str(item.get("stage", model)),
            }
        )
    return {
        "plot_type": "scatter3d",
        "model": str(model).upper(),
        "title": title or f"{str(model).upper()} Solver Trace",
        "color_by": str(color_by),
        "points": points,
    }


def build_objective_surface_payload(
    *,
    model: str,
    asset_x: str,
    asset_y: str,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    z: Sequence[Sequence[float]] | np.ndarray,
    selected_point: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    payload = {
        "plot_type": "surface",
        "model": str(model).upper(),
        "title": title or f"{str(model).upper()} Objective Surface",
        "asset_x": str(asset_x),
        "asset_y": str(asset_y),
        "x": _vector(x),
        "y": _vector(y),
        "z": _matrix(z),
    }
    if selected_point is not None:
        payload["selected_point"] = _jsonable(selected_point)
    return payload


def build_progress_snapshot(
    *,
    model: str,
    stage: str,
    iteration: int | float | None = None,
    progress: float | None = None,
    message: str | None = None,
    trace: Sequence[Mapping[str, Any]] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    surface: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _base_event("progress_snapshot", stage=stage, model=model, iteration=iteration, details=details)
    if progress is not None:
        payload["progress"] = float(progress)
    if message is not None:
        payload["message"] = str(message)
    if trace is not None:
        payload["trace"] = [_jsonable(item) for item in trace]
    if candidates is not None:
        payload["candidates"] = [_jsonable(item) for item in candidates]
    if surface is not None:
        payload["surface"] = _jsonable(surface)
    return payload

