"""
Smoke test for the multi-model portfolio optimization platform.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.api import app
from optimizer import PortfolioOptimizer, PortfolioRequest


def _assert_valid_output(payload: dict, expected_assets: int) -> None:
    weights = np.asarray(payload["weights"], dtype=float)
    assert len(weights) == expected_assets
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-8)
    assert payload["valid"]


def main() -> None:
    request = PortfolioRequest(
        asset_count=12,
        active_assets_target=4,
        budget=5.0,
        lambda_value=1.0,
        liquidity_weight=0.35,
        cost_sensitivity=0.15,
        method="AUTO",
        seed=7,
        monte_carlo_paths=300,
        dataset="balanced",
    )
    optimizer = PortfolioOptimizer(request)
    optimize_result = optimizer.optimize()
    dashboard = optimizer.dashboard_bundle()
    validation = optimizer.validation(dashboard["model_outputs"])

    assert optimize_result["selected_model"] in {"QP", "PSO", "GA", "HYBRID"}
    assert len(optimize_result["assets"]) == 12
    _assert_valid_output(optimize_result["selected_output"], 12)
    assert set(dashboard["model_outputs"].keys()) == {"QP", "PSO", "GA", "HYBRID"}
    for output in dashboard["model_outputs"].values():
        _assert_valid_output(output, 12)
    assert dashboard["validation"]["asset_count_valid"]
    assert len(dashboard["comparison_table"]) == 4
    assert len(dashboard["efficient_frontier"]) == 8
    assert len(dashboard["sensitivity_analysis"]) == 5
    assert set(dashboard["monte_carlo"].keys()) == {"QP", "PSO", "GA", "HYBRID"}
    assert dashboard["intelligence"]["execution_flow"]
    assert dashboard["intelligence"]["model_explanations"]["QP"]["asset_justifications"]
    assert dashboard["intelligence"]["transparency"]["QP"]["solver_trace"]
    assert validation["validation"]["distinct_models"]

    client = TestClient(app)
    payload = {
        "asset_count": 12,
        "active_assets_target": 4,
        "budget": 5.0,
        "lambda_value": 1.0,
        "liquidity_weight": 0.35,
        "cost_sensitivity": 0.15,
        "method": "AUTO",
        "seed": 7,
        "monte_carlo_paths": 300,
        "dataset": "balanced",
    }
    optimize_response = client.post("/optimize", json=payload)
    compare_response = client.post("/compare", json=payload)
    validation_response = client.post("/validation", json=payload)
    dashboard_response = client.post("/dashboard", json=payload)
    live_start_response = client.post("/dashboard/live/start", json=payload)
    demo_response = client.post("/smart-demo")

    assert optimize_response.status_code == 200
    assert compare_response.status_code == 200
    assert validation_response.status_code == 200
    assert dashboard_response.status_code == 200
    assert live_start_response.status_code == 200
    assert demo_response.status_code == 200

    live_job_id = live_start_response.json()["job_id"]
    live_status = None
    for _ in range(120):
        live_status_response = client.get(f"/dashboard/live/{live_job_id}")
        assert live_status_response.status_code == 200
        live_status = live_status_response.json()
        if live_status["status"] == "succeeded":
            break
        time.sleep(0.25)
    assert live_status is not None
    assert live_status["status"] == "succeeded"
    assert live_status["result"]["selected_output"]["valid"]
    assert live_status["result"]["live_console"]["current_trace"]
    artifacts = live_status["metadata"]["artifacts"]
    assert Path(artifacts["snapshot_file"]).exists()
    assert Path(artifacts["history_file"]).exists()
    assert Path(artifacts["latest_payload_file"]).exists()
    assert Path(artifacts["checkpoints_dir"]).exists()

    compare_json = compare_response.json()
    assert set(compare_json["results"].keys()) == {"QP", "PSO", "GA", "HYBRID"}
    assert compare_json["comparison_table"]
    validation_json = validation_response.json()
    assert "distinct_models" in validation_json["validation"]
    dashboard_json = dashboard_response.json()
    assert dashboard_json["assets"]
    assert dashboard_json["selected_output"]["explanation"]["why_this_allocation"]
    assert dashboard_json["selected_output"]["active_assets_target"] == 4

    qp_w = np.asarray(compare_json["results"]["QP"]["weights"], dtype=float)
    pso_w = np.asarray(compare_json["results"]["PSO"]["weights"], dtype=float)
    ga_w = np.asarray(compare_json["results"]["GA"]["weights"], dtype=float)
    assert np.linalg.norm(qp_w - pso_w, ord=1) > 0.03
    assert np.linalg.norm(qp_w - ga_w, ord=1) > 0.03

    print("[OK] Multi-model platform smoke test passed.")


if __name__ == "__main__":
    main()
