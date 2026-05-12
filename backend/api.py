"""
FastAPI backend for the multi-model portfolio optimization platform.
"""

from __future__ import annotations

import json
from functools import lru_cache
from threading import Thread

from fastapi import FastAPI, HTTPException

from api.schemas import CompareResponse, DashboardResponse, OptimizeRequest, OptimizeResponse, ValidationResponse
from backend.live_jobs import JOB_MANAGER, JobNotFoundError
from optimizer import PortfolioOptimizer, PortfolioRequest


app = FastAPI(
    title="BrixSquare AI Investment Optimizer API",
    version="4.0.0",
    description="Multi-model portfolio optimization API with explainability and interactive intelligence outputs.",
)


def _to_request(payload: OptimizeRequest) -> PortfolioRequest:
    return PortfolioRequest(
        asset_count=payload.asset_count,
        active_assets_target=payload.active_assets_target,
        budget=payload.budget,
        max_allocation=payload.max_allocation,
        lambda_value=payload.lambda_value,
        liquidity_weight=payload.liquidity_weight,
        cost_sensitivity=payload.cost_sensitivity,
        tail_risk_weight=payload.tail_risk_weight,
        diversification_weight=payload.diversification_weight,
        concentration_penalty=payload.concentration_penalty,
        regime_aversion=payload.regime_aversion,
        tracking_penalty=payload.tracking_penalty,
        covariance_shrinkage=payload.covariance_shrinkage,
        cvar_alpha=payload.cvar_alpha,
        method=payload.method,
        seed=payload.seed,
        monte_carlo_paths=payload.monte_carlo_paths,
        assets=[asset.model_dump() for asset in payload.assets] if payload.assets else None,
        dataset=payload.dataset,
        exact_cvar_seed=payload.exact_cvar_seed,
        exact_discrete_selection=payload.exact_discrete_selection,
        planning_horizon=payload.planning_horizon,
        backtest_lookback=payload.backtest_lookback,
        historical_returns=payload.historical_returns,
        macro_signal=payload.macro_signal,
        return_engine=payload.return_engine,
        return_prior_strength=payload.return_prior_strength,
        bl_tau=payload.bl_tau,
        bl_view_confidence=payload.bl_view_confidence,
    )


def _run_live_dashboard_job(job_id: str, payload_dict: dict) -> None:
    try:
        payload = OptimizeRequest.model_validate(payload_dict)
        optimizer = PortfolioOptimizer(_to_request(payload))
        JOB_MANAGER.mark_running(job_id, stage="initializing", message="Preparing live optimization workflow")

        def _progress(stage: str, progress: float, bundle: dict, message: str) -> None:
            JOB_MANAGER.mark_progress(
                job_id,
                stage=stage,
                progress=progress,
                payload=bundle,
                message=message,
            )

        result = optimizer.dashboard_bundle_live(progress_callback=_progress)
        JOB_MANAGER.mark_complete(job_id, result=result, payload=result.get("live_console") or {"selected_model": result.get("selected_model")}, message="Live optimization completed")
    except Exception as exc:
        JOB_MANAGER.mark_failed(job_id, error=exc, message="Live optimization failed")


@lru_cache(maxsize=64)
def _cached_dashboard_bundle(payload_json: str) -> dict:
    payload = OptimizeRequest.model_validate(json.loads(payload_json))
    optimizer = PortfolioOptimizer(_to_request(payload))
    return optimizer.dashboard_bundle()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "product": "BrixSquare AI Investment Optimizer", "mode": "multi-model"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> dict:
    try:
        return PortfolioOptimizer(_to_request(payload)).optimize()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc


@app.post("/compare", response_model=CompareResponse)
def compare(payload: OptimizeRequest) -> dict:
    try:
        optimizer = PortfolioOptimizer(_to_request(payload))
        results = optimizer.compare_methods()
        return {"results": results, "comparison_table": optimizer.dashboard_bundle()["comparison_table"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc


@app.post("/validation", response_model=ValidationResponse)
def validation(payload: OptimizeRequest) -> dict:
    try:
        optimizer = PortfolioOptimizer(_to_request(payload))
        return optimizer.validation()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}") from exc


@app.post("/dashboard", response_model=DashboardResponse)
def dashboard(payload: OptimizeRequest) -> dict:
    try:
        payload_json = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
        return _cached_dashboard_bundle(payload_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dashboard bundle failed: {exc}") from exc


@app.post("/backtest")
def backtest(payload: OptimizeRequest) -> dict:
    try:
        return PortfolioOptimizer(_to_request(payload)).backtest()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc


@app.post("/multi-period")
def multi_period(payload: OptimizeRequest) -> dict:
    try:
        return PortfolioOptimizer(_to_request(payload)).multi_period_optimize()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multi-period optimization failed: {exc}") from exc


@app.post("/dashboard/live/start")
def dashboard_live_start(payload: OptimizeRequest) -> dict:
    snapshot = JOB_MANAGER.create_job(
        kind="dashboard_live",
        input_payload=payload.model_dump(mode="json"),
        metadata={"method": payload.method, "dataset": payload.dataset, "asset_count": payload.asset_count},
    )
    thread = Thread(target=_run_live_dashboard_job, args=(snapshot["job_id"], payload.model_dump(mode="json")), daemon=True)
    thread.start()
    return snapshot


@app.get("/dashboard/live/{job_id}")
def dashboard_live_status(job_id: str) -> dict:
    try:
        return JOB_MANAGER.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Live job not found: {job_id}") from exc


@app.post("/smart-demo", response_model=OptimizeResponse)
def smart_demo() -> dict:
    request = PortfolioRequest(
        asset_count=12,
        active_assets_target=4,
        budget=5.0,
        max_allocation=0.35,
        lambda_value=1.0,
        liquidity_weight=0.35,
        cost_sensitivity=0.15,
        tail_risk_weight=0.45,
        diversification_weight=0.30,
        concentration_penalty=0.18,
        regime_aversion=0.28,
        tracking_penalty=0.12,
        covariance_shrinkage=0.15,
        cvar_alpha=0.95,
        method="AUTO",
        seed=42,
        monte_carlo_paths=500,
        dataset="balanced",
        exact_cvar_seed=True,
        exact_discrete_selection=True,
        return_engine="BLACK_LITTERMAN",
    )
    return PortfolioOptimizer(request).optimize()
