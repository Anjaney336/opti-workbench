from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error, request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.dashboard.live_components import (
    LIVE_CONSOLE_CSS,
    build_live_accept_reject_cards,
    build_live_progress_banner,
    build_live_stage_table,
    build_live_trace_3d_figure,
    build_polling_status_cards,
    build_stage_timeline_figure,
)
from frontend.dashboard.ui_components import (
    DASHBOARD_CSS,
    build_asset_dataframe,
    build_asset_simulation_dataframe,
    build_asset_simulation_figure,
    build_asset_value_estimate_dataframe,
    build_benchmark_attainment_figure,
    build_capital_deployment_figure,
    build_combined_comparison_figure,
    build_comparison_dataframe,
    build_decision_space_3d_figure,
    build_efficient_frontier_figure,
    build_pipeline_flow_figure,
    build_model_allocation_figure,
    build_objective_breakdown_figure,
    build_objective_surface_3d_figure,
    build_portfolio_valuation_dataframe,
    build_regime_profile_figure,
    build_regime_scatter_3d_figure,
    build_risk_heatmap,
    build_scenario_valuation_figure,
    build_solver_trace_3d_figure,
    build_solver_manifold_figure,
    build_weight_score_progress_figure,
    stylize_comparison_dataframe,
)
from frontend.dashboard.visibility_components import (
    VISIBILITY_CSS,
    build_candidate_range_dataframe,
    build_candidate_range_figure,
    build_rejection_summary,
    build_solver_accept_reject_cards,
    build_solver_step_html,
    build_solver_visibility_banner,
    stylize_candidate_range_dataframe,
)
from optimizer.asset_engine import generate_asset_universe


DEFAULT_API_URL = "http://127.0.0.1:8000"
DATASET_OPTIONS = ["balanced", "high_growth", "stable", "risky", "low_liquidity", "commercial_heavy"]
METHOD_OPTIONS = ["AUTO", "QP", "PSO", "GA", "HYBRID"]
RETURN_ENGINE_OPTIONS = ["BLACK_LITTERMAN", "BAYESIAN", "RAW"]
REPO_ROOT = Path(__file__).resolve().parents[2]

WORKFLOW_SOURCE_INDEX = {
    "initializing": [
        {"title": "Problem Construction", "path": "optimizer/engine.py", "anchor": "def _build_problem(", "note": "This assembles tensors, constraints, and the optimization problem."},
        {"title": "Return Estimation", "path": "optimizer/engine.py", "anchor": "def _estimate_expected_returns(", "note": "This updates expected returns before any solver starts."},
    ],
    "QP": [
        {"title": "QP Seed Solver", "path": "optimizer/solvers.py", "anchor": "def solve_qp(", "note": "Convex seed stage with feasibility and optional exact CVaR seed."},
        {"title": "Objective Terms", "path": "optimizer/solvers.py", "anchor": "def objective_components(", "note": "All models are scored with this common objective decomposition."},
    ],
    "PSO": [
        {"title": "PSO Search", "path": "optimizer/solvers.py", "anchor": "def solve_pso(", "note": "Swarm search refines the seeded portfolio over the common objective."},
        {"title": "Objective Terms", "path": "optimizer/solvers.py", "anchor": "def objective_components(", "note": "Current PSO candidates are scored here."},
    ],
    "GA": [
        {"title": "GA Refinement", "path": "optimizer/solvers.py", "anchor": "def solve_ga(", "note": "Genetic refinement warm-started from earlier stages."},
        {"title": "Objective Terms", "path": "optimizer/solvers.py", "anchor": "def objective_components(", "note": "GA populations are evaluated with the same breakdown."},
    ],
    "HYBRID": [
        {"title": "Hybrid Selection", "path": "optimizer/solvers.py", "anchor": "def solve_hybrid(", "note": "This blends and selects across QP, PSO, and GA outputs."},
        {"title": "Dashboard Packaging", "path": "optimizer/engine.py", "anchor": "def _dashboard_bundle_from_raw(", "note": "This packages traces, comparisons, validation, and workflow intelligence."},
    ],
    "packaging": [
        {"title": "Dashboard Packaging", "path": "optimizer/engine.py", "anchor": "def _dashboard_bundle_from_raw(", "note": "This converts raw solver outputs into frontend-ready payloads."},
        {"title": "Mathematical Payload", "path": "optimizer/engine.py", "anchor": "def _mathematical_model_payload(", "note": "This creates the math, workflow, and regime payload shown in the UI."},
    ],
}


def call_api(path: str, payload: dict | None, api_url: str, *, method: str = "POST") -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    http_request = request.Request(url=f"{api_url}{path}", data=body, headers=headers, method=method)
    with request.urlopen(http_request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def start_live_dashboard(api_url: str, payload: dict) -> dict:
    return call_api("/dashboard/live/start", payload, api_url, method="POST")


def fetch_live_dashboard(api_url: str, job_id: str) -> dict:
    return call_api(f"/dashboard/live/{job_id}", None, api_url, method="GET")


def ensure_asset_table(asset_count: int, dataset: str, seed: int) -> None:
    signature = f"{asset_count}:{dataset}:{seed}"
    if st.session_state.get("_asset_signature") != signature:
        universe = generate_asset_universe(n_assets=asset_count, scenario=dataset, seed=seed)
        st.session_state["_asset_signature"] = signature
        st.session_state["asset_rows"] = [
            {
                "name": asset.name,
                "mu": round(asset.mu, 4),
                "sigma": round(asset.sigma, 4),
                "liquidity": round(asset.liquidity, 4),
                "cost": round(asset.cost, 4),
                "sector": asset.sector,
            }
            for asset in universe.assets
        ]


def build_payload(
    asset_count: int,
    active_assets_target: int,
    budget: float,
    max_allocation: float,
    lambda_value: float,
    liquidity_weight: float,
    cost_sensitivity: float,
    tail_risk_weight: float,
    diversification_weight: float,
    concentration_penalty: float,
    regime_aversion: float,
    tracking_penalty: float,
    covariance_shrinkage: float,
    cvar_alpha: float,
    method: str,
    dataset: str,
    seed: int,
    assets: list[dict],
    return_engine: str,
    return_prior_strength: float,
    bl_tau: float,
    bl_view_confidence: float,
    exact_cvar_seed: bool,
    exact_discrete_selection: bool,
) -> dict:
    return {
        "asset_count": asset_count,
        "active_assets_target": active_assets_target,
        "budget": budget,
        "max_allocation": max_allocation,
        "lambda_value": lambda_value,
        "liquidity_weight": liquidity_weight,
        "cost_sensitivity": cost_sensitivity,
        "tail_risk_weight": tail_risk_weight,
        "diversification_weight": diversification_weight,
        "concentration_penalty": concentration_penalty,
        "regime_aversion": regime_aversion,
        "tracking_penalty": tracking_penalty,
        "covariance_shrinkage": covariance_shrinkage,
        "cvar_alpha": cvar_alpha,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "monte_carlo_paths": 300,
        "assets": assets,
        "return_engine": return_engine,
        "return_prior_strength": return_prior_strength,
        "bl_tau": bl_tau,
        "bl_view_confidence": bl_view_confidence,
        "exact_cvar_seed": exact_cvar_seed,
        "exact_discrete_selection": exact_discrete_selection,
    }


def metric_card(label: str, value: str, help_text: str, css_class: str) -> None:
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-bottom:0.65rem;">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
            <div class="dashboard-note">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fallback_math_payload(dashboard: dict) -> dict:
    selected_output = dashboard.get("selected_output", {})
    selected_model = dashboard.get("selected_model", "QP")
    regime_metrics = selected_output.get("regime_metrics", {})
    return {
        "multi_objective_formulation": "max ( R(x), -sigma(x), -CVaR_alpha(x), L(x) )",
        "normalization_equation": "f_tilde_i(x) = ( f_i(x) - min f_i ) / ( max f_i - min f_i )",
        "scalarization_equation": "J_scalar(x) = sum_i w_i f_tilde_i(x)",
        "objective_equation": "J(x) = x^T Sigma x - lambda mu^T x + gamma C(x) - delta L(x)",
        "hybrid_equation": "J_H(x) = weighted blend of QP, PSO, and GA stage objectives",
        "covariance_equation": "Sigma_hat = Sigma  (legacy payload fallback)",
        "cvar_equation": "CVaR_alpha(x) estimated from Monte Carlo downside tails",
        "regime_equation": "Legacy run did not persist explicit regime equations.",
        "scalarization_note": "Scalarization converts the multi-objective portfolio problem into a single-objective optimization problem.",
        "discrete_real_estate_model": "x_i in {0,1}  (true indivisible real-estate formulation)",
        "relaxed_real_estate_model": "x_i in [0,1]  (continuous relaxation used for tractable optimization)",
        "constraints": [
            "sum_i x_i = 1",
            "x_i >= 0",
            "legacy run may not include full active-asset metadata",
        ],
        "theoretical_properties": [],
        "coherent_risk_theory": {},
        "decision_variables": [
            {"symbol": "x_i", "meaning": "portfolio weight of asset i"},
            {"symbol": "Sigma", "meaning": "covariance matrix"},
            {"symbol": "mu", "meaning": "expected-return vector"},
        ],
        "parameter_table": [],
        "objective_terms_by_model": {},
        "selected_objective_terms": selected_output.get("objective_breakdown", {}),
        "regime_table": [],
        "regime_3d_points": (
            [
                {
                    "model": selected_model,
                    "bear_return": 0.0,
                    "neutral_return": float(selected_output.get("expected_return", 0.0)),
                    "bull_return": 0.0,
                    "loss_cvar_95": float(regime_metrics.get("loss_cvar_95", 0.0)),
                    "risk": float(selected_output.get("risk", 0.0)),
                }
            ]
            if selected_output
            else []
        ),
        "hybrid_stage_weights": regime_metrics.get("hybrid_stage_weights", {}),
        "pipeline_graph": {"nodes": [], "links": []},
        "benchmark_table": [],
        "key_concepts_and_benchmarks": [],
        "workflow_steps": ["Legacy snapshot loaded without the new mathematical payload; fallback view is being used."],
        "selected_solver_trace_length": 0,
    }


def _resolve_math_payload(dashboard: dict) -> dict:
    intelligence = dashboard.get("intelligence", {})
    return intelligence.get("mathematical_model") or _fallback_math_payload(dashboard)


def _resolve_objective_breakdown(selected_output: dict, math_payload: dict | None = None) -> dict:
    direct = selected_output.get("objective_breakdown") or {}
    if direct:
        return direct
    if math_payload:
        return math_payload.get("selected_objective_terms") or {}
    return {}


def _resolve_regime_metrics(selected_output: dict, math_payload: dict | None = None) -> dict:
    direct = selected_output.get("regime_metrics") or {}
    if direct and direct.get("regime_names"):
        return direct
    if math_payload and math_payload.get("regime_table"):
        regime_table = math_payload.get("regime_table") or []
        return {
            "regime_names": [str(row.get("Regime", "")) for row in regime_table],
            "regime_probabilities": [float(row.get("Probability", 0.0)) for row in regime_table],
            "regime_expected_returns": [float(row.get("Selected Return", 0.0)) for row in regime_table],
            "regime_risks": [float(row.get("Selected Risk", 0.0)) for row in regime_table],
            "loss_var_95": 0.0,
            "loss_cvar_95": 0.0,
            "tracking_error": 0.0,
            "regime_dispersion": 0.0,
            "regime_stress": 0.0,
            "hybrid_stage_weights": math_payload.get("hybrid_stage_weights", {}),
        }
    return {
        "regime_names": [],
        "regime_probabilities": [],
        "regime_expected_returns": [],
        "regime_risks": [],
        "loss_var_95": 0.0,
        "loss_cvar_95": 0.0,
        "tracking_error": 0.0,
        "regime_dispersion": 0.0,
        "regime_stress": 0.0,
        "hybrid_stage_weights": {},
    }


def build_allocation_pie_figure(asset_labels: list[str], weights: list[float], title: str) -> go.Figure:
    positive_pairs = [(label, float(weight)) for label, weight in zip(asset_labels, weights) if float(weight) > 1e-6]
    if not positive_pairs:
        positive_pairs = [("No allocation", 1.0)]
    labels = [item[0] for item in positive_pairs]
    values = [item[1] for item in positive_pairs]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.4f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return figure


def control_label(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="dashboard-subtitle" style="margin-top:0.8rem;">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="dashboard-note" style="margin-top:0.15rem;margin-bottom:0.35rem;">{subtitle}</div>', unsafe_allow_html=True)


def _run_payload(snapshot: dict | None, dashboard: dict | None) -> dict:
    if snapshot and snapshot.get("input_payload"):
        return dict(snapshot["input_payload"])
    if dashboard and dashboard.get("selected_output"):
        return {"budget": 5.0, "dataset": "balanced"}
    return {}


def _budget_from_payload(payload: dict) -> float:
    try:
        return float(payload.get("budget", 5.0))
    except Exception:
        return 5.0


def _valuation_memo_payload(dashboard: dict, run_payload: dict) -> tuple[dict, pd.DataFrame]:
    selected_output = dashboard["selected_output"]
    assets = dashboard["assets"]
    budget = _budget_from_payload(run_payload)
    regime_metrics = _resolve_regime_metrics(selected_output)
    math_payload = _resolve_math_payload(dashboard)
    posterior_mu = ((math_payload.get("return_engine") or {}).get("posterior_mu")) or None
    valuation_frame = build_portfolio_valuation_dataframe(
        assets,
        selected_output["weights"],
        capital_budget=budget,
        expected_returns=posterior_mu,
    )
    expected_profit = budget * float(selected_output["expected_return"])
    downside_reserve = budget * float(regime_metrics.get("loss_cvar_95", 0.0))
    regime_names = list(regime_metrics.get("regime_names", []))
    regime_returns = list(regime_metrics.get("regime_expected_returns", []))
    bear_value = budget
    if regime_names and regime_returns:
        try:
            bear_index = regime_names.index("Bear")
            bear_value = budget * (1.0 + float(regime_returns[bear_index]))
        except Exception:
            bear_value = budget * (1.0 + float(regime_returns[0]))
    return {
        "capital_budget": budget,
        "expected_profit": expected_profit,
        "downside_reserve": downside_reserve,
        "bear_case_value": bear_value,
        "average_liquidity": float(selected_output.get("average_liquidity", 0.0)),
        "objective_value": float(selected_output.get("objective_value", 0.0)),
    }, valuation_frame


def _read_source_snippet(path_fragment: str, anchor: str, *, window: int = 26) -> tuple[str, str]:
    source_path = REPO_ROOT / path_fragment
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return str(source_path), "Source unavailable."

    start_index = 0
    for index, line in enumerate(lines):
        if anchor in line:
            start_index = index
            break
    end_index = min(len(lines), start_index + window)
    snippet = "\n".join(f"{line_no + 1:04d}: {lines[line_no]}" for line_no in range(start_index, end_index))
    return str(source_path), snippet


def _workflow_source_specs(current_model: str | None, stage: str | None) -> list[dict]:
    stage_key = str(stage or "").strip().lower()
    specs: list[dict] = []
    specs.extend(WORKFLOW_SOURCE_INDEX["initializing"])
    if current_model in WORKFLOW_SOURCE_INDEX:
        specs.extend(WORKFLOW_SOURCE_INDEX[current_model])
    if "pack" in stage_key or "complete" in stage_key:
        specs.extend(WORKFLOW_SOURCE_INDEX["packaging"])
    if "hybrid" in stage_key:
        specs.extend(WORKFLOW_SOURCE_INDEX["HYBRID"])

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in specs:
        key = (item["path"], item["anchor"])
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _render_workflow_code_panel(current_model: str | None, stage: str | None) -> None:
    specs = _workflow_source_specs(current_model, stage)
    st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">Code Path Behind This Stage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-note">These snippets are loaded from the actual local source files that are driving the workflow. They change with the active stage and model.</div>',
        unsafe_allow_html=True,
    )
    for item in specs:
        source_path, snippet = _read_source_snippet(item["path"], item["anchor"])
        st.markdown(f'<div class="dashboard-subtitle" style="margin-top:0.8rem;">{item["title"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note">{item["note"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note"><strong>Source:</strong> {source_path}</div>', unsafe_allow_html=True)
        st.code(snippet, language="python")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_workflow_graph_panel(
    *,
    current_model: str | None,
    current_output: dict | None,
    live_payload: dict | None = None,
) -> None:
    st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">What The Current Stage Is Producing</div>', unsafe_allow_html=True)
    if not current_output:
        st.markdown(
            '<div class="dashboard-note">Run the workflow to populate this section with the active model allocation, objective decomposition, regime profile, and latest trace rows from the backend.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return
    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Model", str(current_model or "N/A"), "Current solver stage emitting the visible portfolio.", "metric-accent-orange")
    with metric_cols[1]:
        metric_card("Objective", f"{float(current_output.get('objective_value', 0.0)):.4f}", "Current value of the unified optimization objective.", "metric-accent-green")
    with metric_cols[2]:
        metric_card("Return", f"{float(current_output.get('expected_return', 0.0)):.2%}", "Current stage expected return.", "metric-accent-green")
    with metric_cols[3]:
        metric_card("Risk", f"{float(current_output.get('risk', 0.0)):.2%}", "Current stage portfolio risk.", "metric-accent-red")
    breakdown = current_output.get("objective_breakdown") or {}
    regime_metrics = current_output.get("regime_metrics") or {}
    if breakdown:
        st.plotly_chart(
            build_objective_breakdown_figure(breakdown, title=f"{current_model or 'Stage'} Objective Breakdown"),
            use_container_width=True,
            key=f"workflow_objective_breakdown_{current_model}_{float(current_output.get('objective_value', 0.0)):.4f}",
        )
    if regime_metrics and regime_metrics.get("regime_names"):
        st.plotly_chart(
            build_regime_profile_figure(regime_metrics, title=f"{current_model or 'Stage'} Regime Return / Risk Profile"),
            use_container_width=True,
            key=f"workflow_regime_profile_{current_model}_{len(regime_metrics.get('regime_names', []))}",
        )
    if live_payload and live_payload.get("current_trace"):
        st.dataframe(pd.DataFrame(live_payload["current_trace"]).tail(12), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_storage_panel(snapshot: dict, *, key_suffix: str) -> None:
    if not snapshot:
        return
    history_count = len(snapshot.get("history") or [])
    st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">Stored Run</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dashboard-note">
            This optimization run has been persisted in the backend and can be revisited from future sessions.
        </div>
        <div class="dashboard-note" style="margin-top:0.45rem;"><strong>Job ID:</strong> {snapshot.get('job_id','')}</div>
        <div class="dashboard-note"><strong>Status:</strong> {snapshot.get('status','')}</div>
        <div class="dashboard-note"><strong>Stored events:</strong> {history_count}</div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        label="Download Stored Run Snapshot",
        data=json.dumps(snapshot, indent=2),
        file_name=f"optimization_run_{snapshot.get('job_id','run')}.json",
        mime="application/json",
        key=f"download_run_snapshot_{key_suffix}_{snapshot.get('job_id','run')}",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("dashboard_result", None)
    st.session_state.setdefault("live_job_id", None)
    st.session_state.setdefault("live_snapshot", None)
    st.session_state.setdefault("live_error", None)
    st.session_state.setdefault("run_archive", [])
    st.session_state.setdefault("selected_run_id", None)
    st.session_state.setdefault("dashboard_surface", "workflow")


def archive_snapshot(snapshot: dict | None) -> None:
    if not snapshot or not snapshot.get("job_id"):
        return
    archive = st.session_state.setdefault("run_archive", [])
    existing_ids = {item.get("job_id") for item in archive}
    if snapshot["job_id"] not in existing_ids:
        archive.insert(0, snapshot)
        st.session_state["selected_run_id"] = snapshot["job_id"]


def get_selected_archive_snapshot() -> dict | None:
    archive = collect_archived_runs()
    selected_run_id = st.session_state.get("selected_run_id")
    for item in archive:
        if item.get("job_id") == selected_run_id:
            return item
    return archive[0] if archive else None


def load_saved_run_snapshots() -> list[dict]:
    run_dir = Path("results/live_runs")
    if not run_dir.exists():
        return []
    snapshots: list[dict] = []
    for snapshot_path in sorted(run_dir.glob("*/snapshot.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            snapshots.append(json.loads(snapshot_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return snapshots


def collect_archived_runs() -> list[dict]:
    session_archive = st.session_state.get("run_archive", [])
    disk_archive = load_saved_run_snapshots()
    archive: list[dict] = []
    seen: set[str] = set()
    for item in session_archive + disk_archive:
        job_id = item.get("job_id")
        if job_id and job_id not in seen:
            archive.append(item)
            seen.add(job_id)
    return archive


def sync_live_state(api_url: str, payload: dict, run_live: bool) -> tuple[dict | None, dict | None, bool, str | None]:
    dashboard = st.session_state.get("dashboard_result")
    live_snapshot = st.session_state.get("live_snapshot")
    live_error = st.session_state.get("live_error")
    if run_live:
        try:
            live_snapshot = start_live_dashboard(api_url, payload)
            st.session_state["live_job_id"] = live_snapshot["job_id"]
            st.session_state["live_snapshot"] = live_snapshot
            st.session_state["dashboard_result"] = None
            st.session_state["live_error"] = None
            dashboard = None
            live_error = None
        except Exception as exc:
            live_error = f"Optimization failed to start: {exc}"
            st.session_state["live_error"] = live_error
    job_id = st.session_state.get("live_job_id")
    if job_id:
        try:
            live_snapshot = fetch_live_dashboard(api_url, job_id)
            st.session_state["live_snapshot"] = live_snapshot
            if live_snapshot["status"] == "succeeded" and live_snapshot.get("result"):
                dashboard = live_snapshot["result"]
                st.session_state["dashboard_result"] = dashboard
                st.session_state["live_job_id"] = None
                archive_snapshot(live_snapshot)
            elif live_snapshot["status"] == "failed":
                live_error = live_snapshot.get("error") or "Live optimization failed."
                st.session_state["live_error"] = live_error
                st.session_state["live_job_id"] = None
        except Exception as exc:
            live_error = f"Optimization polling failed: {exc}"
            st.session_state["live_error"] = live_error
    live_snapshot = st.session_state.get("live_snapshot")
    live_running = bool(live_snapshot and live_snapshot.get("status") in {"pending", "running"})
    return dashboard, live_snapshot, live_running, live_error


def render_live_optimizer(center_col, right_col, live_snapshot: dict, live_payload: dict, asset_labels: list[str]) -> None:
    current_model = live_payload.get("current_model")
    current_output = live_payload.get("current_output")
    with center_col:
        st.markdown(
            build_live_progress_banner(
                title="Live Backend Optimization Workflow",
                summary=live_payload.get("message") or "The backend is solving now. Trace points and 3D views update as each model advances.",
                stage=live_snapshot.get("stage"),
                progress=float(live_snapshot.get("progress", 0.0)),
                notes=["QP", "PSO", "GA", "HYBRID", "3D Trace"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_polling_status_cards(
                job_id=live_snapshot.get("job_id", ""),
                stage=live_snapshot.get("stage", "running"),
                status=live_snapshot.get("status", "running"),
                progress=float(live_snapshot.get("progress", 0.0)),
                last_update=live_snapshot.get("updated_at"),
            ),
            unsafe_allow_html=True,
        )
        stage_rows = live_payload.get("stage_rows") or []
        if stage_rows:
            st.plotly_chart(build_stage_timeline_figure(stage_rows, title="Backend Stage Timeline"), use_container_width=True, key="live_stage_timeline")
            st.dataframe(build_live_stage_table(stage_rows), use_container_width=True, hide_index=True)
        current_trace = live_payload.get("current_trace") or []
        candidate_history = live_payload.get("candidate_history") or []
        if current_trace:
            st.plotly_chart(build_live_trace_3d_figure(current_trace, title=f"{live_payload.get('current_model', 'Model')} Live Solver Path"), use_container_width=True, key=f"live_solver_path_{live_payload.get('current_model', 'model')}")
        partial_outputs = live_payload.get("model_outputs") or {}
        assets = live_payload.get("assets") or [{"name": label, "mu": 0.0, "sigma": 0.0, "liquidity": 0.0, "cost": 0.0} for label in asset_labels]
        if current_model and current_output:
            st.plotly_chart(
                build_model_allocation_figure(
                    current_model,
                    current_output["weights"],
                    asset_labels=[asset["name"] for asset in assets],
                    mu=[asset["mu"] for asset in assets],
                    sigma=[asset["sigma"] for asset in assets],
                    liquidity=[asset["liquidity"] for asset in assets],
                    cost=[asset["cost"] for asset in assets],
                ),
                use_container_width=True,
                key=f"live_current_allocation_{current_model}",
            )
        if partial_outputs:
            st.plotly_chart(build_combined_comparison_figure(partial_outputs, asset_labels=asset_labels), use_container_width=True, key="live_combined_comparison")
        _render_workflow_graph_panel(current_model=current_model, current_output=current_output, live_payload=live_payload)
        if candidate_history:
            st.plotly_chart(build_weight_score_progress_figure(candidate_history, title=f"{current_model or 'Model'} Decision Variable Weights And Scores"), use_container_width=True, key=f"live_weight_score_{current_model}")

    with right_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Live Solver Transparency</div>', unsafe_allow_html=True)
        decision_analysis = live_payload.get("decision_analysis")
        if decision_analysis:
            st.markdown(
                build_live_accept_reject_cards(
                    accepted_count=int(decision_analysis.get("accepted_count") or 0),
                    rejected_count=int(decision_analysis.get("rejected_count") or 0),
                    accepted_reason="Accepted assets satisfy the active-position and score rules.",
                    rejected_reason="Rejected assets are filtered because they are dominated on score or projected weight.",
                ),
                unsafe_allow_html=True,
            )
            candidate_frame = build_candidate_range_dataframe(decision_analysis["records"])
            st.plotly_chart(build_candidate_range_figure(decision_analysis["records"], title="Accepted / Rejected Assets During Solve"), use_container_width=True, key=f"live_accept_reject_{live_payload.get('current_model', 'model')}")
            st.dataframe(stylize_candidate_range_dataframe(candidate_frame), use_container_width=True, hide_index=True)
        surface = live_payload.get("current_surface")
        if surface:
            st.plotly_chart(build_objective_surface_3d_figure(surface, title=f"{live_payload.get('current_model', 'Model')} Objective Surface"), use_container_width=True, key=f"live_objective_surface_{live_payload.get('current_model', 'model')}")
        manifold = live_payload.get("current_manifold")
        if manifold and manifold.get("points"):
            st.plotly_chart(build_solver_manifold_figure(manifold, title=f"{live_payload.get('current_model', 'Model')} 3D Solver Manifold"), use_container_width=True, key=f"live_solver_manifold_{live_payload.get('current_model', 'model')}")
        if live_payload.get("candidate_summary"):
            st.markdown(f'<div class="insight-box"><strong>Candidate summary:</strong> {live_payload["candidate_summary"]}</div>', unsafe_allow_html=True)
        if live_payload.get("parameters"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.6rem;">Active Parameter Vector</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([live_payload["parameters"]]), use_container_width=True, hide_index=True)
        if live_payload.get("risk_calibration"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.6rem;">Risk Calibration</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([live_payload["risk_calibration"]]), use_container_width=True, hide_index=True)
        if live_payload.get("step_insights"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.6rem;">Mathematical Step Insight</div>', unsafe_allow_html=True)
            for insight in live_payload["step_insights"]:
                st.markdown(f'<div class="dashboard-note" style="margin-top:0.3rem;">- {insight}</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-subtitle" style="margin-top:0.8rem;">Hybrid Workflow Logic</div>', unsafe_allow_html=True)
        workflow_cards = [
            build_solver_step_html(title="1. QP Seed", body="Construct the first feasible candidate under the convex shrinkage-stabilized baseline.", accent="#0f172a"),
            build_solver_step_html(title="2. PSO Search", body="Use the QP seed as the swarm anchor and explore the global search surface.", accent="#2563eb"),
            build_solver_step_html(title="3. GA Refinement", body="Warm-start genetic evolution from the strongest swarm candidate.", accent="#0f766e"),
            build_solver_step_html(title="4. Score And Blend", body="Compute stage scores, softmax-blend the candidates, and project the blend back into the feasible set.", accent="#b45309"),
            build_solver_step_html(title="5. Final Selection", body="Select the minimum hybrid score among QP, PSO, GA, and the blended portfolio.", accent="#7c3aed"),
        ]
        for card in workflow_cards:
            st.markdown(card, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        _render_workflow_code_panel(current_model, live_snapshot.get("stage"))


def render_final_optimizer(
    center_col,
    right_col,
    dashboard: dict,
    dataset: str,
    active_assets_target: int,
    *,
    run_payload: dict | None = None,
    key_prefix: str = "final",
) -> None:
    selected_output = dashboard["selected_output"]
    selected_model = dashboard["selected_model"]
    assets = dashboard["assets"]
    model_outputs = dashboard["model_outputs"]
    selected_mc = dashboard["monte_carlo"][selected_model]
    decision_analysis = dashboard["intelligence"]["decision_variable_analysis"]
    math_payload = _resolve_math_payload(dashboard)
    objective_breakdown = _resolve_objective_breakdown(selected_output, math_payload)
    regime_metrics = _resolve_regime_metrics(selected_output, math_payload)
    valuation_memo, valuation_frame = _valuation_memo_payload(dashboard, run_payload or {})
    acceptance_summary = build_rejection_summary(
        accepted_count=int(decision_analysis.get("accepted_count") or 0),
        rejected_count=int(decision_analysis.get("rejected_count") or 0),
        accepted_reasons=[row["reason"] for row in decision_analysis["records"] if float(row["selected_weight"]) > 1e-6][:3],
        rejected_reasons=[row["reason"] for row in decision_analysis["records"] if float(row["selected_weight"]) <= 1e-6][:3],
    )
    st.markdown(
        build_solver_visibility_banner(
            title="How The Solver Is Working Now",
            subtitle=f"Selected model: {selected_model} | Active assets accepted: {decision_analysis.get('accepted_count', 0)} | Rejected: {decision_analysis.get('rejected_count', 0)}",
            summary="The engine derives decision-variable bands from mu, sigma, liquidity, and cost, then solves the constrained optimization problem and marks accepted versus rejected assets.",
            steps=["Minimum Features", "Decision Scores", "Accept / Reject", "3D Solver Path", "3D Objective Surface"],
        ),
        unsafe_allow_html=True,
    )
    with center_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-title">Selected Model: {selected_model}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            metric_card("Expected Return", f"{selected_output['expected_return']:.2%}", "Mean portfolio return from the chosen allocation.", "metric-accent-green")
        with cols[1]:
            metric_card("Risk", f"{selected_output['risk']:.2%}", "Quadratic portfolio risk from the covariance matrix.", "metric-accent-orange")
        cols = st.columns(2)
        with cols[0]:
            metric_card("CVaR Proxy", f"{abs(selected_mc['p05']):.2%}", "5th percentile Monte Carlo downside outcome.", "metric-accent-red")
        with cols[1]:
            metric_card("Active Assets", f"{selected_output['diversification']}", f"User target: {selected_output['active_assets_target']} assets.", "metric-accent-green")
        st.plotly_chart(
            build_model_allocation_figure(
                selected_model,
                selected_output["weights"],
                asset_labels=[asset["name"] for asset in assets],
                mu=[asset["mu"] for asset in assets],
                sigma=[asset["sigma"] for asset in assets],
                liquidity=[asset["liquidity"] for asset in assets],
                cost=[asset["cost"] for asset in assets],
            ),
            use_container_width=True,
            key=f"{key_prefix}_allocation_{selected_model}",
        )
        st.markdown(f'<div class="insight-box"><strong>Why this portfolio:</strong> {selected_output["explanation"]["why_this_allocation"]}</div>', unsafe_allow_html=True)
        st.markdown(
            build_solver_accept_reject_cards(
                accepted_count=int(decision_analysis.get("accepted_count") or 0),
                rejected_count=int(decision_analysis.get("rejected_count") or 0),
                accepted_reason="These assets survived the final feasibility and active-position rules.",
                rejected_reason="These assets were filtered out because they were dominated on score, weight band, or active-position selection.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="dashboard-note" style="margin-top:0.5rem;">{acceptance_summary["summary"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Portfolio Valuation Deck</div>', unsafe_allow_html=True)
        value_cols = st.columns(2)
        with value_cols[0]:
            metric_card("Capital Base", f"{valuation_memo['capital_budget']:.2f} Cr", "Investment budget translated into portfolio deployment.", "metric-accent-orange")
        with value_cols[1]:
            metric_card("Expected Profit", f"{valuation_memo['expected_profit']:.2f} Cr", "One-year expected portfolio profit from posterior return estimates.", "metric-accent-green")
        value_cols = st.columns(2)
        with value_cols[0]:
            metric_card("Downside Reserve", f"{valuation_memo['downside_reserve']:.2f} Cr", "CVaR-based loss reserve implied by the selected allocation.", "metric-accent-red")
        with value_cols[1]:
            metric_card("Bear-Case Value", f"{valuation_memo['bear_case_value']:.2f} Cr", "Projected portfolio value under the bear regime return profile.", "metric-accent-orange")
        st.plotly_chart(
            build_capital_deployment_figure(valuation_frame, title="Deployed Capital And Expected Profit By Asset"),
            use_container_width=True,
            key=f"{key_prefix}_capital_deployment",
        )
        st.plotly_chart(
            build_scenario_valuation_figure(regime_metrics, capital_budget=valuation_memo["capital_budget"], title="Scenario Valuation And Regime Risk"),
            use_container_width=True,
            key=f"{key_prefix}_scenario_valuation",
        )
        st.dataframe(valuation_frame, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Valuation Memo CSV",
            data=valuation_frame.to_csv(index=False),
            file_name=f"{key_prefix}_valuation_memo.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_valuation_memo_csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Model Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(build_combined_comparison_figure(model_outputs, asset_labels=[asset["name"] for asset in assets]), use_container_width=True, key=f"{key_prefix}_combined_comparison")
        st.dataframe(stylize_comparison_dataframe(build_comparison_dataframe(model_outputs)), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Accepted vs Rejected Assets</div>', unsafe_allow_html=True)
        candidate_frame = build_candidate_range_dataframe(decision_analysis["records"])
        st.plotly_chart(build_candidate_range_figure(decision_analysis["records"], title="Accepted / Rejected Allocation Outcome"), use_container_width=True, key=f"{key_prefix}_accept_reject_{selected_model}")
        st.dataframe(stylize_candidate_range_dataframe(candidate_frame), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Backend Insight</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="dashboard-note">
                The engine assembled a {len(assets)}-asset universe from the <strong>{dataset}</strong> regime,
                built a positive semidefinite covariance matrix, enforced an active-asset target of <strong>{active_assets_target}</strong>,
                calibrated the quadratic-risk target toward <strong>{float(math_payload.get('risk_target', 0.0087)):.2%}</strong>,
                and solved the portfolio with QP, PSO, GA, and Hybrid methods.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if math_payload.get("objective_latex"):
            st.latex(math_payload["objective_latex"])
        else:
            st.code(math_payload["objective_equation"], language="text")
        st.code(math_payload["hybrid_equation"], language="text")
        if math_payload.get("black_litterman_latex"):
            st.latex(math_payload["black_litterman_latex"])
        elif math_payload.get("return_engine_equation"):
            st.code(math_payload["return_engine_equation"], language="text")
        if math_payload.get("risk_calibration"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.7rem;">Risk Calibration Audit</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([math_payload["risk_calibration"]]), use_container_width=True, hide_index=True)
        if math_payload.get("feasibility_gate"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.7rem;">Feasibility Gate</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(math_payload["feasibility_gate"]).T.reset_index(names="Model"), use_container_width=True, hide_index=True)
        st.markdown(f'<div class="dashboard-note"><strong>Decision formula:</strong> {decision_analysis.get("decision_formula", "")}</div>', unsafe_allow_html=True)
        if math_payload.get("workflow_steps"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.7rem;">How The Math Operated On This Run</div>', unsafe_allow_html=True)
            for index, step in enumerate(math_payload["workflow_steps"], start=1):
                st.markdown(f'<div class="dashboard-note" style="margin-top:0.25rem;">{index}. {step}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Objective Decomposition</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_objective_breakdown_figure(objective_breakdown, title=f"{selected_model} Objective Terms"),
            use_container_width=True,
            key=f"{key_prefix}_objective_breakdown",
        )
        st.plotly_chart(
            build_regime_profile_figure(regime_metrics, title=f"{selected_model} Regime Return/Risk"),
            use_container_width=True,
            key=f"{key_prefix}_regime_profile",
        )
        if math_payload.get("benchmark_table"):
            st.plotly_chart(
                build_benchmark_attainment_figure(math_payload["benchmark_table"], title="Target vs Achieved Benchmarks"),
                use_container_width=True,
                key=f"{key_prefix}_benchmark_attainment",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        if math_payload.get("pipeline_graph"):
            st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
            st.markdown('<div class="dashboard-title">Optimization Pipeline Flow</div>', unsafe_allow_html=True)
            st.plotly_chart(
                build_pipeline_flow_figure(math_payload["pipeline_graph"]),
                use_container_width=True,
                key=f"{key_prefix}_pipeline_flow",
            )
            if math_payload.get("key_concepts_and_benchmarks"):
                for note in math_payload["key_concepts_and_benchmarks"]:
                    st.markdown(f'<div class="dashboard-note" style="margin-top:0.25rem;">- {note}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Frontier and Risk Structure</div>', unsafe_allow_html=True)
        st.plotly_chart(build_efficient_frontier_figure(dashboard["efficient_frontier"], selected_point={"risk": selected_output["risk"], "expected_return": selected_output["expected_return"]}), use_container_width=True, key=f"{key_prefix}_efficient_frontier")
        st.plotly_chart(build_risk_heatmap(dashboard["covariance_matrix"], asset_labels=[asset["name"] for asset in assets]), use_container_width=True, key=f"{key_prefix}_risk_heatmap")
        st.markdown("</div>", unsafe_allow_html=True)


def render_live_intelligence(live_payload: dict) -> None:
    left_col, right_col = st.columns([1.05, 1.0], gap="large")
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Live Execution Flow</div>', unsafe_allow_html=True)
        st.dataframe(build_live_stage_table(live_payload.get("stage_rows") or []), use_container_width=True, hide_index=True)
        if live_payload.get("latest_event"):
            event = live_payload["latest_event"]
            st.markdown(f'<div class="insight-box" style="margin-top:0.75rem;">Latest backend event: {event.get("event_type","")} at {event.get("stage","")}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Decision Variable Agent</div>', unsafe_allow_html=True)
        decision_analysis = live_payload.get("decision_analysis")
        if decision_analysis:
            st.dataframe(pd.DataFrame(decision_analysis["records"]), use_container_width=True, hide_index=True)
            st.plotly_chart(build_decision_space_3d_figure(decision_analysis["records"], title="Live Decision Variable Space"), use_container_width=True, key=f"live_decision_space_{live_payload.get('current_model', 'model')}")
            if live_payload.get("candidate_history"):
                st.plotly_chart(build_weight_score_progress_figure(live_payload.get("candidate_history") or [], title="Live Asset Weights And Scores"), use_container_width=True, key=f"live_weight_score_agent_{live_payload.get('current_model', 'model')}")
        else:
            st.info("Decision-variable classification becomes available once the current model produces a feasible candidate.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_final_intelligence(dashboard: dict) -> None:
    assets = dashboard["assets"]
    model_outputs = dashboard["model_outputs"]
    intelligence = dashboard["intelligence"]
    decision_analysis = intelligence["decision_variable_analysis"]
    math_payload = _resolve_math_payload(dashboard)
    left_col, right_col = st.columns([1.05, 1.0], gap="large")
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Execution Flow</div>', unsafe_allow_html=True)
        for step in intelligence["execution_flow"]:
            st.markdown(f'<div class="dashboard-note" style="margin-bottom:0.45rem;">{step}</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-subtitle" style="margin-top:0.65rem;">Execution Log</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(intelligence["execution_log"]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Decision Variables</div>', unsafe_allow_html=True)
        decision_rows = [{"Variable": key, "Meaning": value} for key, value in intelligence["decision_variables"].items()]
        st.dataframe(pd.DataFrame(decision_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Mathematical Model</div>', unsafe_allow_html=True)
        st.code(math_payload["multi_objective_formulation"], language="text")
        st.code(math_payload["normalization_equation"], language="text")
        st.code(math_payload["scalarization_equation"], language="text")
        if math_payload.get("objective_latex"):
            st.latex(math_payload["objective_latex"])
        else:
            st.code(math_payload["objective_equation"], language="text")
        st.code(math_payload["hybrid_equation"], language="text")
        if math_payload.get("black_litterman_latex"):
            st.latex(math_payload["black_litterman_latex"])
        elif math_payload.get("return_engine_equation"):
            st.code(math_payload["return_engine_equation"], language="text")
        st.markdown(f'<div class="dashboard-note"><strong>Scalarization note:</strong> {math_payload["scalarization_note"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note"><strong>Discrete model:</strong> {math_payload["discrete_real_estate_model"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note"><strong>Relaxed model:</strong> {math_payload["relaxed_real_estate_model"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(math_payload["parameter_table"]), use_container_width=True, hide_index=True)
        if math_payload.get("return_engine"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.65rem;">Return Estimation Engine</div>', unsafe_allow_html=True)
            return_engine = math_payload["return_engine"]
            return_rows = [
                {"Field": "Mode", "Value": return_engine.get("mode")},
                {"Field": "Prior Strength", "Value": return_engine.get("prior_strength")},
                {"Field": "Tau", "Value": return_engine.get("tau")},
                {"Field": "View Confidence", "Value": return_engine.get("view_confidence")},
            ]
            st.dataframe(pd.DataFrame(return_rows), use_container_width=True, hide_index=True)
        if math_payload.get("advanced_capabilities"):
            st.markdown('<div class="dashboard-subtitle" style="margin-top:0.65rem;">Advanced Capability Flags</div>', unsafe_allow_html=True)
            cap_rows = [{"Capability": key, "Enabled": value} for key, value in math_payload["advanced_capabilities"].items()]
            st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)
        for proposition in math_payload.get("theoretical_properties", []):
            st.markdown(
                f'<div class="insight-box" style="margin-top:0.65rem;"><strong>{proposition["title"]}:</strong> {proposition["statement"]} '
                f'<br><span class="dashboard-note">{proposition["interpretation"]}</span></div>',
                unsafe_allow_html=True,
            )
        coherent_risk = math_payload.get("coherent_risk_theory", {})
        if coherent_risk:
            st.markdown(
                f'<div class="dashboard-note" style="margin-top:0.7rem;"><strong>{coherent_risk["headline"]}</strong> '
                f'Properties: {", ".join(coherent_risk.get("properties", []))}. {coherent_risk.get("research_note", "")}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-title">{decision_analysis["agent_name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note">{decision_analysis["summary"]}</div>', unsafe_allow_html=True)
        for step in decision_analysis["methodology"]:
            st.markdown(f'<div class="dashboard-note" style="margin-top:0.3rem;">- {step}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        model_choice = st.selectbox("Explain model", list(intelligence["model_explanations"].keys()), index=list(intelligence["model_explanations"].keys()).index(dashboard["selected_model"]))
        explanation = intelligence["model_explanations"][model_choice]
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-title">{model_choice} Explainability Agent</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note"><strong>Objective:</strong> {explanation["objective_function"]}</div>', unsafe_allow_html=True)
        for step in explanation["solver_steps"]:
            st.markdown(f'<div class="dashboard-note" style="margin-bottom:0.35rem;">- {step}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">{explanation["why_this_allocation"]}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Explanation": explanation["asset_justifications"]}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Asset Matrix</div>', unsafe_allow_html=True)
        st.dataframe(build_asset_dataframe(assets, weights_by_model={model: payload["weights"] for model, payload in model_outputs.items()}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Decision Variable Dataset</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(decision_analysis["records"]), use_container_width=True, hide_index=True)
        st.plotly_chart(build_decision_space_3d_figure(decision_analysis["records"], title="Decision Variable Space"), use_container_width=True, key="final_decision_space")
        trace_model = dashboard["selected_model"]
        candidate_history = dashboard["intelligence"]["transparency"].get(trace_model, {}).get("solver_manifold", {}).get("points", [])
        if candidate_history:
            raw_candidate_history = dashboard["intelligence"]["transparency"].get(trace_model, {}).get("candidate_history", [])
            if raw_candidate_history:
                st.plotly_chart(build_weight_score_progress_figure(raw_candidate_history, title=f"{trace_model} Asset Weights And Scores"), use_container_width=True, key=f"final_weight_score_{trace_model}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Regime Mathematics</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(math_payload["regime_table"]), use_container_width=True, hide_index=True)
        st.plotly_chart(build_regime_scatter_3d_figure(math_payload["regime_3d_points"], title="3D Regime Outcome Surface"), use_container_width=True, key="final_regime_surface")
        if math_payload.get("pipeline_graph"):
            st.plotly_chart(
                build_pipeline_flow_figure(math_payload["pipeline_graph"], title="Backend Pipeline From Inputs To Evaluation"),
                use_container_width=True,
                key="final_pipeline_intelligence",
            )
        if math_payload.get("benchmark_table"):
            st.plotly_chart(
                build_benchmark_attainment_figure(math_payload["benchmark_table"], title="Core Benchmarks: Target vs Achieved"),
                use_container_width=True,
                key="final_benchmark_intelligence",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Comparison Matrix</div>', unsafe_allow_html=True)
        st.dataframe(stylize_comparison_dataframe(build_comparison_dataframe(model_outputs)), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_live_transparency(live_snapshot: dict, live_payload: dict) -> None:
    left_col, right_col = st.columns([1.0, 1.1], gap="large")
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Live Solver Transparency Agent</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note">Current backend stage: <strong>{live_snapshot.get("stage","running")}</strong></div>', unsafe_allow_html=True)
        if live_payload.get("latest_event"):
            st.dataframe(pd.DataFrame([live_payload["latest_event"]]), use_container_width=True, hide_index=True)
        if live_payload.get("current_trace"):
            st.dataframe(pd.DataFrame(live_payload["current_trace"]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        current_trace = live_payload.get("current_trace") or []
        if current_trace:
            st.plotly_chart(build_live_trace_3d_figure(current_trace, title=f"{live_payload.get('current_model', 'Model')} 3D Solver Trace"), use_container_width=True, key=f"live_transparency_trace_{live_payload.get('current_model', 'model')}")
        if live_payload.get("current_surface"):
            st.plotly_chart(build_objective_surface_3d_figure(live_payload["current_surface"], title=f"{live_payload.get('current_model', 'Model')} Objective Surface"), use_container_width=True, key=f"live_transparency_surface_{live_payload.get('current_model', 'model')}")
        if live_payload.get("current_manifold"):
            st.plotly_chart(build_solver_manifold_figure(live_payload["current_manifold"], title=f"{live_payload.get('current_model', 'Model')} 3D Solver Manifold"), use_container_width=True, key=f"live_transparency_manifold_{live_payload.get('current_model', 'model')}")
        render_storage_panel(live_snapshot, key_suffix="live")


def render_final_transparency(dashboard: dict, snapshot: dict | None = None) -> None:
    transparency = dashboard["intelligence"]["transparency"]
    left_col, right_col = st.columns([1.0, 1.1], gap="large")
    trace_model = st.selectbox("Trace model", list(transparency.keys()), index=list(transparency.keys()).index(dashboard["selected_model"]), key="trace_model_select")
    trace_payload = transparency[trace_model]
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-title">{trace_payload["agent_name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-note">{trace_payload["headline"]}</div>', unsafe_allow_html=True)
        for step in trace_payload["mathematical_walkthrough"]:
            st.markdown(f'<div class="dashboard-note" style="margin-bottom:0.35rem;">- {step}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(trace_payload["solver_trace"]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.plotly_chart(build_solver_trace_3d_figure(trace_payload["solver_trace"], title=f"{trace_model} 3D Solver Trace"), use_container_width=True, key=f"final_transparency_trace_{trace_model}")
        st.plotly_chart(build_objective_surface_3d_figure(trace_payload["objective_surface"], title=f"{trace_model} Objective Surface"), use_container_width=True, key=f"final_transparency_surface_{trace_model}")
        if trace_payload.get("solver_manifold"):
            st.plotly_chart(build_solver_manifold_figure(trace_payload["solver_manifold"], title=f"{trace_model} 3D Solver Manifold"), use_container_width=True, key=f"final_transparency_manifold_{trace_model}")
        st.plotly_chart(
            build_efficient_frontier_figure(
                dashboard["efficient_frontier"],
                selected_point={
                    "risk": dashboard["selected_output"]["risk"],
                    "expected_return": dashboard["selected_output"]["expected_return"],
                },
            ),
            use_container_width=True,
            key=f"final_transparency_frontier_{trace_model}",
        )
        if snapshot:
            render_storage_panel(snapshot, key_suffix="final")


def render_workflow_archive() -> None:
    archive = collect_archived_runs()
    if not archive:
        st.info("No completed runs are stored in the frontend yet. Finish a live optimization run and it will appear here.")
        return
    options = [f"{item.get('job_id')} | {item.get('stage')} | {item.get('status')}" for item in archive]
    selected_label = st.selectbox("Stored run", options, index=0, key="archive_select")
    selected_run_id = selected_label.split(" | ")[0]
    st.session_state["selected_run_id"] = selected_run_id
    snapshot = get_selected_archive_snapshot()
    if snapshot is None:
        st.info("No stored snapshot available.")
        return
    history_rows = []
    for item in snapshot.get("history", []):
        history_rows.append(
            {
                "timestamp": item.get("timestamp", ""),
                "event": item.get("event", ""),
                "stage": item.get("stage", ""),
                "progress": item.get("progress", ""),
                "message": item.get("message", ""),
            }
        )
    left_col, right_col = st.columns([1.1, 1.0], gap="large")
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Stored Workflow History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Stored Workflow Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="dashboard-note"><strong>Job ID:</strong> {snapshot.get('job_id','')}</div>
            <div class="dashboard-note"><strong>Status:</strong> {snapshot.get('status','')}</div>
            <div class="dashboard-note"><strong>Stage:</strong> {snapshot.get('stage','')}</div>
            <div class="dashboard-note"><strong>Created:</strong> {snapshot.get('created_at','')}</div>
            <div class="dashboard-note"><strong>Finished:</strong> {snapshot.get('finished_at','')}</div>
            <div class="dashboard-note"><strong>Events captured:</strong> {len(history_rows)}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        render_storage_panel(snapshot, key_suffix="archive")


def render_output_results_page(snapshot: dict | None, dashboard: dict | None) -> None:
    archive = collect_archived_runs()
    selected_run_id = st.session_state.get("selected_run_id")
    current_job_id = (snapshot or {}).get("job_id")
    option_map: dict[str, dict | None] = {}
    options: list[str] = []
    if dashboard is not None:
        current_label = f"{current_job_id or 'current-session'} | current | in-memory"
        options.append(current_label)
        option_map[current_label] = snapshot
    if archive:
        for item in archive:
            label = f"{item.get('job_id')} | {item.get('stage')} | {item.get('status')}"
            options.append(label)
            option_map[label] = item
    if options:
        default_index = 0
        if selected_run_id:
            for idx, label in enumerate(options):
                selected_snapshot = option_map[label]
                if selected_snapshot and selected_snapshot.get("job_id") == selected_run_id:
                    default_index = idx
                    break
        selected_label = st.selectbox("Result run", options, index=default_index, key="results_run_select")
        chosen_snapshot = option_map[selected_label]
        st.session_state["selected_run_id"] = (chosen_snapshot or {}).get("job_id")
    else:
        chosen_snapshot = None

    selected_snapshot = chosen_snapshot if dashboard is not None else (chosen_snapshot or get_selected_archive_snapshot() or snapshot)
    selected_dashboard = dashboard
    if selected_snapshot and selected_snapshot.get("result") and (dashboard is None or selected_snapshot.get("job_id") != current_job_id):
        selected_dashboard = selected_snapshot["result"]
    if selected_dashboard is None:
        st.info("Run a live optimization job first. Final charts, 3D views, and analysis will appear here once the run completes.")
        return
    left_col, right_col = st.columns([1.25, 1.0], gap="large")
    run_payload = _run_payload(selected_snapshot, selected_dashboard)
    dataset = run_payload.get("dataset", "balanced")
    active_assets_target = int(run_payload.get("active_assets_target", selected_dashboard["selected_output"]["active_assets_target"]))
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-bottom:1rem;">
            <div class="dashboard-title">Output Intelligence</div>
            <div class="dashboard-note">
                Final portfolio output for run <strong>{(selected_snapshot or {}).get('job_id', 'current')}</strong>.
                This tab is the investor-facing delivery surface: allocation, return-risk profile, comparison, and 3D diagnostic views.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pie_left, pie_right = st.columns([1.0, 1.0], gap="large")
    with pie_left:
        st.plotly_chart(
            build_allocation_pie_figure(
                [asset["name"] for asset in selected_dashboard["assets"]],
                selected_dashboard["selected_output"]["weights"],
                title=f"{selected_dashboard['selected_model']} Allocation Mix",
            ),
            use_container_width=True,
            key="results_page_allocation_pie",
        )
    with pie_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="dashboard-note"><strong>Selected model:</strong> {selected_dashboard['selected_model']}</div>
            <div class="dashboard-note"><strong>Dataset regime:</strong> {dataset}</div>
            <div class="dashboard-note"><strong>Expected return:</strong> {selected_dashboard['selected_output']['expected_return']:.2%}</div>
            <div class="dashboard-note"><strong>Risk:</strong> {selected_dashboard['selected_output']['risk']:.2%}</div>
            <div class="dashboard-note"><strong>Sharpe:</strong> {selected_dashboard['selected_output']['sharpe']:.2f}</div>
            <div class="dashboard-note"><strong>Active assets:</strong> {selected_dashboard['selected_output']['diversification']} / {active_assets_target}</div>
            <div class="dashboard-note"><strong>Capital base:</strong> {_budget_from_payload(run_payload):.2f} Cr</div>
            <div class="dashboard-note"><strong>Return engine:</strong> {_resolve_math_payload(selected_dashboard).get('return_engine', {}).get('mode', 'N/A')}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    render_final_optimizer(left_col, right_col, selected_dashboard, dataset, active_assets_target, run_payload=run_payload, key_prefix="results_page")
    st.markdown("---")
    st.markdown('<div class="dashboard-title">Output Analysis</div>', unsafe_allow_html=True)
    render_final_intelligence(selected_dashboard)
    st.markdown("---")
    st.markdown('<div class="dashboard-title">3D Solver And Objective Views</div>', unsafe_allow_html=True)
    render_final_transparency(selected_dashboard, selected_snapshot)


def render_page_shell(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="dashboard-card" style="margin-bottom:1rem;">
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_surface_selector() -> str:
    st.markdown(
        """
        <div class="dashboard-card" style="margin-bottom:1rem;">
            <div class="dashboard-title">Platform Views</div>
            <div class="dashboard-note">
                Use these buttons to switch between the live optimization workflow and the final investor-facing system output.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workflow_col, output_col = st.columns(2, gap="large")
    with workflow_col:
        if st.button(
            "Workflow",
            use_container_width=True,
            type="primary" if st.session_state.get("dashboard_surface") == "workflow" else "secondary",
            key="surface_workflow_button",
        ):
            st.session_state["dashboard_surface"] = "workflow"
    with output_col:
        if st.button(
            "System Output",
            use_container_width=True,
            type="primary" if st.session_state.get("dashboard_surface") == "output" else "secondary",
            key="surface_output_button",
        ):
            st.session_state["dashboard_surface"] = "output"
    return st.session_state.get("dashboard_surface", "workflow")


def render_workflow_page() -> None:
    render_page_shell(
        "Workflow Storage",
        "Run the optimizer, inspect the live mathematical workflow, and keep the saved process history for later study.",
    )
    left_col, center_col, right_col = st.columns([1.05, 1.3, 1.15], gap="large")
    with left_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-title">Portfolio Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-note">This page is focused on workflow. Start a run here, watch the backend solve, then inspect the stored workflow history below.</div>', unsafe_allow_html=True)
        control_label("API URL")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, label_visibility="collapsed", key="workflow_api_url")
        control_label("Asset Count", "Number of assets in the generated investment universe.")
        asset_count = st.slider("Asset Count", min_value=10, max_value=15, value=12, step=1, label_visibility="collapsed", key="workflow_asset_count")
        control_label("Diversification Preference", "Choose how many assets should receive meaningful allocation in the final portfolio.")
        active_assets_target = st.slider("Diversification Preference", min_value=2, max_value=asset_count, value=min(4, asset_count), step=1, label_visibility="collapsed", key="workflow_active_assets")
        control_label("Dataset Regime", "Controls the return, risk, liquidity, and sector profile of the asset universe.")
        dataset = st.selectbox("Dataset Regime", DATASET_OPTIONS, index=0, label_visibility="collapsed", key="workflow_dataset")
        control_label("Random Seed", "Keeps the synthetic universe reproducible.")
        seed = st.number_input("Random Seed", min_value=0, value=42, step=1, label_visibility="collapsed", key="workflow_seed")
        control_label("Budget (Cr)", "Capital available for the optimization problem.")
        budget = st.slider("Budget (Cr)", min_value=1.0, max_value=10.0, value=5.0, step=0.5, label_visibility="collapsed", key="workflow_budget")
        control_label("Max Allocation", "Maximum weight allowed in any single asset.")
        max_allocation = st.slider("Max Allocation", min_value=0.10, max_value=0.50, value=0.35, step=0.01, label_visibility="collapsed", key="workflow_max_allocation")
        control_label("Risk Tolerance (lambda)", "Higher lambda pushes the optimizer toward lower-risk portfolios.")
        lambda_value = st.slider("Risk Tolerance (lambda)", min_value=0.2, max_value=3.0, value=1.0, step=0.1, label_visibility="collapsed", key="workflow_lambda")
        control_label("Liquidity Preference", "Higher values reward assets that are easier to trade.")
        liquidity_weight = st.slider("Liquidity Preference", min_value=0.0, max_value=1.5, value=0.35, step=0.05, label_visibility="collapsed", key="workflow_liquidity")
        control_label("Transaction Cost Sensitivity", "Higher values penalize expensive allocations more strongly.")
        cost_sensitivity = st.slider("Transaction Cost Sensitivity", min_value=0.0, max_value=1.0, value=0.15, step=0.05, label_visibility="collapsed", key="workflow_cost")
        control_label("Tail Risk Weight", "Penalizes severe downside outcomes through the loss-based CVaR term.")
        tail_risk_weight = st.slider("Tail Risk Weight", min_value=0.0, max_value=1.5, value=0.45, step=0.05, label_visibility="collapsed", key="workflow_tail")
        control_label("Diversification Gap Weight", "Controls how strongly the optimizer pushes toward the target active-asset structure.")
        diversification_weight = st.slider("Diversification Gap Weight", min_value=0.0, max_value=1.5, value=0.30, step=0.05, label_visibility="collapsed", key="workflow_diversification")
        control_label("Concentration Penalty", "Directly penalizes HHI concentration.")
        concentration_penalty = st.slider("Concentration Penalty", min_value=0.0, max_value=1.5, value=0.18, step=0.05, label_visibility="collapsed", key="workflow_concentration")
        control_label("Regime Aversion", "Penalizes instability across bear, neutral, and bull regimes.")
        regime_aversion = st.slider("Regime Aversion", min_value=0.0, max_value=1.5, value=0.28, step=0.05, label_visibility="collapsed", key="workflow_regime")
        control_label("Tracking Penalty", "Keeps the portfolio closer to a stable reference and makes the hybrid consensus tighter.")
        tracking_penalty = st.slider("Tracking Penalty", min_value=0.0, max_value=1.5, value=0.12, step=0.05, label_visibility="collapsed", key="workflow_tracking")
        control_label("Covariance Shrinkage", "Blends the covariance matrix toward its diagonal for robustness.")
        covariance_shrinkage = st.slider("Covariance Shrinkage", min_value=0.0, max_value=0.8, value=0.15, step=0.05, label_visibility="collapsed", key="workflow_shrinkage")
        control_label("CVaR Confidence", "Confidence level for the downside tail-loss penalty.")
        cvar_alpha = st.slider("CVaR Confidence", min_value=0.80, max_value=0.99, value=0.95, step=0.01, label_visibility="collapsed", key="workflow_cvar_alpha")
        control_label("Optimization Model", "Choose Auto, QP, PSO, GA, or Hybrid.")
        method = st.selectbox("Optimization Model", METHOD_OPTIONS, index=0, label_visibility="collapsed", key="workflow_method")
        with st.expander("Research-Grade Return And Seed Controls", expanded=False):
            control_label("Return Engine", "Choose the expected-return estimator that feeds the unchanged optimizer.")
            return_engine = st.selectbox("Return Engine", RETURN_ENGINE_OPTIONS, index=0, label_visibility="collapsed", key="workflow_return_engine")
            control_label("Prior Strength", "Higher values shrink empirical views more strongly toward the prior.")
            return_prior_strength = st.slider("Prior Strength", min_value=0.0, max_value=0.95, value=0.55, step=0.05, label_visibility="collapsed", key="workflow_prior_strength")
            control_label("Black-Litterman Tau", "Scales prior covariance uncertainty in the return engine.")
            bl_tau = st.slider("Black-Litterman Tau", min_value=0.01, max_value=0.50, value=0.08, step=0.01, label_visibility="collapsed", key="workflow_bl_tau")
            control_label("View Confidence", "Controls how strongly the return engine trusts its Bayesian views.")
            bl_view_confidence = st.slider("View Confidence", min_value=0.10, max_value=0.95, value=0.65, step=0.05, label_visibility="collapsed", key="workflow_bl_confidence")
            exact_cvar_seed = st.checkbox("Enable Exact CVaR Seed", value=True, key="workflow_exact_cvar_seed")
            exact_discrete_selection = st.checkbox("Enable Exact Discrete Selection", value=True, key="workflow_exact_discrete")
        ensure_asset_table(asset_count, dataset, int(seed))
        editor_key = f"workflow_asset_editor_{st.session_state['_asset_signature']}"
        control_label("Asset Matrix", "Edit expected return, risk, liquidity, cost, and sector for each asset.")
        asset_editor = st.data_editor(pd.DataFrame(st.session_state["asset_rows"]), use_container_width=True, hide_index=True, key=editor_key, num_rows="fixed")
        asset_value_preview = build_asset_value_estimate_dataframe(
            asset_editor.to_dict(orient="records"),
            capital_budget=budget,
        )
        st.markdown(
            '<div class="dashboard-note" style="margin-top:0.7rem;">Estimated values are budget-scaled previews from return, risk, liquidity, and cost. They are not final solver outputs, but they let you price every asset before running the optimizer.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(asset_value_preview, use_container_width=True, hide_index=True)
        asset_simulation = build_asset_simulation_dataframe(
            asset_editor.to_dict(orient="records"),
            capital_budget=budget,
            seed=int(seed),
            n_paths=1200,
        )
        st.markdown(
            '<div class="dashboard-subtitle" style="margin-top:0.85rem;">Monte Carlo Asset Simulation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="dashboard-note" style="margin-top:0.25rem;">This simulation runs 1,200 one-year paths per asset using the edited return, volatility, liquidity, and cost inputs. The chart shows the mean simulated value with a 5th-95th percentile range.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_asset_simulation_figure(asset_simulation, title="Monte Carlo Asset Value Simulation"),
            use_container_width=True,
            key=f"workflow_asset_simulation_{st.session_state['_asset_signature']}_{int(seed)}_{budget}",
        )
        st.dataframe(asset_simulation, use_container_width=True, hide_index=True)
        run_live = st.button("Run Live Optimization", type="primary", use_container_width=True, key="workflow_run_live")
        st.markdown("</div>", unsafe_allow_html=True)

    payload = build_payload(
        asset_count=asset_count,
        active_assets_target=active_assets_target,
        budget=budget,
        max_allocation=max_allocation,
        lambda_value=lambda_value,
        liquidity_weight=liquidity_weight,
        cost_sensitivity=cost_sensitivity,
        tail_risk_weight=tail_risk_weight,
        diversification_weight=diversification_weight,
        concentration_penalty=concentration_penalty,
        regime_aversion=regime_aversion,
        tracking_penalty=tracking_penalty,
        covariance_shrinkage=covariance_shrinkage,
        cvar_alpha=cvar_alpha,
        method=method,
        dataset=dataset,
        seed=int(seed),
        assets=asset_editor.to_dict(orient="records"),
        return_engine=return_engine,
        return_prior_strength=return_prior_strength,
        bl_tau=bl_tau,
        bl_view_confidence=bl_view_confidence,
        exact_cvar_seed=exact_cvar_seed,
        exact_discrete_selection=exact_discrete_selection,
    )
    dashboard, live_snapshot, live_running, live_error = sync_live_state(api_url, payload, run_live)
    live_payload = (live_snapshot or {}).get("latest_payload") or {}
    asset_labels = [row["name"] for row in (live_payload.get("assets") or asset_editor.to_dict(orient="records"))]
    archived_snapshot = get_selected_archive_snapshot()

    if live_running and live_snapshot:
        render_live_optimizer(center_col, right_col, live_snapshot, live_payload, asset_labels)
    else:
        completed_snapshot = live_snapshot or archived_snapshot
        completed_payload = {}
        if completed_snapshot:
            completed_payload = (completed_snapshot.get("latest_payload") or {})
            if not completed_payload and completed_snapshot.get("result"):
                completed_payload = completed_snapshot["result"].get("live_console") or {}
        completed_asset_labels = [
            row["name"]
            for row in (
                completed_payload.get("assets")
                or asset_editor.to_dict(orient="records")
            )
        ]
        if completed_snapshot and completed_payload:
            if live_error and dashboard is None and not live_snapshot:
                with center_col:
                    st.warning(f"{live_error} Showing the latest preserved workflow instead.")
            else:
                st.success("Run completed. The workflow view below is preserved on this tab, and the final output remains available in the Output Results tab.")
            st.markdown(
                """
                <div class="dashboard-card" style="margin-bottom:1rem;">
                    <div class="dashboard-title">Workflow Agent Console</div>
                    <div class="dashboard-note">
                        The formulation and solving workflow is preserved below exactly for post-run inspection. The final investor-facing output remains isolated in the Output Results tab.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_live_optimizer(center_col, right_col, completed_snapshot, completed_payload, completed_asset_labels)
        else:
            if live_error and dashboard is None:
                with center_col:
                    st.error(live_error)
            else:
                with center_col:
                    st.info("Start a live optimization run from the left panel to populate the workflow storage and solver audit views.")
                with right_col:
                    st.info("The stored workflow section below will list completed runs from disk as soon as they are available.")
            with center_col:
                _render_workflow_graph_panel(current_model=method if method != "AUTO" else "HYBRID", current_output=None, live_payload=None)
            with right_col:
                _render_workflow_code_panel(method if method != "AUTO" else "HYBRID", "initializing")

    st.markdown("---")
    render_workflow_archive()
    render_workflow_results_section(live_snapshot or archived_snapshot, dashboard)


def render_output_page() -> None:
    render_page_shell(
        "Output Results",
        "Inspect the final portfolio output, allocation charts, scenario views, comparison matrix, and risk-return structure from stored runs.",
    )
    render_output_results_page(st.session_state.get("live_snapshot"), st.session_state.get("dashboard_result"))


def render_workflow_results_section(snapshot: dict | None, dashboard: dict | None) -> None:
    selected_snapshot = get_selected_archive_snapshot() or snapshot
    selected_dashboard = dashboard
    if selected_snapshot and selected_snapshot.get("result"):
        selected_dashboard = selected_snapshot["result"]
    if selected_dashboard is None:
        return
    run_payload = _run_payload(selected_snapshot, selected_dashboard)
    dataset = run_payload.get("dataset", "balanced")
    active_assets_target = int(run_payload.get("active_assets_target", selected_dashboard["selected_output"]["active_assets_target"]))
    st.markdown("---")
    render_page_shell(
        "Workflow Results Snapshot",
        "This is the final optimization output kept directly on the workflow page so the process and result stay together.",
    )
    left_col, right_col = st.columns([1.25, 1.0], gap="large")
    render_final_optimizer(
        left_col,
        right_col,
        selected_dashboard,
        dataset,
        active_assets_target,
        run_payload=run_payload,
        key_prefix="workflow_page_results",
    )


def main() -> None:
    st.set_page_config(page_title="BrixSquare AI Investment Optimizer", page_icon=" ", layout="wide")
    init_state()
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown(VISIBILITY_CSS, unsafe_allow_html=True)
    st.markdown(LIVE_CONSOLE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-card" style="margin-bottom:1rem;">
            <div class="dashboard-title">BrixSquare AI Investment Optimizer</div>
            <div class="dashboard-subtitle">Multi-model portfolio optimization with live workflow persistence, valuation-grade output, and mathematical transparency</div>
            <div class="dashboard-note" style="margin-top:0.35rem;font-weight:700;color:#0f172a;">
                Live Build: Decision Variable Agent + Solver Transparency Agent + 3D Objective Views + Portfolio Valuation Deck
            </div>
            <div class="dashboard-note" style="margin-top:0.45rem;">
                Configure the asset universe, choose how many assets you want actively used in the final portfolio,
                inspect how the optimization engine solves the problem mathematically while it runs, and keep both the workflow and final valuation memo on-screen after completion.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_surface = render_surface_selector()
    if selected_surface == "workflow":
        render_workflow_page()
    else:
        render_output_page()

    if st.session_state.get("live_job_id"):
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
