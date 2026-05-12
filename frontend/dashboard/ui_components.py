"""
Reusable UI helpers for the Streamlit dashboard.

This module is intentionally UI-only:
- Plotly chart builders
- DataFrame builders for model comparisons
- Dashboard CSS for readable white-card styling

No Streamlit page wiring lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from pandas.io.formats.style import Styler


DASHBOARD_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #f3f6fb 100%);
    color: #111111;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

.dashboard-card {
    background: #ffffff;
    color: #111111;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 18px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
    padding: 1rem 1.05rem;
}

.dashboard-title {
    color: #111111;
    font-weight: 800;
    font-size: 1.15rem;
}

.dashboard-subtitle {
    color: #334155;
    font-weight: 600;
    font-size: 0.92rem;
}

.dashboard-note {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.5;
}

.metric-label {
    color: #475569;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.78rem;
}

.metric-value {
    color: #111111;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1.05;
}

.metric-accent-green {
    color: #1f7a4d;
}

.metric-accent-orange {
    color: #b45309;
}

.metric-accent-red {
    color: #b91c1c;
}

.insight-box {
    background: #f8fafc;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-left: 4px solid #0f172a;
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    color: #111111;
}
</style>
"""


MODEL_COLORS = {
    "QP": "#0f172a",
    "PSO": "#2563eb",
    "GA": "#0f766e",
    "HYBRID": "#b45309",
}

DEFAULT_ASSET_COLORS = qualitative.Set2 + qualitative.Plotly


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence of numeric values.")
    return array


def _normalize_labels(labels: Sequence[str] | None, size: int, prefix: str = "Asset") -> list[str]:
    if labels is None:
        return [f"{prefix} {idx + 1}" for idx in range(size)]
    normalized = [str(label) for label in labels]
    if len(normalized) != size:
        raise ValueError("Label count must match the number of weights.")
    return normalized


def _prepare_asset_frame(
    weights: Sequence[float],
    asset_labels: Sequence[str] | None = None,
    mu: Sequence[float] | None = None,
    sigma: Sequence[float] | None = None,
    liquidity: Sequence[float] | None = None,
    cost: Sequence[float] | None = None,
) -> pd.DataFrame:
    weight_array = _as_float_array(weights)
    labels = _normalize_labels(asset_labels, len(weight_array))
    data: dict[str, Any] = {"asset": labels, "weight": weight_array}
    if mu is not None:
        data["mu"] = _as_float_array(mu)
    if sigma is not None:
        data["sigma"] = _as_float_array(sigma)
    if liquidity is not None:
        data["liquidity"] = _as_float_array(liquidity)
    if cost is not None:
        data["cost"] = _as_float_array(cost)
    frame = pd.DataFrame(data)
    if len(frame) != len(weight_array):
        raise ValueError("Input arrays must all have the same length.")
    return frame


def build_model_dataframe(model_outputs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Build a comparison table from a mapping of model name -> model payload.

    Expected payload keys:
    - expected_return
    - risk
    - sharpe (optional; derived if missing)
    - diversification (optional; derived if missing)
    """

    rows: list[dict[str, Any]] = []
    for model_name, payload in model_outputs.items():
        expected_return = float(payload.get("expected_return", 0.0))
        risk = float(payload.get("risk", 0.0))
        sharpe = float(payload.get("sharpe", expected_return / risk if risk else 0.0))
        diversification = int(payload.get("diversification", payload.get("diversification_score", 0)))
        rows.append(
            {
                "Model": model_name,
                "Return": expected_return,
                "Risk": risk,
                "Sharpe": sharpe,
                "Diversification": diversification,
            }
        )
    return pd.DataFrame(rows)


def build_model_allocation_figure(
    model_name: str,
    weights: Sequence[float],
    asset_labels: Sequence[str] | None = None,
    mu: Sequence[float] | None = None,
    sigma: Sequence[float] | None = None,
    liquidity: Sequence[float] | None = None,
    cost: Sequence[float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """
    Create a bar chart for one model's allocation.
    """

    frame = _prepare_asset_frame(weights, asset_labels=asset_labels, mu=mu, sigma=sigma, liquidity=liquidity, cost=cost)
    frame = frame.sort_values("weight", ascending=False).reset_index(drop=True)

    custom_columns = [col for col in ["mu", "sigma", "liquidity", "cost"] if col in frame.columns]
    customdata = np.column_stack([frame[col].to_numpy() for col in custom_columns]) if custom_columns else None
    hover_parts = ["<b>%{x}</b>", "Weight: %{y:.2%}"]
    for index, column in enumerate(custom_columns):
        hover_label = column.capitalize()
        hover_parts.append(f"{hover_label}: %{{customdata[{index}]:.4f}}")

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["asset"],
            y=frame["weight"],
            marker_color=MODEL_COLORS.get(model_name.upper(), "#0f172a"),
            customdata=customdata,
            hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            name=model_name,
        )
    )
    figure.update_layout(
        title=title or f"{model_name} Allocation",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Asset", tickangle=-25, showgrid=False),
        yaxis=dict(title="Weight", tickformat=".0%", gridcolor="rgba(15,23,42,0.08)"),
        showlegend=False,
        height=360,
    )
    return figure


def build_combined_comparison_figure(
    model_outputs: Mapping[str, Mapping[str, Any]],
    asset_labels: Sequence[str] | None = None,
    title: str = "Model Comparison",
) -> go.Figure:
    """
    Overlay all model allocations on one chart.
    """

    if not model_outputs:
        return go.Figure().update_layout(title=title)

    first_weights = next(iter(model_outputs.values())).get("weights", [])
    labels = _normalize_labels(asset_labels, len(first_weights))

    figure = go.Figure()
    for model_name, payload in model_outputs.items():
        weights = _as_float_array(payload.get("weights", []))
        if len(weights) != len(labels):
            raise ValueError("All model outputs must share the same asset universe.")
        figure.add_trace(
            go.Bar(
                x=labels,
                y=weights,
                name=model_name,
                marker_color=MODEL_COLORS.get(model_name.upper(), "#64748b"),
                hovertemplate=f"{model_name}<br>%{{x}}<br>Weight: %{{y:.2%}}<extra></extra>",
            )
        )

    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        barmode="group",
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Asset", tickangle=-25, showgrid=False),
        yaxis=dict(title="Weight", tickformat=".0%", gridcolor="rgba(15,23,42,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=390,
    )
    return figure


def build_efficient_frontier_figure(
    frontier_points: Sequence[Mapping[str, float]],
    selected_point: Mapping[str, float] | None = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Plot frontier points and optionally highlight the selected portfolio.
    """

    frame = pd.DataFrame(frontier_points)
    if frame.empty:
        return go.Figure().update_layout(title=title)

    x_key = "risk" if "risk" in frame.columns else "variance"
    y_key = "expected_return" if "expected_return" in frame.columns else "return"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame[x_key],
            y=frame[y_key],
            mode="lines+markers",
            name="Frontier",
            line=dict(color="#0f172a", width=2.8),
            marker=dict(size=7, color="#b45309"),
            hovertemplate=f"Risk: %{{x:.4f}}<br>Return: %{{y:.4f}}<extra></extra>",
        )
    )

    if selected_point:
        figure.add_trace(
            go.Scatter(
                x=[float(selected_point.get("risk", selected_point.get("variance", 0.0)))],
                y=[float(selected_point.get("expected_return", selected_point.get("return", 0.0)))],
                mode="markers",
                name="Selected Portfolio",
                marker=dict(size=14, color="#1f7a4d", symbol="diamond"),
                hovertemplate="Selected portfolio<extra></extra>",
            )
        )

    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Risk", gridcolor="rgba(15,23,42,0.08)"),
        yaxis=dict(title="Expected Return", gridcolor="rgba(15,23,42,0.08)"),
        height=340,
        hovermode="closest",
    )
    return figure


def build_risk_heatmap(
    covariance_matrix: Sequence[Sequence[float]] | np.ndarray,
    asset_labels: Sequence[str] | None = None,
    title: str = "Risk Heatmap",
) -> go.Figure:
    """
    Show the covariance matrix as a heatmap.
    """

    matrix = np.asarray(covariance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    labels = _normalize_labels(asset_labels, matrix.shape[0])

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Covariance"),
            hovertemplate="%{x} x %{y}<br>Covariance: %{z:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        height=420,
    )
    return figure


def build_solver_trace_3d_figure(
    trace: Sequence[Mapping[str, float | str]],
    *,
    title: str = "3D Solver Trace",
) -> go.Figure:
    frame = pd.DataFrame(trace)
    if frame.empty:
        return go.Figure().update_layout(title=title)
    if "stage" not in frame.columns:
        frame["stage"] = [f"Iter {int(value)}" for value in frame["iteration"]]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=frame["iteration"],
            y=frame["risk"],
            z=frame["objective_value"],
            mode="lines+markers",
            marker=dict(
                size=5,
                color=frame["expected_return"],
                colorscale="Viridis",
                colorbar=dict(title="Return"),
            ),
            line=dict(color="#0f172a", width=5),
            text=frame["stage"],
            hovertemplate=(
                "Step: %{text}<br>"
                "Iteration: %{x:.0f}<br>"
                "Risk: %{y:.4f}<br>"
                "Objective: %{z:.4f}<br>"
                "Return: %{marker.color:.4f}<extra></extra>"
            ),
            name="Solver Path",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title="Iteration",
            yaxis_title="Risk",
            zaxis_title="Objective",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=460,
    )
    return figure


def build_objective_surface_3d_figure(
    surface: Mapping[str, Any],
    *,
    title: str = "Objective Surface",
) -> go.Figure:
    if not surface:
        return go.Figure().update_layout(title=title)
    x = np.asarray(surface.get("x", []), dtype=float)
    y = np.asarray(surface.get("y", []), dtype=float)
    z = np.asarray(surface.get("z", []), dtype=float)
    if x.size == 0 or y.size == 0 or z.size == 0:
        return go.Figure().update_layout(title=title)

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Objective"),
            hovertemplate=(
                f"{surface.get('asset_x', 'Asset X')}: %{{x:.2f}}<br>"
                f"{surface.get('asset_y', 'Asset Y')}: %{{y:.2f}}<br>"
                "Objective: %{z:.4f}<extra></extra>"
            ),
            name="Objective Surface",
        )
    )
    selected = surface.get("selected_point", {})
    if selected:
        figure.add_trace(
            go.Scatter3d(
                x=[float(selected.get("x", 0.0))],
                y=[float(selected.get("y", 0.0))],
                z=[float(selected.get("z", 0.0))],
                mode="markers",
                marker=dict(size=8, color="#b45309"),
                name="Selected point",
                hovertemplate="Selected solution<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title=str(surface.get("asset_x", "Asset X")),
            yaxis_title=str(surface.get("asset_y", "Asset Y")),
            zaxis_title="Objective",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=480,
    )
    return figure


def build_decision_space_3d_figure(
    records: Sequence[Mapping[str, Any]],
    *,
    title: str = "Decision Variable Space",
) -> go.Figure:
    frame = pd.DataFrame(records)
    if frame.empty:
        return go.Figure().update_layout(title=title)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=frame["mu_norm"],
            y=1.0 - frame["risk_norm"],
            z=frame["liquidity_norm"],
            mode="markers+text",
            text=frame["name"],
            textposition="top center",
            marker=dict(
                size=8 + 26 * frame["selected_weight"],
                color=frame["decision_score"],
                colorscale="Viridis",
                colorbar=dict(title="Decision Score"),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Return score: %{x:.3f}<br>"
                "Stability score: %{y:.3f}<br>"
                "Liquidity score: %{z:.3f}<br>"
                "Decision score: %{marker.color:.3f}<extra></extra>"
            ),
            name="Assets",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title="Normalized Return",
            yaxis_title="Stability (1 - Risk)",
            zaxis_title="Normalized Liquidity",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=500,
    )
    return figure


def build_solver_manifold_figure(
    manifold_payload: Mapping[str, Any],
    *,
    title: str = "3D Solver Manifold",
) -> go.Figure:
    points = list(manifold_payload.get("points") or [])
    labels = list(manifold_payload.get("asset_labels") or ["w1", "w2", "w3"])
    frame = pd.DataFrame(points)
    if frame.empty:
        return go.Figure().update_layout(title=title)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=frame["x"],
            y=frame["y"],
            z=frame["z"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=frame["loss_cvar_95"] if "loss_cvar_95" in frame.columns else frame["objective_value"],
                colorscale="Turbo",
                colorbar=dict(title="CVaR"),
            ),
            line=dict(color="#0f172a", width=4),
            text=frame["stage"] if "stage" in frame.columns else None,
            hovertemplate=(
                f"{labels[0]}: %{{x:.3f}}<br>"
                f"{labels[1] if len(labels) > 1 else 'w2'}: %{{y:.3f}}<br>"
                f"{labels[2] if len(labels) > 2 else 'w3'}: %{{z:.3f}}<br>"
                "Risk: %{customdata[0]:.4f}<br>"
                "Return: %{customdata[1]:.4f}<br>"
                "CVaR: %{customdata[2]:.4f}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    frame["risk"] if "risk" in frame.columns else np.zeros(len(frame)),
                    frame["expected_return"] if "expected_return" in frame.columns else np.zeros(len(frame)),
                    frame["loss_cvar_95"] if "loss_cvar_95" in frame.columns else np.zeros(len(frame)),
                ]
            ),
            name="Decision Variable Path",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title=labels[0] if labels else "w1",
            yaxis_title=labels[1] if len(labels) > 1 else "w2",
            zaxis_title=labels[2] if len(labels) > 2 else "w3",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=500,
    )
    return figure


def build_weight_score_progress_figure(
    candidate_history: Sequence[Mapping[str, Any]],
    *,
    title: str = "Decision Variable Weights And Scores",
) -> go.Figure:
    if not candidate_history:
        return go.Figure().update_layout(title=title)
    frame = pd.DataFrame(candidate_history)
    if frame.empty or "weights" not in frame.columns:
        return go.Figure().update_layout(title=title)
    weight_matrix = np.asarray(frame["weights"].tolist(), dtype=float)
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] == 0:
        return go.Figure().update_layout(title=title)
    final_weights = weight_matrix[-1]
    top_idx = np.argsort(final_weights)[::-1][:3]
    figure = go.Figure()
    for idx in top_idx:
        figure.add_trace(
            go.Scatter(
                x=list(range(len(frame))),
                y=weight_matrix[:, int(idx)],
                mode="lines+markers",
                name=f"Weight {int(idx) + 1}",
                hovertemplate="Candidate %{x}<br>Weight: %{y:.3f}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=list(range(len(frame))),
            y=frame["objective_value"].astype(float),
            mode="lines+markers",
            name="Objective",
            yaxis="y2",
            line=dict(color="#b45309", width=3, dash="dash"),
            hovertemplate="Candidate %{x}<br>Objective: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Candidate Snapshot", showgrid=False),
        yaxis=dict(title="Weight", gridcolor="rgba(15,23,42,0.08)"),
        yaxis2=dict(title="Objective", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380,
    )
    return figure


def build_objective_breakdown_figure(
    objective_breakdown: Mapping[str, float],
    *,
    title: str = "Objective Term Decomposition",
) -> go.Figure:
    if not objective_breakdown:
        figure = go.Figure()
        figure.add_annotation(
            text="Objective-term data is not available for this stored run.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#334155"),
        )
        figure.update_layout(
            title=title,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#111111"),
            height=360,
        )
        return figure
    frame = pd.DataFrame(
        [
            {"term": key.replace("_", " ").title(), "value": float(value)}
            for key, value in objective_breakdown.items()
        ]
    )
    figure = go.Figure(
        data=[
            go.Bar(
                x=frame["term"],
                y=frame["value"],
                marker_color=np.where(frame["value"] >= 0.0, "#b45309", "#1f7a4d"),
                hovertemplate="%{x}<br>Contribution: %{y:.4f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=60),
        xaxis=dict(title="Objective term", tickangle=-28, showgrid=False),
        yaxis=dict(title="Contribution", gridcolor="rgba(15,23,42,0.08)"),
        showlegend=False,
        height=360,
    )
    return figure


def build_regime_profile_figure(
    regime_metrics: Mapping[str, Any],
    *,
    title: str = "Regime Return And Risk Profile",
) -> go.Figure:
    regime_names = list(regime_metrics.get("regime_names", []))
    regime_returns = list(regime_metrics.get("regime_expected_returns", []))
    regime_risks = list(regime_metrics.get("regime_risks", []))
    if not regime_names or not regime_returns or not regime_risks:
        figure = go.Figure()
        figure.add_annotation(
            text="Regime return/risk data is not available for this stored run.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#334155"),
        )
        figure.update_layout(
            title=title,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#111111"),
            height=360,
        )
        return figure

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=regime_names,
            y=regime_returns,
            mode="lines+markers",
            name="Expected Return",
            line=dict(color="#1f7a4d", width=3),
            marker=dict(size=9),
            hovertemplate="%{x}<br>Return: %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=regime_names,
            y=regime_risks,
            mode="lines+markers",
            name="Risk",
            yaxis="y2",
            line=dict(color="#b45309", width=3, dash="dash"),
            marker=dict(size=9),
            hovertemplate="%{x}<br>Risk: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Regime", showgrid=False),
        yaxis=dict(title="Expected Return", gridcolor="rgba(15,23,42,0.08)"),
        yaxis2=dict(title="Risk", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=360,
    )
    return figure


def build_regime_scatter_3d_figure(
    regime_points: Sequence[Mapping[str, Any]],
    *,
    title: str = "Regime Outcome Surface",
) -> go.Figure:
    frame = pd.DataFrame(regime_points)
    if frame.empty:
        return go.Figure().update_layout(title=title)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=frame["bear_return"],
            y=frame["neutral_return"],
            z=frame["bull_return"],
            mode="markers+text",
            text=frame["model"],
            textposition="top center",
            marker=dict(
                size=9 + 100 * np.clip(frame["risk"], 0.0, None),
                color=frame["loss_cvar_95"],
                colorscale="YlOrRd",
                colorbar=dict(title="Loss CVaR"),
                opacity=0.92,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Bear return: %{x:.4f}<br>"
                "Neutral return: %{y:.4f}<br>"
                "Bull return: %{z:.4f}<br>"
                "Loss CVaR: %{marker.color:.4f}<extra></extra>"
            ),
            name="Models",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title="Bear Return",
            yaxis_title="Neutral Return",
            zaxis_title="Bull Return",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=500,
    )
    return figure


def build_comparison_dataframe(model_outputs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Build a comparison dataframe ready for display.
    """

    frame = build_model_dataframe(model_outputs)
    if frame.empty:
        return frame
    frame = frame.sort_values("Return", ascending=False).reset_index(drop=True)
    return frame


def build_asset_dataframe(
    assets: Sequence[Mapping[str, Any]],
    weights_by_model: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """
    Build an asset-level dataframe. Optionally include model weights.
    """

    rows: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        row = {
            "Asset": asset.get("name", f"Asset {index + 1}"),
            "mu": float(asset.get("mu", 0.0)),
            "sigma": float(asset.get("sigma", 0.0)),
            "liquidity": float(asset.get("liquidity", 0.0)),
            "cost": float(asset.get("cost", 0.0)),
            "sector": str(asset.get("sector", "Residential")),
        }
        if weights_by_model:
            for model_name, weights in weights_by_model.items():
                weights_array = _as_float_array(weights)
                row[f"{model_name} Weight"] = float(weights_array[index]) if index < len(weights_array) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_asset_value_estimate_dataframe(
    assets: Sequence[Mapping[str, Any]],
    *,
    capital_budget: float,
) -> pd.DataFrame:
    """
    Build a read-only asset valuation preview from the editable asset features.

    The estimates are heuristic and budget-scaled:
    - score uses the same transparent decision-variable ingredients as the
      explainability layer
    - estimated capital allocates the current budget across all assets based on
      that score
    - projected values apply return, cost drag, and a simple stress haircut
    """

    if not assets:
        return pd.DataFrame()

    mu = np.asarray([float(asset.get("mu", 0.0)) for asset in assets], dtype=float)
    sigma = np.asarray([float(asset.get("sigma", 0.0)) for asset in assets], dtype=float)
    liquidity = np.asarray([float(asset.get("liquidity", 0.0)) for asset in assets], dtype=float)
    cost = np.asarray([float(asset.get("cost", 0.0)) for asset in assets], dtype=float)

    mu_norm = (mu - mu.min()) / (mu.max() - mu.min()) if not np.isclose(mu.max(), mu.min()) else np.full_like(mu, 0.5)
    sigma_norm = (sigma - sigma.min()) / (sigma.max() - sigma.min()) if not np.isclose(sigma.max(), sigma.min()) else np.full_like(sigma, 0.5)
    liquidity_norm = (liquidity - liquidity.min()) / (liquidity.max() - liquidity.min()) if not np.isclose(liquidity.max(), liquidity.min()) else np.full_like(liquidity, 0.5)
    cost_norm = (cost - cost.min()) / (cost.max() - cost.min()) if not np.isclose(cost.max(), cost.min()) else np.full_like(cost, 0.5)

    score = 0.50 * mu_norm + 0.25 * liquidity_norm + 0.20 * (1.0 - sigma_norm) - 0.15 * cost_norm
    score_floor = float(np.min(score))
    positive_score = score - score_floor + 0.05
    estimated_weight = positive_score / np.sum(positive_score)
    estimated_capital = float(capital_budget) * estimated_weight

    projected_multiple = np.maximum(0.0, 1.0 + mu - cost)
    stress_multiple = np.maximum(0.0, 1.0 + mu - cost - 1.65 * sigma)
    liquidity_discount = 1.0 - 0.08 * (1.0 - liquidity)

    rows: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        rows.append(
            {
                "Asset": asset.get("name", f"Asset {index + 1}"),
                "Sector": str(asset.get("sector", "Residential")),
                "Score": float(score[index]),
                "Estimated Weight": float(estimated_weight[index]),
                "Estimated Capital (Cr)": float(estimated_capital[index]),
                "Projected Value (Cr)": float(estimated_capital[index] * projected_multiple[index] * liquidity_discount[index]),
                "Stress Value (Cr)": float(estimated_capital[index] * stress_multiple[index] * liquidity_discount[index]),
                "Expected Profit (Cr)": float(estimated_capital[index] * max(mu[index] - cost[index], 0.0)),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Projected Value (Cr)", "Estimated Capital (Cr)"], ascending=[False, False]).reset_index(drop=True)
    return frame


def build_asset_simulation_dataframe(
    assets: Sequence[Mapping[str, Any]],
    *,
    capital_budget: float,
    seed: int = 42,
    n_paths: int = 1000,
) -> pd.DataFrame:
    """
    Simulate one-year asset values from the editable asset matrix.

    Each asset receives a budget-scaled estimated capital slice derived from the
    same score used in the value-estimate preview. The simulation then applies
    normally distributed one-year returns using mu as drift and sigma as
    volatility, with cost drag and a small liquidity haircut.
    """

    estimate_frame = build_asset_value_estimate_dataframe(assets, capital_budget=capital_budget)
    if estimate_frame.empty:
        return estimate_frame

    ordered_assets = {
        str(asset.get("name", f"Asset {index + 1}")): asset
        for index, asset in enumerate(assets)
    }
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []

    for row in estimate_frame.to_dict(orient="records"):
        asset = ordered_assets.get(str(row["Asset"]), {})
        initial_capital = float(row["Estimated Capital (Cr)"])
        mu = float(asset.get("mu", 0.0))
        sigma = float(asset.get("sigma", 0.0))
        cost = float(asset.get("cost", 0.0))
        liquidity = float(asset.get("liquidity", 0.0))

        returns = rng.normal(loc=mu - cost, scale=max(sigma, 1e-6), size=max(int(n_paths), 100))
        returns = np.clip(returns, -0.95, 2.50)
        liquidity_multiplier = float(np.clip(1.0 - 0.08 * (1.0 - liquidity), 0.80, 1.0))
        terminal_values = initial_capital * (1.0 + returns) * liquidity_multiplier

        mean_value = float(np.mean(terminal_values))
        p05 = float(np.quantile(terminal_values, 0.05))
        p50 = float(np.quantile(terminal_values, 0.50))
        p95 = float(np.quantile(terminal_values, 0.95))
        prob_loss = float(np.mean(terminal_values < initial_capital))

        rows.append(
            {
                "Asset": row["Asset"],
                "Sector": row["Sector"],
                "Initial Capital (Cr)": initial_capital,
                "Mean Simulated Value (Cr)": mean_value,
                "P05 Value (Cr)": p05,
                "Median Value (Cr)": p50,
                "P95 Value (Cr)": p95,
                "Expected Profit (Cr)": mean_value - initial_capital,
                "Loss Probability": prob_loss,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Mean Simulated Value (Cr)", "P95 Value (Cr)"], ascending=[False, False]).reset_index(drop=True)
    return frame


def build_asset_simulation_figure(
    simulation_frame: pd.DataFrame,
    *,
    title: str = "Monte Carlo Asset Value Simulation",
) -> go.Figure:
    if simulation_frame.empty:
        return go.Figure().update_layout(title=title)

    mean_values = simulation_frame["Mean Simulated Value (Cr)"].to_numpy(dtype=float)
    p05_values = simulation_frame["P05 Value (Cr)"].to_numpy(dtype=float)
    p95_values = simulation_frame["P95 Value (Cr)"].to_numpy(dtype=float)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=simulation_frame["Asset"],
            y=mean_values,
            mode="markers",
            name="Mean Value",
            marker=dict(size=11, color="#2563eb"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(p95_values - mean_values),
                arrayminus=(mean_values - p05_values),
                color="#94a3b8",
                thickness=1.5,
                width=5,
            ),
            customdata=np.column_stack(
                [
                    simulation_frame["Initial Capital (Cr)"].to_numpy(dtype=float),
                    simulation_frame["Median Value (Cr)"].to_numpy(dtype=float),
                    simulation_frame["Loss Probability"].to_numpy(dtype=float),
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Initial capital: %{customdata[0]:.2f} Cr<br>"
                "Mean simulated value: %{y:.2f} Cr<br>"
                "Median value: %{customdata[1]:.2f} Cr<br>"
                "Loss probability: %{customdata[2]:.1%}<br>"
                "P05-P95 band shown as error bars<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Asset", tickangle=-25, showgrid=False),
        yaxis=dict(title="Simulated Value (Cr)", gridcolor="rgba(15,23,42,0.08)"),
        showlegend=False,
        height=380,
    )
    return figure


def build_portfolio_valuation_dataframe(
    assets: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    capital_budget: float,
    expected_returns: Sequence[float] | None = None,
) -> pd.DataFrame:
    weight_array = _as_float_array(weights)
    posterior_mu = None if expected_returns is None else _as_float_array(expected_returns)
    rows: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        weight = float(weight_array[index]) if index < len(weight_array) else 0.0
        if weight <= 1e-6:
            continue
        deployed = float(capital_budget) * weight
        expected_return = float(posterior_mu[index]) if posterior_mu is not None and index < len(posterior_mu) else float(asset.get("mu", 0.0))
        risk = float(asset.get("sigma", 0.0))
        liquidity = float(asset.get("liquidity", 0.0))
        cost = float(asset.get("cost", 0.0))
        rows.append(
            {
                "Asset": asset.get("name", f"Asset {index + 1}"),
                "Weight": weight,
                "Capital (Cr)": deployed,
                "Expected Return": expected_return,
                "Expected Profit (Cr)": deployed * expected_return,
                "Risk": risk,
                "Liquidity": liquidity,
                "Cost Drag (Cr)": deployed * cost,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Weight", "Expected Profit (Cr)"], ascending=[False, False]).reset_index(drop=True)
    return frame


def build_capital_deployment_figure(
    valuation_frame: pd.DataFrame,
    *,
    title: str = "Capital Deployment And Expected Profit",
) -> go.Figure:
    if valuation_frame.empty:
        return go.Figure().update_layout(title=title)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=valuation_frame["Asset"],
            y=valuation_frame["Capital (Cr)"],
            name="Capital",
            marker_color="#0f172a",
            hovertemplate="%{x}<br>Capital: %{y:.2f} Cr<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=valuation_frame["Asset"],
            y=valuation_frame["Expected Profit (Cr)"],
            mode="lines+markers",
            name="Expected Profit",
            line=dict(color="#1f7a4d", width=3),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x}<br>Expected Profit: %{y:.2f} Cr<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Asset", tickangle=-25, showgrid=False),
        yaxis=dict(title="Capital (Cr)", gridcolor="rgba(15,23,42,0.08)"),
        yaxis2=dict(title="Expected Profit (Cr)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380,
    )
    return figure


def build_scenario_valuation_figure(
    regime_metrics: Mapping[str, Any],
    *,
    capital_budget: float,
    title: str = "Scenario Portfolio Valuation",
) -> go.Figure:
    regime_names = list(regime_metrics.get("regime_names", []))
    regime_returns = np.asarray(regime_metrics.get("regime_expected_returns", []), dtype=float)
    regime_risks = np.asarray(regime_metrics.get("regime_risks", []), dtype=float)
    if not regime_names or regime_returns.size == 0:
        return go.Figure().update_layout(title=title)
    projected_values = float(capital_budget) * (1.0 + regime_returns)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=regime_names,
            y=projected_values,
            name="Projected Value",
            marker_color="#2563eb",
            hovertemplate="%{x}<br>Projected Value: %{y:.2f} Cr<extra></extra>",
        )
    )
    if regime_risks.size == len(regime_names):
        figure.add_trace(
            go.Scatter(
                x=regime_names,
                y=regime_risks,
                mode="lines+markers",
                name="Regime Risk",
                yaxis="y2",
                line=dict(color="#b45309", width=3, dash="dash"),
                marker=dict(size=8),
                hovertemplate="%{x}<br>Risk: %{y:.4f}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Regime", showgrid=False),
        yaxis=dict(title="Projected Portfolio Value (Cr)", gridcolor="rgba(15,23,42,0.08)"),
        yaxis2=dict(title="Regime Risk", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=360,
    )
    return figure


def stylize_comparison_dataframe(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Return a display-friendly styled dataframe.
    """

    if frame.empty:
        return frame.style
    return frame.style.format(
        {
            "Return": "{:.2%}",
            "Risk": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Diversification": "{:d}",
        }
    )


def build_pipeline_flow_figure(
    pipeline_payload: Mapping[str, Any],
    *,
    title: str = "Optimization Pipeline Flow",
) -> go.Figure:
    nodes = list(pipeline_payload.get("nodes") or [])
    links = list(pipeline_payload.get("links") or [])
    if not nodes or not links:
        return go.Figure().update_layout(title=title)

    labels = [str(node.get("label", node.get("id", f"Node {idx + 1}"))) for idx, node in enumerate(nodes)]
    ids = [str(node.get("id", f"node_{idx}")) for idx, node in enumerate(nodes)]
    id_to_index = {node_id: idx for idx, node_id in enumerate(ids)}
    groups = [str(node.get("group", "stage")) for node in nodes]
    group_colors = {
        "input": "#0f172a",
        "estimation": "#1d4ed8",
        "risk": "#0f766e",
        "objective": "#b45309",
        "solver": "#7c3aed",
        "evaluation": "#be123c",
    }
    node_colors = [group_colors.get(group, "#334155") for group in groups]
    link_sources: list[int] = []
    link_targets: list[int] = []
    link_values: list[float] = []
    link_labels: list[str] = []
    for item in links:
        source = str(item.get("source", ""))
        target = str(item.get("target", ""))
        if source not in id_to_index or target not in id_to_index:
            continue
        link_sources.append(id_to_index[source])
        link_targets.append(id_to_index[target])
        link_values.append(max(0.1, float(item.get("value", 1.0))))
        link_labels.append(str(item.get("label", "")))
    if not link_sources:
        return go.Figure().update_layout(title=title)

    figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "pad": 14,
                    "thickness": 16,
                    "line": {"color": "rgba(15,23,42,0.18)", "width": 0.6},
                    "label": labels,
                    "color": node_colors,
                    "hovertemplate": "%{label}<extra></extra>",
                },
                link={
                    "source": link_sources,
                    "target": link_targets,
                    "value": link_values,
                    "label": link_labels,
                    "hovertemplate": "%{source.label} -> %{target.label}<br>%{label}<extra></extra>",
                    "color": "rgba(37,99,235,0.22)",
                },
            )
        ]
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=8, r=8, t=46, b=16),
        height=430,
    )
    return figure


def build_benchmark_attainment_figure(
    benchmark_rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Benchmark Attainment",
) -> go.Figure:
    frame = pd.DataFrame(benchmark_rows)
    if frame.empty:
        return go.Figure().update_layout(title=title)

    for column in ("achieved", "target"):
        if column not in frame.columns:
            frame[column] = 0.0
    if "metric" not in frame.columns:
        frame["metric"] = [f"Metric {idx + 1}" for idx in range(len(frame))]

    frame["achieved"] = frame["achieved"].astype(float)
    frame["target"] = frame["target"].astype(float)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["metric"],
            y=frame["target"],
            name="Target",
            marker_color="#94a3b8",
            hovertemplate="%{x}<br>Target: %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=frame["metric"],
            y=frame["achieved"],
            name="Achieved",
            marker_color="#1d4ed8",
            hovertemplate="%{x}<br>Achieved: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=56),
        xaxis=dict(title="Metric", tickangle=-20, showgrid=False),
        yaxis=dict(title="Value", gridcolor="rgba(15,23,42,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380,
    )
    return figure
