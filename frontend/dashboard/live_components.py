"""
Reusable live-console helpers for the Streamlit optimization workflow.

This module is intentionally helper-only:
- progress banners
- stage timeline visualizations
- polling status cards
- accepted vs rejected cards
- live 3D trace containers

No Streamlit page wiring lives here.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LIVE_CONSOLE_CSS = """
<style>
.live-console-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
    color: #ffffff;
    border-radius: 20px;
    padding: 1rem 1.15rem;
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.live-console-title {
    font-size: 1.08rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 0.25rem;
}

.live-console-copy {
    font-size: 0.92rem;
    line-height: 1.45;
    color: rgba(255, 255, 255, 0.92);
}

.live-status-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
}

.live-status-card {
    background: #ffffff;
    color: #111111;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
}

.live-status-card.accepted {
    border-left: 4px solid #1f7a4d;
}

.live-status-card.rejected {
    border-left: 4px solid #b91c1c;
}

.live-status-card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
    color: #475569;
    margin-bottom: 0.35rem;
}

.live-status-card-value {
    font-size: 1.95rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 0.2rem;
}

.live-status-card-copy {
    font-size: 0.9rem;
    line-height: 1.45;
    color: #334155;
}

.live-pill {
    display: inline-block;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.06);
    color: #0f172a;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

.live-muted {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.5;
}
</style>
"""


def _as_float_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence of numeric values.")
    return array


def _normalize_labels(labels: Sequence[str] | None, size: int, prefix: str = "Step") -> list[str]:
    if labels is None:
        return [f"{prefix} {idx + 1}" for idx in range(size)]
    normalized = [str(label) for label in labels]
    if len(normalized) != size:
        raise ValueError("Label count must match the input length.")
    return normalized


def _escape_text(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _clamp_progress(progress: float | None) -> float:
    if progress is None:
        return 0.0
    return max(0.0, min(float(progress), 1.0))


def build_live_progress_banner(
    *,
    title: str,
    summary: str,
    stage: str | None = None,
    progress: float | None = None,
    notes: Sequence[str] | None = None,
) -> str:
    """Return a high-contrast banner for the current optimization state."""

    safe_title = _escape_text(title)
    safe_summary = _escape_text(summary)
    stage_markup = f'<div class="live-pill">{_escape_text(stage)}</div>' if stage else ""
    progress_markup = ""
    if progress is not None:
        pct = _clamp_progress(progress) * 100.0
        progress_markup = f'<div class="live-muted" style="margin-top:0.45rem;font-weight:700;">Progress: {pct:.0f}%</div>'
    note_markup = ""
    if notes:
        note_markup = "".join(f'<span class="live-pill">{_escape_text(note)}</span>' for note in notes)
    return (
        '<div class="live-console-banner">'
        f'<div class="live-console-title">{safe_title}</div>'
        f'{stage_markup}'
        f'<div class="live-console-copy" style="margin-top:0.25rem;">{safe_summary}</div>'
        f"{progress_markup}"
        f'<div style="margin-top:0.6rem;">{note_markup}</div>'
        "</div>"
    )


def build_stage_timeline_figure(
    stages: Sequence[Mapping[str, Any]],
    *,
    title: str = "Optimization Timeline",
) -> go.Figure:
    """
    Visualize solver stages and their states.

    Each stage may contain:
    - name
    - status: pending | running | accepted | rejected | completed
    - progress
    - duration_sec
    """

    frame = pd.DataFrame(stages)
    if frame.empty:
        return go.Figure().update_layout(title=title)

    if "name" not in frame.columns:
        frame["name"] = [f"Stage {idx + 1}" for idx in range(len(frame))]
    if "status" not in frame.columns:
        frame["status"] = "completed"
    if "progress" not in frame.columns:
        frame["progress"] = np.linspace(0.0, 1.0, len(frame))
    if "duration_sec" not in frame.columns:
        frame["duration_sec"] = 0.0

    color_map = {
        "pending": "#94a3b8",
        "running": "#2563eb",
        "accepted": "#1f7a4d",
        "rejected": "#b91c1c",
        "completed": "#0f172a",
    }
    frame["color"] = frame["status"].map(color_map).fillna("#0f172a")
    frame["y"] = list(range(len(frame), 0, -1))
    frame["duration_sec"] = frame["duration_sec"].fillna(0.0).astype(float)
    frame["progress"] = frame["progress"].fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["progress"],
            y=frame["y"],
            mode="lines+markers+text",
            text=frame["name"],
            textposition="top center",
            marker=dict(size=14, color=frame["color"], line=dict(color="white", width=1.5)),
            line=dict(color="#0f172a", width=3),
            customdata=np.column_stack([frame["status"], frame["duration_sec"]]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Status: %{customdata[0]}<br>"
                "Progress: %{x:.0%}<br>"
                "Duration: %{customdata[1]:.2f}s<extra></extra>"
            ),
            name="Stages",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111"),
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(title="Progress", tickformat=".0%", gridcolor="rgba(15,23,42,0.08)", range=[0, 1]),
        yaxis=dict(title=None, showticklabels=False, gridcolor="rgba(15,23,42,0.08)"),
        height=330,
        hovermode="closest",
        showlegend=False,
    )
    return figure


def build_stage_timeline_cards(
    stages: Sequence[Mapping[str, Any]],
    *,
    title: str = "Stage Timeline",
) -> str:
    """Return compact HTML cards summarizing stage state for Streamlit markdown blocks."""

    frame = build_stage_table(stages)
    if frame.empty:
        return f'<div class="live-muted">{_escape_text(title)}: no stages reported yet.</div>'

    cards = []
    for _, row in frame.iterrows():
        status = str(row["status"])
        status_class = "accepted" if status == "accepted" else "rejected" if status == "rejected" else ""
        cards.append(
            """
            <div class="live-status-card {status_class}" style="margin-bottom:0.5rem;">
                <div class="live-status-card-title">{stage}</div>
                <div class="live-status-card-copy">
                    Status: <strong>{status}</strong><br>
                    Progress: <strong>{progress:.0%}</strong><br>
                    Duration: <strong>{duration:.2f}s</strong>
                </div>
            </div>
            """.format(
                status_class=status_class,
                stage=_escape_text(row["stage"]),
                status=_escape_text(status),
                progress=float(row["progress"]),
                duration=float(row["duration_sec"]),
            )
        )
    return f'<div class="live-console-banner" style="padding:0.85rem 0.95rem;">' \
        f'<div class="live-console-title">{_escape_text(title)}</div>' \
        + "".join(cards) + "</div>"


def build_polling_status_cards(
    *,
    job_id: str,
    stage: str,
    status: str,
    progress: float,
    last_update: str | None = None,
) -> str:
    """Return compact cards for polling state and job metadata."""

    progress_pct = _clamp_progress(progress) * 100.0
    update_text = f"<div class='live-muted'>Last update: {_escape_text(last_update)}</div>" if last_update else ""
    return f"""
    <div class="live-status-grid">
        <div class="live-status-card">
            <div class="live-status-card-title">Job</div>
            <div class="live-status-card-value" style="color:#0f172a;">{_escape_text(job_id)}</div>
            <div class="live-status-card-copy">Current stage: {_escape_text(stage)}</div>
        </div>
        <div class="live-status-card">
            <div class="live-status-card-title">Status</div>
            <div class="live-status-card-value" style="color:#2563eb;">{_escape_text(status)}</div>
            <div class="live-status-card-copy">Progress: {progress_pct:.0f}%</div>
            {update_text}
        </div>
    </div>
    """


def build_live_accept_reject_cards(
    *,
    accepted_count: int,
    rejected_count: int,
    accepted_reason: str,
    rejected_reason: str,
) -> str:
    total = max(accepted_count + rejected_count, 1)
    accepted_share = accepted_count / total
    rejected_share = rejected_count / total
    return f"""
    <div class="live-status-grid">
        <div class="live-status-card accepted">
            <div class="live-status-card-title">Accepted</div>
            <div class="live-status-card-value" style="color:#1f7a4d;">{accepted_count}</div>
            <div class="live-status-card-copy">Share retained: {accepted_share:.1%}. {_escape_text(accepted_reason)}</div>
        </div>
        <div class="live-status-card rejected">
            <div class="live-status-card-title">Rejected</div>
            <div class="live-status-card-value" style="color:#b91c1c;">{rejected_count}</div>
            <div class="live-status-card-copy">Share filtered out: {rejected_share:.1%}. {_escape_text(rejected_reason)}</div>
        </div>
    </div>
    """


def build_live_trace_3d_figure(
    trace: Sequence[Mapping[str, float | str]],
    *,
    title: str = "Live Solver Trace",
) -> go.Figure:
    """Plot a 3D solver trace from iteration, risk, objective, and return."""

    frame = pd.DataFrame(trace)
    if frame.empty:
        return go.Figure().update_layout(title=title)
    if "stage" not in frame.columns:
        frame["stage"] = [f"Step {int(value)}" for value in frame["iteration"]]
    for column in ("iteration", "risk", "objective_value", "expected_return"):
        if column not in frame.columns:
            frame[column] = 0.0
    frame["iteration"] = frame["iteration"].astype(float)
    frame["risk"] = frame["risk"].astype(float)
    frame["objective_value"] = frame["objective_value"].astype(float)
    frame["expected_return"] = frame["expected_return"].astype(float)

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
                "Stage: %{text}<br>"
                "Iteration: %{x:.0f}<br>"
                "Risk: %{y:.4f}<br>"
                "Objective: %{z:.4f}<br>"
                "Return: %{marker.color:.4f}<extra></extra>"
            ),
            name="Trace",
        )
    )
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        font=dict(color="#111111"),
        scene=dict(
            xaxis_title="Iteration",
            yaxis_title="Risk",
            zaxis_title="Objective",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=45, b=20),
        height=470,
    )
    return figure


def build_live_trace_view(
    trace: Sequence[Mapping[str, float | str]],
    *,
    title: str = "Live Solver Trace",
) -> dict[str, Any]:
    """Return a reusable view bundle containing the 3D figure and a compact trace table."""

    frame = pd.DataFrame(trace)
    if frame.empty:
        return {
            "title": title,
            "figure": build_live_trace_3d_figure(trace, title=title),
            "table": pd.DataFrame(),
            "summary": "No trace points have been produced yet.",
        }

    compact = frame.copy()
    if "iteration" in compact.columns:
        compact["iteration"] = compact["iteration"].astype(float)
    if "risk" in compact.columns:
        compact["risk"] = compact["risk"].astype(float)
    if "objective_value" in compact.columns:
        compact["objective_value"] = compact["objective_value"].astype(float)
    if "expected_return" in compact.columns:
        compact["expected_return"] = compact["expected_return"].astype(float)

    summary = (
        f"Trace points: {len(compact)}. "
        f"Best objective: {compact['objective_value'].max():.4f}." if "objective_value" in compact.columns and not compact.empty else
        f"Trace points: {len(compact)}."
    )
    return {
        "title": title,
        "figure": build_live_trace_3d_figure(trace, title=title),
        "table": compact,
        "summary": summary,
    }


def build_live_trace_container(
    *,
    banner_title: str,
    banner_summary: str,
    banner_stage: str | None = None,
    banner_progress: float | None = None,
) -> str:
    """Return an HTML wrapper for the live optimization header."""

    return build_live_progress_banner(
        title=banner_title,
        summary=banner_summary,
        stage=banner_stage,
        progress=banner_progress,
        notes=["Backend", "Solver", "Trace", "3D"],
    )


def build_stage_table(stages: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a compact stage table for display or debugging."""

    rows: list[dict[str, Any]] = []
    for item in stages:
        rows.append(
            {
                "stage": str(item.get("name", item.get("stage", "stage"))),
                "status": str(item.get("status", "pending")),
                "progress": float(item.get("progress", 0.0)),
                "duration_sec": float(item.get("duration_sec", 0.0)),
                "detail": str(item.get("detail", "")),
            }
        )
    return pd.DataFrame(rows)


def build_live_stage_table(stages: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Backward-compatible alias for the stage table helper."""

    return build_stage_table(stages)


__all__ = [
    "build_live_accept_reject_cards",
    "build_live_progress_banner",
    "build_live_stage_table",
    "build_stage_table",
    "build_stage_timeline_cards",
    "build_live_trace_3d_figure",
    "build_live_trace_container",
    "build_live_trace_view",
    "build_polling_status_cards",
    "build_stage_timeline_figure",
    "LIVE_CONSOLE_CSS",
]
