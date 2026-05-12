"""
Reusable visibility helpers for solver transparency panels.

This module is intentionally helper-only:
- HTML snippets for high-contrast dashboard sections
- Plotly figures for accepted vs rejected candidates and variable ranges
- DataFrame builders and stylers for decision-variable tables

No Streamlit page wiring lives here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.io.formats.style import Styler


VISIBILITY_CSS = """
<style>
.visibility-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1f2937 48%, #334155 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 16px 30px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.visibility-hero-title {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 0.25rem;
}

.visibility-hero-copy {
    font-size: 0.92rem;
    line-height: 1.45;
    color: rgba(255, 255, 255, 0.92);
}

.visibility-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.8rem;
}

.visibility-card {
    background: #ffffff;
    color: #111111;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    padding: 0.95rem 1rem;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
}

.visibility-card.accepted {
    border-left: 4px solid #1f7a4d;
}

.visibility-card.rejected {
    border-left: 4px solid #b91c1c;
}

.visibility-card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
    color: #475569;
    margin-bottom: 0.35rem;
}

.visibility-card-value {
    font-size: 2rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 0.22rem;
}

.visibility-card-copy {
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.45;
}

.visibility-pill {
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

.visibility-muted {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.5;
}
</style>
"""


def build_solver_visibility_banner(
    *,
    title: str,
    summary: str,
    steps: Sequence[str] | None = None,
    subtitle: str | None = None,
) -> str:
    step_markup = ""
    if steps:
        step_markup = "".join(f'<span class="visibility-pill">{step}</span>' for step in steps)
    subtitle_markup = f'<div class="visibility-hero-copy" style="margin-bottom:0.35rem;font-weight:700;">{subtitle}</div>' if subtitle else ""
    return (
        '<div class="visibility-hero">'
        f'<div class="visibility-hero-title">{title}</div>'
        f"{subtitle_markup}"
        f'<div class="visibility-hero-copy">{summary}</div>'
        f'<div style="margin-top:0.65rem;">{step_markup}</div>'
        "</div>"
    )


def build_solver_accept_reject_cards(
    *,
    accepted_count: int,
    rejected_count: int,
    accepted_reason: str,
    rejected_reason: str,
    accepted_label: str = "Accepted",
    rejected_label: str = "Rejected",
) -> str:
    total = max(accepted_count + rejected_count, 1)
    accepted_share = accepted_count / total
    rejected_share = rejected_count / total
    return f"""
    <div class="visibility-grid">
        <div class="visibility-card accepted">
            <div class="visibility-card-title">{accepted_label}</div>
            <div class="visibility-card-value" style="color:#1f7a4d;">{accepted_count}</div>
            <div class="visibility-card-copy">Share of candidates retained: {accepted_share:.1%}. {accepted_reason}</div>
        </div>
        <div class="visibility-card rejected">
            <div class="visibility-card-title">{rejected_label}</div>
            <div class="visibility-card-value" style="color:#b91c1c;">{rejected_count}</div>
            <div class="visibility-card-copy">Share of candidates filtered out: {rejected_share:.1%}. {rejected_reason}</div>
        </div>
    </div>
    """


def build_solver_step_html(
    *,
    title: str,
    body: str,
    accent: str = "#0f172a",
) -> str:
    return (
        '<div class="visibility-card" style="border-left:4px solid '
        f"{accent};\">"
        f'<div class="visibility-card-title">{title}</div>'
        f'<div class="visibility-card-copy">{body}</div>'
        "</div>"
    )


def build_candidate_range_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Build a table of candidate ranges from decision-variable records.

    Expected keys:
    - name
    - selected_weight
    - decision_score
    - mu_norm
    - risk_norm
    - liquidity_norm
    - cost_norm
    - rank
    """

    rows: list[dict[str, Any]] = []
    for row in records:
        weight = float(row.get("selected_weight", 0.0))
        rows.append(
            {
                "Asset": str(row.get("name", "Asset")),
                "Weight": weight,
                "Rank": int(row.get("rank", 0)),
                "Decision Score": float(row.get("decision_score", 0.0)),
                "Mu": float(row.get("mu", 0.0)),
                "Sigma": float(row.get("sigma", 0.0)),
                "Liquidity": float(row.get("liquidity", 0.0)),
                "Cost": float(row.get("cost", 0.0)),
                "Accepted": weight > 1e-6,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Accepted", "Weight", "Decision Score"], ascending=[False, False, False]).reset_index(drop=True)
    return frame


def stylize_candidate_range_dataframe(frame: pd.DataFrame) -> Styler:
    if frame.empty:
        return frame.style
    return frame.style.format(
        {
            "Weight": "{:.2%}",
            "Decision Score": "{:.4f}",
            "Mu": "{:.4f}",
            "Sigma": "{:.4f}",
            "Liquidity": "{:.4f}",
            "Cost": "{:.4f}",
        }
    ).apply(
        lambda col: ["background-color: #ecfdf5" if bool(v) else "" for v in col]
        if col.name == "Accepted"
        else ["" for _ in col],
        axis=0,
    )


def build_candidate_range_figure(
    records: Sequence[Mapping[str, Any]],
    *,
    title: str = "Decision Variable Ranges",
) -> go.Figure:
    frame = build_candidate_range_dataframe(records)
    if frame.empty:
        return go.Figure().update_layout(title=title)

    figure = go.Figure()
    accepted = frame[frame["Accepted"]]
    rejected = frame[~frame["Accepted"]]

    figure.add_trace(
        go.Bar(
            x=accepted["Asset"],
            y=accepted["Weight"],
            name="Accepted",
            marker_color="#1f7a4d",
            hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
        )
    )
    if not rejected.empty:
        figure.add_trace(
            go.Bar(
                x=rejected["Asset"],
                y=rejected["Weight"],
                name="Rejected",
                marker_color="#b91c1c",
                hovertemplate="%{x}<br>Weight: %{y:.2%}<extra></extra>",
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
        yaxis=dict(title="Selected Weight", tickformat=".0%", gridcolor="rgba(15,23,42,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=380,
    )
    return figure


def build_rejection_summary(
    *,
    accepted_count: int,
    rejected_count: int,
    accepted_reasons: Sequence[str],
    rejected_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "accepted_count": int(accepted_count),
        "rejected_count": int(rejected_count),
        "accepted_reasons": list(accepted_reasons),
        "rejected_reasons": list(rejected_reasons),
        "summary": (
            f"{accepted_count} candidate positions were retained and {rejected_count} were rejected "
            "after applying the diversification and feasibility rules."
        ),
    }


__all__ = [
    "VISIBILITY_CSS",
    "build_solver_visibility_banner",
    "build_solver_accept_reject_cards",
    "build_solver_step_html",
    "build_candidate_range_dataframe",
    "stylize_candidate_range_dataframe",
    "build_candidate_range_figure",
    "build_rejection_summary",
]
