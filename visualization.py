"""
Publication-quality plotting utilities for the experiment pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "QP": "#1f77b4",
    "PSO": "#ff7f0e",
    "GA": "#2ca02c",
    "Hybrid": "#d62728",
}


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_convergence(convergence_data: dict, output_path: str | Path) -> None:
    set_publication_style()
    baseline = convergence_data["baseline"]

    fig, ax = plt.subplots()
    for method in ["PSO", "GA", "Hybrid"]:
        series = np.asarray(baseline[method]["objective"], dtype=float)
        ax.plot(series, label=method, color=METHOD_COLORS[method])

    ax.set_title("Convergence Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results_df: pd.DataFrame, baseline_lambda: float, output_path: str | Path) -> None:
    set_publication_style()
    baseline = results_df[
        (results_df["analysis_type"] == "baseline")
        & np.isclose(results_df["lambda"], baseline_lambda)
    ].sort_values("method")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("objective", "Objective"),
        ("mc_mean_return", "Monte Carlo Return"),
        ("mc_std_return", "Monte Carlo Risk"),
    ]

    for ax, (column, title) in zip(axes, metrics):
        ax.bar(
            baseline["method"],
            baseline[column],
            color=[METHOD_COLORS[m] for m in baseline["method"]],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_risk_return(results_df: pd.DataFrame, baseline_lambda: float, output_path: str | Path) -> None:
    set_publication_style()
    baseline = results_df[
        (results_df["analysis_type"] == "baseline")
        & np.isclose(results_df["lambda"], baseline_lambda)
    ]

    fig, ax = plt.subplots()
    for _, row in baseline.iterrows():
        ax.scatter(
            row["mc_std_return"],
            row["mc_mean_return"],
            s=90,
            color=METHOD_COLORS[row["method"]],
            label=row["method"],
        )
        ax.annotate(row["method"], (row["mc_std_return"], row["mc_mean_return"]), xytext=(6, 4), textcoords="offset points")

    ax.set_title("Risk-Return Tradeoff")
    ax.set_xlabel("Monte Carlo risk (std dev)")
    ax.set_ylabel("Monte Carlo return")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(results_df: pd.DataFrame, output_path: str | Path) -> None:
    set_publication_style()
    sensitivity = results_df[results_df["analysis_type"] == "sensitivity"].copy()
    sensitivity = sensitivity.sort_values(["method", "lambda"])

    fig, ax = plt.subplots()
    for method, group in sensitivity.groupby("method", observed=True):
        ax.plot(
            group["mc_std_return"],
            group["mc_mean_return"],
            marker="o",
            color=METHOD_COLORS[method],
            label=method,
        )
        for _, row in group.iterrows():
            ax.annotate(f"{row['lambda']:.1f}", (row["mc_std_return"], row["mc_mean_return"]), xytext=(4, 3), textcoords="offset points", fontsize=9)

    ax.set_title("Sensitivity Analysis Across Risk Aversion")
    ax.set_xlabel("Monte Carlo risk (std dev)")
    ax.set_ylabel("Monte Carlo return")
    ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
