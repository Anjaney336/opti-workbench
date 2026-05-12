"""
CLI entrypoint for the multi-model portfolio optimization platform.
"""

from __future__ import annotations

import argparse

from optimizer import PortfolioOptimizer, PortfolioRequest


def run_demo() -> None:
    request = PortfolioRequest(
        asset_count=12,
        active_assets_target=4,
        budget=5.0,
        max_allocation=0.35,
        lambda_value=1.0,
        liquidity_weight=0.35,
        cost_sensitivity=0.15,
        method="AUTO",
        seed=42,
        monte_carlo_paths=500,
        dataset="balanced",
    )
    optimizer = PortfolioOptimizer(request)
    dashboard = optimizer.dashboard_bundle()

    print("=" * 88)
    print("BRIXSQUARE AI INVESTMENT OPTIMIZER")
    print("=" * 88)
    print(f"Selected model : {dashboard['selected_model']}")
    print(f"Assets         : {len(dashboard['assets'])}")
    print(f"Return         : {dashboard['selected_output']['expected_return']:.6f}")
    print(f"Risk           : {dashboard['selected_output']['risk']:.6f}")
    print(f"Sharpe         : {dashboard['selected_output']['sharpe']:.6f}")
    print(f"Diversification: {dashboard['selected_output']['diversification']}")
    print()
    print("Comparison")
    for row in dashboard["comparison_table"]:
        print(
            f"  {row['model']:>6} | return={row['expected_return']:.4f} "
            f"risk={row['risk']:.4f} sharpe={row['sharpe']:.4f} div={row['diversification']}"
        )


def print_instructions() -> None:
    print("Run demo:")
    print("  python main.py --demo")
    print()
    print("Run backend:")
    print("  uvicorn backend.api:app --reload")
    print()
    print("Run frontend:")
    print("  streamlit run frontend/app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="BrixSquare AI Investment Optimizer")
    parser.add_argument("--demo", action="store_true", help="Run the demo bundle.")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print_instructions()


if __name__ == "__main__":
    main()
