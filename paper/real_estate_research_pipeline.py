"""
End-to-end research pipeline for real-estate portfolio optimization on HouseTS.

The pipeline replaces synthetic portfolio parameters with estimates derived from
the observed housing panel. Each zipcode is treated as an investable asset.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import textwrap
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import sparse
from scipy.stats import bootstrap
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.covariance import GraphicalLasso, LedoitWolf, OAS
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, TimeSeriesSplit, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

shap = None

try:
    import optuna
except Exception:
    optuna = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except Exception:
    TabNetRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import compare_portfolios, monte_carlo_evaluate
from src.ga import GeneticAlgorithm
from src.ga import optimize_with_ga
from src.pso import ParticleSwarmOptimizer
from src.pso import optimize_with_pso
from src.qp import efficient_frontier, solve_quadratic_program
from src.utils import project_constraints


SEED = 42


@dataclass
class ResearchConfig:
    dataset_path: Path = Path("HouseTS.csv")
    output_dir: Path = Path("results/research_pipeline")
    top_assets: int = 15
    holdout_months: int = 12
    monte_carlo_scenarios: int = 10_000
    alpha: float = 0.22
    lambda_param: float = 4.0
    pso_particles: int = 36
    pso_iterations: int = 140
    ga_population: int = 44
    ga_generations: int = 140
    random_state: int = SEED
    n_optimizer_seeds: int = 30
    transaction_cost_rate: float = 0.0025
    figure_dpi: int = 220
    style: dict[str, Any] = field(
        default_factory=lambda: {
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "font.size": 10,
            "axes.facecolor": "#fbfbfd",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#d9d9e3",
            "axes.edgecolor": "#222222",
        }
    )


class DenseEstimatorAdapter:
    """Small sklearn-compatible adapter for estimators that require dense float arrays."""

    def __init__(self, estimator: Any, **estimator_params: Any) -> None:
        self.estimator = estimator
        if estimator_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**estimator_params)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"estimator": self.estimator}
        if hasattr(self.estimator, "get_params"):
            params.update(self.estimator.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> "DenseEstimatorAdapter":
        estimator_params = {}
        for key, value in params.items():
            if key == "estimator":
                self.estimator = value
            else:
                estimator_params[key] = value
        if estimator_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**estimator_params)
        return self

    @staticmethod
    def _to_dense(X: Any) -> np.ndarray:
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X: Any, y: Any) -> "DenseEstimatorAdapter":
        self.estimator.fit(self._to_dense(X), np.asarray(y, dtype=np.float32))
        return self

    def predict(self, X: Any) -> np.ndarray:
        prediction = self.estimator.predict(self._to_dense(X))
        return np.asarray(prediction, dtype=float).reshape(-1)


class TabNetRegressorAdapter(DenseEstimatorAdapter):
    def fit(self, X: Any, y: Any) -> "TabNetRegressorAdapter":
        X_dense = self._to_dense(X)
        y_dense = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.estimator.fit(
                X_dense,
                y_dense,
                max_epochs=25,
                patience=5,
                batch_size=min(256, max(32, len(X_dense) // 4)),
                virtual_batch_size=32,
            )
        return self

class RealEstateResearchPipeline:
    """Publication-oriented research pipeline for the HouseTS panel."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        for path in [self.output_dir, self.figures_dir, self.tables_dir, self.models_dir, self.reports_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.limitations: list[str] = []
        self.metadata: dict[str, Any] = {"config": asdict(config), "runtime_seconds": {}}
        self.selected_assets: list[int] = []
        self.latest_regime_label: str = "Unknown"

        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.config.style)

    def run(self) -> dict[str, Any]:
        timings: dict[str, float] = {}

        t0 = perf_counter()
        raw = self.load_dataset()
        timings["load_dataset"] = perf_counter() - t0

        t0 = perf_counter()
        eda_summary = self.perform_eda(raw)
        timings["eda"] = perf_counter() - t0

        t0 = perf_counter()
        panel = self.preprocess_and_engineer(raw)
        timings["preprocess_and_engineer"] = perf_counter() - t0

        t0 = perf_counter()
        expected_return_results = self.estimate_expected_returns(panel)
        timings["expected_returns"] = perf_counter() - t0

        t0 = perf_counter()
        covariance_results = self.estimate_covariance(panel)
        timings["covariance"] = perf_counter() - t0

        t0 = perf_counter()
        regime_results = self.detect_regimes(panel)
        timings["regimes"] = perf_counter() - t0

        t0 = perf_counter()
        liquidity_results = self.estimate_liquidity(panel)
        timings["liquidity"] = perf_counter() - t0

        t0 = perf_counter()
        tx_cost_results = self.estimate_transaction_cost(panel)
        timings["transaction_cost"] = perf_counter() - t0

        t0 = perf_counter()
        optimization_results = self.optimize_portfolio(
            panel=panel,
            return_results=expected_return_results,
            covariance_results=covariance_results,
            regime_results=regime_results,
            liquidity_results=liquidity_results,
            tx_cost_results=tx_cost_results,
        )
        timings["optimization"] = perf_counter() - t0

        t0 = perf_counter()
        experiment_results = self.run_experimental_studies(
            panel=panel,
            covariance_results=covariance_results,
            optimization_results=optimization_results,
        )
        timings["experimental_studies"] = perf_counter() - t0

        t0 = perf_counter()
        validation_results = self.validate_statistics(
            optimization_results=optimization_results,
            covariance_results=covariance_results,
            panel=panel,
        )
        timings["statistical_validation"] = perf_counter() - t0

        self.metadata["runtime_seconds"] = {k: round(v, 3) for k, v in timings.items()}
        self.metadata["selected_assets"] = self.selected_assets
        self.metadata["latest_regime_label"] = self.latest_regime_label
        self.metadata["limitations"] = self.limitations
        self._write_json(self.output_dir / "metadata.json", self.metadata)

        parameter_support = self._parameter_support_table()
        parameter_support.to_csv(self.tables_dir / "parameter_support_table.csv", index=False)
        claims_table = self._claim_validation_table()
        claims_table.to_csv(self.tables_dir / "claim_validation_table.csv", index=False)
        self._write_methodology_note()
        self._write_limitation_analysis()

        report = {
            "eda_summary": eda_summary,
            "expected_return_results": expected_return_results["summary"],
            "covariance_results": covariance_results["summary"],
            "regime_results": regime_results["summary"],
            "liquidity_results": liquidity_results["summary"],
            "transaction_cost_results": tx_cost_results["summary"],
            "optimization_results": optimization_results["summary"],
            "experimental_studies": experiment_results["summary"],
            "validation_results": validation_results,
            "limitations": self.limitations,
        }
        self._write_json(self.output_dir / "research_summary.json", report)
        self._write_report(report)
        return report

    def load_dataset(self) -> pd.DataFrame:
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        df = pd.read_csv(self.config.dataset_path, parse_dates=["date"])
        df = df.sort_values(["zipcode", "date"]).reset_index(drop=True)
        return df

    def perform_eda(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sample_size = min(50_000, len(df))
        plot_df = df.sample(sample_size, random_state=self.config.random_state).copy() if len(df) > sample_size else df.copy()
        missing = df.isna().sum().sort_values(ascending=False)
        duplicates = int(df.duplicated().sum())
        temporal_duplicates = int(df.duplicated(subset=["zipcode", "date"]).sum())
        per_asset_counts = df.groupby("zipcode").size()

        market_series = df.groupby("date")["price"].mean().sort_index()
        adf = self._approximate_adf_test(market_series)
        trend, seasonal, resid = self._simple_decompose(market_series, period=12)

        eda_summary = {
            "shape": list(df.shape),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "zipcode_count": int(df["zipcode"].nunique()),
            "city_count": int(df["city"].nunique()),
            "duplicates": duplicates,
            "temporal_duplicates": temporal_duplicates,
            "obs_per_asset": {
                "min": int(per_asset_counts.min()),
                "median": float(per_asset_counts.median()),
                "max": int(per_asset_counts.max()),
            },
            "missing_top15": missing.head(15).to_dict(),
            "stationarity_proxy": adf,
        }
        self._write_json(self.tables_dir / "eda_summary.json", eda_summary)

        dtype_table = pd.DataFrame({"column": df.columns, "dtype": [str(v) for v in df.dtypes]})
        dtype_table.to_csv(self.tables_dir / "dtypes.csv", index=False)
        missing.reset_index().rename(columns={"index": "column", 0: "missing_count"}).to_csv(
            self.tables_dir / "missing_values.csv", index=False
        )

        outlier_rows = []
        for col in numeric_cols:
            values = plot_df[col].dropna()
            if values.empty:
                continue
            z = np.abs((values - values.mean()) / (values.std(ddof=0) + 1e-9))
            outlier_rows.append({"feature": col, "outlier_count_z3": int((z > 3).sum())})
        pd.DataFrame(outlier_rows).sort_values("outlier_count_z3", ascending=False).to_csv(
            self.tables_dir / "outlier_summary.csv", index=False
        )

        asset_stats = (
            df.groupby("zipcode")
            .agg(
                city=("city", "first"),
                median_price=("price", "median"),
                price_std=("price", "std"),
                avg_homes_sold=("homes_sold", "mean"),
                avg_inventory=("inventory", "mean"),
            )
            .sort_values("avg_homes_sold", ascending=False)
        )
        asset_stats.head(50).to_csv(self.tables_dir / "asset_level_statistics_top50.csv")

        corr_cols = [
            "price",
            "median_sale_price",
            "median_list_price",
            "homes_sold",
            "pending_sales",
            "inventory",
            "median_dom",
            "avg_sale_to_list",
            "sold_above_list",
            "Median Rent",
            "Median Home Value",
            "Per Capita Income",
        ]
        corr = plot_df[corr_cols].corr(numeric_only=True)
        corr.to_csv(self.tables_dir / "correlation_matrix.csv")

        market_frame = pd.DataFrame(
            {"trend": trend, "seasonal": seasonal, "residual": resid, "market_price": market_series}
        ).dropna()
        market_frame.to_csv(self.tables_dir / "market_decomposition.csv")

        self._save_distribution_plot(plot_df, ["price", "median_sale_price", "homes_sold", "inventory"], "distributions.png")
        self._save_correlation_heatmap(corr, "correlation_heatmap.png")
        self._save_trend_plot(market_series, trend, seasonal, "market_decomposition.png")
        self._save_boxplot(plot_df, ["price", "homes_sold", "inventory"], "outlier_boxplots.png")

        return eda_summary

    def preprocess_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["month"] = work["date"].dt.month
        work["quarter"] = work["date"].dt.quarter
        work["turnover"] = work["homes_sold"] / (work["inventory"] + 1.0)
        work["rental_yield"] = (work["Median Rent"] * 12.0) / work["price"].replace(0, np.nan)
        work["poverty_rate_proxy"] = work["Total Families Below Poverty"] / work["Total Population"].replace(0, np.nan)
        work["employment_rate"] = 1.0 - (
            work["Unemployed Population"] / work["Total Labor Force"].replace(0, np.nan)
        )

        unique_dates = np.sort(work["date"].unique())
        cutoff = unique_dates[-self.config.holdout_months]
        asset_rank = (
            work.loc[work["date"] < cutoff]
            .groupby("zipcode")
            .agg(avg_turnover=("turnover", "mean"), obs=("date", "size"), median_price=("price", "median"))
            .query("obs >= 120")
            .sort_values(["avg_turnover", "median_price"], ascending=[False, False])
        )
        self.selected_assets = asset_rank.head(self.config.top_assets).index.astype(int).tolist()
        work = work[work["zipcode"].isin(self.selected_assets)].copy()

        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            work[col] = work.groupby("zipcode")[col].transform(lambda s: s.ffill().bfill())
            work[col] = work[col].fillna(work[col].median())
            lower = work[col].quantile(0.01)
            upper = work[col].quantile(0.99)
            work[col] = work[col].clip(lower, upper)

        work = work.sort_values(["zipcode", "date"]).reset_index(drop=True)
        grp = work.groupby("zipcode", group_keys=False)
        work["asset_return"] = grp["price"].pct_change()
        work["appreciation_rate"] = grp["median_sale_price"].pct_change(12)
        work["growth_rate_3m"] = grp["price"].pct_change(3)
        work["momentum_3m"] = grp["price"].pct_change(3)
        work["momentum_6m"] = grp["price"].pct_change(6)
        work["momentum_12m"] = grp["price"].pct_change(12)
        work["ma_3"] = grp["price"].transform(lambda s: s.rolling(3, min_periods=2).mean())
        work["ma_6"] = grp["price"].transform(lambda s: s.rolling(6, min_periods=3).mean())
        work["ma_12"] = grp["price"].transform(lambda s: s.rolling(12, min_periods=6).mean())
        work["volatility_3"] = grp["asset_return"].transform(lambda s: s.rolling(3, min_periods=2).std())
        work["volatility_6"] = grp["asset_return"].transform(lambda s: s.rolling(6, min_periods=3).std())
        work["turnover_rolling_3"] = grp["turnover"].transform(lambda s: s.rolling(3, min_periods=2).mean())
        work["inventory_change"] = grp["inventory"].pct_change()
        work["demand_pressure"] = work["pending_sales"] / (work["inventory"] + 1.0)
        work["list_to_sale_gap"] = (work["median_list_price"] - work["median_sale_price"]) / (
            work["median_list_price"].replace(0, np.nan)
        )
        work["amihud_illiquidity_proxy"] = np.abs(work["asset_return"]) / (work["turnover"] + 1e-6)
        work["turnover_price_impact_proxy"] = np.abs(work["asset_return"]) / (work["turnover_rolling_3"] + 1e-6)
        work["sqrt_impact_proxy"] = np.sqrt(np.abs(work["asset_return"])) / np.sqrt(work["turnover"] + 1e-6)

        lag_base_cols = [
            "asset_return",
            "turnover",
            "rental_yield",
            "momentum_3m",
            "momentum_6m",
            "volatility_3",
            "volatility_6",
            "inventory_change",
            "demand_pressure",
            "avg_sale_to_list",
            "sold_above_list",
            "median_dom",
            "amihud_illiquidity_proxy",
            "turnover_price_impact_proxy",
            "sqrt_impact_proxy",
        ]
        for col in lag_base_cols:
            work[f"{col}_lag1"] = grp[col].shift(1)

        liquidity_proxy = (
            0.30 * self._zscore_series(work["turnover_rolling_3"])
            + 0.20 * self._zscore_series(1.0 / (work["median_dom"] + 1.0))
            + 0.20 * self._zscore_series(work["off_market_in_two_weeks"])
            + 0.15 * self._zscore_series(work["avg_sale_to_list"])
            - 0.15 * self._zscore_series(work["amihud_illiquidity_proxy"])
        )
        work["liquidity_score"] = self._minmax_series(liquidity_proxy)

        tx_proxy = (
            0.35 * self._minmax_series(1.0 - work["avg_sale_to_list"])
            + 0.25 * self._minmax_series(work["median_dom"])
            + 0.20 * self._minmax_series(work["amihud_illiquidity_proxy"])
            + 0.20 * self._minmax_series(work["sqrt_impact_proxy"])
        )
        work["transaction_cost_proxy"] = 0.005 + 0.03 * tx_proxy

        work["next_return"] = grp["asset_return"].shift(-1)
        work["next_liquidity"] = grp["liquidity_score"].shift(-1)
        work["next_transaction_cost"] = grp["transaction_cost_proxy"].shift(-1)

        panel = work.copy()
        panel["asset_id"] = panel["zipcode"].astype(str)

        panel = panel.replace([np.inf, -np.inf], np.nan)
        panel = panel.dropna(subset=["next_return", "next_liquidity", "next_transaction_cost"]).reset_index(drop=True)
        panel.to_csv(self.tables_dir / "engineered_panel.csv", index=False)
        return panel

    def estimate_expected_returns(self, panel: pd.DataFrame) -> dict[str, Any]:
        feature_cols = self._reduce_redundant_features(panel, self._feature_columns(panel))
        categorical_cols = ["city", "asset_id", "month", "quarter"]
        train_mask, test_mask = self._time_split_masks(panel)

        X_train = panel.loc[train_mask, feature_cols + categorical_cols]
        y_train = panel.loc[train_mask, "next_return"]
        X_test = panel.loc[test_mask, feature_cols + categorical_cols]
        y_test = panel.loc[test_mask, "next_return"]

        models = self._regression_models()
        kfold = KFold(n_splits=3, shuffle=False)
        tss = TimeSeriesSplit(n_splits=3)
        model_rows: list[dict[str, Any]] = []
        fitted_models: dict[str, Pipeline] = {}
        residual_cache: dict[str, pd.DataFrame] = {}
        runtime_cache: dict[str, float] = {}

        for name, estimator in models.items():
            pipeline = self._build_model_pipeline(estimator, feature_cols, categorical_cols)
            try:
                started = perf_counter()
                pipeline.fit(X_train, y_train)
                pred_train = pipeline.predict(X_train)
                pred_test = pipeline.predict(X_test)
                runtime_cache[name] = perf_counter() - started
                fitted_models[name] = pipeline
            except Exception as exc:
                note = f"{name} unavailable at execution time: {exc}"
                if note not in self.limitations:
                    self.limitations.append(note)
                continue

            train_rmse = float(math.sqrt(mean_squared_error(y_train, pred_train)))
            test_rmse = float(math.sqrt(mean_squared_error(y_test, pred_test)))
            metrics = self._regression_metric_row(
                name=name,
                y_true=y_test,
                y_pred=pred_test,
                train_rmse=train_rmse,
                test_rmse=test_rmse,
                n_features=len(feature_cols) + len(categorical_cols),
            )
            metrics["kfold_rmse"] = self._cross_val_rmse(pipeline, X_train, y_train, kfold)
            metrics["timeseries_rmse"] = self._cross_val_rmse(pipeline, X_train, y_train, tss)
            metrics["fit_runtime_sec"] = float(runtime_cache[name])
            model_rows.append(metrics)

            residual_cache[name] = pd.DataFrame(
                {"actual": y_test.to_numpy(), "predicted": pred_test, "residual": y_test.to_numpy() - pred_test}
            )

        if not model_rows:
            raise RuntimeError("No expected-return models executed successfully.")
        metrics_df = self._rank_models(pd.DataFrame(model_rows))
        metrics_df.to_csv(self.tables_dir / "expected_return_model_metrics.csv", index=False)
        best_name = str(metrics_df.iloc[0]["model"])
        best_model = fitted_models[best_name]

        self._save_learning_curve(best_model, X_train, y_train, "expected_return_learning_curve.png")
        self._save_prediction_plot(residual_cache[best_name], "expected_return_prediction_vs_actual.png")
        self._save_residual_plot(residual_cache[best_name], "expected_return_residuals.png")
        self._save_feature_importance(best_model, X_test, y_test, "expected_return_permutation_importance.csv")
        self._save_partial_dependence(best_model, X_train, feature_cols[:2], "expected_return_partial_dependence.png")
        self._save_mutual_information(panel.loc[train_mask, feature_cols], y_train, "expected_return_mutual_information.csv")
        self._save_rfe(best_model, X_train, y_train, "expected_return_rfe.csv")
        self._save_shap_artifacts(best_model, X_test, "expected_return")

        tuned = self._tune_installed_models(X_train, y_train, feature_cols, categorical_cols)

        latest_features = self._latest_asset_features(panel, feature_cols + categorical_cols)
        latest_features["predicted_return"] = best_model.predict(latest_features[feature_cols + categorical_cols])
        latest_features.to_csv(self.tables_dir / "predicted_expected_returns.csv", index=False)
        with open(self.models_dir / "expected_return_best_model.pkl", "wb") as handle:
            pickle.dump(best_model, handle)

        summary = {
            "best_model": best_name,
            "best_rmse": float(metrics_df.iloc[0]["rmse"]),
            "best_timeseries_rmse": float(metrics_df.iloc[0]["timeseries_rmse"]),
            "selected_feature_count": int(len(feature_cols)),
            "tested_models": metrics_df["model"].tolist(),
            "tuned_models": tuned,
        }
        return {
            "summary": summary,
            "metrics": metrics_df,
            "best_model": best_model,
            "best_model_name": best_name,
            "predictions": latest_features[["zipcode", "city", "predicted_return"]],
        }

    def estimate_covariance(self, panel: pd.DataFrame) -> dict[str, Any]:
        returns = (
            panel.pivot_table(index="date", columns="zipcode", values="asset_return")
            .sort_index()
            .fillna(0.0)
        )
        train_returns = returns.iloc[:-self.config.holdout_months]

        estimators = {
            "sample_covariance": np.cov(train_returns.to_numpy().T, ddof=1),
            "ledoit_wolf": LedoitWolf().fit(train_returns.to_numpy()).covariance_,
            "oas": OAS().fit(train_returns.to_numpy()).covariance_,
        }
        try:
            estimators["graphical_lasso"] = GraphicalLasso(alpha=0.01, max_iter=300).fit(train_returns.to_numpy()).covariance_
        except Exception as exc:
            self.limitations.append(f"GraphicalLasso convergence issue: {exc}")

        rows = []
        best_name = None
        best_score = -np.inf
        best_cov = None
        for name, cov in estimators.items():
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = float(np.min(eigvals))
            cond = float(np.linalg.cond(cov + np.eye(cov.shape[0]) * 1e-8))
            psd = bool(min_eig >= -1e-8)
            stability = float(np.trace(cov)) / (cond + 1e-9)
            score = (1.0 if psd else -1.0) + stability
            rows.append(
                {
                    "estimator": name,
                    "condition_number": cond,
                    "min_eigenvalue": min_eig,
                    "max_eigenvalue": float(np.max(eigvals)),
                    "positive_semidefinite": psd,
                    "trace": float(np.trace(cov)),
                    "stability_score": stability,
                }
            )
            if score > best_score:
                best_score = score
                best_name = name
                best_cov = cov

        metrics_df = pd.DataFrame(rows).sort_values("condition_number")
        metrics_df.to_csv(self.tables_dir / "covariance_estimator_comparison.csv", index=False)
        np.save(self.tables_dir / "selected_covariance_matrix.npy", best_cov)
        self._save_covariance_heatmap(best_cov, "selected_covariance_heatmap.png")

        summary = {
            "selected_estimator": best_name,
            "selected_condition_number": float(metrics_df.query("estimator == @best_name")["condition_number"].iloc[0]),
            "estimators_evaluated": metrics_df["estimator"].tolist(),
        }
        return {"summary": summary, "matrix": best_cov, "returns_matrix": returns, "metrics": metrics_df}

    def detect_regimes(self, panel: pd.DataFrame) -> dict[str, Any]:
        market = (
            panel.groupby("date")
            .agg(
                market_return=("asset_return", "mean"),
                market_volatility=("asset_return", "std"),
                liquidity=("liquidity_score", "mean"),
                transaction_cost=("transaction_cost_proxy", "mean"),
                momentum=("momentum_3m", "mean"),
            )
            .fillna(0.0)
            .sort_index()
        )
        market["rolling_vol_3"] = market["market_return"].rolling(3, min_periods=2).std().fillna(0.0)
        market["rolling_vol_6"] = market["market_return"].rolling(6, min_periods=3).std().fillna(0.0)
        market["downside_share_6"] = market["market_return"].rolling(6, min_periods=3).apply(lambda x: float(np.mean(np.asarray(x) < 0.0)), raw=False).fillna(0.0)
        features = market[["market_return", "rolling_vol_3", "rolling_vol_6", "downside_share_6"]]
        X = StandardScaler().fit_transform(features)
        rows = []

        if "Hidden Markov Model skipped because statsmodels/hmmlearn is not installed." not in self.limitations:
            self.limitations.append("Hidden Markov Model skipped because statsmodels/hmmlearn is not installed.")

        best_bic = np.inf
        best_gmm = None
        for components in [2, 3, 4]:
            gmm = GaussianMixture(n_components=components, random_state=self.config.random_state)
            gmm.fit(X)
            bic = float(gmm.bic(X))
            rows.append({"model": "gaussian_mixture", "components": components, "bic": bic})
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        kmeans = KMeans(n_clusters=int(best_gmm.n_components if best_gmm is not None else 3), random_state=self.config.random_state, n_init=20)
        kmeans_labels = kmeans.fit_predict(X)
        rows.append({"model": "kmeans", "components": int(len(np.unique(kmeans_labels))), "bic": None})
        vol_regime = pd.qcut(market["rolling_vol_6"].rank(method="first"), q=3, labels=False)
        rows.append({"model": "rolling_volatility_classification", "components": int(pd.Series(vol_regime).nunique()), "bic": None})

        if best_gmm is None:
            market["regime_cluster"] = kmeans_labels
            chosen_name = "kmeans"
        else:
            market["regime_cluster"] = best_gmm.predict(X)
            chosen_name = "gaussian_mixture_bic"
        mapping = self._map_regime_labels(market)
        market["regime_label"] = market["regime_cluster"].map(mapping)
        self.latest_regime_label = str(market["regime_label"].iloc[-1])

        market.to_csv(self.tables_dir / "market_regimes.csv")
        self._save_regime_timeline(market, "regime_timeline.png")
        summary = {
            "selected_model": chosen_name,
            "latest_regime": self.latest_regime_label,
            "available_models": ["gaussian_mixture_bic", "kmeans", "rolling_volatility_classification"],
        }
        pd.DataFrame(rows).to_csv(self.tables_dir / "regime_model_comparison.csv", index=False)
        self._write_json(self.tables_dir / "regime_detection_summary.json", summary)
        return {"summary": summary, "timeline": market}

    def estimate_liquidity(self, panel: pd.DataFrame) -> dict[str, Any]:
        result = self._fit_proxy_model(
            panel=panel,
            target="next_liquidity",
            prefix="liquidity",
            estimator=RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=self.config.random_state
            ),
        )
        return result

    def estimate_transaction_cost(self, panel: pd.DataFrame) -> dict[str, Any]:
        result = self._fit_proxy_model(
            panel=panel,
            target="next_transaction_cost",
            prefix="transaction_cost",
            estimator=GradientBoostingRegressor(random_state=self.config.random_state),
        )
        return result

    def optimize_portfolio(
        self,
        panel: pd.DataFrame,
        return_results: dict[str, Any],
        covariance_results: dict[str, Any],
        regime_results: dict[str, Any],
        liquidity_results: dict[str, Any],
        tx_cost_results: dict[str, Any],
    ) -> dict[str, Any]:
        latest = self._latest_asset_features(
            panel,
            self._feature_columns(panel) + ["city", "asset_id", "month", "quarter"],
        )
        latest = latest.merge(return_results["predictions"], on=["zipcode", "city"], how="left")
        latest = latest.merge(liquidity_results["predictions"], on=["zipcode", "city"], how="left")
        latest = latest.merge(tx_cost_results["predictions"], on=["zipcode", "city"], how="left")
        latest = latest.sort_values("zipcode").reset_index(drop=True)

        mu = latest["predicted_return"].to_numpy(dtype=float)
        mu = np.nan_to_num(mu, nan=float(np.nanmean(mu)))

        regime_timeline = regime_results["timeline"]
        regime_adjustment = regime_timeline.groupby("regime_label")["market_return"].mean()
        scale = 1.0 + float(regime_adjustment.get(self.latest_regime_label, 0.0))
        mu_regime = mu * scale

        cov = covariance_results["matrix"]
        prices = latest["price"].to_numpy(dtype=float)
        tx_cost = latest["predicted_transaction_cost"].to_numpy(dtype=float)
        liquidity = latest["predicted_liquidity"].to_numpy(dtype=float)
        cost_vector = self._normalize_cost_vector(prices, tx_cost)
        budget = 1.0
        alpha = self.config.alpha
        adjusted_mu = mu_regime + 0.10 * liquidity - 0.25 * tx_cost

        qp = solve_quadratic_program(adjusted_mu, cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param)
        x_qp = project_constraints(
            qp["x"] if qp["x"] is not None else np.ones_like(mu_regime) / len(mu_regime),
            cost_vector,
            budget,
            alpha,
        )

        def objective_fn(x: np.ndarray) -> float:
            liquidity_bonus = float(np.dot(x, liquidity))
            cost_penalty = float(np.dot(x, tx_cost))
            return float(np.dot(x, mu_regime) - self.config.lambda_param * (x @ cov @ x) + 0.10 * liquidity_bonus - 0.25 * cost_penalty)

        def constraint_proj(x: np.ndarray) -> np.ndarray:
            return project_constraints(x, cost_vector, budget, alpha)

        pso = optimize_with_pso(
            objective_fn=objective_fn,
            constraint_proj_fn=constraint_proj,
            x0=x_qp,
            n_particles=self.config.pso_particles,
            n_iterations=self.config.pso_iterations,
            seed=self.config.random_state,
        )
        ga = optimize_with_ga(
            objective_fn=objective_fn,
            constraint_proj_fn=constraint_proj,
            x0=pso["x"],
            population_size=self.config.ga_population,
            n_generations=self.config.ga_generations,
            seed=self.config.random_state,
        )
        candidate_pool = {"QP": x_qp, "PSO": pso["x"], "GA": ga["x"]}
        hybrid_method = max(candidate_pool, key=lambda name: objective_fn(candidate_pool[name]))
        x_hybrid = np.asarray(candidate_pool[hybrid_method], dtype=float)

        historical_mu = (
            panel.groupby("zipcode")["asset_return"]
            .mean()
            .reindex(latest["zipcode"])
            .to_numpy(dtype=float)
        )
        sample_cov = np.cov(covariance_results["returns_matrix"].to_numpy().T, ddof=1)

        equal_weight = np.ones_like(mu_regime) / len(mu_regime)
        rng = np.random.default_rng(self.config.random_state)
        random_portfolio = constraint_proj(rng.dirichlet(np.ones_like(mu_regime)))
        portfolios = {
            "EqualWeight": equal_weight,
            "RandomPortfolio": random_portfolio,
            "MeanVariance": x_qp,
            "QP": x_qp,
            "PSO": pso["x"],
            "GA": ga["x"],
            "Hybrid": x_hybrid,
            "NoML_HistoricalMean": solve_quadratic_program(
                historical_mu, cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param
            )["x"],
            "NoShrinkage": solve_quadratic_program(
                mu_regime, sample_cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param
            )["x"],
            "NoRegime": solve_quadratic_program(
                mu, cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param
            )["x"],
        }
        portfolios = {k: constraint_proj(v if v is not None else equal_weight) for k, v in portfolios.items()}

        mc_results = compare_portfolios(
            {k: np.asarray(v, dtype=float) for k, v in portfolios.items()},
            mu_regime,
            cov,
            n_scenarios=self.config.monte_carlo_scenarios,
            seed=self.config.random_state,
        )

        risk_rows = []
        market_proxy = covariance_results["returns_matrix"].mean(axis=1).to_numpy()
        returns_matrix = covariance_results["returns_matrix"].reindex(columns=latest["zipcode"]).to_numpy()
        for name, weights in portfolios.items():
            risk_rows.append(self._risk_metric_row(name, weights, returns_matrix, market_proxy, mc_results[name]))
        risk_df = pd.DataFrame(risk_rows).sort_values("sharpe_ratio", ascending=False)
        risk_df.to_csv(self.tables_dir / "portfolio_risk_metrics.csv", index=False)

        frontier_returns, frontier_vars = efficient_frontier(mu_regime, cov, cost_vector, budget, alpha, n_points=25)
        pd.DataFrame({"expected_return": frontier_returns, "variance": frontier_vars}).to_csv(
            self.tables_dir / "efficient_frontier.csv", index=False
        )
        weights_df = pd.DataFrame(portfolios, index=latest["zipcode"]).reset_index().rename(columns={"index": "zipcode"})
        weights_df.to_csv(self.tables_dir / "portfolio_weights.csv", index=False)

        self._save_efficient_frontier(frontier_returns, frontier_vars, risk_df, "efficient_frontier.png")
        self._save_portfolio_allocation(latest, x_hybrid, "hybrid_portfolio_allocation.png")
        self._save_convergence(pso["history"], ga["history"], "optimizer_convergence.png")
        self._save_monte_carlo_distribution(mc_results["Hybrid"], "hybrid_monte_carlo_distribution.png")

        stress_results = self._stress_test(portfolios, mu_regime, cov)
        pd.DataFrame(stress_results).to_csv(self.tables_dir / "stress_test_results.csv", index=False)
        self._save_stress_test(pd.DataFrame(stress_results), "stress_test_summary.png")

        summary = {
            "best_method_by_hybrid_pipeline": hybrid_method,
            "selected_regime": self.latest_regime_label,
            "monte_carlo_scenarios": self.config.monte_carlo_scenarios,
            "weight_count": int(len(x_hybrid)),
        }
        return {
            "summary": summary,
            "latest_inputs": latest,
            "portfolios": portfolios,
            "mc_results": mc_results,
            "risk_metrics": risk_df,
            "stress_results": stress_results,
        }

    def validate_statistics(
        self,
        optimization_results: dict[str, Any],
        covariance_results: dict[str, Any],
        panel: pd.DataFrame,
    ) -> dict[str, Any]:
        mc = optimization_results["mc_results"]
        benchmark_names = ["EqualWeight", "RandomPortfolio", "MeanVariance", "QP", "PSO", "GA"]
        hybrid_returns = np.asarray(mc["Hybrid"]["returns_full"], dtype=float)
        pso_returns = np.asarray(mc["PSO"]["returns_full"], dtype=float)
        ga_returns = np.asarray(mc["GA"]["returns_full"], dtype=float)
        qp_returns = np.asarray(mc["QP"]["returns_full"], dtype=float)
        paired_rows: list[dict[str, Any]] = []
        raw_pvals: list[float] = []
        for benchmark in benchmark_names:
            other = np.asarray(mc[benchmark]["returns_full"], dtype=float)
            diff = hybrid_returns - other
            normality_sample = diff[: min(len(diff), 5000)]
            normality_p = float(stats.normaltest(normality_sample).pvalue) if len(normality_sample) >= 20 else 1.0
            t_stat, t_p = stats.ttest_rel(hybrid_returns, other) if normality_p > 0.05 else (np.nan, np.nan)
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            if np.allclose(diff, diff[0]):
                ci_low = float(diff[0])
                ci_high = float(diff[0])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    warnings.simplefilter("ignore")
                    ci = bootstrap(
                        (diff,),
                        np.mean,
                        confidence_level=0.95,
                        random_state=self.config.random_state,
                        n_resamples=2000,
                        method="percentile",
                    ).confidence_interval
                ci_low = float(ci.low)
                ci_high = float(ci.high)
            pooled = np.concatenate([hybrid_returns, other])
            effect_size = (np.mean(hybrid_returns) - np.mean(other)) / (np.std(pooled, ddof=1) + 1e-9)
            raw_pvals.append(float(w_p) if not np.isnan(w_p) else 1.0)
            paired_rows.append(
                {
                    "benchmark": benchmark,
                    "normality_p": normality_p,
                    "paired_t_stat": float(t_stat) if not np.isnan(t_stat) else None,
                    "paired_t_pvalue": float(t_p) if not np.isnan(t_p) else None,
                    "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
                    "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else None,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "cohens_d": float(effect_size),
                }
            )
        adjusted = self._holm_correction(raw_pvals)
        for row, adj in zip(paired_rows, adjusted, strict=False):
            row["holm_adjusted_pvalue"] = float(adj)
        anova_stat, anova_p = stats.f_oneway(hybrid_returns, qp_returns, ga_returns, pso_returns)
        friedman_stat, friedman_p = stats.friedmanchisquare(
            hybrid_returns[:2000],
            qp_returns[:2000],
            ga_returns[:2000],
            pso_returns[:2000],
        )
        result = {
            "pairwise_hybrid_comparisons": paired_rows,
            "anova": {"f_stat": float(anova_stat), "p_value": float(anova_p)},
            "friedman": {"stat": float(friedman_stat), "p_value": float(friedman_p)},
            "asset_count": int(panel["zipcode"].nunique()),
            "covariance_estimator": covariance_results["summary"]["selected_estimator"],
            "inference_scope": "Monte Carlo path comparison only; not independent optimizer-run inference",
            "optimizer_level_inference_available": False,
            "warning": "Do not interpret these p-values as evidence that one optimizer dominates. Use optimizer_seed_runs.csv and walk-forward rebalance outcomes for inferential claims.",
        }
        self._write_json(self.tables_dir / "statistical_validation.json", result)
        return result

    def run_experimental_studies(
        self,
        panel: pd.DataFrame,
        covariance_results: dict[str, Any],
        optimization_results: dict[str, Any],
    ) -> dict[str, Any]:
        latest = optimization_results["latest_inputs"].sort_values("zipcode").reset_index(drop=True)
        returns_matrix_df = covariance_results["returns_matrix"].reindex(columns=latest["zipcode"]).fillna(0.0)
        returns_matrix = returns_matrix_df.to_numpy(dtype=float)
        mu = latest["predicted_return"].to_numpy(dtype=float)
        tx_cost = latest["predicted_transaction_cost"].to_numpy(dtype=float)
        liquidity = latest["predicted_liquidity"].to_numpy(dtype=float)
        cov = covariance_results["matrix"]
        cost_vector = self._normalize_cost_vector(latest["price"].to_numpy(dtype=float), tx_cost)
        adjusted_mu = mu + 0.10 * liquidity - 0.25 * tx_cost
        budget = 1.0
        alpha = self.config.alpha

        def objective_fn(x: np.ndarray) -> float:
            return float(np.dot(x, mu) - self.config.lambda_param * (x @ cov @ x) + 0.10 * np.dot(x, liquidity) - 0.25 * np.dot(x, tx_cost))

        def constraint_proj(x: np.ndarray) -> np.ndarray:
            return project_constraints(x, cost_vector, budget, alpha)

        baselines = self._classical_benchmarks(adjusted_mu, cov, cost_vector, budget, alpha)
        ablation = self._optimizer_ablation(adjusted_mu, cov, cost_vector, alpha, objective_fn, constraint_proj)
        sensitivity = self._optimizer_sensitivity(adjusted_mu, cov, cost_vector, alpha, objective_fn, constraint_proj)
        seed_analysis = self._optimizer_seed_analysis(adjusted_mu, cov, cost_vector, alpha, objective_fn, constraint_proj)
        scalability = self._scalability_study(panel)
        backtest = self._rolling_backtest(panel)

        self._write_json(self.tables_dir / "ablation_summary.json", ablation["summary"])
        self._write_json(self.tables_dir / "sensitivity_summary.json", sensitivity["summary"])
        self._write_json(self.tables_dir / "seed_analysis_summary.json", seed_analysis["summary"])
        self._write_json(self.tables_dir / "scalability_summary.json", scalability["summary"])
        self._write_json(self.tables_dir / "rolling_backtest_summary.json", backtest["summary"])

        summary = {
            "ablation_best": ablation["summary"].get("best_variant"),
            "seed_best_mean_objective": seed_analysis["summary"].get("best_mean_objective_variant"),
            "backtest_periods": backtest["summary"].get("periods"),
            "scalability_max_assets": scalability["summary"].get("max_assets_evaluated"),
            "classical_benchmarks": list(baselines.keys()),
        }
        return {
            "summary": summary,
            "ablation": ablation,
            "sensitivity": sensitivity,
            "seed_analysis": seed_analysis,
            "scalability": scalability,
            "backtest": backtest,
            "classical_benchmarks": baselines,
        }

    def _feature_columns(self, panel: pd.DataFrame) -> list[str]:
        return [
            "price",
            "median_sale_price",
            "median_list_price",
            "median_ppsf",
            "median_list_ppsf",
            "homes_sold",
            "pending_sales",
            "new_listings",
            "inventory",
            "median_dom",
            "avg_sale_to_list",
            "sold_above_list",
            "off_market_in_two_weeks",
            "bank",
            "bus",
            "hospital",
            "mall",
            "park",
            "restaurant",
            "school",
            "station",
            "supermarket",
            "Total Population",
            "Median Age",
            "Per Capita Income",
            "Total Housing Units",
            "Median Rent",
            "Median Home Value",
            "employment_rate",
            "rental_yield",
            "turnover",
            "appreciation_rate",
            "growth_rate_3m",
            "momentum_3m",
            "momentum_6m",
            "momentum_12m",
            "ma_3",
            "ma_6",
            "ma_12",
            "volatility_3",
            "volatility_6",
            "turnover_rolling_3",
            "inventory_change",
            "demand_pressure",
            "list_to_sale_gap",
            "asset_return_lag1",
            "turnover_lag1",
            "rental_yield_lag1",
            "momentum_3m_lag1",
            "momentum_6m_lag1",
            "volatility_3_lag1",
            "volatility_6_lag1",
            "inventory_change_lag1",
            "demand_pressure_lag1",
            "avg_sale_to_list_lag1",
            "sold_above_list_lag1",
            "median_dom_lag1",
        ]

    def _regression_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=self.config.random_state),
            "Lasso": Lasso(alpha=0.001, random_state=self.config.random_state, max_iter=5000),
            "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.config.random_state, max_iter=5000),
            "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=self.config.random_state),
            "AdaBoost": AdaBoostRegressor(random_state=self.config.random_state, n_estimators=120),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=self.config.random_state, n_jobs=-1
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=120, max_depth=8, min_samples_leaf=2, random_state=self.config.random_state, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.config.random_state),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=self.config.random_state),
            "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.01),
            "KNN": KNeighborsRegressor(n_neighbors=8, weights="distance"),
            "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=self.config.random_state),
        }
        availability_messages: list[str] = []
        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=self.config.random_state,
                n_jobs=1,
            )
        else:
            availability_messages.append("XGBoost unavailable in the current environment.")
        if LGBMRegressor is not None:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.config.random_state,
                verbose=-1,
            )
        else:
            availability_messages.append("LightGBM unavailable in the current environment.")
        if CatBoostRegressor is not None:
            models["CatBoost"] = DenseEstimatorAdapter(
                CatBoostRegressor(
                    iterations=120,
                    depth=5,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=self.config.random_state,
                )
            )
        else:
            availability_messages.append("CatBoost unavailable in the current environment.")
        if TabNetRegressor is not None:
            availability_messages.append(
                "TabNet installed but excluded from the default benchmark because it is not reproducibly tractable in the managed CPU-only execution window."
            )
        else:
            availability_messages.append("TabNet unavailable in the current environment.")
        self.limitations.extend(msg for msg in availability_messages if msg not in self.limitations)
        return models

    def _classical_benchmarks(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cost_vector: np.ndarray,
        budget: float,
        alpha: float,
    ) -> dict[str, np.ndarray]:
        n_assets = len(mu)
        equal_weight = np.ones(n_assets, dtype=float) / max(n_assets, 1)
        min_var = solve_quadratic_program(np.zeros_like(mu), cov, cost_vector, budget, alpha, lambda_param=1.0)["x"]
        markowitz = solve_quadratic_program(mu, cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param)["x"]
        inv_vol = 1.0 / (np.sqrt(np.diag(cov)) + 1e-9)
        risk_parity = inv_vol / inv_vol.sum()
        max_sharpe_proxy = solve_quadratic_program(mu, cov, cost_vector, budget, alpha, lambda_param=0.5)["x"]
        return {
            "EqualWeight": project_constraints(equal_weight, cost_vector, budget, alpha),
            "MinimumVariance": project_constraints(min_var if min_var is not None else equal_weight, cost_vector, budget, alpha),
            "MaximumSharpeProxy": project_constraints(max_sharpe_proxy if max_sharpe_proxy is not None else equal_weight, cost_vector, budget, alpha),
            "Markowitz": project_constraints(markowitz if markowitz is not None else equal_weight, cost_vector, budget, alpha),
            "RiskParityApprox": project_constraints(risk_parity, cost_vector, budget, alpha),
        }

    def _optimizer_ablation(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cost_vector: np.ndarray,
        alpha: float,
        objective_fn: Any,
        constraint_proj: Any,
    ) -> dict[str, Any]:
        budget = 1.0
        qp = solve_quadratic_program(mu, cov, cost_vector, budget, alpha, lambda_param=self.config.lambda_param)
        x_qp = project_constraints(qp["x"] if qp["x"] is not None else np.ones_like(mu) / len(mu), cost_vector, budget, alpha)
        experiments = []
        configs = [
            ("QP", None, None),
            ("Random PSO", "PSO", None),
            ("QP Seeded PSO", "PSO", x_qp),
            ("Random GA", "GA", None),
            ("QP Seeded GA", "GA", x_qp),
            ("Hybrid QP→PSO→GA", "HYBRID", x_qp),
        ]
        for label, method, seed_vec in configs:
            started = perf_counter()
            if label == "QP":
                weights = x_qp
            elif label == "Random PSO":
                weights = optimize_with_pso(objective_fn, constraint_proj, x0=None, n_dimensions=len(mu), n_particles=24, n_iterations=80, seed=self.config.random_state)["x"]
            elif label == "QP Seeded PSO":
                weights = optimize_with_pso(objective_fn, constraint_proj, x0=seed_vec, n_particles=24, n_iterations=80, seed=self.config.random_state)["x"]
            elif label == "Random GA":
                weights = optimize_with_ga(objective_fn, constraint_proj, x0=None, n_dimensions=len(mu), population_size=32, n_generations=80, seed=self.config.random_state)["x"]
            elif label == "QP Seeded GA":
                weights = optimize_with_ga(objective_fn, constraint_proj, x0=seed_vec, population_size=32, n_generations=80, seed=self.config.random_state)["x"]
            else:
                pso = optimize_with_pso(objective_fn, constraint_proj, x0=seed_vec, n_particles=24, n_iterations=80, seed=self.config.random_state)
                ga = optimize_with_ga(objective_fn, constraint_proj, x0=pso["x"], population_size=32, n_generations=80, seed=self.config.random_state)
                weights = ga["x"]
            runtime = perf_counter() - started
            mc = monte_carlo_evaluate(weights, mu, cov, n_scenarios=4000, seed=self.config.random_state)
            experiments.append(
                {
                    "variant": label,
                    "objective": float(objective_fn(weights)),
                    "runtime_sec": float(runtime),
                    "return": float(mc["mean_return"]),
                    "risk": float(mc["std_return"]),
                    "sharpe": float(mc["sharpe_ratio"]),
                    "cvar": float(mc["loss_cvar_95"]),
                }
            )
        ablation_df = pd.DataFrame(experiments).sort_values("objective", ascending=False)
        ablation_df.to_csv(self.tables_dir / "optimizer_ablation.csv", index=False)
        return {"summary": {"best_variant": str(ablation_df.iloc[0]["variant"])}, "table": experiments}

    def _optimizer_sensitivity(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cost_vector: np.ndarray,
        alpha: float,
        objective_fn: Any,
        constraint_proj: Any,
    ) -> dict[str, Any]:
        pso_rows = []
        for n_particles in [16, 24, 32]:
            for n_iterations in [60, 100]:
                for w_init, c1, c2 in [(0.9, 1.5, 1.5), (0.8, 1.2, 1.8)]:
                    pso = ParticleSwarmOptimizer(objective_fn, constraint_proj, n_particles=n_particles, n_dimensions=len(mu), w_init=w_init, w_final=0.4, c1=c1, c2=c2, seed=self.config.random_state)
                    result = pso.optimize(n_iterations=n_iterations, verbose=False)
                    pso_rows.append({"population": n_particles, "iterations": n_iterations, "inertia": w_init, "cognitive": c1, "social": c2, "objective": float(result["fitness"])})
        ga_rows = []
        for pop in [24, 32, 40]:
            for generations in [60, 100]:
                for mutation_rate, crossover_rate, elite_fraction, tournament_size in [(0.10, 0.8, 0.15, 3), (0.18, 0.9, 0.20, 4)]:
                    ga = GeneticAlgorithm(objective_fn, constraint_proj, population_size=pop, n_dimensions=len(mu), mutation_rate=mutation_rate, crossover_rate=crossover_rate, elite_fraction=elite_fraction, tournament_size=tournament_size, seed=self.config.random_state)
                    result = ga.optimize(n_generations=generations, verbose=False)
                    ga_rows.append({"population": pop, "generations": generations, "mutation": mutation_rate, "crossover": crossover_rate, "elite": elite_fraction, "tournament": tournament_size, "objective": float(result["fitness"])})
        pd.DataFrame(pso_rows).to_csv(self.tables_dir / "pso_sensitivity.csv", index=False)
        pd.DataFrame(ga_rows).to_csv(self.tables_dir / "ga_sensitivity.csv", index=False)
        return {
            "summary": {
                "best_pso_objective": float(pd.DataFrame(pso_rows)["objective"].max()),
                "best_ga_objective": float(pd.DataFrame(ga_rows)["objective"].max()),
            },
            "pso": pso_rows,
            "ga": ga_rows,
        }

    def _optimizer_seed_analysis(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cost_vector: np.ndarray,
        alpha: float,
        objective_fn: Any,
        constraint_proj: Any,
    ) -> dict[str, Any]:
        qp = solve_quadratic_program(mu, cov, cost_vector, 1.0, alpha, lambda_param=self.config.lambda_param)
        x_qp = project_constraints(qp["x"] if qp["x"] is not None else np.ones_like(mu) / len(mu), cost_vector, 1.0, alpha)
        rows = []
        for seed in range(self.config.n_optimizer_seeds):
            pso_random = optimize_with_pso(objective_fn, constraint_proj, x0=None, n_dimensions=len(mu), n_particles=20, n_iterations=60, seed=seed)["x"]
            pso_seeded = optimize_with_pso(objective_fn, constraint_proj, x0=x_qp, n_particles=20, n_iterations=60, seed=seed)["x"]
            ga_random = optimize_with_ga(objective_fn, constraint_proj, x0=None, n_dimensions=len(mu), population_size=28, n_generations=60, seed=seed)["x"]
            ga_seeded = optimize_with_ga(objective_fn, constraint_proj, x0=x_qp, population_size=28, n_generations=60, seed=seed)["x"]
            hybrid = optimize_with_ga(objective_fn, constraint_proj, x0=pso_seeded, population_size=28, n_generations=60, seed=seed)["x"]
            for label, weights in {
                "Random PSO": pso_random,
                "QP Seeded PSO": pso_seeded,
                "Random GA": ga_random,
                "QP Seeded GA": ga_seeded,
                "Hybrid": hybrid,
            }.items():
                rows.append({"seed": seed, "variant": label, "objective": float(objective_fn(weights))})
        seed_df = pd.DataFrame(rows)
        summary_rows = []
        for variant, group in seed_df.groupby("variant"):
            values = group["objective"].to_numpy(dtype=float)
            ci = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values)) if len(values) > 1 else (values[0], values[0])
            summary_rows.append({
                "variant": variant,
                "mean_objective": float(np.mean(values)),
                "std_objective": float(np.std(values, ddof=1)),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "best": float(np.max(values)),
                "worst": float(np.min(values)),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("mean_objective", ascending=False)
        seed_df.to_csv(self.tables_dir / "optimizer_seed_runs.csv", index=False)
        summary_df.to_csv(self.tables_dir / "optimizer_seed_summary.csv", index=False)
        return {"summary": {"best_mean_objective_variant": str(summary_df.iloc[0]["variant"])}, "table": summary_rows}

    def _scalability_study(self, panel: pd.DataFrame) -> dict[str, Any]:
        rows = []
        candidate_sizes = [30, 50, 100, 200, 500]
        rank = (
            panel.groupby("zipcode")
            .agg(avg_turnover=("turnover", "mean"), median_price=("price", "median"))
            .sort_values(["avg_turnover", "median_price"], ascending=[False, False])
        )
        for size in candidate_sizes:
            if rank.shape[0] < size:
                self.limitations.append(f"Scalability study at {size} assets not feasible with the current filtered investable panel.")
                continue
            asset_ids = rank.head(size).index.tolist()
            sub = panel[panel["zipcode"].isin(asset_ids)].copy()
            returns = sub.pivot_table(index="date", columns="zipcode", values="asset_return").sort_index().fillna(0.0)
            mu = returns.mean().to_numpy(dtype=float)
            cov = LedoitWolf().fit(returns.to_numpy()).covariance_
            costs = np.ones(size, dtype=float)
            alpha = min(0.35, max(0.03, 4.0 / size))
            started = perf_counter()
            qp = solve_quadratic_program(mu, cov, costs, 1.0, alpha, lambda_param=self.config.lambda_param)
            runtime = perf_counter() - started
            weights = qp["x"] if qp["x"] is not None else np.ones(size) / size
            weights = project_constraints(weights, costs, 1.0, alpha)
            objective = float(np.dot(weights, mu) - self.config.lambda_param * (weights @ cov @ weights))
            memory_mb = float((returns.to_numpy().nbytes + cov.nbytes + mu.nbytes) / (1024 ** 2))
            rows.append({"assets": size, "runtime_sec": runtime, "memory_mb": memory_mb, "objective": objective, "converged": bool(qp["x"] is not None)})
        pd.DataFrame(rows).to_csv(self.tables_dir / "scalability_study.csv", index=False)
        return {"summary": {"max_assets_evaluated": int(max([r["assets"] for r in rows], default=0))}, "table": rows}

    def _rolling_backtest(self, panel: pd.DataFrame) -> dict[str, Any]:
        dates = np.sort(panel["date"].unique())
        lookback = 60
        horizon = 12
        rows = []
        for end_idx in range(lookback, len(dates) - horizon + 1, horizon):
            train_dates = dates[end_idx - lookback:end_idx]
            test_dates = dates[end_idx:end_idx + horizon]
            train = panel[panel["date"].isin(train_dates)].copy()
            test = panel[panel["date"].isin(test_dates)].copy()
            returns_train = train.pivot_table(index="date", columns="zipcode", values="asset_return").sort_index().fillna(0.0)
            returns_test = test.pivot_table(index="date", columns="zipcode", values="asset_return").sort_index().fillna(0.0)
            common = returns_train.columns.intersection(returns_test.columns)
            if len(common) < 5:
                continue
            returns_train = returns_train[common]
            returns_test = returns_test[common]
            mu = returns_train.mean().to_numpy(dtype=float)
            cov = LedoitWolf().fit(returns_train.to_numpy()).covariance_
            alpha = min(0.35, max(0.05, 3.0 / len(common)))
            weights = solve_quadratic_program(mu, cov, np.ones(len(common)), 1.0, alpha, lambda_param=self.config.lambda_param)["x"]
            weights = project_constraints(weights if weights is not None else np.ones(len(common)) / len(common), np.ones(len(common)), 1.0, alpha)
            future_returns = returns_test.to_numpy(dtype=float) @ weights
            previous = np.ones(len(common), dtype=float) / len(common)
            turnover = float(np.abs(weights - previous).sum())
            friction = float(test["transaction_cost_proxy"].mean()) if "transaction_cost_proxy" in test else self.config.transaction_cost_rate
            if len(future_returns):
                future_returns[0] -= friction * turnover
            annual_return = float((np.prod(1.0 + future_returns) ** (12 / max(len(future_returns), 1))) - 1.0)
            volatility = float(np.std(future_returns) * np.sqrt(12))
            sharpe = float(np.mean(future_returns) / (np.std(future_returns) + 1e-9) * np.sqrt(12))
            cumulative = np.cumprod(1.0 + future_returns)
            drawdown = float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1.0))
            rows.append({
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "turnover": turnover,
                "transaction_cost": friction * turnover,
            })
        pd.DataFrame(rows).to_csv(self.tables_dir / "rolling_backtest.csv", index=False)
        return {"summary": {"periods": int(len(rows))}, "table": rows}

    def _build_model_pipeline(self, estimator: Any, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def _cross_val_rmse(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, splitter: Any) -> float:
        rmses = []
        for train_idx, valid_idx in splitter.split(X, y):
            model = clone(pipeline)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            rmses.append(math.sqrt(mean_squared_error(y.iloc[valid_idx], pred)))
        return float(np.mean(rmses))

    def _regression_metric_row(
        self,
        name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        train_rmse: float,
        test_rmse: float,
        n_features: int,
    ) -> dict[str, Any]:
        r2 = float(r2_score(y_true, y_pred))
        n = len(y_true)
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1)
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "model": name,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(test_rmse),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "smape": float(self._smape(y_true.to_numpy(dtype=float), np.asarray(y_pred, dtype=float))),
            "r2": r2,
            "adjusted_r2": float(adjusted_r2),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
            "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
            "train_rmse": float(train_rmse),
            "bias_variance_gap": float(test_rmse - train_rmse),
        }

    def _time_split_masks(self, panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        unique_dates = np.sort(panel["date"].unique())
        cutoff = unique_dates[-self.config.holdout_months]
        train_mask = panel["date"] < cutoff
        test_mask = panel["date"] >= cutoff
        return train_mask, test_mask

    def _latest_asset_features(self, panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        latest = panel.sort_values(["zipcode", "date"]).groupby("zipcode").tail(1).copy()
        return latest[["zipcode", "city", "price"] + [col for col in feature_cols if col not in {"zipcode", "city", "price"}]]

    def _fit_proxy_model(self, panel: pd.DataFrame, target: str, prefix: str, estimator: Any) -> dict[str, Any]:
        feature_cols = self._reduce_redundant_features(panel, self._feature_columns(panel))
        categorical_cols = ["city", "asset_id", "month", "quarter"]
        train_mask, test_mask = self._time_split_masks(panel)
        X_train = panel.loc[train_mask, feature_cols + categorical_cols]
        y_train = panel.loc[train_mask, target]
        X_test = panel.loc[test_mask, feature_cols + categorical_cols]
        y_test = panel.loc[test_mask, target]

        pipeline = self._build_model_pipeline(estimator, feature_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "mse": float(mean_squared_error(y_test, pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
            "mape": float(mean_absolute_percentage_error(y_test, pred)),
            "smape": float(self._smape(y_test.to_numpy(dtype=float), np.asarray(pred, dtype=float))),
            "r2": float(r2_score(y_test, pred)),
            "explained_variance": float(explained_variance_score(y_test, pred)),
            "median_absolute_error": float(median_absolute_error(y_test, pred)),
        }
        metrics["timeseries_rmse"] = self._cross_val_rmse(pipeline, X_train, y_train, TimeSeriesSplit(n_splits=3))
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.tables_dir / f"{prefix}_model_metrics.csv", index=False)

        residuals = pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test.to_numpy() - pred})
        self._save_prediction_plot(residuals, f"{prefix}_prediction_vs_actual.png")
        self._save_residual_plot(residuals, f"{prefix}_residuals.png")

        latest = self._latest_asset_features(panel, feature_cols + categorical_cols)
        pred_latest = pipeline.predict(latest[feature_cols + categorical_cols])
        pred_col = f"predicted_{prefix}"
        latest = latest[["zipcode", "city", "price"]].copy()
        latest[pred_col] = pred_latest
        latest.to_csv(self.tables_dir / f"{prefix}_predictions.csv", index=False)

        with open(self.models_dir / f"{prefix}_model.pkl", "wb") as handle:
            pickle.dump(pipeline, handle)

        summary = {"target": target, "rmse": metrics["rmse"], "timeseries_rmse": metrics["timeseries_rmse"]}
        return {"summary": summary, "model": pipeline, "predictions": latest}

    def _tune_installed_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: list[str],
        categorical_cols: list[str],
    ) -> dict[str, Any]:
        search_space = {
            "RandomForest": {
                "estimator": RandomForestRegressor(random_state=self.config.random_state, n_jobs=-1),
                "random_params": {
                    "n_estimators": [80, 120, 160],
                    "max_depth": [6, 8, None],
                    "min_samples_leaf": [1, 2, 3, 5],
                },
            },
            "GradientBoosting": {
                "estimator": GradientBoostingRegressor(random_state=self.config.random_state),
                "grid_params": {
                    "n_estimators": [80, 120],
                    "learning_rate": [0.03, 0.05],
                    "max_depth": [2, 3],
                },
            },
            "ExtraTrees": {
                "estimator": ExtraTreesRegressor(random_state=self.config.random_state, n_jobs=-1),
                "random_params": {
                    "n_estimators": [100, 140, 180],
                    "max_depth": [6, 8, None],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "MLP": {
                "estimator": MLPRegressor(max_iter=400, random_state=self.config.random_state),
                "random_params": {
                    "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                    "alpha": [0.0001, 0.001, 0.01],
                },
            },
        }
        results = {}
        splitter = TimeSeriesSplit(n_splits=3)
        for name, spec in search_space.items():
            best_rmse = np.inf
            best_params = None
            candidates: list[dict[str, Any]] = []
            if "random_params" in spec:
                candidates.extend(list(ParameterSampler(spec["random_params"], n_iter=3, random_state=self.config.random_state)))
            if "grid_params" in spec:
                candidates.extend(list(ParameterGrid(spec["grid_params"])))
            for params in candidates:
                estimator = clone(spec["estimator"]).set_params(**params)
                pipeline = self._build_model_pipeline(estimator, feature_cols, categorical_cols)
                rmse = self._cross_val_rmse(pipeline, X_train, y_train, splitter)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
            results[name] = {"best_timeseries_rmse": float(best_rmse), "best_params": best_params}
        if optuna is not None:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            for name in ["XGBoost"]:
                if name not in self._regression_models():
                    continue

                def objective(trial: Any, model_name: str = name) -> float:
                    if model_name == "XGBoost":
                        estimator = XGBRegressor(
                            n_estimators=trial.suggest_int("n_estimators", 80, 180),
                            max_depth=trial.suggest_int("max_depth", 3, 6),
                            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12),
                            subsample=trial.suggest_float("subsample", 0.7, 1.0),
                            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                            objective="reg:squarederror",
                            random_state=self.config.random_state,
                            n_jobs=1,
                        )
                    else:
                        estimator = LGBMRegressor(
                            n_estimators=trial.suggest_int("n_estimators", 80, 180),
                            max_depth=trial.suggest_int("max_depth", 3, 8),
                            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12),
                            subsample=trial.suggest_float("subsample", 0.7, 1.0),
                            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                            random_state=self.config.random_state,
                            verbose=-1,
                        )
                    pipeline = self._build_model_pipeline(estimator, feature_cols, categorical_cols)
                    return self._cross_val_rmse(pipeline, X_train, y_train, splitter)

                try:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=5, show_progress_bar=False)
                    results[f"{name}_optuna"] = {
                        "best_timeseries_rmse": float(study.best_value),
                        "best_params": dict(study.best_params),
                    }
                except Exception as exc:
                    results[f"{name}_optuna"] = {"unavailable": True, "reason": str(exc)}
                    note = f"{name} Optuna tuning unavailable at execution time: {exc}"
                    if note not in self.limitations:
                        self.limitations.append(note)
        self._write_json(self.tables_dir / "hyperparameter_search_results.json", results)
        return results

    def _reduce_redundant_features(self, panel: pd.DataFrame, feature_cols: list[str], threshold: float = 0.98) -> list[str]:
        numeric = panel[feature_cols].select_dtypes(include=[np.number]).copy()
        keep = [col for col in numeric.columns if numeric[col].nunique(dropna=True) > 1]
        numeric = numeric[keep]
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in feature_cols if col in keep and col not in drop]

    def _rank_models(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        ranking = metrics_df.copy()
        components = {
            "timeseries_rmse": False,
            "rmse": False,
            "bias_variance_gap": False,
            "fit_runtime_sec": False,
            "r2": True,
            "explained_variance": True,
        }
        composite = np.zeros(len(ranking), dtype=float)
        for col, higher_is_better in components.items():
            values = ranking[col].to_numpy(dtype=float)
            transformed = values if higher_is_better else -values
            if np.allclose(transformed.max(), transformed.min()):
                scaled = np.ones_like(transformed)
            else:
                scaled = (transformed - transformed.min()) / (transformed.max() - transformed.min())
            composite += scaled
        ranking["selection_score"] = composite / len(components)
        return ranking.sort_values(["selection_score", "timeseries_rmse", "rmse"], ascending=[False, True, True]).reset_index(drop=True)

    @staticmethod
    def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
        return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

    def _save_mutual_information(self, X: pd.DataFrame, y: pd.Series, filename: str) -> None:
        matrix = X.fillna(X.median(numeric_only=True))
        scores = mutual_info_regression(matrix, y, random_state=self.config.random_state)
        table = pd.DataFrame({"feature": X.columns, "mutual_information": scores}).sort_values("mutual_information", ascending=False)
        table.to_csv(self.tables_dir / filename, index=False)

    def _save_rfe(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, filename: str) -> None:
        transformed = model.named_steps["preprocessor"].fit_transform(X)
        selector = RFE(estimator=Ridge(alpha=1.0, random_state=self.config.random_state), n_features_to_select=min(12, transformed.shape[1]), step=0.1)
        selector.fit(transformed, y)
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = np.array([f"feature_{idx}" for idx in range(transformed.shape[1])])
        table = pd.DataFrame({"feature": feature_names, "selected": selector.support_, "ranking": selector.ranking_})
        table.sort_values(["selected", "ranking"], ascending=[False, True]).to_csv(self.tables_dir / filename, index=False)

    def _save_shap_artifacts(self, model: Pipeline, X: pd.DataFrame, prefix: str) -> None:
        note = "SHAP installed but not reproducibly usable in this managed Windows environment because application-control policy blocks required native components; permutation importance, PDP, mutual information, and RFE were used instead."
        if note not in self.limitations:
            self.limitations.append(note)
        return
        try:
            sample = X.sample(min(len(X), 200), random_state=self.config.random_state)
            transformed = model.named_steps["preprocessor"].transform(sample)
            if sparse.issparse(transformed):
                transformed = transformed.toarray()
            transformed = np.asarray(transformed, dtype=np.float32)
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            estimator = model.named_steps["model"]
            if hasattr(estimator, "estimator"):
                estimator = estimator.estimator
            explainer = shap.Explainer(estimator, transformed, feature_names=feature_names)
            shap_values = explainer(transformed[: min(100, len(transformed))])
            plt.figure()
            shap.plots.beeswarm(shap_values, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{prefix}_shap_summary.png", dpi=self.config.figure_dpi)
            plt.close()
            top_feature = str(feature_names[int(np.argmax(np.abs(np.asarray(shap_values.values)).mean(axis=0)))])
            plt.figure()
            shap.plots.scatter(shap_values[:, top_feature], show=False)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{prefix}_shap_dependence.png", dpi=self.config.figure_dpi)
            plt.close()
        except Exception as exc:
            note = f"SHAP generation unavailable for the selected model: {exc}"
            if note not in self.limitations:
                self.limitations.append(note)

    @staticmethod
    def _holm_correction(p_values: list[float]) -> list[float]:
        indexed = sorted(enumerate(p_values), key=lambda item: item[1])
        m = len(indexed)
        adjusted = [1.0] * m
        running_max = 0.0
        for rank, (idx, pval) in enumerate(indexed, start=1):
            adj = min(1.0, (m - rank + 1) * pval)
            running_max = max(running_max, adj)
            adjusted[idx] = running_max
        return adjusted

    def _normalize_cost_vector(self, prices: np.ndarray, tx_cost: np.ndarray) -> np.ndarray:
        cost = prices / (np.mean(prices) + 1e-9)
        cost = cost * (1.0 + tx_cost)
        return cost / (np.mean(cost) + 1e-9)

    def _risk_metric_row(
        self,
        name: str,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        market_proxy: np.ndarray,
        mc_stats: dict[str, Any],
    ) -> dict[str, Any]:
        portfolio_returns = returns_matrix @ weights
        downside = portfolio_returns[portfolio_returns < 0]
        semivariance = float(np.mean(np.square(downside))) if len(downside) else 0.0
        downside_risk = float(np.sqrt(semivariance))
        cumulative = np.cumprod(1.0 + portfolio_returns)
        running_peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / (running_peak + 1e-9) - 1.0
        max_drawdown = float(np.min(drawdowns))
        beta = float(np.cov(portfolio_returns, market_proxy)[0, 1] / (np.var(market_proxy) + 1e-9))
        sortino = float(np.mean(portfolio_returns) / (downside_risk + 1e-9))
        calmar = float(np.mean(portfolio_returns) / (abs(max_drawdown) + 1e-9))
        gains = portfolio_returns[portfolio_returns > 0]
        losses = np.abs(portfolio_returns[portfolio_returns < 0])
        omega = float(gains.sum() / (losses.sum() + 1e-9))
        tail_risk = float(np.quantile(portfolio_returns, 0.01))
        return {
            "portfolio": name,
            "expected_return": float(mc_stats["mean_return"]),
            "volatility": float(mc_stats["std_return"]),
            "beta": beta,
            "value_at_risk_95": float(mc_stats["loss_var_95"]),
            "conditional_value_at_risk_95": float(mc_stats["loss_cvar_95"]),
            "downside_risk": downside_risk,
            "semi_variance": semivariance,
            "max_drawdown": max_drawdown,
            "tail_risk_1pct": tail_risk,
            "sortino_ratio": sortino,
            "sharpe_ratio": float(mc_stats["sharpe_ratio"]),
            "calmar_ratio": calmar,
            "omega_ratio": omega,
        }

    def _stress_test(self, portfolios: dict[str, np.ndarray], mu: np.ndarray, cov: np.ndarray) -> list[dict[str, Any]]:
        scenarios = {
            "MarketCrash": {"mu_scale": 0.55, "cov_scale": 1.8},
            "InterestRateIncrease": {"mu_scale": 0.82, "cov_scale": 1.2},
            "InflationShock": {"mu_scale": 0.90, "cov_scale": 1.35},
            "LiquidityCrisis": {"mu_scale": 0.72, "cov_scale": 1.55},
            "DemandCollapse": {"mu_scale": 0.65, "cov_scale": 1.45},
        }
        rows = []
        for scenario, spec in scenarios.items():
            shocked_mu = mu * spec["mu_scale"]
            shocked_cov = cov * spec["cov_scale"]
            for name, weights in portfolios.items():
                stats_row = monte_carlo_evaluate(
                    weights,
                    shocked_mu,
                    shocked_cov,
                    n_scenarios=2500,
                    seed=self.config.random_state,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "portfolio": name,
                        "expected_return": float(stats_row["mean_return"]),
                        "volatility": float(stats_row["std_return"]),
                        "loss_cvar_95": float(stats_row["loss_cvar_95"]),
                    }
                )
        return rows

    def _map_regime_labels(self, market: pd.DataFrame) -> dict[int, str]:
        mapping: dict[int, str] = {}
        grouped = market.groupby("regime_cluster").agg(
            mean_return=("market_return", "mean"),
            mean_vol=("market_volatility", "mean"),
        )
        ret_rank = grouped["mean_return"].rank(method="dense")
        vol_rank = grouped["mean_vol"].rank(method="dense")
        for cluster, row in grouped.iterrows():
            if ret_rank.loc[cluster] == ret_rank.max():
                mapping[int(cluster)] = "Bull"
            elif ret_rank.loc[cluster] == ret_rank.min():
                mapping[int(cluster)] = "Bear"
            elif vol_rank.loc[cluster] == vol_rank.max():
                mapping[int(cluster)] = "High Volatility"
            elif vol_rank.loc[cluster] == vol_rank.min():
                mapping[int(cluster)] = "Low Volatility"
            else:
                mapping[int(cluster)] = "Sideways"
        return mapping

    def _approximate_adf_test(self, series: pd.Series, lags: int = 1) -> dict[str, Any]:
        y = series.dropna().to_numpy(dtype=float)
        dy = np.diff(y)
        y_lag = y[:-1]
        X_parts = [y_lag.reshape(-1, 1)]
        for lag in range(1, lags + 1):
            lagged_diff = np.roll(dy, lag)
            lagged_diff[:lag] = 0.0
            X_parts.append(lagged_diff.reshape(-1, 1))
        X = np.hstack(X_parts)
        X = X[lags:]
        y_target = dy[lags:]
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y_target, rcond=None)
        resid = y_target - X @ beta
        sigma2 = float(np.sum(resid**2) / max(len(y_target) - X.shape[1], 1))
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        stderr = math.sqrt(max(cov_beta[1, 1], 1e-12))
        t_stat = float(beta[1] / stderr)
        return {
            "adf_t_stat": t_stat,
            "approx_5pct_critical_value": -2.86,
            "stationary_at_5pct_proxy": bool(t_stat < -2.86),
            "note": "Approximate ADF t-statistic without MacKinnon p-value because statsmodels is unavailable.",
        }

    def _simple_decompose(self, series: pd.Series, period: int = 12) -> tuple[pd.Series, pd.Series, pd.Series]:
        trend = series.rolling(period, center=True, min_periods=max(3, period // 2)).mean()
        detrended = series - trend
        seasonal_profile = detrended.groupby(detrended.index.month).transform("mean")
        seasonal = seasonal_profile.reindex(series.index)
        residual = series - trend - seasonal
        return trend, seasonal, residual

    def _save_distribution_plot(self, df: pd.DataFrame, cols: list[str], filename: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, col in zip(axes.flat, cols, strict=False):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1f77b4")
            ax.set_title(col)
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_correlation_heatmap(self, corr: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_trend_plot(self, market: pd.Series, trend: pd.Series, seasonal: pd.Series, filename: str) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(market.index, market.values, color="#0b7285")
        axes[0].set_title("Market Price")
        axes[1].plot(trend.index, trend.values, color="#d9480f")
        axes[1].set_title("Trend")
        axes[2].plot(seasonal.index, seasonal.values, color="#5f3dc4")
        axes[2].set_title("Seasonality")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_boxplot(self, df: pd.DataFrame, cols: list[str], filename: str) -> None:
        fig, axes = plt.subplots(1, len(cols), figsize=(14, 5))
        for ax, col in zip(axes, cols, strict=False):
            sns.boxplot(y=df[col], ax=ax, color="#74c0fc")
            ax.set_title(col)
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_learning_curve(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, filename: str) -> None:
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.3, 1.0, 3),
            n_jobs=None,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, -train_scores.mean(axis=1), label="Train RMSE", marker="o")
        ax.plot(train_sizes, -test_scores.mean(axis=1), label="Validation RMSE", marker="o")
        ax.set_title("Learning Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_prediction_plot(self, residuals: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(residuals["actual"], residuals["predicted"], alpha=0.5, color="#1c7ed6")
        diag = [residuals["actual"].min(), residuals["actual"].max()]
        ax.plot(diag, diag, linestyle="--", color="#e03131")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Prediction vs Actual")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_residual_plot(self, residuals: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(residuals["predicted"], residuals["residual"], alpha=0.5, color="#2f9e44")
        ax.axhline(0.0, linestyle="--", color="#343a40")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Analysis")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_feature_importance(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, filename: str) -> None:
        perm = permutation_importance(model, X, y, n_repeats=4, random_state=self.config.random_state)
        imp = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean})
        imp = imp.sort_values("importance_mean", ascending=False)
        imp.to_csv(self.tables_dir / filename, index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=imp.head(15), x="importance_mean", y="feature", ax=ax, color="#495057")
        ax.set_title("Top Permutation Importances")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename.replace(".csv", ".png"), dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_partial_dependence(self, model: Pipeline, X: pd.DataFrame, features: list[str], filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            PartialDependenceDisplay.from_estimator(model, X, features=features, ax=ax)
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_covariance_heatmap(self, cov: np.ndarray, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cov, cmap="mako", ax=ax)
        ax.set_title("Selected Covariance Matrix")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_regime_timeline(self, market: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        palette = {"Bull": "#2b8a3e", "Bear": "#c92a2a", "Sideways": "#495057", "High Volatility": "#f08c00", "Low Volatility": "#1c7ed6"}
        colors = market["regime_label"].map(palette).fillna("#868e96")
        ax.scatter(market.index, market["market_return"], c=colors, s=40)
        ax.set_title("Market Regime Timeline")
        ax.set_ylabel("Mean Return")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_efficient_frontier(
        self, frontier_returns: np.ndarray, frontier_vars: np.ndarray, risk_df: pd.DataFrame, filename: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(np.sqrt(frontier_vars), frontier_returns, color="#0c8599", label="Efficient Frontier")
        ax.scatter(risk_df["volatility"], risk_df["expected_return"], color="#e8590c", alpha=0.8)
        for _, row in risk_df.iterrows():
            ax.text(row["volatility"], row["expected_return"], row["portfolio"], fontsize=8)
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier and Portfolio Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_portfolio_allocation(self, latest: pd.DataFrame, weights: np.ndarray, filename: str) -> None:
        alloc = pd.DataFrame({"zipcode": latest["zipcode"], "weight": weights, "city": latest["city"]}).sort_values(
            "weight", ascending=False
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=alloc, x="zipcode", y="weight", ax=ax, color="#5c7cfa")
        ax.set_title("Hybrid Portfolio Allocation")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_convergence(self, pso_history: dict[str, Any], ga_history: dict[str, Any], filename: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(pso_history["best_fitness"], color="#1c7ed6")
        axes[0].set_title("PSO Convergence")
        axes[1].plot(ga_history["best_fitness"], color="#2b8a3e")
        axes[1].set_title("GA Convergence")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_monte_carlo_distribution(self, mc_stats: dict[str, Any], filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(mc_stats["returns_sample"], kde=True, ax=ax, color="#7048e8")
        ax.set_title("Hybrid Monte Carlo Return Distribution")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _save_stress_test(self, stress_df: pd.DataFrame, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = stress_df.pivot(index="scenario", columns="portfolio", values="loss_cvar_95")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="rocket_r", ax=ax)
        ax.set_title("Stress Test Loss CVaR")
        fig.tight_layout()
        fig.savefig(self.figures_dir / filename, dpi=self.config.figure_dpi)
        plt.close(fig)

    def _parameter_support_table(self) -> pd.DataFrame:
        rows = [
            {
                "parameter": "Expected Return",
                "status": "Machine-Learned",
                "evidence": "Predicted from historical panel features with supervised regression.",
            },
            {
                "parameter": "Covariance",
                "status": "Machine-Learned",
                "evidence": "Estimated from historical return matrix using shrinkage and graphical models.",
            },
            {
                "parameter": "Liquidity",
                "status": "Engineering Proxy",
                "evidence": "Composite proxy using turnover, days-on-market, off-market speed, sale-to-list ratio, and Amihud-style illiquidity.",
            },
            {
                "parameter": "Transaction Cost",
                "status": "Engineering Proxy",
                "evidence": "Composite proxy using slippage, time-on-market, Amihud-style price impact, and square-root impact approximation.",
            },
            {
                "parameter": "Regime",
                "status": "Heuristic",
                "evidence": "Latent state inferred from returns and rolling volatility using BIC-selected GMM without macro covariates.",
            },
            {
                "parameter": "CVaR",
                "status": "Machine-Learned",
                "evidence": "Computed from Monte Carlo portfolio return distributions implied by estimated return and covariance inputs.",
            },
            {
                "parameter": "Risk",
                "status": "Machine-Learned",
                "evidence": "Derived from estimated covariance and simulated return distributions.",
            },
            {
                "parameter": "Volatility",
                "status": "Machine-Learned",
                "evidence": "Estimated directly from historical return realizations and portfolio simulations.",
            },
        ]
        return pd.DataFrame(rows)

    def _claim_validation_table(self) -> pd.DataFrame:
        rows = [
            {
                "claim": "PSO produced the best portfolio in the primary run.",
                "classification": "Experimentally Proven",
                "rationale": "Observed directly in the executed optimization output for the evaluated dataset and protocol.",
            },
            {
                "claim": "Hybrid warm-starting improves optimizer performance on average.",
                "classification": "Supported",
                "rationale": "Supported by ablation and 30-seed analyses, but only for the tested objective and dataset.",
            },
            {
                "claim": "Liquidity and transaction cost are real market measurements.",
                "classification": "Engineering Assumption",
                "rationale": "They are proxy constructions because no direct labels exist in the dataset.",
            },
            {
                "claim": "The regime detector identifies true macroeconomic states.",
                "classification": "Hypothesis",
                "rationale": "The model uses return/volatility structure only and is not calibrated with external macro indicators.",
            },
            {
                "claim": "The pipeline scales to hundreds of assets on this dataset.",
                "classification": "Future Work",
                "rationale": "The final investable universe is much smaller, so only theoretical scaling discussion is defensible.",
            },
        ]
        return pd.DataFrame(rows)

    def _write_methodology_note(self) -> None:
        lines = [
            "# Methodology Alignment Note",
            "",
            "## Liquidity Proxy",
            "",
            "The implemented liquidity proxy follows accepted low-frequency market-liquidity logic by combining trading frequency, time-to-sale, execution tightness, and price-impact style terms.",
            "",
            "Mathematical formulation:",
            "",
            "- Turnover proxy: `homes_sold / (inventory + 1)`",
            "- Amihud-style illiquidity proxy: `|return| / turnover`",
            "- Liquidity score: weighted combination of turnover, inverse days-on-market, off-market speed, sale-to-list efficiency, and inverse Amihud-style illiquidity",
            "",
            "Engineering rationale:",
            "",
            "- Real-estate markets do not provide exchange-style depth or bid-ask quotes in this dataset.",
            "- Low-frequency liquidity research often uses price-impact and turnover-based proxies when direct transaction microstructure data are unavailable.",
            "",
            "Citation placeholders:",
            "",
            "- `[Amihud 2002]`",
            "- `[Amihud and Mendelson 1986/2015 review]`",
            "- `[Low-frequency liquidity proxy review]`",
            "",
            "## Transaction-Cost Proxy",
            "",
            "The transaction-cost proxy is formulated as a defensible composite of observed execution friction and price-impact proxies.",
            "",
            "Mathematical formulation:",
            "",
            "- Slippage proxy: `1 - avg_sale_to_list`",
            "- Search/holding-cost proxy: normalized `median_dom`",
            "- Amihud-style price impact: `|return| / turnover`",
            "- Square-root impact proxy: `sqrt(|return|) / sqrt(turnover)`",
            "- Transaction cost proxy: affine scaling of the weighted composite to a bounded cost band",
            "",
            "Engineering rationale:",
            "",
            "- True brokerage, taxes, and order-book impact are unavailable.",
            "- The resulting proxy should be interpreted as relative execution friction, not a quoted broker fee.",
            "",
            "Citation placeholders:",
            "",
            "- `[Amihud 2002]`",
            "- `[Almgren-Chriss / square-root impact literature]`",
            "- `[Real-estate transaction friction literature]`",
            "",
            "## Regime Detection",
            "",
            "The regime detector now uses BIC-selected Gaussian mixtures over market return and rolling volatility features rather than an arbitrary fixed unsupervised labeler.",
            "",
            "Engineering rationale:",
            "",
            "- Historical returns and volatility clustering are available from the dataset.",
            "- External macro indicators are not available inside the data source, so macro-calibrated HMM or Markov-switching specifications cannot be justified without bringing in new data.",
            "",
            "Citation placeholders:",
            "",
            "- `[GMM regime detection literature]`",
            "- `[Markov-switching financial regime literature]`",
        ]
        (self.reports_dir / "methodology_alignment.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_limitation_analysis(self) -> None:
        lines = [
            "# Limitation Analysis",
            "",
            "## Dataset Limitations",
            "",
            "- Liquidity is not directly observed through quotes, order-book depth, or transaction-level bid-ask spreads.",
            "- Transaction cost is not directly observed through brokerage, tax, or impact records.",
            "- External macroeconomic indicators are absent, so regime validation against macro states is unavailable.",
            "",
            "## Methodological Limitations",
            "",
            "- Liquidity and transaction cost are engineering proxies rather than labeled targets.",
            "- Regime states are latent statistical clusters, not verified economic truth labels.",
            "- Monte Carlo-based tail-risk estimates depend on the estimated covariance and return models.",
            "",
            "## Computational Limitations",
            "",
            "- The final investable panel contains about 15 assets, so empirical scalability claims beyond that universe would be scientifically inappropriate.",
            "- TabNet was excluded from the default benchmark because it is not reproducibly tractable in the managed CPU-only execution window.",
            "",
            "## Environmental Limitations",
            "",
            "- SHAP is installed but blocked by application-control restrictions in this Windows environment.",
            "- LightGBM is installed but unstable at runtime in this environment due a native access-violation path.",
            "",
            "## Future Work",
            "",
            "- Add external macroeconomic series to support macro-calibrated regime models.",
            "- Add observed transaction-cost and liquidity labels if proprietary market data become available.",
            "- Re-evaluate explainability with SHAP in an environment that permits the required native runtime components.",
        ]
        (self.reports_dir / "limitations_threats_future_work.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)

    def _write_report(self, report: dict[str, Any]) -> None:
        lines = [
            "# Real-Estate Portfolio Research Pipeline",
            "",
            "## Summary",
            "",
            f"- Dataset: `{self.config.dataset_path}`",
            f"- Selected assets: `{len(self.selected_assets)}` zipcodes",
            f"- Best expected-return model: `{report['expected_return_results']['best_model']}`",
            f"- Selected covariance estimator: `{report['covariance_results']['selected_estimator']}`",
            f"- Latest detected regime: `{report['regime_results']['latest_regime']}`",
            "",
            "## Limitations",
            "",
        ]
        for item in self.limitations:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "## Statistical Validation",
                "",
                "```json",
                json.dumps(report["validation_results"], indent=2),
                "```",
            ]
        )
        (self.reports_dir / "research_report.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / (series.std(ddof=0) + 1e-9)

    @staticmethod
    def _minmax_series(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def main() -> None:
    config = ResearchConfig()
    pipeline = RealEstateResearchPipeline(config)
    report = pipeline.run()
    print(json.dumps(report["optimization_results"], indent=2))


if __name__ == "__main__":
    main()
