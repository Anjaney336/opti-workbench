"""
Dynamic multi-asset data engine for portfolio optimization.

This module is intentionally self-contained so the orchestration layer can
consume structured assets, covariance matrices, and JSON-ready payloads without
depending on the rest of the optimization stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "Asset",
    "AssetUniverse",
    "AssetEngineConfig",
    "generate_asset_universe",
    "generate_assets",
    "build_covariance_matrix",
    "asset_to_dict",
    "assets_to_dicts",
    "universe_to_dict",
    "serialize_for_api",
]


_SECTORS = ("Residential", "Commercial", "Land")
_SCENARIOS = {
    "high_growth": {
        "mu_range": (0.105, 0.205),
        "sigma_range": (0.12, 0.28),
        "liquidity_range": (0.42, 0.82),
        "cost_range": (0.010, 0.032),
        "sector_probs": (0.24, 0.36, 0.40),
    },
    "stable": {
        "mu_range": (0.055, 0.115),
        "sigma_range": (0.06, 0.18),
        "liquidity_range": (0.68, 0.96),
        "cost_range": (0.006, 0.020),
        "sector_probs": (0.55, 0.30, 0.15),
    },
    "risky": {
        "mu_range": (0.085, 0.180),
        "sigma_range": (0.16, 0.36),
        "liquidity_range": (0.30, 0.70),
        "cost_range": (0.012, 0.040),
        "sector_probs": (0.20, 0.34, 0.46),
    },
    "low_liquidity": {
        "mu_range": (0.060, 0.140),
        "sigma_range": (0.10, 0.24),
        "liquidity_range": (0.18, 0.55),
        "cost_range": (0.010, 0.036),
        "sector_probs": (0.34, 0.30, 0.36),
    },
    "commercial_heavy": {
        "mu_range": (0.070, 0.155),
        "sigma_range": (0.08, 0.22),
        "liquidity_range": (0.40, 0.84),
        "cost_range": (0.010, 0.028),
        "sector_probs": (0.18, 0.62, 0.20),
    },
    "balanced": {
        "mu_range": (0.065, 0.145),
        "sigma_range": (0.08, 0.22),
        "liquidity_range": (0.50, 0.90),
        "cost_range": (0.008, 0.026),
        "sector_probs": (0.42, 0.34, 0.24),
    },
}


@dataclass(frozen=True)
class Asset:
    """Structured asset object used across the optimization stack."""

    name: str
    mu: float
    sigma: float
    liquidity: float
    cost: float
    sector: str = "Residential"


@dataclass(frozen=True)
class AssetEngineConfig:
    """Generation controls for deterministic synthetic universes."""

    n_assets: int = 12
    seed: int = 42
    scenario: str = "balanced"
    correlation_strength: float = 0.55
    factor_rank: int = 3
    asset_prefix: str = "Asset"
    allow_user_assets: bool = True


@dataclass(frozen=True)
class AssetUniverse:
    """Complete data bundle for downstream solvers and API responses."""

    assets: list[Asset]
    cov_matrix: np.ndarray
    scenario: str
    seed: int
    correlation_strength: float

    @property
    def mu(self) -> np.ndarray:
        return np.asarray([asset.mu for asset in self.assets], dtype=float)

    @property
    def sigma(self) -> np.ndarray:
        return np.asarray([asset.sigma for asset in self.assets], dtype=float)

    @property
    def liquidity(self) -> np.ndarray:
        return np.asarray([asset.liquidity for asset in self.assets], dtype=float)

    @property
    def cost(self) -> np.ndarray:
        return np.asarray([asset.cost for asset in self.assets], dtype=float)

    @property
    def sectors(self) -> list[str]:
        return [asset.sector for asset in self.assets]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(0 if seed is None else int(seed))


def _scenario_spec(scenario: str) -> dict[str, Any]:
    return _SCENARIOS.get((scenario or "balanced").lower(), _SCENARIOS["balanced"])


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if np.isclose(lo, hi):
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def _sample_sector(rng: np.random.Generator, sector_probs: Iterable[float]) -> str:
    probs = np.asarray(tuple(sector_probs), dtype=float)
    probs = probs / np.sum(probs)
    return str(rng.choice(np.asarray(_SECTORS, dtype=object), p=probs))


def _make_asset(
    rng: np.random.Generator,
    index: int,
    spec: Mapping[str, Any],
    asset_prefix: str,
) -> Asset:
    sector = _sample_sector(rng, spec["sector_probs"])
    mu = float(rng.uniform(*spec["mu_range"]))
    sigma = float(rng.uniform(*spec["sigma_range"]))
    liquidity = float(rng.uniform(*spec["liquidity_range"]))
    cost = float(rng.uniform(*spec["cost_range"]))

    sector_tilt_mu = {"Residential": 0.99, "Commercial": 1.05, "Land": 1.12}[sector]
    sector_tilt_sigma = {"Residential": 0.92, "Commercial": 1.00, "Land": 1.14}[sector]
    sector_tilt_liquidity = {"Residential": 1.08, "Commercial": 0.98, "Land": 0.84}[sector]
    sector_tilt_cost = {"Residential": 0.95, "Commercial": 1.02, "Land": 1.12}[sector]

    return Asset(
        name=f"{asset_prefix} {index + 1}",
        mu=mu * sector_tilt_mu,
        sigma=max(1e-6, sigma * sector_tilt_sigma),
        liquidity=float(np.clip(liquidity * sector_tilt_liquidity, 0.05, 1.0)),
        cost=max(1e-6, cost * sector_tilt_cost),
        sector=sector,
    )


def _coerce_asset(value: Asset | Mapping[str, Any], index: int) -> Asset:
    if isinstance(value, Asset):
        return value
    required = {"name", "mu", "sigma", "liquidity", "cost"}
    missing = required.difference(value.keys())
    if missing:
        raise ValueError(f"Provided asset at index {index} is missing fields: {sorted(missing)}")
    return Asset(
        name=str(value["name"]),
        mu=float(value["mu"]),
        sigma=float(value["sigma"]),
        liquidity=float(value["liquidity"]),
        cost=float(value["cost"]),
        sector=str(value.get("sector", "Residential")),
    )


def build_covariance_matrix(
    assets: Iterable[Asset],
    seed: int | None = 42,
    correlation_strength: float = 0.55,
    factor_rank: int = 3,
) -> np.ndarray:
    """
    Build a PSD covariance matrix using a factor model with sector-aware loadings.

    The construction is:
        Sigma = D @ Corr @ D
    where Corr is PSD by construction and D is a diagonal volatility matrix.
    """

    asset_list = list(assets)
    n_assets = len(asset_list)
    if n_assets == 0:
        raise ValueError("At least one asset is required to build a covariance matrix.")

    rng = _rng(seed)
    sigma = np.asarray([asset.sigma for asset in asset_list], dtype=float)
    sigma = np.maximum(sigma, 1e-6)

    sectors = np.asarray([asset.sector for asset in asset_list], dtype=object)
    sector_codes = np.array([{"Residential": 0, "Commercial": 1, "Land": 2}.get(str(s), 0) for s in sectors], dtype=float)

    # Latent factor exposures.  The first factor is a market factor; the rest
    # create sector-specific co-movement and preserve PSD.
    rank = max(2, int(factor_rank))
    base_loadings = rng.normal(0.0, 1.0, size=(n_assets, rank))
    base_loadings[:, 0] = 0.85 + 0.15 * _normalize(np.asarray([asset.mu for asset in asset_list], dtype=float))
    if rank > 1:
        base_loadings[:, 1] += 0.65 * (sector_codes == 1).astype(float) - 0.20 * (sector_codes == 0).astype(float)
    if rank > 2:
        base_loadings[:, 2] += 0.65 * (sector_codes == 2).astype(float) - 0.15 * (sector_codes == 1).astype(float)

    factor_cov = base_loadings @ base_loadings.T
    diag = np.sqrt(np.maximum(np.diag(factor_cov), 1e-12))
    corr_from_factors = factor_cov / np.outer(diag, diag)
    corr_from_factors = np.clip(corr_from_factors, -0.95, 0.999)
    corr_from_factors = 0.5 * (corr_from_factors + corr_from_factors.T)

    # Blend factor correlation with an identity floor and sector clustering.
    sector_affinity = np.equal.outer(sector_codes, sector_codes).astype(float)
    sector_affinity = 0.65 * sector_affinity + 0.35 * np.eye(n_assets)
    corr = (
        correlation_strength * corr_from_factors
        + 0.25 * sector_affinity
        + 0.15 * np.eye(n_assets)
    )
    corr = 0.5 * (corr + corr.T)

    # Numerical PSD repair through eigenvalue clipping.
    evals, evecs = np.linalg.eigh(corr)
    evals = np.maximum(evals, 1e-8)
    corr = evecs @ np.diag(evals) @ evecs.T
    corr_diag = np.sqrt(np.maximum(np.diag(corr), 1e-12))
    corr = corr / np.outer(corr_diag, corr_diag)
    corr = 0.5 * (corr + corr.T)
    corr = corr + 1e-10 * np.eye(n_assets)

    cov_matrix = np.diag(sigma) @ corr @ np.diag(sigma)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    cov_matrix += 1e-10 * np.eye(n_assets)
    return cov_matrix


def generate_asset_universe(
    n_assets: int | None = None,
    *,
    seed: int = 42,
    scenario: str = "balanced",
    provided_assets: Iterable[Asset | Mapping[str, Any]] | None = None,
    correlation_strength: float = 0.55,
    factor_rank: int = 3,
    asset_prefix: str = "Asset",
) -> AssetUniverse:
    """
    Generate a deterministic 10-15 asset universe with optional user overrides.

    If `provided_assets` is supplied, it will be normalized into `Asset` objects
    and used as the leading slice of the universe. Synthetic assets fill any
    remaining slots.
    """

    spec = _scenario_spec(scenario)
    rng = _rng(seed)
    supplied = []
    if provided_assets is not None:
        supplied = [_coerce_asset(asset, idx) for idx, asset in enumerate(provided_assets)]

    if n_assets is None:
        if supplied:
            n_assets = min(15, max(10, len(supplied)))
        else:
            n_assets = 12
    if not 10 <= int(n_assets) <= 15:
        raise ValueError("n_assets must be in [10, 15].")
    n_assets = int(n_assets)

    if len(supplied) > n_assets:
        raise ValueError("provided_assets cannot exceed n_assets.")

    assets: list[Asset] = list(supplied)
    for index in range(len(assets), n_assets):
        assets.append(_make_asset(rng, index, spec, asset_prefix))

    # Re-seed the covariance generator so asset composition fully determines
    # the final matrix while keeping the outer call deterministic.
    cov_seed = seed + 997
    cov_matrix = build_covariance_matrix(
        assets,
        seed=cov_seed,
        correlation_strength=correlation_strength,
        factor_rank=factor_rank,
    )
    return AssetUniverse(
        assets=assets,
        cov_matrix=cov_matrix,
        scenario=(scenario or "balanced").lower(),
        seed=int(seed),
        correlation_strength=float(correlation_strength),
    )


def generate_assets(
    n_assets: int = 12,
    *,
    seed: int = 42,
    scenario: str = "balanced",
    provided_assets: Iterable[Asset | Mapping[str, Any]] | None = None,
    correlation_strength: float = 0.55,
    factor_rank: int = 3,
    asset_prefix: str = "Asset",
) -> dict[str, Any]:
    """
    Backward-friendly wrapper returning plain Python types.
    """

    universe = generate_asset_universe(
        n_assets=n_assets,
        seed=seed,
        scenario=scenario,
        provided_assets=provided_assets,
        correlation_strength=correlation_strength,
        factor_rank=factor_rank,
        asset_prefix=asset_prefix,
    )
    return serialize_for_api(universe)


def asset_to_dict(asset: Asset) -> dict[str, Any]:
    return asdict(asset)


def assets_to_dicts(assets: Iterable[Asset]) -> list[dict[str, Any]]:
    return [asset_to_dict(asset) for asset in assets]


def universe_to_dict(universe: AssetUniverse) -> dict[str, Any]:
    return {
        "assets": assets_to_dicts(universe.assets),
        "cov_matrix": universe.cov_matrix.tolist(),
        "scenario": universe.scenario,
        "seed": universe.seed,
        "correlation_strength": universe.correlation_strength,
    }


def serialize_for_api(
    universe: AssetUniverse | Mapping[str, Any],
) -> dict[str, Any]:
    """
    Serialize the generated universe into API-friendly JSON primitives.
    """

    if isinstance(universe, AssetUniverse):
        return universe_to_dict(universe)

    if "assets" not in universe or "cov_matrix" not in universe:
        raise ValueError("serialize_for_api expects an AssetUniverse or a mapping with assets and cov_matrix.")

    assets = universe["assets"]
    cov_matrix = universe["cov_matrix"]
    if isinstance(cov_matrix, np.ndarray):
        cov_matrix = cov_matrix.tolist()
    return {
        "assets": [asset_to_dict(asset) if isinstance(asset, Asset) else dict(asset) for asset in assets],
        "cov_matrix": cov_matrix,
        "scenario": str(universe.get("scenario", "balanced")),
        "seed": int(universe.get("seed", 42)),
        "correlation_strength": float(universe.get("correlation_strength", 0.55)),
    }


def _demo() -> None:
    universe = generate_asset_universe(seed=7, scenario="high_growth")
    print(f"Assets: {len(universe.assets)}")
    print(f"PSD covariance shape: {universe.cov_matrix.shape}")
    print(f"Min eigenvalue: {np.min(np.linalg.eigvalsh(universe.cov_matrix)):.3e}")


if __name__ == "__main__":
    _demo()
