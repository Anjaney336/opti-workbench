"""
Bayesian and Black-Litterman expected-return estimation helpers.

These functions sit upstream of the optimizer and only modify the expected
return vector fed into the existing objective. The optimization structure
itself remains unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 1e-8)
    total = float(np.sum(clipped))
    if total <= 1e-12:
        return np.full_like(clipped, 1.0 / max(len(clipped), 1), dtype=float)
    return clipped / total


def _rescale_to_reference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref))
    cand_mean = float(np.mean(cand))
    cand_std = float(np.std(cand))
    centered = cand - cand_mean
    if cand_std <= 1e-10 or ref_std <= 1e-10:
        return np.full_like(cand, ref_mean, dtype=float)
    scaled = ref_mean + centered * (ref_std / cand_std)
    lower = float(np.min(ref) - 0.35 * ref_std)
    upper = float(np.max(ref) + 0.35 * ref_std)
    return np.clip(scaled, lower, upper)


def _sample_mean(history: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    if history is None:
        return np.asarray(fallback, dtype=float)
    arr = np.asarray(history, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(fallback):
        return np.asarray(fallback, dtype=float)
    return np.mean(arr, axis=0)


def market_implied_prior(
    cov: np.ndarray,
    *,
    market_weights: np.ndarray,
    raw_mu: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    weights = _normalize_positive(market_weights)
    pi = float(max(risk_aversion, 1e-4)) * cov @ weights
    return _rescale_to_reference(pi, raw_mu)


def bayesian_expected_returns(
    raw_mu: np.ndarray,
    cov: np.ndarray,
    *,
    liquidity: np.ndarray,
    cost: np.ndarray,
    historical_returns: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    prior_strength: float = 0.55,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Shrink empirical views toward a market-implied prior.
    """
    raw_mu = np.asarray(raw_mu, dtype=float)
    empirical_mu = _sample_mean(historical_returns, raw_mu)
    investability_score = _normalize_positive(np.asarray(liquidity, dtype=float) / np.maximum(np.asarray(cost, dtype=float), 1e-6))
    prior_mu = market_implied_prior(
        cov,
        market_weights=investability_score,
        raw_mu=raw_mu,
        risk_aversion=risk_aversion,
    )
    shrinkage = float(np.clip(prior_strength, 0.0, 0.95))
    posterior = shrinkage * prior_mu + (1.0 - shrinkage) * empirical_mu
    return posterior, {
        "engine": "BAYESIAN",
        "prior_mu": prior_mu,
        "empirical_mu": empirical_mu,
        "posterior_mu": posterior,
        "prior_strength": shrinkage,
        "market_weights": investability_score,
    }


def black_litterman_expected_returns(
    raw_mu: np.ndarray,
    cov: np.ndarray,
    *,
    liquidity: np.ndarray,
    cost: np.ndarray,
    historical_returns: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    prior_strength: float = 0.55,
    tau: float = 0.08,
    view_confidence: float = 0.65,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Absolute-view Black-Litterman with a Bayesian-smoothed view vector.
    """
    raw_mu = np.asarray(raw_mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    empirical_mu = _sample_mean(historical_returns, raw_mu)
    investability_score = _normalize_positive(np.asarray(liquidity, dtype=float) / np.maximum(np.asarray(cost, dtype=float), 1e-6))
    prior_mu = market_implied_prior(
        cov,
        market_weights=investability_score,
        raw_mu=raw_mu,
        risk_aversion=risk_aversion,
    )

    shrinkage = float(np.clip(prior_strength, 0.0, 0.95))
    bayesian_view_mu = shrinkage * prior_mu + (1.0 - shrinkage) * empirical_mu
    n_assets = len(raw_mu)
    p_matrix = np.eye(n_assets, dtype=float)
    q_vector = bayesian_view_mu
    tau_value = float(np.clip(tau, 1e-4, 1.0))
    confidence = float(np.clip(view_confidence, 0.05, 0.99))

    tau_cov = tau_value * cov
    omega_diag = np.maximum(np.diag(tau_cov), 1e-8) * ((1.0 - confidence) / confidence)
    omega = np.diag(np.maximum(omega_diag, 1e-8))
    inv_tau_cov = np.linalg.pinv(tau_cov)
    inv_omega = np.linalg.pinv(omega)
    posterior_cov = np.linalg.pinv(inv_tau_cov + p_matrix.T @ inv_omega @ p_matrix)
    posterior_mu = posterior_cov @ (inv_tau_cov @ prior_mu + p_matrix.T @ inv_omega @ q_vector)
    posterior_mu = _rescale_to_reference(posterior_mu, raw_mu)

    return posterior_mu, {
        "engine": "BLACK_LITTERMAN",
        "prior_mu": prior_mu,
        "empirical_mu": empirical_mu,
        "bayesian_view_mu": bayesian_view_mu,
        "posterior_mu": posterior_mu,
        "prior_strength": shrinkage,
        "tau": tau_value,
        "view_confidence": confidence,
        "market_weights": investability_score,
    }


def estimate_expected_returns(
    engine: str,
    raw_mu: np.ndarray,
    cov: np.ndarray,
    *,
    liquidity: np.ndarray,
    cost: np.ndarray,
    historical_returns: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    prior_strength: float = 0.55,
    tau: float = 0.08,
    view_confidence: float = 0.65,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(engine or "BLACK_LITTERMAN").strip().upper()
    if mode == "RAW":
        base = np.asarray(raw_mu, dtype=float)
        return base, {
            "engine": "RAW",
            "prior_mu": base,
            "empirical_mu": _sample_mean(historical_returns, base),
            "posterior_mu": base,
            "market_weights": _normalize_positive(np.asarray(liquidity, dtype=float) / np.maximum(np.asarray(cost, dtype=float), 1e-6)),
        }
    if mode == "BAYESIAN":
        return bayesian_expected_returns(
            raw_mu,
            cov,
            liquidity=liquidity,
            cost=cost,
            historical_returns=historical_returns,
            risk_aversion=risk_aversion,
            prior_strength=prior_strength,
        )
    return black_litterman_expected_returns(
        raw_mu,
        cov,
        liquidity=liquidity,
        cost=cost,
        historical_returns=historical_returns,
        risk_aversion=risk_aversion,
        prior_strength=prior_strength,
        tau=tau,
        view_confidence=view_confidence,
    )


__all__ = [
    "estimate_expected_returns",
    "bayesian_expected_returns",
    "black_litterman_expected_returns",
    "market_implied_prior",
]
