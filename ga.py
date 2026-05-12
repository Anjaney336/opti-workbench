"""
Real-valued Genetic Algorithm for quadratic programs.

Each individual is x in R^D. The minimized fitness is the penalized objective

    f_pen(x) = 0.5 x^T Q x + c^T x + rho * penalty(x)

Selection methods:
    1. Roulette wheel selection
    2. Tournament selection

Crossover:
    x_child = alpha x1 + (1 - alpha) x2, alpha ~ U(0, 1)

Mutation:
    x <- x + epsilon, epsilon ~ N(0, sigma^2 I)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qp_solver import QuadraticProgram, constraint_violation_norm, is_feasible, objective_value


Array = np.ndarray


def population_diversity(population: Array) -> float:
    centroid = np.mean(population, axis=0)
    return float(np.mean(np.linalg.norm(population - centroid, axis=1)))


@dataclass
class GAConfig:
    population_size: int = 64
    generations: int = 120
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.12
    elitism_count: int = 4
    selection_method: str = "tournament"
    tournament_size: int = 3
    penalty_weight: float = 500.0
    seed: int = 42


class GeneticAlgorithm:
    """
    Real-coded GA with elitism, configurable selection, arithmetic crossover,
    and Gaussian mutation.
    """

    def __init__(
        self,
        problem: QuadraticProgram,
        lower_bound: Array,
        upper_bound: Array,
        config: GAConfig,
    ) -> None:
        self.problem = problem
        self.lower_bound = np.asarray(lower_bound, dtype=float)
        self.upper_bound = np.asarray(upper_bound, dtype=float)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        if self.lower_bound.shape != (problem.dimension,) or self.upper_bound.shape != (problem.dimension,):
            raise ValueError("Search bounds must match the problem dimension.")
        if config.selection_method not in {"roulette", "tournament"}:
            raise ValueError("selection_method must be 'roulette' or 'tournament'.")
        if not 0 <= config.elitism_count < config.population_size:
            raise ValueError("elitism_count must be in [0, population_size).")

        self.population = self.rng.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(config.population_size, problem.dimension),
        )
        self.initial_population = np.copy(self.population)

        self.scores = np.full(config.population_size, np.inf)
        self.objectives = np.full(config.population_size, np.inf)
        self.penalties = np.full(config.population_size, np.inf)

        self.best_position = np.copy(self.population[0])
        self.best_score = np.inf
        self.best_objective = np.inf
        self.best_penalty = np.inf

        self.best_feasible_position: Array | None = None
        self.best_feasible_objective = np.inf

        self.history = {
            "best_penalized_value": [],
            "best_objective_value": [],
            "best_penalty": [],
            "best_feasible_objective": [],
            "position_error": [],
            "population_diversity": [],
            "mean_penalized_value": [],
        }

        self._evaluate_population()
        self._record_history()

    def _penalty_value(self, x: Array) -> float:
        return self.config.penalty_weight * constraint_violation_norm(self.problem, x)

    def _augmented_objective(self, x: Array) -> tuple[float, float, float]:
        raw_objective = objective_value(self.problem, x)
        penalty = self._penalty_value(x)
        return raw_objective + penalty, raw_objective, penalty

    def _evaluate_population(self) -> None:
        for idx in range(self.config.population_size):
            score, raw_objective, penalty = self._augmented_objective(self.population[idx])
            self.scores[idx] = score
            self.objectives[idx] = raw_objective
            self.penalties[idx] = penalty

            if score < self.best_score:
                self.best_score = score
                self.best_objective = raw_objective
                self.best_penalty = penalty
                self.best_position = np.copy(self.population[idx])

            if is_feasible(self.problem, self.population[idx]) and raw_objective < self.best_feasible_objective:
                self.best_feasible_objective = raw_objective
                self.best_feasible_position = np.copy(self.population[idx])

    def _record_history(self) -> None:
        best_position = self.best_feasible_position if self.best_feasible_position is not None else self.best_position
        best_objective = (
            self.best_feasible_objective if self.best_feasible_position is not None else self.best_objective
        )

        self.history["best_penalized_value"].append(float(self.best_score))
        self.history["best_objective_value"].append(float(best_objective))
        self.history["best_penalty"].append(float(self.best_penalty))
        self.history["best_feasible_objective"].append(
            float(self.best_feasible_objective) if self.best_feasible_position is not None else np.nan
        )
        self.history["population_diversity"].append(population_diversity(self.population))
        self.history["mean_penalized_value"].append(float(np.mean(self.scores)))

        if self.problem.x_true is None:
            self.history["position_error"].append(np.nan)
        else:
            self.history["position_error"].append(float(np.linalg.norm(best_position - self.problem.x_true)))

    def _roulette_select(self) -> Array:
        shifted = self.scores - np.min(self.scores)
        weights = 1.0 / (1.0 + shifted)
        probabilities = weights / np.sum(weights)
        index = self.rng.choice(self.config.population_size, p=probabilities)
        return np.copy(self.population[index])

    def _tournament_select(self) -> Array:
        contestants = self.rng.choice(
            self.config.population_size,
            size=self.config.tournament_size,
            replace=False,
        )
        winner = contestants[np.argmin(self.scores[contestants])]
        return np.copy(self.population[winner])

    def _select_parent(self) -> Array:
        if self.config.selection_method == "roulette":
            return self._roulette_select()
        return self._tournament_select()

    def _arithmetic_crossover(self, parent_a: Array, parent_b: Array) -> tuple[Array, Array]:
        alpha = self.rng.random(self.problem.dimension)
        child_a = alpha * parent_a + (1.0 - alpha) * parent_b
        child_b = alpha * parent_b + (1.0 - alpha) * parent_a
        return child_a, child_b

    def _gaussian_mutation(self, individual: Array) -> Array:
        mask = self.rng.random(self.problem.dimension) < self.config.mutation_rate
        if np.any(mask):
            individual = np.copy(individual)
            individual[mask] += self.rng.normal(
                loc=0.0,
                scale=self.config.mutation_sigma,
                size=np.count_nonzero(mask),
            )
        return np.clip(individual, self.lower_bound, self.upper_bound)

    def step(self) -> None:
        elite_indices = np.argsort(self.scores)[: self.config.elitism_count]
        next_population = [np.copy(self.population[idx]) for idx in elite_indices]

        while len(next_population) < self.config.population_size:
            parent_a = self._select_parent()
            parent_b = self._select_parent()

            if self.rng.random() < self.config.crossover_rate:
                child_a, child_b = self._arithmetic_crossover(parent_a, parent_b)
            else:
                child_a, child_b = np.copy(parent_a), np.copy(parent_b)

            child_a = self._gaussian_mutation(child_a)
            next_population.append(child_a)

            if len(next_population) < self.config.population_size:
                child_b = self._gaussian_mutation(child_b)
                next_population.append(child_b)

        self.population = np.asarray(next_population, dtype=float)
        self._evaluate_population()
        self._record_history()

    def solve(self, verbose: bool = False) -> dict[str, object]:
        for generation in range(self.config.generations):
            self.step()

            if verbose and (generation + 1) % 20 == 0:
                print(
                    f"  ga_gen={generation + 1:3d} "
                    f"best_aug={self.history['best_penalized_value'][-1]:.6e} "
                    f"best_obj={self.history['best_objective_value'][-1]:.6e}"
                )

        if self.best_feasible_position is not None:
            solution = self.best_feasible_position
            objective = self.best_feasible_objective
            feasible = True
            selected_from = "best_feasible"
        else:
            solution = self.best_position
            objective = self.best_objective
            feasible = is_feasible(self.problem, solution)
            selected_from = "best_penalized"

        ranking = np.argsort(self.scores)

        return {
            "x": np.copy(solution),
            "objective": float(objective),
            "penalized_objective": float(self.best_score),
            "penalty": float(self.best_penalty),
            "feasible": bool(feasible),
            "history": self.history,
            "selected_from": selected_from,
            "config": self.config,
            "population": np.copy(self.population),
            "ranked_population": np.copy(self.population[ranking]),
            "scores": np.copy(self.scores),
            "initial_population": np.copy(self.initial_population),
            "initial_diversity": population_diversity(self.initial_population),
            "final_diversity": population_diversity(self.population),
        }


def solve_with_ga(
    problem: QuadraticProgram,
    lower_bound: Array,
    upper_bound: Array,
    config: GAConfig,
    verbose: bool = False,
) -> dict[str, object]:
    optimizer = GeneticAlgorithm(
        problem=problem,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        config=config,
    )
    return optimizer.solve(verbose=verbose)
