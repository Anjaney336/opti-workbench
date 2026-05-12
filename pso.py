"""
Particle Swarm Optimization for quadratic programs.

The implemented update equations are

    v_i(t+1) = w(t) v_i(t)
               + c1 r1 (pbest_i - x_i(t))
               + c2 r2 (gbest - x_i(t))

    x_i(t+1) = x_i(t) + v_i(t+1)

with optional Clerc constriction

    v_i(t+1) <- chi v_i(t+1)

where x_i, v_i, pbest_i, gbest in R^D and r1, r2 ~ U(0, 1)^D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qp_solver import QuadraticProgram, constraint_violation_norm, is_feasible, objective_value


Array = np.ndarray


def population_diversity(population: Array) -> float:
    """
    Mean distance to the population centroid.
    """
    centroid = np.mean(population, axis=0)
    return float(np.mean(np.linalg.norm(population - centroid, axis=1)))


def clerc_constriction_factor(c1: float, c2: float) -> float:
    """
    Clerc constriction factor for phi = c1 + c2 > 4.
    """
    phi = c1 + c2
    if phi <= 4.0:
        raise ValueError("Clerc constriction requires c1 + c2 > 4.")
    return 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))


@dataclass
class PSOConfig:
    n_particles: int = 48
    iterations: int = 160
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    c1: float = 2.05
    c2: float = 2.05
    use_constriction: bool = True
    constriction: float | None = None
    penalty_weight: float = 500.0
    velocity_clamp_fraction: float = 0.25
    stochastic: bool = True
    seed: int = 42


class ParticleSwarmOptimizer:
    """
    Minimization-oriented PSO with quadratic constraint penalties.
    """

    def __init__(
        self,
        problem: QuadraticProgram,
        lower_bound: Array,
        upper_bound: Array,
        config: PSOConfig,
        initial_positions: Array | None = None,
    ) -> None:
        self.problem = problem
        self.lower_bound = np.asarray(lower_bound, dtype=float)
        self.upper_bound = np.asarray(upper_bound, dtype=float)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        if self.lower_bound.shape != (problem.dimension,) or self.upper_bound.shape != (problem.dimension,):
            raise ValueError("Search bounds must match the problem dimension.")

        self.span = self.upper_bound - self.lower_bound
        self.velocity_limit = config.velocity_clamp_fraction * self.span
        self.chi = config.constriction
        if config.use_constriction and self.chi is None:
            self.chi = clerc_constriction_factor(config.c1, config.c2)
        if not config.use_constriction:
            self.chi = 1.0

        self.positions = self._initialize_positions(initial_positions)
        self.initial_positions = np.copy(self.positions)
        self.velocities = self.rng.uniform(
            low=-self.velocity_limit,
            high=self.velocity_limit,
            size=(config.n_particles, problem.dimension),
        )

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(config.n_particles, np.inf)
        self.personal_best_objectives = np.full(config.n_particles, np.inf)

        self.global_best_position = np.copy(self.positions[0])
        self.global_best_score = np.inf
        self.global_best_objective = np.inf
        self.global_best_penalty = np.inf

        self.best_feasible_position: Array | None = None
        self.best_feasible_objective = np.inf

        self.history = {
            "best_penalized_value": [],
            "best_objective_value": [],
            "best_penalty": [],
            "best_feasible_objective": [],
            "position_error": [],
            "inertia_weight": [],
            "mean_penalized_value": [],
            "population_diversity": [],
        }

        self._evaluate_and_update_bests()
        self._record_history(self._inertia_weight(0))

    def _initialize_positions(self, initial_positions: Array | None) -> Array:
        if initial_positions is None:
            positions = self.rng.uniform(
                low=self.lower_bound,
                high=self.upper_bound,
                size=(self.config.n_particles, self.problem.dimension),
            )
            return positions

        initial_positions = np.asarray(initial_positions, dtype=float)
        if initial_positions.ndim != 2 or initial_positions.shape[1] != self.problem.dimension:
            raise ValueError("initial_positions must have shape (n, dimension).")

        positions = np.empty((self.config.n_particles, self.problem.dimension), dtype=float)
        n_seeded = min(self.config.n_particles, initial_positions.shape[0])
        positions[:n_seeded] = initial_positions[:n_seeded]

        if n_seeded < self.config.n_particles:
            positions[n_seeded:] = self.rng.uniform(
                low=self.lower_bound,
                high=self.upper_bound,
                size=(self.config.n_particles - n_seeded, self.problem.dimension),
            )

        return np.clip(positions, self.lower_bound, self.upper_bound)

    def _inertia_weight(self, iteration: int) -> float:
        """
        Linear schedule:

            w(t) = w_max - (w_max - w_min) * t / (T - 1)
        """
        if self.config.iterations <= 1:
            return self.config.inertia_end
        fraction = iteration / (self.config.iterations - 1)
        return self.config.inertia_start + fraction * (self.config.inertia_end - self.config.inertia_start)

    def _sample_random_vectors(self) -> tuple[Array, Array]:
        if self.config.stochastic:
            return self.rng.random(self.problem.dimension), self.rng.random(self.problem.dimension)
        return np.full(self.problem.dimension, 0.5), np.full(self.problem.dimension, 0.5)

    def _penalty_value(self, x: Array) -> float:
        return self.config.penalty_weight * constraint_violation_norm(self.problem, x)

    def _augmented_objective(self, x: Array) -> tuple[float, float, float]:
        raw_objective = objective_value(self.problem, x)
        penalty = self._penalty_value(x)
        return raw_objective + penalty, raw_objective, penalty

    def _evaluate_and_update_bests(self) -> Array:
        penalized_scores = np.empty(self.config.n_particles)

        for particle in range(self.config.n_particles):
            penalized, raw_objective, penalty = self._augmented_objective(self.positions[particle])
            penalized_scores[particle] = penalized

            if penalized < self.personal_best_scores[particle]:
                self.personal_best_scores[particle] = penalized
                self.personal_best_objectives[particle] = raw_objective
                self.personal_best_positions[particle] = np.copy(self.positions[particle])

            if penalized < self.global_best_score:
                self.global_best_score = penalized
                self.global_best_objective = raw_objective
                self.global_best_penalty = penalty
                self.global_best_position = np.copy(self.positions[particle])

            if is_feasible(self.problem, self.positions[particle]) and raw_objective < self.best_feasible_objective:
                self.best_feasible_objective = raw_objective
                self.best_feasible_position = np.copy(self.positions[particle])

        return penalized_scores

    def _record_history(self, inertia_weight: float) -> None:
        best_position = self.best_feasible_position if self.best_feasible_position is not None else self.global_best_position
        best_objective = (
            self.best_feasible_objective if self.best_feasible_position is not None else self.global_best_objective
        )

        self.history["best_penalized_value"].append(self.global_best_score)
        self.history["best_objective_value"].append(best_objective)
        self.history["best_penalty"].append(self.global_best_penalty)
        self.history["best_feasible_objective"].append(
            self.best_feasible_objective if self.best_feasible_position is not None else np.nan
        )
        self.history["mean_penalized_value"].append(float(np.mean(self.personal_best_scores)))
        self.history["inertia_weight"].append(inertia_weight)
        self.history["population_diversity"].append(population_diversity(self.positions))

        if self.problem.x_true is None:
            self.history["position_error"].append(np.nan)
        else:
            self.history["position_error"].append(float(np.linalg.norm(best_position - self.problem.x_true)))

    def step(self, iteration: int) -> None:
        """
        Advance one iteration using the exact PSO equations.
        """
        inertia_weight = self._inertia_weight(iteration)

        for particle in range(self.config.n_particles):
            r1, r2 = self._sample_random_vectors()

            cognitive = self.config.c1 * r1 * (
                self.personal_best_positions[particle] - self.positions[particle]
            )
            social = self.config.c2 * r2 * (self.global_best_position - self.positions[particle])

            self.velocities[particle] = self.chi * (
                inertia_weight * self.velocities[particle] + cognitive + social
            )
            self.velocities[particle] = np.clip(
                self.velocities[particle],
                -self.velocity_limit,
                self.velocity_limit,
            )

            self.positions[particle] = self.positions[particle] + self.velocities[particle]
            self.positions[particle] = np.clip(self.positions[particle], self.lower_bound, self.upper_bound)

        self._evaluate_and_update_bests()
        self._record_history(inertia_weight)

    def solve(self, verbose: bool = False) -> dict[str, object]:
        for iteration in range(self.config.iterations):
            self.step(iteration)

            if verbose and (iteration + 1) % 20 == 0:
                print(
                    f"  pso_iter={iteration + 1:3d} "
                    f"best_aug={self.history['best_penalized_value'][-1]:.6e} "
                    f"best_obj={self.history['best_objective_value'][-1]:.6e}"
                )

        if self.best_feasible_position is not None:
            solution = self.best_feasible_position
            objective = self.best_feasible_objective
            feasible = True
            selected_from = "best_feasible"
        else:
            solution = self.global_best_position
            objective = self.global_best_objective
            feasible = is_feasible(self.problem, solution)
            selected_from = "best_penalized"

        return {
            "x": np.copy(solution),
            "objective": float(objective),
            "penalized_objective": float(self.global_best_score),
            "penalty": float(self.global_best_penalty),
            "feasible": bool(feasible),
            "history": self.history,
            "selected_from": selected_from,
            "config": self.config,
            "population": np.copy(self.positions),
            "initial_population": np.copy(self.initial_positions),
            "initial_diversity": population_diversity(self.initial_positions),
            "final_diversity": population_diversity(self.positions),
        }


def solve_with_pso(
    problem: QuadraticProgram,
    lower_bound: Array,
    upper_bound: Array,
    config: PSOConfig,
    initial_positions: Array | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    optimizer = ParticleSwarmOptimizer(
        problem=problem,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        config=config,
        initial_positions=initial_positions,
    )
    return optimizer.solve(verbose=verbose)
