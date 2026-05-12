"""
Particle Swarm Optimization implementation (custom, no libraries).
Hybrid approach: PSO with constraint projection and warm start from QP solution.
"""

import numpy as np
from typing import Dict, Callable, List, Tuple
import copy


def population_diversity(population: np.ndarray) -> float:
    """Mean Euclidean distance to the population centroid."""
    centroid = np.mean(population, axis=0)
    return float(np.mean(np.linalg.norm(population - centroid, axis=1)))


class ParticleSwarmOptimizer:
    """
    Custom PSO implementation for constrained portfolio optimization.
    
    Features:
    - Velocity update with inertia weight decay
    - Constraint projection after each step
    - Warm start from initial solution
    - Convergence tracking
    - Particle history recording
    """
    
    def __init__(self,
                 objective_fn: Callable,
                 constraint_proj_fn: Callable,
                 n_particles: int = 30,
                 n_dimensions: int = 40,
                 w_init: float = 0.9,
                 w_final: float = 0.4,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 seed: int = 42):
        """
        Initialize PSO.
        
        Parameters:
        -----------
        objective_fn : callable
            Function to maximize: f(x) -> float
        constraint_proj_fn : callable
            Constraint projection: f(x) -> x_feasible
        n_particles : int
            Population size (30-50 typical)
        n_dimensions : int
            Problem dimension
        w_init : float
            Initial inertia weight
        w_final : float
            Final inertia weight
        c1, c2 : float
            Cognitive and social parameters
        seed : int
            Random seed
        """
        self.objective_fn = objective_fn
        self.constraint_proj_fn = constraint_proj_fn
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.w_init = w_init
        self.w_final = w_final
        self.c1 = c1
        self.c2 = c2
        
        np.random.seed(seed)
        
        # Initialize particles
        self.positions = np.random.dirichlet(np.ones(n_dimensions), size=n_particles)
        self.velocities = np.random.uniform(-0.1, 0.1, size=(n_particles, n_dimensions))
        
        # Project to feasible region
        for i in range(n_particles):
            self.positions[i] = self.constraint_proj_fn(self.positions[i])
        
        # Evaluate initial fitness
        self.fitness = np.array([self.objective_fn(p) for p in self.positions])
        
        # Best personal positions
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.copy(self.fitness)
        
        # Global best
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest_position = np.copy(self.pbest_positions[best_idx])
        self.gbest_fitness = self.pbest_fitness[best_idx]
        
        # History
        self.history = {
            'best_fitness': [self.gbest_fitness],
            'mean_fitness': [np.mean(self.fitness)],
            'std_fitness': [np.std(self.fitness)],
            'gbest_position': [np.copy(self.gbest_position)],
            'population_diversity': [population_diversity(self.positions)],
        }
    
    def warm_start(self, x0: np.ndarray) -> None:
        """
        Initialize particles around a given solution (warm start).
        
        Parameters:
        -----------
        x0 : np.ndarray
            Initial solution (e.g., from QP)
        """
        # Start from a feasible anchor point.
        self.positions[0] = self.constraint_proj_fn(np.copy(x0))
        
        # Add noise to other particles around x0
        for i in range(1, self.n_particles):
            noise = np.random.normal(0, 0.05, self.n_dimensions)
            candidate = x0 + noise
            candidate = np.clip(candidate, 0, 1)
            self.positions[i] = self.constraint_proj_fn(candidate)
        
        # Re-evaluate
        self.fitness = np.array([self.objective_fn(p) for p in self.positions])
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.copy(self.fitness)
        
        # Update global best
        best_idx = np.argmax(self.pbest_fitness)
        self.gbest_position = np.copy(self.pbest_positions[best_idx])
        self.gbest_fitness = self.pbest_fitness[best_idx]
        self.history['best_fitness'][0] = self.gbest_fitness
    
    def step(self, iteration: int, n_iterations: int) -> None:
        """
        Perform one PSO iteration.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        n_iterations : int
            Total iterations for decay schedule
        """
        # Inertia weight decay
        w = self.w_init - (self.w_init - self.w_final) * (iteration / n_iterations)
        
        for i in range(self.n_particles):
            # Random coefficients
            r1 = np.random.random(self.n_dimensions)
            r2 = np.random.random(self.n_dimensions)
            
            # Velocity update
            self.velocities[i] = (
                w * self.velocities[i] +
                self.c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                self.c2 * r2 * (self.gbest_position - self.positions[i])
            )
            
            # Clip velocity for stability
            self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)
            
            # Position update
            self.positions[i] = self.positions[i] + self.velocities[i]
            
            # Project to feasible region
            self.positions[i] = self.constraint_proj_fn(self.positions[i])
            
            # Evaluate
            fitness = self.objective_fn(self.positions[i])
            
            # Update personal best
            if fitness > self.pbest_fitness[i]:
                self.pbest_fitness[i] = fitness
                self.pbest_positions[i] = np.copy(self.positions[i])
                
                # Update global best
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = np.copy(self.positions[i])
        
        # Record history
        self.history['best_fitness'].append(self.gbest_fitness)
        self.history['mean_fitness'].append(np.mean(self.pbest_fitness))
        self.history['std_fitness'].append(np.std(self.pbest_fitness))
        self.history['gbest_position'].append(np.copy(self.gbest_position))
        self.history['population_diversity'].append(population_diversity(self.positions))
    
    def optimize(self, n_iterations: int = 200, verbose: bool = False) -> Dict:
        """
        Run PSO optimization.
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations
        verbose : bool
            Print progress
        
        Returns:
        --------
        dict : Results including best position and fitness
        """
        for iteration in range(n_iterations):
            self.step(iteration, n_iterations)
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"  PSO Iteration {iteration+1:3d}: "
                      f"Best={self.gbest_fitness:.6f}, "
                      f"Mean={self.history['mean_fitness'][-1]:.6f}")
        
        return {
            'x': self.gbest_position,
            'fitness': self.gbest_fitness,
            'n_iterations': n_iterations,
            'history': self.history,
            'n_particles': self.n_particles
        }
    
    def get_population(self) -> np.ndarray:
        """Return current particle positions."""
        return np.copy(self.positions)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Return global best position and fitness."""
        return np.copy(self.gbest_position), self.gbest_fitness


def optimize_with_pso(objective_fn: Callable,
                     constraint_proj_fn: Callable,
                     x0: np.ndarray = None,
                     n_dimensions: int = None,
                     n_particles: int = 30,
                     n_iterations: int = 200,
                     verbose: bool = False,
                     seed: int = 42) -> Dict:
    """
    High-level interface for PSO optimization.
    
    Parameters:
    -----------
    objective_fn : callable
        Objective function f(x) -> float (to maximize)
    constraint_proj_fn : callable
        Constraint projection function
    x0 : np.ndarray, optional
        Warm start point (e.g., from QP solution)
    n_dimensions : int, optional
        Problem dimension when no warm start is supplied
    n_particles : int
        Swarm size
    n_iterations : int
        Number of PSO iterations
    verbose : bool
        Print progress
    seed : int
        Random seed
    
    Returns:
    --------
    dict : Optimization results
    """
    if x0 is not None:
        n_dims = len(x0)
    elif n_dimensions is not None:
        n_dims = n_dimensions
    else:
        n_dims = 40
    
    pso = ParticleSwarmOptimizer(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        n_particles=n_particles,
        n_dimensions=n_dims,
        seed=seed
    )
    
    if x0 is not None:
        pso.warm_start(x0)
    
    if verbose:
        print(f"\n{'PSO OPTIMIZATION':^50}")
        print(f"  Particles: {n_particles}, Iterations: {n_iterations}")
    
    result = pso.optimize(n_iterations, verbose=verbose)
    
    if verbose:
        improvement = result['fitness'] - pso.history['best_fitness'][0]
        print(f"  Final best fitness: {result['fitness']:.6f}")
        print(f"  Improvement: {improvement:+.6f}")
    
    return result


from typing import Tuple

if __name__ == '__main__':
    from data import generate_portfolio_data
    from utils import project_constraints
    
    data = generate_portfolio_data(n_assets=20, seed=42)
    
    # Define objective: expected return (simplified)
    def objective(x):
        return np.dot(x, data['mu'])
    
    # Define constraint projection
    def constraint_proj(x):
        return project_constraints(x, data['c'], data['B'], data['alpha'])
    
    # Warm start
    x0 = np.ones(20) / 20
    
    # Optimize
    result = optimize_with_pso(
        objective_fn=objective,
        constraint_proj_fn=constraint_proj,
        x0=x0,
        n_particles=20,
        n_iterations=100,
        verbose=True,
        seed=42
    )
    
    print(f"\nBest solution: {result['x'][:5]}")
    print(f"Best fitness: {result['fitness']:.6f}")
