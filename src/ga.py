"""
Genetic Algorithm implementation (custom, no libraries).
Real-coded GA with tournament selection, blend crossover, and Gaussian mutation.
"""

import numpy as np
from typing import Dict, Callable, List, Tuple


def population_diversity(population: np.ndarray) -> float:
    """Mean Euclidean distance to the population centroid."""
    centroid = np.mean(population, axis=0)
    return float(np.mean(np.linalg.norm(population - centroid, axis=1)))


class GeneticAlgorithm:
    """
    Real-coded Genetic Algorithm for constrained portfolio optimization.
    
    Features:
    - Tournament selection
    - Blend crossover (BLX-α)
    - Gaussian mutation with adaptive rates
    - Elitism
    - Constraint projection after genetic operators
    - Convergence tracking
    """
    
    def __init__(self,
                 objective_fn: Callable,
                 constraint_proj_fn: Callable,
                 population_size: int = 50,
                 n_dimensions: int = 40,
                 tournament_size: int = 3,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_sigma: float = 0.05,
                 elite_fraction: float = 0.15,
                 seed: int = 42):
        """
        Initialize Genetic Algorithm.
        
        Parameters:
        -----------
        objective_fn : callable
            Function to maximize: f(x) -> float
        constraint_proj_fn : callable
            Constraint projection: f(x) -> x_feasible
        population_size : int
            Population size (40-100 typical)
        n_dimensions : int
            Problem dimension
        tournament_size : int
            Tournament size for selection
        crossover_rate : float
            Probability of crossover
        mutation_rate : float
            Mutation rate (fraction of genes mutated)
        mutation_sigma : float
            Standard deviation of Gaussian mutation
        elite_fraction : float
            Fraction of population to preserve (elitism)
        seed : int
            Random seed
        """
        self.objective_fn = objective_fn
        self.constraint_proj_fn = constraint_proj_fn
        self.population_size = population_size
        self.n_dimensions = n_dimensions
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elite_fraction = elite_fraction
        self.n_elite = max(1, int(population_size * elite_fraction))
        
        np.random.seed(seed)
        
        # Initialize population
        self.population = np.random.dirichlet(np.ones(n_dimensions), size=population_size)
        
        # Project to feasible region
        for i in range(population_size):
            self.population[i] = self.constraint_proj_fn(self.population[i])
        
        # Evaluate fitness
        self.fitness = np.array([self.objective_fn(ind) for ind in self.population])
        
        # Best individual
        best_idx = np.argmax(self.fitness)
        self.best_individual = np.copy(self.population[best_idx])
        self.best_fitness = self.fitness[best_idx]
        
        # History
        self.history = {
            'best_fitness': [self.best_fitness],
            'mean_fitness': [np.mean(self.fitness)],
            'std_fitness': [np.std(self.fitness)],
            'best_individual': [np.copy(self.best_individual)],
            'population_diversity': [population_diversity(self.population)],
        }
    
    def warm_start(self, x0: np.ndarray) -> None:
        """
        Initialize population around a given solution.
        
        Parameters:
        -----------
        x0 : np.ndarray
            Initial solution (e.g., from PSO)
        """
        # Start from a feasible anchor point.
        self.population[0] = self.constraint_proj_fn(np.copy(x0))
        
        # Add variation to others
        for i in range(1, self.population_size):
            noise = np.random.normal(0, 0.05, self.n_dimensions)
            candidate = x0 + noise
            candidate = np.clip(candidate, 0, 1)
            self.population[i] = self.constraint_proj_fn(candidate)
        
        # Re-evaluate
        self.fitness = np.array([self.objective_fn(ind) for ind in self.population])
        
        best_idx = np.argmax(self.fitness)
        self.best_individual = np.copy(self.population[best_idx])
        self.best_fitness = self.fitness[best_idx]
        self.history['best_fitness'][0] = self.best_fitness
    
    def tournament_selection(self, tournament_size: int = None) -> np.ndarray:
        """
        Select individual via tournament selection.
        
        Parameters:
        -----------
        tournament_size : int
            Size of tournament (if None, uses self.tournament_size)
        
        Returns:
        --------
        np.ndarray : Selected individual
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        indices = np.random.choice(self.population_size, tournament_size, replace=False)
        best_idx = indices[np.argmax(self.fitness[indices])]
        return np.copy(self.population[best_idx])
    
    def blend_crossover(self, parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple:
        """
        Blend crossover (BLX-α).
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent chromosomes
        alpha : float
            Blend parameter
        
        Returns:
        --------
        tuple : (offspring1, offspring2)
        """
        n = len(parent1)
        offspring1 = np.zeros(n)
        offspring2 = np.zeros(n)
        
        for i in range(n):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            offspring1[i] = np.random.uniform(lower, upper)
            offspring2[i] = np.random.uniform(lower, upper)
        
        return offspring1, offspring2
    
    def gaussian_mutation(self, individual: np.ndarray, mutation_rate: float = None) -> np.ndarray:
        """
        Apply Gaussian mutation.
        
        Parameters:
        -----------
        individual : np.ndarray
            Chromosome
        mutation_rate : float
            Fraction of genes to mutate (if None, uses self.mutation_rate)
        
        Returns:
        --------
        np.ndarray : Mutated individual
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        mutant = np.copy(individual)
        n_mutations = max(1, int(len(individual) * mutation_rate))
        mutation_indices = np.random.choice(len(individual), n_mutations, replace=False)
        
        for idx in mutation_indices:
            noise = np.random.normal(0, self.mutation_sigma)
            mutant[idx] = mutant[idx] + noise
        
        mutant = np.clip(mutant, 0, 1)
        return mutant
    
    def step(self, iteration: int, n_iterations: int) -> None:
        """
        Perform one GA generation.
        
        Parameters:
        -----------
        iteration : int
            Current generation
        n_iterations : int
            Total generations
        """
        # Elitism: keep best individuals
        elite_indices = np.argsort(-self.fitness)[:self.n_elite]
        elite_population = self.population[elite_indices]
        elite_fitness = self.fitness[elite_indices]
        
        # Generate new population
        new_population = np.copy(elite_population)
        new_fitness = np.copy(elite_fitness)
        
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                offspring1, offspring2 = self.blend_crossover(parent1, parent2)
            else:
                # Selection only (duplication)
                parent1 = self.tournament_selection()
                offspring1 = np.copy(parent1)
                parent2 = self.tournament_selection()
                offspring2 = np.copy(parent2)
            
            # Mutation
            offspring1 = self.gaussian_mutation(offspring1)
            offspring2 = self.gaussian_mutation(offspring2)
            
            # Constraint projection
            offspring1 = self.constraint_proj_fn(offspring1)
            if len(new_population) < self.population_size:
                offspring2 = self.constraint_proj_fn(offspring2)
            
            # Evaluate
            fitness1 = self.objective_fn(offspring1)
            new_population = np.vstack([new_population, offspring1])
            new_fitness = np.append(new_fitness, fitness1)
            
            if len(new_population) < self.population_size:
                fitness2 = self.objective_fn(offspring2)
                new_population = np.vstack([new_population, offspring2])
                new_fitness = np.append(new_fitness, fitness2)
        
        # Trim to population size
        self.population = new_population[:self.population_size]
        self.fitness = new_fitness[:self.population_size]
        
        # Update best
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = np.copy(self.population[best_idx])
        
        # Record history
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        self.history['std_fitness'].append(np.std(self.fitness))
        self.history['best_individual'].append(np.copy(self.best_individual))
        self.history['population_diversity'].append(population_diversity(self.population))
    
    def optimize(self, n_generations: int = 200, verbose: bool = False) -> Dict:
        """
        Run GA optimization.
        
        Parameters:
        -----------
        n_generations : int
            Number of generations
        verbose : bool
            Print progress
        
        Returns:
        --------
        dict : Results including best individual and fitness
        """
        for generation in range(n_generations):
            self.step(generation, n_generations)
            
            if verbose and (generation + 1) % 20 == 0:
                print(f"  GA Generation {generation+1:3d}: "
                      f"Best={self.best_fitness:.6f}, "
                      f"Mean={self.history['mean_fitness'][-1]:.6f}")
        
        return {
            'x': self.best_individual,
            'fitness': self.best_fitness,
            'n_generations': n_generations,
            'history': self.history,
            'population_size': self.population_size
        }
    
    def get_population(self) -> np.ndarray:
        """Return current population."""
        return np.copy(self.population)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Return best individual and fitness."""
        return np.copy(self.best_individual), self.best_fitness


def optimize_with_ga(objective_fn: Callable,
                    constraint_proj_fn: Callable,
                    x0: np.ndarray = None,
                    n_dimensions: int = None,
                    population_size: int = 50,
                    n_generations: int = 200,
                    verbose: bool = False,
                    seed: int = 42) -> Dict:
    """
    High-level interface for GA optimization.
    
    Parameters:
    -----------
    objective_fn : callable
        Objective function f(x) -> float (to maximize)
    constraint_proj_fn : callable
        Constraint projection function
    x0 : np.ndarray, optional
        Warm start point
    n_dimensions : int, optional
        Problem dimension when no warm start is supplied
    population_size : int
        Population size
    n_generations : int
        Number of generations
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
    
    ga = GeneticAlgorithm(
        objective_fn=objective_fn,
        constraint_proj_fn=constraint_proj_fn,
        population_size=population_size,
        n_dimensions=n_dims,
        seed=seed
    )
    
    if x0 is not None:
        ga.warm_start(x0)
    
    if verbose:
        print(f"\n{'GA OPTIMIZATION':^50}")
        print(f"  Population: {population_size}, Generations: {n_generations}")
    
    result = ga.optimize(n_generations, verbose=verbose)
    
    if verbose:
        improvement = result['fitness'] - ga.history['best_fitness'][0]
        print(f"  Final best fitness: {result['fitness']:.6f}")
        print(f"  Improvement: {improvement:+.6f}")
    
    return result


if __name__ == '__main__':
    from data import generate_portfolio_data
    from utils import project_constraints
    
    data = generate_portfolio_data(n_assets=20, seed=42)
    
    # Define objective
    def objective(x):
        return np.dot(x, data['mu'])
    
    # Define constraint projection
    def constraint_proj(x):
        return project_constraints(x, data['c'], data['B'], data['alpha'])
    
    # Warm start
    x0 = np.ones(20) / 20
    
    # Optimize
    result = optimize_with_ga(
        objective_fn=objective,
        constraint_proj_fn=constraint_proj,
        x0=x0,
        population_size=40,
        n_generations=100,
        verbose=True,
        seed=42
    )
    
    print(f"\nBest solution: {result['x'][:5]}")
    print(f"Best fitness: {result['fitness']:.6f}")
