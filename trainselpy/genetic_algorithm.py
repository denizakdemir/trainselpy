"""
Genetic algorithm implementation for TrainSelPy.
"""

import numpy as np
import time
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
import random
from joblib import Parallel, delayed


class Solution:
    """
    Class to represent a solution in the genetic algorithm.
    """
    def __init__(
        self, 
        int_values: List[List[int]] = None, 
        dbl_values: List[List[float]] = None, 
        fitness: float = float('-inf'), 
        multi_fitness: List[float] = None
    ):
        self.int_values = int_values if int_values is not None else []
        self.dbl_values = dbl_values if dbl_values is not None else []
        self.fitness = fitness
        self.multi_fitness = multi_fitness if multi_fitness is not None else []
    
    def copy(self):
        """Create a deep copy of the solution."""
        # Copy integer values (each sublist is explicitly copied)
        int_copy = [list(x) for x in self.int_values]
        # Copy double values similarly
        dbl_copy = [list(x) for x in self.dbl_values]
        # Always copy multi_fitness as a list (even if empty)
        multi_fit_copy = self.multi_fitness.copy()
        return Solution(int_copy, dbl_copy, self.fitness, multi_fit_copy)

    
    def __lt__(self, other):
        """Comparison for sorting (by fitness)."""
        return self.fitness < other.fitness


def initialize_population(
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    pop_size: int
) -> List[Solution]:
    """
    Initialize a random population for the genetic algorithm.
    
    Parameters
    ----------
    candidates : List[List[int]]
        List of lists of candidate indices
    setsizes : List[int]
        List of set sizes to select
    settypes : List[str]
        List of set types
    pop_size : int
        Population size
        
    Returns
    -------
    List[Solution]
        List of random solutions
    """
    population = []
    
    for _ in range(pop_size):
        # Initialize a new solution
        sol = Solution()
        
        # For each set type
        for i, (cand, size, type_) in enumerate(zip(candidates, setsizes, settypes)):
            if type_ == "DBL":
                # For double variables, generate random values between 0 and 1
                sol.dbl_values.append(np.random.uniform(0, 1, size))
            elif type_ == "BOOL":
                # For boolean variables, generate random 0/1 values
                sol.int_values.append(np.random.choice([0, 1], size=len(cand)).tolist())
            else:
                # For set types (UOS, OS, UOMS, OMS)
                if size > len(cand):
                    raise ValueError(f"Set size {size} is larger than number of candidates {len(cand)}")
                
                if type_ in ["UOS", "OS"]:
                    # Unordered or ordered set (no repetition)
                    selected = np.random.choice(cand, size=size, replace=False).tolist()
                    
                    if type_ == "OS":
                        # For ordered set, keep the random order
                        pass
                    else:
                        # For unordered set, sort the values
                        selected.sort()
                    
                    sol.int_values.append(selected)
                elif type_ in ["UOMS", "OMS"]:
                    # Unordered or ordered multiset (with repetition)
                    selected = np.random.choice(cand, size=size, replace=True).tolist()
                    
                    if type_ == "OMS":
                        # For ordered multiset, keep the random order
                        pass
                    else:
                        # For unordered multiset, sort the values
                        selected.sort()
                    
                    sol.int_values.append(selected)
        
        population.append(sol)
    
    return population


def evaluate_fitness(
    population: List[Solution],
    stat_func: Callable,
    data: Dict[str, Any],
    n_stat: int = 1,
    is_parallel: bool = False,
    control: Dict[str, Any] = None
) -> None:
    """
    Evaluate the fitness of each solution in the population.
    
    Parameters
    ----------
    population : List[Solution]
        List of solutions
    stat_func : Callable
        Fitness function
    data : Dict[str, Any]
        Data for fitness evaluation
    n_stat : int, optional
        Number of objectives for multi-objective optimization
    is_parallel : bool, optional
        Whether to use parallel evaluation
    control : Dict[str, Any], optional
        Control parameters
        
    Returns
    -------
    None
        The population list is modified in-place
    """
    # Check if we have both integer and double variables
    has_int = any(sol.int_values for sol in population)
    has_dbl = any(sol.dbl_values for sol in population)
    
    if n_stat == 1:
        # Single-objective optimization
        if is_parallel:
            # Parallel evaluation
            if has_int and has_dbl:
                # Both integer and double variables
                int_sols = [sol.int_values[0] for sol in population]
                dbl_sols = [sol.dbl_values[0] for sol in population]
                fitness_values = stat_func(int_sols, dbl_sols, data)
            else:
                # Only integer or only double variables
                if has_int:
                    sols = [sol.int_values[0] for sol in population]
                else:
                    sols = [sol.dbl_values[0] for sol in population]
                fitness_values = stat_func(sols, data)
            
            # Assign fitness values
            for i, sol in enumerate(population):
                sol.fitness = fitness_values[i]
        else:
            # Sequential evaluation
            for sol in population:
                if has_int and has_dbl:
                    # Both integer and double variables
                    sol.fitness = stat_func(sol.int_values[0], sol.dbl_values[0], data)
                else:
                    # Only integer or only double variables
                    if has_int:
                        sol.fitness = stat_func(sol.int_values[0], data)
                    else:
                        sol.fitness = stat_func(sol.dbl_values[0], data)
    else:
        # Multi-objective optimization
        for sol in population:
            if has_int and has_dbl:
                # Both integer and double variables
                sol.multi_fitness = stat_func(sol.int_values[0], sol.dbl_values[0], data)
            else:
                # Only integer or only double variables
                if has_int:
                    sol.multi_fitness = stat_func(sol.int_values[0], data)
                else:
                    sol.multi_fitness = stat_func(sol.dbl_values[0], data)
            
            # For sorting purposes, use the sum (or other aggregation) as the main fitness
            sol.fitness = sum(sol.multi_fitness)


def selection(
    population: List[Solution],
    n_elite: int,
    tournament_size: int = 3
) -> List[Solution]:
    """
    Select parents for the next generation using tournament selection.
    For multi-objective optimization, uses non-dominated sorting.
    
    Parameters
    ----------
    population : List[Solution]
        List of solutions
    n_elite : int
        Number of elite solutions to keep
    tournament_size : int, optional
        Size of the tournament
        
    Returns
    -------
    List[Solution]
        List of selected parents
    """
    # Check if this is a multi-objective optimization problem
    is_multi_objective = any(len(sol.multi_fitness) > 0 for sol in population) if population else False
    
    if is_multi_objective:
        # Perform non-dominated sorting
        fronts = []
        remaining = population.copy()
        
        while remaining and len(sum(fronts, [])) < n_elite:
            # Find the current Pareto front
            pareto_front = []
            for sol in remaining:
                is_dominated = False
                for other in remaining:
                    if sol is not other:
                        if all(other.multi_fitness[i] >= sol.multi_fitness[i] for i in range(len(sol.multi_fitness))) and \
                           any(other.multi_fitness[i] > sol.multi_fitness[i] for i in range(len(sol.multi_fitness))):
                            is_dominated = True
                            break
                if not is_dominated:
                    pareto_front.append(sol)
            
            fronts.append(pareto_front)
            remaining = [sol for sol in remaining if sol not in pareto_front]
        
        # Select elite solutions from the fronts
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_elite:
                selected.extend([sol.copy() for sol in front])
            else:
                # If we can't fit the entire front, sort by a diversity measure
                # Calculate distances between solutions in objective space
                front_copy = front.copy()
                distances = []
                
                for i, sol1 in enumerate(front_copy):
                    distance = 0.0
                    for sol2 in front_copy:
                        if sol1 is not sol2:
                            # Calculate Euclidean distance in objective space
                            dist = sum((sol1.multi_fitness[j] - sol2.multi_fitness[j])**2 
                                       for j in range(len(sol1.multi_fitness)))
                            distance += dist**0.5
                    distances.append(distance)
                
                # Sort the front by distance (more diverse solutions first)
                sorted_front = [x for _, x in sorted(zip(distances, front_copy), key=lambda pair: pair[0], reverse=True)]
                selected.extend([sol.copy() for sol in sorted_front[:n_elite - len(selected)]])
                break
    else:
        # For single-objective, just sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        selected = [sol.copy() for sol in sorted_pop[:n_elite]]
    
    # Fill the rest with tournament selection
    pop_size = len(population)
    while len(selected) < pop_size:
        # Select tournament_size random individuals
        tournament = random.sample(population, tournament_size)
        
        if is_multi_objective:
            # For multi-objective, select a non-dominated solution from the tournament
            non_dominated = []
            for sol in tournament:
                is_dominated = False
                for other in tournament:
                    if sol is not other:
                        if all(other.multi_fitness[i] >= sol.multi_fitness[i] for i in range(len(sol.multi_fitness))) and \
                           any(other.multi_fitness[i] > sol.multi_fitness[i] for i in range(len(sol.multi_fitness))):
                            is_dominated = True
                            break
                if not is_dominated:
                    non_dominated.append(sol)
            
            if non_dominated:
                # If there are multiple non-dominated solutions, select randomly
                winner = random.choice(non_dominated)
            else:
                # Fallback to regular fitness
                winner = max(tournament, key=lambda x: x.fitness)
        else:
            # For single-objective, select the best by fitness
            winner = max(tournament, key=lambda x: x.fitness)
        
        selected.append(winner.copy())
    
    return selected


def crossover(
    parents: List[Solution],
    crossprob: float,
    crossintensity: float,
    settypes: List[str] = None,
    candidates: List[List[int]] = None
) -> List[Solution]:
    """
    Perform crossover on the population.
    
    Parameters
    ----------
    parents : List[Solution]
        List of parent solutions
    crossprob : float
        Probability of crossover
    crossintensity : float
        Intensity of crossover
    settypes : List[str], optional
        List of set types for each variable
    candidates : List[List[int]], optional
        List of candidate sets for each variable
        
    Returns
    -------
    List[Solution]
        List of offspring solutions
    """
    offspring = []
    n_parents = len(parents)
    
    # Create pairs of parents
    for i in range(0, n_parents, 2):
        if i + 1 < n_parents:
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # Decide whether to perform crossover
            if random.random() < crossprob:
                # Create two new offspring
                child1 = parent1.copy()
                child2 = parent2.copy()
                
                # Crossover integer values
                for j in range(len(parent1.int_values)):
                    if parent1.int_values and parent2.int_values:
                        # Calculate crossover points
                        size = len(parent1.int_values[j])
                        n_points = max(1, int(size * crossintensity))
                        
                        # Select crossover points
                        points = sorted(random.sample(range(1, size), min(n_points, size - 1)))
                        
                        # Perform crossover
                        for k in range(len(points)):
                            if k % 2 == 0:
                                # Swap segments
                                start = points[k - 1] if k > 0 else 0
                                end = points[k]
                                
                                temp = child1.int_values[j][start:end].copy()
                                child1.int_values[j][start:end] = child2.int_values[j][start:end].copy()
                                child2.int_values[j][start:end] = temp
                                
                                # Check set type and fix if needed
                                if settypes and j < len(settypes):
                                    # Fix unordered sets by sorting
                                    if settypes[j] in ["UOS", "UOMS"]:
                                        child1.int_values[j].sort()
                                        child2.int_values[j].sort()
                                    
                                    # Fix sets without repetition by removing duplicates
                                    if settypes[j] in ["UOS", "OS"] and candidates and j < len(candidates):
                                        # Fix child1
                                        values_set = set()
                                        fixed_values = []
                                        for val in child1.int_values[j]:
                                            if val not in values_set:
                                                values_set.add(val)
                                                fixed_values.append(val)
                                        
                                        # If we lost any values, fill in with unused candidates
                                        while len(fixed_values) < len(child1.int_values[j]):
                                            for c in candidates[j]:
                                                if c not in values_set:
                                                    values_set.add(c)
                                                    fixed_values.append(c)
                                                    break
                                        
                                        child1.int_values[j] = fixed_values
                                        
                                        # Fix child2
                                        values_set = set()
                                        fixed_values = []
                                        for val in child2.int_values[j]:
                                            if val not in values_set:
                                                values_set.add(val)
                                                fixed_values.append(val)
                                        
                                        # If we lost any values, fill in with unused candidates
                                        while len(fixed_values) < len(child2.int_values[j]):
                                            for c in candidates[j]:
                                                if c not in values_set:
                                                    values_set.add(c)
                                                    fixed_values.append(c)
                                                    break
                                        
                                        child2.int_values[j] = fixed_values
                                        
                                        # Re-sort if needed
                                        if settypes[j] == "UOS":
                                            child1.int_values[j].sort()
                                            child2.int_values[j].sort()
                
                # Crossover double values
                for j in range(len(parent1.dbl_values)):
                    if parent1.dbl_values and parent2.dbl_values:
                        # Calculate crossover points
                        size = len(parent1.dbl_values[j])
                        n_points = max(1, int(size * crossintensity))
                        
                        # Select crossover points
                        points = sorted(random.sample(range(1, size), min(n_points, size - 1)))
                        
                        # Perform crossover
                        for k in range(len(points)):
                            if k % 2 == 0:
                                # Swap segments
                                start = points[k - 1] if k > 0 else 0
                                end = points[k]
                                
                                temp = child1.dbl_values[j][start:end]
                                child1.dbl_values[j][start:end] = child2.dbl_values[j][start:end]
                                child2.dbl_values[j][start:end] = temp
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # No crossover, just copy the parents
                offspring.append(parent1.copy())
                offspring.append(parent2.copy())
        else:
            # Odd number of parents, just copy the last one
            offspring.append(parents[i].copy())
    
    return offspring


def mutation(
    population: List[Solution],
    candidates: List[List[int]],
    settypes: List[str],
    mutprob: float,
    mutintensity: float
) -> None:
    """
    Perform mutation on the population.
    
    Parameters
    ----------
    population : List[Solution]
        List of solutions
    candidates : List[List[int]]
        List of lists of candidate indices
    settypes : List[str]
        List of set types
    mutprob : float
        Probability of mutation
    mutintensity : float
        Intensity of mutation
        
    Returns
    -------
    None
        The population list is modified in-place
    """
    for sol in population:
        # Mutation for integer values
        for i, values in enumerate(sol.int_values):
            # Determine number of mutations
            n_mutations = int(len(values) * mutintensity) + 1
            
            # For each potential mutation
            for _ in range(n_mutations):
                if random.random() < mutprob:
                    # Select a random position to mutate
                    pos = random.randrange(len(values))
                    
                    if settypes[i] == "BOOL":
                        # For boolean variables, flip the bit
                        sol.int_values[i][pos] = 1 - sol.int_values[i][pos]
                    else:
                        # For set types, replace with a random value
                        old_val = sol.int_values[i][pos]
                        cand = candidates[i]
                        
                        # Handle the different set types
                        if settypes[i] in ["UOS", "OS"]:
                            # For sets without repetition, avoid values already in the set
                            current_set = set(sol.int_values[i])
                            available = [c for c in cand if c != old_val and c not in current_set]
                            if available:
                                sol.int_values[i][pos] = random.choice(available)
                        else:  # UOMS, OMS
                            # For multisets, allow any value from the candidates
                            available = cand
                            if available:
                                sol.int_values[i][pos] = random.choice(available)
                        
                        # For unordered sets, ensure the values are sorted
                        if settypes[i] in ["UOS", "UOMS"]:
                            sol.int_values[i].sort()
        
        # Mutation for double values
        for i, values in enumerate(sol.dbl_values):
            # Determine number of mutations
            n_mutations = int(len(values) * mutintensity) + 1
            
            # For each potential mutation
            for _ in range(n_mutations):
                if random.random() < mutprob:
                    # Select a random position to mutate
                    pos = random.randrange(len(values))
                    
                    # Mutate the value
                    delta = np.random.normal(0, mutintensity)
                    sol.dbl_values[i][pos] += delta
                    
                    # Ensure the value is in the valid range [0, 1]
                    sol.dbl_values[i][pos] = max(0, min(1, sol.dbl_values[i][pos]))


def simulated_annealing(
    solution: Solution,
    candidates: List[List[int]],
    settypes: List[str],
    stat_func: Callable,
    data: Dict[str, Any],
    n_stat: int = 1,
    n_iter: int = 50,
    temp_init: float = 100.0,
    temp_final: float = 0.1
) -> Solution:
    """
    Perform simulated annealing on a solution.
    
    Parameters
    ----------
    solution : Solution
        Solution to optimize
    candidates : List[List[int]]
        List of lists of candidate indices
    settypes : List[str]
        List of set types
    stat_func : Callable
        Fitness function
    data : Dict[str, Any]
        Data for fitness evaluation
    n_stat : int, optional
        Number of objectives for multi-objective optimization
    n_iter : int, optional
        Number of iterations
    temp_init : float, optional
        Initial temperature
    temp_final : float, optional
        Final temperature
        
    Returns
    -------
    Solution
        Optimized solution
    """
    # Check if we have both integer and double variables
    has_int = bool(solution.int_values)
    has_dbl = bool(solution.dbl_values)
    
    # Initialize the current solution
    current = solution.copy()
    best = solution.copy()
    
    # Calculate cooling rate
    cooling_rate = (temp_init / temp_final) ** (1.0 / n_iter)
    temp = temp_init
    
    for t in range(n_iter):
        # Generate a neighbor solution
        neighbor = current.copy()
        
        # Modify integer values
        for i, values in enumerate(neighbor.int_values):
            if random.random() < 0.5:  # 50% chance to modify this set
                if settypes[i] == "BOOL":
                    # For boolean variables, flip a random bit
                    pos = random.randrange(len(values))
                    neighbor.int_values[i][pos] = 1 - neighbor.int_values[i][pos]
                else:
                    # For set types, replace a random value
                    pos = random.randrange(len(values))
                    old_val = neighbor.int_values[i][pos]
                    cand = candidates[i]
                    
                    # Handle the different set types
                    if settypes[i] in ["UOS", "OS"]:
                        # For sets without repetition, avoid values already in the set
                        current_set = set(neighbor.int_values[i])
                        available = [c for c in cand if c != old_val and c not in current_set]
                        if available:
                            neighbor.int_values[i][pos] = random.choice(available)
                    else:  # UOMS, OMS
                        # For multisets, allow any value from the candidates
                        available = cand
                        if available:
                            neighbor.int_values[i][pos] = random.choice(available)
                    
                    # For unordered sets, ensure the values are sorted
                    if settypes[i] in ["UOS", "UOMS"]:
                        neighbor.int_values[i].sort()
        
        # Modify double values
        for i, values in enumerate(neighbor.dbl_values):
            if random.random() < 0.5:  # 50% chance to modify this set
                # Modify a random value
                pos = random.randrange(len(values))
                delta = np.random.normal(0, 0.1)  # Small random change
                neighbor.dbl_values[i][pos] += delta
                
                # Ensure the value is in the valid range [0, 1]
                neighbor.dbl_values[i][pos] = max(0, min(1, neighbor.dbl_values[i][pos]))
        
        # Evaluate the neighbor
        if n_stat == 1:
            # Single-objective
            if has_int and has_dbl:
                # Both integer and double variables
                neighbor.fitness = stat_func(neighbor.int_values[0], neighbor.dbl_values[0], data)
            else:
                # Only integer or only double variables
                if has_int:
                    neighbor.fitness = stat_func(neighbor.int_values[0], data)
                else:
                    neighbor.fitness = stat_func(neighbor.dbl_values[0], data)
                    
            # Decide whether to accept the neighbor
            # Handle cases where fitness might be NaN or infinite
            if isinstance(neighbor.fitness, (int, float)) and isinstance(current.fitness, (int, float)):
                if np.isfinite(neighbor.fitness) and np.isfinite(current.fitness):
                    delta = neighbor.fitness - current.fitness
                    if delta > 0 or random.random() < np.exp(delta / temp):
                        current = neighbor.copy()
                elif np.isfinite(neighbor.fitness):
                    # Current fitness is not finite but neighbor's is
                    current = neighbor.copy()
                
                # Update the best solution
                if np.isfinite(current.fitness) and (not np.isfinite(best.fitness) or current.fitness > best.fitness):
                    best = current.copy()
        else:
            # Multi-objective
            if has_int and has_dbl:
                # Both integer and double variables
                neighbor.multi_fitness = stat_func(neighbor.int_values[0], neighbor.dbl_values[0], data)
            else:
                # Only integer or only double variables
                if has_int:
                    neighbor.multi_fitness = stat_func(neighbor.int_values[0], data)
                else:
                    neighbor.multi_fitness = stat_func(neighbor.dbl_values[0], data)
            
            # For sorting purposes, use the sum as the main fitness
            neighbor.fitness = sum(neighbor.multi_fitness)
            
            # Decide whether to accept the neighbor
            # Handle cases where fitness might be NaN or infinite
            if isinstance(neighbor.fitness, (int, float)) and isinstance(current.fitness, (int, float)):
                if np.isfinite(neighbor.fitness) and np.isfinite(current.fitness):
                    delta = neighbor.fitness - current.fitness
                    if delta > 0 or random.random() < np.exp(delta / temp):
                        current = neighbor.copy()
                elif np.isfinite(neighbor.fitness):
                    # Current fitness is not finite but neighbor's is
                    current = neighbor.copy()
                
                # Update the best solution
                if np.isfinite(current.fitness) and (not np.isfinite(best.fitness) or current.fitness > best.fitness):
                    best = current.copy()
        
        # Cool down the temperature
        temp /= cooling_rate
    
    return best


def genetic_algorithm(
    data: Dict[str, Any],
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    stat_func: Callable,
    target: List[int] = None,
    control: Dict[str, Any] = None,
    init_sol: Dict[str, Any] = None,
    n_stat: int = 1,
    is_parallel: bool = False,
    solution_diversity: bool = True
) -> Dict[str, Any]:
    """
    Run the genetic algorithm for optimization.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data for the optimization
    candidates : List[List[int]]
        List of lists of candidate indices
    setsizes : List[int]
        List of set sizes to select
    settypes : List[str]
        List of set types
    stat_func : Callable
        Fitness function
    target : List[int], optional
        List of target indices
    control : Dict[str, Any], optional
        Control parameters for the algorithm
    init_sol : Dict[str, Any], optional
        Initial solution
    n_stat : int, optional
        Number of objectives for multi-objective optimization
    is_parallel : bool, optional
        Whether to use parallel evaluation
    solution_diversity : bool, optional
        Whether to enforce uniqueness of solutions on the Pareto front. When True,
        duplicate solutions are eliminated, ensuring more diverse Pareto front.
        
    Returns
    -------
    Dict[str, Any]
        Results of the optimization
    """
    if control is None:
        control = {}
    
    # Extract control parameters
    n_iterations = control.get("niterations", 500)
    min_iter_before_stop = control.get("minitbefstop", 100)
    n_elite_saved = control.get("nEliteSaved", 5)
    n_elite = control.get("nelite", 100)
    pop_size = control.get("npop", 500)
    mut_prob = control.get("mutprob", 0.01)
    mut_intensity = control.get("mutintensity", 0.1)
    cross_prob = control.get("crossprob", 0.5)
    cross_intensity = control.get("crossintensity", 0.75)
    n_iter_sann = control.get("niterSANN", 50)
    temp_init = control.get("tempini", 100.0)
    temp_final = control.get("tempfin", 0.1)
    dynamic_n_elite = control.get("dynamicNelite", True)
    show_progress = control.get("progress", True)
    save_trace = control.get("trace", False)
    
    # Initialize population
    population = initialize_population(candidates, setsizes, settypes, pop_size)
    
    # Incorporate initial solution if provided
    if init_sol is not None:
        if "solnIntMat" in init_sol and init_sol["solnIntMat"].size > 0:
            for i in range(min(n_elite_saved, init_sol["solnIntMat"].shape[0])):
                population[i].int_values = [init_sol["solnIntMat"][i].tolist()]
        
        if "solnDBLMat" in init_sol and init_sol["solnDBLMat"].size > 0:
            for i in range(min(n_elite_saved, init_sol["solnDBLMat"].shape[0])):
                population[i].dbl_values = [init_sol["solnDBLMat"][i].tolist()]
    
    # Evaluate initial population
    evaluate_fitness(population, stat_func, data, n_stat, is_parallel, control)
    
    # Keep track of the best solution and fitness history
    best_solution = max(population, key=lambda x: x.fitness).copy()
    fitness_history = [best_solution.fitness]
    no_improvement_count = 0
    
    if show_progress:
        print(f"Starting GA with population size {pop_size}")
        print(f"Initial best fitness: {best_solution.fitness}")
    
    # Main GA loop
    for gen in range(n_iterations):
        # Select parents
        parents = selection(population, n_elite, tournament_size=3)
        
        # Create offspring through crossover
        offspring = crossover(parents, cross_prob, cross_intensity, settypes, candidates)
        
        # Apply mutation
        mutation(offspring, candidates, settypes, mut_prob, mut_intensity)
        
        # Update population
        population = offspring
        
        # Evaluate fitness for the current generation
        evaluate_fitness(population, stat_func, data, n_stat, is_parallel, control)
        
        # --- New: Apply simulated annealing (SA) on elite solutions at each iteration ---
        if n_iter_sann > 0:
            # Sort population to get the elite solutions
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            # Apply SA to each elite individual
            for i in range(min(n_elite_saved, len(sorted_pop))):
                refined = simulated_annealing(
                    sorted_pop[i],
                    candidates,
                    settypes,
                    stat_func,
                    data,
                    n_stat,
                    n_iter_sann,
                    temp_init,
                    temp_final
                )
                # Replace the original if the refined solution is better
                if refined.fitness > sorted_pop[i].fitness:
                    sorted_pop[i] = refined.copy()
            # Update the population with the refined elites
            for i in range(min(n_elite_saved, len(sorted_pop))):
                population[i] = sorted_pop[i].copy()
        # -------------------------------------------------------------------------------
        
        # Update best solution based on the (possibly refined) population
        current_best = max(population, key=lambda x: x.fitness)
        if current_best.fitness > best_solution.fitness:
            best_solution = current_best.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Record fitness history
        fitness_history.append(best_solution.fitness)
        
        # Dynamic adjustment of elite size if enabled
        if dynamic_n_elite and gen > 0 and gen % 10 == 0:
            if no_improvement_count > 10:
                n_elite = max(10, int(n_elite * 0.9))
            else:
                n_elite = min(int(pop_size * 0.5), int(n_elite * 1.1))
        
        # Show progress every 10 generations
        if show_progress and gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {best_solution.fitness}")
        
        # Check for early stopping
        if no_improvement_count >= min_iter_before_stop and gen >= min_iter_before_stop:
            if show_progress:
                print(f"Stopping early at generation {gen} due to no improvement")
            break
    
    # --- Removed final SA block since SA is now applied at every iteration ---
    # The following block has been removed:
    # if n_iter_sann > 0:
    #     ...
    
    # Process the result
    if n_stat == 1:
        result = {
            "selected_indices": best_solution.int_values,
            "selected_values": best_solution.dbl_values,
            "fitness": best_solution.fitness,
            "fitness_history": fitness_history
        }
    else:
        # Multi-objective result processing
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        pareto_front = []
        pareto_solutions = []
        
        for sol in population:
            is_dominated = False
            for pf_sol in pareto_solutions:
                if all(pf_sol.multi_fitness[i] >= sol.multi_fitness[i] for i in range(n_stat)) and \
                   any(pf_sol.multi_fitness[i] > sol.multi_fitness[i] for i in range(n_stat)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # First, remove any solutions that this one dominates
                new_pareto = []
                for pf_sol in pareto_solutions:
                    if not (all(sol.multi_fitness[i] >= pf_sol.multi_fitness[i] for i in range(n_stat)) and \
                            any(sol.multi_fitness[i] > pf_sol.multi_fitness[i] for i in range(n_stat))):
                        new_pareto.append(pf_sol)
                pareto_solutions = new_pareto
                
                # Then check for duplicate solutions if solution_diversity is enabled
                if solution_diversity:
                    # Check if this solution is unique based on the int_values (selected indices)
                    is_duplicate = False
                    for pf_sol in pareto_solutions:
                        if len(sol.int_values) == len(pf_sol.int_values):
                            all_match = True
                            for i in range(len(sol.int_values)):
                                if set(sol.int_values[i]) != set(pf_sol.int_values[i]):
                                    all_match = False
                                    break
                            if all_match:
                                is_duplicate = True
                                # If it's a duplicate but has better fitness, replace the existing one
                                if sol.fitness > pf_sol.fitness:
                                    idx = pareto_solutions.index(pf_sol)
                                    pareto_solutions[idx] = sol
                                    pareto_front[idx] = sol.multi_fitness
                                break
                    
                    if not is_duplicate:
                        pareto_solutions.append(sol)
                        pareto_front.append(sol.multi_fitness)
                else:
                    # Add all non-dominated solutions (may include duplicates)
                    pareto_solutions.append(sol)
                    pareto_front.append(sol.multi_fitness)
        
        result = {
            "selected_indices": best_solution.int_values,
            "selected_values": best_solution.dbl_values,
            "fitness": best_solution.fitness,
            "fitness_history": fitness_history,
            "pareto_front": pareto_front,
            "pareto_solutions": [
                {
                    "selected_indices": sol.int_values,
                    "selected_values": sol.dbl_values,
                    "multi_fitness": sol.multi_fitness
                }
                for sol in pareto_solutions
            ]
        }
    
    return result



def island_model_ga(
    data: Dict[str, Any],
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    stat_func: Callable,
    target: List[int] = None,
    control: Dict[str, Any] = None,
    init_sol: Dict[str, Any] = None,
    n_stat: int = 1,
    n_islands: int = 4,
    n_jobs: int = 1,
    solution_diversity: bool = True
) -> Dict[str, Any]:
    """
    Run the island model genetic algorithm for optimization.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data for the optimization
    candidates : List[List[int]]
        List of lists of candidate indices
    setsizes : List[int]
        List of set sizes to select
    settypes : List[str]
        List of set types
    stat_func : Callable
        Fitness function
    target : List[int], optional
        List of target indices
    control : Dict[str, Any], optional
        Control parameters for the algorithm
    init_sol : Dict[str, Any], optional
        Initial solution
    n_stat : int, optional
        Number of objectives for multi-objective optimization
    n_islands : int, optional
        Number of islands
    n_jobs : int, optional
        Number of parallel jobs
    solution_diversity : bool, optional
        Whether to enforce uniqueness of solutions on the Pareto front. When True,
        duplicate solutions are eliminated, ensuring more diverse Pareto front.
        
    Returns
    -------
    Dict[str, Any]
        Results of the optimization
    """
    if control is None:
        control = {}
    
    # Show progress
    show_progress = control.get("progress", True)
    
    if show_progress:
        print(f"Starting island model GA with {n_islands} islands")
    
    # Define a function to run GA on one island
    def run_island(island_id):
        # Create a modified control object for this island
        island_control = control.copy()
        island_control["progress"] = False  # Disable progress output for islands
        
        # Add some randomization to hyperparameters for diversity
        island_control["mutprob"] = control.get("mutprob", 0.01) * (0.8 + 0.4 * random.random())
        island_control["crossprob"] = control.get("crossprob", 0.5) * (0.8 + 0.4 * random.random())
        
        # Run GA on this island
        result = genetic_algorithm(
            data=data,
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat_func=stat_func,
            target=target,
            control=island_control,
            init_sol=init_sol,
            n_stat=n_stat,
            is_parallel=False  # No parallel within island
        )
        
        return result
    
    # Run GAs on all islands
    if n_jobs > 1:
        # Run in parallel
        island_results = Parallel(n_jobs=min(n_jobs, n_islands))(
            delayed(run_island)(i) for i in range(n_islands)
        )
    else:
        # Run sequentially
        island_results = [run_island(i) for i in range(n_islands)]
    
    if show_progress:
        print("Island optimizations completed, consolidating results...")
    
    # Consolidate results
    if n_stat == 1:
        # Single-objective: find the best solution across all islands
        best_island = max(island_results, key=lambda x: x["fitness"])
        result = best_island
    else:
        # Multi-objective: combine Pareto fronts from all islands
        combined_pareto = []
        combined_solutions = []
        
        for island_result in island_results:
            if "pareto_solutions" in island_result:
                combined_pareto.extend(island_result["pareto_front"])
                combined_solutions.extend(island_result["pareto_solutions"])
        
        # Re-identify the Pareto front from combined solutions
        pareto_front = []
        pareto_solutions = []
        
        for i, sol in enumerate(combined_solutions):
            is_dominated = False
            for j, pf_sol in enumerate(combined_solutions):
                if i != j:
                    # Check if sol is dominated by pf_sol
                    if all(pf_sol["multi_fitness"][k] >= sol["multi_fitness"][k] for k in range(n_stat)) and \
                       any(pf_sol["multi_fitness"][k] > sol["multi_fitness"][k] for k in range(n_stat)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                # Check for duplicate solutions (if solution_diversity is enabled)
                if solution_diversity:
                    # Check if this solution is unique based on selected indices
                    is_duplicate = False
                    for existing_sol in pareto_solutions:
                        # Compare integer solutions (selected indices)
                        if len(sol["selected_indices"]) == len(existing_sol["selected_indices"]):
                            all_match = True
                            for set_idx in range(len(sol["selected_indices"])):
                                if set(sol["selected_indices"][set_idx]) != set(existing_sol["selected_indices"][set_idx]):
                                    all_match = False
                                    break
                            if all_match:
                                is_duplicate = True
                                # Keep the solution with better sum of objectives
                                if sum(sol["multi_fitness"]) > sum(existing_sol["multi_fitness"]):
                                    # Replace the existing solution with this one
                                    idx = pareto_solutions.index(existing_sol)
                                    pareto_solutions[idx] = sol
                                    pareto_front[idx] = sol["multi_fitness"]
                                break
                    
                    if not is_duplicate:
                        pareto_solutions.append(sol)
                        pareto_front.append(sol["multi_fitness"])
                else:
                    # Add all non-dominated solutions (may include duplicates)
                    pareto_solutions.append(sol)
                    pareto_front.append(sol["multi_fitness"])
        
        # Find the "best compromise" solution (highest sum of objectives)
        best_compromise = max(pareto_solutions, key=lambda x: sum(x["multi_fitness"]))
        
        # Combine results from all islands
        result = {
            "selected_indices": best_compromise["selected_indices"],
            "selected_values": best_compromise["selected_values"],
            "fitness": sum(best_compromise["multi_fitness"]),
            "fitness_history": [max(r["fitness_history"][i] if i < len(r["fitness_history"]) else r["fitness_history"][-1] 
                                for r in island_results) 
                                for i in range(max(len(r["fitness_history"]) for r in island_results))],
            "pareto_front": pareto_front,
            "pareto_solutions": pareto_solutions
        }
    
    if show_progress:
        print(f"Island model optimization completed, best fitness: {result['fitness']}")
    
    return result