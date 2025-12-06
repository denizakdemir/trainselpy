"""
Genetic algorithm implementations for TrainSelPy.
"""

import numpy as np
import time
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
import random
from joblib import Parallel, delayed
import math

from trainselpy.solution import Solution, flatten_dbl_values, unflatten_dbl_values
from trainselpy.cma_es import CMAESOptimizer
from trainselpy.surrogate import SurrogateModel
from trainselpy.nsga3 import nsga3_selection, generate_reference_points
from trainselpy.operators import crossover, mutation
from trainselpy.selection import selection, fast_non_dominated_sort, calculate_crowding_distance


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


def _prepare_function_args(sol, has_int, has_dbl):
    """Helper to prepare arguments for fitness function call."""
    if has_int and has_dbl:
        int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
        dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
        return (int_arg, dbl_arg)
    elif has_int:
        int_arg = sol.int_values if len(sol.int_values) > 1 else sol.int_values[0]
        return (int_arg,)
    else:  # has_dbl
        dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]
        return (dbl_arg,)


def _prepare_batch_args(population, has_int, has_dbl):
    """Helper to prepare batch arguments for vectorized fitness evaluation."""
    if has_int and has_dbl:
        if len(population[0].int_values) > 1:
            int_sols = [sol.int_values for sol in population]
        else:
            int_sols = [sol.int_values[0] for sol in population]

        if len(population[0].dbl_values) > 1:
            dbl_sols = [sol.dbl_values for sol in population]
        else:
            dbl_sols = [sol.dbl_values[0] for sol in population]
        return (int_sols, dbl_sols)
    elif has_int:
        if len(population[0].int_values) > 1:
            sols = [sol.int_values for sol in population]
        else:
            sols = [sol.int_values[0] for sol in population]
        return (sols,)
    else:  # has_dbl
        if len(population[0].dbl_values) > 1:
            sols = [sol.dbl_values for sol in population]
        else:
            sols = [sol.dbl_values[0] for sol in population]
        return (sols,)


def evaluate_fitness(
    population: List[Solution],
    stat_func: Callable,
    data: Dict[str, Any],
    n_stat: int = 1,
    is_parallel: bool = False,
    control: Dict[str, Any] = None,
    fitness_cache: Dict[int, Union[float, List[float]]] = None,
    surrogate_model: Optional[SurrogateModel] = None
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
    fitness_cache : Dict[int, Union[float, List[float]]], optional
        Cache to avoid re-evaluating identical solutions.
        IMPORTANT: Surrogate evaluations and real evaluations use separate cache keys
        to prevent cache corruption.
    surrogate_model : SurrogateModel, optional
        Surrogate model for fitness prediction

    Returns
    -------
    None
        The population list is modified in-place
    """
    if fitness_cache is None:
        fitness_cache = {}

    # Check if we should use surrogate for objective
    use_surrogate_objective = (
        control and
        control.get("use_surrogate_objective", False) and
        surrogate_model and
        surrogate_model.is_fitted
    )

    if use_surrogate_objective:
        # Use surrogate model for prediction
        # CRITICAL FIX: Use separate cache namespace for surrogate predictions
        # to prevent pollution of real fitness cache
        means, _ = surrogate_model.predict(population)
        for i, sol in enumerate(population):
            sol.fitness = means[i]
            if n_stat > 1:
                print("Warning: Surrogate model currently only supports scalar fitness. "
                      "Multi-objective optimization with surrogate may not work as expected.")
                sol.multi_fitness = [means[i]] * n_stat

            # Cache surrogate predictions with special prefix to distinguish from real evaluations
            # This allows us to re-evaluate with real function without cache pollution
            sol_hash = sol.get_hash()
            cache_key = ("surrogate", sol_hash)
            fitness_cache[cache_key] = sol.fitness
        return

    # Check if we have integer and/or double variables
    has_int = any(sol.int_values for sol in population)
    has_dbl = any(sol.dbl_values for sol in population)
    use_vectorized = control.get("vectorized_stat", False) if control else False

    # Batched/vectorized evaluation path
    if is_parallel or use_vectorized:
        args = _prepare_batch_args(population, has_int, has_dbl)
        results = stat_func(*args, data)

        for i, sol in enumerate(population):
            sol_hash = sol.get_hash()
            cache_key = ("real", sol_hash)  # Use "real" prefix for consistency

            if n_stat == 1:
                sol.fitness = results[i]
                fitness_cache[cache_key] = results[i]
            else:
                sol.multi_fitness = list(results[i])
                sol.fitness = float(sum(sol.multi_fitness))
                # Cache multi-objective results
                fitness_cache[cache_key] = sol.multi_fitness
        return

    # Sequential evaluation with caching
    for sol in population:
        sol_hash = sol.get_hash()

        # Use "real" prefix for real fitness evaluations to distinguish from surrogate cache
        cache_key = ("real", sol_hash)

        if cache_key in fitness_cache:
            # Use cached result (REAL fitness only)
            cached_value = fitness_cache[cache_key]
            if n_stat == 1:
                sol.fitness = cached_value
            else:
                sol.multi_fitness = cached_value
                sol.fitness = sum(sol.multi_fitness)
        else:
            # Evaluate fitness
            args = _prepare_function_args(sol, has_int, has_dbl)
            result = stat_func(*args, data)

            if n_stat == 1:
                sol.fitness = result
                fitness_cache[cache_key] = result
            else:
                sol.multi_fitness = result
                sol.fitness = sum(sol.multi_fitness)
                fitness_cache[cache_key] = sol.multi_fitness


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

    # NUMERICAL STABILITY FIX: Use log-space for more stable temperature calculation
    # Instead of: cooling_rate = (temp_init / temp_final) ** (1.0 / n_iter)
    # Use exponential decay schedule: temp = temp_init * exp(-t * decay_rate)
    # This avoids floating-point errors from repeated divisions
    import math
    if n_iter > 0:
        decay_rate = math.log(temp_init / temp_final) / n_iter
    else:
        decay_rate = 0
    temp = temp_init

    for t in range(n_iter):
        # Generate a neighbor solution
        neighbor = current.copy()

        # Use the mutation operator from operators.py for consistency and code reuse
        # For SA, we want small local moves:
        # - mutintensity=0.0 ensures n_mutations = int(len * 0) + 1 = 1 (one position per set)
        # - mutprob=0.5 gives 50% chance to mutate each set
        # This matches the original SA semantics while reusing tested mutation logic
        mutation([neighbor], candidates, settypes, mutprob=0.5, mutintensity=0.0)

        # Evaluate the neighbor
        if n_stat == 1:
            # Single-objective
            if has_int and has_dbl:
                # Both integer and double variables
                int_arg = neighbor.int_values if len(neighbor.int_values) > 1 else neighbor.int_values[0]
                dbl_arg = neighbor.dbl_values if len(neighbor.dbl_values) > 1 else neighbor.dbl_values[0]
                neighbor.fitness = stat_func(int_arg, dbl_arg, data)
            else:
                # Only integer or only double variables
                if has_int:
                    int_arg = neighbor.int_values if len(neighbor.int_values) > 1 else neighbor.int_values[0]
                    neighbor.fitness = stat_func(int_arg, data)
                else:
                    dbl_arg = neighbor.dbl_values if len(neighbor.dbl_values) > 1 else neighbor.dbl_values[0]
                    neighbor.fitness = stat_func(dbl_arg, data)
                    
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
                int_arg = neighbor.int_values if len(neighbor.int_values) > 1 else neighbor.int_values[0]
                dbl_arg = neighbor.dbl_values if len(neighbor.dbl_values) > 1 else neighbor.dbl_values[0]
                neighbor.multi_fitness = stat_func(int_arg, dbl_arg, data)
            else:
                # Only integer or only double variables
                if has_int:
                    int_arg = neighbor.int_values if len(neighbor.int_values) > 1 else neighbor.int_values[0]
                    neighbor.multi_fitness = stat_func(int_arg, data)
                else:
                    dbl_arg = neighbor.dbl_values if len(neighbor.dbl_values) > 1 else neighbor.dbl_values[0]
                    neighbor.multi_fitness = stat_func(dbl_arg, data)
            
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

        # Cool down the temperature using exponential decay
        # temp = temp_init * exp(-(t+1) * decay_rate)
        temp = temp_init * math.exp(-(t + 1) * decay_rate)

    return best


def generate_from_surrogate(
    surrogate_model: SurrogateModel,
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    n_solutions: int = 1,
    n_iter: int = 50
) -> List[Solution]:
    """
    Generate solutions by optimizing the surrogate model.
    
    Parameters
    ----------
    surrogate_model : SurrogateModel
        Fitted surrogate model
    candidates : List[List[int]]
        List of lists of candidate indices
    setsizes : List[int]
        List of set sizes
    settypes : List[str]
        List of set types
    n_solutions : int
        Number of solutions to generate
    n_iter : int
        Number of iterations for optimization
        
    Returns
    -------
    List[Solution]
        List of optimized solutions
    """
    generated = []
    
    generated = []
    
    # Define a wrapper for the fitness function to use the surrogate
    def surrogate_fitness_wrapper(int_vals, dbl_vals, data=None):
        # Reconstruct a temporary solution object
        # Note: int_vals and dbl_vals are passed as they are stored in the solution
        # (List[List[int]] and List[List[float]])
        # BUT simulated_annealing passes:
        # int_arg = neighbor.int_values if len > 1 else neighbor.int_values[0]
        # This inconsistency in SA needs to be handled or we need to fix SA.
        # The current SA implementation (and my refactor) still does:
        # int_arg = neighbor.int_values if len(neighbor.int_values) > 1 else neighbor.int_values[0]
        
        # So we need to handle both cases here.
        
        temp_sol = Solution()
        
        # Handle int_vals
        if int_vals:
            if isinstance(int_vals[0], list):
                temp_sol.int_values = int_vals
            else:
                temp_sol.int_values = [int_vals]
        
        # Handle dbl_vals
        if dbl_vals:
            if isinstance(dbl_vals[0], list):
                temp_sol.dbl_values = dbl_vals
            else:
                temp_sol.dbl_values = [dbl_vals]
                
        # Predict using surrogate
        means, _ = surrogate_model.predict([temp_sol])
        return means[0]

    for _ in range(n_solutions):
        # Start with a random solution
        sol = initialize_population(candidates, setsizes, settypes, 1)[0]
        
        # Optimize using SA with the surrogate fitness function
        optimized_sol = simulated_annealing(
            solution=sol,
            candidates=candidates,
            settypes=settypes,
            stat_func=surrogate_fitness_wrapper,
            data=None, # Not needed for surrogate wrapper
            n_stat=1, # Surrogate is scalar
            n_iter=n_iter,
            temp_init=1.0,
            temp_final=0.01
        )
        
        generated.append(optimized_sol)
        
    return generated


def _initialize_nsga3(n_stat: int, pop_size: int, control: Dict[str, Any]) -> Tuple[bool, Optional[List[List[float]]]]:
    """
    Initialize NSGA-III reference points.
    """
    use_nsga3 = control.get("use_nsga3", n_stat > 2)
    reference_points = None
    
    if use_nsga3 and n_stat > 1:
        # Determine p (divisions) to match pop_size
        p = 1
        while True:
            n_ref = math.comb(n_stat + p - 1, p)
            if n_ref >= pop_size:
                break
            p += 1
            if p > 20: # Safety break
                break
        
        # If n_ref is much larger than pop_size, maybe reduce p?
        if p > 1 and math.comb(n_stat + (p-1) - 1, p-1) > pop_size * 0.8:
            p -= 1
            
        reference_points = generate_reference_points(n_stat, p)
        if control.get("progress", True):
            print(f"Initialized NSGA-III with {len(reference_points)} reference points (p={p})")
            
    return use_nsga3, reference_points


def _initialize_surrogate(
    control: Dict[str, Any], 
    candidates: List[List[int]], 
    settypes: List[str]
) -> Tuple[Optional[SurrogateModel], List[Solution]]:
    """
    Initialize the surrogate model.
    """
    use_surrogate = control.get("use_surrogate", False)
    surrogate_model = None
    surrogate_archive = []
    
    if use_surrogate:
        try:
            surrogate_model = SurrogateModel(candidates, settypes, model_type="gp")
            if control.get("progress", True):
                print("Initialized Surrogate Model")
        except ImportError:
            print("Warning: scikit-learn not found, disabling surrogate optimization")
            
    return surrogate_model, surrogate_archive


def _initialize_cma_es(
    settypes: List[str],
    population: List[Solution],
    control: Dict[str, Any]
) -> Optional[CMAESOptimizer]:
    """
    Initialize CMA-ES optimizer if double variables are present.
    """
    cma_optimizer = None
    has_dbl = any(st == "DBL" for st in settypes)
    use_cma_es = control.get("use_cma_es", True)

    if has_dbl and population and use_cma_es:
        # Create initial mean from the best solution
        best_solution = max(population, key=lambda x: x.fitness)
        initial_mean = flatten_dbl_values(best_solution.dbl_values)
        # Initialize CMA-ES with configurable sigma
        sigma = control.get("cma_es_sigma", 0.2)
        cma_optimizer = CMAESOptimizer(mean=initial_mean, sigma=sigma)
        if control.get("progress", True):
            print(f"Initialized CMA-ES for continuous variables (sigma={sigma})")

    return cma_optimizer


def _generate_surrogate_offspring(
    surrogate_model: SurrogateModel,
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    pop_size: int,
    control: Dict[str, Any]
) -> List[Solution]:
    """
    Generate offspring using surrogate optimization.
    """
    surrogate_generation_prob = control.get("surrogate_generation_prob", 0.0)
    
    if (surrogate_model and surrogate_model.is_fitted and 
        random.random() < surrogate_generation_prob):
        # Generate some offspring using surrogate optimization
        n_gen = max(1, int(pop_size * 0.1)) # Generate 10% of population
        return generate_from_surrogate(
            surrogate_model, candidates, setsizes, settypes, n_solutions=n_gen
        )
    return []


def _prescreen_offspring(
    surrogate_model: SurrogateModel,
    offspring: List[Solution],
    population: List[Solution],
    candidates: List[List[int]],
    settypes: List[str],
    control: Dict[str, Any],
    gen: int
) -> List[Solution]:
    """
    Prescreen offspring using the surrogate model.
    """
    use_surrogate = control.get("use_surrogate", False)
    surrogate_start_gen = control.get("surrogate_start_gen", 10)
    surrogate_prescreen_factor = control.get("surrogate_prescreen_factor", 5)
    
    if (use_surrogate and gen >= surrogate_start_gen and surrogate_model and surrogate_model.is_fitted):
        # Generate more offspring for pre-screening
        n_extra = int((surrogate_prescreen_factor - 1) * len(offspring))
        if n_extra > 0:
            # Generate extra offspring
            extra_offspring = []
            
            # Extract mutation/crossover params
            mut_prob = control.get("mutprob", 0.01)
            mut_intensity = control.get("mutintensity", 0.1)
            cross_prob = control.get("crossprob", 0.5)
            cross_intensity = control.get("crossintensity", 0.75)
            
            while len(extra_offspring) < n_extra:
                # Quick selection
                p = random.sample(population, 2)
                # Crossover
                children = crossover(p, cross_prob, cross_intensity, settypes, candidates)
                mutation(children, candidates, settypes, mut_prob, mut_intensity)
                extra_offspring.extend(children)
            
            # Combine all potential offspring
            all_candidates = offspring + extra_offspring[:n_extra]

            # Predict fitness
            means, stds = surrogate_model.predict(all_candidates)

            # IMPROVED: Use configurable acquisition function for better exploration-exploitation trade-off
            # Options: "ucb" (Upper Confidence Bound), "ei" (Expected Improvement), "mean" (greedy)
            acquisition_function = control.get("surrogate_acquisition", "mean")

            if acquisition_function == "ucb":
                # UCB balances exploitation (high mean) and exploration (high uncertainty)
                # Lower kappa = more exploitation, higher kappa = more exploration
                kappa = control.get("surrogate_kappa", 1.0)  # Reduced from 2.0 for less exploration bias
                scores = means + kappa * stds
            elif acquisition_function == "ei":
                # Expected Improvement: balance between mean and uncertainty
                # Best for expensive functions where we want good solutions quickly
                from scipy.stats import norm
                best_f = max(means)  # Current best predicted fitness
                z = (means - best_f) / (stds + 1e-9)  # Normalize by std
                scores = (means - best_f) * norm.cdf(z) + stds * norm.pdf(z)
            else:  # "mean" or default
                # Greedy selection based on predicted mean (pure exploitation)
                # Best when surrogate is well-fitted and we trust predictions
                scores = means

            # Select top len(offspring) based on scores
            n_keep = len(offspring)
            top_indices = np.argsort(scores)[-n_keep:]

            # Update offspring list
            return [all_candidates[i] for i in top_indices]
            
    return offspring


def _generate_cma_offspring(
    cma_optimizer: Optional[CMAESOptimizer],
    parents: List[Solution],
    stat_func: Callable,
    data: Dict[str, Any],
    n_stat: int,
    is_parallel: bool,
    control: Dict[str, Any],
    fitness_cache: Dict,
    surrogate_model: Optional[SurrogateModel]
) -> Tuple[List[Solution], Optional[List[np.ndarray]]]:
    """
    Generate and evaluate offspring using CMA-ES.
    """
    cma_offspring = []
    cma_candidates = None
    
    if cma_optimizer:
        # Generate candidates from CMA-ES
        cma_candidates = cma_optimizer.ask()
        
        valid_candidates = []
        valid_indices = []

        # Create offspring solutions
        for i, cand_vec in enumerate(cma_candidates):
            if np.isnan(cand_vec).any():
                continue
                
            valid_candidates.append(cand_vec)
            valid_indices.append(i)

            # Create a new solution
            # For integer parts, copy from a random parent to maintain diversity
            parent = random.choice(parents)
            new_sol = parent.copy()
            
            # Update double values from CMA candidate
            # Clip to [0, 1] as per problem constraints
            cand_vec = np.clip(cand_vec, 0, 1)
            new_sol.dbl_values = unflatten_dbl_values(cand_vec, new_sol.dbl_values)

            # Genome has changed, invalidate cached hash
            new_sol.invalidate_hash()
            
            cma_offspring.append(new_sol)
        
        # Update cma_candidates to only include valid ones
        if len(cma_offspring) < len(cma_candidates):
            cma_candidates = [cma_candidates[i] for i in valid_indices]

        # Evaluate CMA offspring
        if cma_offspring:
            evaluate_fitness(cma_offspring, stat_func, data, n_stat, is_parallel, control, fitness_cache, surrogate_model)
        
    return cma_offspring, cma_candidates


def _update_cma_es(
    cma_optimizer: Optional[CMAESOptimizer],
    cma_candidates: Optional[List[np.ndarray]],
    cma_offspring: List[Solution]
) -> None:
    """
    Update CMA-ES optimizer with evaluated offspring.
    """
    if cma_optimizer and cma_candidates is not None and cma_offspring:
        # Use scalar fitness (sum of objectives for MO)
        cma_fitnesses = [sol.fitness for sol in cma_offspring]
        cma_optimizer.tell(cma_candidates, cma_fitnesses)


def knapsack_repair(
    population: List[Solution],
    candidates: List[List[int]],
    setsizes: List[int],
    settypes: List[str],
    data: Dict[str, Any],
    control: Optional[Dict[str, Any]] = None
) -> None:
    """
    Repair operator for BOOL knapsack problems.

    When ``data`` contains ``\"weights\"`` (1D array-like) and ``\"capacity\"``,
    this operator enforces the capacity constraint by turning off items in
    overweight solutions until the total weight is <= capacity. Items are
    removed in order of decreasing weight.
    """
    weights = data.get("weights", None)
    capacity = data.get("capacity", None)
    if weights is None or capacity is None:
        return

    weights_arr = np.asarray(weights, dtype=float)
    cap = float(capacity)

    # Vectorized repair over the entire population for BOOL sets
    for idx, stype in enumerate(settypes):
        if stype != "BOOL":
            continue

        # Collect BOOL genes for this index across population
        gene_rows = []
        sol_indices = []
        for s_idx, sol in enumerate(population):
            if idx < len(sol.int_values):
                gene_rows.append(sol.int_values[idx])
                sol_indices.append(s_idx)

        if not gene_rows:
            continue

        genes_mat = np.asarray(gene_rows, dtype=int)
        # Compute total weights per solution
        total_weights = genes_mat @ weights_arr

        # Identify overweight solutions
        overweight_mask = total_weights > cap
        if not np.any(overweight_mask):
            # No repair needed for this set index
            continue

        # Repair each overweight solution independently (still using NumPy arrays)
        for local_idx, s_idx in enumerate(sol_indices):
            if not overweight_mask[local_idx]:
                continue

            genes = genes_mat[local_idx].copy()
            mask = genes.astype(bool)
            total_weight = float(weights_arr[mask].sum())
            if total_weight <= cap:
                population[s_idx].int_values[idx] = genes.tolist()
                continue

            selected_indices = np.where(mask)[0]
            order = np.argsort(weights_arr[selected_indices])[::-1]
            for pos in order:
                item = int(selected_indices[pos])
                genes[item] = 0
                total_weight -= float(weights_arr[item])
                if total_weight <= cap:
                    break

            population[s_idx].int_values[idx] = genes.tolist()
            population[s_idx].invalidate_hash()


def _select_next_generation(
    combined_pop: List[Solution],
    pop_size: int,
    n_stat: int,
    use_nsga3: bool,
    reference_points: Optional[List[List[float]]],
    solution_diversity: bool = True,
    candidates: Optional[List[List[int]]] = None,
    setsizes: Optional[List[int]] = None,
    settypes: Optional[List[str]] = None,
    stat_func: Optional[Callable] = None,
    data: Optional[Dict[str, Any]] = None,
    is_parallel: bool = False,
    control: Optional[Dict[str, Any]] = None,
    fitness_cache: Optional[Dict[int, Union[float, List[float]]]] = None,
    surrogate_model: Optional[SurrogateModel] = None
) -> List[Solution]:
    """
    Select the next generation of solutions.
    """
    if n_stat > 1:
        if use_nsga3 and reference_points is not None:
            # Use NSGA-III selection
            return nsga3_selection(combined_pop, pop_size, reference_points)
        else:
            # Multi-objective: use Pareto ranking (NSGA-II)
            fronts = fast_non_dominated_sort(combined_pop)
            population = []
            for front in fronts:
                if len(population) + len(front) <= pop_size:
                    population.extend(front)
                else:
                    # Use crowding distance to select most diverse
                    crowding_distances = calculate_crowding_distance(front)
                    sorted_indices = sorted(range(len(front)),
                                          key=lambda i: crowding_distances[i],
                                          reverse=True)
                    
                    n_needed = pop_size - len(population)
                    for i in range(n_needed):
                        population.append(front[sorted_indices[i]])
                    break
            return population
    else:
        # Single-objective: sort by fitness
        combined_pop.sort(key=lambda x: x.fitness, reverse=True)
        
        if not solution_diversity:
            return combined_pop[:pop_size]
        
        # Enforce diversity by removing exact duplicates (by hash)
        unique_pop = []
        seen_hashes = set()
        for sol in combined_pop:
            sol_hash = sol.get_hash()
            if sol_hash in seen_hashes:
                continue
            seen_hashes.add(sol_hash)
            unique_pop.append(sol)
            if len(unique_pop) >= pop_size:
                break

        # CRITICAL FIX: Improved refill logic to prevent quality degradation
        # Only refill if population dropped below 30% (was 50%, which was too aggressive)
        # This balances natural convergence with maintaining diversity
        min_population_fraction = 0.3  # Balanced threshold (was 0.5)
        if len(unique_pop) < pop_size * min_population_fraction:
            can_refill = all(v is not None for v in (candidates, setsizes, settypes, stat_func))
            if can_refill:
                # Generate extra candidates and keep only the best
                # This prevents diluting population with poor random solutions
                n_needed = pop_size - len(unique_pop)
                n_generate = int(n_needed * 2)  # Generate 2x and select best

                new_samples = initialize_population(candidates, setsizes, settypes, n_generate)
                evaluate_fitness(new_samples, stat_func, data, n_stat, is_parallel, control, fitness_cache, surrogate_model)

                # Sort and keep only the best n_needed
                new_samples.sort(key=lambda x: x.fitness, reverse=True)
                unique_pop.extend(new_samples[:n_needed])

                # Re-sort combined population
                unique_pop.sort(key=lambda x: x.fitness, reverse=True)
                return unique_pop[:pop_size]

        # If we have enough unique solutions (>30% of pop_size), just return them
        # Even if less than pop_size, maintaining quality is more important than quantity
        # Natural convergence is allowed - duplicate solutions indicate local optimum
        return unique_pop[:pop_size] if len(unique_pop) >= pop_size else unique_pop


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
    n_iter_sann = control.get("niterSANN", 20)  # Reduced from 50 to 20 for efficiency
    sann_frequency = control.get("sannFrequency", 5)  # Apply SA every N generations
    temp_init = control.get("tempini", 100.0)
    temp_final = control.get("tempfin", 0.1)
    dynamic_n_elite = control.get("dynamicNelite", True)
    show_progress = control.get("progress", True)
    save_trace = control.get("trace", False)
    
    # Surrogate control
    use_surrogate = control.get("use_surrogate", False)
    surrogate_start_gen = control.get("surrogate_start_gen", 10)
    surrogate_update_freq = control.get("surrogate_update_freq", 5)
    surrogate_prescreen_factor = control.get("surrogate_prescreen_factor", 5)  # Generate 5x offspring
    use_surrogate_objective = control.get("use_surrogate_objective", False)
    surrogate_generation_prob = control.get("surrogate_generation_prob", 0.0)
    
    # NSGA-III Setup
    use_nsga3, reference_points = _initialize_nsga3(n_stat, pop_size, control)

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

    # Create fitness cache for the entire run
    fitness_cache = {}

    # Surrogate initialization
    surrogate_model, surrogate_archive = _initialize_surrogate(control, candidates, settypes)

    # Evaluate initial population
    evaluate_fitness(population, stat_func, data, n_stat, is_parallel, control, fitness_cache, surrogate_model)
    
    # Keep track of the best solution and fitness history
    best_solution = max(population, key=lambda x: x.fitness).copy()
    fitness_history = [best_solution.fitness]
    no_improvement_count = 0
    
    # Add initial population to surrogate archive if using surrogate
    if use_surrogate and surrogate_model:
        # Only add if we have real fitness (which we do for initial pop unless we used surrogate objective?)
        # If use_surrogate_objective is True, initial pop is evaluated with surrogate?
        # No, surrogate is not fitted yet!
        # So initial pop MUST be evaluated with real function.
        # But wait, if use_surrogate_objective is True, evaluate_fitness uses surrogate if fitted.
        # Here surrogate is NOT fitted yet. So it uses real function.
        # So we can add to archive.
        surrogate_archive.extend([s.copy() for s in population])
    
    if show_progress:
        print(f"Starting GA with population size {pop_size}")
        print(f"Initial best fitness: {best_solution.fitness}")
    
    # Initialize CMA-ES if there are double variables
    cma_optimizer = _initialize_cma_es(settypes, population, control)

    # Main GA loop
    for gen in range(n_iterations):
        # Select parents
        parents = selection(population, n_elite, tournament_size=3)
        
        # Create offspring through crossover
        if "crossover_func" in control:
            offspring = control["crossover_func"](parents, cross_prob, cross_intensity, settypes, candidates)
        else:
            offspring = crossover(parents, cross_prob, cross_intensity, settypes, candidates)
        
        # Apply mutation
        if "mutation_func" in control:
            control["mutation_func"](offspring, candidates, settypes, mut_prob, mut_intensity)
        else:
            mutation(offspring, candidates, settypes, mut_prob, mut_intensity)

        # --- Surrogate Generation ---
        surrogate_offspring = _generate_surrogate_offspring(
            surrogate_model, candidates, setsizes, settypes, pop_size, control
        )
        offspring.extend(surrogate_offspring)

        # --- Surrogate Pre-screening ---
        offspring = _prescreen_offspring(
            surrogate_model, offspring, population, candidates, settypes, control, gen
        )

        # Optional repair step (e.g. knapsack capacity enforcement)
        repair_func = control.get("repair_func", None)
        if repair_func is not None:
            repair_func(offspring, candidates, setsizes, settypes, data, control)

        # Evaluate fitness for the offspring (using cache)
        # If use_surrogate_objective is True, this will use the surrogate
        evaluate_fitness(offspring, stat_func, data, n_stat, is_parallel, control, fitness_cache, surrogate_model)

        # --- CMA-ES Injection ---
        cma_offspring, cma_candidates = _generate_cma_offspring(
            cma_optimizer, parents, stat_func, data, n_stat, is_parallel, control, fitness_cache, surrogate_model
        )
        
        # Update CMA-ES
        _update_cma_es(cma_optimizer, cma_candidates, cma_offspring)
            
        # Implement elitism: combine population, offspring, and CMA offspring
        # This ensures best solutions are always preserved
        combined = population + offspring + cma_offspring

        # Select the best pop_size individuals from combined population
        population = _select_next_generation(
            combined,
            pop_size,
            n_stat,
            use_nsga3,
            reference_points,
            solution_diversity,
            candidates,
            setsizes,
            settypes,
            stat_func,
            data,
            is_parallel,
            control,
            fitness_cache,
            surrogate_model
        )

        # --- Apply simulated annealing (SA) on elite solutions periodically ---
        # Changed to apply every sann_frequency generations instead of every generation for efficiency
        if n_iter_sann > 0 and (gen % sann_frequency == 0 or gen == n_iterations - 1):
            # For single-objective, population is already sorted from elitism
            # For multi-objective, we need to sort by scalar fitness
            if n_stat > 1:
                sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            else:
                # Already sorted from elitism step
                sorted_pop = population

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
                # For both single and multi-objective, SA already performs acceptance
                # based on fitness improvement, so we simply use the refined solution
                # if it has better scalar fitness (sum for multi-objective)
                if refined.fitness > sorted_pop[i].fitness:
                    sorted_pop[i] = refined.copy()
            # Update the population with the refined elites
            for i in range(min(n_elite_saved, len(sorted_pop))):
                population[i] = sorted_pop[i].copy()
        # -------------------------------------------------------------------------------

        # Update best solution and track improvement
        # CRITICAL FIX: For multi-objective, track Pareto front quality, not scalar fitness sum
        if n_stat > 1:
            # For multi-objective: use Pareto front size as improvement metric
            # This is a simple but effective metric - more sophisticated approaches
            # could use hypervolume indicator
            fronts = fast_non_dominated_sort(population)
            if fronts and fronts[0]:
                current_pareto_size = len(fronts[0])
                current_best = max(population, key=lambda x: x.fitness)  # Keep best for logging

                # Track if Pareto front improved (more diverse non-dominated solutions)
                if not hasattr(best_solution, '_pareto_size'):
                    best_solution._pareto_size = 0

                if current_pareto_size > best_solution._pareto_size:
                    best_solution = current_best.copy()
                    best_solution._pareto_size = current_pareto_size
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # No fronts found (shouldn't happen)
                current_best = max(population, key=lambda x: x.fitness)
                no_improvement_count += 1
        else:
            # For single-objective: use scalar fitness comparison
            current_best = population[0]  # Already sorted from elitism
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
                # Reduce elite size but not below 5% of population (minimum 20)
                min_elite = max(20, int(pop_size * 0.05))
                n_elite = max(min_elite, int(n_elite * 0.9))
            else:
                # Increase elite size but not above 50% of population
                n_elite = min(int(pop_size * 0.5), int(n_elite * 1.1))
        
        # Show progress every 10 generations
        if show_progress and gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {best_solution.fitness}")
        
        # Check for early stopping
        if no_improvement_count >= min_iter_before_stop and gen >= min_iter_before_stop:
            if show_progress:
                print(f"Stopping early at generation {gen} due to no improvement")
            break
            
        # Update Surrogate Model
        if use_surrogate:
            # Add evaluated offspring to archive
            # Only add those that were actually evaluated (offspring + cma_offspring)
            # Note: cma_offspring are already evaluated
            
            # If using surrogate objective, these have predicted fitness, NOT real fitness.
            # So DO NOT add them to archive yet.
            if not use_surrogate_objective:
                new_data = offspring + cma_offspring
                surrogate_archive.extend([s.copy() for s in new_data])
            
            # Limit archive size? (e.g. keep last 1000 or best 1000)
            if len(surrogate_archive) > 2000:
                # Keep best 1000 and random 1000? Or just last 2000?
                # Last 2000 is better for tracking moving landscape (if dynamic)
                # But for static, best is good.
                # Let's keep random to avoid bias
                surrogate_archive = random.sample(surrogate_archive, 2000)
            
            # Retrain
            if gen % surrogate_update_freq == 0 and gen >= surrogate_start_gen:
                # If we are using surrogate as objective, we need to evaluate some solutions with REAL function
                # to update the surrogate.
                if use_surrogate_objective:
                    # Select a subset of archive to re-evaluate?
                    # Or just evaluate the current best?
                    # Let's evaluate the top 5% of the current population with the real function
                    n_eval = max(1, int(len(population) * 0.05))
                    to_eval = population[:n_eval] # Population is sorted
                    
                    # Temporarily disable surrogate objective to force real evaluation
                    temp_control = control.copy()
                    temp_control["use_surrogate_objective"] = False
                    
                    # We need to bypass the cache if it stores surrogate values!
                    # But cache keys are hashes of solution.
                    # If we overwrite cache with real values, that's good.
                    evaluate_fitness(to_eval, stat_func, data, n_stat, is_parallel, temp_control, fitness_cache)
                    
                    # Add these to archive (they are already in population, so maybe just update archive?)
                    # We need to make sure archive has the REAL fitness.
                    # The archive update above `surrogate_archive.extend` added solutions with SURROGATE fitness.
                    # We should probably clear those or update them.
                    
                    # Actually, if use_surrogate_objective is True, `offspring` have surrogate fitness.
                    # We shouldn't add them to archive as training data!
                    # We should ONLY add solutions evaluated with REAL function.
                    
                    # So, if use_surrogate_objective is True, we only add the re-evaluated ones.
                    surrogate_archive.extend([s.copy() for s in to_eval])
                
                # Prepare training data
                # Use scalar fitness for surrogate
                fitnesses = [s.fitness for s in surrogate_archive]
                surrogate_model.fit(surrogate_archive, fitnesses)
                if show_progress and gen % 10 == 0:
                    print(f"Updated Surrogate Model with {len(surrogate_archive)} samples")

        # Invoke user callback if provided
        if "callback" in control and control["callback"] is not None:
            # Compute current Pareto front for multi-objective problems
            current_pareto_front = None
            current_pareto_solutions = None
            if n_stat > 1:
                fronts = fast_non_dominated_sort(population)
                if fronts:
                    pareto_sols = fronts[0]
                    # Apply diversity filter if enabled
                    if solution_diversity and pareto_sols:
                        unique_pareto = []
                        seen_hashes = set()
                        for sol in pareto_sols:
                            sol_hash = sol.get_hash()
                            if sol_hash not in seen_hashes:
                                unique_pareto.append(sol)
                                seen_hashes.add(sol_hash)
                        pareto_sols = unique_pareto

                    current_pareto_front = [sol.multi_fitness for sol in pareto_sols]
                    current_pareto_solutions = pareto_sols

            # Build callback state dictionary
            callback_state = {
                "generation": gen,
                "population": population,
                "best_solution": best_solution,
                "fitness_history": fitness_history,
                "no_improvement_count": no_improvement_count,
                "pareto_front": current_pareto_front,
                "pareto_solutions": current_pareto_solutions,
                "n_stat": n_stat,
                "control": control,
                "data": data,
                "candidates": candidates,
                "setsizes": setsizes,
                "settypes": settypes
            }

            # Call the user's callback
            try:
                control["callback"](callback_state)
            except Exception as e:
                if show_progress:
                    print(f"Warning: Callback raised exception at generation {gen}: {e}")

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
        # Use fast_non_dominated_sort to get the true Pareto front
        fronts = fast_non_dominated_sort(population)
        pareto_solutions = fronts[0] if fronts else []  # First front is the Pareto front
        
        # Apply solution diversity filtering if enabled
        if solution_diversity and pareto_solutions:
            unique_pareto = []
            seen_hashes = set()
            
            for sol in pareto_solutions:
                sol_hash = sol.get_hash()
                if sol_hash not in seen_hashes:
                    unique_pareto.append(sol)
                    seen_hashes.add(sol_hash)
            
            pareto_solutions = unique_pareto
        
        # Extract fitness values for the Pareto front
        pareto_front = [sol.multi_fitness for sol in pareto_solutions]
        
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
        # Convert solution dicts to Solution objects for fast_non_dominated_sort
        combined_sol_objects = []

        for island_result in island_results:
            if "pareto_solutions" in island_result:
                for sol_dict in island_result["pareto_solutions"]:
                    sol_obj = Solution(
                        int_values=sol_dict["selected_indices"],
                        dbl_values=sol_dict["selected_values"],
                        multi_fitness=sol_dict["multi_fitness"],
                        fitness=sum(sol_dict["multi_fitness"])
                    )
                    combined_sol_objects.append(sol_obj)

        # Use fast_non_dominated_sort to efficiently find the Pareto front
        if combined_sol_objects:
            fronts = fast_non_dominated_sort(combined_sol_objects)
            pareto_sol_objects = fronts[0] if fronts else []
        else:
            pareto_sol_objects = []

        # Apply diversity filtering if enabled
        if solution_diversity and pareto_sol_objects:
            unique_pareto = []
            seen_hashes = set()

            for sol in pareto_sol_objects:
                sol_hash = sol.get_hash()
                if sol_hash not in seen_hashes:
                    unique_pareto.append(sol)
                    seen_hashes.add(sol_hash)

            pareto_sol_objects = unique_pareto

        # Convert back to dict format for result
        pareto_front = [sol.multi_fitness for sol in pareto_sol_objects]
        pareto_solutions = [
            {
                "selected_indices": sol.int_values,
                "selected_values": sol.dbl_values,
                "multi_fitness": sol.multi_fitness
            }
            for sol in pareto_sol_objects
        ]
        
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
