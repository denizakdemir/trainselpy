"""
Genetic algorithm selection logic for TrainSelPy.
"""

import random
from typing import List
from trainselpy.solution import Solution

def fast_non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
    """
    Fast non-dominated sorting algorithm (NSGA-II style).
    Significantly faster than naive approach for multi-objective optimization.

    Parameters
    ----------
    population : List[Solution]
        List of solutions with multi_fitness attributes

    Returns
    -------
    List[List[Solution]]
        List of Pareto fronts, where fronts[0] is the best (non-dominated) front
    """
    n = len(population)

    # For each solution, track:
    # - domination_count: how many solutions dominate it
    # - dominated_solutions: which solutions it dominates
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]

    fronts = [[]]

    # Compare all pairs of solutions
    for i in range(n):
        for j in range(i + 1, n):
            sol_i = population[i]
            sol_j = population[j]

            # Check if i dominates j
            i_dominates_j = (all(sol_i.multi_fitness[k] >= sol_j.multi_fitness[k]
                                for k in range(len(sol_i.multi_fitness))) and
                           any(sol_i.multi_fitness[k] > sol_j.multi_fitness[k]
                               for k in range(len(sol_i.multi_fitness))))

            # Check if j dominates i
            j_dominates_i = (all(sol_j.multi_fitness[k] >= sol_i.multi_fitness[k]
                                for k in range(len(sol_i.multi_fitness))) and
                           any(sol_j.multi_fitness[k] > sol_i.multi_fitness[k]
                               for k in range(len(sol_i.multi_fitness))))

            if i_dominates_j:
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif j_dominates_i:
                dominated_solutions[j].append(i)
                domination_count[i] += 1

        # If solution i is not dominated by anyone, it's in the first front
        if domination_count[i] == 0:
            fronts[0].append(population[i])

    # Build subsequent fronts
    front_idx = 0
    while fronts[front_idx]:
        next_front = []
        for sol in fronts[front_idx]:
            # Get the index of this solution
            sol_idx = population.index(sol)
            # For each solution that this one dominates
            for dominated_idx in dominated_solutions[sol_idx]:
                domination_count[dominated_idx] -= 1
                # If domination count becomes 0, it belongs to the next front
                if domination_count[dominated_idx] == 0:
                    next_front.append(population[dominated_idx])

        front_idx += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def calculate_crowding_distance(front: List[Solution]) -> List[float]:
    """
    Calculate crowding distance for solutions in a front (NSGA-II style).

    Parameters
    ----------
    front : List[Solution]
        Solutions in the same Pareto front

    Returns
    -------
    List[float]
        Crowding distance for each solution (higher is better for diversity)
    """
    n = len(front)
    if n <= 2:
        return [float('inf')] * n  # Boundary solutions get infinite distance

    n_obj = len(front[0].multi_fitness)
    distances = [0.0] * n

    # For each objective
    for m in range(n_obj):
        # Sort by this objective
        sorted_indices = sorted(range(n), key=lambda i: front[i].multi_fitness[m])

        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # Normalize by objective range
        obj_min = front[sorted_indices[0]].multi_fitness[m]
        obj_max = front[sorted_indices[-1]].multi_fitness[m]
        obj_range = obj_max - obj_min

        if obj_range > 0:
            # Calculate crowding distance for middle solutions
            for i in range(1, n - 1):
                if distances[sorted_indices[i]] != float('inf'):
                    distances[sorted_indices[i]] += (
                        (front[sorted_indices[i + 1]].multi_fitness[m] -
                         front[sorted_indices[i - 1]].multi_fitness[m]) / obj_range
                    )

    return distances


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
        # Use fast non-dominated sorting (NSGA-II style) - much more efficient
        fronts = fast_non_dominated_sort(population)

        # Select elite solutions from the fronts
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= n_elite:
                # Entire front fits
                selected.extend([sol.copy() for sol in front])
            else:
                # Partial front - use crowding distance to select most diverse solutions
                crowding_distances = calculate_crowding_distance(front)

                # Sort by crowding distance (higher is better for diversity)
                sorted_indices = sorted(range(len(front)),
                                      key=lambda i: crowding_distances[i],
                                      reverse=True)

                # Select most diverse solutions
                n_to_select = n_elite - len(selected)
                for i in range(n_to_select):
                    selected.append(front[sorted_indices[i]].copy())
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
