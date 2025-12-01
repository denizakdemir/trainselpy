"""
Genetic algorithm operators for TrainSelPy.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Set
from trainselpy.solution import Solution

def _get_valid_replacement(
    current_val: Optional[int],
    current_set: Set[int],
    candidates: List[int],
    set_type: str,
    max_attempts: int = 20
) -> int:
    """
    Select a valid replacement value using rejection sampling with fallback.
    
    Parameters
    ----------
    current_val : Optional[int]
        The value being replaced (if any).
    current_set : Set[int]
        The set of values currently in the solution (for uniqueness checks).
    candidates : List[int]
        List of candidate values.
    set_type : str
        Type of the set (UOS, OS, UOMS, OMS).
    max_attempts : int
        Maximum number of rejection sampling attempts before fallback.
        
    Returns
    -------
    int
        A valid replacement value.
    """
    # For multisets (UOMS, OMS), any candidate is valid
    if set_type in ["UOMS", "OMS"]:
        return random.choice(candidates)

    # For unique sets (UOS, OS), we need a value not in current_set and not current_val
    # Rejection sampling
    n_candidates = len(candidates)
    
    # If candidates are much larger than the set size, rejection sampling is efficient
    # If the set is nearly full (close to n_candidates), it might be slow, so we fallback
    
    for _ in range(max_attempts):
        # Pick a random candidate
        # Optimization: if candidates is a range or simple list, random.choice is fast
        # If candidates is very large, random.choice is O(1) if it's a list
        val = random.choice(candidates)
        
        if val != current_val and val not in current_set:
            return val
            
    # Fallback: calculate available candidates explicitly
    # This is expensive but guarantees a result
    available = [c for c in candidates if c != current_val and c not in current_set]
    if not available:
        # This should theoretically not happen if set size < n_candidates
        # But if it does, return current_val or a random one to avoid crash
        return current_val if current_val is not None else random.choice(candidates)
        
    return random.choice(available)


def _crossover_int_values(
    parent1_vals: List[List[int]],
    parent2_vals: List[List[int]],
    child1_vals: List[List[int]],
    child2_vals: List[List[int]],
    crossintensity: float,
    settypes: Optional[List[str]] = None,
    candidates: Optional[List[List[int]]] = None
) -> None:
    """Helper for integer crossover."""
    for j in range(len(parent1_vals)):
        if parent1_vals and parent2_vals:
            # Calculate crossover points
            size = len(parent1_vals[j])
            n_points = max(1, int(size * crossintensity))
            
            # Select crossover points
            points = sorted(random.sample(range(1, size), min(n_points, size - 1)))
            
            # Perform crossover
            for k in range(len(points)):
                if k % 2 == 0:
                    # Swap segments
                    start = points[k - 1] if k > 0 else 0
                    end = points[k]
                    
                    temp = child1_vals[j][start:end]
                    child1_vals[j][start:end] = child2_vals[j][start:end]
                    child2_vals[j][start:end] = temp
                    
                    # Check set type and fix if needed
                    if settypes and j < len(settypes):
                        # Fix unordered sets by sorting
                        if settypes[j] in ["UOS", "UOMS"]:
                            child1_vals[j].sort()
                            child2_vals[j].sort()
                        
                        # Fix sets without repetition by removing duplicates
                        if settypes[j] in ["UOS", "OS"] and candidates and j < len(candidates):
                            for child_val in [child1_vals, child2_vals]:
                                # Fix child - optimized set-based approach
                                original_len = len(child_val[j])
                                unique_values = list(dict.fromkeys(child_val[j]))  # Preserves order, removes dups

                                if len(unique_values) < original_len:
                                    # Need to fill in missing values
                                    current_set = set(unique_values)
                                    n_missing = original_len - len(unique_values)
                                    
                                    # Use optimized replacement
                                    cand = candidates[j]
                                    stype = settypes[j]
                                    
                                    for _ in range(n_missing):
                                        new_val = _get_valid_replacement(None, current_set, cand, stype)
                                        unique_values.append(new_val)
                                        current_set.add(new_val)

                                child_val[j] = unique_values

                                # Re-sort if needed (only once at the end)
                                if settypes[j] == "UOS":
                                    child_val[j].sort()

def _crossover_dbl_values(
    parent1_vals: List[List[float]],
    parent2_vals: List[List[float]],
    child1_vals: List[List[float]],
    child2_vals: List[List[float]],
    crossintensity: float
) -> None:
    """Helper for double crossover."""
    for j in range(len(parent1_vals)):
        if parent1_vals and parent2_vals:
            # Calculate crossover points
            size = len(parent1_vals[j])
            n_points = max(1, int(size * crossintensity))
            
            # Select crossover points
            points = sorted(random.sample(range(1, size), min(n_points, size - 1)))
            
            # Perform crossover
            for k in range(len(points)):
                if k % 2 == 0:
                    # Swap segments
                    start = points[k - 1] if k > 0 else 0
                    end = points[k]
                    
                    temp = child1_vals[j][start:end]
                    child1_vals[j][start:end] = child2_vals[j][start:end]
                    child2_vals[j][start:end] = temp

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
                _crossover_int_values(
                    parent1.int_values, parent2.int_values,
                    child1.int_values, child2.int_values,
                    crossintensity, settypes, candidates
                )
                
                # Crossover double values
                _crossover_dbl_values(
                    parent1.dbl_values, parent2.dbl_values,
                    child1.dbl_values, child2.dbl_values,
                    crossintensity
                )
                
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
                        stype = settypes[i]
                        
                        # Use optimized replacement
                        if stype in ["UOS", "OS"]:
                            current_set = set(sol.int_values[i])
                        else:
                            current_set = set() # Not used for multisets
                            
                        new_val = _get_valid_replacement(old_val, current_set, cand, stype)
                        sol.int_values[i][pos] = new_val
                        
                        # For unordered sets, ensure the values are sorted
                        if stype in ["UOS", "UOMS"]:
                            sol.int_values[i].sort()
        
        # Mutation for double values
        for i, values in enumerate(sol.dbl_values):
            
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
