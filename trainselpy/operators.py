"""
Genetic algorithm operators for TrainSelPy.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
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


def _pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Partially Mapped Crossover (PMX) for ordered sets (permutations).
    Preserves the permutation property - no duplicates, all values present.
    
    Parameters
    ----------
    parent1 : List[int]
        First parent permutation
    parent2 : List[int]
        Second parent permutation
        
    Returns
    -------
    Tuple[List[int], List[int]]
        Two offspring permutations
    """
    size = len(parent1)
    
    # Select two random crossover points
    cx_point1 = random.randint(0, size - 1)
    cx_point2 = random.randint(0, size - 1)
    
    if cx_point1 > cx_point2:
        cx_point1, cx_point2 = cx_point2, cx_point1
    
    # Initialize offspring as copies of parents
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Create mapping between the crossover segments
    mapping1 = {}
    mapping2 = {}
    
    for i in range(cx_point1, cx_point2 + 1):
        # Swap the segment
        child1[i] = parent2[i]
        child2[i] = parent1[i]
        
        # Build mappings
        mapping1[parent2[i]] = parent1[i]
        mapping2[parent1[i]] = parent2[i]
    
    # Fix conflicts outside the crossover segment
    for i in list(range(0, cx_point1)) + list(range(cx_point2 + 1, size)):
        # Fix child1
        val = child1[i]
        while val in mapping1:
            val = mapping1[val]
        child1[i] = val
        
        # Fix child2
        val = child2[i]
        while val in mapping2:
            val = mapping2[val]
        child2[i] = val
    
    return child1, child2



def _crossover_int_values(
    parent1_vals: List[List[int]],
    parent2_vals: List[List[int]],
    child1_vals: List[List[int]],
    child2_vals: List[List[int]],
    crossintensity: float,
    settypes: Optional[List[str]] = None,
    candidates: Optional[List[List[int]]] = None
) -> None:
    """
    Helper for integer crossover.

    Internally uses NumPy arrays for segment swapping where beneficial,
    but preserves the external list-of-lists representation.
    """
    if not parent1_vals or not parent2_vals:
        return

    n_sets = len(parent1_vals)
    for j in range(n_sets):
        size = len(parent1_vals[j])
        if size <= 1:
            continue

        if settypes and j < len(settypes) and settypes[j] == "OS":
            # Ordered permutation: keep original PMX logic on lists
            child1_vals[j], child2_vals[j] = _pmx_crossover(parent1_vals[j], parent2_vals[j])
            continue

        # Vectorized-style crossover for other types using NumPy views
        n_points = max(1, int(size * crossintensity))
        n_points = min(n_points, size - 1)
        if n_points <= 0:
            continue

        # Use NumPy to generate sorted unique crossover points efficiently
        points = np.sort(
            np.random.choice(np.arange(1, size, dtype=int), size=n_points, replace=False)
        ).tolist()

        # Work on NumPy copies for segment operations
        c1 = np.asarray(child1_vals[j], dtype=int)
        c2 = np.asarray(child2_vals[j], dtype=int)

        for k, point in enumerate(points):
            if k % 2 != 0:
                continue
            start = points[k - 1] if k > 0 else 0
            end = point
            # Swap segments using NumPy slicing
            tmp = c1[start:end].copy()
            c1[start:end] = c2[start:end]
            c2[start:end] = tmp

        # Convert back to Python lists for downstream code
        child1_vals[j] = c1.tolist()
        child2_vals[j] = c2.tolist()

        # Post-processing per set type
        if settypes and j < len(settypes):
            stype = settypes[j]
            if stype in ["UOS", "UOMS"]:
                child1_vals[j].sort()
                child2_vals[j].sort()

            if stype == "UOS" and candidates and j < len(candidates):
                # Efficient duplicate repair using sets and _get_valid_replacement
                for child_val in (child1_vals, child2_vals):
                    vals = child_val[j]
                    original_len = len(vals)
                    unique_values = list(dict.fromkeys(vals))
                    if len(unique_values) < original_len:
                        current_set = set(unique_values)
                        n_missing = original_len - len(unique_values)
                        cand = candidates[j]
                        for _ in range(n_missing):
                            new_val = _get_valid_replacement(None, current_set, cand, stype)
                            unique_values.append(new_val)
                            current_set.add(new_val)
                    child_val[j] = sorted(unique_values)

def _crossover_dbl_values(
    parent1_vals: List[List[float]],
    parent2_vals: List[List[float]],
    child1_vals: List[List[float]],
    child2_vals: List[List[float]],
    crossintensity: float
) -> None:
    """
    Helper for double crossover.

    Uses NumPy arrays for efficient segment swapping but keeps the
    list-of-lists API intact.
    """
    if not parent1_vals or not parent2_vals:
        return

    n_sets = len(parent1_vals)
    for j in range(n_sets):
        size = len(parent1_vals[j])
        if size <= 1:
            continue

        n_points = max(1, int(size * crossintensity))
        n_points = min(n_points, size - 1)
        if n_points <= 0:
            continue

        points = np.sort(
            np.random.choice(np.arange(1, size, dtype=int), size=n_points, replace=False)
        ).tolist()

        c1 = np.asarray(child1_vals[j], dtype=float)
        c2 = np.asarray(child2_vals[j], dtype=float)

        for k, point in enumerate(points):
            if k % 2 != 0:
                continue
            start = points[k - 1] if k > 0 else 0
            end = point
            tmp = c1[start:end].copy()
            c1[start:end] = c2[start:end]
            c2[start:end] = tmp

        child1_vals[j] = c1.tolist()
        child2_vals[j] = c2.tolist()

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

                # Invalidate cached hashes since genomes have changed
                child1.invalidate_hash()
                child2.invalidate_hash()
                
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

    This implementation uses NumPy arrays internally to apply mutations
    across the population in a more vectorized fashion while preserving
    the existing public API and semantics.
    """
    if not population:
        return

    pop_size = len(population)

    # ----- Integer-valued genes -----
    # We process each int set index i across the whole population in a batched way.
    max_int_sets = max(len(sol.int_values) for sol in population)

    for i in range(max_int_sets):
        stype = settypes[i]

        # Collect this int set across all solutions that have it
        cols = []
        idx_map = []
        for s_idx, sol in enumerate(population):
            if i < len(sol.int_values):
                cols.append(sol.int_values[i])
                idx_map.append(s_idx)
        if not cols:
            continue

        arr = np.asarray(cols, dtype=int)
        n_rows, n_cols = arr.shape

        if stype == "OS":
            # Ordered permutations: swap-based mutation
            # Random mask of positions to mutate
            mask = np.random.rand(n_rows, n_cols) < mutprob
            # For each row, swap selected positions with random other positions
            for r in range(n_rows):
                row = arr[r]
                pos_indices = np.where(mask[r])[0]
                if pos_indices.size == 0:
                    continue
                # Choose random swap positions for each selected index
                swap_pos = np.random.randint(0, n_cols, size=pos_indices.size)
                for pos, sp in zip(pos_indices, swap_pos):
                    if pos != sp:
                        row[pos], row[sp] = row[sp], row[pos]
            # Write back
            for r, s_idx in enumerate(idx_map):
                population[s_idx].int_values[i] = arr[r].tolist()
                population[s_idx].invalidate_hash()
        else:
            # Standard mutation for BOOL / set types
            mask = np.random.rand(n_rows, n_cols) < mutprob

            if stype == "BOOL":
                # Flip bits using XOR-like behavior
                arr[mask] = 1 - arr[mask]
                for r, s_idx in enumerate(idx_map):
                    population[s_idx].int_values[i] = arr[r].tolist()
                    population[s_idx].invalidate_hash()
            else:
                # Set types: UOS, UOMS, OMS
                cand = candidates[i]
                cand_arr = np.asarray(cand, dtype=int)

                for r, s_idx in enumerate(idx_map):
                    row = arr[r]
                    row_mask = mask[r]
                    if not np.any(row_mask):
                        continue

                    if stype in ["UOS", "OS"]:
                        current_set = set(row.tolist())
                    else:
                        current_set = set()

                    # Mutate each marked position
                    positions = np.where(row_mask)[0]
                    for pos in positions:
                        old_val = int(row[pos])
                        new_val = _get_valid_replacement(old_val, current_set, cand, stype)
                        row[pos] = new_val
                        if stype in ["UOS", "OS"]:
                            current_set.discard(old_val)
                            current_set.add(new_val)

                    # Post-process ordering for unordered sets
                    new_list = row.tolist()
                    if stype in ["UOS", "UOMS"]:
                        new_list.sort()
                    population[s_idx].int_values[i] = new_list
                    population[s_idx].invalidate_hash()

    # ----- Double-valued genes -----
    # For continuous variables we keep the original per-gene mutation
    # semantics to preserve convergence behavior on benchmark problems.
    for sol in population:
        dbl_mutated = False
        for i, values in enumerate(sol.dbl_values):
            for pos in range(len(values)):
                if random.random() < mutprob:
                    delta = np.random.normal(0, mutintensity)
                    new_val = values[pos] + delta
                    # Ensure the value is in the valid range [0, 1]
                    if new_val < 0.0:
                        new_val = 0.0
                    elif new_val > 1.0:
                        new_val = 1.0
                    sol.dbl_values[i][pos] = new_val
                    dbl_mutated = True

        if dbl_mutated:
            sol.invalidate_hash()
