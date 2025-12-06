"""
Solution class and helpers for TrainSelPy.
"""

import numpy as np
from typing import List, Dict, Callable, Union, Optional, Any, Tuple

class Solution:
    """
    Class to represent a solution in the genetic algorithm.

    Attributes
    ----------
    int_values : List[List[int]]
        Integer-valued decision variables (for UOS, OS, UOMS, OMS, BOOL set types)
    dbl_values : List[List[float]]
        Continuous decision variables (for DBL set type)
    fitness : float
        Scalar fitness value (sum of objectives for multi-objective)
    multi_fitness : List[float]
        Individual objective values for multi-objective optimization
    """
    def __init__(
        self,
        int_values: List[List[int]] = None,
        dbl_values: List[List[float]] = None,
        fitness: float = float('-inf'),
        multi_fitness: List[float] = None
    ):
        # Normalize inputs to lists (handle numpy arrays from initialization)
        if int_values is None:
            self.int_values = []
        else:
            self.int_values = [
                list(x) if not isinstance(x, list) else x
                for x in int_values
            ]

        if dbl_values is None:
            self.dbl_values = []
        else:
            self.dbl_values = []
            for x in dbl_values:
                if isinstance(x, np.ndarray):
                    self.dbl_values.append(x.tolist())
                elif isinstance(x, list):
                    self.dbl_values.append(x)
                else:
                    self.dbl_values.append(list(x))

        self.fitness = float(fitness) if fitness is not None else float('-inf')
        self.multi_fitness = list(multi_fitness) if multi_fitness is not None else []

        # Cache for hash value to avoid repeated computation (performance optimization)
        self._hash_cache = None
        self._hash_valid = False

    def copy(self):
        """Create a deep copy of the solution."""
        # Copy integer values (each sublist is explicitly copied)
        int_copy = [list(x) if isinstance(x, list) else x.tolist() for x in self.int_values]
        # Copy double values - handle both lists and numpy arrays
        dbl_copy = []
        for x in self.dbl_values:
            if isinstance(x, np.ndarray):
                dbl_copy.append(x.tolist())
            elif isinstance(x, list):
                dbl_copy.append(list(x))
            else:
                # Handle other iterables
                dbl_copy.append(list(x))
        # Always copy multi_fitness as a list (even if empty)
        multi_fit_copy = self.multi_fitness.copy() if self.multi_fitness else []
        new_sol = Solution(int_copy, dbl_copy, self.fitness, multi_fit_copy)

        # Copy cached hash if it's valid (performance optimization)
        if self._hash_valid:
            new_sol._hash_cache = self._hash_cache
            new_sol._hash_valid = True

        return new_sol

    def get_hash(self):
        """
        Get a hash of the solution for caching.

        Uses cached hash value for performance. Hash is invalidated when solution
        is modified (mutation/crossover create new solutions, so hash remains valid).
        """
        if self._hash_valid and self._hash_cache is not None:
            return self._hash_cache

        # Create hash from integer and double values
        int_tuple = tuple(tuple(iv) for iv in self.int_values)
        # Round doubles to avoid precision issues, but ensure mutated values (which change significantly) are different
        # Using 8 decimals should be safe for caching within a run
        dbl_tuple = tuple(tuple(round(v, 8) for v in dv) for dv in self.dbl_values)

        self._hash_cache = hash((int_tuple, dbl_tuple))
        self._hash_valid = True
        return self._hash_cache

    def invalidate_hash(self):
        """Invalidate cached hash (call after in-place modification)."""
        self._hash_valid = False
        self._hash_cache = None

    def __lt__(self, other):
        """Comparison for sorting (by fitness)."""
        return self.fitness < other.fitness


def flatten_dbl_values(dbl_values: List[List[float]]) -> np.ndarray:
    """Flatten double values into a single numpy array."""
    if not dbl_values:
        return np.array([])
    return np.concatenate([np.array(x) for x in dbl_values])


def unflatten_dbl_values(flat_values: np.ndarray, template: List[List[float]]) -> List[List[float]]:
    """Unflatten numpy array back into list of lists structure."""
    result = []
    idx = 0
    for sublist in template:
        size = len(sublist)
        result.append(flat_values[idx:idx+size].tolist())
        idx += size
    return result
