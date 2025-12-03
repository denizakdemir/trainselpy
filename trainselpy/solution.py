"""
Solution class and helpers for TrainSelPy.
"""

import numpy as np
from typing import List, Dict, Callable, Union, Optional, Any, Tuple

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
        self._hash = None  # Cache for hash value

    def copy(self):
        """Create a deep copy of the solution."""
        # Copy integer values (each sublist is explicitly copied)
        int_copy = [list(x) for x in self.int_values]
        # Copy double values similarly
        dbl_copy = [list(x) for x in self.dbl_values]
        # Always copy multi_fitness as a list (even if empty)
        multi_fit_copy = self.multi_fitness.copy()
        return Solution(int_copy, dbl_copy, self.fitness, multi_fit_copy)

    def get_hash(self):
        """
        Get a hash of the solution for caching.
        Uses only int_values for hashing as dbl_values may have precision issues.
        """
        if self._hash is None:
            # Create hash from integer and double values
            int_tuple = tuple(tuple(iv) for iv in self.int_values)
            # Round doubles to avoid precision issues, but ensure mutated values (which change significantly) are different
            # Using 8 decimals should be safe for caching within a run
            dbl_tuple = tuple(tuple(round(v, 8) for v in dv) for dv in self.dbl_values)
            self._hash = hash((int_tuple, dbl_tuple))
        return self._hash

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
