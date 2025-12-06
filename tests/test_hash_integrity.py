import random

import numpy as np

from trainselpy.solution import Solution
from trainselpy.operators import mutation


def test_solution_hash_changes_after_manual_invalidated_edit():
    """Hash should change when genome is edited and hash is invalidated."""
    sol = Solution(int_values=[[0, 1, 2]], dbl_values=[[0.1, 0.2]])

    h_before = sol.get_hash()

    # Manually edit genome and invalidate hash
    sol.int_values[0][0] = 99
    sol.invalidate_hash()

    h_after = sol.get_hash()
    assert h_before != h_after


def test_mutation_invalidates_hash_when_genome_changes():
    """
    When mutation modifies a solution, its cached hash must be invalidated
    so that get_hash() reflects the new genome.
    """
    np.random.seed(123)
    random.seed(123)

    # One solution with both BOOL and DBL genes so both paths are exercised
    sol = Solution(
        int_values=[[0, 0, 1, 1]],  # BOOL-style
        dbl_values=[[0.2, 0.4, 0.6]]
    )
    population = [sol]

    candidates = [[0, 1, 2, 3]]  # Only used for non-BOOL int sets
    settypes = ["BOOL"]

    h_before = sol.get_hash()

    # Force mutation on all positions
    mutation(population, candidates, settypes, mutprob=1.0, mutintensity=0.5)

    h_after = sol.get_hash()

    # With mutprob=1.0 for BOOL and non-zero mutintensity for DBL, the genome
    # should change and the hash must change accordingly.
    assert h_before != h_after

