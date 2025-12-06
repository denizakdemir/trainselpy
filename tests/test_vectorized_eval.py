import numpy as np
import random

from trainselpy.genetic_algorithm import genetic_algorithm


def test_vectorized_single_objective_matches_scalar():
    """
    Verify that a vectorized objective function (called once per generation)
    produces the same result as a scalar objective evaluated per solution.
    """
    # Simple integer-only problem: maximize sum of selected indices
    candidates = [list(range(10))]
    setsizes = [3]
    settypes = ["UOS"]

    def scalar_stat(int_vals, data):
        # int_vals is a single solution
        return float(sum(int_vals))

    def vectorized_stat(int_solutions, data):
        """
        Vectorized objective matching the single-solution interface.

        During GA evaluation (with vectorized_stat=True) this will receive a
        list of solutions and should return a 1D array of fitness values.
        When called from simulated annealing (single solution), it should
        behave like the scalar version.
        """
        arr = np.asarray(int_solutions, dtype=float)
        if arr.ndim == 1:
            return float(arr.sum())
        return arr.sum(axis=1)

    control = {
        "niterations": 30,
        "npop": 40,
        "progress": False
    }

    # Run GA with scalar objective
    np.random.seed(123)
    random.seed(123)
    res_scalar = genetic_algorithm(
        data={},
        candidates=candidates,
        setsizes=setsizes,
        settypes=settypes,
        stat_func=scalar_stat,
        control=control
    )

    # Run GA with vectorized objective
    np.random.seed(123)
    random.seed(123)
    control_vec = control.copy()
    control_vec["vectorized_stat"] = True
    res_vec = genetic_algorithm(
        data={},
        candidates=candidates,
        setsizes=setsizes,
        settypes=settypes,
        stat_func=vectorized_stat,
        control=control_vec
    )

    # Best fitness and solution should match
    assert np.isclose(res_scalar["fitness"], res_vec["fitness"])
    assert res_scalar["selected_indices"] == res_vec["selected_indices"]
