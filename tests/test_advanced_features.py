"""
Verification script for TrainSelPy advanced features.
"""

import numpy as np
import time
from trainselpy.genetic_algorithm import genetic_algorithm
from trainselpy.solution import Solution
from trainselpy.cma_es import CMAESOptimizer
from trainselpy.surrogate import SurrogateModel
from trainselpy.nsga3 import generate_reference_points

def test_cma_es():
    print("\n--- Testing CMA-ES Integration ---")
    # Problem: Minimize sphere function (Maximize -sum(x^2))
    # 5 continuous variables
    n_vars = 5
    # 5 continuous variables in 1 set
    n_vars = 5
    candidates = [[]] 
    setsizes = [n_vars]
    settypes = ["DBL"]
    
    def sphere_fitness(dbl_vals, data):
        # dbl_vals is the list of values for the set
        x = np.array(dbl_vals)
        # Target is 0.5 (since range is [0,1])
        return -np.sum((x - 0.5)**2)
    
    control = {
        "niterations": 50,
        "npop": 20,
        "progress": True
    }
    
    result = genetic_algorithm(
        data={},
        candidates=candidates,
        setsizes=setsizes,
        settypes=settypes,
        stat_func=sphere_fitness,
        control=control
    )
    
    print(f"Best fitness: {result['fitness']}")
    print(f"Best solution: {result['selected_values']}")
    # Should be close to 0.5
    vals = np.concatenate(result['selected_values'])
    if np.allclose(vals, 0.5, atol=0.1):
        print("CMA-ES Test PASSED")
    else:
        print("CMA-ES Test FAILED (Convergence issue)")


def test_surrogate():
    print("\n--- Testing Surrogate Optimization ---")
    # Problem: Maximize sum of selected indices
    # 10 candidates, select 3
    candidates = [list(range(10))]
    setsizes = [3]
    settypes = ["UOS"]
    
    def expensive_fitness(int_vals, data):
        # Simulate expense
        # time.sleep(0.01) 
        # int_vals is [1, 2, 3]
        return float(sum(int_vals))
    
    control = {
        "niterations": 20,
        "npop": 20,
        "use_surrogate": True,
        "surrogate_start_gen": 2,
        "surrogate_update_freq": 2,
        "progress": True
    }
    
    try:
        result = genetic_algorithm(
            data={},
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat_func=expensive_fitness,
            control=control
        )
        print(f"Best fitness: {result['fitness']}")
        # Max sum of 3 distinct from 0..9 is 9+8+7 = 24
        if result['fitness'] == 24:
            print("Surrogate Test PASSED")
        else:
            print(f"Surrogate Test FAILED (Suboptimal: {result['fitness']})")
    except Exception as e:
        print(f"Surrogate Test CRASHED: {e}")


def test_nsga3():
    print("\n--- Testing NSGA-III ---")
    # Problem: DTLZ1-like (3 objectives)
    # 3 continuous variables in 1 set
    n_vars = 3
    candidates = [[]]
    setsizes = [n_vars]
    settypes = ["DBL"]
    
    def multi_obj_fitness(dbl_vals, data):
        x = np.array(dbl_vals)
        f1 = x[0]
        f2 = x[1]
        # f3 conflicts with f1 and f2
        f3 = 2.0 - x[0] - x[1]
        # Maximize all
        return [f1, f2, f3]
    
    control = {
        "niterations": 20,
        "npop": 50,
        "use_nsga3": True,
        "progress": True
    }
    
    try:
        result = genetic_algorithm(
            data={},
            candidates=candidates,
            setsizes=setsizes,
            settypes=settypes,
            stat_func=multi_obj_fitness,
            n_stat=3,
            control=control
        )
        
        pareto_front = result['pareto_front']
        print(f"Pareto front size: {len(pareto_front)}")
        if len(pareto_front) > 0:
            print("NSGA-III Test PASSED")
        else:
            print("NSGA-III Test FAILED (Empty front)")
    except Exception as e:
        print(f"NSGA-III Test CRASHED: {e}")


if __name__ == "__main__":
    test_cma_es()
    test_surrogate()
    test_nsga3()
