
import unittest
import numpy as np
from trainselpy import train_sel, set_control_default, make_data

import random

class TestConvergence(unittest.TestCase):
    """
    Convergence tests for TrainSelPy.
    These tests use simple problems with known optimal solutions to verify
    that the algorithm can recover them.
    """

    def setUp(self):
        np.random.seed(42)
        random.seed(42)
        # Common control parameters for fast convergence
        self.control = set_control_default()
        self.control["niterations"] = 500
        self.control["npop"] = 200
        self.control["nelite"] = 20
        self.control["mutprob"] = 0.2
        self.control["mutintensity"] = 0.1
        self.control["progress"] = False

    def test_uos_max_sum(self):
        """
        Test UOS (Unordered Set without replacement).
        Problem: Select k items to maximize the sum of their values.
        """
        n_items = 50
        k = 10
        values = np.arange(n_items) # 0, 1, ..., 49
        
        # True solution: the last k items [40, 41, ..., 49]
        true_sol = set(range(n_items - k, n_items))
        max_fitness = sum(true_sol)

        data = {"values": values}
        
        def fitness(sol, data):
            # sol is a list of indices
            return np.sum(data["values"][sol])

        result = train_sel(
            data=data,
            candidates=[list(range(n_items))],
            setsizes=[k],
            settypes=["UOS"],
            stat=fitness,
            control=self.control,
            verbose=False
        )

        best_sol = set(result.selected_indices[0])
        best_fitness = result.fitness
        
        self.assertEqual(best_fitness, max_fitness, "Did not find optimal fitness for UOS")
        self.assertEqual(best_sol, true_sol, "Did not find optimal solution for UOS")

    def test_uoms_max_sum(self):
        """
        Test UOMS (Unordered Multiset with replacement).
        Problem: Select k items to maximize sum.
        Optimal: Select the largest item k times.
        """
        n_items = 20
        k = 5
        values = np.arange(n_items) # 0, ..., 19
        
        # True solution: select 19, five times
        true_sol = sorted([19] * k)
        max_fitness = 19 * k

        data = {"values": values}
        
        def fitness(sol, data):
            return np.sum(data["values"][sol])

        result = train_sel(
            data=data,
            candidates=[list(range(n_items))],
            setsizes=[k],
            settypes=["UOMS"],
            stat=fitness,
            control=self.control,
            verbose=False
        )

        best_sol = sorted(result.selected_indices[0])
        best_fitness = result.fitness
        
        self.assertEqual(best_fitness, max_fitness, "Did not find optimal fitness for UOMS")
        self.assertEqual(best_sol, true_sol, "Did not find optimal solution for UOMS")

    def test_bool_knapsack(self):
        """
        Test BOOL (Binary selection).
        Problem: Simple Knapsack-like problem without weight constraint (just max sum).
        Actually, let's do a subset sum target.
        Target sum = 100. Values = [10, 20, 30, 40, 50, ...].
        Optimal: Select items that sum exactly to 100.
        Let's make it simpler: Maximize sum of positive values, avoid negatives.
        Values: [-10, 10, -20, 20, -30, 30]
        Optimal: Select indices 1, 3, 5. Sum = 60.
        """
        values = np.array([-10, 10, -20, 20, -30, 30])
        n_items = len(values)
        
        true_indices = [1, 3, 5]
        max_fitness = 60

        data = {"values": values}
        
        def fitness(sol, data):
            # sol is a binary vector [0, 1, 0, 1, ...]
            # Convert to boolean mask
            mask = np.array(sol, dtype=bool)
            if not any(mask): return -1e9
            return np.sum(data["values"][mask])

        # For BOOL, setsizes is ignored (or length of vector), candidates is usually just [0, 1] implied
        # In trainselpy, for BOOL, candidates[i] is the list of possible values? 
        # No, for BOOL, the "solution" is a list of 0/1 of length setsizes[0].
        # Let's check how BOOL is initialized.
        # initialize_population:
        # if settypes[j] == "BOOL":
        #    sol.int_values.append([random.choice([0, 1]) for _ in range(setsizes[j])])
        
        result = train_sel(
            data=data,
            candidates=[list(range(n_items))], # Candidates determine the size of the boolean vector
            setsizes=[n_items], # Ignored for BOOL but good practice
            settypes=["BOOL"],
            stat=fitness,
            control=self.control,
            verbose=False
        )
        
        best_sol_vector = result.selected_indices[0]
        best_fitness = result.fitness
        
        # Check if selected indices match true indices
        selected_indices = [i for i, x in enumerate(best_sol_vector) if x == 1]
        
        self.assertEqual(best_fitness, max_fitness, "Did not find optimal fitness for BOOL")
        self.assertEqual(sorted(selected_indices), sorted(true_indices), "Did not find optimal solution for BOOL")

    def test_dbl_sphere(self):
        """
        Test DBL (Continuous variables).
        Problem: Minimize Sphere function f(x) = sum(x^2).
        We maximize -sum(x^2).
        Optimal: x = [0, 0, ...], Fitness = 0.
        """
        dim = 5
        
        def fitness(sol_dbl, data):
            # sol_dbl is a list of values (unwrapped for single set)
            # or list of lists (for multiple sets)
            
            # Since we have 1 set, it's unwrapped? Let's check algorithms.py logic.
            # "dbl_arg = sol.dbl_values if len(sol.dbl_values) > 1 else sol.dbl_values[0]"
            # So it is unwrapped. It is a list of floats.
            vals = np.array(sol_dbl)
            # vals are in [0, 1]. Shift to [-0.5, 0.5] for sphere centered at 0.5?
            # Let's just target 0.5. Maximize -(x - 0.5)^2
            target = 0.5
            return -np.sum((vals - target)**2)

        # Optimal fitness is 0
        
        result = train_sel(
            data={},
            candidates=[[]], # No int candidates
            setsizes=[dim],
            settypes=["DBL"],
            stat=fitness,
            control=self.control,
            verbose=False
        )
        
        best_vals = np.array(result.selected_values[0])
        best_fitness = result.fitness
        
        # Check if close to 0.5
        np.testing.assert_allclose(best_vals, 0.5, atol=0.05, err_msg="Did not converge to 0.5 for DBL")
        self.assertGreater(best_fitness, -0.01, "Fitness should be close to 0")

    def test_mixed_uos_dbl(self):
        """
        Test Mixed (UOS + DBL).
        Problem: Select index i from [0..9] and value x in [0,1].
        Maximize: values[i] - (x - target_x)^2
        values = [0, 10, 20, ..., 90]
        Optimal: i=9 (value 90), x=target_x.
        """
        n_items = 10
        values = np.arange(n_items) * 10
        target_x = 0.75
        
        data = {"values": values}
        
        def fitness(sol_int, sol_dbl, data):
            # sol_int is unwrapped (list of ints) because 1 int set
            # sol_dbl is unwrapped (list of floats) because 1 dbl set
            
            idx = sol_int[0] # First item of the first set
            val = sol_dbl[0] # First item of the first set
            
            score = data["values"][idx] - 100 * (val - target_x)**2
            return score

        result = train_sel(
            data=data,
            candidates=[list(range(n_items)), []], # Int candidates, DBL empty
            setsizes=[1, 1], # 1 int, 1 dbl
            settypes=["UOS", "DBL"],
            stat=fitness,
            control=self.control,
            verbose=False
        )
        
        best_idx = result.selected_indices[0][0]
        best_val = result.selected_values[0][0]
        
        self.assertEqual(best_idx, 9, "Did not select optimal integer index")
        self.assertAlmostEqual(best_val, target_x, delta=0.05, msg="Did not converge to optimal double value")

    def test_moo_quadratic(self):
        """
        Test MOO (Multi-Objective Optimization).
        Problem: Bi-objective quadratic.
        Maximize f1(x) = -x^2
        Maximize f2(x) = -(x - 2)^2
        Variable x in [-1, 3].
        Pareto optimal set: x in [0, 2].
        
        Mapping: gene g in [0, 1] -> x = 4*g - 1.
        Pareto set in gene space:
        0 = 4g - 1 => g = 0.25
        2 = 4g - 1 => g = 0.75
        So optimal genes are in [0.25, 0.75].
        """
        
        def fitness(sol_dbl, data):
            # sol_dbl is list of floats (1 variable)
            g = sol_dbl[0]
            x = 4 * g - 1
            f1 = -x**2
            f2 = -(x - 2)**2
            return [f1, f2]

        # Enable solution diversity to get a good spread
        self.control["solution_diversity"] = True
        
        result = train_sel(
            data={},
            candidates=[[]],
            setsizes=[1],
            settypes=["DBL"],
            stat=fitness,
            n_stat=2,
            control=self.control,
            verbose=False
        )
        
        pareto_front = result.pareto_front
        pareto_solutions = result.pareto_solutions
        
        self.assertIsNotNone(pareto_front, "Pareto front should not be None")
        self.assertGreater(len(pareto_front), 5, "Should find multiple Pareto solutions")
        
        # Verify solutions are in valid range [0.25, 0.75] (approx [0, 2] in real space)
        # Allow small tolerance
        valid_count = 0
        for sol in pareto_solutions:
            g = sol["selected_values"][0][0] # DBL values are wrapped in list of lists in result?
            # Let's check result structure. 
            # In algorithms.py: "selected_values": best_solution.dbl_values
            # best_solution.dbl_values is List[List[float]].
            # So yes, sol["selected_values"][0][0] is correct.
            
            if 0.24 <= g <= 0.76:
                valid_count += 1
                
        # Most solutions should be in the Pareto set
        self.assertGreater(valid_count / len(pareto_solutions), 0.8, 
                           f"Most solutions should be in Pareto set [0.25, 0.75]. Found {valid_count}/{len(pareto_solutions)}")

if __name__ == "__main__":
    unittest.main()
