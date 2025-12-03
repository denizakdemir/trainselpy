
import unittest
import numpy as np
import random
from trainselpy import train_sel, set_control_default

class TestConvergenceMedium(unittest.TestCase):
    """
    Medium difficulty convergence tests for TrainSelPy.
    These tests target problems known to cause stagnation or premature convergence.
    """

    def setUp(self):
        np.random.seed(42)
        random.seed(42)
        # Control parameters - starting with robust settings
        self.control = set_control_default()
        self.control["niterations"] = 1000  # More iterations for harder problems
        self.control["npop"] = 500          # Larger population for exploration
        self.control["nelite"] = 20         # Keep elitism low to avoid premature convergence
        self.control["mutprob"] = 0.2       # High mutation probability
        self.control["mutintensity"] = 0.1  # Moderate mutation step
        self.control["progress"] = False
        self.control["solution_diversity"] = True

    def test_rosenbrock(self):
        """
        Test Rosenbrock function (Continuous).
        f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
        Global minimum at x = 1, f(x) = 0.
        Search space: [-2.048, 2.048]
        """
        dim = 5
        
        def fitness(sol_dbl, data):
            x = np.array(sol_dbl)
            # Map [0, 1] to [-2.048, 2.048]
            x_mapped = x * 4.096 - 2.048
            
            val = 0
            for i in range(dim - 1):
                val += 100 * (x_mapped[i+1] - x_mapped[i]**2)**2 + (1 - x_mapped[i])**2
            
            # We maximize negative fitness
            return -val

        # Rosenbrock needs more iterations to traverse the valley
        control = self.control.copy()
        control["niterations"] = 2000
        control["npop"] = 1000

        result = train_sel(
            data={},
            candidates=[[]],
            setsizes=[dim],
            settypes=["DBL"],
            stat=fitness,
            control=control,
            verbose=False
        )
        
        best_fitness = result.fitness
        best_vals = np.array(result.selected_values[0])
        best_vals_mapped = best_vals * 4.096 - 2.048
        
        # Rosenbrock is hard, so we accept a small error
        print(f"Rosenbrock Best Fitness: {best_fitness}")
        print(f"Rosenbrock Best Solution: {best_vals_mapped}")
        
        self.assertGreater(best_fitness, -1.0, "Failed to converge on Rosenbrock (fitness < -1.0)")
        np.testing.assert_allclose(best_vals_mapped, 1.0, atol=0.2, err_msg="Did not converge to [1, 1, ...] for Rosenbrock")

    def test_rastrigin(self):
        """
        Test Rastrigin function (Continuous).
        f(x) = 10*d + sum(x[i]^2 - 10*cos(2*pi*x[i]))
        Global minimum at x = 0, f(x) = 0.
        Search space: [-5.12, 5.12]
        """
        dim = 5
        
        def fitness(sol_dbl, data):
            x = np.array(sol_dbl)
            # Map [0, 1] to [-5.12, 5.12]
            x_mapped = x * 10.24 - 5.12
            
            val = 10 * dim + np.sum(x_mapped**2 - 10 * np.cos(2 * np.pi * x_mapped))
            
            # We maximize negative fitness
            return -val

        result = train_sel(
            data={},
            candidates=[[]],
            setsizes=[dim],
            settypes=["DBL"],
            stat=fitness,
            control=self.control,
            verbose=False
        )
        
        best_fitness = result.fitness
        best_vals = np.array(result.selected_values[0])
        best_vals_mapped = best_vals * 10.24 - 5.12
        
        print(f"Rastrigin Best Fitness: {best_fitness}")
        print(f"Rastrigin Best Solution: {best_vals_mapped}")
        
        self.assertGreater(best_fitness, -1.0, "Failed to converge on Rastrigin (fitness < -1.0)")
        np.testing.assert_allclose(best_vals_mapped, 0.0, atol=0.2, err_msg="Did not converge to [0, 0, ...] for Rastrigin")

    def test_knapsack_medium(self):
        """
        Test Medium Knapsack (Discrete Constrained).
        50 items, correlated weights and values.
        Maximize value subject to weight <= capacity.
        """
        n_items = 50
        # Generate correlated weights and values
        # w ~ U[10, 100], v = w + U[-10, 10]
        # This makes the problem harder as value density is similar
        weights = np.linspace(10, 100, n_items)
        values = weights + np.sin(np.linspace(0, 10, n_items)) * 10
        
        capacity = np.sum(weights) * 0.5
        
        def fitness(sol, data):
            # sol is binary vector
            mask = np.array(sol, dtype=bool)
            total_weight = np.sum(weights[mask])
            total_value = np.sum(values[mask])
            
            if total_weight > capacity:
                # Penalty: reduce value significantly
                return total_value - 10 * (total_weight - capacity)
            return total_value

        # Use default control but ensure enough iterations
        control = self.control.copy()
        control["niterations"] = 500
        
        result = train_sel(
            data={},
            candidates=[list(range(n_items))],
            setsizes=[n_items],
            settypes=["BOOL"],
            stat=fitness,
            control=control,
            verbose=False
        )
        
        best_sol = np.array(result.selected_indices[0], dtype=bool)
        best_weight = np.sum(weights[best_sol])
        best_value = result.fitness
        
        print(f"Knapsack Best Value: {best_value}, Weight: {best_weight}/{capacity}")
        
        self.assertLessEqual(best_weight, capacity, "Knapsack solution exceeded capacity")
        # Hard to know exact optimal, but should be reasonably high
        # Simple greedy (by value density) gives a baseline
        density = values / weights
        indices = np.argsort(density)[::-1]
        greedy_val = 0
        greedy_wt = 0
        for idx in indices:
            if greedy_wt + weights[idx] <= capacity:
                greedy_wt += weights[idx]
                greedy_val += values[idx]
        
        print(f"Greedy Baseline: {greedy_val}")
        self.assertGreater(best_value, greedy_val * 0.95, "GA performed significantly worse than greedy baseline")

    def test_zdt1(self):
        """
        Test ZDT1 (Multi-Objective).
        f1(x) = x1
        g(x) = 1 + 9 * sum(xi) / (n-1) for i=2..n
        f2(x) = g(x) * (1 - sqrt(f1/g))
        x in [0, 1]. Pareto front at g(x) = 1.
        """
        dim = 30
        
        def fitness(sol_dbl, data):
            x = np.array(sol_dbl)
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (dim - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
            return [f1, f2] # We maximize, so we should return negative?
            # Wait, standard ZDT is minimization.
            # TrainSelPy maximizes.
            # So we return [-f1, -f2].
            
        def fitness_max(sol_dbl, data):
            x = np.array(sol_dbl)
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (dim - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
            return [-f1, -f2]

        control = self.control.copy()
        control["niterations"] = 500
        control["npop"] = 200
        control["solution_diversity"] = True
        
        result = train_sel(
            data={},
            candidates=[[]],
            setsizes=[dim],
            settypes=["DBL"],
            stat=fitness_max,
            n_stat=2,
            control=control,
            verbose=False
        )
        
        pareto_front = result.pareto_front
        
        self.assertGreater(len(pareto_front), 10, "Should find many Pareto solutions for ZDT1")
        
        # Verify convergence to g(x) = 1
        # In maximization, f1_max = -f1, f2_max = -f2
        # So f1 = -f1_max, f2 = -f2_max
        # Theoretical relation: f2 = 1 - sqrt(f1) (since g=1)
        # So check if f2 approx 1 - sqrt(f1)
        
        errors = []
        for sol in result.pareto_solutions:
            f1 = -sol["multi_fitness"][0]
            f2 = -sol["multi_fitness"][1]
            
            # Theoretical f2
            f2_theory = 1 - np.sqrt(f1)
            errors.append(abs(f2 - f2_theory))
        mean_error = np.mean(errors)
        print(f"ZDT1 Mean Error from Pareto Front: {mean_error}")
        self.assertLess(mean_error, 0.05, "ZDT1 solutions not close to Pareto front")

    def test_moo_portfolio_selection(self):
        """
        Test Multi-Objective Portfolio Selection (UOS + DBL).
        Select subset of assets (UOS) and allocate weights (DBL).
        Objective 1: Maximize expected return
        Objective 2: Minimize risk (variance)
        """
        n_assets = 20
        
        # Generate synthetic asset data
        np.random.seed(42)
        expected_returns = np.random.uniform(0.05, 0.20, n_assets)
        # Create covariance matrix
        A = np.random.randn(n_assets, n_assets)
        cov_matrix = np.dot(A, A.T) / n_assets
        
        def fitness(selected_assets, weights, data):
            # Normalize weights to sum to 1
            weights_array = np.array(weights)
            weights_norm = weights_array / (np.sum(weights_array) + 1e-10)
            
            # Calculate portfolio return
            portfolio_return = np.sum(expected_returns[selected_assets] * weights_norm)
            
            # Calculate portfolio risk (variance)
            portfolio_cov = cov_matrix[np.ix_(selected_assets, selected_assets)]
            portfolio_variance = np.dot(weights_norm, np.dot(portfolio_cov, weights_norm))
            
            # Return [maximize return, minimize risk = maximize -risk]
            return [portfolio_return, -portfolio_variance]
        
        n_select = 5
        control = self.control.copy()
        control["niterations"] = 300
        control["npop"] = 200
        control["solution_diversity"] = True
        
        result = train_sel(
            data={},
            candidates=[list(range(n_assets)), [1.0] * n_select],
            setsizes=[n_select, n_select],
            settypes=["UOS", "DBL"],
            stat=fitness,
            n_stat=2,
            control=control,
            verbose=False
        )
        
        pareto_front = result.pareto_front
        
        print(f"Portfolio: Found {len(pareto_front)} Pareto solutions")
        self.assertGreater(len(pareto_front), 3, "Should find multiple Pareto solutions for portfolio")
        
        # Verify trade-off: higher return should correlate with higher risk
        returns = [pf[0] for pf in pareto_front]
        risks = [-pf[1] for pf in pareto_front]  # Convert back to positive
        
        # Check that solutions span a range
        self.assertGreater(max(returns) - min(returns), 0.01, "Returns should vary across Pareto front")
        self.assertGreater(max(risks) - min(risks), 0.001, "Risks should vary across Pareto front")

    def test_moo_resource_allocation(self):
        """
        Test Multi-Objective Resource Allocation (BOOL + DBL).
        Select which projects to fund (BOOL) and allocate budget (DBL).
        Objective 1: Maximize total value
        Objective 2: Minimize total cost
        """
        n_projects = 15
        
        # Generate project data with trade-off between value and cost
        np.random.seed(42)
        base_values = np.random.uniform(50, 150, n_projects)
        base_costs = np.random.uniform(10, 50, n_projects)
        
        # Create correlation: higher value projects tend to cost more
        costs = base_costs + base_values * 0.3
        values = base_values + costs * 0.2
        
        def fitness(selected_bool, budget_allocation, data):
            selected = np.array(selected_bool, dtype=bool)
            budgets = np.array(budget_allocation)
            
            # Only consider selected projects
            total_value = 0
            total_cost = 0
            
            for i, is_selected in enumerate(selected):
                if is_selected:
                    # Budget allocation affects value (diminishing returns)
                    # budget in [0, 1], map to [0.5, 1.5] multiplier
                    budget_multiplier = 0.5 + budgets[i]
                    total_value += values[i] * budget_multiplier
                    total_cost += costs[i] * budget_multiplier
            
            # Return [maximize value, minimize cost = maximize -cost]
            return [total_value, -total_cost]
        
        control = self.control.copy()
        control["niterations"] = 300
        control["npop"] = 200
        control["solution_diversity"] = True
        
        result = train_sel(
            data={},
            candidates=[list(range(n_projects)), [0.5] * n_projects],
            setsizes=[n_projects, n_projects],
            settypes=["BOOL", "DBL"],
            stat=fitness,
            n_stat=2,
            control=control,
            verbose=False
        )
        
        pareto_front = result.pareto_front
        
        print(f"Resource Allocation: Found {len(pareto_front)} Pareto solutions")
        self.assertGreater(len(pareto_front), 3, "Should find multiple Pareto solutions for resource allocation")
        
        # Verify solutions are non-dominated
        for i, pf1 in enumerate(pareto_front):
            for j, pf2 in enumerate(pareto_front):
                if i != j:
                    # Check that pf1 doesn't dominate pf2
                    dominates = all(pf1[k] >= pf2[k] for k in range(2)) and any(pf1[k] > pf2[k] for k in range(2))
                    self.assertFalse(dominates, f"Solution {i} dominates solution {j} in Pareto front")

    def test_moo_scheduling_3obj(self):
        """
        Test Multi-Objective Scheduling with 3 objectives (OS + DBL).
        Order tasks (OS) and assign durations (DBL).
        Objective 1: Minimize total time
        Objective 2: Minimize cost
        Objective 3: Maximize quality
        This tests NSGA-III with 3 objectives.
        """
        n_tasks = 8
        
        # Task properties
        np.random.seed(42)
        base_durations = np.random.uniform(1, 10, n_tasks)
        task_costs = np.random.uniform(5, 20, n_tasks)
        task_quality = np.random.uniform(0.5, 1.0, n_tasks)
        
        def fitness(task_order, duration_multipliers, data):
            order = task_order
            multipliers = np.array(duration_multipliers)
            
            # Duration multiplier in [0, 1] maps to [0.5, 1.5]
            # Lower multiplier = faster but lower quality
            actual_multipliers = 0.5 + multipliers
            
            # Total time (considering dependencies - sequential)
            total_time = np.sum(base_durations * actual_multipliers)
            
            # Total cost (rushing costs more)
            rush_penalty = np.sum(task_costs * (2.0 - actual_multipliers))
            
            # Quality (rushing reduces quality)
            avg_quality = np.mean(task_quality * actual_multipliers)
            
            # Return [minimize time, minimize cost, maximize quality]
            # = [maximize -time, maximize -cost, maximize quality]
            return [-total_time, -rush_penalty, avg_quality]
        
        control = self.control.copy()
        control["niterations"] = 400
        control["npop"] = 300
        control["solution_diversity"] = True
        control["use_nsga3"] = True  # Use NSGA-III for 3 objectives
        
        result = train_sel(
            data={},
            candidates=[list(range(n_tasks)), [0.75] * n_tasks],
            setsizes=[n_tasks, n_tasks],
            settypes=["OS", "DBL"],
            stat=fitness,
            n_stat=3,
            control=control,
            verbose=False
        )
        
        pareto_front = result.pareto_front
        
        print(f"Scheduling (3-obj): Found {len(pareto_front)} Pareto solutions")
        self.assertGreater(len(pareto_front), 5, "Should find multiple Pareto solutions for 3-objective scheduling")
        
        # Verify all objectives vary across the front
        obj1_vals = [pf[0] for pf in pareto_front]
        obj2_vals = [pf[1] for pf in pareto_front]
        obj3_vals = [pf[2] for pf in pareto_front]
        
        self.assertGreater(max(obj1_vals) - min(obj1_vals), 0.1, "Objective 1 should vary")
        self.assertGreater(max(obj2_vals) - min(obj2_vals), 0.1, "Objective 2 should vary")
        self.assertGreater(max(obj3_vals) - min(obj3_vals), 0.01, "Objective 3 should vary")

if __name__ == "__main__":
    unittest.main()
