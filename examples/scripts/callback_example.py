"""
Callback Example for TrainSelPy

Demonstrates how to use callbacks for:
1. Checkpointing optimization progress
2. Tracking Pareto front evolution (multi-objective)
3. Custom logging and visualization
4. Real-time monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

from trainselpy import (
    make_data,
    train_sel,
    set_control_default
)


def generate_synthetic_data(n_samples=100, n_features=20, random_seed=42):
    """Generate synthetic data for demonstration."""
    np.random.seed(random_seed)

    # Create feature matrix
    features = np.random.randn(n_samples, n_features)

    # Create importance scores (for objective 1)
    importance = np.random.uniform(0.3, 1.0, n_features)

    # Create efficiency scores (for objective 2, slightly conflicting)
    efficiency = np.random.uniform(0.2, 1.0, n_features)
    # Make some features with high importance have lower efficiency
    efficiency[:n_features//3] *= 0.6

    return features, importance, efficiency


def multi_objective_fitness(selected_features, data):
    """
    Multi-objective fitness function.

    Objectives:
    1. Maximize total importance
    2. Maximize total efficiency
    """
    importance = data['importance']
    efficiency = data['efficiency']

    total_importance = np.sum(importance[selected_features])
    total_efficiency = np.sum(efficiency[selected_features])

    return [total_importance, total_efficiency]


class OptimizationMonitor:
    """
    Callback class for monitoring optimization progress.

    Features:
    - Saves checkpoints periodically
    - Tracks Pareto front evolution
    - Logs generation statistics
    - Can generate plots
    """

    def __init__(self, output_dir="optimization_output", save_every=10):
        self.output_dir = Path(output_dir)
        self.save_every = save_every
        self.generation_log = []

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.pareto_dir = self.output_dir / "pareto_fronts"
        self.log_dir = self.output_dir / "logs"

        for directory in [self.checkpoint_dir, self.pareto_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, state):
        """
        Callback function invoked at the end of each generation.

        Parameters
        ----------
        state : dict
            Dictionary containing optimization state information.
        """
        gen = state["generation"]
        best_solution = state["best_solution"]
        fitness_history = state["fitness_history"]
        n_stat = state["n_stat"]

        # Log generation statistics
        log_entry = {
            "generation": gen,
            "best_fitness": best_solution.fitness,
            "no_improvement": state["no_improvement_count"]
        }

        if n_stat > 1:
            log_entry["pareto_front_size"] = (
                len(state["pareto_front"]) if state["pareto_front"] else 0
            )
            log_entry["multi_fitness"] = best_solution.multi_fitness

        self.generation_log.append(log_entry)

        # Print progress
        if gen % self.save_every == 0 or gen == 0:
            print(f"\n{'='*60}")
            print(f"Generation {gen}")
            print(f"Best fitness: {best_solution.fitness:.4f}")

            if n_stat > 1:
                print(f"Multi-objective fitness: {best_solution.multi_fitness}")
                if state["pareto_front"]:
                    print(f"Pareto front size: {len(state['pareto_front'])}")

            print(f"No improvement count: {state['no_improvement_count']}")
            print(f"{'='*60}\n")

        # Save checkpoint
        if gen % self.save_every == 0:
            self._save_checkpoint(state)

        # Save Pareto front for multi-objective
        if n_stat > 1 and state["pareto_front"] is not None:
            if gen % self.save_every == 0:
                self._save_pareto_front(state)

        # Save generation log
        if gen % (self.save_every * 2) == 0:
            self._save_log()

    def _save_checkpoint(self, state):
        """Save optimization checkpoint."""
        gen = state["generation"]

        checkpoint = {
            "generation": gen,
            "best_solution": state["best_solution"],
            "fitness_history": state["fitness_history"],
            "population_size": len(state["population"]),
            "n_stat": state["n_stat"]
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{gen:04d}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved: {checkpoint_path.name}")

    def _save_pareto_front(self, state):
        """Save Pareto front to CSV."""
        gen = state["generation"]
        pareto_front = state["pareto_front"]

        if not pareto_front:
            return

        # Create DataFrame with objective values
        df = pd.DataFrame(
            pareto_front,
            columns=[f"Objective_{i+1}" for i in range(len(pareto_front[0]))]
        )

        # Add solution indices
        pareto_solutions = state["pareto_solutions"]
        for i, sol in enumerate(pareto_solutions):
            df.loc[i, "selected_indices"] = str(sol.int_values[0])

        pareto_path = self.pareto_dir / f"pareto_{gen:04d}.csv"
        df.to_csv(pareto_path, index=False)

        print(f"Pareto front saved: {pareto_path.name} ({len(df)} solutions)")

    def _save_log(self):
        """Save generation log to CSV."""
        df = pd.DataFrame(self.generation_log)
        log_path = self.log_dir / "generation_log.csv"
        df.to_csv(log_path, index=False)
        print(f"Generation log updated: {log_path.name}")

    def plot_convergence(self, save_path=None):
        """Plot fitness convergence over generations."""
        df = pd.DataFrame(self.generation_log)

        plt.figure(figsize=(12, 5))

        # Plot 1: Fitness history
        plt.subplot(1, 2, 1)
        plt.plot(df["generation"], df["best_fitness"], 'b-', linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Convergence")
        plt.grid(True, alpha=0.3)

        # Plot 2: Pareto front size (if multi-objective)
        if "pareto_front_size" in df.columns:
            plt.subplot(1, 2, 2)
            plt.plot(df["generation"], df["pareto_front_size"], 'r-', linewidth=2)
            plt.xlabel("Generation")
            plt.ylabel("Pareto Front Size")
            plt.title("Pareto Front Evolution")
            plt.grid(True, alpha=0.3)
        else:
            # Plot no improvement count
            plt.subplot(1, 2, 2)
            plt.plot(df["generation"], df["no_improvement"], 'g-', linewidth=2)
            plt.xlabel("Generation")
            plt.ylabel("No Improvement Count")
            plt.title("Stagnation Monitor")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved: {save_path}")

        plt.show()


def example_single_objective_with_callback():
    """Example: Single-objective optimization with callback monitoring."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single-Objective Optimization with Callback")
    print("="*70 + "\n")

    # Generate data
    features, importance, efficiency = generate_synthetic_data()
    n_features = features.shape[1]

    # Prepare data
    data = {
        'importance': importance,
        'efficiency': efficiency
    }

    # Define fitness function (maximize importance)
    def fitness_func(selected_features, data):
        return np.sum(data['importance'][selected_features])

    # Create monitor
    monitor = OptimizationMonitor(
        output_dir="callback_output_single",
        save_every=5
    )

    # Configure optimization
    control = set_control_default()
    control["niterations"] = 50
    control["npop"] = 100
    control["progress"] = False  # We'll use callback for progress
    control["callback"] = monitor  # Attach callback

    # Run optimization
    result = train_sel(
        data=None,
        candidates=[list(range(n_features))],
        setsizes=[8],
        settypes=["UOS"],
        stat=fitness_func,
        n_stat=1,
        control=control
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best fitness: {result.fitness:.4f}")
    print(f"Selected features: {result.selected_indices[0]}")
    print(f"Total generations: {len(monitor.generation_log)}")
    print(f"Output saved to: {monitor.output_dir}")

    # Plot convergence
    monitor.plot_convergence(save_path=monitor.output_dir / "convergence.png")

    return result, monitor


def example_multi_objective_with_callback():
    """Example: Multi-objective optimization with Pareto front tracking."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Objective Optimization with Pareto Tracking")
    print("="*70 + "\n")

    # Generate data
    features, importance, efficiency = generate_synthetic_data()
    n_features = features.shape[1]

    # Prepare data
    data = {
        'importance': importance,
        'efficiency': efficiency
    }

    # Create monitor
    monitor = OptimizationMonitor(
        output_dir="callback_output_multi",
        save_every=10
    )

    # Configure optimization
    control = set_control_default()
    control["niterations"] = 100
    control["npop"] = 200
    control["progress"] = False
    control["solution_diversity"] = True
    control["callback"] = monitor

    # Run optimization
    result = train_sel(
        data=None,
        candidates=[list(range(n_features))],
        setsizes=[8],
        settypes=["UOS"],
        stat=multi_objective_fitness,
        n_stat=2,
        control=control
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Final Pareto front size: {len(result.pareto_front)}")
    print(f"Best aggregated fitness: {result.fitness:.4f}")
    print(f"Total generations: {len(monitor.generation_log)}")
    print(f"Output saved to: {monitor.output_dir}")

    # Plot convergence
    monitor.plot_convergence(save_path=monitor.output_dir / "convergence.png")

    # Plot final Pareto front
    plot_pareto_front(result.pareto_front, monitor.output_dir)

    return result, monitor


def plot_pareto_front(pareto_front, output_dir):
    """Plot the final Pareto front."""
    if not pareto_front:
        return

    pareto_array = np.array(pareto_front)

    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_array[:, 0], pareto_array[:, 1],
                c='blue', s=100, alpha=0.6, edgecolors='black')
    plt.xlabel("Objective 1 (Importance)", fontsize=12)
    plt.ylabel("Objective 2 (Efficiency)", fontsize=12)
    plt.title("Final Pareto Front", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    save_path = Path(output_dir) / "pareto_front_final.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Pareto front plot saved: {save_path}")
    plt.show()


def simple_callback_example():
    """Simple inline callback example."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Simple Inline Callback")
    print("="*70 + "\n")

    # Track best fitness at specific generations
    tracked_generations = []

    def simple_callback(state):
        """Simple callback that tracks specific generations."""
        gen = state["generation"]

        # Track every 10th generation
        if gen % 10 == 0:
            tracked_generations.append({
                "generation": gen,
                "fitness": state["best_solution"].fitness,
                "selected": state["best_solution"].int_values[0]
            })
            print(f"Gen {gen}: Fitness = {state['best_solution'].fitness:.4f}")

    # Generate simple data
    features, importance, efficiency = generate_synthetic_data(n_samples=50)

    def fitness_func(selected_features, data):
        return np.sum(data['importance'][selected_features])

    # Run with simple callback
    control = set_control_default()
    control["niterations"] = 50
    control["npop"] = 50
    control["progress"] = False
    control["callback"] = simple_callback

    result = train_sel(
        data=None,
        candidates=[list(range(20))],
        setsizes=[5],
        settypes=["UOS"],
        stat=fitness_func,
        n_stat=1,
        control=control
    )

    print(f"\nTracked {len(tracked_generations)} generations")
    print(f"Final fitness: {result.fitness:.4f}")

    return result, tracked_generations


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# TrainSelPy Callback Examples")
    print("#"*70)

    # Example 1: Single-objective with monitoring
    result1, monitor1 = example_single_objective_with_callback()

    # Example 2: Multi-objective with Pareto tracking
    result2, monitor2 = example_multi_objective_with_callback()

    # Example 3: Simple inline callback
    result3, tracked = simple_callback_example()

    print("\n" + "#"*70)
    print("# All examples completed successfully!")
    print("#"*70 + "\n")
