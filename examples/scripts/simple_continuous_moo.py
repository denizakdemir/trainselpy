"""
Simple Multi-Objective Continuous Variable Optimization Example for TrainSelPy.

This example demonstrates a simplified optimization problem with continuous
variables and three clearly conflicting objectives, designed to produce a
diverse Pareto front.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    train_sel_control
)

def simple_moo_objectives(weights, data):
    """
    Multi-objective function with three clearly conflicting objectives.
    
    Objectives:
    1. Minimize the sum of squares (weights closer to zero are better)
    2. Maximize the product of weights (weights closer to 1 are better)
    3. Maximize the entropy (weights should be diverse)
    
    Parameters
    ----------
    weights : list
        List of weights (continuous variables)
    data : dict
        Data dictionary (not used in this simple example)
        
    Returns
    -------
    list
        List of objective values [obj1, obj2, obj3] (all to be maximized)
    """
    weights = np.array(weights)
    n = len(weights)
    
    # Normalize weights to 0-1 range
    weights = np.clip(weights, 0, 1)
    
    # Objective 1: Minimize sum of squares (return negative for maximization)
    # This favors solutions with weights close to 0
    obj1 = -np.sum(weights**2)
    
    # Objective 2: Maximize product of weights
    # This favors solutions with weights close to 1
    obj2 = np.prod(weights + 1e-10)  # Small epsilon to avoid zero products
    
    # Objective 3: Maximize entropy (diversity of weights)
    # This favors solutions with diverse weights
    entropy = -np.sum((weights + 1e-10) * np.log(weights + 1e-10))
    obj3 = entropy / np.log(n) if n > 1 else 0  # Normalize by max entropy
    
    return [obj1, obj2, obj3]

def main():
    """Run a simple multi-objective continuous variable optimization example."""
    
    print("TrainSelPy Simple Multi-Objective Continuous Optimization Example")
    print("---------------------------------------------------------------")
    
    # Number of variables to optimize
    n_vars = 5
    
    # Create a simple data object
    features = np.eye(n_vars)
    ts_data = make_data(M=features)
    
    # Set control parameters for multi-objective optimization
    print("\nSetting control parameters...")
    control = train_sel_control(
        size="free",
        niterations=200,      # Number of iterations
        minitbefstop=50,      # Minimum iterations before stopping
        nEliteSaved=20,       # Number of elite solutions to save
        nelite=50,            # Number of elite solutions
        npop=200,             # Population size
        mutprob=0.1,          # Mutation probability
        mutintensity=0.3,     # Mutation intensity
        crossprob=0.8,        # Crossover probability
        crossintensity=0.5,   # Crossover intensity
        niterSANN=30,         # Simulated annealing iterations
        tempini=100.0,        # Initial temperature
        tempfin=0.1,          # Final temperature
        solution_diversity=True  # Ensure diversity of solutions
    )
    
    # Run the multi-objective optimization
    print("\nRunning optimization...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_vars))],  # Indices for continuous variables
        setsizes=[n_vars],                 # Number of variables
        settypes=["DBL"],                  # Continuous variables
        stat=simple_moo_objectives,        # Multi-objective function
        n_stat=3,                          # Three objectives
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract results
    if result.pareto_front:
        print(f"Found {len(result.pareto_front)} solutions on the Pareto front")
        
        # Display results
        if len(result.pareto_front) > 1:
            # Print objectives for each solution on the Pareto front
            for i, (solution, objectives) in enumerate(zip(result.pareto_solutions, result.pareto_front)):
                if i < 5:  # Limit to first 5 solutions
                    weights = solution['selected_values'][0]
                    print(f"\nSolution {i+1}:")
                    print(f"  Weights: {', '.join([f'{w:.4f}' for w in weights])}")
                    print(f"  Objectives: [-SumSquares={-objectives[0]:.4f}, Product={objectives[1]:.4f}, Entropy={objectives[2]:.4f}]")
            
            # Print ellipsis if more than 5 solutions
            if len(result.pareto_front) > 5:
                print("...")
            
            # Visualize the Pareto front (3D)
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract objective values
            x = [-sol[0] for sol in result.pareto_front]  # Convert back to positive for clarity
            y = [sol[1] for sol in result.pareto_front]
            z = [sol[2] for sol in result.pareto_front]
            
            # Plot Pareto front
            scatter = ax.scatter(x, y, z, c=z, marker='o', s=50, alpha=0.7, cmap='viridis')
            
            # Add labels
            ax.set_xlabel('Sum of Squares (minimize)')
            ax.set_ylabel('Product (maximize)')
            ax.set_zlabel('Entropy (maximize)')
            ax.set_title('Pareto Front')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Entropy')
            
            plt.savefig('simple_continuous_pareto_front.png')
            print("\nSaved 3D Pareto front to 'simple_continuous_pareto_front.png'")
            
            # 2D projections
            plt.figure(figsize=(15, 5))
            
            # Sum of Squares vs Product
            plt.subplot(1, 3, 1)
            plt.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Entropy')
            plt.xlabel('Sum of Squares (minimize)')
            plt.ylabel('Product (maximize)')
            plt.title('Sum of Squares vs Product')
            plt.grid(True)
            
            # Sum of Squares vs Entropy
            plt.subplot(1, 3, 2)
            plt.scatter(x, z, c=y, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Product')
            plt.xlabel('Sum of Squares (minimize)')
            plt.ylabel('Entropy (maximize)')
            plt.title('Sum of Squares vs Entropy')
            plt.grid(True)
            
            # Product vs Entropy
            plt.subplot(1, 3, 3)
            plt.scatter(y, z, c=x, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Sum of Squares')
            plt.xlabel('Product (maximize)')
            plt.ylabel('Entropy (maximize)')
            plt.title('Product vs Entropy')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('simple_continuous_pareto_front_2d.png')
            print("Saved 2D Pareto front projections to 'simple_continuous_pareto_front_2d.png'")
            
            # Visualize solutions in weight space
            # Choose diverse solutions
            indices = [0, len(result.pareto_front)//4, len(result.pareto_front)//2, 
                      3*len(result.pareto_front)//4, len(result.pareto_front)-1]
            indices = [i for i in indices if i < len(result.pareto_front)]
            
            plt.figure(figsize=(12, 8))
            
            # For each solution
            for i, idx in enumerate(indices):
                solution = result.pareto_solutions[idx]
                weights = solution['selected_values'][0]
                objectives = result.pareto_front[idx]
                
                # Plot weights
                plt.subplot(len(indices), 1, i+1)
                plt.bar(range(n_vars), weights, alpha=0.7)
                plt.ylim(0, 1)
                plt.title(f"Solution {idx+1} - SumSq: {-objectives[0]:.2f}, " +
                         f"Prod: {objectives[1]:.2f}, Ent: {objectives[2]:.2f}")
                plt.xticks(range(n_vars), [f"W{j+1}" for j in range(n_vars)])
                
                # Add value labels
                for j, v in enumerate(weights):
                    plt.text(j, v+0.05, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig('simple_continuous_solutions.png')
            print("Saved solution visualizations to 'simple_continuous_solutions.png'")
            
        else:
            print("Only one solution found on the Pareto front. Not enough for visualization.")
    else:
        print("No Pareto front found.")
    
    print("\nSimple multi-objective continuous optimization example completed.")

if __name__ == "__main__":
    main()