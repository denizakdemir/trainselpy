"""
True Multi-objective optimization example for TrainSelPy.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import List

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    train_sel_control,
    dopt,
    cdmean_opt
)

def multi_objective_fitness(solution: List[int], data: dict) -> List[float]:
    """
    Multi-objective fitness function returning multiple objective values.
    
    This function calculates three objectives:
    1. D-optimality (to be maximized)
    2. Diversity (mean pairwise distance between selected samples, to be maximized)
    3. Representativeness (mean distance to closest non-selected sample, to be minimized)
    
    Parameters
    ----------
    solution : list
        List of selected indices
    data : dict
        Data dictionary containing the feature matrix
        
    Returns
    -------
    list
        List of objective values [dopt_value, diversity, representativeness]
    """
    # Calculate D-optimality (higher values are better)
    dopt_value = dopt(solution, data)
    
    # Get feature matrix
    M = data["FeatureMat"]
    n_samples = M.shape[0]
    
    # Get selected features
    selected_features = M[solution, :]
    
    # Calculate diversity (mean distance between selected points)
    diversity = 0
    if len(solution) > 1:
        distances = []
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                dist = np.linalg.norm(selected_features[i] - selected_features[j])
                distances.append(dist)
        diversity = np.mean(distances)
    
    # Calculate representativeness (mean distance to closest non-selected sample)
    # Lower values indicate better representation of the entire dataset
    non_selected = [i for i in range(n_samples) if i not in solution]
    
    if len(non_selected) > 0:
        min_distances = []
        for i in non_selected:
            dists = [np.linalg.norm(M[i] - M[s]) for s in solution]
            min_distances.append(min(dists))
        representativeness = np.mean(min_distances)
    else:
        representativeness = 0
    
    # Return three objectives
    # For a consistent maximization approach, we negate representativeness (since we want to minimize it)
    return [dopt_value, diversity, -representativeness]

def main():
    """Run a true multi-objective optimization example with solution diversity."""
    
    print("TrainSelPy True Multi-objective Optimization Example - With Solution Diversity")
    print("--------------------------------------------------------------------------")
    
    # Create a test dataset
    print("\nCreating test dataset...")
    n_samples = 200
    n_features = 30
    
    # Create a marker matrix with some structure
    np.random.seed(42)  # For reproducibility
    
    # Create two clusters of data with different characteristics
    M1 = np.random.normal(0, 1, size=(n_samples//2, n_features))
    M2 = np.random.normal(3, 1.5, size=(n_samples//2, n_features))
    M = np.vstack([M1, M2])
    
    # Add some correlation between features
    for i in range(1, n_features):
        M[:, i] = M[:, i] + 0.3 * M[:, i-1]
    
    print(f"Created dataset with {n_samples} samples and {n_features} features")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    
    # Create relationship matrix
    K = np.dot(M, M.T) / n_features
    
    # Add a small value to the diagonal to ensure positive definiteness
    K += np.eye(n_samples) * 1e-6
    
    ts_data = make_data(K=K)
    ts_data["FeatureMat"] = M  # Add the feature matrix for diversity calculations
    
    # Set control parameters for multi-objective optimization
    print("\nSetting control parameters...")
    control = train_sel_control(
        size="free",
        niterations=100,    # Number of iterations
        minitbefstop=30,    # Stop if no improvement for this many iterations
        nEliteSaved=10,     # Number of elite solutions to save
        nelite=50,          # Number of elite solutions for selection
        npop=200,           # Population size
        mutprob=0.05,       # Mutation probability
        mutintensity=0.2,   # Mutation intensity
        crossprob=0.7,      # Crossover probability
        crossintensity=0.5, # Crossover intensity
        niterSANN=5,       # Simulated annealing iterations
        tempini=100.0,      # Initial temperature for SA
        tempfin=0.1,        # Final temperature for SA
        nislands=1,         # Number of islands for parallel optimization
        progress=True       # Show progress
    )
    
    # Run true multi-objective optimization with solution diversity
    print("\nRunning true multi-objective optimization with solution diversity...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_samples))],  # Select from all samples
        setsizes=[20],                        # Select 20 samples
        settypes=["UOS"],                     # Unordered set
        stat=multi_objective_fitness,         # Multi-objective fitness function
        n_stat=3,                             # Number of objectives (critical parameter!)
        control=control,
        solution_diversity=True,              # Ensure unique solutions on Pareto front
        verbose=True
    )
    
    runtime = time.time() - start_time
    
    print(f"\nMulti-objective optimization completed in {runtime:.2f} seconds")
    
    # Extract the Pareto front
    if result.pareto_front:
        print(f"Found {len(result.pareto_front)} solutions on the Pareto front")
        
        # Visualize the Pareto front (3D plot for 3 objectives)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract the objective values
        x = [sol[0] for sol in result.pareto_front]  # D-optimality
        y = [sol[1] for sol in result.pareto_front]  # Diversity
        z = [sol[2] for sol in result.pareto_front]  # Representativeness (negated)
        
        # Plot the Pareto front
        scatter = ax.scatter(x, y, z, c='b', marker='o', s=50, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('D-optimality')
        ax.set_ylabel('Diversity')
        ax.set_zlabel('Representativeness (-)')
        ax.set_title('Pareto Front for Multi-objective Optimization')
        
        # Add grid
        ax.grid(True)
        
        # Save the figure
        plt.savefig('pareto_front_3d.png')
        print("\nSaved 3D Pareto front plot to 'pareto_front_3d.png'")
        
        # Also create 2D projections for better visualization
        plt.figure(figsize=(18, 6))
        
        # D-optimality vs Diversity
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, c='b', marker='o', s=50, alpha=0.7)
        plt.xlabel('D-optimality')
        plt.ylabel('Diversity')
        plt.title('D-optimality vs Diversity')
        plt.grid(True)
        
        # D-optimality vs Representativeness
        plt.subplot(1, 3, 2)
        plt.scatter(x, z, c='r', marker='o', s=50, alpha=0.7)
        plt.xlabel('D-optimality')
        plt.ylabel('Representativeness (-)')
        plt.title('D-optimality vs Representativeness')
        plt.grid(True)
        
        # Diversity vs Representativeness
        plt.subplot(1, 3, 3)
        plt.scatter(y, z, c='g', marker='o', s=50, alpha=0.7)
        plt.xlabel('Diversity')
        plt.ylabel('Representativeness (-)')
        plt.title('Diversity vs Representativeness')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pareto_front_2d_projections.png')
        print("Saved 2D Pareto front projections to 'pareto_front_2d_projections.png'")
        
        # Display details of solutions on the Pareto front
        print("\nExample solutions from the Pareto front:")
        for i in range(min(3, len(result.pareto_front))):
            print(f"\nSolution {i+1}:")
            print(f"  Objectives: D-opt={result.pareto_front[i][0]:.4f}, " + 
                  f"Diversity={result.pareto_front[i][1]:.4f}, " + 
                  f"Representativeness={-result.pareto_front[i][2]:.4f}")
            
            # Display the selected indices for this solution
            if hasattr(result, 'pareto_solutions'):
                sol_indices = result.pareto_solutions[i]['selected_indices'][0]
                print(f"  Selected indices: {sol_indices}")
                
                # Calculate how many points are selected from each cluster
                cluster1_count = sum(1 for idx in sol_indices if idx < n_samples//2)
                cluster2_count = len(sol_indices) - cluster1_count
                print(f"  Points from Cluster 1: {cluster1_count}")
                print(f"  Points from Cluster 2: {cluster2_count}")
    else:
        print("No Pareto front found. The optimization may not have run in true multi-objective mode.")
    
    # Visualize one of the solutions on the Pareto front in the feature space
    if result.pareto_front and hasattr(result, 'pareto_solutions'):
        try:
            from sklearn.decomposition import PCA
            
            plt.figure(figsize=(10, 8))
            
            # Perform PCA to visualize high-dimensional data in 2D
            pca = PCA(n_components=2)
            M_pca = pca.fit_transform(M)
            
            # Plot all points
            plt.scatter(M_pca[:, 0], M_pca[:, 1], c='gray', alpha=0.3, label='All samples')
            
            # Choose one solution from the Pareto front (e.g., the first one)
            sol_idx = 0
            if sol_idx < len(result.pareto_solutions):
                selected = result.pareto_solutions[sol_idx]['selected_indices'][0]
                selected_pca = pca.transform(M[selected])
                
                plt.scatter(selected_pca[:, 0], selected_pca[:, 1], c='red', s=100, 
                            edgecolor='black', label=f'Selected (Solution {sol_idx+1})')
                
                plt.title(f'Feature Space Visualization (PCA) - Solution {sol_idx+1}')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.legend()
                plt.grid(True)
                plt.savefig('feature_space_visualization.png')
                print("\nSaved feature space visualization to 'feature_space_visualization.png'")
        except ImportError:
            print("Could not import sklearn for PCA visualization. Skipping feature space plot.")
    
    print("\nTrue multi-objective optimization example completed.")

if __name__ == "__main__":
    main()