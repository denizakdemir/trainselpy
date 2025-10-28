"""
Multi-Objective Discrete Optimization Example for TrainSelPy.

This example demonstrates multi-objective optimization with discrete variables,
selecting optimal feature subsets while balancing multiple competing objectives.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    train_sel_control
)

def generate_synthetic_data(n_samples=100, n_features=20, random_seed=42):
    """
    Generate synthetic data with feature clusters and multiple metrics.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        Features, importances, costs
    """
    np.random.seed(random_seed)
    
    # Create three feature clusters
    cluster1 = np.random.normal(0, 1, size=(n_samples, n_features // 4))
    cluster2 = np.random.normal(2, 1, size=(n_samples, n_features // 4))
    cluster3 = np.random.normal(-2, 1, size=(n_samples, n_features // 2))
    
    features = np.hstack((cluster1, cluster2, cluster3))
    
    # Create two importance metrics (conflicting to some extent)
    # Metric 1: Information content - higher for clusters 1 and 2
    information = np.zeros(n_features)
    information[:n_features//2] = np.random.uniform(0.6, 1.0, n_features//2)
    information[n_features//2:] = np.random.uniform(0.1, 0.5, n_features//2)
    
    # Metric 2: Cost efficiency - higher for clusters 2 and 3
    efficiency = np.zeros(n_features)
    efficiency[n_features//4:] = np.random.uniform(0.7, 1.0, 3*n_features//4)
    efficiency[:n_features//4] = np.random.uniform(0.2, 0.6, n_features//4)
    
    # Metric 3: Feature costs (to be minimized)
    costs = np.random.uniform(1, 10, n_features)
    # Make some features in each cluster particularly expensive
    costs[np.random.choice(range(n_features//4), size=2)] *= 2
    costs[np.random.choice(range(n_features//4, n_features//2), size=2)] *= 2
    costs[np.random.choice(range(n_features//2, n_features), size=4)] *= 2
    
    return features, information, efficiency, costs

def moo_feature_selection(selected_features, data):
    """
    Multi-objective fitness function for feature selection.
    
    Returns three objective values:
    1. Information (maximize)
    2. Efficiency (maximize)
    3. Cost (minimize, converted to negative for maximization)
    
    Parameters
    ----------
    selected_features : list
        List of selected feature indices
    data : dict
        Data dictionary
        
    Returns
    -------
    list
        List of objective values [info, efficiency, -cost]
    """
    # Extract feature metrics
    information = data['information']
    efficiency = data['efficiency']
    costs = data['costs']
    
    # Calculate total metrics for selected features
    total_information = np.sum(information[selected_features])
    total_efficiency = np.sum(efficiency[selected_features])
    total_cost = np.sum(costs[selected_features])
    
    # Return objectives (all set up to be maximized)
    return [total_information, total_efficiency, -total_cost]

def main():
    """Run a multi-objective discrete variable optimization example."""
    
    print("TrainSelPy Multi-Objective Discrete Optimization Example")
    print("------------------------------------------------------")
    
    # Generate synthetic data
    n_samples = 100
    n_features = 24  # Divisible by 4 for easier clustering
    print(f"\nGenerating synthetic data with {n_samples} samples and {n_features} features...")
    
    features, information, efficiency, costs = generate_synthetic_data(n_samples, n_features)
    
    # Display feature metrics
    print("\nFeature Metrics (first 5 features):")
    print("Feature | Information | Efficiency | Cost")
    print("-" * 45)
    for i in range(5):
        print(f"{i+1:7} | {information[i]:11.4f} | {efficiency[i]:10.4f} | {costs[i]:4.2f}")
    print("...")
    
    # Create TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(M=features)
    ts_data['information'] = information
    ts_data['efficiency'] = efficiency
    ts_data['costs'] = costs
    
    # Set control parameters for multi-objective optimization
    print("\nSetting control parameters...")
    control = train_sel_control(
        size="free",
        niterations=150,       # Number of iterations
        minitbefstop=50,       # Stop if no improvement for this many iterations
        nEliteSaved=20,        # Number of elite solutions to save
        nelite=50,             # Number of elite solutions to keep
        npop=200,              # Population size
        mutprob=0.1,           # Mutation probability
        mutintensity=0.2,      # Mutation intensity
        crossprob=0.7,         # Crossover probability
        crossintensity=0.5,    # Crossover intensity
        progress=True,         # Show progress
        solution_diversity=True  # Ensure diversity of solutions
    )
    
    # Run the multi-objective optimization
    print("\nRunning multi-objective feature selection...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_features))],  # All features are candidates
        setsizes=[n_features // 3],            # Select one-third of features
        settypes=["UOS"],                      # Unordered Set (feature indices)
        stat=moo_feature_selection,            # Multi-objective fitness function
        n_stat=3,                              # Three objectives
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract the Pareto front
    if result.pareto_front:
        n_solutions = len(result.pareto_front)
        print(f"Found {n_solutions} solutions on the Pareto front")
        
        # Display selected solutions
        if n_solutions > 0:
            # Sample solutions across the Pareto front
            n_display = min(5, n_solutions)
            display_indices = np.linspace(0, n_solutions-1, n_display, dtype=int)
            
            for i, idx in enumerate(display_indices):
                solution = result.pareto_solutions[idx]
                selected = solution['selected_indices'][0]
                objectives = result.pareto_front[idx]
                
                print(f"\nSolution {i+1}:")
                print(f"  Selected features: {selected}")
                print(f"  Information: {objectives[0]:.4f}")
                print(f"  Efficiency: {objectives[1]:.4f}")
                print(f"  Cost: {-objectives[2]:.4f}")
            
            # Visualize the Pareto front
            if n_solutions > 1:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Extract objective values
                x = [sol[0] for sol in result.pareto_front]  # Information
                y = [sol[1] for sol in result.pareto_front]  # Efficiency
                z = [-sol[2] for sol in result.pareto_front]  # Cost (convert back to positive)
                
                # Plot the Pareto front
                scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=50, alpha=0.7)
                
                # Add labels
                ax.set_xlabel('Information')
                ax.set_ylabel('Efficiency')
                ax.set_zlabel('Cost')
                ax.set_title('Pareto Front for Feature Selection')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label('Cost')
                
                # Add grid
                ax.grid(True)
                
                plt.savefig('discrete_moo_pareto_front.png')
                print("\nSaved Pareto front visualization to 'discrete_moo_pareto_front.png'")
                
                # 2D projections
                plt.figure(figsize=(15, 5))
                
                # Information vs Efficiency
                plt.subplot(1, 3, 1)
                plt.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.7)
                plt.colorbar(label='Cost')
                plt.xlabel('Information')
                plt.ylabel('Efficiency')
                plt.title('Information vs. Efficiency')
                plt.grid(True)
                
                # Information vs Cost
                plt.subplot(1, 3, 2)
                plt.scatter(x, z, c=y, cmap='viridis', s=50, alpha=0.7)
                plt.colorbar(label='Efficiency')
                plt.xlabel('Information')
                plt.ylabel('Cost')
                plt.title('Information vs. Cost')
                plt.grid(True)
                
                # Efficiency vs Cost
                plt.subplot(1, 3, 3)
                plt.scatter(y, z, c=x, cmap='viridis', s=50, alpha=0.7)
                plt.colorbar(label='Information')
                plt.xlabel('Efficiency')
                plt.ylabel('Cost')
                plt.title('Efficiency vs. Cost')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('discrete_moo_pareto_2d.png')
                print("Saved 2D Pareto front projections to 'discrete_moo_pareto_2d.png'")
                
                # Visualize selected features for different solutions
                # Find extreme and balanced solutions
                max_info_idx = np.argmax([sol[0] for sol in result.pareto_front])
                max_eff_idx = np.argmax([sol[1] for sol in result.pareto_front])
                min_cost_idx = np.argmin([-sol[2] for sol in result.pareto_front])
                
                # Get a balanced solution (closest to middle of normalized objectives)
                norm_x = (np.array(x) - min(x)) / (max(x) - min(x) if max(x) > min(x) else 1)
                norm_y = (np.array(y) - min(y)) / (max(y) - min(y) if max(y) > min(y) else 1)
                norm_z = 1 - (np.array(z) - min(z)) / (max(z) - min(z) if max(z) > min(z) else 1)  # Invert cost
                
                distances = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2 + (norm_z - 0.5)**2)
                balanced_idx = np.argmin(distances)
                
                # Collect solutions to show
                solution_indices = [max_info_idx, max_eff_idx, min_cost_idx, balanced_idx]
                solution_names = ["Max Information", "Max Efficiency", "Min Cost", "Balanced"]
                
                # Keep unique indices
                unique_indices = []
                unique_names = []
                for i, idx in enumerate(solution_indices):
                    if idx not in unique_indices:
                        unique_indices.append(idx)
                        unique_names.append(solution_names[i])
                
                # Feature importance visualization
                plt.figure(figsize=(15, 10))
                
                for i, (idx, name) in enumerate(zip(unique_indices, unique_names)):
                    solution = result.pareto_solutions[idx]
                    selected_features = solution['selected_indices'][0]
                    objectives = result.pareto_front[idx]
                    
                    # Create feature importance plot
                    plt.subplot(2, 2, i+1)
                    
                    # Prepare data
                    all_indices = np.arange(n_features)
                    selected_mask = np.zeros(n_features, dtype=bool)
                    selected_mask[selected_features] = True
                    
                    # Create bar colors (cluster-coded)
                    cluster_colors = ['red' if j < n_features//4 else 
                                     'blue' if j < n_features//2 else 
                                     'green' for j in range(n_features)]
                    
                    # Adjust alpha based on selection
                    alpha_values = [1.0 if selected else 0.3 for selected in selected_mask]
                    
                    # Plot all features with cluster coloring
                    for j in range(n_features):
                        plt.bar(j, information[j], color=cluster_colors[j], alpha=alpha_values[j])
                    
                    # Add title with solution metrics
                    plt.title(f"{name}\nInfo: {objectives[0]:.2f}, " +
                             f"Eff: {objectives[1]:.2f}, Cost: {-objectives[2]:.2f}")
                    
                    plt.xlabel('Feature Index')
                    plt.ylabel('Information')
                    plt.ylim(0, max(information) * 1.1)
                    
                    # Add cluster legend
                    plt.text(n_features//8, max(information)*0.9, "Cluster 1", 
                            color='red', ha='center')
                    plt.text(3*n_features//8, max(information)*0.9, "Cluster 2", 
                            color='blue', ha='center')
                    plt.text(3*n_features//4, max(information)*0.9, "Cluster 3", 
                            color='green', ha='center')
                
                plt.tight_layout()
                plt.savefig('discrete_moo_solutions.png')
                print("Saved solution visualizations to 'discrete_moo_solutions.png'")
                
            else:
                print("Only one solution found on the Pareto front. Not enough for visualization.")
    else:
        print("No Pareto front found.")
        
    print("\nMulti-objective discrete optimization example completed.")

if __name__ == "__main__":
    main()