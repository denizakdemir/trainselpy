"""
Advanced example of using the TrainSelPy package with optimization criteria.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to ensure we find our fixes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    train_sel_control,
    set_control_default,
    dopt, 
    maximin_opt
)

# Import our custom implementation of cdmean
from custom_cdmean import custom_cdmean as cdmean_opt

# Load example data
from trainselpy.data import wheat_data

def custom_multi_objective(solution, data):
    """
    Custom multi-objective function combining CDMean and D-optimality.
    
    Parameters
    ----------
    solution : list
        List of selected indices
    data : dict
        Data dictionary
        
    Returns
    -------
    float
        Combined objective value (weighted average)
    """
    cdmean_value = cdmean_opt(solution, data)
    dopt_value = dopt(solution, data)
    
    # Scale values to make them comparable
    # Adjust weights to prioritize different objectives
    return 0.7 * cdmean_value + 0.3 * (dopt_value / 100)  # Scale down D-optimality

def main():
    """Run an advanced example of TrainSelPy."""
    
    print("TrainSelPy Advanced Example (Fixed)")
    print("--------------------------------")
    
    # Load wheat data
    print("\nLoading wheat data...")
    M = wheat_data['M']
    K = wheat_data['K']
    Y = wheat_data['Y']
    
    print(f"Data dimensions: {M.shape[0]} lines, {M.shape[1]} markers")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(K=K)
    ts_data["FeatureMat"] = M  # For D-optimality
    ts_data["G"] = K           # For CDMean
    ts_data["lambda"] = 0.01   # For CDMean
    
    # Run the selection algorithm with custom optimization criterion
    print("\nRunning TrainSel with custom multi-objective criterion...")
    control = set_control_default()
    control["niterations"] = 50
    control["npop"] = 200
    
    start_time = time.time()
    result_custom = train_sel(
        data=ts_data,
        candidates=[list(range(400))],  # Select from first 400 lines
        setsizes=[40],                 # Select 40 lines
        settypes=["UOS"],             # Unordered set
        stat=custom_multi_objective,  # Use custom criterion
        control=control,
        verbose=True
    )
    runtime = time.time() - start_time
    
    print(f"\nCustom optimization completed in {runtime:.2f} seconds")
    print(f"Final fitness: {result_custom.fitness:.6f}")
    selected_custom = result_custom.selected_indices[0]
    print(f"Selected {len(selected_custom)} lines")
    print(f"First 10 selected indices: {selected_custom[:10]}...")
    
    # Calculate individual components
    cdmean_val = cdmean_opt(selected_custom, ts_data)
    dopt_val = dopt(selected_custom, ts_data)
    print(f"CDMean value: {cdmean_val:.6f}")
    print(f"D-optimality value: {dopt_val:.6f}")
    
    # Example with different selection types
    print("\n\nComparing different selection types...")
    
    # Run with Unordered Set type (UOS)
    result_uos = train_sel(
        data=ts_data,
        candidates=[list(range(400))],
        setsizes=[20],
        settypes=["UOS"],  # Unordered set
        stat=dopt,
        control=control,
        verbose=False
    )
    
    # Run with Ordered Set type (OS)
    result_os = train_sel(
        data=ts_data,
        candidates=[list(range(400))],
        setsizes=[20],
        settypes=["OS"],  # Ordered set
        stat=dopt,
        control=control,
        verbose=False
    )
    
    print("\nResults comparison:")
    print(f"UOS D-optimality: {result_uos.fitness:.6f}")
    print(f"OS D-optimality: {result_os.fitness:.6f}")
    
    # Example with mixed set types (multiple selection sets)
    print("\n\nRunning example with multiple selection sets...")
    
    # Define a fitness function for multiple sets
    def multi_set_fitness(sets, data):
        # Calculate average D-optimality across all sets
        total_dopt = 0
        for set_indices in sets:
            set_dopt = dopt(set_indices, data)
            total_dopt += set_dopt
        return total_dopt / len(sets)
    
    # Run the optimization with multiple sets
    result_multi = train_sel(
        data=ts_data,
        candidates=[list(range(200)), list(range(200, 400))],  # Two candidate sets
        setsizes=[15, 15],                                   # Select 15 from each
        settypes=["UOS", "UOS"],                            # Both unordered
        stat=multi_set_fitness,                            # Multi-set fitness function
        control=control,
        verbose=True
    )
    
    print(f"\nMulti-set optimization fitness: {result_multi.fitness:.6f}")
    print(f"Selected set 1: {result_multi.selected_indices[0][:5]}... ({len(result_multi.selected_indices[0])} items)")
    print(f"Selected set 2: {result_multi.selected_indices[1][:5]}... ({len(result_multi.selected_indices[1])} items)")
    
    # Visualize the selections
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to the marker matrix
        pca = PCA(n_components=2)
        M_reduced = pca.fit_transform(M)
        
        # Plot custom multi-objective selection
        plt.figure(figsize=(10, 8))
        plt.scatter(M_reduced[:400, 0], M_reduced[:400, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[selected_custom, 0], M_reduced[selected_custom, 1], alpha=0.8, c='blue')
        plt.title('Custom Multi-objective Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig('trainsel_advanced_selection.png')
        print("\nSelection plot saved as 'trainsel_advanced_selection.png'")
        
        # Plot multi-set selection
        plt.figure(figsize=(12, 6))
        
        # First set
        plt.subplot(1, 2, 1)
        plt.scatter(M_reduced[:200, 0], M_reduced[:200, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[result_multi.selected_indices[0], 0], 
                   M_reduced[result_multi.selected_indices[0], 1], alpha=0.8, c='blue')
        plt.title('Set 1 Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # Second set
        plt.subplot(1, 2, 2)
        plt.scatter(M_reduced[200:400, 0], M_reduced[200:400, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[result_multi.selected_indices[1], 0], 
                   M_reduced[result_multi.selected_indices[1], 1], alpha=0.8, c='red')
        plt.title('Set 2 Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('trainsel_multiset_selection.png')
        print("Multi-set selection plot saved as 'trainsel_multiset_selection.png'")
    except Exception as e:
        print(f"\nCould not create visualization: {str(e)}")
    
    print("\nAdvanced example completed.")

if __name__ == "__main__":
    main()