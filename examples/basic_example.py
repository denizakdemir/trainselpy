"""
Basic example of using the TrainSelPy package.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to ensure we find our fixes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    set_control_default,
    dopt
)

# Import our custom implementation of cdmean
from custom_cdmean import custom_cdmean as cdmean_opt

# Load example data
from trainselpy.data import wheat_data

def main():
    """Run a basic example of TrainSelPy."""
    
    print("TrainSelPy Basic Example (Fixed)")
    print("-----------------------------")
    
    # Load wheat data
    print("\nLoading wheat data...")
    M = wheat_data['M']
    K = wheat_data['K']
    Y = wheat_data['Y']
    
    print(f"Data dimensions: {M.shape[0]} lines, {M.shape[1]} markers")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    # Use make_data with the same dimensions as the data
    ts_data = make_data(K=K)
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 10  # Reduce for faster example
    control["npop"] = 100
    
    # Run the selection algorithm with CDMean criterion (default)
    print("\nRunning TrainSel with CDMean criterion...")
    start_time = time.time()
    result = train_sel(
        data=ts_data,
        candidates=[list(range(200))],  # Select from first 200 lines
        setsizes=[30],                 # Select 30 lines
        settypes=["UOS"],             # Unordered set
        stat=cdmean_opt,              # Use CDMean criterion from fixed module
        control=control,
        verbose=True
    )
    runtime = time.time() - start_time
    
    print(f"\nSelection completed in {runtime:.2f} seconds")
    print(f"Final fitness (CDMean): {result.fitness:.6f}")
    
    # Get the selected lines
    selected = result.selected_indices[0]
    print(f"Selected {len(selected)} lines:")
    print(f"First 10 selected indices: {selected[:10]}...")
    
    # Run the selection algorithm with D-optimality
    print("\n\nRunning TrainSel with D-optimality criterion...")
    ts_data["FeatureMat"] = M  # Add feature matrix for D-optimality
    
    start_time = time.time()
    result_dopt = train_sel(
        data=ts_data,
        candidates=[list(range(200))],  # Select from first 200 lines
        setsizes=[30],                 # Select 30 lines
        settypes=["UOS"],             # Unordered set
        stat=dopt,                    # Use D-optimality criterion
        control=control,
        verbose=True
    )
    runtime = time.time() - start_time
    
    print(f"\nSelection completed in {runtime:.2f} seconds")
    print(f"Final fitness (D-optimality): {result_dopt.fitness:.6f}")
    
    # Get the selected lines
    selected_dopt = result_dopt.selected_indices[0]
    print(f"Selected {len(selected_dopt)} lines:")
    print(f"First 10 selected indices: {selected_dopt[:10]}...")
    
    # Compare the selected sets
    common = set(selected).intersection(set(selected_dopt))
    print(f"\nNumber of lines in common between the two criteria: {len(common)}")
    
    # Visualize the selections (if G matrix is available)
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to the marker matrix
        pca = PCA(n_components=2)
        M_reduced = pca.fit_transform(M)
        
        # Plot both selections
        plt.figure(figsize=(12, 6))
        
        # CDMean selection
        plt.subplot(1, 2, 1)
        plt.scatter(M_reduced[:600, 0], M_reduced[:600, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[selected, 0], M_reduced[selected, 1], alpha=0.8, c='blue')
        plt.title('CDMean Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # D-optimality selection
        plt.subplot(1, 2, 2)
        plt.scatter(M_reduced[:600, 0], M_reduced[:600, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[selected_dopt, 0], M_reduced[selected_dopt, 1], alpha=0.8, c='red')
        plt.title('D-optimality Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('trainsel_basic_comparison.png')
        print("\nComparison plot saved as 'trainsel_basic_comparison.png'")
    except Exception as e:
        print(f"\nCould not create visualization: {str(e)}")
    
    print("\nBasic example completed.")

if __name__ == "__main__":
    main()