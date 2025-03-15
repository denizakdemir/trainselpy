"""
Basic example of using the TrainSelPy package, restricted to the first 200 genotypes.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to ensure we find our local fixes
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
    """Run a restricted example of TrainSelPy with the first 200 genotypes only."""
    
    print("TrainSelPy Basic Example (Restricted to 200 Genotypes)")
    print("------------------------------------------------------")
    
    # -------------------------------------------------------------------------
    # Load wheat data.
    # -------------------------------------------------------------------------
    print("\nLoading and subsetting wheat data to the first 200 genotypes...")
    M_full = wheat_data['M']  # Marker data (DataFrame or NumPy array)
    K_full = wheat_data['K']  # Kinship data (DataFrame or NumPy array)
    Y_full = wheat_data['Y']  # Phenotype data (DataFrame or Series)
    
    # -------------------------------------------------------------------------
    # IMPORTANT: Use .iloc for pandas objects, or .values to convert to NumPy.
    # -------------------------------------------------------------------------
    # Example 1: Keep as pandas objects, but with correct subsetting:
    M_200 = M_full.iloc[:200, :]
    K_200 = K_full.iloc[:200, :200]
    Y_200 = Y_full.iloc[:200]
    

    print(f"Subset data dimensions: {M_200.shape[0]} lines, {M_200.shape[1]} markers")
    
    # -------------------------------------------------------------------------
    # Create the TrainSel data object.
    # -------------------------------------------------------------------------
    print("\nCreating the TrainSel data object...")
    # TrainSelPy typically expects NumPy arrays for K, so ensure the correct format:
    if isinstance(K_200, pd.DataFrame):
        K_200 = K_200.values
    ts_data = make_data(K=K_200)
    
    # -------------------------------------------------------------------------
    # Set control parameters.
    # -------------------------------------------------------------------------
    print("\nSetting control parameters for the selection algorithm...")
    control = set_control_default()
    control["niterations"] = 10  # Reduced for a faster example
    control["npop"] = 100
    
    # -------------------------------------------------------------------------
    # Run the selection algorithm with the CDMean (Custom) criterion.
    # -------------------------------------------------------------------------
    print("\nRunning TrainSel with CDMean criterion...")
    start_time = time.time()
    
    # Make sure we pass a list of candidates the same size as M_200
    candidates = [list(range(M_200.shape[0]))]
    
    result_cdmean = train_sel(
        data=ts_data,
        candidates=candidates,       # All 200 genotypes
        setsizes=[30],               # Select 30 lines
        settypes=["UOS"],            # Unordered set
        stat=cdmean_opt,             # Use custom CDMean criterion
        control=control,
        verbose=True
    )
    runtime = time.time() - start_time
    
    print(f"\nSelection completed in {runtime:.2f} seconds")
    print(f"Final fitness (CDMean): {result_cdmean.fitness:.6f}")
    
    # Extract the selected indices
    selected_cdmean = result_cdmean.selected_indices[0]
    print(f"Selected {len(selected_cdmean)} lines:")
    print(f"First 10 selected indices: {selected_cdmean[:10]}...")
    
    # -------------------------------------------------------------------------
    # Run the selection algorithm with the D-optimality criterion.
    # -------------------------------------------------------------------------
    print("\n\nRunning TrainSel with D-optimality criterion...")
    
    # For D-optimality, we need a "FeatureMat" in ts_data
    # Convert M_200 to NumPy array if it's still a DataFrame
    if isinstance(M_200, pd.DataFrame):
        M_200_values = M_200.values
    else:
        M_200_values = M_200
    # sample 1000 markers for speed
    M_200_values = M_200_values[:, :1000]
    ts_data["FeatureMat"] = M_200_values
    
    start_time = time.time()
    result_dopt = train_sel(
        data=ts_data,
        candidates=candidates,       # All 200 genotypes
        setsizes=[30],               # Select 30 lines
        settypes=["UOS"],            # Unordered set
        stat=dopt,                   # D-optimality criterion
        control=control,
        verbose=True
    )
    runtime = time.time() - start_time
    
    print(f"\nSelection completed in {runtime:.2f} seconds")
    print(f"Final fitness (D-optimality): {result_dopt.fitness:.6f}")
    
    # Extract the selected indices
    selected_dopt = result_dopt.selected_indices[0]
    print(f"Selected {len(selected_dopt)} lines:")
    print(f"First 10 selected indices: {selected_dopt[:10]}...")
    
    # -------------------------------------------------------------------------
    # Compare the selected sets.
    # -------------------------------------------------------------------------
    common = set(selected_cdmean).intersection(set(selected_dopt))
    print(f"\nNumber of lines in common between the two selections: {len(common)}")
    
    # -------------------------------------------------------------------------
    # Visualization (optional) via PCA, if possible.
    # -------------------------------------------------------------------------
    try:
        from sklearn.decomposition import PCA
        
        # Perform PCA on the 200-line marker matrix
        pca = PCA(n_components=2)
        M_reduced = pca.fit_transform(M_200_values)
        
        # Create a comparison plot of both selections
        plt.figure(figsize=(12, 6))
        
        # CDMean selection
        plt.subplot(1, 2, 1)
        plt.scatter(M_reduced[:, 0], M_reduced[:, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[selected_cdmean, 0], M_reduced[selected_cdmean, 1],
                    alpha=0.8, c='blue')
        plt.title('CDMean Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # D-optimality selection
        plt.subplot(1, 2, 2)
        plt.scatter(M_reduced[:, 0], M_reduced[:, 1], alpha=0.2, c='gray')
        plt.scatter(M_reduced[selected_dopt, 0], M_reduced[selected_dopt, 1],
                    alpha=0.8, c='red')
        plt.title('D-optimality Selection')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('trainsel_basic_comparison.png')
        print("\nComparison plot saved as 'trainsel_basic_comparison.png'.")
    except Exception as e:
        print(f"\nCould not create visualization: {str(e)}")
    
    print("\nBasic example completed.")

if __name__ == "__main__":
    main()
