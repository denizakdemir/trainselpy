"""
A simple example using our fixed CDMean criterion.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add the examples directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the custom CDMean implementation
from custom_cdmean import custom_cdmean

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    set_control_default,
    dopt
)

# Create a simple dataset for testing
def create_test_dataset(n_samples=50, n_features=30):
    """Create a simple test dataset."""
    np.random.seed(42)
    
    # Create a marker matrix
    M = np.random.choice([-1, 0, 1], size=(n_samples, n_features), p=[0.25, 0.5, 0.25])
    
    # Create a relationship matrix
    K = np.dot(M, M.T) / n_features
    K += np.eye(n_samples) * 1e-6  # Add small diagonal for stability
    
    return M, K

def main():
    """Run CDMean example with the fixed implementation."""
    
    print("TrainSelPy CDMean Example (Fixed Implementation)")
    print("---------------------------------------------")
    
    # Create test dataset
    print("\nCreating test dataset...")
    M, K = create_test_dataset(n_samples=50, n_features=30)
    
    print(f"Created dataset with {M.shape[0]} samples and {M.shape[1]} features")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(K=K)
    ts_data["FeatureMat"] = M  # For D-optimality
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 30
    control["npop"] = 100
    
    # Run with fixed CDMean
    print("\nRunning TrainSel with fixed CDMean criterion...")
    result = train_sel(
        data=ts_data,
        candidates=[list(range(len(M)))],
        setsizes=[10],
        settypes=["UOS"],
        stat=custom_cdmean,  # Use our fixed implementation
        control=control,
        verbose=True
    )
    
    print(f"\nSelected {len(result.selected_indices[0])} samples:")
    print(f"Selected indices: {result.selected_indices[0]}")
    print(f"Final fitness (CDMean): {result.fitness:.6f}")
    
    # Run with D-optimality for comparison
    print("\nRunning TrainSel with D-optimality for comparison...")
    result_dopt = train_sel(
        data=ts_data,
        candidates=[list(range(len(M)))],
        setsizes=[10],
        settypes=["UOS"],
        stat=dopt,
        control=control,
        verbose=True
    )
    
    print(f"\nSelected {len(result_dopt.selected_indices[0])} samples:")
    print(f"Selected indices: {result_dopt.selected_indices[0]}")
    print(f"Final fitness (D-optimality): {result_dopt.fitness:.6f}")
    
    # Compare selections
    cdmean_set = set(result.selected_indices[0])
    dopt_set = set(result_dopt.selected_indices[0])
    common = cdmean_set.intersection(dopt_set)
    
    print(f"\nCommon elements between CDMean and D-optimality: {len(common)}")
    print(f"Common indices: {sorted(list(common))}")
    
    print("\nCDMean example completed successfully.")

if __name__ == "__main__":
    main()