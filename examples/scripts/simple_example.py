""" Minimal working example for TrainSelPy. """
import numpy as np
import pandas as pd
from trainselpy import (
    make_data,
    train_sel,
    set_control_default,
    dopt
)

def main():
    """Run a minimal working example of TrainSelPy."""
    print("TrainSelPy Simple Example")
    print("-----------------------")

    # Create a small test dataset
    print("\nCreating test dataset...")
    n_samples = 1000
    n_features = 20

    # Create a marker matrix
    M = np.random.choice([-1, 0, 1], size=(n_samples, n_features), p=[0.25, 0.5, 0.25])

    # Create a relationship matrix - need to transpose M correctly
    K = np.dot(M, M.T) / n_features
    # Add a small value to the diagonal to ensure positive definiteness
    K += np.eye(n_samples) * 1e-6

    print(f"Created dataset with {n_samples} samples and {n_features} features")

    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(M=M)
    # For D-optimality
    ts_data["FeatureMat"] = M
    # Add the relationship matrix to the data object
    ts_data["K"] = K

    # Set control parameters (limited iterations for example)
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 2000
    control["npop"] = 100

    # Run the selection algorithm with D-optimality
    print("\nRunning TrainSel with D-optimality criterion...")
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_samples))],  # Select from all samples
        setsizes=[10],  # Select 10 samples
        settypes=["UOS"],  # Unordered set
        stat=dopt,  # Use D-optimality
        control=control,
        verbose=True
    )

    print(f"\nSelected {len(result.selected_indices[0])} samples:")
    print(f"Selected indices: {result.selected_indices[0]}")
    print(f"Final fitness (D-optimality): {result.fitness:.6f}")
    print("\nSimple example completed.")

if __name__ == "__main__":
    main()