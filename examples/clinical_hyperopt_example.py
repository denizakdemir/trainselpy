"""
Example demonstrating hyperparameter tuning and model architecture search with TrainSelPy.

This example uses the Breast Cancer Wisconsin (Diagnostic) dataset to optimize 
a Multi-Layer Perceptron (MLP) classifier. It demonstrates how to use TrainSelPy
to simultaneously select:
1. Model Architecture (Discrete choice)
2. Activation Function (Discrete choice)
3. Regularization Parameter (Continuous)
4. Learning Rate (Continuous)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    set_control_default
)

# Suppress convergence warnings for faster/cleaner output during optimization
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define the search space
ARCHITECTURES = [
    (50,),
    (100,),
    (50, 25),
    (100, 50),
    (100, 50, 25),
    (200, 100, 50),
    (50, 50, 50)
]

ACTIVATIONS = ['tanh', 'relu', 'logistic']

def get_hyperparameters(arch_idx, act_idx, alpha_val, lr_val):
    """Decode the optimization variables into actual hyperparameters."""
    # Architecture
    hidden_layer_sizes = ARCHITECTURES[arch_idx]
    
    # Activation
    activation = ACTIVATIONS[act_idx]
    
    # Alpha (L2 penalty) - Map 0-1 to 1e-6 to 1e-1 (log scale)
    # 10^(-6 + 5*val)
    alpha = 10**(-6 + 5 * alpha_val)
    
    # Learning Rate - Map 0-1 to 1e-5 to 1e-1 (log scale)
    # 10^(-5 + 4*val)
    learning_rate_init = 10**(-5 + 4 * lr_val)
    
    return {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init
    }

def fitness_function(int_solution_arch, int_solution_act, dbl_solution_alpha, dbl_solution_lr, data):
    """
    Fitness function for hyperparameter tuning.
    
    Parameters:
    - int_solution_arch: Selected index for architecture
    - int_solution_act: Selected index for activation
    - dbl_solution_alpha: Selected value for alpha (0-1)
    - dbl_solution_lr: Selected value for learning rate (0-1)
    - data: Dictionary containing the dataset
    
    Returns:
    - Mean Cross-Validation Accuracy
    """
    # Handle list wrapping (TrainSelPy might pass lists even for single items)
    # Also handle numpy arrays if present
    def extract_scalar(val):
        if isinstance(val, list):
            val = val[0]
        if hasattr(val, 'item'):
            return val.item()
        return val

    arch_idx = extract_scalar(int_solution_arch)
    act_idx = extract_scalar(int_solution_act)
    alpha_val = extract_scalar(dbl_solution_alpha)
    lr_val = extract_scalar(dbl_solution_lr)
    
    # Get hyperparameters
    params = get_hyperparameters(arch_idx, act_idx, alpha_val, lr_val)
    
    # Create model
    clf = MLPClassifier(
        random_state=42,
        max_iter=200, # Keep it low for speed in this example
        **params
    )
    
    # Get data
    X = data['X_scaled']
    y = data['y']
    
    # Perform Cross-Validation
    # Use 3-fold for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    return np.mean(scores)

# Wrapper for TrainSelPy which expects specific signature based on settypes
def fitness_wrapper(int_sols, dbl_sols, data):
    """
    Wrapper to unpack the solutions correctly.
    TrainSelPy passes:
    - int_sols: list of integer solutions (one per 'UOS'/'OS' set)
    - dbl_sols: list of double solutions (one per 'DBL' set)
    """
    # We have 2 integer sets (Arch, Act) and 2 double sets (Alpha, LR)
    # int_sols will be [arch_indices, act_indices]
    # dbl_sols will be [alpha_values, lr_values]
    
    return fitness_function(int_sols[0], int_sols[1], dbl_sols[0], dbl_sols[1], data)

def main():
    print("TrainSelPy Clinical Hyperparameter Tuning Example")
    print("-----------------------------------------------")
    
    # 1. Load Data
    print("Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create TrainSel Data Object
    # We don't really need 'M' or 'K' for this specific usage as we use custom fitness,
    # but make_data requires something. We can pass a dummy or the actual data.
    ts_data = make_data(M=X_scaled) 
    ts_data['X_scaled'] = X_scaled
    ts_data['y'] = y
    
    # 2. Setup Optimization
    
    # Candidates
    # 1. Architecture indices
    cand_arch = list(range(len(ARCHITECTURES)))
    # 2. Activation indices
    cand_act = list(range(len(ACTIVATIONS)))
    # 3. Alpha (continuous 0-1) - Initial guess
    cand_alpha = [0.5]
    # 4. Learning Rate (continuous 0-1) - Initial guess
    cand_lr = [0.5]
    
    candidates = [cand_arch, cand_act, cand_alpha, cand_lr]
    
    # Set Sizes (Select 1 of each)
    setsizes = [1, 1, 1, 1]
    
    # Set Types
    # UOS for discrete choices, DBL for continuous
    settypes = ["UOS", "UOS", "DBL", "DBL"]
    
    # Control Parameters
    control = set_control_default()
    control["niterations"] = 20
    control["npop"] = 20
    control["niterSANN"] = 10
    control["mutprob"] = 0.1
    control["mutintensity"] = 0.2
    
    print("\nSearch Space:")
    print(f"- Architectures: {len(ARCHITECTURES)} configurations")
    print(f"- Activations: {ACTIVATIONS}")
    print("- Alpha: 1e-6 to 1e-1 (log scale)")
    print("- Learning Rate: 1e-5 to 1e-1 (log scale)")
    
    print("\nStarting Optimization...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=candidates,
        setsizes=setsizes,
        settypes=settypes,
        stat=fitness_wrapper,
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization finished in {runtime:.2f} seconds.")
    
    # 3. Analyze Results
    
    # Extract selected values
    # result.selected_indices contains the indices for UOS sets
    # result.selected_values contains the values for DBL sets (and indices for UOS)
    
    # Note: For DBL sets, selected_values has the actual float values.
    # For UOS sets, selected_indices has the indices.
    
    # Let's look at how train_sel returns things.
    # Based on core.py, it returns selected_indices and selected_values.
    # For UOS, selected_indices[i] is the list of selected indices.
    # For DBL, selected_values[i] is the list of selected values.
    
    # However, the indices in result.selected_indices correspond to the order of sets.
    # But wait, `train_sel` usually returns `selected_indices` for all sets.
    # For DBL sets, `selected_indices` might not be meaningful or might be empty/dummy?
    # Actually, looking at `genetic_algorithm.py` (implied), DBL variables are handled separately.
    # Let's rely on what we passed to fitness_wrapper.
    
    # The result object structure:
    # selected_indices: List[List[int]]
    # selected_values: List[Any]
    
    # For UOS sets (0 and 1):
    sel_arch_idx = result.selected_indices[0][0]
    sel_act_idx = result.selected_indices[1][0]
    
    # For DBL sets (2 and 3):
    # The `selected_values` list should contain the optimized values for DBL sets at the corresponding positions?
    # Or maybe `selected_values` contains everything?
    # Let's inspect `result.selected_values`.
    
    # In `mixed_type_optimization_example.py`:
    # result.selected_values = [result.selected_values[0], optimal_doses]
    # It seems `selected_values` has an entry for each set.
    
    sel_alpha_val = result.selected_values[0][0]
    sel_lr_val = result.selected_values[1][0]
    
    best_params = get_hyperparameters(sel_arch_idx, sel_act_idx, sel_alpha_val, sel_lr_val)
    
    print("\nBest Configuration Found:")
    print(f"Architecture: {best_params['hidden_layer_sizes']}")
    print(f"Activation: {best_params['activation']}")
    print(f"Alpha: {best_params['alpha']:.6f}")
    print(f"Learning Rate: {best_params['learning_rate_init']:.6f}")
    print(f"Best CV Accuracy: {result.fitness:.4f}")
    
    # Compare with default parameters
    print("\nComparison with Default Parameters:")
    default_clf = MLPClassifier(random_state=42, max_iter=200)
    default_scores = cross_val_score(default_clf, X_scaled, y, cv=3, scoring='accuracy')
    print(f"Default MLP Accuracy: {np.mean(default_scores):.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(result.fitness_history)
    plt.title('Optimization History')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (CV Accuracy)')
    plt.grid(True)
    plt.savefig('clinical_hyperopt_history.png')
    print("\nHistory plot saved to 'clinical_hyperopt_history.png'")

if __name__ == "__main__":
    main()
