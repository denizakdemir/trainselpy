"""
Discrete Variable Optimization Example for TrainSelPy.

This example demonstrates how to use TrainSelPy for optimizing discrete variables
in a real-world setting: Feature Selection on the Breast Cancer dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    train_sel_control
)

def load_real_data():
    """
    Load the Breast Cancer dataset from scikit-learn.
    
    Returns
    -------
    tuple
        X (features), y (target), feature_names
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names

def feature_selection_fitness(selected_indices, data):
    """
    Calculate fitness for feature selection using Cross-Validation accuracy.
    
    Parameters
    ----------
    selected_indices : list
        List of selected feature indices
    data : dict
        Data dictionary with 'X', 'y', and 'model'
        
    Returns
    -------
    float
        Fitness value (higher is better)
    """
    # If no features selected, return poor fitness
    if not selected_indices:
        return 0.5  # Return baseline accuracy (random guess)
        
    X = data['X']
    y = data['y']
    
    # Subset features
    X_subset = X[:, selected_indices]
    
    # Create valid pipeline with scaler (important for Logistic Regression)
    # Using a simpler model or fewer folds to keep it fast
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=42))
    
    # Calculate cross-validation score
    scores = cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy')
    mean_score = np.mean(scores)
    
    # Penalize for number of features to encourage parsimony
    # Penalty: 0.001 per feature
    n_features = X.shape[1]
    n_selected = len(selected_indices)
    penalty = 0.001 * n_selected
    
    return mean_score - penalty

def main():
    """Run feature selection optimization example."""
    print("TrainSelPy Feature Selection Example (Breast Cancer Dataset)")
    print("----------------------------------------------------------")
    
    # Load data
    print("\nLoading Breast Cancer dataset...")
    X, y, feature_names = load_real_data()
    n_samples, n_features = X.shape
    print(f"Dataset: {n_samples} samples, {n_features} features")
    
    # Calculate baseline performance (all features)
    print("\nCalculating baseline performance (all features)...")
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=42))
    baseline_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    baseline_acc = np.mean(baseline_scores)
    print(f"Baseline Accuracy (3-fold CV): {baseline_acc:.4f}")
    
    # Create TrainSel data object
    print("\nSetting up optimization...")
    # Pass X and y in the data dict for the fitness function
    ts_data = make_data(M=X) # M is required by make_data but mostly acts as a placeholder or grid here
    ts_data['X'] = X
    ts_data['y'] = y
    
    
    # Set control parameters
    control = train_sel_control(
        size="free",
        niterations=50,      # Fewer iterations for demo speed
        minitbefstop=10,     # Stop early if converged
        npop=50,             # Population size
        mutprob=0.1,         # Mutation probability
        mutintensity=0.2,    # Mutation intensity
        crossprob=0.7,       # Crossover probability
        crossintensity=0.5,  # Crossover intensity
        progress=True        # Show progress
    )
    
    # Run optimization
    print("\nRunning optimization to find optimal feature subset...")
    print("Objective: Maximize CV Accuracy - 0.001 * n_features")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_features))],  # All features are candidates
        setsizes=[n_features // 2],            # Start with half features (initial size hint)
        settypes=["UOS"],                      # Unordered Set selection
        stat=feature_selection_fitness,        # Custom fitness function
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract results
    best_indices = result.selected_indices[0]
    best_indices.sort()
    
    final_fitness = feature_selection_fitness(best_indices, ts_data)
    
    # Get accuracy without penalty for reporting
    X_subset = X[:, best_indices]
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=42))
    final_acc = np.mean(cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy'))
    
    print("\nResults (Phase 1: Main Effects):")
    print(f"Selected {len(best_indices)}/{n_features} features")
    print(f"Selected Feature Names:\n{feature_names[best_indices]}")
    print(f"\nBaseline Accuracy: {baseline_acc:.4f} (with {n_features} features)")
    print(f"Selected Accuracy: {final_acc:.4f} (with {len(best_indices)} features)")
    print(f"Fitness Score:     {final_fitness:.4f}")
    
    # ---------------------------------------------------------
    # Phase 2: Interaction Search
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 2: Searching for Interaction Effects (Degree 2 & 3)")
    print("="*60)
    print("Generating interactions relative to the selected variables found in Phase 1...")
    
    # Generate interactions only for the selected features
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X[:, best_indices])
    interaction_names = poly.get_feature_names_out(feature_names[best_indices])
    
    n_interactions = X_interactions.shape[1]
    print(f"Generated {n_interactions} features (main effects + interactions) from {len(best_indices)} original variables.")
    
    # Setup data for Phase 2
    ts_data_phase2 = make_data(M=X_interactions)
    ts_data_phase2['X'] = X_interactions
    ts_data_phase2['y'] = y
    
    print("\nRunning optimization on main effects + interactions...")
    start_time_2 = time.time()
    
    # Use lighter control for Phase 2 as the feature space is larger
    control_phase2 = train_sel_control(
        size="free",
        niterations=20,      # Reduced iterations
        minitbefstop=5,      
        npop=30,             # Reduced population
        mutprob=0.1,         
        mutintensity=0.2,    
        crossprob=0.7,       
        crossintensity=0.5,  
        progress=True        
    )

    # We can reuse the same fitness function as it just operates on the provided 'X' matrix
    result_phase2 = train_sel(
        data=ts_data_phase2,
        candidates=[list(range(n_interactions))],
        setsizes=[n_interactions // 3],  # Start smaller
        settypes=["UOS"],
        stat=feature_selection_fitness,
        control=control_phase2,
        verbose=True
    )
    
    runtime_2 = time.time() - start_time_2
    print(f"\nPhase 2 Optimization completed in {runtime_2:.2f} seconds")
    
    # Extract results Phase 2
    best_indices_2 = result_phase2.selected_indices[0]
    best_indices_2.sort()
    
    final_fitness_2 = feature_selection_fitness(best_indices_2, ts_data_phase2)
    
    # Get accuracy without penalty
    X_subset_2 = X_interactions[:, best_indices_2]
    clf_2 = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=42))
    final_acc_2 = np.mean(cross_val_score(clf_2, X_subset_2, y, cv=3, scoring='accuracy'))
    
    print("\nResults (Phase 2: Interactions):")
    print(f"Selected {len(best_indices_2)}/{n_interactions} features")
    print("Top 10 Selected Features/Interactions:")
    for idx in best_indices_2[:10]:
        print(f"  - {interaction_names[idx]}")
    if len(best_indices_2) > 10:
        print(f"  ... and {len(best_indices_2)-10} more.")
        
    print(f"\nPhase 1 Accuracy: {final_acc:.4f}")
    print(f"Phase 2 Accuracy: {final_acc_2:.4f}")
    print(f"Improvement:      {(final_acc_2 - final_acc):.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    # Bar chart of accuracy
    labels = ['All Features', 'Phase 1 (Main Effects)', 'Phase 2 (With Interactions)']
    accuracies = [baseline_acc, final_acc, final_acc_2]
    bars = plt.bar(labels, accuracies, color=['gray', 'green', 'purple'])
    plt.ylim(min(accuracies)*0.98, 1.0) 
    plt.ylabel('CV Accuracy')
    plt.title('Feature Selection Performance: Main Effects vs Interactions')
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig('feature_selection_interactions_result.png')
    print("\nVisualization saved as 'feature_selection_interactions_result.png'")

if __name__ == "__main__":
    main()