"""
Discrete Variable Optimization Example for TrainSelPy.

This example demonstrates how to use TrainSelPy for optimizing discrete variables
in a simplified setting, focusing on selecting optimal subsets of features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    Generate synthetic data with feature clusters and target variable.
    
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
        Features, target, feature importance
    """
    np.random.seed(random_seed)
    
    # Create feature clusters with varying importance
    # Cluster 1: Important features highly correlated with target
    important_features = np.random.normal(0, 1, size=(n_samples, n_features // 4))
    
    # Cluster 2: Moderately important features
    moderate_features = np.random.normal(0, 1, size=(n_samples, n_features // 4))
    
    # Cluster 3: Noisy features with little relevance
    noisy_features = np.random.normal(0, 1, size=(n_samples, n_features // 2))
    
    # Combine features
    features = np.hstack((important_features, moderate_features, noisy_features))
    
    # Create target variable
    # Highly dependent on important features
    target = np.sum(important_features, axis=1) * 0.8
    
    # Somewhat dependent on moderate features
    target += np.sum(moderate_features, axis=1) * 0.3
    
    # Add noise
    target += np.random.normal(0, 0.5, size=n_samples)
    
    # Create feature importance values (for evaluation)
    importance = np.zeros(n_features)
    importance[:n_features//4] = np.random.uniform(0.7, 1.0, n_features//4)
    importance[n_features//4:n_features//2] = np.random.uniform(0.3, 0.7, n_features//4)
    importance[n_features//2:] = np.random.uniform(0.0, 0.3, n_features//2)
    
    return features, target, importance

def feature_selection_fitness(selected_features, data):
    """
    Calculate fitness for feature selection.
    
    Parameters
    ----------
    selected_features : list
        List of selected feature indices
    data : dict
        Data dictionary with feature importance
        
    Returns
    -------
    float
        Fitness value (higher is better)
    """
    # Extract feature importance
    importance = data['importance']
    
    # Sum importance of selected features
    total_importance = np.sum(importance[selected_features])
    
    # Penalize for selecting too many features (encourage parsimony)
    n_features = len(importance)
    parsimony_penalty = len(selected_features) / n_features
    
    # Final fitness combines importance and parsimony
    fitness = total_importance * (1 - 0.3 * parsimony_penalty)
    
    return fitness

def main():
    """Run a discrete variable optimization example using TrainSelPy."""
    print("TrainSelPy Discrete Variable Optimization Example")
    print("-----------------------------------------------")
    
    # Generate synthetic data
    n_samples = 100
    n_features = 20
    print(f"\nGenerating synthetic data with {n_samples} samples and {n_features} features...")
    features, target, importance = generate_synthetic_data(n_samples, n_features)
    
    # Display feature importance
    print("\nFeature Importance:")
    for i in range(n_features):
        category = "High" if importance[i] >= 0.7 else "Medium" if importance[i] >= 0.3 else "Low"
        print(f"Feature {i+1}: {importance[i]:.4f} ({category})")
    
    # Create TrainSel data object
    print("\nSetting up optimization...")
    ts_data = make_data(M=features)
    ts_data['target'] = target
    ts_data['importance'] = importance
    
    # Set control parameters
    control = train_sel_control(
        size="free",
        niterations=100,     # Number of iterations
        minitbefstop=30,     # Stop if no improvement for this many iterations
        npop=100,            # Population size
        mutprob=0.1,         # Mutation probability
        mutintensity=0.2,    # Mutation intensity
        crossprob=0.7,       # Crossover probability
        crossintensity=0.5,  # Crossover intensity
        progress=True        # Show progress
    )
    
    # Run the discrete variable optimization (feature selection)
    print("\nRunning optimization to find optimal feature subset...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[list(range(n_features))],  # All features are candidates
        setsizes=[n_features // 3],            # Select one-third of features
        settypes=["UOS"],                      # Unordered Set selection
        stat=feature_selection_fitness,        # Custom fitness function
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract the optimized feature subset
    selected_features = result.selected_indices[0]
    selected_features.sort()  # Sort for better readability
    
    # Calculate fitness and total importance
    fitness = feature_selection_fitness(selected_features, ts_data)
    total_importance = np.sum(importance[selected_features])
    
    # Calculate importance for a naive selection (first n_features//3 features)
    naive_selection = list(range(n_features // 3))
    naive_importance = np.sum(importance[naive_selection])
    
    # Calculate importance for random selection (average of 10 random selections)
    random_importance = 0
    for _ in range(10):
        random_selection = np.random.choice(range(n_features), size=n_features//3, replace=False)
        random_importance += np.sum(importance[random_selection])
    random_importance /= 10
    
    # Print results
    print("\nResults:")
    print(f"Selected {len(selected_features)} features: {selected_features}")
    print(f"Total importance of selected features: {total_importance:.4f}")
    print(f"Fitness score: {fitness:.4f}")
    print(f"Naive selection importance: {naive_importance:.4f}")
    print(f"Random selection importance (avg): {random_importance:.4f}")
    print(f"Improvement over naive: {((total_importance/naive_importance)-1)*100:.2f}%")
    print(f"Improvement over random: {((total_importance/random_importance)-1)*100:.2f}%")
    
    # Visualize the results
    plt.figure(figsize=(12, 10))
    
    # 1. Feature importance comparison
    plt.subplot(2, 1, 1)
    
    # Prepare data for visualization
    all_indices = np.arange(n_features)
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_features] = True
    
    # Create bar colors
    colors = ['blue' if selected else 'lightgray' for selected in selected_mask]
    
    # Plot feature importance
    plt.bar(all_indices, importance, color=colors, alpha=0.7)
    plt.title('Feature Importance (Selected features in blue)')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    
    # Add horizontal line for average importance of selected features
    avg_selected = np.mean(importance[selected_features])
    plt.axhline(y=avg_selected, color='red', linestyle='--', 
                label=f'Avg Selected: {avg_selected:.4f}')
    
    # Add horizontal line for average importance of all features
    avg_all = np.mean(importance)
    plt.axhline(y=avg_all, color='green', linestyle=':', 
                label=f'Avg All: {avg_all:.4f}')
    
    plt.legend()
    
    # 2. PCA of features with selected features highlighted
    plt.subplot(2, 1, 2)
    
    # Standardize features for PCA
    features_scaled = StandardScaler().fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled.T)  # Transpose to get PCA of features
    
    # Create labels for feature clusters
    labels = ['High Importance' if i < n_features//4 else 
             'Medium Importance' if i < n_features//2 else 
             'Low Importance' for i in range(n_features)]
    
    # Prepare data for scatter plot
    unique_labels = ['High Importance', 'Medium Importance', 'Low Importance']
    colors = {'High Importance': 'red', 'Medium Importance': 'blue', 'Low Importance': 'green'}
    
    # Plot all features by category
    for label in unique_labels:
        mask = [l == label for l in labels]
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   alpha=0.6, label=label, color=colors[label])
    
    # Highlight selected features
    plt.scatter(pca_result[selected_features, 0], pca_result[selected_features, 1], 
               s=100, facecolors='none', edgecolors='black', linewidth=2, 
               label='Selected Features')
    
    plt.title('PCA of Features (features in feature space)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('discrete_optimization.png')
    print("\nVisualization saved as 'discrete_optimization.png'")
    
    print("\nDiscrete optimization example completed.")

if __name__ == "__main__":
    main()