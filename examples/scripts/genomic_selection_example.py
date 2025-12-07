"""
Genomic selection example for TrainSelPy.

This example demonstrates how to use TrainSelPy for optimizing training
populations in a genomic selection context with a simulated dataset.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    set_control_default,
    calculate_relationship_matrix,
    cdmean_opt,
)

def simulate_breeding_population(n_individuals=200, n_markers=500, h2=0.5):
    """
    Simulate a breeding population with genetic markers and phenotypes.
    
    Parameters
    ----------
    n_individuals : int
        Number of individuals
    n_markers : int
        Number of markers
    h2 : float
        Heritability (proportion of phenotypic variance explained by genetics)
        
    Returns
    -------
    dict
        Dictionary containing simulated data
    """
    # Simulate marker matrix
    M = np.random.choice([-1, 0, 1], size=(n_individuals, n_markers), p=[0.25, 0.5, 0.25])
    
    # Simulate QTL effects (most markers have small or no effect)
    qtl_effects = np.zeros(n_markers)
    # Select 10% of markers to be QTLs
    qtl_indices = np.random.choice(n_markers, size=int(n_markers * 0.1), replace=False)
    qtl_effects[qtl_indices] = np.random.normal(0, 1, size=len(qtl_indices))
    
    # Calculate genetic values
    genetic_values = np.dot(M, qtl_effects)
    
    # Scale genetic variance to desired heritability
    genetic_var = np.var(genetic_values)
    error_var = genetic_var * (1 - h2) / h2
    
    # Add environmental effects
    env_effects = np.random.normal(0, np.sqrt(error_var), size=n_individuals)
    phenotypes = genetic_values + env_effects
    
    # Calculate genomic relationship matrix
    K = calculate_relationship_matrix(M, method='additive')
    
    # Return the simulated data
    return {
        'M': M,
        'K': K,
        'genetic_values': genetic_values,
        'phenotypes': phenotypes,
        'qtl_effects': qtl_effects
    }

def prediction_accuracy_stat(training_indices, data):
    """
    Helper function to evaluate the prediction accuracy of a model
    trained on the selected individuals.

    Note: In this example, TrainSelPy optimizes the CDMean criterion
    via ``cdmean_opt``. Prediction accuracy is used only for
    post-hoc evaluation of the selected training sets.
    
    Parameters
    ----------
    training_indices : list
        Indices of individuals selected for training
    data : dict
        Data dictionary containing all required information
        
    Returns
    -------
    float
        Prediction accuracy (correlation between predicted and true genetic values)
    """
    # Extract data
    M = data["M"]
    phenotypes = data["phenotypes"]
    true_genetic_values = data["genetic_values"]
    test_indices = data["test_indices"]
    
    # Split data into training and testing sets
    X_train = M[training_indices, :]
    y_train = phenotypes[training_indices]
    X_test = M[test_indices, :]
    y_test_true = true_genetic_values[test_indices]
    
    # Train a Ridge regression model (commonly used in genomic selection)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predict genetic values for the test set
    y_test_pred = model.predict(X_test)
    
    # Calculate prediction accuracy as correlation
    accuracy = np.corrcoef(y_test_true, y_test_pred)[0, 1]
    
    return accuracy

def main():
    """Run a genomic selection example."""
    
    print("TrainSelPy Genomic Selection Example")
    print("----------------------------------")
    
    # Simulate a breeding population
    print("\nSimulating breeding population...")
    n_individuals = 200
    n_markers = 500
    sim_data = simulate_breeding_population(n_individuals, n_markers, h2=0.5)
    
    # Split into training candidates and test set
    print("\nSplitting data into training candidates and test set...")
    all_indices = np.arange(n_individuals)
    candidate_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=42
    )
    
    print(f"Number of candidates: {len(candidate_indices)}")
    print(f"Number of test individuals: {len(test_indices)}")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(M=sim_data["M"])
    
    # Add required data for our custom statistic function
    ts_data["M"] = sim_data["M"]
    ts_data["phenotypes"] = sim_data["phenotypes"]
    ts_data["genetic_values"] = sim_data["genetic_values"]
    ts_data["test_indices"] = test_indices
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = set_control_default()
    control["niterations"] = 30
    control["npop"] = 150
    
    # Compare random vs. optimized selection
    print("\nComparing random selection vs. optimized selection...")
    
    # Try different training set sizes
    training_sizes = [20, 40, 60, 80]
    random_accuracies = []
    optimized_accuracies = []
    random_cdmeans = []
    optimized_cdmeans = []
    
    for size in training_sizes:
        print(f"\nTraining set size: {size}")
        
        # Random selection
        print("  Random selection:")
        random_accuracy_sum = 0
        random_cdmean_sum = 0
        n_repeats = 5
        
        for i in range(n_repeats):
            random_indices = np.random.choice(candidate_indices, size=size, replace=False)
            accuracy = prediction_accuracy_stat(random_indices, ts_data)
            cd_val = cdmean_opt(random_indices.tolist(), ts_data)
            random_accuracy_sum += accuracy
            random_cdmean_sum += cd_val
            print(f"    Repeat {i+1}: Accuracy = {accuracy:.4f}, CDMean = {cd_val:.4f}")
        
        avg_random_accuracy = random_accuracy_sum / n_repeats
        avg_random_cdmean = random_cdmean_sum / n_repeats
        random_accuracies.append(avg_random_accuracy)
        random_cdmeans.append(avg_random_cdmean)
        print(f"  Average random selection accuracy: {avg_random_accuracy:.4f}")
        print(f"  Average random selection CDMean:   {avg_random_cdmean:.4f}")
        
        # Optimized selection (CDMean-optimized)
        print("  Optimized selection:")
        result = train_sel(
            data=ts_data,
            candidates=[candidate_indices.tolist()],
            setsizes=[size],
            settypes=["UOS"],
            stat=cdmean_opt,
            control=control,
            verbose=False
        )

        optimized_indices = result.selected_indices[0]
        optimized_cdmean = result.fitness
        optimized_accuracy = prediction_accuracy_stat(optimized_indices, ts_data)

        optimized_accuracies.append(optimized_accuracy)
        optimized_cdmeans.append(optimized_cdmean)

        print(f"  Optimized selection CDMean:   {optimized_cdmean:.4f}")
        print(f"  Optimized selection accuracy: {optimized_accuracy:.4f}")
        print(f"  Accuracy improvement vs random: {(optimized_accuracy - avg_random_accuracy) / avg_random_accuracy * 100:.2f}%")
        print(f"  CDMean improvement vs random:   {(optimized_cdmean - avg_random_cdmean) / avg_random_cdmean * 100:.2f}%")
    
    # Plot prediction accuracy results
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, random_accuracies, 'o-', label='Random Selection')
    plt.plot(training_sizes, optimized_accuracies, 'o-', label='Optimized Selection')
    plt.title('Prediction Accuracy vs. Training Set Size\n(CDMean-Optimized vs Random)')
    plt.xlabel('Training Set Size')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('genomic_selection_results.png')
    print("\nSaved results plot to 'genomic_selection_results.png'")

    # Plot CDMean results
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, random_cdmeans, 'o-', label='Random Selection')
    plt.plot(training_sizes, optimized_cdmeans, 'o-', label='Optimized Selection')
    plt.title('CDMean vs. Training Set Size')
    plt.xlabel('Training Set Size')
    plt.ylabel('CDMean')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('genomic_selection_cdmean_results.png')
    print("Saved CDMean plot to 'genomic_selection_cdmean_results.png'")
    
    print("\nGenomic selection example completed.")

if __name__ == "__main__":
    main()
