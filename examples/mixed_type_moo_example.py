"""
Example demonstrating multivariate mixed type multi-objective optimization with TrainSelPy.

This example shows how to optimize multiple variables of different types simultaneously
for drug discovery experiment design, focusing on two conflicting objectives:
1. Maximizing information gain (efficiency)
2. Minimizing total cost

It combines:
1. Selecting compounds to test (UOS - Unordered Set)
2. Determining optimal dosage levels (DBL - Continuous variables)
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
    set_control_default
)


def generate_synthetic_data(n_compounds=50, n_features=5, n_targets=2, random_state=42):
    """Generate synthetic compound data with features and target activities."""
    np.random.seed(random_state)
    
    # Generate molecular features
    compound_features = np.random.normal(0, 1, size=(n_compounds, n_features))
    
    # Generate target activities based on features (with some noise)
    target_activities = np.zeros((n_compounds, n_targets))
    
    for i in range(n_targets):
        weights = np.random.normal(0, 1, size=n_features)
        target_activities[:, i] = np.dot(compound_features, weights) + np.random.normal(0, 0.5, size=n_compounds)
        # Normalize to 0-1 range
        target_activities[:, i] = (target_activities[:, i] - np.min(target_activities[:, i])) / \
                                 (np.max(target_activities[:, i]) - np.min(target_activities[:, i]))
    
    # Generate compound costs
    complexity = np.sum(np.abs(compound_features), axis=1)
    compound_costs = 100 + 50 * (complexity - np.min(complexity)) / (np.max(complexity) - np.min(complexity))
    
    return compound_features, target_activities, compound_costs


def dose_response_curve(dose, ec50=0.5, hill_slope=2.0, max_response=1.0):
    """Calculate the response from a dose using a Hill equation."""
    return max_response * (dose ** hill_slope) / (ec50 ** hill_slope + dose ** hill_slope)

def compound_specific_response(compound_idx, dose, compound_features):
    """Calculate compound-specific response based on molecular features."""
    # Create compound-specific parameters based on features
    feature_sum = np.sum(np.abs(compound_features[compound_idx]))
    
    # Adjust EC50 based on compound features - some compounds need higher/lower doses to be effective
    compound_ec50 = 0.3 + 0.4 * (feature_sum / 5)  # Range approximately 0.3-0.7
    
    # Adjust hill slope based on different feature aspects
    compound_hill_slope = 1.0 + 2.0 * np.abs(compound_features[compound_idx, 0])  # Range approximately 1.0-3.0
    
    # Adjust max response based on another feature aspect
    compound_max_response = 0.7 + 0.3 * np.abs(compound_features[compound_idx, 1])  # Range 0.7-1.0
    
    return dose_response_curve(dose, ec50=compound_ec50, hill_slope=compound_hill_slope, max_response=compound_max_response)


def moo_fitness_function(int_solution, dbl_solution, data):
    """
    Multi-objective fitness function for mixed type optimization.
    
    Parameters:
    - int_solution: List of selected compound indices
    - dbl_solution: List of dosage levels for each compound
    - data: Dictionary with compound information
    
    Returns:
    - List of fitness values [Information Gain, -Cost] (both to be maximized)
    """
    selected_compounds = int_solution if isinstance(int_solution, list) else [int_solution]
    dosage_levels = dbl_solution
    
    # Extract data
    target_activities = data["TargetActivities"]
    compound_costs = data["CompoundCosts"]
    compound_features = data["G"].values  # Get the feature matrix
    
    # Calculate cost (Objective 2: Minimize Cost -> Maximize -Cost)
    total_cost = np.sum(compound_costs[selected_compounds])
    
    # Calculate information gain (Objective 1: Maximize Information Gain)
    selected_activities = target_activities[selected_compounds, :]
    
    # Apply compound-specific dose-response effects
    dose_effects = np.array([
        compound_specific_response(compound_idx, dose, compound_features) 
        for compound_idx, dose in zip(selected_compounds, dosage_levels)
    ])
    
    dose_adjusted_activities = selected_activities * dose_effects[:, np.newaxis]
    
    # Calculate diversity of responses
    activity_diversity = np.sum(np.std(dose_adjusted_activities, axis=0))
    
    # Calculate coverage of activity space
    coverage = np.mean(np.max(dose_adjusted_activities, axis=0))
    
    # Calculate dose optimization score with stronger penalty for suboptimal doses
    dose_optimization_score = 0
    for i, compound_idx in enumerate(selected_compounds):
        # Calculate "optimal" dose for this compound based on its features
        feature_sum = np.sum(np.abs(compound_features[compound_idx]))
        optimal_dose = 0.3 + 0.6 * (feature_sum / 5)  # Optimal dose varies by compound
        
        # Add score based on how close the selected dose is to the optimal dose
        # Using a quadratic penalty for distance from optimal
        distance = abs(dosage_levels[i] - optimal_dose)
        dose_optimization_score += 1.0 - min(distance * 4, 1.0)**2
    
    dose_optimization_score /= len(selected_compounds)  # Normalize
    
    # Information gain combines diversity, coverage, and dose optimality
    information_gain = activity_diversity * coverage * len(selected_compounds) * (1.0 + dose_optimization_score)
    
    # Return both objectives
    return [information_gain, -total_cost]


def main():
    """Run a mixed type multi-objective optimization example using TrainSelPy."""
    print("TrainSelPy Mixed Type Multi-Objective Optimization Example")
    print("----------------------------------------------------------")
    
    # Generate synthetic data
    n_compounds = 50
    n_features = 5
    n_targets = 2
    
    print(f"Generating {n_compounds} compounds with {n_features} features and {n_targets} targets...")
    compound_features, target_activities, compound_costs = generate_synthetic_data(
        n_compounds=n_compounds,
        n_features=n_features,
        n_targets=n_targets
    )
    
    # Create the TrainSel data object
    ts_data = make_data(M=compound_features)
    ts_data["TargetActivities"] = target_activities
    ts_data["CompoundCosts"] = compound_costs
    
    # Set control parameters
    control = set_control_default()
    control["niterations"] = 300  # Increased for better convergence and front filling
    control["npop"] = 300         # Increased population size for better exploration
    control["nEliteSaved"] = 50   # Save more elite solutions to maintain the front
    control["nelite"] = 100       # Carry over more elites
    control["mutprob"] = 0.2      # Higher mutation probability for exploration
    control["mutintensity"] = 0.3 # Higher mutation intensity
    control["niterSANN"] = 50     # Simulated annealing iterations
    control["solution_diversity"] = True # Ensure diverse solutions on Pareto front
    control["use_nsga3"] = True   # Use NSGA-III for better diversity
    
    # Define the optimization problem
    n_compounds_to_select = 5
    
    print("Running multi-objective optimization (Maximize Info Gain vs. Minimize Cost)...")
    start_time = time.time()
    
    # Run multi-objective optimization
    result = train_sel(
        data=ts_data,
        candidates=[
            list(range(n_compounds)),  # Candidate compounds (UOS)
            [0.5] * n_compounds_to_select  # Initial dosage values (DBL)
        ],
        setsizes=[
            n_compounds_to_select,  # Select 5 compounds
            n_compounds_to_select   # One dosage per selected compound
        ],
        settypes=[
            "OS",  # Unordered set for compounds
            "DBL"   # Continuous values for dosages (0-1)
        ],
        stat=moo_fitness_function,
        n_stat=2,  # Two objectives
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"Optimization completed in {runtime:.2f} seconds")
    
    # Check if Pareto front exists
    if result.pareto_front:
        print(f"\nFound {len(result.pareto_front)} solutions on the Pareto front.")
        
        # Extract objectives for plotting
        info_gains = [sol[0] for sol in result.pareto_front]
        costs = [-sol[1] for sol in result.pareto_front]  # Convert back to positive cost
        
        # Sort by cost for better visualization
        sorted_indices = np.argsort(costs)
        info_gains = np.array(info_gains)[sorted_indices]
        costs = np.array(costs)[sorted_indices]
        
        # Print some solutions
        print("\nSample Pareto Solutions:")
        print(f"{'Solution':<10} | {'Info Gain':<15} | {'Cost':<10} | {'Efficiency':<10}")
        print("-" * 55)
        
        # Pick a few representative solutions (min cost, max info, and some in between)
        indices_to_show = [0, len(costs)//2, len(costs)-1]
        if len(costs) > 5:
            indices_to_show = np.linspace(0, len(costs)-1, 5, dtype=int)
            
        for i in indices_to_show:
            efficiency = info_gains[i] / costs[i]
            print(f"{i:<10} | {info_gains[i]:<15.4f} | {costs[i]:<10.2f} | {efficiency:<10.4f}")
            
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(costs, info_gains, c=info_gains/costs, cmap='viridis', s=100, edgecolors='k')
        plt.colorbar(label='Efficiency (Info/Cost)')
        plt.plot(costs, info_gains, 'k--', alpha=0.3)
        
        plt.title('Pareto Front: Information Gain vs. Cost')
        plt.xlabel('Total Cost (Minimize)')
        plt.ylabel('Information Gain (Maximize)')
        plt.grid(True, alpha=0.3)
        
        # Annotate extreme points
        plt.annotate('Lowest Cost', xy=(costs[0], info_gains[0]), xytext=(costs[0], info_gains[0]+1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.annotate('Highest Info', xy=(costs[-1], info_gains[-1]), xytext=(costs[-1]-50, info_gains[-1]),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        plt.savefig('pareto_front_mixed_type.png')
        print("\nPareto front visualization saved as 'pareto_front_mixed_type.png'")
        
    else:
        print("\nNo Pareto front found. Optimization might have converged to a single solution.")

if __name__ == "__main__":
    main()
