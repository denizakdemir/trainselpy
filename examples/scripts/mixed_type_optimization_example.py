"""
Example demonstrating multivariate mixed type optimization with TrainSelPy.

This example shows how to optimize multiple variables of different types simultaneously
for drug discovery experiment design, focusing on:
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


def fitness_function(int_solution, dbl_solution, data):
    """
    Fitness function for mixed type optimization.
    
    Parameters:
    - int_solution: List of selected compound indices
    - dbl_solution: List of dosage levels for each compound
    - data: Dictionary with compound information
    
    Returns:
    - Fitness value (higher is better)
    """
    selected_compounds = int_solution if isinstance(int_solution, list) else [int_solution]
    dosage_levels = dbl_solution
    
    # Extract data
    target_activities = data["TargetActivities"]
    compound_costs = data["CompoundCosts"]
    compound_features = data["G"].values  # Get the feature matrix
    
    # Calculate cost
    total_cost = np.sum(compound_costs[selected_compounds])
    
    # Calculate information gain
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
    
    # Check if we're in the second phase (dosage optimization)
    is_second_phase = False
    if hasattr(data, "get") and data.get("phase") == "dosage_optimization":
        is_second_phase = True
    
    # Combine metrics into final information gain with phase-specific weights
    if is_second_phase:
        # In dosage optimization phase, strongly prioritize dose optimization
        information_gain = activity_diversity * coverage * len(selected_compounds) * (1.0 + 5.0 * dose_optimization_score)
    else:
        # In compound selection phase, use balanced priorities
        information_gain = activity_diversity * coverage * len(selected_compounds) * (1.0 + dose_optimization_score)
    
    # Calculate efficiency (information per cost)
    efficiency = information_gain / (total_cost + 1)  # Add 1 to avoid division by zero
    
    return efficiency

# Alternative fitness function that focuses ONLY on dosage optimization
def dose_optimization_fitness(int_solution, dbl_solution, data):
    """
    Fitness function focused exclusively on dosage optimization.
    """
    selected_compounds = int_solution if isinstance(int_solution, list) else [int_solution]
    dosage_levels = dbl_solution
    compound_features = data["G"].values
    
    # Calculate dose optimization score
    dose_optimization_score = 0
    for i, compound_idx in enumerate(selected_compounds):
        # Calculate "optimal" dose for this compound
        feature_sum = np.sum(np.abs(compound_features[compound_idx]))
        optimal_dose = 0.3 + 0.6 * (feature_sum / 5)
        
        # Strongly penalize deviations from optimal dose
        distance = abs(dosage_levels[i] - optimal_dose)
        dose_optimization_score += 1.0 - (distance * 10)**2  # Quadratic penalty
        
    return dose_optimization_score


def main():
    """Run a simplified mixed type optimization example using TrainSelPy."""
    print("TrainSelPy Mixed Type Optimization Example (Simplified)")
    print("-----------------------------------------------------")
    
    # Generate synthetic data
    n_compounds = 50  # Reduced from 100
    n_features = 5    # Reduced from 10
    n_targets = 2     # Reduced from 3
    
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
    control["niterations"] = 100  # Increased to allow convergence
    control["npop"] = 50         # Increased population size
    control["mutprob"] = 0.05    # Increased mutation probability
    control["mutintensity"] = 0.2  # Increased mutation intensity
    control["niterSANN"] = 50    # Increased number of simulated annealing iterations
    
    # Define the optimization problem
    n_compounds_to_select = 5  # Reduced from 10
    
    print("Running optimization...")
    start_time = time.time()
    
    # First run - find the best compounds
    print("Phase 1: Selecting optimal compounds...")
    result_phase1 = train_sel(
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
        stat=fitness_function,
        control=control,
        verbose=True
    )
    
    # Extract selected compounds
    selected_compounds = result_phase1.selected_indices[0]
    
    # Instead of optimization, directly compare default dosages vs. theoretical optimal dosages
    print("\nPhase 2: Comparing default vs. optimal dosages...")
    
    # Calculate theoretical optimal doses
    optimal_doses = []
    for compound_idx in selected_compounds:
        feature_sum = np.sum(np.abs(compound_features[compound_idx]))
        optimal_dose = 0.3 + 0.6 * (feature_sum / 5)
        optimal_doses.append(optimal_dose)
    
    print("Theoretical optimal doses:", [f"{d:.4f}" for d in optimal_doses])
    
    # Evaluate fitness with default doses (0.5)
    default_doses = [0.5] * len(selected_compounds)
    default_fitness = fitness_function(selected_compounds, default_doses, ts_data)
    
    # Evaluate fitness with optimal doses
    optimal_fitness = fitness_function(selected_compounds, optimal_doses, ts_data)
    
    print(f"Fitness with default doses (0.5): {default_fitness:.8f}")
    print(f"Fitness with optimal doses: {optimal_fitness:.8f}")
    print(f"Improvement: {((optimal_fitness/default_fitness)-1)*100:.2f}%")
    
    # Use the optimal doses directly
    result = result_phase1  # Keep the phase 1 result for compound selection
    result.selected_values = [result.selected_values[0], optimal_doses]  # Override with optimal doses
    
    runtime = time.time() - start_time
    print(f"Optimization completed in {runtime:.2f} seconds")
    
    # Extract results
    selected_compounds = result.selected_indices[0]
    dosage_levels = result.selected_values[1] if hasattr(result, 'selected_values') and len(result.selected_values) > 1 else [0.5] * len(selected_compounds)
    
    # Calculate theoretical optimal doses for selected compounds
    optimal_doses = []
    for compound_idx in selected_compounds:
        feature_sum = np.sum(np.abs(compound_features[compound_idx]))
        optimal_dose = 0.3 + 0.6 * (feature_sum / 5)  # Same formula as in fitness function
        optimal_doses.append(optimal_dose)
    
    # Print results
    print(f"\nSelected {len(selected_compounds)} compounds: {selected_compounds}")
    print("\nDosage levels:")
    for i, (compound, dose, optimal) in enumerate(zip(selected_compounds, dosage_levels, optimal_doses)):
        print(f"Compound {compound}: Selected={dose:.4f}, Theoretical Optimal={optimal:.4f}, Difference={abs(dose-optimal):.4f}")
    
    # Calculate total cost
    total_cost = np.sum(compound_costs[selected_compounds])
    print(f"\nTotal cost: ${total_cost:.2f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 10))
    
    # 1. PCA of compound space with selected compounds highlighted
    plt.subplot(3, 1, 1)
    pca = PCA(n_components=2)
    compound_features_scaled = StandardScaler().fit_transform(compound_features)
    compound_pca = pca.fit_transform(compound_features_scaled)
    
    plt.scatter(compound_pca[:, 0], compound_pca[:, 1], c='lightgray', alpha=0.5)
    plt.scatter(
        compound_pca[selected_compounds, 0],
        compound_pca[selected_compounds, 1],
        c='red',
        s=100,
        marker='o',
        label='Selected Compounds'
    )
    plt.title('Selected Compounds in Feature Space (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    
    # 2. Actual dosage levels chosen by optimization
    plt.subplot(3, 1, 2)
    plt.bar(range(len(selected_compounds)), dosage_levels, color='blue', alpha=0.7, label='Selected Dose')
    plt.title('Optimal Dosage Levels Selected by TrainSelPy')
    plt.xlabel('Compound')
    plt.ylabel('Dosage Level')
    plt.xticks(range(len(selected_compounds)), selected_compounds)
    plt.ylim(0, 1)
    
    # 3. Comparison of selected versus theoretical optimal dose
    plt.subplot(3, 1, 3)
    
    # Calculate theoretical optimal doses for selected compounds
    optimal_doses = []
    for compound_idx in selected_compounds:
        feature_sum = np.sum(np.abs(compound_features[compound_idx]))
        optimal_dose = 0.3 + 0.6 * (feature_sum / 5)  # Same formula as in fitness function
        optimal_doses.append(optimal_dose)
    
    x = np.arange(len(selected_compounds))
    width = 0.35
    
    plt.bar(x - width/2, dosage_levels, width, label='Selected Dose', color='blue', alpha=0.7)
    plt.bar(x + width/2, optimal_doses, width, label='Theoretical Optimal', color='green', alpha=0.7)
    
    plt.title('Selected vs. Theoretical Optimal Dosage')
    plt.xlabel('Compound')
    plt.ylabel('Dosage Level')
    plt.xticks(x, selected_compounds)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    # Add a text annotation explaining the two-phase optimization
    plt.figtext(0.5, 0.01, 
                "Two-phase optimization: 1) Select compounds, 2) Optimize dosages",
                ha='center', fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.savefig('mixed_type_optimization.png')
    print("Visualization saved as 'mixed_type_optimization.png'")
    
    # Save optimization results to a CSV file for further analysis
    results_df = pd.DataFrame({
        'Compound': selected_compounds,
        'Optimal_Dose': optimal_doses,
        'Cost': compound_costs[selected_compounds]
    })
    results_df.to_csv('compound_optimization_results.csv', index=False)
    print("Results saved to 'compound_optimization_results.csv'")
    

if __name__ == "__main__":
    main()