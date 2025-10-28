"""
Continuous Variable Optimization Example for TrainSelPy.

This example demonstrates how to use TrainSelPy for optimizing continuous variables
in a simplified resource allocation problem where we need to determine optimal investment
levels across different projects to maximize returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    train_sel_control
)

def generate_projects(n_projects=10, random_seed=42):
    """
    Generate a set of synthetic projects with different return characteristics.
    
    Each project has:
    - Base return rate
    - Diminishing returns factor (how quickly returns diminish with investment)
    - Risk factor (variability in returns)
    - Maximum investment capacity
    """
    np.random.seed(random_seed)
    
    projects = {}
    
    # Generate project characteristics
    projects['base_return'] = np.random.uniform(0.05, 0.20, n_projects)  # 5-20% base return
    projects['diminishing_factor'] = np.random.uniform(0.3, 0.9, n_projects)  # Higher = faster diminishing
    projects['risk_factor'] = np.random.uniform(0.01, 0.10, n_projects)  # 1-10% risk
    projects['max_capacity'] = np.random.uniform(0.5, 1.0, n_projects)  # Maximum investment proportion
    
    # Project names
    projects['names'] = [f"Project_{i+1}" for i in range(n_projects)]
    
    # Create a feature matrix for the projects
    features = np.column_stack([
        projects['base_return'],
        projects['diminishing_factor'],
        projects['risk_factor'],
        projects['max_capacity']
    ])
    
    return projects, features

def calculate_returns(investment_levels, projects):
    """
    Calculate the expected return and risk for given investment levels.
    
    Parameters
    ----------
    investment_levels : list
        List of investment proportions for each project (0.0 to 1.0)
    projects : dict
        Dictionary with project characteristics
        
    Returns
    -------
    float
        Expected total return
    """
    n_projects = len(investment_levels)
    
    # Calculate return for each project with diminishing returns
    project_returns = []
    for i in range(n_projects):
        # Apply diminishing returns formula
        level = min(investment_levels[i], projects['max_capacity'][i])
        base = projects['base_return'][i]
        diminish = projects['diminishing_factor'][i]
        
        # Returns follow a curve where they diminish as investment increases
        # R = base_return * investment_level^(diminishing_factor)
        project_return = base * (level ** diminish)
        project_returns.append(project_return)
    
    # Calculate weighted total return
    total_return = sum(project_returns)
    
    return total_return

def investment_fitness(dbl_solution, data):
    """
    Calculate fitness for a particular investment allocation.
    
    Parameters
    ----------
    dbl_solution : list
        List of investment levels for each project (continuous values)
    data : dict
        Data dictionary with project information
    
    Returns
    -------
    float
        Fitness value (expected return, higher is better)
    """
    # Extract project information
    projects = data['projects']
    
    # Ensure investment levels sum to 1.0 (normalize if needed)
    investment_levels = np.array(dbl_solution)
    investment_sum = np.sum(investment_levels)
    
    if investment_sum > 0:
        normalized_levels = investment_levels / investment_sum
    else:
        normalized_levels = np.zeros_like(investment_levels)
    
    # Calculate expected return
    expected_return = calculate_returns(normalized_levels, projects)
    
    return expected_return

def main():
    """Run a continuous variable optimization example using TrainSelPy."""
    print("TrainSelPy Continuous Variable Optimization Example")
    print("--------------------------------------------------")
    
    # Generate synthetic projects
    n_projects = 8
    print(f"\nGenerating {n_projects} synthetic projects...")
    projects, project_features = generate_projects(n_projects=n_projects)
    
    # Display project information
    print("\nProject Information:")
    for i in range(n_projects):
        print(f"Project {i+1}: Base Return={projects['base_return'][i]:.2f}, " +
              f"Diminishing Factor={projects['diminishing_factor'][i]:.2f}, " +
              f"Risk={projects['risk_factor'][i]:.2f}, " +
              f"Max Capacity={projects['max_capacity'][i]:.2f}")
    
    # Create TrainSel data object
    print("\nSetting up optimization...")
    ts_data = make_data(M=project_features)
    ts_data['projects'] = projects
    
    # Set control parameters
    control = train_sel_control(
        size="free",
        niterations=150,     # Number of iterations
        minitbefstop=30,     # Stop if no improvement for this many iterations
        npop=100,            # Population size
        mutprob=0.1,         # Mutation probability
        mutintensity=0.2,    # Mutation intensity
        crossprob=0.7,       # Crossover probability
        crossintensity=0.5,  # Crossover intensity
        progress=True        # Show progress
    )
    
    # Run the continuous variable optimization
    print("\nRunning optimization to find optimal investment levels...")
    start_time = time.time()
    
    result = train_sel(
        data=ts_data,
        candidates=[
            [0.125] * n_projects  # Initial equal investment across all projects
        ],
        setsizes=[
            n_projects            # One investment level per project
        ],
        settypes=[
            "DBL"                 # DBL = Double (continuous variables)
        ],
        stat=investment_fitness,  # Custom fitness function
        control=control,
        verbose=True
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract the optimized investment levels
    investment_levels = np.array(result.selected_values[0])
    investment_sum = np.sum(investment_levels)
    normalized_levels = investment_levels / investment_sum if investment_sum > 0 else investment_levels
    
    # Calculate expected return with optimal allocation
    optimal_return = calculate_returns(normalized_levels, projects)
    
    # Calculate return with equal allocation (baseline)
    equal_allocation = np.ones(n_projects) / n_projects
    baseline_return = calculate_returns(equal_allocation, projects)
    
    improvement = ((optimal_return / baseline_return) - 1) * 100
    
    print("\nResults:")
    print(f"Optimal total return: {optimal_return:.4f}")
    print(f"Baseline return (equal allocation): {baseline_return:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    print("\nOptimal investment allocation:")
    for i in range(n_projects):
        print(f"Project {i+1}: {normalized_levels[i]:.4f} ({normalized_levels[i]*100:.1f}%)")
    
    # Visualize the results
    plt.figure(figsize=(12, 10))
    
    # 1. Optimal investment allocation
    plt.subplot(2, 1, 1)
    plt.bar(projects['names'], normalized_levels, color='blue', alpha=0.7)
    plt.title('Optimal Investment Allocation')
    plt.xlabel('Project')
    plt.ylabel('Allocation Proportion')
    plt.ylim(0, max(normalized_levels) * 1.1)
    
    # Add percentage labels on bars
    for i, v in enumerate(normalized_levels):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')
    
    # 2. Project characteristics comparison
    plt.subplot(2, 1, 2)
    
    # Create a dataframe for better visualization
    df = pd.DataFrame({
        'Project': projects['names'],
        'Base Return': projects['base_return'],
        'Diminishing Factor': projects['diminishing_factor'],
        'Risk Factor': projects['risk_factor'],
        'Max Capacity': projects['max_capacity'],
        'Allocation': normalized_levels
    })
    
    # Sort by allocation for better visualization
    df = df.sort_values('Allocation', ascending=False)
    
    # Plot project characteristics with allocation overlay
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Bar chart for characteristics
    characteristics = df[['Base Return', 'Diminishing Factor', 'Risk Factor', 'Max Capacity']].values.T
    x = np.arange(len(df))
    width = 0.2
    
    for i, label in enumerate(['Base Return', 'Diminishing Factor', 'Risk Factor', 'Max Capacity']):
        ax1.bar(x + (i-1.5)*width, characteristics[i], width, label=label, alpha=0.7)
    
    # Line chart for allocation
    ax2.plot(x, df['Allocation'].values, 'r-', marker='o', linewidth=2, label='Allocation')
    
    # Add labels and legend
    ax1.set_xlabel('Project')
    ax1.set_ylabel('Project Characteristics')
    ax2.set_ylabel('Allocation Proportion', color='r')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Project'], rotation=45)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title('Project Characteristics vs. Optimal Allocation')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('continuous_optimization.png')
    print("\nSaved visualization to 'continuous_optimization.png'")
    
    print("\nContinuous variable optimization example completed.")

if __name__ == "__main__":
    main()