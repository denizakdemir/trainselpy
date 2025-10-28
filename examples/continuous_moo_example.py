"""
Multi-Objective Continuous Variable Optimization Example for TrainSelPy.

This example demonstrates multi-objective optimization of continuous variables
in a portfolio allocation problem, balancing multiple objectives:
1. Maximizing expected return
2. Minimizing risk (volatility)
3. Maximizing diversification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
from typing import List, Dict, Any

# Import TrainSelPy functions
from trainselpy import (
    make_data,
    train_sel,
    train_sel_control
)

def generate_assets(n_assets=10, random_seed=42):
    """
    Generate a set of synthetic assets with different return and risk characteristics.
    
    Parameters
    ----------
    n_assets : int
        Number of assets to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with asset information
    numpy.ndarray
        Feature matrix for the assets
    """
    np.random.seed(random_seed)
    
    assets = {}
    
    # Generate asset characteristics
    assets['expected_return'] = np.random.uniform(0.03, 0.18, n_assets)  # 3-18% return
    assets['volatility'] = np.random.uniform(0.05, 0.35, n_assets)       # 5-35% volatility
    
    # Assets with higher returns typically have higher volatility (but with noise)
    for i in range(n_assets):
        assets['volatility'][i] = 0.6 * assets['expected_return'][i] + 0.1 * assets['volatility'][i]
    
    # Generate asset correlation matrix (higher correlation between similar assets)
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Correlation depends on how similar their return/risk profiles are
            similarity = 1.0 - min(1.0, abs(assets['expected_return'][i] - assets['expected_return'][j]) * 5)
            correlation[i, j] = 0.2 + 0.6 * similarity + 0.1 * np.random.random()
            correlation[j, i] = correlation[i, j]  # Symmetric matrix
    
    assets['correlation'] = correlation
    
    # Asset names and sectors
    sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Energy', 'Industrial']
    assets['names'] = [f"Asset_{i+1}" for i in range(n_assets)]
    assets['sectors'] = [random.choice(sectors) for _ in range(n_assets)]
    
    # Create feature matrix for the assets (for visualization)
    features = np.column_stack([
        assets['expected_return'],
        assets['volatility']
    ])
    
    return assets, features

def portfolio_metrics(weights, assets):
    """
    Calculate portfolio metrics (return, risk, diversification).
    
    Parameters
    ----------
    weights : ndarray
        Portfolio weights
    assets : dict
        Asset data
        
    Returns
    -------
    tuple
        (expected_return, portfolio_risk, diversification, sector_diversity)
    """
    # Calculate expected portfolio return
    expected_return = np.sum(weights * assets['expected_return'])
    
    # Calculate portfolio risk (volatility)
    correlation = assets['correlation']
    volatilities = assets['volatility']
    
    # Construct covariance matrix from volatilities and correlation
    n_assets = len(volatilities)
    covariance = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            covariance[i, j] = volatilities[i] * volatilities[j] * correlation[i, j]
    
    portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # Calculate diversification (higher when weights are more evenly distributed)
    # Using entropy-like measure
    non_zero_weights = weights[weights > 1e-10]
    if len(non_zero_weights) > 0:
        # Entropy calculation as a measure of diversification
        entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
        # Normalize to 0-1 range (max entropy is log(n))
        max_entropy = np.log(len(weights))
        diversification = entropy / max_entropy if max_entropy > 0 else 0
    else:
        diversification = 0
    
    # Calculate sector concentration
    sectors = assets['sectors']
    sector_weights = {}
    for i, weight in enumerate(weights):
        sector = sectors[i]
        if sector not in sector_weights:
            sector_weights[sector] = 0
        sector_weights[sector] += weight
    
    # Calculate Herfindahl-Hirschman Index (HHI) for sector concentration
    hhi = sum(w**2 for w in sector_weights.values())
    sector_diversity = 1 - hhi  # Transform so higher is better
    
    return expected_return, portfolio_risk, diversification, sector_diversity

def portfolio_objectives(dbl_solution, data):
    """
    Calculate multiple objectives for portfolio allocation.
    
    Creates strongly conflicting objectives to produce a diverse Pareto front:
    1. Return (maximize) - Focus on high returns, ignoring risk
    2. Risk (minimize) - Focus on low risk, ignoring returns
    3. Diversification (maximize) - Focus on diversification across assets and sectors
    
    Parameters
    ----------
    dbl_solution : list
        List of investment weights for each asset (continuous values)
    data : dict
        Data dictionary with asset information
        
    Returns
    -------
    list
        List of objective values [return_obj, risk_obj, diversity_obj]
    """
    # Extract asset information
    assets = data['assets']
    
    # Ensure weights sum to 1.0 (normalize)
    weights = np.array(dbl_solution)
    weight_sum = np.sum(weights)
    
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones_like(weights) / len(weights)
    
    # Calculate portfolio metrics
    expected_return, portfolio_risk, diversification, sector_diversity = portfolio_metrics(weights, assets)
    
    # Create objective functions with strong trade-offs
    # These objectives are intentionally designed to conflict with each other
    # to generate a diverse Pareto front
    
    # 1. Return objective: Maximize return
    # This objective strongly favors high-return assets regardless of risk
    return_objective = expected_return * 10.0  # Scale for better comparison
    
    # 2. Risk objective: Minimize risk (as negative to maximize)
    # This objective strongly favors low-risk assets regardless of return
    risk_objective = -portfolio_risk * 10.0  # Negative because we want to maximize
    
    # 3. Diversification objective: Maximize diversification
    # This objective favors portfolios with many assets and sector diversification
    # regardless of return or risk
    diversity_objective = (diversification + sector_diversity) / 2.0
    
    return [return_objective, risk_objective, diversity_objective]

def main():
    """Run a multi-objective continuous variable optimization example using TrainSelPy."""
    print("TrainSelPy Multi-Objective Continuous Optimization Example")
    print("--------------------------------------------------------")
    
    # Generate synthetic assets
    n_assets = 12
    print(f"\nGenerating {n_assets} synthetic assets...")
    assets, asset_features = generate_assets(n_assets=n_assets)
    
    # Display asset information
    print("\nAsset Information:")
    for i in range(n_assets):
        print(f"Asset {i+1} ({assets['sectors'][i]}): Expected Return={assets['expected_return'][i]:.2f}, " +
              f"Volatility={assets['volatility'][i]:.2f}")
    
    # Create TrainSel data object
    print("\nSetting up multi-objective optimization...")
    ts_data = make_data(M=asset_features)
    ts_data['assets'] = assets
    
    # Set control parameters for multi-objective optimization
    # We'll use a single-island approach with higher population diversity
    control = train_sel_control(
        size="free",
        niterations=100,       # Number of iterations
        minitbefstop=30,       # Stop if no improvement for this many iterations
        nEliteSaved=30,        # Save more elite solutions for a diverse Pareto front
        nelite=80,             # Keep more elite solutions for selection
        npop=200,              # Larger population for more diversity
        mutprob=0.3,           # Higher mutation probability
        mutintensity=0.4,      # Higher mutation intensity  
        crossprob=0.8,         # High crossover probability
        crossintensity=0.7,    # High crossover intensity
        progress=True,         # Show progress
        solution_diversity=True,  # Ensure diverse solutions on Pareto front
        dynamicNelite=True     # Dynamically adjust elite population
    )
    
    # Run the multi-objective optimization
    print("\nRunning multi-objective optimization to find optimal portfolio allocations...")
    start_time = time.time()
    
    # Create specialized initial solutions to help the algorithm explore the solution space
    # These solutions target different objectives to ensure a diverse Pareto front
    
    # 1. Create initial high-return biased solution (higher weights on high-return assets)
    high_return_init = [0.1] * n_assets
    return_order = np.argsort(-assets['expected_return'])
    for i, idx in enumerate(return_order[:3]):  # Top 3 highest-return assets
        high_return_init[idx] = 0.2 + (3-i)*0.1  # Higher weights for higher return assets
    
    # 2. Create initial low-risk biased solution (higher weights on low-risk assets)
    low_risk_init = [0.1] * n_assets
    risk_order = np.argsort(assets['volatility'])
    for i, idx in enumerate(risk_order[:3]):  # Top 3 lowest-risk assets
        low_risk_init[idx] = 0.2 + (3-i)*0.1  # Higher weights for lower risk assets
    
    # 3. Create initial diversification-biased solution (equal weights)
    diversification_init = [1.0/n_assets] * n_assets
    
    # Create 10 more random solutions for diversity
    random_inits = []
    for _ in range(10):
        rand_weights = np.random.random(n_assets)
        rand_weights = rand_weights / np.sum(rand_weights)  # Normalize
        random_inits.append(rand_weights.tolist())
    
    # Combine all initial solutions to form our initial population
    initial_population = [high_return_init, low_risk_init, diversification_init] + random_inits
    
    # Run the optimization with multiple initial solutions
    result = train_sel(
        data=ts_data,
        candidates=[
            list(range(n_assets))  # Full range of assets to choose from
        ],
        setsizes=[
            n_assets               # One weight per asset
        ],
        settypes=[
            "DBL"                  # DBL = Double (continuous variables)
        ],
        init_sol={"values": [initial_population]},  # Provide our diverse initial solutions
        stat=portfolio_objectives,  # Multi-objective fitness function
        n_stat=3,                   # Number of objectives - VERY IMPORTANT!
        control=control,
        verbose=True,
        solution_diversity=True     # Ensure diverse solutions on Pareto front
    )
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract the Pareto front
    if result.pareto_front:
        print(f"Found {len(result.pareto_front)} solutions on the Pareto front")
        
        # Extract and display a few diverse solutions
        print("\nSample portfolio solutions from the Pareto front:")
        
        # Choose a few representative solutions to show
        n_display = min(5, len(result.pareto_solutions))
        indices_to_display = np.linspace(0, len(result.pareto_solutions)-1, n_display, dtype=int)
        
        for idx_num, i in enumerate(indices_to_display):
            solution = result.pareto_solutions[i]
            weights = np.array(solution['selected_values'][0])
            weight_sum = np.sum(weights)
            normalized_weights = weights / weight_sum if weight_sum > 0 else weights
            
            # Calculate raw metrics for display using our portfolio_metrics function
            expected_return, portfolio_risk, diversification, sector_diversity = portfolio_metrics(normalized_weights, assets)
            
            # Get the actual optimization objectives
            obj_values = portfolio_objectives(normalized_weights, ts_data)
            
            print(f"\nPortfolio {idx_num+1}:")
            print(f"  Expected Return: {expected_return:.4f} ({expected_return*100:.2f}%)")
            print(f"  Risk (Volatility): {portfolio_risk:.4f} ({portfolio_risk*100:.2f}%)")
            print(f"  Asset Diversification: {diversification:.4f}")
            print(f"  Sector Diversification: {sector_diversity:.4f}")
            print(f"  Optimization Objectives: [Return={obj_values[0]:.2f}, Risk={obj_values[1]:.2f}, Diversity={obj_values[2]:.2f}]")
            
            # Show top allocations for this portfolio
            top_indices = np.argsort(-normalized_weights)[:5]  # Top 5 allocations
            print("  Top allocations:")
            for j in top_indices:
                if normalized_weights[j] > 0.01:  # Only show allocations > 1%
                    print(f"    Asset {j+1} ({assets['sectors'][j]}): {normalized_weights[j]:.4f} ({normalized_weights[j]*100:.1f}%)")
        
        # Visualize the Pareto front (3D plot for 3 objectives)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract the objective values
        x = [sol[0] for sol in result.pareto_front]  # Expected return objective
        y = [sol[1] for sol in result.pareto_front]  # Risk objective (already negative)
        z = [sol[2] for sol in result.pareto_front]  # Diversification objective
        
        # Plot the Pareto front
        scatter = ax.scatter(x, y, z, c=y, cmap='viridis', marker='o', s=50, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Return Objective')
        ax.set_ylabel('Risk Objective')
        ax.set_zlabel('Diversification Objective')
        ax.set_title('Pareto Front for Portfolio Optimization')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Risk Objective')
        
        # Add grid
        ax.grid(True)
        
        # Save the figure
        plt.savefig('portfolio_pareto_front_3d.png')
        print("\nSaved 3D Pareto front plot to 'portfolio_pareto_front_3d.png'")
        
        # 2D projections for easier interpretation
        plt.figure(figsize=(15, 5))
        
        # Return vs Risk
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Diversification Objective')
        plt.xlabel('Return Objective')
        plt.ylabel('Risk Objective')
        plt.title('Return vs. Risk')
        plt.grid(True)
        
        # Return vs Diversification
        plt.subplot(1, 3, 2)
        plt.scatter(x, z, c=y, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Risk Objective')
        plt.xlabel('Return Objective')
        plt.ylabel('Diversification Objective')
        plt.title('Return vs. Diversification')
        plt.grid(True)
        
        # Risk vs Diversification
        plt.subplot(1, 3, 3)
        plt.scatter(y, z, c=x, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Return Objective')
        plt.xlabel('Risk Objective')
        plt.ylabel('Diversification Objective')
        plt.title('Risk vs. Diversification')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('portfolio_pareto_front_2d.png')
        print("Saved 2D Pareto front projections to 'portfolio_pareto_front_2d.png'")
        
        # Visualize asset allocations for selected portfolio solutions
        # Pick a few representative solutions (high return, low risk, balanced)
        solution_indices = []
        
        # Find highest return solution
        max_return_idx = np.argmax([sol[0] for sol in result.pareto_front])
        solution_indices.append(max_return_idx)
        
        # Find lowest risk solution
        min_risk_idx = np.argmin([-sol[1] for sol in result.pareto_front])
        solution_indices.append(min_risk_idx)
        
        # Find most diversified solution
        max_div_idx = np.argmax([sol[2] for sol in result.pareto_front])
        solution_indices.append(max_div_idx)
        
        # Find balanced solution (closest to "ideal" point)
        # Normalize objectives to 0-1 range
        normalized_x = (np.array(x) - min(x)) / (max(x) - min(x)) if max(x) > min(x) else np.array(x)
        normalized_y = (np.array(y) - min(y)) / (max(y) - min(y)) if max(y) > min(y) else np.array(y)
        normalized_z = (np.array(z) - min(z)) / (max(z) - min(z)) if max(z) > min(z) else np.array(z)
        
        # Ideal point is (1,0,1) - high return, low risk, high diversification
        ideal_point = np.array([1, 0, 1])
        distances = []
        for i in range(len(result.pareto_front)):
            point = np.array([normalized_x[i], 1 - normalized_y[i], normalized_z[i]])
            distances.append(np.linalg.norm(point - ideal_point))
        
        balanced_idx = np.argmin(distances)
        solution_indices.append(balanced_idx)
        
        # Make solution indices unique
        solution_indices = list(set(solution_indices))
        
        # Visualize allocations for selected portfolios
        plt.figure(figsize=(15, 10))
        
        portfolio_names = []
        for i, sol_idx in enumerate(solution_indices):
            solution = result.pareto_solutions[sol_idx]
            weights = np.array(solution['selected_values'][0])
            weight_sum = np.sum(weights)
            normalized_weights = weights / weight_sum if weight_sum > 0 else weights
            
            # Calculate objectives
            obj_values = portfolio_objectives(normalized_weights, ts_data)
            
            # Determine portfolio type based on objectives
            if sol_idx == max_return_idx:
                portfolio_type = "High Return"
            elif sol_idx == min_risk_idx:
                portfolio_type = "Low Risk"
            elif sol_idx == max_div_idx:
                portfolio_type = "High Diversification"
            elif sol_idx == balanced_idx:
                portfolio_type = "Balanced"
            else:
                portfolio_type = f"Portfolio {i+1}"
            
            portfolio_names.append(portfolio_type)
            
            # Sort assets by allocation for this portfolio
            sorted_indices = np.argsort(-normalized_weights)
            sorted_assets = [assets['names'][j] for j in sorted_indices]
            sorted_weights = normalized_weights[sorted_indices]
            
            # Plot allocations
            plt.subplot(2, 2, i+1)
            bars = plt.bar(np.arange(n_assets), sorted_weights, alpha=0.7)
            
            # Color bars by sector
            sector_colors = {'Technology': 'blue', 'Financial': 'green', 'Healthcare': 'red', 
                            'Consumer': 'purple', 'Energy': 'orange', 'Industrial': 'brown'}
            
            for j, bar in enumerate(bars):
                asset_idx = sorted_indices[j]
                sector = assets['sectors'][asset_idx]
                bar.set_color(sector_colors.get(sector, 'gray'))
            
            plt.title(f"{portfolio_type} Portfolio\nReturn: {obj_values[0]*100:.2f}%, " + 
                     f"Risk: {-obj_values[1]*100:.2f}%, Div: {obj_values[2]:.2f}")
            plt.xlabel('Asset Rank')
            plt.ylabel('Allocation')
            plt.xticks([])  # Hide x-axis labels for cleaner display
            
            # Add percentage labels for top allocations
            for j in range(min(5, n_assets)):
                if sorted_weights[j] > 0.05:  # Only label allocations > 5%
                    plt.text(j, sorted_weights[j] + 0.01, 
                            f"{sorted_weights[j]*100:.1f}%\n{assets['names'][sorted_indices[j]]}", 
                            ha='center', fontsize=8)
        
        # Add a legend for sectors
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in sector_colors.values()]
        labels = list(sector_colors.keys())
        plt.figlegend(handles, labels, loc='lower center', ncol=len(sector_colors), 
                    bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for the legend
        plt.savefig('portfolio_allocations.png')
        print("Saved portfolio allocations visualization to 'portfolio_allocations.png'")
        
        # Create a comparison of sectoral allocations
        plt.figure(figsize=(12, 8))
        
        # Create a dictionary to store sector allocations for each portfolio
        sector_allocations = {name: {sector: 0 for sector in set(assets['sectors'])} 
                             for name in portfolio_names}
        
        # Calculate sector allocations for each portfolio
        for i, sol_idx in enumerate(solution_indices):
            solution = result.pareto_solutions[sol_idx]
            weights = np.array(solution['selected_values'][0])
            weight_sum = np.sum(weights)
            normalized_weights = weights / weight_sum if weight_sum > 0 else weights
            
            for j, weight in enumerate(normalized_weights):
                sector = assets['sectors'][j]
                sector_allocations[portfolio_names[i]][sector] += weight
        
        # Convert to dataframe for easier plotting
        sector_df = pd.DataFrame(sector_allocations)
        
        # Plot sectoral allocations
        ax = sector_df.plot(kind='bar', figsize=(12, 8), width=0.7)
        
        plt.title('Sector Allocations by Portfolio Type')
        plt.xlabel('Sector')
        plt.ylabel('Allocation')
        plt.legend(title='Portfolio Type')
        plt.grid(axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, 
                        label_type='edge', fontsize=8, 
                        rotation=90)
        
        plt.tight_layout()
        plt.savefig('portfolio_sector_allocations.png')
        print("Saved sector allocations visualization to 'portfolio_sector_allocations.png'")
        
    else:
        print("No Pareto front found. The optimization may not have run in true multi-objective mode.")
    
    print("\nMulti-objective continuous optimization example completed.")

if __name__ == "__main__":
    main()