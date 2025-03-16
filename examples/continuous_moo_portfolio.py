"""
Multi-Objective Continuous Variable Portfolio Optimization Example for TrainSelPy.

This example demonstrates multi-objective optimization with continuous variables
for a portfolio allocation problem balancing:
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

def generate_asset_data(n_assets=20, random_seed=42):
    """Generate synthetic data for assets."""
    np.random.seed(random_seed)
    
    # Create two clusters of assets with different risk/return profiles
    # Cluster 1: Lower return, lower risk assets
    returns1 = np.random.uniform(0.03, 0.08, n_assets // 2)
    risks1 = returns1 * 0.8 + np.random.uniform(0.01, 0.03, n_assets // 2)
    
    # Cluster 2: Higher return, higher risk assets
    returns2 = np.random.uniform(0.10, 0.20, n_assets - (n_assets // 2))
    risks2 = returns2 * 1.2 + np.random.uniform(0.02, 0.05, n_assets - (n_assets // 2))
    
    # Combine the clusters
    returns = np.concatenate([returns1, returns2])
    risks = np.concatenate([risks1, risks2])
    
    # Add some industry sectors
    sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Energy', 'Industrial']
    asset_sectors = [random.choice(sectors) for _ in range(n_assets)]
    
    # Create feature vectors for each asset
    features = np.column_stack((returns, risks))
    
    # Create asset data
    assets = {
        'names': [f"Asset_{i+1}" for i in range(n_assets)],
        'returns': returns,
        'risks': risks,
        'sectors': asset_sectors
    }
    
    # Create correlation matrix (assets within same sector are more correlated)
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Base correlation
            base_corr = 0.2
            
            # Add correlation if in same sector
            sector_corr = 0.4 if asset_sectors[i] == asset_sectors[j] else 0
            
            # Add correlation based on similarity of risk/return profiles
            profile_diff = np.abs(returns[i] - returns[j]) + np.abs(risks[i] - risks[j])
            profile_corr = max(0, 0.3 * (1 - profile_diff))
            
            # Total correlation
            total_corr = min(0.95, base_corr + sector_corr + profile_corr)
            
            # Apply symmetrically
            corr_matrix[i, j] = total_corr
            corr_matrix[j, i] = total_corr
    
    assets['corr_matrix'] = corr_matrix
    
    return assets, features

def portfolio_metrics(weights, assets):
    """Calculate portfolio metrics (return, risk, diversification)."""
    n_assets = len(weights)
    
    # Expected return
    expected_return = np.sum(weights * assets['returns'])
    
    # Risk (volatility)
    risk_vector = assets['risks']
    corr_matrix = assets['corr_matrix']
    
    # Construct covariance matrix
    cov_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            cov_matrix[i, j] = risk_vector[i] * risk_vector[j] * corr_matrix[i, j]
    
    # Portfolio variance and volatility
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    volatility = np.sqrt(variance)
    
    # Asset diversification (entropy-based measure)
    non_zero_weights = weights[weights > 1e-8]
    if len(non_zero_weights) > 0:
        entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
        max_entropy = np.log(n_assets)
        asset_diversity = entropy / max_entropy if max_entropy > 0 else 0
    else:
        asset_diversity = 0
    
    # Sector diversification
    sectors = assets['sectors']
    sector_weights = {}
    
    for i, weight in enumerate(weights):
        sector = sectors[i]
        if sector not in sector_weights:
            sector_weights[sector] = 0
        sector_weights[sector] += weight
    
    # Herfindahl-Hirschman Index (HHI) for sector concentration
    sector_hhi = sum(w**2 for w in sector_weights.values())
    sector_diversity = 1 - sector_hhi  # Transform to diversity measure
    
    return expected_return, volatility, asset_diversity, sector_diversity

def portfolio_objectives(dbl_solution, data):
    """
    Multi-objective fitness function for portfolio optimization.
    
    Returns three objective values:
    1. Return (to be maximized)
    2. Risk (to be minimized, returned as negative to be maximized)
    3. Diversification (to be maximized)
    """
    assets = data['assets']
    
    # Ensure weights sum to 1.0
    weights = np.array(dbl_solution)
    weight_sum = np.sum(weights)
    
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    # Calculate metrics
    return_val, risk, asset_div, sector_div = portfolio_metrics(weights, assets)
    
    # Scale objectives and ensure they are in very different ranges to force trade-offs
    # This helps create a more diverse Pareto front
    return_obj = return_val * 10 + 0.1  # Scale returns to ~[0.3, 2.1] range
    
    # Scale risk objective based on how much risk we're taking
    # Make the penalty for high risk more severe
    risk_scale = 5.0 if risk > 0.15 else 3.0
    risk_obj = -risk * risk_scale
    
    # Make diversity more important and separate it into two components
    # This makes the algorithm consider asset diversity and sector diversity separately
    asset_div_obj = asset_div * 0.8
    sector_div_obj = sector_div * 1.2
    
    # Average the diversity components
    div_obj = (asset_div_obj + sector_div_obj) / 2
    
    # Add a small random noise to break ties and encourage exploration
    # This can help find more diverse solutions
    noise = np.random.normal(0, 0.01, 3)
    
    return [
        return_obj + noise[0], 
        risk_obj + noise[1], 
        div_obj + noise[2]
    ]

def main():
    """Run a multi-objective continuous variable optimization example for portfolio allocation."""
    
    print("TrainSelPy Multi-Objective Continuous Portfolio Optimization Example")
    print("------------------------------------------------------------------")
    
    # Generate asset data
    n_assets = 20
    print(f"\nGenerating synthetic data for {n_assets} assets...")
    assets, features = generate_asset_data(n_assets=n_assets)
    
    # Display asset information
    print("\nAsset Information:")
    print("Index | Name      | Sector      | Return | Risk")
    print("-" * 50)
    for i in range(n_assets):
        print(f"{i:5} | {assets['names'][i]:10} | {assets['sectors'][i]:11} | {assets['returns'][i]:6.2f} | {assets['risks'][i]:4.2f}")
    
    # Create the TrainSel data object
    print("\nCreating TrainSel data object...")
    ts_data = make_data(M=features)
    ts_data['assets'] = assets
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = train_sel_control(
        size="free",
        niterations=60,       # Reduced for faster runtime
        minitbefstop=30,      # Reduced for faster runtime
        nEliteSaved=30,       # Number of elite solutions to save
        nelite=100,           # Number of elite solutions
        npop=300,             # Population size
        mutprob=0.2,          # Mutation probability
        mutintensity=0.3,     # Mutation intensity
        crossprob=0.8,        # Crossover probability
        crossintensity=0.6,   # Crossover intensity
        niterSANN=15,         # Simulated annealing iterations
        tempini=100.0,        # Initial temperature
        tempfin=0.1,          # Final temperature
        progress=True,        # Show progress
        solution_diversity=False,  # Allow similar solutions on Pareto front
        nislands=4,           # Use 4 islands
        niterIslands=40,      # Number of iterations per island
        minitbefstopIslands=20, # Minimum iterations before stopping on each island
        npopIslands=150,      # Population size per island
        dynamicNelite=True    # Enable dynamic elite size adjustment
    )
    
    # Run the multi-objective optimization
    print("\nRunning multi-objective portfolio optimization...")
    start_time = time.time()
    
    # Create multiple optimization runs with different objectives priorities
    # This approach forces the algorithm to explore different areas of the Pareto front
    print("\nRunning multiple optimizations with different objective priorities...")
    
    all_results = []
    
    # Original balanced run (all objectives equally weighted)
    print("\n1. Running balanced optimization (all objectives equally weighted)...")
    result1 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],  # Choose from all assets
        setsizes=[n_assets],                 # Weights for each asset
        settypes=["DBL"],                    # Continuous variables
        stat=portfolio_objectives,           # Multi-objective fitness function
        n_stat=3,                            # Three objectives
        control=control,
        verbose=False,
        solution_diversity=False,            # Allow similar solutions (will combine later)
        n_jobs=2                             # Use parallel processing
    )
    all_results.append(result1)
    
    # Define a return-biased objective function (for high return portfolios)
    def return_biased_objectives(dbl_solution, data):
        base_objs = portfolio_objectives(dbl_solution, data)
        # Boost return objective significantly
        return [base_objs[0] * 3.0, base_objs[1] * 0.3, base_objs[2] * 0.3]
    
    # Run return-biased optimization
    print("2. Running return-biased optimization...")
    result2 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=return_biased_objectives,       # Return-biased objective function
        n_stat=3,
        control=control,
        verbose=False,
        solution_diversity=False,
        n_jobs=2
    )
    all_results.append(result2)
    
    # Define a risk-biased objective function (for low risk portfolios)
    def risk_biased_objectives(dbl_solution, data):
        base_objs = portfolio_objectives(dbl_solution, data)
        # Boost risk objective significantly
        return [base_objs[0] * 0.3, base_objs[1] * 3.0, base_objs[2] * 0.3]
    
    # Run risk-biased optimization
    print("3. Running risk-biased optimization...")
    result3 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=risk_biased_objectives,         # Risk-biased objective function
        n_stat=3,
        control=control,
        verbose=False,
        solution_diversity=False,
        n_jobs=2
    )
    all_results.append(result3)
    
    # Define a diversity-biased objective function
    def diversity_biased_objectives(dbl_solution, data):
        base_objs = portfolio_objectives(dbl_solution, data)
        # Boost diversity objective significantly
        return [base_objs[0] * 0.3, base_objs[1] * 0.3, base_objs[2] * 3.0]
    
    # Run diversity-biased optimization
    print("4. Running diversity-biased optimization...")
    result4 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=diversity_biased_objectives,    # Diversity-biased objective function
        n_stat=3,
        control=control,
        verbose=False,
        solution_diversity=False,
        n_jobs=2
    )
    all_results.append(result4)
    
    # Add extreme optimizations for more variety
    
    # Extreme return focus
    def extreme_return_objectives(dbl_solution, data):
        base_objs = portfolio_objectives(dbl_solution, data)
        return [base_objs[0] * 5.0, base_objs[1] * 0.1, base_objs[2] * 0.1]
    
    print("5. Running extreme return-focused optimization...")
    result5 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=extreme_return_objectives,
        n_stat=3,
        control=control,
        verbose=False,
        solution_diversity=False,
        n_jobs=2
    )
    all_results.append(result5)
    
    # Return-risk balanced (for efficient frontier)
    def return_risk_balanced(dbl_solution, data):
        base_objs = portfolio_objectives(dbl_solution, data)
        return [base_objs[0] * 2.0, base_objs[1] * 2.0, base_objs[2] * 0.2]
    
    print("6. Running return-risk balanced optimization...")
    result6 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=return_risk_balanced,
        n_stat=3,
        control=control,
        verbose=False,
        solution_diversity=False,
        n_jobs=2
    )
    all_results.append(result6)
    
    # Combine results and find the overall Pareto front
    print("\nCombining results to find the overall Pareto front...")
    
    # Combine all pareto solutions
    combined_solutions = []
    for result in all_results:
        if result.pareto_solutions:
            combined_solutions.extend(result.pareto_solutions)
    
    # If we have solutions, find the overall Pareto front
    if combined_solutions:
        # Extract multi-objective fitness values
        fitness_values = [sol["multi_fitness"] for sol in combined_solutions]
        
        # Find non-dominated solutions
        is_dominated = [False] * len(combined_solutions)
        for i in range(len(combined_solutions)):
            for j in range(len(combined_solutions)):
                if i != j:
                    fi = fitness_values[i]
                    fj = fitness_values[j]
                    if all(fj[k] >= fi[k] for k in range(3)) and any(fj[k] > fi[k] for k in range(3)):
                        is_dominated[i] = True
                        break
        
        # Filter non-dominated solutions
        pareto_solutions = [sol for i, sol in enumerate(combined_solutions) if not is_dominated[i]]
        pareto_front = [sol["multi_fitness"] for sol in pareto_solutions]
        
        # Update the first result with the combined Pareto front
        result1.pareto_solutions = pareto_solutions
        result1.pareto_front = pareto_front
        print(f"Found {len(pareto_front)} solutions on the combined Pareto front.")
    
    # Use the first result (with combined Pareto front) as the main result
    result = result1
    
    runtime = time.time() - start_time
    print(f"\nOptimization completed in {runtime:.2f} seconds")
    
    # Extract results
    if result.pareto_front:
        num_solutions = len(result.pareto_front)
        print(f"Found {num_solutions} solutions on the Pareto front")
        
        if num_solutions > 0:
            # Find distinct types of portfolios
            return_values = np.array([sol[0] for sol in result.pareto_front])
            risk_values = np.array([-sol[1] for sol in result.pareto_front])  # Convert back to positive
            div_values = np.array([sol[2] for sol in result.pareto_front])
            
            # Find extreme portfolios
            max_return_idx = np.argmax(return_values)
            min_risk_idx = np.argmin(risk_values)
            max_div_idx = np.argmax(div_values)
            
            # Find a balanced portfolio (closest to the middle of normalized objectives)
            if num_solutions >= 4:
                # Normalize values to [0,1] range
                norm_return = (return_values - np.min(return_values)) / (np.max(return_values) - np.min(return_values) + 1e-10)
                norm_risk = 1 - (risk_values - np.min(risk_values)) / (np.max(risk_values) - np.min(risk_values) + 1e-10)  # Invert so 1 is best
                norm_div = (div_values - np.min(div_values)) / (np.max(div_values) - np.min(div_values) + 1e-10)
                
                # Calculate distance to ideal point (0.33, 0.33, 0.33) for balanced portfolio
                distances = np.sqrt((norm_return - 0.33)**2 + (norm_risk - 0.33)**2 + (norm_div - 0.33)**2)
                balanced_idx = np.argmin(distances)
                
                # Special indices for display
                display_indices = [max_return_idx, min_risk_idx, max_div_idx, balanced_idx]
                display_names = ["High Return", "Low Risk", "High Diversification", "Balanced"]
                
                # Remove duplicates
                seen = set()
                unique_indices = []
                unique_names = []
                for i, idx in enumerate(display_indices):
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                        unique_names.append(display_names[i])
            else:
                # If we don't have many solutions, just show all of them
                unique_indices = list(range(num_solutions))
                unique_names = [f"Portfolio {i+1}" for i in range(num_solutions)]
            
            # Display the chosen portfolios
            print(f"\nDisplaying {len(unique_indices)} distinct portfolio types:")
            for i, (idx, name) in enumerate(zip(unique_indices, unique_names)):
                solution = result.pareto_solutions[idx]
                weights = np.array(solution['selected_values'][0])
                sum_weights = np.sum(weights)
                weights = weights / sum_weights if sum_weights > 0 else weights
                
                metrics = portfolio_metrics(weights, assets)
                
                print(f"\n{name} Portfolio:")
                print(f"  Expected Return: {metrics[0]:.4f} ({metrics[0]*100:.2f}%)")
                print(f"  Risk: {metrics[1]:.4f} ({metrics[1]*100:.2f}%)")
                print(f"  Asset Diversification: {metrics[2]:.4f}")
                print(f"  Sector Diversification: {metrics[3]:.4f}")
                print(f"  Objective values: Return={solution['multi_fitness'][0]:.4f}, Risk={solution['multi_fitness'][1]:.4f}, Div={solution['multi_fitness'][2]:.4f}")
                
                # Show top allocations
                top_indices = np.argsort(-weights)[:5]
                print("  Top Allocations:")
                for j in top_indices:
                    if weights[j] > 0.01:  # Only show allocations > 1%
                        print(f"    {assets['names'][j]} ({assets['sectors'][j]}): {weights[j]:.4f} ({weights[j]*100:.1f}%)")
        
        # Visualize Pareto front in 3D
        if len(result.pareto_front) > 1:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract the objective values
            x = np.array([sol[0] for sol in result.pareto_front])  # Return
            y = np.array([-sol[1] for sol in result.pareto_front])  # Risk (convert back to positive)
            z = np.array([sol[2] for sol in result.pareto_front])  # Diversification
            
            # Plot all solutions
            scatter = ax.scatter(x, y, z, c=y, marker='o', s=80, alpha=0.6, cmap='viridis', 
                                edgecolors='black', linewidths=1)
            
            # Add labels with larger font
            ax.set_xlabel('Expected Return', fontsize=14, labelpad=10)
            ax.set_ylabel('Risk (Volatility)', fontsize=14, labelpad=10)
            ax.set_zlabel('Diversification', fontsize=14, labelpad=10)
            ax.set_title('Pareto Front for Portfolio Optimization', fontsize=16, pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
            cbar.set_label('Risk Level', fontsize=12)
            
            # Highlight special portfolios with distinct markers if they exist
            if 'max_return_idx' in locals() and 'min_risk_idx' in locals() and 'max_div_idx' in locals():
                # Highlight max return portfolio
                ax.scatter(x[max_return_idx], y[max_return_idx], z[max_return_idx], 
                          color='red', marker='*', s=300, label='Max Return', edgecolors='black')
                
                # Highlight min risk portfolio
                ax.scatter(x[min_risk_idx], y[min_risk_idx], z[min_risk_idx], 
                          color='blue', marker='*', s=300, label='Min Risk', edgecolors='black')
                
                # Highlight max diversification portfolio
                ax.scatter(x[max_div_idx], y[max_div_idx], z[max_div_idx], 
                          color='green', marker='*', s=300, label='Max Diversity', edgecolors='black')
                
                # Highlight balanced portfolio if it exists
                if 'balanced_idx' in locals() and balanced_idx not in [max_return_idx, min_risk_idx, max_div_idx]:
                    ax.scatter(x[balanced_idx], y[balanced_idx], z[balanced_idx], 
                              color='purple', marker='*', s=300, label='Balanced', edgecolors='black')
            
            # Add legend
            ax.legend(fontsize=12, loc='upper left')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Improve the view angle
            ax.view_init(elev=30, azim=45)
            
            # Save figure with high DPI
            plt.savefig('portfolio_pareto_front_3d.png', dpi=300, bbox_inches='tight')
            print("\nSaved 3D Pareto front visualization to 'portfolio_pareto_front_3d.png'")
            
            # 2D projections
            plt.figure(figsize=(15, 5))
            
            # Return vs Risk
            plt.subplot(1, 3, 1)
            plt.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Diversification')
            plt.xlabel('Expected Return')
            plt.ylabel('Risk (Volatility)')
            plt.title('Return vs. Risk')
            plt.grid(True)
            
            # Return vs Diversification
            plt.subplot(1, 3, 2)
            plt.scatter(x, z, c=y, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Risk')
            plt.xlabel('Expected Return')
            plt.ylabel('Diversification')
            plt.title('Return vs. Diversification')
            plt.grid(True)
            
            # Risk vs Diversification
            plt.subplot(1, 3, 3)
            plt.scatter(y, z, c=x, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Return')
            plt.xlabel('Risk (Volatility)')
            plt.ylabel('Diversification')
            plt.title('Risk vs. Diversification')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('portfolio_pareto_front_2d.png')
            print("Saved 2D Pareto front projections to 'portfolio_pareto_front_2d.png'")
            
            # Visualize example portfolios
            # Identify specific portfolio types
            max_return_idx = np.argmax([sol[0] for sol in result.pareto_front])
            min_risk_idx = np.argmin([-sol[1] for sol in result.pareto_front])
            max_div_idx = np.argmax([sol[2] for sol in result.pareto_front])
            
            # Get a balanced portfolio (closest to middle of normalized objectives)
            normalized_x = (np.array(x) - min(x)) / (max(x) - min(x) if max(x) > min(x) else 1)
            normalized_y = 1 - (np.array(y) - min(y)) / (max(y) - min(y) if max(y) > min(y) else 1)  # Invert so 1 is best
            normalized_z = (np.array(z) - min(z)) / (max(z) - min(z) if max(z) > min(z) else 1)
            
            distances = np.sqrt((normalized_x - 0.5)**2 + (normalized_y - 0.5)**2 + (normalized_z - 0.5)**2)
            balanced_idx = np.argmin(distances)
            
            portfolio_indices = [max_return_idx, min_risk_idx, max_div_idx, balanced_idx]
            portfolio_names = ["High Return", "Low Risk", "High Diversification", "Balanced"]
            
            # Keep unique indices only
            unique_indices = []
            unique_names = []
            for i, idx in enumerate(portfolio_indices):
                if idx not in unique_indices:
                    unique_indices.append(idx)
                    unique_names.append(portfolio_names[i])
            
            if len(unique_indices) > 1:
                # Visualize sector allocations for different portfolio types
                plt.figure(figsize=(12, 6 * len(unique_indices)//2))
                
                for i, (idx, name) in enumerate(zip(unique_indices, unique_names)):
                    solution = result.pareto_solutions[idx]
                    weights = np.array(solution['selected_values'][0])
                    sum_weights = np.sum(weights)
                    weights = weights / sum_weights if sum_weights > 0 else weights
                    
                    # Calculate metrics
                    metrics = portfolio_metrics(weights, assets)
                    
                    # Get sector allocations
                    sector_allocations = {}
                    for j, weight in enumerate(weights):
                        sector = assets['sectors'][j]
                        if sector not in sector_allocations:
                            sector_allocations[sector] = 0
                        sector_allocations[sector] += weight
                    
                    # Plot
                    plt.subplot(len(unique_indices)//2 + len(unique_indices)%2, 
                               2, i+1)
                    
                    # Sort by allocation
                    sorted_sectors = sorted(sector_allocations.items(), 
                                           key=lambda x: x[1], reverse=True)
                    sectors = [x[0] for x in sorted_sectors]
                    allocations = [x[1] for x in sorted_sectors]
                    
                    # Create sector bar chart
                    plt.bar(sectors, allocations, alpha=0.7)
                    plt.title(f"{name} Portfolio\nReturn: {metrics[0]*100:.2f}%, " + 
                             f"Risk: {metrics[1]*100:.2f}%, Div: {metrics[2]:.2f}")
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Allocation')
                    plt.ylim(0, max(allocations) * 1.1)
                    
                    # Add percentage labels
                    for j, v in enumerate(allocations):
                        plt.text(j, v+0.01, f"{v*100:.1f}%", ha='center')
                
                plt.tight_layout()
                plt.savefig('portfolio_allocations.png')
                print("Saved portfolio allocations visualization to 'portfolio_allocations.png'")
                
                # Efficient frontier visualization (Return vs Risk only)
                plt.figure(figsize=(10, 8))
                
                # Calculate the efficient frontier boundary
                # Sort points by return
                frontier_points = sorted(zip(x, y), key=lambda point: point[0])
                frontier_x = [point[0] for point in frontier_points]
                frontier_y = [point[1] for point in frontier_points]
                
                # Plot all solutions
                plt.scatter(x, y, c=z, cmap='viridis', s=50, alpha=0.6)
                plt.colorbar(label='Diversification')
                
                # Highlight specific portfolios
                for idx, name in zip(unique_indices, unique_names):
                    plt.scatter(result.pareto_front[idx][0], 
                              -result.pareto_front[idx][1], 
                              s=200, marker='*', label=name)
                
                plt.xlabel('Expected Return')
                plt.ylabel('Risk (Volatility)')
                plt.title('Efficient Frontier - Return vs. Risk')
                plt.grid(True)
                plt.legend()
                plt.savefig('efficient_frontier.png')
                print("Saved efficient frontier visualization to 'efficient_frontier.png'")
                
        else:
            print("Only one solution found on the Pareto front. Not enough for visualization.")
            
    else:
        print("No Pareto front found.")
    
    print("\nMulti-objective portfolio optimization example completed.")

if __name__ == "__main__":
    main()