"""
Simple Return-Risk Multi-Objective Portfolio Optimization Example for TrainSelPy.

This example demonstrates a classic Markowitz portfolio optimization with just two objectives:
1. Maximize expected return
2. Minimize risk (volatility)

The example focuses on generating and visualizing the Pareto frontier (efficient frontier).
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import TrainSelPy functions
from trainselpy import (
    make_data, 
    train_sel, 
    train_sel_control
)

def generate_asset_data(n_assets=10, random_seed=42):
    """Generate synthetic data for assets."""
    np.random.seed(random_seed)
    
    # Generate returns (mix of low and high)
    returns = np.concatenate([
        np.random.uniform(0.03, 0.08, n_assets // 2),  # Low-return assets
        np.random.uniform(0.10, 0.20, n_assets - (n_assets // 2))  # High-return assets
    ])
    
    # Generate risks (correlated with returns but with noise)
    risks = returns * 1.2 + np.random.uniform(0.01, 0.05, n_assets)
    
    # Create correlation matrix (realistic with moderate correlation)
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Generate correlation between 0.1 and 0.7
            corr = 0.1 + 0.6 * np.random.random()
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    assets = {
        'names': [f"Asset_{i+1}" for i in range(n_assets)],
        'returns': returns,
        'risks': risks,
        'corr_matrix': corr_matrix
    }
    
    return assets

def portfolio_metrics(weights, assets):
    """Calculate portfolio return and risk (volatility)."""
    weights = np.array(weights)
    
    # Ensure weights sum to 1.0
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones_like(weights) / len(weights)
    
    # Expected return
    expected_return = np.sum(weights * assets['returns'])
    
    # Risk (volatility)
    risk_vector = assets['risks']
    corr_matrix = assets['corr_matrix']
    
    # Construct covariance matrix
    n = len(weights)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = risk_vector[i] * risk_vector[j] * corr_matrix[i, j]
    
    # Portfolio variance and volatility
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    volatility = np.sqrt(variance)
    
    return expected_return, volatility

def return_risk_objectives(dbl_solution, data):
    """
    Classic Markowitz portfolio optimization objectives.
    
    Returns two objective values:
    1. Return (to be maximized)
    2. Risk (to be minimized, returned as negative to be maximized)
    """
    assets = data['assets']
    
    # Calculate metrics
    return_val, risk = portfolio_metrics(dbl_solution, assets)
    
    # Create objectives
    return_obj = return_val  # Maximize return
    risk_obj = -risk         # Minimize risk (negative for maximization)
    
    # Add a tiny bit of noise to help with dominated solution pruning
    noise = np.random.normal(0, 0.0001, 2)
    
    return [return_obj + noise[0], risk_obj + noise[1]]

def main():
    """Run a simple return-risk multi-objective portfolio optimization."""
    
    print("TrainSelPy Simple Return-Risk Portfolio Optimization Example")
    print("----------------------------------------------------------")
    
    # Generate asset data
    n_assets = 10
    print(f"\nGenerating synthetic data for {n_assets} assets...")
    assets = generate_asset_data(n_assets)
    
    # Display asset information
    print("\nAsset Information:")
    print("Index | Name      | Return  | Risk")
    print("-" * 40)
    for i in range(n_assets):
        print(f"{i:5} | {assets['names'][i]:10} | {assets['returns'][i]:7.4f} | {assets['risks'][i]:7.4f}")
    
    # Create empty features matrix for data object
    features = np.eye(n_assets)
    ts_data = make_data(M=features)
    ts_data['assets'] = assets
    
    # Set control parameters
    print("\nSetting control parameters...")
    control = train_sel_control(
        size="free",
        niterations=50,       # Number of iterations (reduced for speed)
        minitbefstop=20,      # Minimum iterations before stopping
        nEliteSaved=40,       # Number of elite solutions to save
        nelite=100,           # Number of elite solutions
        npop=300,             # Population size 
        mutprob=0.3,          # Mutation probability (increased)
        mutintensity=0.4,     # Mutation intensity (increased)
        crossprob=0.8,        # Crossover probability (increased)
        crossintensity=0.7,   # Crossover intensity (increased)
        niterSANN=15,         # Simulated annealing iterations
        tempini=100.0,        # Initial temperature
        tempfin=0.1,          # Final temperature
        solution_diversity=False,  # Allow similar solutions on Pareto front
        nislands=3,           # Use island model for better Pareto front exploration
        niterIslands=30       # Number of iterations per island
    )
    
    # Multiple optimization runs with different weightings to get a complete efficient frontier
    print("\nRunning optimization with different return-risk weightings...")
    all_results = []
    
    # 1. Balanced optimization (equal weight on return and risk)
    result1 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],  # Choose from all assets
        setsizes=[n_assets],                 # Weights for each asset
        settypes=["DBL"],                    # Continuous variables
        stat=return_risk_objectives,         # Two-objective function
        n_stat=2,                            # Two objectives
        control=control,
        verbose=False
    )
    all_results.append(result1)
    
    # 2. Return-focused optimization (strong bias)
    def return_focused(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 5.0, base[1] * 0.2]
    
    result2 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=return_focused,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result2)
    
    # 3. Risk-focused optimization (strong bias)
    def risk_focused(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 0.2, base[1] * 5.0]
    
    result3 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=risk_focused,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result3)
    
    # 4. Slightly return-biased optimization
    def slight_return_focused(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 2.0, base[1] * 0.8]
    
    result4 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=slight_return_focused,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result4)
    
    # 5. Slightly risk-biased optimization
    def slight_risk_focused(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 0.8, base[1] * 2.0]
    
    result5 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=slight_risk_focused,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result5)
    
    # 6. Extreme return optimization
    def extreme_return(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 10.0, base[1] * 0.1]
    
    result6 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=extreme_return,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result6)
    
    # 7. Extreme risk minimization optimization
    def extreme_risk(dbl_solution, data):
        base = return_risk_objectives(dbl_solution, data)
        return [base[0] * 0.1, base[1] * 10.0]
    
    result7 = train_sel(
        data=ts_data,
        candidates=[list(range(n_assets))],
        setsizes=[n_assets],
        settypes=["DBL"],
        stat=extreme_risk,
        n_stat=2,
        control=control,
        verbose=False
    )
    all_results.append(result7)
    
    # Combine all solutions
    print("\nCombining results to create a complete efficient frontier...")
    combined_solutions = []
    for result in all_results:
        if result.pareto_solutions:
            combined_solutions.extend(result.pareto_solutions)
    
    # Find non-dominated solutions
    if combined_solutions:
        # Extract objective values
        fitness_values = [sol["multi_fitness"] for sol in combined_solutions]
        
        # Find non-dominated solutions
        is_dominated = [False] * len(combined_solutions)
        for i in range(len(combined_solutions)):
            for j in range(len(combined_solutions)):
                if i != j:
                    fi = fitness_values[i]
                    fj = fitness_values[j]
                    if all(fj[k] >= fi[k] for k in range(2)) and any(fj[k] > fi[k] for k in range(2)):
                        is_dominated[i] = True
                        break
        
        # Filter non-dominated solutions
        pareto_solutions = [sol for i, sol in enumerate(combined_solutions) if not is_dominated[i]]
        pareto_front = [sol["multi_fitness"] for sol in pareto_solutions]
        
        # Update the result with the combined Pareto front
        result1.pareto_solutions = pareto_solutions
        result1.pareto_front = pareto_front
        print(f"Found {len(pareto_front)} solutions on the efficient frontier.")
    
    # Use the first result with the combined Pareto front
    result = result1
    
    # Extract results for visualization
    if result.pareto_front and len(result.pareto_front) > 1:
        # Convert to return and risk values
        returns = [obj[0] for obj in result.pareto_front]
        risks = [-obj[1] for obj in result.pareto_front]  # Convert back to positive
        
        # Sort by risk for a smooth frontier
        sorted_indices = np.argsort(risks)
        sorted_returns = [returns[i] for i in sorted_indices]
        sorted_risks = [risks[i] for i in sorted_indices]
        
        # Highlight special portfolios
        min_risk_idx = np.argmin(risks)
        max_return_idx = np.argmax(returns)
        
        # Plot the efficient frontier
        plt.figure(figsize=(12, 8))
        
        # Plot all solutions
        plt.scatter(risks, returns, c=returns, cmap='viridis', s=80, alpha=0.7)
        
        # Connect points with a line to show the frontier
        plt.plot(sorted_risks, sorted_returns, 'k--', alpha=0.5)
        
        # Highlight minimum risk portfolio
        plt.scatter(risks[min_risk_idx], returns[min_risk_idx], 
                   s=300, color='blue', marker='*', edgecolors='black', 
                   label='Minimum Risk Portfolio')
        
        # Highlight maximum return portfolio
        plt.scatter(risks[max_return_idx], returns[max_return_idx], 
                   s=300, color='red', marker='*', edgecolors='black', 
                   label='Maximum Return Portfolio')
        
        # Find a balanced portfolio (approximate equal distance from min risk and max return)
        distances = np.sqrt((np.array(risks) - risks[min_risk_idx])**2 + 
                            (np.array(returns) - returns[max_return_idx])**2)
        balanced_idx = np.argmin(distances)
        
        # Highlight balanced portfolio
        plt.scatter(risks[balanced_idx], returns[balanced_idx], 
                   s=300, color='green', marker='*', edgecolors='black', 
                   label='Balanced Portfolio')
        
        # Add labels and title
        plt.xlabel('Risk (Volatility)', fontsize=14)
        plt.ylabel('Expected Return', fontsize=14)
        plt.title('Efficient Frontier', fontsize=16)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save the figure
        plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
        print("\nSaved efficient frontier visualization to 'efficient_frontier.png'")
        
        # Show portfolio compositions
        plt.figure(figsize=(15, 8))
        
        # Select three key portfolios to display
        key_indices = [min_risk_idx, balanced_idx, max_return_idx]
        key_names = ["Minimum Risk", "Balanced", "Maximum Return"]
        
        for i, (idx, name) in enumerate(zip(key_indices, key_names)):
            solution = result.pareto_solutions[idx]
            weights = np.array(solution['selected_values'][0])
            
            # Normalize weights to sum to 1
            weight_sum = np.sum(weights)
            weights = weights / weight_sum if weight_sum > 0 else weights
            
            # Calculate metrics
            portfolio_return, portfolio_risk = portfolio_metrics(weights, assets)
            
            # Plot portfolio composition
            plt.subplot(1, 3, i+1)
            
            # Sort weights for better visualization
            sorted_indices = np.argsort(-weights)
            sorted_weights = weights[sorted_indices]
            sorted_names = [assets['names'][j] for j in sorted_indices]
            
            # Only show weights > 1%
            significant_indices = [j for j, w in enumerate(sorted_weights) if w > 0.01]
            significant_weights = [sorted_weights[j] for j in significant_indices]
            significant_names = [sorted_names[j] for j in significant_indices]
            
            # Create pie chart
            plt.pie(significant_weights, labels=significant_names, autopct='%1.1f%%',
                   shadow=False, startangle=90)
            
            plt.title(f"{name} Portfolio\nReturn: {portfolio_return*100:.2f}%, Risk: {portfolio_risk*100:.2f}%")
        
        plt.tight_layout()
        plt.savefig('portfolio_allocations.png', dpi=300, bbox_inches='tight')
        print("Saved portfolio allocations visualization to 'portfolio_allocations.png'")
        
    else:
        print("Not enough solutions found on the efficient frontier for visualization.")
    
    print("\nSimple return-risk portfolio optimization completed.")

if __name__ == "__main__":
    main()