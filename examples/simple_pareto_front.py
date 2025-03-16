"""
Simple Return-Risk Portfolio Optimization Example - Direct Pareto Front Calculation.

This example demonstrates a classic Markowitz portfolio optimization with just two objectives:
1. Maximize expected return
2. Minimize risk (volatility)

The example focuses on directly calculating and visualizing the Pareto frontier (efficient frontier).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

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
    
    # Create covariance matrix
    cov_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            cov_matrix[i, j] = risks[i] * risks[j] * corr_matrix[i, j]
    
    assets = {
        'names': [f"Asset_{i+1}" for i in range(n_assets)],
        'returns': returns,
        'risks': risks,
        'corr_matrix': corr_matrix,
        'cov_matrix': cov_matrix
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
    cov_matrix = assets['cov_matrix']
    variance = weights.dot(cov_matrix).dot(weights)
    volatility = np.sqrt(variance)
    
    return expected_return, volatility

def objective_function(weights, cov_matrix, exp_returns, risk_aversion):
    """
    Objective function for portfolio optimization.
    
    Combines return and risk in a single objective with a risk aversion parameter.
    Risk aversion = 0 means care only about return.
    Risk aversion = 1 means care only about risk.
    """
    weights = np.array(weights)
    portfolio_return = np.sum(weights * exp_returns)
    portfolio_variance = weights.dot(cov_matrix).dot(weights)
    
    # Objective to minimize: -return + risk_aversion * variance
    return -portfolio_return + risk_aversion * portfolio_variance

def optimize_portfolio(assets, risk_aversion, initial_weights=None):
    """Optimize portfolio weights for a given risk aversion parameter."""
    n_assets = len(assets['returns'])
    cov_matrix = assets['cov_matrix']
    exp_returns = assets['returns']
    
    # Initial guess (equal weight if not provided)
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: each weight between 0 and 1
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        objective_function,
        initial_weights,
        args=(cov_matrix, exp_returns, risk_aversion),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimal weights
    weights = result['x']
    
    # Calculate metrics
    return_val, risk = portfolio_metrics(weights, assets)
    
    return weights, return_val, risk

def generate_efficient_frontier(assets, n_points=20):
    """Generate points along the efficient frontier."""
    # Range of risk aversion parameters (from return-focused to risk-focused)
    risk_aversions = np.logspace(-3, 2, n_points)
    
    # Store results
    frontier_weights = []
    frontier_returns = []
    frontier_risks = []
    
    # Initial weights (equal allocation)
    n_assets = len(assets['returns'])
    initial_weights = np.ones(n_assets) / n_assets
    
    # For each risk aversion parameter
    for ra in risk_aversions:
        # Optimize portfolio
        weights, ret, risk = optimize_portfolio(assets, ra, initial_weights)
        
        # Use these weights as starting point for next optimization
        initial_weights = weights
        
        # Store results
        frontier_weights.append(weights)
        frontier_returns.append(ret)
        frontier_risks.append(risk)
    
    return frontier_weights, frontier_returns, frontier_risks

def main():
    """Run the direct Pareto front calculation example."""
    
    print("Simple Return-Risk Portfolio Optimization - Direct Pareto Front Calculation")
    print("------------------------------------------------------------------------")
    
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
    
    # Generate efficient frontier
    print("\nCalculating efficient frontier...")
    start_time = time.time()
    weights, returns, risks = generate_efficient_frontier(assets, n_points=30)
    end_time = time.time()
    
    print(f"Calculation completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(returns)} points on the efficient frontier")
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    
    # Plot all frontier points
    plt.scatter(risks, returns, c=returns, cmap='viridis', s=100, alpha=0.7)
    
    # Connect points with a line
    # Sort by risk for a smooth curve
    sorted_indices = np.argsort(risks)
    sorted_risks = [risks[i] for i in sorted_indices]
    sorted_returns = [returns[i] for i in sorted_indices]
    plt.plot(sorted_risks, sorted_returns, 'k--', alpha=0.5)
    
    # Find and mark notable portfolios
    min_risk_idx = np.argmin(risks)
    max_return_idx = np.argmax(returns)
    
    # Highlight minimum risk portfolio
    plt.scatter(risks[min_risk_idx], returns[min_risk_idx], 
               s=300, color='blue', marker='*', edgecolors='black', 
               label='Minimum Risk Portfolio')
    
    # Highlight maximum return portfolio
    plt.scatter(risks[max_return_idx], returns[max_return_idx], 
               s=300, color='red', marker='*', edgecolors='black', 
               label='Maximum Return Portfolio')
    
    # Find a balanced portfolio (approximately equal distance from min risk and max return)
    # Use Sharpe ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratios = [(r - risk_free_rate) / risk for r, risk in zip(returns, risks)]
    max_sharpe_idx = np.argmax(sharpe_ratios)
    
    # Highlight balanced portfolio (maximum Sharpe ratio)
    plt.scatter(risks[max_sharpe_idx], returns[max_sharpe_idx], 
               s=300, color='green', marker='*', edgecolors='black', 
               label='Optimal Sharpe Ratio Portfolio')
    
    # Add labels and title
    plt.xlabel('Risk (Volatility)', fontsize=14)
    plt.ylabel('Expected Return', fontsize=14)
    plt.title('Efficient Frontier', fontsize=16)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save figure
    plt.savefig('efficient_frontier_direct.png', dpi=300, bbox_inches='tight')
    print("\nSaved efficient frontier visualization to 'efficient_frontier_direct.png'")
    
    # Show portfolio compositions for key portfolios
    plt.figure(figsize=(15, 8))
    
    # Key portfolios to visualize
    key_indices = [min_risk_idx, max_sharpe_idx, max_return_idx]
    key_names = ["Minimum Risk", "Maximum Sharpe", "Maximum Return"]
    
    for i, (idx, name) in enumerate(zip(key_indices, key_names)):
        portfolio_weights = weights[idx]
        portfolio_return = returns[idx]
        portfolio_risk = risks[idx]
        
        # Plot portfolio composition
        plt.subplot(1, 3, i+1)
        
        # Sort weights for better visualization
        sorted_indices = np.argsort(-portfolio_weights)
        sorted_weights = portfolio_weights[sorted_indices]
        sorted_names = [assets['names'][j] for j in sorted_indices]
        
        # Only show weights > 1%
        significant_indices = [j for j, w in enumerate(sorted_weights) if w > 0.01]
        significant_weights = [sorted_weights[j] for j in significant_indices]
        significant_names = [sorted_names[j] for j in significant_indices]
        
        # Create pie chart
        plt.pie(significant_weights, labels=significant_names, autopct='%1.1f%%',
               shadow=False, startangle=90)
        
        if name == "Maximum Sharpe":
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            plt.title(f"{name} Portfolio\nReturn: {portfolio_return*100:.2f}%, Risk: {portfolio_risk*100:.2f}%\nSharpe: {sharpe:.2f}")
        else:
            plt.title(f"{name} Portfolio\nReturn: {portfolio_return*100:.2f}%, Risk: {portfolio_risk*100:.2f}%")
    
    plt.tight_layout()
    plt.savefig('portfolio_allocations_direct.png', dpi=300, bbox_inches='tight')
    print("Saved portfolio allocations visualization to 'portfolio_allocations_direct.png'")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()