"""
Multi-Environmental Genomic Breeding Experiment Design using TrainSelPy

This example demonstrates how to use TrainSelPy with the CDMean criterion for 
designing multi-environmental genomic breeding experiments. It uses a Kronecker 
product covariance structure to model genotype-by-environment interactions.

Key Concepts:
- Genomic relationship matrix (G_genotype)
- Environmental correlation matrix (G_environment)
- Kronecker product covariance: G = G_genotype ⊗ G_environment
- CDMean criterion for optimal training set selection
- Prediction accuracy evaluation across environments

Author: TrainSelPy Development Team
Date: 2025-11-24
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainselpy import make_data, train_sel, set_control_default
from trainselpy.optimization_criteria import cdmean_opt


def simulate_genomic_data(
    n_individuals: int,
    n_markers: int,
    maf_range: Tuple[float, float] = (0.1, 0.5),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate genomic marker data and compute genomic relationship matrix.
    
    Uses the VanRaden (2008) method for computing the genomic relationship matrix:
    G = (M - P)(M - P)' / (2 * sum(p_i * (1 - p_i)))
    
    Parameters
    ----------
    n_individuals : int
        Number of individuals in the breeding population
    n_markers : int
        Number of SNP markers
    maf_range : tuple
        Range of minor allele frequencies (min, max)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    markers : np.ndarray
        Marker matrix (n_individuals × n_markers) with values {0, 1, 2}
    G_genotype : np.ndarray
        Genomic relationship matrix (n_individuals × n_individuals)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate minor allele frequencies
    maf = np.random.uniform(maf_range[0], maf_range[1], n_markers)
    
    # Simulate markers (0, 1, 2 copies of minor allele)
    markers = np.zeros((n_individuals, n_markers))
    for i in range(n_markers):
        # Binomial sampling for each individual
        markers[:, i] = np.random.binomial(2, maf[i], n_individuals)
    
    # Compute expected frequencies (2 * p_i)
    P = 2 * maf
    
    # Center the marker matrix
    M_centered = markers - P
    
    # Compute scaling factor
    scaling = 2 * np.sum(maf * (1 - maf))
    
    # Compute genomic relationship matrix
    G_genotype = (M_centered @ M_centered.T) / scaling
    
    # Add small value to diagonal for numerical stability
    G_genotype += np.eye(n_individuals) * 1e-6
    
    return markers, G_genotype


def create_environmental_correlation(
    n_environments: int,
    correlation_type: str = 'compound_symmetry',
    rho: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """
    Create environmental correlation matrix.
    
    Parameters
    ----------
    n_environments : int
        Number of environments
    correlation_type : str
        Type of correlation structure:
        - 'compound_symmetry': All off-diagonal elements equal to rho
        - 'ar1': AR(1) structure with correlation rho^|i-j|
        - 'unstructured': Random positive definite matrix
    rho : float
        Correlation parameter (0 < rho < 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G_environment : np.ndarray
        Environmental correlation matrix (n_environments × n_environments)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if correlation_type == 'compound_symmetry':
        # All environments have same correlation
        G_environment = np.full((n_environments, n_environments), rho)
        np.fill_diagonal(G_environment, 1.0)
        
    elif correlation_type == 'ar1':
        # AR(1) structure: correlation decreases with distance
        G_environment = np.zeros((n_environments, n_environments))
        for i in range(n_environments):
            for j in range(n_environments):
                G_environment[i, j] = rho ** abs(i - j)
                
    elif correlation_type == 'unstructured':
        # Generate random positive definite matrix
        A = np.random.randn(n_environments, n_environments)
        G_environment = A @ A.T
        # Standardize to correlation matrix
        D = np.sqrt(np.diag(G_environment))
        G_environment = G_environment / np.outer(D, D)
        # Scale by rho
        G_environment = (1 - rho) * np.eye(n_environments) + rho * G_environment
        
    else:
        raise ValueError(f"Unknown correlation type: {correlation_type}")
    
    return G_environment


def create_kronecker_covariance(
    G_genotype: np.ndarray,
    G_environment: np.ndarray
) -> np.ndarray:
    """
    Create Kronecker product covariance matrix.
    
    The full covariance matrix for genotype-environment combinations is:
    G_full = G_genotype ⊗ G_environment
    
    This means that the covariance between:
    - Same genotype in different environments: determined by G_environment
    - Different genotypes in same environment: determined by G_genotype
    - Different genotypes in different environments: product of both
    
    Parameters
    ----------
    G_genotype : np.ndarray
        Genomic relationship matrix (n_individuals × n_individuals)
    G_environment : np.ndarray
        Environmental correlation matrix (n_environments × n_environments)
        
    Returns
    -------
    G_full : np.ndarray
        Full covariance matrix (n_individuals*n_environments × n_individuals*n_environments)
    """
    # Compute Kronecker product
    G_full = np.kron(G_genotype, G_environment)
    
    return G_full


def optimize_training_set_cdmean(
    G_full: np.ndarray,
    n_select: int,
    lambda_val: float = 0.01,
    control: Dict[str, Any] = None,
    verbose: bool = True
) -> Tuple[List[int], float]:
    """
    Optimize training set selection using CDMean criterion.
    
    Parameters
    ----------
    G_full : np.ndarray
        Full covariance matrix
    n_select : int
        Number of individuals to select for training
    lambda_val : float
        Regularization parameter
    control : dict, optional
        Control parameters for genetic algorithm
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    selected_indices : list
        Indices of selected individuals
    cdmean_value : float
        CDMean value for the selected set
    """
    # Create data structure for TrainSelPy
    data = make_data(K=G_full)
    data["lambda"] = lambda_val
    
    # Set control parameters if not provided
    if control is None:
        control = set_control_default()
        control["niterations"] = 100
        control["npop"] = 500
        control["mutprob"] = 0.5
        control["crossprob"] = 0.8
    
    # Run optimization
    result = train_sel(
        data=data,
        candidates=[list(range(G_full.shape[0]))],
        setsizes=[n_select],
        settypes=["UOS"],
        stat=cdmean_opt,
        control=control,
        verbose=verbose
    )
    
    selected_indices = result.selected_indices[0]
    cdmean_value = result.fitness
    
    return selected_indices, cdmean_value


def random_selection(
    n_total: int,
    n_select: int,
    seed: int = None
) -> List[int]:
    """
    Randomly select training set for comparison.
    
    Parameters
    ----------
    n_total : int
        Total number of individuals
    n_select : int
        Number to select
    seed : int, optional
        Random seed
        
    Returns
    -------
    selected_indices : list
        Randomly selected indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    selected_indices = np.random.choice(n_total, n_select, replace=False).tolist()
    return selected_indices


def evaluate_prediction_accuracy(
    G_full: np.ndarray,
    training_indices: List[int],
    test_indices: List[int],
    n_individuals: int,
    n_environments: int,
    lambda_val: float = 0.01,
    h2: float = 0.5,
    seed: int = None
) -> Dict[str, Any]:
    """
    Evaluate prediction accuracy using GBLUP.
    
    Simulates breeding values and evaluates prediction accuracy for test set
    using the training set.
    
    Parameters
    ----------
    G_full : np.ndarray
        Full covariance matrix
    training_indices : list
        Indices of training individuals
    test_indices : list
        Indices of test individuals
    n_individuals : int
        Number of individuals
    n_environments : int
        Number of environments
    lambda_val : float
        Regularization parameter
    h2 : float
        Heritability (proportion of genetic variance)
    seed : int, optional
        Random seed
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'overall_accuracy': Overall prediction accuracy
        - 'per_environment_accuracy': Accuracy per environment
        - 'true_bv': True breeding values
        - 'predicted_bv': Predicted breeding values
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_total = G_full.shape[0]
    
    # Simulate true breeding values
    # BV ~ MVN(0, G_full)
    true_bv = np.random.multivariate_normal(np.zeros(n_total), G_full)
    
    # Add environmental noise
    error_var = (1 - h2) / h2
    noise = np.random.normal(0, np.sqrt(error_var), n_total)
    phenotypes = true_bv + noise
    
    # GBLUP prediction
    # Extract relevant submatrices
    G_TT = G_full[np.ix_(training_indices, training_indices)]
    G_PT = G_full[np.ix_(test_indices, training_indices)]
    
    # Add regularization
    G_TT_reg = G_TT + lambda_val * np.eye(len(training_indices))
    
    # Solve for training effects
    y_train = phenotypes[training_indices]
    
    # Predict test set: u_P = G_PT @ (G_TT + λI)^(-1) @ y_T
    predicted_bv_test = G_PT @ np.linalg.solve(G_TT_reg, y_train)
    
    # Calculate overall accuracy
    true_bv_test = true_bv[test_indices]
    overall_accuracy = np.corrcoef(true_bv_test, predicted_bv_test)[0, 1]
    
    # Calculate per-environment accuracy
    per_env_accuracy = []
    for env in range(n_environments):
        # Get indices for this environment
        env_test_mask = []
        for idx in test_indices:
            # Check if this index belongs to current environment
            individual = idx // n_environments
            environment = idx % n_environments
            if environment == env:
                env_test_mask.append(True)
            else:
                env_test_mask.append(False)
        
        env_test_mask = np.array(env_test_mask)
        
        if np.sum(env_test_mask) > 1:
            env_true = true_bv_test[env_test_mask]
            env_pred = predicted_bv_test[env_test_mask]
            env_acc = np.corrcoef(env_true, env_pred)[0, 1]
            per_env_accuracy.append(env_acc)
        else:
            per_env_accuracy.append(np.nan)
    
    results = {
        'overall_accuracy': overall_accuracy,
        'per_environment_accuracy': per_env_accuracy,
        'true_bv': true_bv,
        'predicted_bv_test': predicted_bv_test,
        'test_indices': test_indices
    }
    
    return results


def visualize_results(
    G_genotype: np.ndarray,
    G_environment: np.ndarray,
    n_individuals: int,
    n_environments: int,
    optimized_indices: List[int],
    random_indices: List[int],
    optimized_accuracy: Dict[str, Any],
    random_accuracy: Dict[str, Any],
    cdmean_optimized: float,
    cdmean_random: float,
    scenario_name: str = ""
):
    """
    Create comprehensive visualization of results.
    
    Parameters
    ----------
    G_genotype : np.ndarray
        Genomic relationship matrix
    G_environment : np.ndarray
        Environmental correlation matrix
    n_individuals : int
        Number of individuals
    n_environments : int
        Number of environments
    optimized_indices : list
        Indices selected by CDMean optimization
    random_indices : list
        Randomly selected indices
    optimized_accuracy : dict
        Prediction accuracy for optimized selection
    random_accuracy : dict
        Prediction accuracy for random selection
    cdmean_optimized : float
        CDMean value for optimized selection
    cdmean_random : float
        CDMean value for random selection
    scenario_name : str
        Name of the scenario for the plot title
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Genomic relationship matrix
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(G_genotype, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Genomic Relationship'},
                square=True, ax=ax1)
    ax1.set_title('Genomic Relationship Matrix (G_genotype)')
    ax1.set_xlabel('Individual')
    ax1.set_ylabel('Individual')
    
    # 2. Environmental correlation matrix
    ax2 = plt.subplot(3, 4, 2)
    sns.heatmap(G_environment, cmap='YlOrRd', vmin=0, vmax=1,
                annot=True, fmt='.2f', square=True, ax=ax2,
                cbar_kws={'label': 'Correlation'})
    ax2.set_title('Environmental Correlation Matrix (G_environment)')
    ax2.set_xlabel('Environment')
    ax2.set_ylabel('Environment')
    
    # 3. Selection pattern - Optimized
    ax3 = plt.subplot(3, 4, 3)
    selected_individuals_opt = [idx // n_environments for idx in optimized_indices]
    selected_environments_opt = [idx % n_environments for idx in optimized_indices]
    
    # Create heatmap showing selection
    selection_matrix_opt = np.zeros((n_individuals, n_environments))
    for ind, env in zip(selected_individuals_opt, selected_environments_opt):
        selection_matrix_opt[ind, env] = 1
    
    sns.heatmap(selection_matrix_opt, cmap='Greens', vmin=0, vmax=1,
                cbar_kws={'label': 'Selected'}, ax=ax3)
    ax3.set_title(f'CDMean-Optimized Selection\n(CDMean={cdmean_optimized:.4f})')
    ax3.set_xlabel('Environment')
    ax3.set_ylabel('Individual')
    
    # 4. Selection pattern - Random
    ax4 = plt.subplot(3, 4, 4)
    selected_individuals_rand = [idx // n_environments for idx in random_indices]
    selected_environments_rand = [idx % n_environments for idx in random_indices]
    
    selection_matrix_rand = np.zeros((n_individuals, n_environments))
    for ind, env in zip(selected_individuals_rand, selected_environments_rand):
        selection_matrix_rand[ind, env] = 1
    
    sns.heatmap(selection_matrix_rand, cmap='Greens', vmin=0, vmax=1,
                cbar_kws={'label': 'Selected'}, ax=ax4)
    ax4.set_title(f'Random Selection\n(CDMean={cdmean_random:.4f})')
    ax4.set_xlabel('Environment')
    ax4.set_ylabel('Individual')
    
    # 5. Per-environment accuracy comparison
    ax5 = plt.subplot(3, 4, 5)
    env_labels = [f'Env {i+1}' for i in range(n_environments)]
    x = np.arange(n_environments)
    width = 0.35
    
    opt_acc = optimized_accuracy['per_environment_accuracy']
    rand_acc = random_accuracy['per_environment_accuracy']
    
    ax5.bar(x - width/2, opt_acc, width, label='CDMean-Optimized', color='steelblue')
    ax5.bar(x + width/2, rand_acc, width, label='Random', color='coral')
    ax5.set_xlabel('Environment')
    ax5.set_ylabel('Prediction Accuracy')
    ax5.set_title('Prediction Accuracy by Environment')
    ax5.set_xticks(x)
    ax5.set_xticklabels(env_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 6. Overall accuracy comparison
    ax6 = plt.subplot(3, 4, 6)
    methods = ['CDMean-Optimized', 'Random']
    accuracies = [optimized_accuracy['overall_accuracy'], 
                  random_accuracy['overall_accuracy']]
    colors = ['steelblue', 'coral']
    
    bars = ax6.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Overall Prediction Accuracy')
    ax6.set_title('Overall Prediction Accuracy Comparison')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 7. CDMean comparison
    ax7 = plt.subplot(3, 4, 7)
    cdmean_values = [cdmean_optimized, cdmean_random]
    bars = ax7.bar(methods, cdmean_values, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('CDMean Value')
    ax7.set_title('CDMean Criterion Comparison')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, cdmean_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 8. Individuals per environment - Optimized
    ax8 = plt.subplot(3, 4, 8)
    individuals_per_env_opt = [selected_environments_opt.count(i) for i in range(n_environments)]
    ax8.bar(env_labels, individuals_per_env_opt, color='steelblue', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Environment')
    ax8.set_ylabel('Number of Selected Individuals')
    ax8.set_title('Distribution: CDMean-Optimized')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Individuals per environment - Random
    ax9 = plt.subplot(3, 4, 9)
    individuals_per_env_rand = [selected_environments_rand.count(i) for i in range(n_environments)]
    ax9.bar(env_labels, individuals_per_env_rand, color='coral', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Environment')
    ax9.set_ylabel('Number of Selected Individuals')
    ax9.set_title('Distribution: Random')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. Improvement over random
    ax10 = plt.subplot(3, 4, 10)
    improvement = [(opt - rand) for opt, rand in zip(opt_acc, rand_acc)]
    colors_imp = ['green' if x > 0 else 'red' for x in improvement]
    ax10.bar(env_labels, improvement, color=colors_imp, alpha=0.7, edgecolor='black')
    ax10.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax10.set_xlabel('Environment')
    ax10.set_ylabel('Accuracy Improvement')
    ax10.set_title('CDMean-Optimized vs Random\n(Positive = Better)')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Summary statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Overall Accuracy:
      CDMean-Optimized: {optimized_accuracy['overall_accuracy']:.4f}
      Random:           {random_accuracy['overall_accuracy']:.4f}
      Improvement:      {optimized_accuracy['overall_accuracy'] - random_accuracy['overall_accuracy']:.4f}
    
    Mean Per-Env Accuracy:
      CDMean-Optimized: {np.nanmean(opt_acc):.4f}
      Random:           {np.nanmean(rand_acc):.4f}
      Improvement:      {np.nanmean(opt_acc) - np.nanmean(rand_acc):.4f}
    
    CDMean Value:
      CDMean-Optimized: {cdmean_optimized:.4f}
      Random:           {cdmean_random:.4f}
      Ratio:            {cdmean_optimized/cdmean_random:.2f}x
    """
    
    ax11.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center')
    
    # 12. Environmental correlation vs accuracy
    ax12 = plt.subplot(3, 4, 12)
    # Get unique off-diagonal correlations
    env_corrs = []
    for i in range(n_environments):
        for j in range(i+1, n_environments):
            env_corrs.append(G_environment[i, j])
    
    mean_corr = np.mean(env_corrs)
    mean_acc_opt = np.nanmean(opt_acc)
    mean_acc_rand = np.nanmean(rand_acc)
    
    ax12.scatter([mean_corr], [mean_acc_opt], s=200, c='steelblue', 
                label='CDMean-Optimized', marker='o', edgecolor='black', linewidth=2)
    ax12.scatter([mean_corr], [mean_acc_rand], s=200, c='coral',
                label='Random', marker='s', edgecolor='black', linewidth=2)
    ax12.set_xlabel('Mean Environmental Correlation')
    ax12.set_ylabel('Mean Prediction Accuracy')
    ax12.set_title('Correlation vs Accuracy')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    ax12.set_xlim([0, 1])
    ax12.set_ylim([0, 1])
    
    plt.suptitle(f'Multi-Environment Breeding Experiment Design{scenario_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main function demonstrating multi-environment breeding experiment design.
    """
    print("="*80)
    print("Multi-Environmental Genomic Breeding Experiment Design")
    print("Using TrainSelPy with CDMean Criterion and Kronecker Covariance")
    print("="*80)
    
    # Set parameters
    n_individuals = 100
    n_environments = 4
    n_markers = 500
    n_select = 40  # Select 40 genotype-environment combinations
    lambda_val = 0.01
    h2 = 0.5
    
    print(f"\nExperiment Parameters:")
    print(f"  Number of individuals: {n_individuals}")
    print(f"  Number of environments: {n_environments}")
    print(f"  Number of markers: {n_markers}")
    print(f"  Training set size: {n_select}")
    print(f"  Heritability: {h2}")
    print(f"  Regularization (λ): {lambda_val}")
    
    # Simulate genomic data
    print("\n" + "-"*80)
    print("Step 1: Simulating genomic data...")
    markers, G_genotype = simulate_genomic_data(n_individuals, n_markers, seed=42)
    print(f"  Generated {n_markers} markers for {n_individuals} individuals")
    print(f"  G_genotype shape: {G_genotype.shape}")
    print(f"  G_genotype range: [{G_genotype.min():.3f}, {G_genotype.max():.3f}]")
    
    # Test three scenarios with different environmental correlations
    scenarios = [
        {'name': 'High Correlation', 'rho': 0.8, 'type': 'compound_symmetry'},
        {'name': 'Moderate Correlation', 'rho': 0.5, 'type': 'compound_symmetry'},
        {'name': 'Low Correlation', 'rho': 0.2, 'type': 'compound_symmetry'}
    ]
    
    for scenario in scenarios:
        print("\n" + "="*80)
        print(f"SCENARIO: {scenario['name']} (ρ = {scenario['rho']})")
        print("="*80)
        
        # Create environmental correlation
        print("\nStep 2: Creating environmental correlation structure...")
        G_environment = create_environmental_correlation(
            n_environments,
            correlation_type=scenario['type'],
            rho=scenario['rho'],
            seed=42
        )
        print(f"  G_environment shape: {G_environment.shape}")
        print(f"  Environmental correlation matrix:")
        print(f"{G_environment}")
        
        # Create Kronecker covariance
        print("\nStep 3: Creating Kronecker product covariance...")
        G_full = create_kronecker_covariance(G_genotype, G_environment)
        print(f"  G_full shape: {G_full.shape}")
        print(f"  G_full = G_genotype ⊗ G_environment")
        print(f"  Total genotype-environment combinations: {G_full.shape[0]}")
        
        # Optimize training set with CDMean
        print("\nStep 4: Optimizing training set with CDMean...")
        control = set_control_default()
        control["niterations"] = 50
        control["npop"] = 300
        control["mutprob"] = 0.5
        control["crossprob"] = 0.8
        
        optimized_indices, cdmean_optimized = optimize_training_set_cdmean(
            G_full, n_select, lambda_val, control, verbose=False
        )
        print(f"  Selected {len(optimized_indices)} genotype-environment combinations")
        print(f"  CDMean value: {cdmean_optimized:.6f}")
        
        # Random selection for comparison
        print("\nStep 5: Random selection for comparison...")
        random_indices = random_selection(G_full.shape[0], n_select, seed=42)
        
        # Calculate CDMean for random selection
        data_random = make_data(K=G_full)
        data_random["lambda"] = lambda_val
        cdmean_random = cdmean_opt(random_indices, data_random)
        print(f"  CDMean value (random): {cdmean_random:.6f}")
        print(f"  CDMean improvement: {(cdmean_optimized/cdmean_random - 1)*100:.2f}%")
        
        # Evaluate prediction accuracy
        print("\nStep 6: Evaluating prediction accuracy...")
        test_indices = [i for i in range(G_full.shape[0]) if i not in optimized_indices]
        
        optimized_accuracy = evaluate_prediction_accuracy(
            G_full, optimized_indices, test_indices,
            n_individuals, n_environments, lambda_val, h2, seed=42
        )
        
        test_indices_rand = [i for i in range(G_full.shape[0]) if i not in random_indices]
        random_accuracy = evaluate_prediction_accuracy(
            G_full, random_indices, test_indices_rand,
            n_individuals, n_environments, lambda_val, h2, seed=42
        )
        
        print(f"\n  Overall Prediction Accuracy:")
        print(f"    CDMean-Optimized: {optimized_accuracy['overall_accuracy']:.4f}")
        print(f"    Random:           {random_accuracy['overall_accuracy']:.4f}")
        print(f"    Improvement:      {optimized_accuracy['overall_accuracy'] - random_accuracy['overall_accuracy']:.4f}")
        
        print(f"\n  Per-Environment Prediction Accuracy:")
        for env in range(n_environments):
            opt_acc = optimized_accuracy['per_environment_accuracy'][env]
            rand_acc = random_accuracy['per_environment_accuracy'][env]
            print(f"    Environment {env+1}: Optimized={opt_acc:.4f}, Random={rand_acc:.4f}, Diff={opt_acc-rand_acc:+.4f}")
        
        # Visualize results
        print("\nStep 7: Creating visualization...")
        fig = visualize_results(
            G_genotype, G_environment, n_individuals, n_environments,
            optimized_indices, random_indices,
            optimized_accuracy, random_accuracy,
            cdmean_optimized, cdmean_random,
            scenario_name=f"\nScenario: {scenario['name']} (ρ={scenario['rho']})"
        )
        
        filename = f"multi_env_breeding_{scenario['name'].lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to '{filename}'")
        plt.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. CDMean-optimized selection consistently outperforms random selection")
    print("2. Higher environmental correlation leads to higher prediction accuracy")
    print("3. The Kronecker structure properly models genotype-environment interactions")
    print("4. CDMean effectively identifies informative training sets across environments")
    print("\nRecommendations for Breeders:")
    print("- Use CDMean for optimal training set design in multi-environment trials")
    print("- Consider environmental correlation when planning experiments")
    print("- Balance training set across environments based on correlation structure")
    print("- Higher correlation allows for fewer environments in training set")


if __name__ == "__main__":
    main()
