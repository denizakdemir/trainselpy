# Multi-Environmental Genomic Breeding Experiment Design

## Overview

This example demonstrates how to use TrainSelPy with the **CDMean criterion** for designing multi-environmental genomic breeding experiments. It uses a **Kronecker product covariance structure** to model genotype-by-environment (G×E) interactions.

## Key Concepts

### Kronecker Product Covariance Structure

In multi-environment breeding trials, genotypes are evaluated across multiple environments. The breeding values (BVs) have a covariance structure modeled as:

**G = G_genotype ⊗ G_environment**

Where:
- **G_genotype**: Genomic relationship matrix (n × n individuals)
- **G_environment**: Environmental correlation matrix (e × e environments)
- **⊗**: Kronecker product operator
- **G**: Full covariance matrix (ne × ne combinations)

This structure assumes:
1. The same genetic variance in all environments
2. Environmental correlations are constant across all genotypes
3. G×E interactions follow a specific multiplicative pattern

### CDMean Criterion

The **CDMean (Coefficient of Determination Mean)** criterion measures the average prediction accuracy for non-selected individuals. It's particularly useful for:
- Maximizing prediction accuracy across environments
- Balancing training set across correlated environments
- Optimizing resource allocation in breeding programs

## Example Usage

### Basic Example

```python
from multi_environment_breeding_example import (
    simulate_genomic_data,
    create_environmental_correlation,
    create_kronecker_covariance,
    optimize_training_set_cdmean
)

# 1. Simulate genomic data
n_individuals = 100
n_markers = 500
markers, G_genotype = simulate_genomic_data(n_individuals, n_markers, seed=42)

# 2. Create environmental correlation (4 environments, ρ=0.6)
n_environments = 4
G_environment = create_environmental_correlation(
    n_environments,
    correlation_type='compound_symmetry',
    rho=0.6
)

# 3. Create Kronecker covariance
G_full = create_kronecker_covariance(G_genotype, G_environment)

# 4. Optimize training set (select 40 genotype-environment combinations)
selected_indices, cdmean_value = optimize_training_set_cdmean(
    G_full,
    n_select=40,
    lambda_val=0.01
)

print(f"Selected {len(selected_indices)} combinations")
print(f"CDMean value: {cdmean_value:.4f}")
```

### Running the Full Example

```bash
cd examples
python multi_environment_breeding_example.py
```

This will:
1. Simulate a breeding population with 100 individuals and 500 markers
2. Test three scenarios with different environmental correlations (ρ = 0.8, 0.5, 0.2)
3. Compare CDMean-optimized vs. random selection
4. Generate comprehensive visualizations for each scenario

## Results

### High Correlation Scenario (ρ = 0.8)

**CDMean Values:**
- Optimized: 0.2515
- Random: 0.2132
- Improvement: 17.97%

**Prediction Accuracy:**
- Optimized: 0.428
- Random: 0.350
- Improvement: +0.078

**Key Insight:** High environmental correlation allows for effective prediction across environments with fewer training samples per environment.

### Moderate Correlation Scenario (ρ = 0.5)

**CDMean Values:**
- Optimized: 0.1112
- Random: 0.0973
- Improvement: 14.21%

**Key Insight:** Moderate correlation requires more balanced training set across environments.

### Low Correlation Scenario (ρ = 0.2)

**CDMean Values:**
- Optimized: 0.0368
- Random: 0.0334
- Improvement: 10.23%

**Key Insight:** Low correlation requires training samples in each environment for accurate prediction.

## Visualizations

Each scenario generates a comprehensive 12-panel visualization showing:

1. **Genomic Relationship Matrix** - Heatmap of genetic similarities
2. **Environmental Correlation Matrix** - Structure of environment relationships
3. **CDMean-Optimized Selection Pattern** - Which genotype-environment combinations were selected
4. **Random Selection Pattern** - Comparison baseline
5. **Per-Environment Accuracy** - Prediction accuracy in each environment
6. **Overall Accuracy Comparison** - Bar chart comparing methods
7. **CDMean Value Comparison** - Criterion values for both methods
8. **Distribution (Optimized)** - How training samples are distributed across environments
9. **Distribution (Random)** - Random distribution for comparison
10. **Improvement Over Random** - Environment-specific gains
11. **Summary Statistics** - Numerical summary of results
12. **Correlation vs Accuracy** - Relationship between environmental correlation and prediction accuracy

## Functions

### Data Simulation

**`simulate_genomic_data(n_individuals, n_markers, maf_range=(0.1, 0.5), seed=None)`**

Simulates SNP marker data and computes genomic relationship matrix using VanRaden (2008) method.

**Parameters:**
- `n_individuals`: Number of individuals in breeding population
- `n_markers`: Number of SNP markers
- `maf_range`: Range of minor allele frequencies (default: 0.1-0.5)
- `seed`: Random seed for reproducibility

**Returns:**
- `markers`: Marker matrix (n_individuals × n_markers) with values {0, 1, 2}
- `G_genotype`: Genomic relationship matrix (n_individuals × n_individuals)

### Environmental Correlation

**`create_environmental_correlation(n_environments, correlation_type='compound_symmetry', rho=0.5, seed=None)`**

Creates environmental correlation matrix with different structures.

**Parameters:**
- `n_environments`: Number of environments
- `correlation_type`: 'compound_symmetry', 'ar1', or 'unstructured'
- `rho`: Correlation parameter (0 < rho < 1)
- `seed`: Random seed

**Returns:**
- `G_environment`: Environmental correlation matrix (n_environments × n_environments)

**Correlation Types:**
- **Compound Symmetry**: All environments have same correlation ρ
- **AR(1)**: Correlation decreases with distance: ρ^|i-j|
- **Unstructured**: Random positive definite matrix scaled by ρ

### Kronecker Product

**`create_kronecker_covariance(G_genotype, G_environment)`**

Computes Kronecker product of genomic and environmental matrices.

**Parameters:**
- `G_genotype`: Genomic relationship matrix (n × n)
- `G_environment`: Environmental correlation matrix (e × e)

**Returns:**
- `G_full`: Full covariance matrix (ne × ne)

**Mathematical Details:**

The Kronecker product creates a block matrix where each block corresponds to one environment:

```
G_full = [G_genotype * G_env[0,0]  G_genotype * G_env[0,1]  ...]
         [G_genotype * G_env[1,0]  G_genotype * G_env[1,1]  ...]
         [...]
```

### Optimization

**`optimize_training_set_cdmean(G_full, n_select, lambda_val=0.01, control=None, verbose=True)`**

Optimizes training set selection using CDMean criterion with genetic algorithm.

**Parameters:**
- `G_full`: Full covariance matrix
- `n_select`: Number of genotype-environment combinations to select
- `lambda_val`: Regularization parameter (default: 0.01)
- `control`: GA control parameters (optional)
- `verbose`: Print progress (default: True)

**Returns:**
- `selected_indices`: List of selected indices
- `cdmean_value`: CDMean value for selected set

### Evaluation

**`evaluate_prediction_accuracy(G_full, training_indices, test_indices, n_individuals, n_environments, lambda_val=0.01, h2=0.5, seed=None)`**

Evaluates prediction accuracy using GBLUP (Genomic Best Linear Unbiased Prediction).

**Parameters:**
- `G_full`: Full covariance matrix
- `training_indices`: Training set indices
- `test_indices`: Test set indices
- `n_individuals`: Number of individuals
- `n_environments`: Number of environments
- `lambda_val`: Regularization parameter
- `h2`: Heritability (proportion of genetic variance)
- `seed`: Random seed

**Returns:**
- Dictionary with:
  - `overall_accuracy`: Overall prediction accuracy (correlation)
  - `per_environment_accuracy`: List of accuracies per environment
  - `true_bv`: True breeding values
  - `predicted_bv_test`: Predicted breeding values for test set

## Practical Recommendations

### For Breeders

1. **Use CDMean for Training Set Design**
   - CDMean consistently outperforms random selection
   - Improvement ranges from 10-18% depending on environmental correlation

2. **Consider Environmental Correlation**
   - High correlation (ρ > 0.7): Can use fewer environments in training
   - Low correlation (ρ < 0.3): Need training samples in each environment
   - Moderate correlation (0.3 < ρ < 0.7): Balance training across environments

3. **Optimize Resource Allocation**
   - CDMean helps identify which genotype-environment combinations are most informative
   - Can reduce phenotyping costs while maintaining prediction accuracy

4. **Training Set Size**
   - Typical recommendation: 10-20% of total genotype-environment combinations
   - Adjust based on heritability and environmental correlation

### For Researchers

1. **Model Assumptions**
   - Kronecker structure assumes multiplicative G×E
   - May not fit all biological scenarios (e.g., crossover interactions)
   - Consider testing model fit before optimization

2. **Regularization**
   - Lambda parameter (λ) controls shrinkage
   - Typical range: 0.001 - 0.1
   - Higher λ for smaller training sets or lower heritability

3. **Computational Considerations**
   - Kronecker product creates large matrices (ne × ne)
   - For large problems (>1000 combinations), consider:
     - Sparse matrix implementations
     - Approximate methods
     - Parallel computing

## References

1. **VanRaden, P. M. (2008).** Efficient methods to compute genomic predictions. *Journal of Dairy Science*, 91(11), 4414-4423.

2. **Rincent, R., et al. (2012).** Maximizing the reliability of genomic selection by optimizing the calibration set of reference individuals. *Genetics*, 192(2), 715-728.

3. **Akdemir, D., et al. (2015).** Design of training populations for selective phenotyping in genomic prediction. *Scientific Reports*, 5, 11102.

4. **Jarquín, D., et al. (2014).** A reaction norm model for genomic selection using high-dimensional genomic and environmental data. *Theoretical and Applied Genetics*, 127(3), 595-607.

## Citation

If you use this example in your research, please cite:

```
TrainSelPy: A Python package for optimal training set design in genomic selection
```

## License

This example is part of TrainSelPy and is distributed under the same license.

## Contact

For questions or issues, please open an issue on the TrainSelPy GitHub repository.
