"""
Test to verify the CDMean implementation is now correct.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainselpy import make_data
from trainselpy.optimization_criteria import cdmean_opt, cdmean_opt_target


def test_cdmean_correct_implementation():
    """
    Verify that the CDMean implementation now produces non-zero values
    and matches the correct mathematical formula.
    """
    # Create a simple test case
    n_samples = 20
    n_features = 10
    
    np.random.seed(42)
    M = np.random.choice([-1, 0, 1], size=(n_samples, n_features), p=[0.25, 0.5, 0.25])
    K = np.dot(M, M.T) / n_features
    K += np.eye(n_samples) * 1e-6
    
    # Create data
    data = make_data(K=K)
    
    # Select a subset
    soln = [5, 6, 7, 8, 9]
    
    # Compute CDMean
    cdmean_value = cdmean_opt(soln, data)
    
    print("CDMean Correctness Test")
    print("=" * 60)
    print(f"Selected samples: {soln}")
    print(f"CDMean value: {cdmean_value}")
    print(f"Is CDMean non-zero? {abs(cdmean_value) > 1e-10}")
    print(f"Is CDMean finite? {np.isfinite(cdmean_value)}")
    
    # Verify manually
    G = data["G"]
    # Support both legacy 'lambda' and newer 'lambda_val' keys
    lambda_val = data.get("lambda", data.get("lambda_val"))
    
    if hasattr(G, 'values'):
        G_matrix = G.values
    else:
        G_matrix = G
    
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Compute V_inv
    V_inv = np.linalg.inv(G_soln_soln + lambda_val * np.eye(len(soln)))
    
    # Correct formula
    ones = np.ones(len(soln))
    V_inv_1 = V_inv @ ones
    sum_V_inv = ones @ V_inv_1
    V_inv_2 = np.outer(V_inv_1, V_inv_1) / sum_V_inv
    
    # Verify V_inv - V_inv_2 is NOT zero
    diff = V_inv - V_inv_2
    max_diff = np.max(np.abs(diff))
    
    print(f"\nMax |V_inv - V_inv_2|: {max_diff}")
    print(f"Is (V_inv - V_inv_2) non-zero? {max_diff > 1e-10}")
    
    # Compute CDMean manually
    outmat = G_all_soln @ diff @ G_all_soln.T
    G_diag = np.diag(G_matrix)
    outmat = outmat / G_diag[:, np.newaxis]
    
    mask = np.ones(G_matrix.shape[0], dtype=bool)
    mask[soln] = False
    
    cdmean_manual = np.mean(np.diag(outmat)[mask])
    
    print(f"\nCDMean (from function): {cdmean_value}")
    print(f"CDMean (manual calculation): {cdmean_manual}")
    print(f"Difference: {abs(cdmean_value - cdmean_manual)}")
    print(f"Match? {np.isclose(cdmean_value, cdmean_manual)}")
    
    # Assertions
    assert abs(cdmean_value) > 1e-10, "CDMean should be non-zero"
    assert np.isfinite(cdmean_value), "CDMean should be finite"
    assert np.isclose(cdmean_value, cdmean_manual), "CDMean should match manual calculation"
    assert max_diff > 1e-10, "(V_inv - V_inv_2) should be non-zero"
    
    print("\n" + "=" * 60)
    print("✅ CDMean implementation is CORRECT!")
    print("=" * 60)


def test_cdmean_target_correct():
    """
    Verify that the CDMean with target implementation is also correct.
    """
    # Create a simple test case
    n_samples = 20
    n_features = 10
    
    np.random.seed(42)
    M = np.random.choice([-1, 0, 1], size=(n_samples, n_features), p=[0.25, 0.5, 0.25])
    K = np.dot(M, M.T) / n_features
    K += np.eye(n_samples) * 1e-6
    
    # Create data
    data = make_data(K=K)
    data["Target"] = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]  # Some target samples
    
    # Select a subset
    soln = [5, 6, 7, 8, 9]
    
    # Compute CDMean with target
    cdmean_value = cdmean_opt_target(soln, data)
    
    print("\nCDMean Target Correctness Test")
    print("=" * 60)
    print(f"Selected samples: {soln}")
    print(f"Target samples: {data['Target']}")
    print(f"CDMean target value: {cdmean_value}")
    print(f"Is CDMean target non-zero? {abs(cdmean_value) > 1e-10}")
    print(f"Is CDMean target finite? {np.isfinite(cdmean_value)}")
    
    # Assertions
    assert abs(cdmean_value) > 1e-10, "CDMean target should be non-zero"
    assert np.isfinite(cdmean_value), "CDMean target should be finite"
    
    print("\n" + "=" * 60)
    print("✅ CDMean target implementation is CORRECT!")
    print("=" * 60)


if __name__ == "__main__":
    test_cdmean_correct_implementation()
    test_cdmean_target_correct()
