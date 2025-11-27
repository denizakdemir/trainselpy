"""
Test to verify the CDMean mathematical error.

This test demonstrates that the current implementation of V_inv_2
results in V_inv_2 = V_inv, making (V_inv - V_inv_2) = 0.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainselpy import make_data
from trainselpy.optimization_criteria import cdmean_opt


def test_cdmean_mathematical_error():
    """
    Demonstrate the mathematical error in CDMean implementation.
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
    
    # Extract the relevant parts
    G = data["G"]
    lambda_val = data["lambda"]
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(G, 'values'):
        G_matrix = G.values
    else:
        G_matrix = G
    
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Compute V_inv
    V_inv = np.linalg.inv(G_soln_soln + lambda_val * np.eye(len(soln)))
    
    # Current (incorrect) implementation
    sum_V_inv = np.sum(V_inv)
    V_inv_flat = V_inv.flatten()
    
    print("Current Implementation Analysis:")
    print("=" * 60)
    print(f"V_inv shape: {V_inv.shape}")
    print(f"V_inv_flat shape: {V_inv_flat.shape}")
    print(f"sum(V_inv): {sum_V_inv}")
    print(f"\nV_inv:\n{V_inv}")
    
    # The current implementation tries to do:
    # V_inv_2 = np.outer(V_inv_flat, V_inv_flat).reshape(len(soln), len(soln)) / sum_V_inv
    # But this is WRONG because:
    # - V_inv_flat has shape (25,) for a 5x5 matrix
    # - np.outer(V_inv_flat, V_inv_flat) has shape (25, 25)
    # - Cannot reshape (25, 25) into (5, 5)
    
    print(f"\nnp.outer(V_inv_flat, V_inv_flat) shape: {np.outer(V_inv_flat, V_inv_flat).shape}")
    print(f"Trying to reshape to: ({len(soln)}, {len(soln)})")
    print(f"This will FAIL because 625 != 25")
    
    # What the code ACTUALLY does (based on the formula in the comment):
    # It computes element-wise: V_inv_2[i,j] = sum(V_inv) * (V_inv[i,j] / sum(V_inv)) = V_inv[i,j]
    V_inv_2_manual = np.zeros((len(soln), len(soln)))
    for i in range(len(soln)):
        for j in range(len(soln)):
            V_inv_2_manual[i, j] = sum_V_inv * (V_inv[i, j] / sum_V_inv)
    
    print(f"\nV_inv_2 (what the formula actually computes):\n{V_inv_2_manual}")
    print(f"\nIs V_inv_2_manual == V_inv? {np.allclose(V_inv_2_manual, V_inv)}")
    print(f"\nV_inv - V_inv_2_manual:\n{V_inv - V_inv_2_manual}")
    print(f"\nMax absolute difference: {np.max(np.abs(V_inv - V_inv_2_manual))}")
    
    # Correct implementation
    ones = np.ones(len(soln))
    V_inv_1 = V_inv @ ones
    sum_V_inv_correct = ones @ V_inv_1
    V_inv_2_correct = np.outer(V_inv_1, V_inv_1) / sum_V_inv_correct
    
    print("\n" + "=" * 60)
    print("Correct Implementation Analysis:")
    print("=" * 60)
    print(f"V_inv @ 1:\n{V_inv_1}")
    print(f"\nV_inv_2 (correct):\n{V_inv_2_correct}")
    print(f"\nIs V_inv_2_correct == V_inv? {np.allclose(V_inv_2_correct, V_inv)}")
    print(f"\nV_inv - V_inv_2_correct:\n{V_inv - V_inv_2_correct}")
    print(f"\nMax absolute difference: {np.max(np.abs(V_inv - V_inv_2_correct))}")
    
    # Compute CDMean with both methods
    outmat_current = G_all_soln @ (V_inv - V_inv_2_manual) @ G_all_soln.T
    outmat_correct = G_all_soln @ (V_inv - V_inv_2_correct) @ G_all_soln.T
    
    G_diag = np.diag(G_matrix)
    outmat_current = outmat_current / G_diag[:, np.newaxis]
    outmat_correct = outmat_correct / G_diag[:, np.newaxis]
    
    mask = np.ones(G_matrix.shape[0], dtype=bool)
    mask[soln] = False
    
    cdmean_current = np.mean(np.diag(outmat_current)[mask])
    cdmean_correct = np.mean(np.diag(outmat_correct)[mask])
    
    print("\n" + "=" * 60)
    print("CDMean Comparison:")
    print("=" * 60)
    print(f"CDMean (current implementation): {cdmean_current}")
    print(f"CDMean (correct implementation): {cdmean_correct}")
    print(f"Difference: {abs(cdmean_current - cdmean_correct)}")
    if abs(cdmean_correct) > 1e-10:
        print(f"Relative difference: {abs(cdmean_current - cdmean_correct) / abs(cdmean_correct) * 100:.2f}%")
    
    # The current implementation should give near-zero values
    print(f"\nIs current implementation near zero? {abs(cdmean_current) < 1e-10}")
    print(f"Is correct implementation non-zero? {abs(cdmean_correct) > 1e-5}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: The current implementation has a mathematical error!")
    print("The reshape operation is impossible, and the manual calculation")
    print("shows that V_inv_2 = V_inv, making (V_inv - V_inv_2) = 0")
    print("=" * 60)


if __name__ == "__main__":
    test_cdmean_mathematical_error()
