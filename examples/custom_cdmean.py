"""
Custom CDMean implementation to fix the reshape issue.
"""

import numpy as np
from typing import List, Dict, Any

def custom_cdmean(soln: List[int], data: Dict[str, Any]) -> float:
    """
    CDMean criterion (Coefficient of Determination Mean) with fixed reshape.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing G and lambda
        
    Returns
    -------
    float
        CDMean value
    """
    G = data["G"]
    lambda_val = data.get("lambda_val", data.get("lambda", 1))
    
    # Get G matrices - handle both DataFrame and ndarray
    if hasattr(G, 'iloc'):
        G_matrix = G.values  # Convert to numpy array if it's a DataFrame
    else:
        G_matrix = G
        
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Add lambda to diagonal
    V_inv = np.linalg.inv(G_soln_soln + lambda_val * np.eye(len(soln)))
    
    # Compute the sum vector: V_inv @ 1 (where 1 is a vector of ones)
    ones = np.ones(len(soln))
    V_inv_1 = V_inv @ ones
    
    # Compute the scalar: 1' @ V_inv @ 1 (total sum of V_inv)
    sum_V_inv = ones @ V_inv_1
    
    # Compute V_inv_2 as the outer product of the sum vector divided by the total sum
    # V_inv_2 = (V_inv @ 1)(V_inv @ 1)' / (1' @ V_inv @ 1)
    V_inv_2 = np.outer(V_inv_1, V_inv_1) / sum_V_inv

    
    # Compute V_inv_G = V_inv @ G_all_soln.T
    V_inv_G = V_inv @ G_all_soln.T
    
    # Compute diagonal efficiently:
    # diag(G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T)
    # = sum over axis 1 of: G_all_soln * (V_inv_G - outer_contrib).T
    # where outer_contrib = V_inv_2 @ G_all_soln.T
    
    # Precompute V_inv_2 @ G_all_soln.T
    # V_inv_2 = (V_inv_1 @ V_inv_1.T) / sum_V_inv
    # So V_inv_2 @ G_all_soln.T = (V_inv_1 / sum_V_inv) * (V_inv_1.T @ G_all_soln.T)
    # Note: V_inv_1 is (k,), G_all_soln.T is (k, N)
    
    term1 = V_inv_1 / sum_V_inv
    term2 = V_inv_1 @ G_all_soln.T
    outer_contrib_T = np.outer(term1, term2) # (k, N)
    
    # Now compute diagonal elements
    # (V_inv_G - outer_contrib_T).T is (N, k)
    # G_all_soln is (N, k)
    # Element-wise multiply and sum across columns (axis 1)
    
    diff_T = V_inv_G - outer_contrib_T
    diag_vals = np.sum(G_all_soln * diff_T.T, axis=1)
    
    # Normalize by G diagonal
    G_diag = np.diag(G_matrix)
    diag_vals = diag_vals / G_diag
    
    # Exclude the diagonal elements corresponding to the selected samples
    mask = np.ones(G_matrix.shape[0], dtype=bool)
    mask[soln] = False
    
    # Return the mean of the diagonal elements (excluding selected samples)
    return np.mean(diag_vals[mask])