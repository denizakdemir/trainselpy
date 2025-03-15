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
    lambda_val = data["lambda"]
    
    # Get G matrices - handle both DataFrame and ndarray
    if hasattr(G, 'iloc'):
        G_matrix = G.values  # Convert to numpy array if it's a DataFrame
    else:
        G_matrix = G
        
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Add lambda to diagonal
    V_inv = np.linalg.inv(G_soln_soln + lambda_val * np.eye(len(soln)))
    
    # Compute sum of V_inv
    sum_V_inv = np.sum(V_inv)
    
    # Compute V_inv_2 (the second term) - Using a different approach to avoid reshape
    # Using matrix outer product directly
    V_inv_2 = np.zeros((len(soln), len(soln)))
    for i in range(len(soln)):
        for j in range(len(soln)):
            V_inv_2[i, j] = V_inv.sum() * (V_inv[i, j] / sum_V_inv)
    
    # Compute the complete matrix
    outmat = G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T
    G_diag = np.diag(G_matrix)
    outmat = outmat / G_diag[:, np.newaxis]
    
    # Exclude the diagonal elements corresponding to the selected samples
    mask = np.ones(G_matrix.shape[0], dtype=bool)
    mask[soln] = False
    
    # Return the mean of the diagonal elements (excluding selected samples)
    return np.mean(np.diag(outmat)[mask])