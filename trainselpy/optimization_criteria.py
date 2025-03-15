"""
Optimization criteria for TrainSelPy (fixed version).
"""

import numpy as np
from scipy.linalg import det, solve
from typing import List, Dict, Any, Union


def dopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    D-optimality criterion.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing FeatureMat
        
    Returns
    -------
    float
        D-optimality value (log determinant)
    """
    fmat = data["FeatureMat"]
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cross_prod)
    return logdet


def maximin_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Maximin criterion (maximize the minimum distance).
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing DistMat
        
    Returns
    -------
    float
        Minimum distance between any pair of selected samples
    """
    dist_mat = data["DistMat"]
    soln_dist = dist_mat.iloc[soln, soln] if hasattr(dist_mat, 'iloc') else dist_mat[soln, :][:, soln]
    
    # Extract lower triangular part (excluding diagonal)
    tri_indices = np.tril_indices(len(soln), k=-1)
    dist_values = soln_dist.values[tri_indices] if hasattr(soln_dist, 'values') else soln_dist[tri_indices]
    
    # Return the minimum distance
    return np.min(dist_values) if len(dist_values) > 0 else float('inf')


def pev_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Prediction Error Variance criterion.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing FeatureMat and Target
        
    Returns
    -------
    float
        Mean prediction error variance
    """
    fmat = data["FeatureMat"]
    targ = data["Target"]
    
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    target_features = fmat.iloc[targ, :] if hasattr(fmat, 'iloc') else fmat[targ, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Invert X'X
    inv_cross_prod = np.linalg.inv(cross_prod)
    
    # Compute PEV for targets
    pev_matrix = target_features @ inv_cross_prod @ target_features.T if hasattr(target_features, 'T') else np.dot(target_features, np.dot(inv_cross_prod, target_features.T))
    
    # Return mean PEV
    return np.mean(np.diag(pev_matrix))


def cdmean_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    CDMean criterion (Coefficient of Determination Mean).
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing G, R, and lambda
        
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
    
    # Compute V_inv_2 (the second term) - FIXED the reshape issue
    V_inv_flat = V_inv.flatten()
    V_inv_2 = np.outer(V_inv_flat, V_inv_flat).reshape(len(soln), len(soln)) / sum_V_inv
    
    # Compute the complete matrix
    outmat = G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T
    G_diag = np.diag(G_matrix)
    outmat = outmat / G_diag[:, np.newaxis]
    
    # Exclude the diagonal elements corresponding to the selected samples
    mask = np.ones(G_matrix.shape[0], dtype=bool)
    mask[soln] = False
    
    # Return the mean of the diagonal elements (excluding selected samples)
    return np.mean(np.diag(outmat)[mask])


def cdmean_opt_target(soln: List[int], data: Dict[str, Any]) -> float:
    """
    CDMean criterion with target samples.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing G, R, lambda, and Target
        
    Returns
    -------
    float
        CDMean value for target samples
    """
    G = data["G"]
    lambda_val = data["lambda"]
    targ = data["Target"]
    
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
    
    # Compute V_inv_2 (the second term) - FIXED the reshape issue
    V_inv_flat = V_inv.flatten()
    V_inv_2 = np.outer(V_inv_flat, V_inv_flat).reshape(len(soln), len(soln)) / sum_V_inv
    
    # Compute the complete matrix
    outmat = G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T
    G_diag = np.diag(G_matrix)
    outmat = outmat / G_diag[:, np.newaxis]
    
    # Return the mean of the diagonal elements for target samples
    return np.mean(np.diag(outmat)[targ])


def fun_opt_prop(soln_int, soln_dbl, data):
    """
    Optimization function for proportions.
    
    Parameters
    ----------
    soln_int : List[int]
        Integer solution (indices)
    soln_dbl : List[float]
        Double solution (proportions)
    data : Dict[str, Any]
        Data structure containing matrices
        
    Returns
    -------
    List[float]
        [Breeding value, -Inbreeding]
    """
    # Normalize proportions
    props = np.array(soln_dbl) / np.sum(soln_dbl)
    
    # Get matrices
    bv_matrix = data[0]
    inb_matrix = data[1]
    
    # Compute breeding value
    bv = bv_matrix[soln_int, :].T @ props
    
    # Compute inbreeding
    inb = props.T @ inb_matrix[soln_int, :][:, soln_int] @ props
    
    # Return both objectives
    return [bv, -inb]