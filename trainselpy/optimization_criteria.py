"""
Optimization criteria for TrainSelPy (fixed version).
"""

import numpy as np
from scipy.linalg import det, solve
from typing import List, Dict, Any, Union, Optional


def _check_inputs(soln: List[int], n_samples: int) -> None:
    """
    Validate solution indices.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    n_samples : int
        Total number of samples available
        
    Raises
    ------
    ValueError
        If solution is empty or contains invalid indices
    """
    if not soln:
        raise ValueError("Solution set is empty")
    
    # Convert to array for efficient checking
    soln_arr = np.array(soln)
    if np.any((soln_arr < 0) | (soln_arr >= n_samples)):
        raise ValueError(f"Solution contains invalid indices. Must be between 0 and {n_samples-1}")


def _validate_matrix(matrix: np.ndarray, name: str = "Matrix") -> None:
    """
    Check for NaN or Inf values in matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    name : str
        Name of the matrix for error message
        
    Raises
    ------
    ValueError
        If matrix contains NaN or Inf values
    """
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or Inf values")



def dopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    D-optimality criterion with numerical stability.
    
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
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)
    
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Convert to float to avoid type casting issues
    cross_prod = np.asarray(cross_prod, dtype=np.float64)
    
    # Validate matrix
    try:
        _validate_matrix(cross_prod, "Cross product matrix")
    except ValueError:
        return float('-inf')
    
    # Add small regularization for numerical stability
    n_features = cross_prod.shape[0]
    epsilon = 1e-10
    cross_prod = cross_prod + epsilon * np.eye(n_features)
    
    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cross_prod)
    
    # Check for numerical issues
    if sign <= 0 or not np.isfinite(logdet):
        return float('-inf')
    
    return logdet


def aopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    A-optimality criterion (minimize average variance).
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing FeatureMat
        
    Returns
    -------
    float
        Negative trace of the inverse information matrix (maximization)
    """
    fmat = data["FeatureMat"]
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)
    
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Convert to float
    cross_prod = np.asarray(cross_prod, dtype=np.float64)
    
    # Add regularization
    n_features = cross_prod.shape[0]
    epsilon = 1e-10
    cross_prod_reg = cross_prod + epsilon * np.eye(n_features)
    
    try:
        _validate_matrix(cross_prod_reg, "Regularized cross product")
        inv_cross_prod = np.linalg.inv(cross_prod_reg)
        # We want to minimize trace(inv(X'X)), so maximize -trace
        return -np.trace(inv_cross_prod)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


def eopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    E-optimality criterion (minimize worst-case variance).
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing FeatureMat
        
    Returns
    -------
    float
        Minimum eigenvalue of the information matrix (maximization)
    """
    fmat = data["FeatureMat"]
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)
    
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Convert to float
    cross_prod = np.asarray(cross_prod, dtype=np.float64)
    
    try:
        _validate_matrix(cross_prod, "Cross product matrix")
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(cross_prod)
        # We want to maximize the minimum eigenvalue (which minimizes the maximum eigenvalue of the inverse)
        return np.min(eigvals)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


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
    n_samples = dist_mat.shape[0]
    _check_inputs(soln, n_samples)
    
    soln_dist = dist_mat.iloc[soln, soln] if hasattr(dist_mat, 'iloc') else dist_mat[soln, :][:, soln]
    
    # Extract lower triangular part (excluding diagonal)
    tri_indices = np.tril_indices(len(soln), k=-1)
    dist_values = soln_dist.values[tri_indices] if hasattr(soln_dist, 'values') else soln_dist[tri_indices]
    
    # Return the minimum distance
    return np.min(dist_values) if len(dist_values) > 0 else float('inf')


def coverage_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Coverage criterion (Minimax distance).
    Ensures every candidate point is close to at least one selected point.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing DistMat
        
    Returns
    -------
    float
        Negative maximum distance from any candidate to nearest selected (maximization)
    """
    dist_mat = data["DistMat"]
    n_samples = dist_mat.shape[0]
    _check_inputs(soln, n_samples)
    
    # Get distances from all candidates (rows) to selected points (cols)
    # dist_mat is assumed to be square (N x N)
    if hasattr(dist_mat, 'iloc'):
        d_sub = dist_mat.iloc[:, soln]
    else:
        d_sub = dist_mat[:, soln]
        
    # For each candidate, find distance to nearest selected point
    if hasattr(d_sub, 'values'):
        min_dists = np.min(d_sub.values, axis=1)
    else:
        min_dists = np.min(d_sub, axis=1)
        
    # We want to minimize the maximum of these gaps (minimax)
    # Since GA maximizes, we return negative max distance
    return -np.max(min_dists)


def pev_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Prediction Error Variance criterion with numerical stability.
    
    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing FeatureMat and Target
        
    Returns
    -------
    float
        Mean prediction error variance (negative for maximization)
    """
    fmat = data["FeatureMat"]
    targ = data["Target"]
    
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)
    
    selected_features = fmat.iloc[soln, :] if hasattr(fmat, 'iloc') else fmat[soln, :]
    target_features = fmat.iloc[targ, :] if hasattr(fmat, 'iloc') else fmat[targ, :]
    
    # Compute X'X (cross product)
    cross_prod = selected_features.T @ selected_features if hasattr(selected_features, 'T') else np.dot(selected_features.T, selected_features)
    
    # Add regularization for numerical stability
    lambda_reg = data.get("lambda", 1e-6)
    n_features = cross_prod.shape[0]
    cross_prod_reg = cross_prod + lambda_reg * np.eye(n_features)
    
    # Invert X'X with error handling
    try:
        _validate_matrix(cross_prod_reg, "Regularized cross product")
        inv_cross_prod = np.linalg.inv(cross_prod_reg)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')  # Return worst possible PEV if inversion fails
    
    # Compute PEV for targets
    pev_matrix = target_features @ inv_cross_prod @ target_features.T if hasattr(target_features, 'T') else np.dot(target_features, np.dot(inv_cross_prod, target_features.T))
    
    # Return mean PEV (we want to minimize this, so return positive value)
    mean_pev = np.mean(np.diag(pev_matrix))
    
    # Return negative for maximization (GA maximizes fitness)
    return -mean_pev


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
        
    n_samples = G_matrix.shape[0]
    _check_inputs(soln, n_samples)
        
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Add lambda to diagonal
    V = G_soln_soln + lambda_val * np.eye(len(soln))
    
    try:
        _validate_matrix(V, "V matrix")
        V_inv = np.linalg.inv(V)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')
    
    # Compute the sum vector: V_inv @ 1 (where 1 is a vector of ones)
    ones = np.ones(len(soln))
    V_inv_1 = V_inv @ ones
    
    # Compute the scalar: 1' @ V_inv @ 1 (total sum of V_inv)
    sum_V_inv = ones @ V_inv_1
    
    # Compute V_inv_2 as the outer product of the sum vector divided by the total sum
    # V_inv_2 = (V_inv @ 1)(V_inv @ 1)' / (1' @ V_inv @ 1)
    V_inv_2 = np.outer(V_inv_1, V_inv_1) / sum_V_inv
    
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
        
    n_samples = G_matrix.shape[0]
    _check_inputs(soln, n_samples)
        
    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]
    
    # Add lambda to diagonal
    V = G_soln_soln + lambda_val * np.eye(len(soln))
    
    try:
        _validate_matrix(V, "V matrix")
        V_inv = np.linalg.inv(V)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')
    
    # Compute the sum vector: V_inv @ 1 (where 1 is a vector of ones)
    ones = np.ones(len(soln))
    V_inv_1 = V_inv @ ones
    
    # Compute the scalar: 1' @ V_inv @ 1 (total sum of V_inv)
    sum_V_inv = ones @ V_inv_1
    
    # Compute V_inv_2 as the outer product of the sum vector divided by the total sum
    # V_inv_2 = (V_inv @ 1)(V_inv @ 1)' / (1' @ V_inv @ 1)
    V_inv_2 = np.outer(V_inv_1, V_inv_1) / sum_V_inv
    
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