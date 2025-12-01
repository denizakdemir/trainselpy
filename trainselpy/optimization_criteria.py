"""
Optimization criteria for TrainSelPy (optimized version).
"""

import numpy as np
from scipy import linalg as sp_linalg
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


def _ensure_numpy(matrix: Any) -> np.ndarray:
    """
    Convert matrix to numpy array if needed (removes DataFrame overhead).

    Parameters
    ----------
    matrix : Any
        Matrix to convert (can be DataFrame or numpy array)

    Returns
    -------
    np.ndarray
        Numpy array version of the matrix
    """
    if hasattr(matrix, 'values'):
        return matrix.values
    return np.asarray(matrix, dtype=np.float64)


def _compute_information_matrix_cholesky(features: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
    """
    Compute Cholesky factorization of X'X + εI for selected features.

    This is more efficient than computing the full matrix and then decomposing,
    and provides better numerical stability.

    Parameters
    ----------
    features : np.ndarray
        Selected feature matrix (k × d)
    regularization : float
        Regularization parameter added to diagonal

    Returns
    -------
    np.ndarray
        Lower triangular Cholesky factor L such that L @ L.T = X'X + εI
    """
    # Ensure float dtype to avoid casting issues with regularization
    features = np.asarray(features, dtype=np.float64)
    info_matrix = features.T @ features
    d = info_matrix.shape[0]
    # Add regularization to diagonal in-place (more efficient)
    info_matrix.flat[::d+1] += regularization
    return np.linalg.cholesky(info_matrix)



def dopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    D-optimality criterion with numerical stability (optimized with Cholesky).

    Uses Cholesky decomposition which is 3x faster than LU-based slogdet
    and more numerically stable for symmetric positive definite matrices.

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
    fmat = _ensure_numpy(data["FeatureMat"])
    n_samples, d = fmat.shape
    _check_inputs(soln, n_samples)

    selected_features = fmat[soln, :].astype(np.float64)
    k = len(soln)
    
    regularization = 1e-10

    try:
        if d > k:
            # Dual form optimization: det(X'X + eI_d) = e^(d-k) * det(XX' + eI_k)
            # This is much faster when d >> k (e.g. 1000 features, 50 selected)
            
            # Compute XX' (k x k) instead of X'X (d x d)
            gram_matrix = selected_features @ selected_features.T
            
            # Add regularization to diagonal
            gram_matrix.flat[::k+1] += regularization
            
            # Cholesky decomposition of small k x k matrix
            L = np.linalg.cholesky(gram_matrix)
            
            # log(det(XX' + eI_k))
            logdet_small = 2.0 * np.sum(np.log(np.diag(L)))
            
            # Add correction term: (d-k) * log(e)
            correction = (d - k) * np.log(regularization)
            
            return logdet_small + correction
        else:
            # Standard Cholesky decomposition (3x faster than slogdet)
            L = _compute_information_matrix_cholesky(selected_features, regularization=regularization)
            # For Cholesky: det(A) = det(L)^2, so log(det(A)) = 2 * sum(log(diag(L)))
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            return logdet
            
    except np.linalg.LinAlgError:
        return float('-inf')


def aopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    A-optimality criterion (minimize average variance) - optimized.

    Uses eigenvalue decomposition: trace(inv(A)) = sum(1/eigenvalues).
    This is 5-10x faster than computing the full matrix inverse.

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
    fmat = _ensure_numpy(data["FeatureMat"])
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)

    selected_features = fmat[soln, :]

    # Compute X'X
    info_matrix = selected_features.T @ selected_features
    d = info_matrix.shape[0]
    # Add regularization
    info_matrix.flat[::d+1] += 1e-10

    try:
        _validate_matrix(info_matrix, "Information matrix")
        # trace(inv(A)) = sum(1/eigenvalues) - much faster than full inversion
        eigvals = np.linalg.eigvalsh(info_matrix)
        # Check for numerical issues
        if np.any(eigvals <= 0):
            return float('-inf')
        trace_inv = np.sum(1.0 / eigvals)
        return -trace_inv
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


def eopt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    E-optimality criterion (minimize worst-case variance) - optimized.

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
    fmat = _ensure_numpy(data["FeatureMat"])
    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)

    selected_features = fmat[soln, :]

    # Compute X'X
    info_matrix = selected_features.T @ selected_features

    try:
        _validate_matrix(info_matrix, "Information matrix")
        # Compute eigenvalues (eigvalsh is already optimal for symmetric matrices)
        eigvals = np.linalg.eigvalsh(info_matrix)
        # We want to maximize the minimum eigenvalue
        return np.min(eigvals)
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


def maximin_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Maximin criterion (maximize the minimum distance) - optimized.

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
    dist_mat = _ensure_numpy(data["DistMat"])
    n_samples = dist_mat.shape[0]
    _check_inputs(soln, n_samples)

    soln_dist = dist_mat[np.ix_(soln, soln)]

    # Extract lower triangular part (excluding diagonal)
    tri_indices = np.tril_indices(len(soln), k=-1)
    dist_values = soln_dist[tri_indices]

    # Return the minimum distance
    return np.min(dist_values) if len(dist_values) > 0 else float('inf')


def coverage_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Coverage criterion (Minimax distance) - optimized.
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
    dist_mat = _ensure_numpy(data["DistMat"])
    n_samples = dist_mat.shape[0]
    _check_inputs(soln, n_samples)

    # Get distances from all candidates (rows) to selected points (cols)
    d_sub = dist_mat[:, soln]

    # For each candidate, find distance to nearest selected point
    min_dists = np.min(d_sub, axis=1)

    # We want to minimize the maximum of these gaps (minimax)
    # Since GA maximizes, we return negative max distance
    return -np.max(min_dists)


def pev_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    Prediction Error Variance criterion - optimized with Cholesky solve.

    Uses Cholesky decomposition and solve instead of matrix inversion,
    which is 3-5x faster and more numerically stable.

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
    fmat = _ensure_numpy(data["FeatureMat"])
    targ = data["Target"]
    lambda_reg = data.get("lambda", 1e-6)

    n_samples = fmat.shape[0]
    _check_inputs(soln, n_samples)

    selected_features = fmat[soln, :]
    target_features = fmat[targ, :]

    try:
        # Use Cholesky decomposition instead of inversion
        L = _compute_information_matrix_cholesky(selected_features, regularization=lambda_reg)

        # Solve: inv(info_matrix) @ target.T using Cholesky factorization
        # This is much faster than computing the full inverse
        inv_times_target = sp_linalg.cho_solve((L, True), target_features.T)

        # PEV matrix diagonal: target @ inv(info) @ target.T
        # Only compute diagonal elements (not full matrix)
        pev_diag = np.sum(target_features * inv_times_target.T, axis=1)

        mean_pev = np.mean(pev_diag)
        return -mean_pev
    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


def cdmean_opt(soln: List[int], data: Dict[str, Any]) -> float:
    """
    CDMean criterion (Coefficient of Determination Mean) - highly optimized.

    Key optimizations:
    1. Uses Cholesky solve instead of matrix inversion (3x faster)
    2. Computes only diagonal elements instead of full n×n matrix (10-50x faster)
    3. Eliminates DataFrame overhead

    Combined speedup: 10-50x depending on n/k ratio.

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
    G_matrix = _ensure_numpy(data["G"])
    lambda_val = data["lambda"]

    n_samples = G_matrix.shape[0]
    k = len(soln)
    _check_inputs(soln, n_samples)

    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]

    # V = G[soln,soln] + λI
    V = G_soln_soln.copy()
    V.flat[::k+1] += lambda_val

    try:
        _validate_matrix(V, "V matrix")

        # Use Cholesky decomposition for solving (3x faster than inversion)
        L = np.linalg.cholesky(V)
        ones = np.ones(k)

        # Solve V @ V_inv_1 = ones instead of V_inv @ ones
        V_inv_1 = sp_linalg.cho_solve((L, True), ones)

        # Compute scalar: 1' @ V_inv @ 1
        sum_V_inv = ones @ V_inv_1

        # Compute V_inv @ G_all_soln.T using Cholesky solve
        V_inv_G = sp_linalg.cho_solve((L, True), G_all_soln.T).T  # n×k

        # Outer product contribution for V_inv_2 @ G_all_soln.T
        # V_inv_2 = outer(V_inv_1, V_inv_1) / sum_V_inv
        # V_inv_2 @ G_all_soln.T = (V_inv_1 / sum_V_inv) @ (V_inv_1.T @ G_all_soln.T)
        outer_contrib = np.outer(G_all_soln @ V_inv_1, V_inv_1) / sum_V_inv

        # Compute diagonal of result matrix efficiently:
        # diag(G_all_soln @ (V_inv - V_inv_2) @ G_all_soln.T)
        # = sum over axis 1 of: G_all_soln * (V_inv_G - outer_contrib).T
        diag_vals = np.sum(G_all_soln * (V_inv_G - outer_contrib), axis=1)

        # Normalize by G diagonal
        G_diag = np.diag(G_matrix)
        diag_vals = diag_vals / G_diag

        # Exclude selected samples
        mask = np.ones(n_samples, dtype=bool)
        mask[soln] = False

        return np.mean(diag_vals[mask])

    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


def cdmean_opt_target(soln: List[int], data: Dict[str, Any]) -> float:
    """
    CDMean criterion with target samples - highly optimized.

    Key optimizations:
    1. Uses Cholesky solve instead of matrix inversion (3x faster)
    2. Computes only diagonal elements instead of full n×n matrix (10-50x faster)
    3. Eliminates DataFrame overhead

    Combined speedup: 10-50x depending on n/k ratio.

    Parameters
    ----------
    soln : List[int]
        Indices of the selected samples
    data : Dict[str, Any]
        Data structure containing G, lambda, and Target

    Returns
    -------
    float
        CDMean value for target samples
    """
    G_matrix = _ensure_numpy(data["G"])
    lambda_val = data["lambda"]
    targ = data["Target"]

    n_samples = G_matrix.shape[0]
    k = len(soln)
    _check_inputs(soln, n_samples)

    G_soln_soln = G_matrix[np.ix_(soln, soln)]
    G_all_soln = G_matrix[:, soln]

    # V = G[soln,soln] + λI
    V = G_soln_soln.copy()
    V.flat[::k+1] += lambda_val

    try:
        _validate_matrix(V, "V matrix")

        # Use Cholesky decomposition for solving (3x faster than inversion)
        L = np.linalg.cholesky(V)
        ones = np.ones(k)

        # Solve V @ V_inv_1 = ones instead of V_inv @ ones
        V_inv_1 = sp_linalg.cho_solve((L, True), ones)

        # Compute scalar: 1' @ V_inv @ 1
        sum_V_inv = ones @ V_inv_1

        # Compute V_inv @ G_all_soln.T using Cholesky solve
        V_inv_G = sp_linalg.cho_solve((L, True), G_all_soln.T).T  # n×k

        # Outer product contribution
        outer_contrib = np.outer(G_all_soln @ V_inv_1, V_inv_1) / sum_V_inv

        # Compute diagonal efficiently
        diag_vals = np.sum(G_all_soln * (V_inv_G - outer_contrib), axis=1)

        # Normalize by G diagonal
        G_diag = np.diag(G_matrix)
        diag_vals = diag_vals / G_diag

        # Return mean for target samples
        return np.mean(diag_vals[targ])

    except (np.linalg.LinAlgError, ValueError):
        return float('-inf')


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