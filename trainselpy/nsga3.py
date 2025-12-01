"""
NSGA-III implementation for TrainSelPy.
Reference: Deb, K., & Jain, H. (2013). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Transactions on Evolutionary Computation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from itertools import combinations_with_replacement

from trainselpy.solution import Solution

def generate_reference_points(M: int, p: int) -> np.ndarray:
    """
    Generate reference points for NSGA-III using Das and Dennis's method.
    
    Parameters
    ----------
    M : int
        Number of objectives
    p : int
        Number of divisions
        
    Returns
    -------
    np.ndarray
        Reference points (N x M)
    """
    # Generate all combinations of M integers that sum to p
    # This corresponds to partitioning p items into M bins
    # We use combinations_with_replacement(range(p+1), M-1) to find dividers
    
    # Alternative approach: recursion
    def get_points(m_left, p_left):
        if m_left == 1:
            return [[p_left]]
        points = []
        for i in range(p_left + 1):
            for sub_point in get_points(m_left - 1, p_left - i):
                points.append([i] + sub_point)
        return points

    points = np.array(get_points(M, p))
    return points / p


def normalize_objectives(population: List[Solution], intercepts: np.ndarray, ideal_point: np.ndarray) -> np.ndarray:
    """
    Normalize objective values.
    
    Parameters
    ----------
    population : List[Solution]
        Population
    intercepts : np.ndarray
        Intercepts of the hyperplane
    ideal_point : np.ndarray
        Ideal point (min values)
        
    Returns
    -------
    np.ndarray
        Normalized objective values
    """
    # Extract objectives
    objs = np.array([sol.multi_fitness for sol in population])
    
    # Shift by ideal point
    objs = objs - ideal_point
    
    # Normalize by intercepts
    # Avoid division by zero
    denom = intercepts - ideal_point
    denom[denom < 1e-6] = 1e-6
    
    return objs / denom


def associate_to_reference_points(
    normalized_objs: np.ndarray,
    reference_points: np.ndarray
) -> Tuple[List[int], List[float]]:
    """
    Associate each solution to the nearest reference point.
    
    Parameters
    ----------
    normalized_objs : np.ndarray
        Normalized objective values (N x M)
    reference_points : np.ndarray
        Reference points (K x M)
        
    Returns
    -------
    Tuple[List[int], List[float]]
        Indices of nearest reference point for each solution, and distances
    """
    n_sol = normalized_objs.shape[0]
    n_ref = reference_points.shape[0]
    
    association = []
    distances = []
    
    for i in range(n_sol):
        f = normalized_objs[i]
        
        # Calculate perpendicular distance to each reference line
        # d(f, w) = || f - (w'f / ||w||^2) * w ||
        
        # Vectorized calculation
        # w_norm_sq = sum(w^2)
        w_norm_sq = np.sum(reference_points**2, axis=1)
        
        # dot_prod = sum(f * w)
        dot_prod = np.dot(reference_points, f)
        
        # Projection length along w
        proj_len = dot_prod / w_norm_sq
        
        # Projection vector
        proj = reference_points * proj_len[:, np.newaxis]
        
        # Distance vector
        dist_vec = f - proj
        
        # Perpendicular distance
        dists = np.linalg.norm(dist_vec, axis=1)
        
        # Find nearest
        min_idx = np.argmin(dists)
        association.append(min_idx)
        distances.append(dists[min_idx])
        
    return association, distances


def niche_preservation(
    population: List[Solution],
    n_remaining: int,
    reference_points: np.ndarray,
    ideal_point: np.ndarray,
    intercepts: np.ndarray
) -> List[Solution]:
    """
    Select solutions using niche preservation logic.
    
    Parameters
    ----------
    population : List[Solution]
        Combined population (last front)
    n_remaining : int
        Number of solutions to select
    reference_points : np.ndarray
        Reference points
    ideal_point : np.ndarray
        Ideal point
    intercepts : np.ndarray
        Intercepts
        
    Returns
    -------
    List[Solution]
        Selected solutions
    """
    # Normalize objectives
    norm_objs = normalize_objectives(population, intercepts, ideal_point)
    
    # Associate to reference points
    association, distances = associate_to_reference_points(norm_objs, reference_points)
    
    # Count niche counts (rho)
    # Note: This assumes we are selecting from the LAST front.
    # But NSGA-III considers niche counts from ALL selected fronts (P_t+1).
    # The function signature implies we only pass the candidates from the last front.
    # We need the counts from already selected solutions (if any).
    # For simplicity, let's assume 'population' contains ONLY the last front candidates,
    # and we start with 0 counts.
    # BUT, correct NSGA-III needs to account for solutions already selected in previous fronts.
    # So we should pass 'selected_so_far' or similar.
    
    # Simplified logic: Just select from this front based on counts within this front?
    # No, that defeats the purpose.
    # We need to pass the full set of "already selected" solutions to compute rho correctly.
    
    # Let's assume this function is called with the *entire* set of candidates (previous fronts + current front)
    # and we need to select K total solutions.
    # But usually NSGA-III is called when |St| > N. St = F1 + F2 + ... + Fl.
    # We want to select N solutions.
    
    # Let's change signature to take 'fronts' and 'n_needed'.
    pass

def nsga3_selection(
    population: List[Solution],
    n_select: int,
    reference_points: np.ndarray
) -> List[Solution]:
    """
    Perform NSGA-III selection.
    
    Parameters
    ----------
    population : List[Solution]
        Population to select from
    n_select : int
        Number of solutions to select
    reference_points : np.ndarray
        Reference points
        
    Returns
    -------
    List[Solution]
        Selected solutions
    """
    from trainselpy.genetic_algorithm import fast_non_dominated_sort
    
    # 1. Non-dominated sorting
    fronts = fast_non_dominated_sort(population)
    
    selected = []
    last_front_idx = 0
    
    # 2. Fill with complete fronts
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            last_front = front
            break
        last_front_idx += 1
        
    if len(selected) == n_select:
        return selected
        
    # 3. Selection from last front
    n_needed = n_select - len(selected)
    
    # Determine Ideal Point (min values for each objective)
    # Using all solutions in St (selected + last_front)
    st = selected + last_front
    objs = np.array([s.multi_fitness for s in st])
    
    # Note: We MAXIMIZE in TrainSelPy.
    # NSGA-III is usually defined for MINIMIZATION.
    # We should negate objectives for standard NSGA-III logic, or adapt logic.
    # Let's negate to use standard minimization logic.
    objs = -objs
    ideal_point = np.min(objs, axis=0)
    
    # Translate objectives
    trans_objs = objs - ideal_point
    
    # Find Extreme Points (for intercept calculation)
    # ASF: Minimize max(f_i / w_i)
    # We use identity matrix rows as weight vectors for extreme points
    n_obj = objs.shape[1]
    extreme_points = []
    
    for i in range(n_obj):
        w = np.zeros(n_obj) + 1e-6
        w[i] = 1.0
        
        # ASF
        asf = np.max(trans_objs / w, axis=1)
        best_idx = np.argmin(asf)
        extreme_points.append(trans_objs[best_idx])
        
    # Calculate Intercepts (Gaussian elimination)
    # Plane equation: sum(f_i / a_i) = 1
    try:
        # Solve linear system if possible
        # If extreme points are degenerate, fallback to max values
        E = np.array(extreme_points)
        if np.linalg.matrix_rank(E) == n_obj:
            # Solve E * x = 1, where x_i = 1/a_i
            x = np.linalg.solve(E, np.ones(n_obj))
            intercepts = 1.0 / x
        else:
            intercepts = np.max(trans_objs, axis=0)
    except np.linalg.LinAlgError:
        intercepts = np.max(trans_objs, axis=0)
        
    # Normalize objectives
    # f_n = (f - z_min) / (a - z_min)
    # But we already shifted by z_min (trans_objs)
    # So f_n = trans_objs / intercepts
    # (intercepts are relative to z_min)
    
    # We need to normalize ALL solutions in St (to compute niche counts)
    # But we only care about association for St
    
    norm_objs = trans_objs / intercepts
    
    # Associate
    association, distances = associate_to_reference_points(norm_objs, reference_points)
    
    # Compute niche counts for solutions already in 'selected' (P_t+1 \ F_l)
    rho = np.zeros(len(reference_points), dtype=int)
    
    # Indices of 'selected' in 'st' are 0 to len(selected)-1
    for i in range(len(selected)):
        ref_idx = association[i]
        rho[ref_idx] += 1
        
    # Select from last front
    # Indices of 'last_front' in 'st' are len(selected) to len(st)-1
    last_front_indices = list(range(len(selected), len(st)))
    
    final_selection = []
    potential_indices = last_front_indices.copy()
    
    while len(final_selection) < n_needed:
        # Find reference points with min rho
        min_rho = np.min(rho)
        j_min = np.where(rho == min_rho)[0]
        
        # Pick a random reference point from j_min
        j_bar = np.random.choice(j_min)
        
        # Find solutions in last front associated with j_bar
        associated_indices = [idx for idx in potential_indices if association[idx] == j_bar]
        
        if associated_indices:
            if rho[j_bar] == 0:
                # Select the one with min perpendicular distance
                # distances is aligned with st
                dists = [distances[idx] for idx in associated_indices]
                best_rel_idx = np.argmin(dists)
                best_idx = associated_indices[best_rel_idx]
            else:
                # Select randomly
                best_idx = np.random.choice(associated_indices)
                
            final_selection.append(st[best_idx])
            rho[j_bar] += 1
            potential_indices.remove(best_idx)
        else:
            # No solution associated with this reference point
            # Exclude this reference point from consideration?
            # Standard NSGA-III: remove j_bar from consideration for current step
            # But here we just increment rho to move to next min
            rho[j_bar] += 1
            
    return selected + final_selection

