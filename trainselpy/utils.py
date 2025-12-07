"""
Utility functions for TrainSelPy.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Union
import pickle


def r_data_to_python(r_data_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Convert R data (.rda, .rdata) to Python format.
    
    This function requires the rpy2 package to be installed.
    
    Parameters
    ----------
    r_data_path : str
        Path to the R data file
    output_path : str, optional
        Path to save the converted data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the converted data
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
    except ImportError:
        raise ImportError("rpy2 is required to convert R data to Python format. Install it with 'pip install rpy2'.")
    
    # Load the R data
    robjects.r(f"load('{r_data_path}')")
    
    # Get the names of the objects in the R environment
    r_objects = robjects.r("ls()")
    
    # Convert each object to Python
    python_data = {}
    for obj_name in r_objects:
        r_obj = robjects.r[obj_name]
        
        # Try to convert to pandas DataFrame or Series
        try:
            python_data[obj_name] = pandas2ri.rpy2py(r_obj)
        except:
            # If conversion fails, try to convert to numpy array
            try:
                python_data[obj_name] = np.array(r_obj)
            except:
                # If all else fails, just keep the R object
                python_data[obj_name] = r_obj
    
    # Save the converted data if output_path is provided
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(python_data, f)
    
    return python_data


def create_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Create a distance matrix from a feature matrix.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix (n_samples, n_features)
    metric : str, optional
        Distance metric to use
        
    Returns
    -------
    ndarray
        Distance matrix (n_samples, n_samples)
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Compute the pairwise distances
    distances = pdist(X, metric=metric)
    
    # Convert to a square matrix
    distance_matrix = squareform(distances)
    
    return distance_matrix


def calculate_relationship_matrix(M: np.ndarray, method: str = 'additive') -> np.ndarray:
    """
    Calculate a relationship matrix from a marker matrix.
    
    Parameters
    ----------
    M : ndarray
        Marker matrix (n_samples, n_markers)
    method : str, optional
        Method to use for calculating the relationship matrix
        
    Returns
    -------
    ndarray
        Relationship matrix (n_samples, n_samples)
    """
    if method == 'additive':
        # Center the marker matrix
        M_centered = M - np.mean(M, axis=0)
        
        # Calculate the relationship matrix
        K = np.dot(M_centered, M_centered.T) / M.shape[1]
        
        # Make sure the matrix is positive definite
        min_eig = np.min(np.linalg.eigvalsh(K))
        if min_eig < 1e-10:
            K += np.eye(K.shape[0]) * (1e-10 - min_eig)
    
    elif method == 'dominance':
        # Calculate dominance matrix (assuming markers are coded as -1, 0, 1)
        D = (M == 0).astype(float)
        D_centered = D - np.mean(D, axis=0)
        
        # Calculate the relationship matrix
        K = np.dot(D_centered, D_centered.T) / D.shape[1]
        
        # Make sure the matrix is positive definite
        min_eig = np.min(np.linalg.eigvalsh(K))
        if min_eig < 1e-10:
            K += np.eye(K.shape[0]) * (1e-10 - min_eig)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return K


def create_mixed_model_data(
    G: np.ndarray, 
    R: np.ndarray = None, 
    lambda_val: float = 1.0,
    X: np.ndarray = None
) -> Dict[str, Any]:
    """
    Create data for mixed model optimization.
    
    Parameters
    ----------
    G : ndarray
        Genetic relationship matrix
    R : ndarray, optional
        Residual relationship matrix
    lambda_val : float, optional
        Ratio of residual to genetic variance
    X : ndarray, optional
        Design matrix
        
    Returns
    -------
    Dict[str, Any]
        Data for mixed model optimization
    """
    # Create a TrainSel_Data object
    data = {}
    
    # Add the genetic relationship matrix
    data["G"] = G
    
    # Add the residual relationship matrix
    if R is None:
        R = np.eye(G.shape[0])
    data["R"] = R
    
    # Add lambda
    data["lambda"] = lambda_val
    
    # Add the design matrix
    if X is not None:
        data["X"] = X
    
    # Add metadata
    data["Nind"] = G.shape[0]
    data["class"] = "TrainSel_Data"
    
    return data


def plot_optimization_progress(
    fitness_history: List[float],
    title: str = 'Optimization Progress',
    xlabel: str = 'Generation',
    ylabel: str = 'Fitness',
    output_file: str = None
):
    """
    Plot the optimization progress.
    
    Parameters
    ----------
    fitness_history : List[float]
        List of fitness values for each generation
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    output_file : str, optional
        Path to save the plot
        
    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot optimization progress. Install it with 'pip install matplotlib'.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def compute_hypervolume(front: List[List[float]], reference_point: List[float]) -> float:
    """
    Compute the hypervolume of a Pareto front.
    Currently optimized for 2D.
    
    Parameters
    ----------
    front : List[List[float]]
        List of points on the Pareto front (should be non-dominated)
    reference_point : List[float]
        Reference point for hypervolume calculation (should be worse than all points)
        
    Returns
    -------
    float
        Hypervolume value
    """
    if not front:
        return 0.0
    
    # Convert to numpy array
    front_arr = np.array(front)
    
    # Filter points that do not dominate the reference point
    # We want points that are BETTER than reference point.
    # Assuming maximization? Or minimization?
    # TrainSelPy usually assumes maximization of fitness.
    # So reference point should be SMALLER than all points (e.g. [0, 0])
    # And we calculate volume between point and reference.
    # BUT, hypervolume is usually defined for minimization (volume dominated by front bounded by ref).
    # If maximization, we can transform to minimization: val' = -val, ref' = -ref.
    # OR, we just define it as volume of union of hyper-rectangles [ref, point].
    
    # Let's assume maximization and we want Volume(Union([ref, point_i]))
    # Reference point should be the lower bound.
    
    n_obj = len(reference_point)
    
    if n_obj == 2:
        # Sort by first objective
        # For maximization:
        # We need volume of union of rectangles defined by (ref[0], ref[1]) to (p[0], p[1])
        # Sort points by first objective ascending
        sorted_indices = np.argsort(front_arr[:, 0])
        sorted_front = front_arr[sorted_indices]
        
        volume = 0.0
        current_max_y = reference_point[1]
        
        for point in sorted_front:
            # For each point, add the area of the rectangle covering [point[0], prev_x] * [point[1], ref[1]]
            # BUT ONLY if point[1] > current_max_y
            # Actually simpler:
            # Iterate sorted by X.
            # Volume += (x_i - x_{i-1}) * (max(y_j for j >= i) - ref_y) ?? No.
            
            # Standard algorithm for maximization 2D:
            # Sort by X descending.
            # Volume += (x_i - ref_x) * (y_i - max_y_seen_so_far) (if y_i > max_y)
            # No, that's not right.
            
            # Correct 2D Hypervolume for Maximization:
            # Reference point R = (r1, r2). Points P = {(x1, y1), ...} where xi >= r1, yi >= r2.
            # Sort P by x descending: p1, p2, ..., pk such that x1 > x2 > ... > xk.
            # Then necessarily y1 < y2 < ... < yk (since non-dominated).
            # Area = (p1.x - r1) * (p1.y - r2) 
            #      + (p2.x - r1) * (p2.y - p1.y)  <-- wait, p2.x is SMALLER.
            
            # Let's try standard slicing:
            # sorted by X ascending: x1 < x2 < ...
            # Then y1 > y2 > ...
            # Area = sum (x_{i} - x_{i-1}) * (y_i - ref_y)  <-- simple!
            # where x_0 = ref_x.
            
            pass
        
        # Implementation for Sort-by-X Ascending:
        sorted_indices = np.argsort(front_arr[:, 0])
        sorted_front = front_arr[sorted_indices]
        
        volume = 0.0
        prev_x = reference_point[0]
        
        # We need to consider the "envelope"
        # Since they are non-dominated, and sorted by X ascending, Y must be descending?
        # Let's verify:
        # P1=(10, 2), P2=(5, 5). Sorted X: P2, P1.
        # P2=(5, 5), P1=(10, 2).
        # Union area:
        # Rect1: [r_x, 5] * [r_y, 5]
        # Rect2: [r_x, 10] * [r_y, 2]
        # Their union...
        
        # Actually simpler:
        # Area = (P2.x - r_x) * (P2.y - r_y) + (P1.x - P2.x) * (P1.y - r_y)
        # Yes!
        # Because P2.y > P1.y (since P1 has higher X, it must have lower Y to be non-dominated).
        # So for the range [P2.x, P1.x], the height is constrained by P1.y.
        # For [r_x, P2.x], height is P2.y.
        
        # Generalizing:
        # Sort by X ascending.
        # Area = 0
        # for i in range(len):
        #    width = p[i].x - (p[i-1].x if i>0 else ref_x)
        #    height = p[i].y - ref_y  <-- NO, this assumes p[i] dictates height for determining the block to its LEFT?
        # No, p[i] is to the RIGHT of p[i-1].
        # So the interval (p[i-1].x, p[i].x] is covered by... whom?
        # It's covered by p[i] and all subsequent points.
        # But subsequent points have LOWER Y.
        # So the height in this interval is max(y_k for k >= i).
        # Which is just p[i].y.
        
        # Wait, let's reverse:
        # Sort by X Ascending.
        # P1(5,5), P2(10,2). Ref(0,0).
        # i=0: P1. Area += (5 - 0) * (5 - 0) = 25.
        # i=1: P2. Area += (10 - 5) * (2 - 0) = 5 * 2 = 10.
        # Total = 35.
        # Check: Union of [0,5]x[0,5] and [0,10]x[0,2].
        # [0,10]x[0,2] has area 20.
        # [0,5]x[0,5] has area 25.
        # Intersection is [0,5]x[0,2] = 10.
        # Union = 20 + 25 - 10 = 35. Correct.
        
        # So algorithm:
        # Sort by X ascending.
        # Term (width) = (current.x - prev.x).
        # Term (height) = (current.y - ref.y).
        # Sum them up?
        # In example:
        # i=0: (5 - 0) * (5 - 0) = 25.
        # i=1: (10 - 5) * (2 - 0) = 10.
        # It works because for x in (prev_x, curr_x], the max Y available is curr_y?
        # NO.
        # Example: P1(5,5), P2(10,2).
        # Range [0, 5]: Covered by P1 (y=5) AND P2 (y=2). Max y = 5.
        # Range [5, 10]: Covered ONLY by P2 (y=2). Max y = 2.
        
        # My loop calculation:
        # i=0 (P1): width=5, height=5. Area += 25. (Correct for range [0,5])
        # i=1 (P2): width=(10-5)=5, height=2. Area += 10. (Correct for range [5,10])
        
        # BUT this assumes sorted by X ascending implies Y descending.
        # If P3(12, 6) exists, it dominates P2 and P1? No.
        # P3(12, 6) dominates P2(10, 2) since 12>10 and 6>2.
        # So P2 should NOT be in the front.
        # So yes, a valid Pareto front sorted by X ascending MUST have Y descending.
        
        # Algorithm is:
        # 1. Sort by objective 1 ascending.
        # 2. Iterate. Area += (p[i].obj1 - p[i-1].obj1) * (p[i].obj2 - ref_obj2).
        #    (Use ref for p[-1])
        # BUT wait.
        # Range [0, 5] corresponds to P1(5,5). Term: (5-0)*5 = 25.
        # Range [5, 10] corresponds to P2(10,2). Term: (10-5)*2 = 10.
        # This seems correct IF we iterate 0..N-1?
        # My logic was:
        # i=0: P1. width=5-0. height=5.
        # i=1: P2. width=10-5. height=2.
        # This matches.
        
        # Wait, what if X goes up and Y goes DOWN?
        # P1(5,5), P2(10,2).
        # X: 5 -> 10. Y: 5 -> 2.
        # The first chunk [0,5] has height 5 (from P1).
        # The second chunk [5,10] has height 2 (from P2).
        # So area is (5-0)*5 + (10-5)*2.
        # Yes.
        
        # What if X goes up and Y goes UP? (Not Pareto optimal)
        # P1(5,2), P2(10,5).
        # P2 dominates P1. P1 shouldn't be there.
        # The code assumes `front` is non-dominated.
        # But to be safe, we should filter it or handle it.
        # Let's just assume valid front for now, or ensure we take max Y.
        
        # Robust 2D algorithm:
        # Sort X ascending.
        # Max Y from right to left?
        # Actually, let's just stick to the calculation assuming non-dominated.
        # Or even better: calculate union of rectangles.
        # Since N is small, we can just be precise.
        # But the slicing method is standard.
        
        volume = 0.0
        prev_x = reference_point[0]
        
        # Sort by first objective
        # We must filter out points worse than reference?
        # Assume valid inputs.
        
        sorted_indices = np.argsort(front_arr[:, 0])
        sorted_front = front_arr[sorted_indices]
        
        # Filter dominated points just in case?
        # Ideally the input is a Pareto front.
        
        for p in sorted_front:
            width = p[0] - prev_x
            height = p[1] - reference_point[1]
            if width > 0 and height > 0:
                volume += width * height
            prev_x = p[0]
            
        # WAIT!
        # In my manual trace:
        # P1(5,5), P2(10,2). Ref(0,0).
        # Sorted: P1, P2.
        # i=0 (P1): width = 5-0=5. height = 5-0=5. Vol += 25. prev_x=5.
        # i=1 (P2): width = 10-5=5. height = 2-0=2. Vol += 10. prev_x=10.
        # Total 35.
        # This works ONLY if Y is strictly decreasing (or non-increasing).
        # If Y increased, say P2 was (10, 6) [dominating P1],
        # then i=1: width=5, height=6 -> vol+=30. Total 55.
        # Correct union: [0,10]x[0,6] = 60.
        # My result 55 is WRONG.
        # So this ONLY works for non-dominated front.
        
        # Better robust approach:
        # Discretize? No.
        # Just compute union of rectangles properly.
        # But since we assume `front` comes from `fast_non_dominated_sort`, it SHOULD be non-dominated.
        # Let's rely on that.
        
        return volume

    else:
        # Fallback for >2D: Monte Carlo estimation?
        # Or just sum of fitnesses as a heuristic proxy if expensive?
        # No, let's do Monte Carlo.
        # Sample points in bounding box defined by Ref and Max(Front).
        # Count % dominated by at least one point in Front.
        
        # Bounding box
        max_vals = np.max(front_arr, axis=0)
        min_vals = np.array(reference_point)
        
        # Volume of box
        box_vol = np.prod(max_vals - min_vals)
        if box_vol <= 0:
            return 0.0
            
        n_samples = 1000
        samples = np.random.uniform(min_vals, max_vals, (n_samples, n_obj))
        
        # Check dominance
        # A sample S is dominated by Front if exists P in Front s.t. P dominates S.
        # i.e., P_j >= S_j for all j.
        
        dominated_count = 0
        for sample in samples:
            # Check if any point in front dominates sample
            # (assuming maximization)
            is_dominated = np.any(np.all(front_arr >= sample, axis=1))
            if is_dominated:
                dominated_count += 1
                
        return box_vol * (dominated_count / n_samples)


def plot_pareto_front(
    pareto_front: List[List[float]],
    obj_names: List[str] = None,
    title: str = 'Pareto Front',
    output_file: str = None
):
    """
    Plot the Pareto front for multi-objective optimization.
    
    Parameters
    ----------
    pareto_front : List[List[float]]
        List of points on the Pareto front
    obj_names : List[str], optional
        Names of the objectives
    title : str, optional
        Plot title
    output_file : str, optional
        Path to save the plot
        
    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to plot Pareto fronts. Install it with 'pip install matplotlib'.")
    
    # Get the number of objectives
    n_obj = len(pareto_front[0])
    
    if n_obj == 2:
        # 2D plot
        plt.figure(figsize=(10, 6))
        x = [point[0] for point in pareto_front]
        y = [point[1] for point in pareto_front]
        plt.scatter(x, y, s=50, alpha=0.7)
        plt.title(title)
        
        if obj_names and len(obj_names) >= 2:
            plt.xlabel(obj_names[0])
            plt.ylabel(obj_names[1])
        else:
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
        
        plt.grid(True)
        plt.tight_layout()
    
    elif n_obj == 3:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = [point[0] for point in pareto_front]
        y = [point[1] for point in pareto_front]
        z = [point[2] for point in pareto_front]
        
        ax.scatter(x, y, z, s=50, alpha=0.7)
        ax.set_title(title)
        
        if obj_names and len(obj_names) >= 3:
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
        else:
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
        
    else:
        # Matrix of 2D plots for more than 3 objectives
        import seaborn as sns
        
        # Create a dataframe with the Pareto front points
        df = pd.DataFrame(pareto_front)
        
        # Set column names
        if obj_names and len(obj_names) >= n_obj:
            df.columns = obj_names
        else:
            df.columns = [f'Objective {i+1}' for i in range(n_obj)]
        
        # Create a matrix of scatter plots
        plt.figure(figsize=(12, 10))
        sns.pairplot(df)
        plt.suptitle(title, y=1.02, fontsize=16)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()