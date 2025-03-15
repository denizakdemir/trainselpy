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
    else:
        plt.show()


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