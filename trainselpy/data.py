"""
Data module for TrainSelPy.

This module provides access to example datasets.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
import pickle


def load_wheat_data() -> Dict[str, Any]:
    """
    Load the wheat example dataset.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing wheat data:
        - Y: Phenotypic data for adult plant height
        - M: Marker data (4670 markers for 1182 wheat lines)
        - K: Genomic relationship matrix
    """
    # Check if the wheat data is already available
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_file = os.path.join(data_dir, 'wheat_data.pkl')
    
    if os.path.exists(data_file):
        # Load the data from the file
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    # If the data is not available, create simulated data
    print("Wheat dataset not found, creating simulated data.")
    # Create simulated data with similar properties to the original wheat dataset
    n_lines = 1182
    n_markers = 4670
    
    # Create marker data (-1, 0, 1 coding)
    M = np.random.choice([-1, 0, 1], size=(n_lines, n_markers), p=[0.25, 0.5, 0.25])
    
    # Create line names
    line_names = [f"Line_{i+1}" for i in range(n_lines)]
    
    # Create dataframe with line names as index
    M_df = pd.DataFrame(M, index=line_names)
    
    # Compute genomic relationship matrix
    K = np.dot(M, M.T) / M.shape[1]
    K_df = pd.DataFrame(K, index=line_names, columns=line_names)
    
    # Simulate phenotypic data
    u = np.random.multivariate_normal(np.zeros(n_lines), K, 1)[0]
    e = np.random.normal(0, 0.5, n_lines)
    Y = u + e
    Y_df = pd.Series(Y, index=line_names)
    
    # Create the data dictionary
    data = {
        'Y': Y_df,
        'M': M_df,
        'K': K_df
    }
    
    # Save the data for future use
    os.makedirs(data_dir, exist_ok=True)
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data


# Create a module-level variable for easy access to the wheat data
wheat_data = load_wheat_data()